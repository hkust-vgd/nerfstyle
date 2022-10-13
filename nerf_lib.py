from pathlib import Path
from typing import Optional, Tuple
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torchtyping import TensorType

from common import Intrinsics, RayBatch
import utils


def assert_ready(func):
    def new_func(self, *args, **kwargs):
        assert self._ready, 'Please assign a GPU to nerf_lib.device first.'
        return func(self, *args, **kwargs)

    return new_func


class NerfLib:
    EXT_NAME = 'nerf_cuda_lib'
    EXT_MAIN_CPP_FN = 'nerf_lib.cpp'

    def __init__(self):
        self._device = None
        self._ready = False

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        assert device.type == 'cuda', 'Device must be GPU'
        self._device = device
        self._ready = True

    @assert_ready
    def load_cuda_ext(self) -> None:
        """
        Activate GPU mode for NeRF library object.
        """
        cuda_lib = self._load_cuda_ext()
        for method in dir(cuda_lib):
            if method[:2] == '__':
                continue
            # Overload all functions from CUDA extension.
            setattr(self, method, getattr(cuda_lib, method))

    @staticmethod
    def _load_cuda_ext():
        cuda_dir = Path('./cuda')
        cuda_paths = [p for p in cuda_dir.iterdir() if p.suffix == '.cu']
        cuda_load_paths = [str(cuda_dir / NerfLib.EXT_MAIN_CPP_FN)] + [str(p) for p in cuda_paths]

        include_dirs = ['/usr/local/magma/include']
        cuda_ext = load(
            NerfLib.EXT_NAME,
            cuda_load_paths,
            extra_include_paths=include_dirs,
            extra_cflags=['-w', '-D_GLIBCXX_USE_CXX11_ABI=0'],
            extra_ldflags=['/usr/local/magma/lib/libmagma.a'],
            verbose=True
        )
        return cuda_ext

    @assert_ready
    def generate_rays(
        self,
        pose: TensorType[4, 4],
        intr: Intrinsics,
        img: Optional[TensorType['H', 'W', 3]] = None,
        precrop: float = 1.,
        bsize: Optional[int] = None,
        camera_flip: int = 0
    ) -> Tuple[RayBatch, Optional[TensorType['K', 3]]]:
        """Generate a batch of rays.

        Args:
            pose (TensorType[4, 4]): Camera-to-world transformation matrix.
            intr (Intrinsics): Intrinsics data for render camera.
            img (Optional[TensorType['h', 'w', 3]]): Ground truth image.
            precrop (float): Square cropping factor. Defaults to 1.0 (no cropping).
            bsize (Optional[int]): Size of ray batch. All rays are used if not specified.

        Returns:
            rays (RayBatch): A batch of K rays.
            target (Optional[TensorType['K', 3]]): Pixel values corresponding to rays.
        """
        assert (precrop >= 0.) and (precrop <= 1.)
        target = None

        # Symmetric samples in pixel coords system: [0.5, 1.5, 2.5, ...]
        fh, fw = intr.h, intr.w
        x_coords = np.linspace(0, fw, num=2*fw+1, dtype=np.float32)[1::2]
        y_coords = np.linspace(0, fh, num=2*fh+1, dtype=np.float32)[1::2]

        w, h = intr.w, intr.h
        dx, dy = 0, 0
        pose_r, pose_t = pose[:3, :3], pose[:3, 3]

        if precrop < 1.:
            w, h = int(intr.w * precrop), int(intr.h * precrop)
            dx, dy = (intr.w - w) // 2, (intr.h - h) // 2
            x_coords, y_coords = x_coords[dx:dx+w], y_coords[dy:dy+h]

        i, j = np.meshgrid(x_coords, y_coords, indexing='xy')
        k = np.ones_like(i)

        # Pixel coords to camera frame
        dirs = torch.tensor(np.stack([
            (i - intr.cx) / intr.fx, (j - intr.cy) / intr.fy, k
        ], axis=-1), device=self._device)
        flip = np.where([(camera_flip >> i) & 1 for i in [2, 1, 0]], -1, 1)
        dirs *= torch.tensor(flip, device=self._device)

        # Camera frame to world frame
        rays_d = torch.einsum('ij, hwj -> hwi', pose_r, dirs)

        if bsize is None:
            rays_d = rays_d.reshape((-1, 3))
            if img is not None:
                if fh != img.shape[-2] or fw != img.shape[-1]:
                    img = F.interpolate(img.unsqueeze(0), size=(fh, fw)).squeeze(0)
                target = einops.rearrange(img, 'c h w -> (h w) c')
        else:
            indices_1d = np.random.choice(np.arange(w * h), bsize, replace=False)
            indices_2d = (indices_1d // w, indices_1d % w)
            coords_y, coords_x = indices_2d[0] + dy, indices_2d[1] + dx
            rays_d = rays_d[indices_2d]
            if img is not None:
                target = einops.rearrange(img, 'c h w -> h w c')[coords_y, coords_x]

        rays = RayBatch(pose_t, rays_d)
        return rays, target

    @assert_ready
    def sample_points(
        self,
        rays: RayBatch,
        near: float,
        far: float,
        num_samples: int
    ) -> Tuple[TensorType['N', 'K', 3], TensorType['N', 'K']]:
        """Given a batch of N rays, sample K points per ray.

        Args:
            rays (RayBatch): Ray batch of size N.

        Returns:
            pts (TensorType[N, K, 3]): Coordinates of the samples.
            dists (TensorType[N, K]): Distances between samples.
            near (float): Distance from origin to start sampling points.
            far (float): Distance from origin to stop sampling points.
            num_samples (int): No. of point samples per ray.
        """
        z_vals = torch.linspace(near, far, steps=(num_samples + 1), device=self._device)
        z_vals = z_vals.expand([len(rays), num_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape, device=self._device)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        last_dist = torch.ones((len(dists), 1), device=self._device) * 1e10
        dists = torch.cat([dists, last_dist], dim=-1)
        return pts, dists

    @assert_ready
    def integrate_points(
        self,
        dists: TensorType['N', 'K'],
        rgbs: TensorType['N', 'K', 3],
        densities: TensorType['N', 'K'],
        prev_rgb: TensorType['N', 3],
        prev_acc: TensorType['N', 1],
        prev_trans: TensorType['N', 1]
    ) -> Tuple[TensorType['N', 3], TensorType['N', 1]]:
        """Evaluate the volumetric rendering equation for N rays, each with K samples.

        Args:
            dists (TensorType[N, K]): Distances between samples.
            rgbs (TensorType[N, K, 3]): Colors of the samples.
            densities (TensorType[N, K]): Densities of the samples.
            prev_rgb (TensorType[N, 3]): Colors from last pass. (First pass = 0)
            prev_acc (TensorType[N, 1]): Accumulated weights from last pass. (First pass = 0)
            prev_trans (TensorType[N, 1]): Transmittance from last pass. (First pass = 1)

        Returns:
            rgb_map (TensorType[N, 3]): Color values.
            acc_map (TensorType[N, 1]): Accumulated weights.
            trans_map (TensorType[N, 1]): Transmittance values.
        """

        # a_s, ..., a_{e-1}
        alpha = utils.density2alpha(densities, dists)

        # t_s, (1 - a_s), ..., (1 - a_{e-2})
        alpha_tmp = torch.cat([prev_trans, (1. - alpha[:, :-1])], dim=-1)

        # t_s, ..., t_{e-1}
        trans = torch.cumprod(alpha_tmp, dim=-1)

        weights = alpha * trans
        rgb_map = prev_rgb + einops.reduce(weights[..., None] * rgbs, 'n k c -> n c', 'sum')
        acc_map = prev_acc + einops.reduce(weights, 'n k -> n', 'sum').unsqueeze(-1)

        # t_e = t_{e-1} * (1 - a_{e-1})
        trans_map = (trans[:, -1] * (1. - alpha[:, -1]))[:, None]
        return rgb_map, acc_map, trans_map

    @assert_ready
    def global_to_local(
        self,
        points: TensorType['batch_size', 3],
        mid_points: TensorType['num_nets', 3],
        voxel_size: TensorType[3],
        batch_sizes: TensorType['num_nets']
    ) -> TensorType['batch_size', 3]:
        local_points = torch.empty_like(points)
        ptr = 0
        for mid_point, bsize in zip(mid_points, batch_sizes):
            local_points[ptr:ptr+bsize] = points[ptr:ptr+bsize] - mid_point
            ptr += bsize
        local_points /= (voxel_size / 2)
        return local_points

    # Stub CUDA methods
    @assert_ready
    def init_stream_pool(*_):
        pass

    @assert_ready
    def destroy_stream_pool(_):
        pass

    @assert_ready
    def init_magma(_):
        pass

    @assert_ready
    def init_multimatmul_aux_data(*_):
        return None

    @assert_ready
    def deinit_multimatmul_aux_data(_):
        pass


# Global instance
nerf_lib = NerfLib()
