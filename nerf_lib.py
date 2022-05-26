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
        bsize: Optional[int] = None
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

        # Camera frame to world frame
        rays_d = torch.einsum('ij, hwj -> hwi', pose_r, dirs)

        if bsize is None:
            rays_d = rays_d.reshape((-1, 3))
            if img is not None:
                img = F.interpolate(img.unsqueeze(0), size=(fh, fw)).squeeze(0)
                target = einops.rearrange(img, 'c h w -> (h w) c')
        else:
            indices_1d = np.random.choice(np.arange(w * h), bsize, replace=False)
            indices_2d = (indices_1d // h, indices_1d % h)
            coords = (indices_2d[0] + dy, indices_2d[1] + dx)
            rays_d = rays_d[indices_2d]
            if img is not None:
                target = img[coords]

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
        z_vals = torch.linspace(near, far, steps=(num_samples + 1)).to(self._device)
        z_vals = z_vals.expand([len(rays), num_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape).to(self._device)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones((len(dists), 1)).to(self._device) * 1e10], dim=-1)
        return pts, dists

    @assert_ready
    def integrate_points(
        self,
        dists: TensorType['N', 'K'],
        rgbs: TensorType['N', 'K', 3],
        densities: TensorType['N', 'K'],
        bg_color: TensorType[3]
    ) -> TensorType['N', 3]:
        """Evaluate the volumetric rendering equation for N rays, each with K samples.

        Args:
            dists (TensorType[N, K]): Distances between samples.
            rgbs (TensorType[N, K, 3]): Colors of the samples.
            densities (TensorType[N, K]): Densities of the samples.
            bg_color (TensorType[3]): Background color.

        Returns:
            TensorType[N, 3]: Evaluation results.
        """

        alpha = utils.density2alpha(densities, dists)
        # 1, (1 - a_1), ..., (1 - a_(K-1))
        alpha_tmp = torch.cat([
            torch.ones((len(alpha), 1)).to(self._device),
            (1. - alpha[:, :-1])
        ], dim=-1)
        ts = torch.cumprod(alpha_tmp, dim=-1)
        weights = alpha * ts  # (N, K)
        rgb_map = einops.reduce(weights[..., None] * rgbs, 'n k c -> n c', 'sum')
        acc_map = einops.reduce(weights, 'n k -> n', 'sum')

        res = (1 - acc_map)[..., None]
        rgb_map = rgb_map + res * bg_color
        return rgb_map

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
