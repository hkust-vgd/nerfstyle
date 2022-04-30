from pathlib import Path
from anyio import run_async_from_thread
import einops
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from typing import List, Optional, Tuple
from torch.utils.cpp_extension import load
from torchtyping import TensorType

from config import NetworkConfig, TrainConfig
from data.nsvf_dataset import NSVFDataset as Dataset
from common import RayBatch
import utils


class NerfLib:
    EXT_NAME = 'nerf_cuda_lib'
    EXT_MAIN_CPP_FN = 'nerf_lib.cpp'

    def __init__(self, net_cfg: NetworkConfig, train_cfg: TrainConfig, device):
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg
        self.device = device

        # Load cuda
        cuda_lib = self._load_cuda_ext()
        for method in dir(cuda_lib):
            if method[:2] == '__':
                continue
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

    def generate_rays(
        self,
        img: TensorType[3, 'H', 'W'],
        pose: TensorType[4, 4],
        dataset: Dataset,
        precrop: Optional[float] = None,
        bsize: Optional[int] = None,
        grid_dims: Optional[Tuple[int, int]] = None
    ) -> Tuple[TensorType['K', 3], RayBatch]:
        """Generate a batch of rays.

        Args:
            img (TensorType[3, 'H', 'W']): Ground truth image.
            pose (TensorType[4, 4]): Camera-to-world transformation matrix.
            dataset (Dataset): Dataset object.
            precrop (Optional[float]): Precrop factor; None if not specified.
            bsize (Optional[int]): Size of ray batch. All rays are used if not specified.
            grid_dims (Optional[Tuple[int, int]]): Dimension of sampling grid. If None, the output
                image dimensions are used.

        Returns:
            target (TensorType['K', 3]): Pixel values corresponding to rays.
            rays (RayBatch): Batch of rays.
        """

        intr = dataset.intrinsics
        near, far = dataset.near, dataset.far

        fh, fw = intr.h, intr.w
        if grid_dims is not None:
            fh, fw = grid_dims

        x_coords = np.linspace(0, intr.w, num=2*fw+1, dtype=np.float32)[1::2]
        y_coords = np.linspace(0, intr.h, num=2*fh+1, dtype=np.float32)[1::2]

        w, h = intr.w, intr.h
        dx, dy = 0, 0
        pose_r, pose_t = pose[:3, :3], pose[:3, 3]

        if precrop is not None:
            # TODO: Verify if dims paramter work
            assert grid_dims is None
            w, h = int(intr.w * precrop), int(intr.h * precrop)
            dx, dy = (intr.w - w) // 2, (intr.h - h) // 2
            x_coords, y_coords = x_coords[dx:dx+w], y_coords[dy:dy+h]

        i, j = np.meshgrid(x_coords, y_coords, indexing='xy')
        k = np.ones_like(i)

        # Pixel coords to camera frame
        dirs = torch.tensor(np.stack([
            (i - intr.cx) / intr.fx, (j - intr.cy) / intr.fy, k
        ], axis=-1), device=self.device)

        # Camera frame to world frame
        rays_d = torch.einsum('ij, hwj -> hwi', pose_r, dirs)

        if bsize is None:
            img = F.interpolate(img.unsqueeze(0), size=(fh, fw)).squeeze(0)
            target = einops.rearrange(img, 'c h w -> (h w) c')
            rays_d = rays_d.reshape((-1, 3))
        else:
            # TODO: Verify if dims paramter work
            assert grid_dims is None
            indices_1d = np.random.choice(np.arange(w * h), bsize, replace=False)
            indices_2d = (indices_1d // h, indices_1d % h)
            coords = (indices_2d[0] + dy, indices_2d[1] + dx)
            target = img[:, coords].T
            rays_d = rays_d[indices_2d]

        rays = RayBatch(pose_t, rays_d, near, far)
        return target, rays

    def sample_points(
        self,
        rays: RayBatch
    ) -> Tuple[TensorType['N', 'K', 3], TensorType['N', 'K']]:
        """Given a batch of N rays, sample K points per ray.

        Args:
            rays (RayBatch): Ray batch of size N.

        Returns:
            pts (TensorType[N, K, 3]): Coordinates of the samples.
            dists (TensorType[N, K]): Distances between samples.
        """
        n_samples = self.net_cfg.num_samples_per_ray
        z_vals = torch.linspace(rays.near, rays.far, steps=(n_samples + 1)).to(self.device)
        z_vals = z_vals.expand([len(rays), n_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape).to(self.device)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones((len(dists), 1)).to(self.device) * 1e10], dim=-1)
        return pts, dists

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
            torch.ones((len(alpha), 1)).to(self.device),
            (1. - alpha[:, :-1])
        ], dim=-1)
        ts = torch.cumprod(alpha_tmp, dim=-1)
        weights = alpha * ts  # (N, K)
        rgb_map = reduce(weights[..., None] * rgbs, 'n k c -> n c', 'sum')
        acc_map = reduce(weights, 'n k -> n', 'sum')

        res = (1 - acc_map)[..., None]
        rgb_map = rgb_map + res * bg_color
        return rgb_map

    @staticmethod
    def global_to_local(
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
    @staticmethod
    def init_stream_pool(_):
        pass

    @staticmethod
    def destroy_stream_pool():
        pass

    @staticmethod
    def init_magma():
        pass

    @staticmethod
    def init_multimatmul_aux_data(*_):
        return None

    @staticmethod
    def deinit_multimatmul_aux_data():
        pass


class NerfLibManager:
    def __init__(self) -> None:
        self._lib = None

    def init(self, net_cfg: NetworkConfig, train_cfg: TrainConfig, device):
        self._lib = NerfLib(net_cfg, train_cfg, device)

    def __getattr__(self, name):
        if self._lib is None:
            raise RuntimeError('Library not initialized; run "init()" first')
        return getattr(self._lib, name)


# Global instance
nerf_lib = NerfLibManager()
