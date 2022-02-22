from pathlib import Path
import numpy as np
import torch
from einops import reduce
from typing import Optional, Tuple
from torch.utils.cpp_extension import load
from torchtyping import TensorType

from config import NetworkConfig, TrainConfig
from data.nsvf_dataset import NSVFDataset as Dataset
from ray_batch import RayBatch
import utils


class NerfLib:
    def __init__(self, net_cfg: NetworkConfig, train_cfg: TrainConfig, device):
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg
        self.device = device

        # Load cuda
        EXT_NAME = 'nerf_cuda_lib'
        EXT_MAIN_CPP_FN = 'nerf_lib.cpp'

        cuda_dir = Path('./cuda')
        cuda_paths = [p for p in cuda_dir.iterdir() if p.suffix == '.cu']
        cuda_modules = [p.stem for p in cuda_paths]
        cuda_load_paths = [str(cuda_dir / EXT_MAIN_CPP_FN)] + \
            [str(p) for p in cuda_paths]

        cuda_lib = load(
            EXT_NAME, cuda_load_paths, verbose=True, extra_cflags=['-w'])
        for module in cuda_modules:
            assert module in dir(cuda_lib)
            setattr(self, module, getattr(cuda_lib, module))

    def generate_rays(
        self,
        img: TensorType['H', 'W', 3],
        pose: TensorType[4, 4],
        dataset: Dataset,
        precrop: Optional[float] = None
    ) -> Tuple[TensorType['K', 3], RayBatch]:
        """Generate a batch of rays.

        Args:
            img (TensorType['h', 'w', 3]): Image tensor.
            pose (TensorType[4, 4]): Camera pose tensor.
            dataset (Dataset): Dataset object.
            precrop (Optional[float]): Precrop factor; None if not used.

        Returns:
            target (TensorType['K', 3]): Pixel values corresponding to rays.
            rays (RayBatch): A batch of K rays.
        """

        intr = dataset.intrinsics
        near, far = dataset.near, dataset.far

        x_coords = np.arange(intr.w, dtype=np.float32)
        y_coords = np.arange(intr.h, dtype=np.float32)
        w, h = intr.w, intr.h
        dx, dy = 0, 0
        pose_r, pose_t = pose[:3, :3], pose[:3, 3]

        if precrop is not None:
            w, h = int(intr.w * precrop), int(intr.h * precrop)
            dx, dy = (intr.w - w) // 2, (intr.h - h) // 2
            x_coords, y_coords = x_coords[dx:dx+w], y_coords[dy:dy+h]

        i, j = np.meshgrid(x_coords, y_coords, indexing='xy')
        # Transform by inverse intrinsic matrix
        dirs = torch.FloatTensor(np.stack([
            (i - intr.cx) / intr.fx,
            -(j - intr.cy) / intr.fy,
            -np.ones_like(i)
        ], axis=-1)).to(self.device)

        # Transform by camera pose (camera to world coords)
        rays_d = torch.einsum('ij, hwj -> hwi', pose_r, dirs)

        indices_1d = np.random.choice(
            np.arange(w * h), self.train_cfg.num_rays_per_batch, replace=False)
        indices_2d = (indices_1d // h, indices_1d % h)
        coords = (indices_2d[0] + dy, indices_2d[1] + dx)
        rays_d = rays_d[indices_2d]

        target = img[coords]
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
        z_vals = torch.linspace(rays.near, rays.far, steps=(
            n_samples + 1)).to(self.device)
        z_vals = z_vals.expand([len(rays), n_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape).to(self.device)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]) * \
            torch.norm(rays.dests, dim=-1, keepdim=True)
        dists = torch.cat([dists, torch.ones(
            (len(dists), 1)).to(self.device) * 1e10], dim=-1)
        return pts, dists

    def integrate_points(
        self,
        dists: TensorType['N', 'K'],
        rgbs: TensorType['N', 'K', 3],
        densities: TensorType['N', 'K'],
        bg_color: TensorType[3]
    ) -> TensorType['N', 3]:
        """Evaluate the volumetric rendering equation for N rays, each with
        K samples.

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
