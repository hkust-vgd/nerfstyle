import numpy as np
import torch

from ray_batch import RayBatch
from data.common import Intrinsics
from networks.embedder import Embedder


class NerfLib:
    def __init__(self, conf):
        self.x_embedder = Embedder(conf['x_enc_count'])
        self.d_embedder = Embedder(conf['d_enc_count'])
        self.conf = conf

    def embed_x(self, x):
        return self.x_embedder(x)

    def embed_d(self, d):
        return self.d_embedder(d)

    def generate_rays(self, intr: Intrinsics, img, pose):
        i, j = np.meshgrid(np.arange(intr.w, dtype=np.float32), np.arange(intr.h, dtype=np.float32), indexing='xy')
        # K_inv * (u, v, 1)
        dirs = np.stack([(i - intr.cx) / intr.fx, -(j - intr.cy) / intr.fy, -np.ones_like(i)], -1)

        rays_d = np.einsum('ij, hwj -> hwi', pose[:3, :3], dirs).reshape(-1, 3)

        N = intr.h * intr.w
        mask = np.random.choice(np.arange(N), self.conf['num_rays_per_batch'], replace=False)
        rays_d = torch.tensor(rays_d[mask], dtype=torch.float32)
        rays_o = torch.tensor(pose[:3, -1], dtype=torch.float32)
        target = img.reshape((N, -1))[mask]

        return target, rays_o, rays_d

    def sample_points(self, rays: RayBatch):
        n_samples = self.conf['num_samples_per_ray']
        z_vals = torch.linspace(rays.near, rays.far, steps=(n_samples + 1))
        z_vals = z_vals.expand([len(rays), n_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)
        return pts
