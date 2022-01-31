import numpy as np
import torch
import torch.nn.functional as F

from config import NetworkConfig, TrainConfig
from ray_batch import RayBatch
from data.common import Intrinsics
from networks.embedder import Embedder


class NerfLib:
    def __init__(self, net_cfg: NetworkConfig, train_cfg: TrainConfig, device):
        self.x_embedder = Embedder(net_cfg.x_enc_count).to(device)
        self.d_embedder = Embedder(net_cfg.d_enc_count).to(device)
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg
        self.device = device

    def embed_x(self, x):
        return self.x_embedder(x)

    def embed_d(self, d):
        return self.d_embedder(d)

    def generate_rays(self, intr: Intrinsics, pose, precrop=None):
        x_coords = np.arange(intr.w, dtype=np.float32)
        y_coords = np.arange(intr.h, dtype=np.float32)
        w, h = intr.w, intr.h
        dx, dy = 0, 0

        if precrop is not None:
            w, h = int(intr.w * precrop), int(intr.h * precrop)
            dx, dy = (intr.w - w) // 2, (intr.h - h) // 2
            x_coords, y_coords = x_coords[dx:dx+w], y_coords[dy:dy+h]

        i, j = np.meshgrid(x_coords, y_coords, indexing='xy')
        # K_inv * (u, v, 1)
        dirs = np.stack([(i - intr.cx) / intr.fx, -(j - intr.cy) / intr.fy, -np.ones_like(i)], -1)
        rays_d = np.einsum('ij, hwj -> hwi', pose[:3, :3], dirs).reshape(-1, 3)

        indices = np.random.choice(np.arange(w * h), self.train_cfg.num_rays_per_batch, replace=False)
        rays_coords = (indices // h + dy, indices % h + dx)

        rays_d = torch.FloatTensor(rays_d[indices]).to(self.device)
        rays_o = torch.FloatTensor(pose[:3, -1]).to(self.device)

        return rays_o, rays_d, rays_coords

    def sample_points(self, rays: RayBatch):
        n_samples = self.net_cfg.num_samples_per_ray
        z_vals = torch.linspace(rays.near, rays.far, steps=(n_samples + 1)).to(self.device)
        z_vals = z_vals.expand([len(rays), n_samples + 1])

        lower = z_vals[:, :-1]
        upper = z_vals[:, 1:]
        t_rand = torch.rand(lower.shape).to(self.device)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays.lerp(z_vals)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]) * torch.norm(rays.dests, dim=-1, keepdim=True)
        dists = torch.cat([dists, torch.ones((len(dists), 1)).to(self.device) * 1e10], dim=-1)
        return pts, dists

    def integrate_points(self, dists, rgbs, densities):
        """
        Evaluate the volumetric rendering equation for multiple rays.
        :param dists:       (N, K) array of distances between samples
        :param rgbs:        (N, K, 3) array of sampled colors
        :param densities:   (N, K) array of sampled densities
        :return:            (N, 3) array of colors
        """
        alpha = 1. - torch.exp(-F.relu(densities) * dists)
        # 1, (1 - a_1), ..., (1 - a_(K-1))
        alpha_tmp = torch.cat([torch.ones((len(alpha), 1)).to(self.device), (1. - alpha[:, :-1])], dim=-1)
        transmittance = torch.cumprod(alpha_tmp, dim=-1)
        rgb_map = torch.sum((alpha * transmittance).unsqueeze(-1) * rgbs, dim=1)

        return rgb_map
