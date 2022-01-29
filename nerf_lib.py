import numpy as np
from data.common import Intrinsics
from networks.embedder import Embedder


class NerfLib:
    def __init__(self, conf):
        self.x_embedder = Embedder(conf['x_enc_count'])
        self.d_embedder = Embedder(conf['d_enc_count'])

        self.num_rays_per_batch = conf['num_rays_per_batch']

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
        mask = np.random.choice(np.arange(N), self.num_rays_per_batch, replace=False)
        rays_d = rays_d[mask]
        rays_o = np.broadcast_to(pose[:3, -1], np.shape(rays_d)).reshape(-1, 3)
        target = img.reshape((N, -1))[mask]

        return target, rays_o, rays_d
