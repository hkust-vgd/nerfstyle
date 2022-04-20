import torch
from nerf_lib import nerf_lib
import utils

class Renderer:
    def __init__(
        self,
        model,
        dataset,
        net_cfg,
        train_cfg,
        all_rays=True,
        device='cuda',
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg

        self.precrop = False
        self.all_rays = all_rays
        self.device = device
    
    def render(self, img, pose):
        # Generate rays
        precrop_frac, rays_bsize = None, None
        if self.precrop:
            precrop_frac = self.train_cfg.precrop_fraction
        if not self.all_rays:
            rays_bsize = self.train_cfg.num_rays_per_batch

        target, rays = nerf_lib.generate_rays(
            img, pose, self.dataset,
            precrop=precrop_frac, bsize=rays_bsize)
        dirs = rays.viewdirs()
        
        # Sample points
        pts, dists = nerf_lib.sample_points(rays)
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = torch.repeat_interleave(
            dirs, repeats=self.net_cfg.num_samples_per_ray, dim=0)
        del pts, dirs

        # Evaluate model
        rgbs = torch.empty((len(pts_flat), 3), device=self.device)
        densities = torch.empty((len(pts_flat), 1), device=self.device)
        utils.batch_exec(self.model, rgbs, densities,
                         bsize=self.net_cfg.pts_bsize)(pts_flat, dirs_flat)
        rgbs = rgbs.reshape(*dists.shape, 3)
        densities = densities.reshape(dists.shape)
        del pts_flat, dirs_flat

        # Integrate points
        bg_color = torch.tensor(self.dataset.bg_color).to(self.device)
        rgb_map = nerf_lib.integrate_points(dists, rgbs, densities, bg_color)
        del dists, rgbs, densities

        return rgb_map, target
