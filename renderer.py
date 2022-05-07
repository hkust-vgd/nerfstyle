from typing import Tuple, TypeVar
import torch
from torchtyping import TensorType

from config import NetworkConfig, TrainConfig
from data.base_dataset import BaseDataset
from nerf_lib import nerf_lib
from networks.nerf import Nerf
import utils

T = TypeVar('T', bound='Renderer')


class Renderer:
    def __init__(
        self: T,
        model: Nerf,
        dataset: BaseDataset,
        net_cfg: NetworkConfig,
        train_cfg: TrainConfig,
        all_rays: bool = True
    ) -> None:
        """NeRF renderer.

        Args:
            model (Nerf): Backbone model.
            dataset (BaseDataset): Dataset which the model is trained on.
            net_cfg (NetworkConfig): Network configuration.
            train_cfg (TrainConfig): Training configuration.
            all_rays (bool, optional): If True, renders all rays in the image; if False, renders a
                randomized subset of rays. Defaults to True.
        """
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.net_cfg = net_cfg
        self.train_cfg = train_cfg

        self.precrop = False
        self.all_rays = all_rays
        self.device = self.model.device

        # Set BG color as white
        self.bg_color = torch.ones(3, device='cuda')

    def render(
        self: T,
        img: TensorType['H', 'W', 3],
        pose: TensorType[4, 4]
    ) -> Tuple[TensorType[..., 3], TensorType[..., 3]]:
        """Render a new image, given a camera pose.

        Args:
            img (TensorType['H', 'W', 3]): Ground truth image.
            pose (TensorType[4, 4]): Input camera pose.

        Returns:
            tuple[rgb_map, target], where:
                rgb_map (TensorType[..., 3]): Rendered RGB values.
                target (TensorType[..., 3]): Ground truth RGB values.
        """
        # TODO: make "img" parameter optional

        # Generate rays
        precrop_frac, rays_bsize = None, None
        if self.precrop:
            precrop_frac = self.train_cfg.precrop_fraction
        if not self.all_rays:
            rays_bsize = self.train_cfg.num_rays_per_batch

        target, rays = nerf_lib.generate_rays(
            img, pose, self.dataset, precrop=precrop_frac, bsize=rays_bsize)
        dirs = rays.viewdirs()

        # Sample points
        pts, dists = nerf_lib.sample_points(rays)
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = torch.repeat_interleave(dirs, repeats=self.net_cfg.num_samples_per_ray, dim=0)
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
        bg_color = torch.tensor(self.bg_color).to(self.device)
        rgb_map = nerf_lib.integrate_points(dists, rgbs, densities, bg_color)
        del dists, rgbs, densities

        return rgb_map, target
