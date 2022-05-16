from typing import List, Optional, Tuple, TypeVar
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
        all_rays: bool = True,
        reduce_size: bool = False
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
        self.reduce_size = reduce_size
        self.device = self.model.device

        # Set BG color as white
        self.bg_color = torch.ones(3, device='cuda')

    def render(
        self: T,
        pose: TensorType[4, 4],
        img: Optional[TensorType['H', 'W', 3]] = None,
        ret_flags: Optional[List[str]] = None
    ) -> Tuple[TensorType[..., 3], TensorType[..., 3]]:
        """Render a new image, given a camera pose.

        Args:
            pose (TensorType[4, 4]): Input camera pose.
            img (TensorType['H', 'W', 3]): Ground truth image (optional). Used for getting target
                color values.

        Returns:
            output (dict): Dict of all output tensors. See key below:
                'rgb_map' (TensorType['K', 3]): Output RGB values for each pixel. Always included.
                'target' (TensorType['K', 3]): Target RGB values for each pixel. Included if 'img'
                    is not None.
                'pts' (TensorType['K*N', 3]): Positions of all point samples. (*)
                'dirs' (TensorType['K*N', 3]): Directions of all point samples. (*)
                'rgbs' (TensorType['K', 'N', 3]): Predicted RGB colors of all point samples. (*)
                'densities' (TensorType['K', 'N']): Predicted densities of all point samples. (*)

                (*) Included if specified in 'ret_flags'.
        """
        output = {}
        if ret_flags is None:
            ret_flags = []

        # Generate rays
        precrop_frac, rays_bsize = None, None
        if self.precrop:
            precrop_frac = self.train_cfg.precrop_fraction
        if not self.all_rays:
            rays_bsize = self.train_cfg.num_rays_per_batch

        grid_dims = (256, 256) if self.reduce_size else None
        rays, output['target'] = nerf_lib.generate_rays(
            pose, self.dataset, img, precrop=precrop_frac, bsize=rays_bsize, grid_dims=grid_dims)
        dirs = rays.viewdirs()

        # Sample points
        pts, dists = nerf_lib.sample_points(rays)
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = torch.repeat_interleave(dirs, repeats=self.net_cfg.num_samples_per_ray, dim=0)

        # Evaluate model
        # TODO: fix hardcode
        rgbs = torch.empty((len(pts_flat), 6), device=self.device)
        densities = torch.empty((len(pts_flat), 1), device=self.device)
        utils.batch_exec(self.model, rgbs, densities,
                         bsize=self.net_cfg.pts_bsize)(pts_flat, dirs_flat)
        rgbs = rgbs.reshape(*dists.shape, 6)
        densities = densities.reshape(dists.shape)

        output['pts'] = pts_flat if 'pts' in ret_flags else None
        output['dirs'] = dirs_flat if 'dirs' in ret_flags else None
        del pts_flat, dirs_flat

        # Integrate points
        rgb_c, rgb_s = torch.split(rgbs, [3, 3], dim=-1)
        del rgbs

        output['rgb_map'] = nerf_lib.integrate_points(dists, rgb_c, densities, self.bg_color)
        output['style_map'] = nerf_lib.integrate_points(dists, rgb_s, densities, self.bg_color)

        output['rgb_c'] = rgb_c if 'rgb_c' in ret_flags else None
        output['rgb_s'] = rgb_s if 'rgb_s' in ret_flags else None
        output['densities'] = densities if 'densities' in ret_flags else None
        return output
