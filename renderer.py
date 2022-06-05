from typing import Dict, List, Optional, TypeVar
from matplotlib import use
import numpy as np
import torch
from torchtyping import TensorType

from common import Intrinsics
from config import NetworkConfig
from nerf_lib import nerf_lib
from networks.nerf import Nerf
import utils

T = TypeVar('T', bound='Renderer')


class Renderer:
    def __init__(
        self: T,
        model: Nerf,
        net_cfg: NetworkConfig,
        intr: Intrinsics,
        near: float,
        far: float,
        name: str = 'Renderer',
        precrop_frac: float = 1.,
        num_rays: Optional[int] = None,
        use_ert: bool = False
    ) -> None:
        """
        NeRF renderer.

        Args:
            model (Nerf): Backbone model.
            net_cfg (NetworkConfig): Network configuration.
            intr (Intrinsics): Render camera intrinsics.
            near (float): Near plane distance.
            far (float): Far plane distance.
            name (str, optional): Logging name. Defaults to 'Renderer'.
            precrop_frac (float, optional): Cropping fraction. Defaults to 1.
            num_rays (Optional[int], optional): No. of rays to sample randomly. Defaults to None
                (render all rays in image).
            use_ert (bool, optional): Turn on early ray termination (ERT). Defaults to False.
        """
        super().__init__()
        self.model = model
        self.net_cfg = net_cfg
        self.logger = utils.create_logger(name)

        self.intr = intr
        self.near = near
        self.far = far

        self._use_precrop = False
        self.precrop_frac = precrop_frac
        self.use_ert = use_ert

        self.num_rays = num_rays
        self.device = self.model.device

        # Set BG color as white
        self.bg_color = None
        # self.bg_color = torch.ones(3, device='cuda')

        self.logger.info('Renderer "{}" initialized'.format(name))
        self.clock = utils.Clock()

    @property
    def use_precrop(self):
        return self._use_precrop

    @use_precrop.setter
    def use_precrop(self, value: bool):
        if value != self._use_precrop:
            msg = 'Turining {} square cropping'.format('on' if value else 'off')
            self.logger.info(msg)
            self._use_precrop = value

    def render(
        self: T,
        pose: TensorType[4, 4],
        img: Optional[TensorType['H', 'W', 3]] = None,
        ret_flags: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
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
                'trans_map' (TensorType['K', 1]): No. of ERT passes for each pixel. (*)
                'pts' (TensorType['K', 'N', 3]): Positions of all point samples. (*)
                'dirs' (TensorType['K', 3]): Directions of all point samples. (*)
                'rgbs' (TensorType['K', 'N', 3]): Predicted RGB colors of all point samples. (*)
                'densities' (TensorType['K', 'N']): Predicted densities of all point samples. (*)

                (*) Included if specified in 'ret_flags'.
        """

        torch.cuda.empty_cache()
        output = {}
        if ret_flags is None:
            ret_flags = []

        # Generate rays
        precrop_frac = self.precrop_frac if self._use_precrop else 1.
        rays, output['target'] = nerf_lib.generate_rays(
            pose, self.intr, img, precrop=precrop_frac, bsize=self.num_rays)
        dirs = rays.viewdirs()

        # Sample points
        num_samples = self.net_cfg.num_samples_per_ray
        pts, dists = nerf_lib.sample_points(rays, self.near, self.far, num_samples)
        dirs = dirs.unsqueeze(1).tile((1, num_samples, 1))

        # Evaluate model
        num_samples_per_pass = self.net_cfg.ert_bsize if self.use_ert else num_samples
        trans_threshold = self.net_cfg.ert_trans_thres

        rgb_buf = torch.zeros((len(pts), 3), device=self.device)
        acc_buf = torch.zeros((len(pts), 1), device=self.device)
        trans_buf = torch.ones((len(pts), 1), device=self.device)
        if 'trans_map' in ret_flags:
            output['trans_map'] = torch.zeros((len(pts), 1), device=self.device)

        output['pts'] = pts if 'pts' in ret_flags else None
        output['dirs'] = dirs[:, 0, :] if 'dirs' in ret_flags else None

        if 'rgbs' in ret_flags:
            output['rgbs'] = torch.zeros((len(pts), num_samples, 3), device=self.device)
        if 'densities' in ret_flags:
            output['densities'] = torch.zeros((len(pts), num_samples), device=self.device)

        total_passes = np.ceil(num_samples / num_samples_per_pass)
        for start in range(0, num_samples, num_samples_per_pass):
            end = min(num_samples, start + num_samples_per_pass)
            pts_flat = pts[:, start:end].reshape(-1, 3)
            dirs_flat = dirs[:, start:end].reshape(-1, 3)

            active_rays = (trans_buf > trans_threshold).squeeze(1)
            if torch.sum(active_rays) == 0:
                break

            if 'trans_map' in ret_flags:
                output['trans_map'][:, 0] += active_rays.to(torch.float32) / total_passes
            active_pts = active_rays.repeat_interleave(end - start)

            rgbs = torch.empty((len(pts_flat), 3), device=self.device)
            densities = torch.empty((len(pts_flat), 1), device=self.device)
            utils.batch_exec(self.model, rgbs, densities,
                             bsize=self.net_cfg.pts_bsize)(pts_flat, dirs_flat, active_pts)

            rgbs = rgbs.reshape((len(pts), end - start, 3))
            densities = densities.reshape((len(pts), end - start))

            if 'rgbs' in ret_flags:
                output['rgbs'][:, start:end] = rgbs
            if 'densities' in ret_flags:
                output['densities'][:, start:end] = densities

            integrate_bsize = self.net_cfg.pixels_bsize
            utils.batch_exec(nerf_lib.integrate_points, rgb_buf, acc_buf, trans_buf,
                             bsize=integrate_bsize)(
                dists[:, start:end], rgbs, densities, rgb_buf, acc_buf, trans_buf)

        output['rgb_map'] = rgb_buf
        if self.bg_color is not None:
            output['rgb_map'] += (1 - acc_buf) * self.bg_color
        return output
