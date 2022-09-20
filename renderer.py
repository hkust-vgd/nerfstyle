import itertools
from operator import ne
from packaging import version as pver
from typing import Callable, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import torch
from torch import Tensor
from torchtyping import TensorType

from common import Intrinsics, RayBatch, TensorModule
from config import NetworkConfig
from nerf_lib import nerf_lib
from networks.nerf import Nerf
import raymarching
import utils

T = TypeVar('T', bound='Renderer')


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


class Raymarcher(TensorModule):
    def __init__(
        self,
        model: Callable,
        bound: float,
        update_iter: int,
        cascade: int = 1,
        grid_size: int = 128,
        min_near: float = 0.2,
        t_thresh: float = 1e-4
    ) -> None:
        super().__init__()

        self.model = model
        self.bound = bound
        self.cascade = cascade
        self.grid_size = grid_size
        self.update_iter = update_iter
        self.min_near = min_near
        self.t_thresh = t_thresh

        self.max_steps = 1024

        self.aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound])

        bitfield_size = cascade * (grid_size ** 3) // 8
        self.density_grid = torch.zeros((cascade, grid_size ** 3))
        self.density_bitfield = torch.zeros((bitfield_size, ), dtype=torch.uint8)
        self.step_counter = torch.zeros((update_iter, 2), dtype=torch.int32)
        self.local_step = 0
        self.mean_count = 0

        self.density_scale = 1
        self.density_thresh = 0.01
        self.mean_density = 0
        self.iter_density = 0
        self.full_iter_thresh = 16

    def _compute_occ_sigmas(self, xyzs, cas):
        bound = min(2 ** cas, self.bound)
        half_grid_size = bound / self.grid_size
        cas_xyzs = xyzs * (bound - half_grid_size)
        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size

        sigmas = self.model(cas_xyzs).reshape(-1).detach() * self.density_scale
        return sigmas

    @torch.no_grad()
    def update_state(
        self,
        decay: float = 0.95,
        S: int = 128
    ) -> None:
        tmp_grid = -torch.ones_like(self.density_grid)

        if self.iter_density < self.full_iter_thresh:
            # Full update
            X, Y, Z = [torch.arange(
                self.grid_size, dtype=torch.int32, device=self.device).split(S)
                for _ in range(3)]

            for (xs, ys, zs) in itertools.product(X, Y, Z):
                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                coords = torch.cat([
                    xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                indices = raymarching.morton3D(coords).long()
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1

                for cas in range(self.cascade):
                    tmp_grid[cas, indices] = self._compute_occ_sigmas(xyzs, cas)

        else:
            # Partial update
            N = self.grid_size ** 3 // 4
            for cas in range(self.cascade):
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.device)
                indices = raymarching.morton3D(coords).long()

                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)
                rand_mask = torch.randint(0, occ_indices.shape[0], [N],
                                          dtype=torch.long, device=self.device)
                occ_indices = occ_indices[rand_mask]
                occ_coords = raymarching.morton3D_invert(occ_indices)

                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)

                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1
                tmp_grid[cas, indices] = self._compute_occ_sigmas(xyzs, cas)

        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(
            self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()
        self.iter_density += 1

        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(
            self.density_grid, density_thresh, self.density_bitfield)

        total_step = min(self.update_iter, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def render(
        self,
        rays: RayBatch
    ) -> Tuple[Tensor, Tensor]:
        origin = torch.tile(rays.origin, (len(rays.dests), 1))
        nears, fars = raymarching.near_far_from_aabb(origin, rays.dests, self.aabb, self.min_near)

        counter = self.step_counter[self.local_step % self.update_iter]
        counter.zero_()
        self.local_step += 1

        xyzs, dirs, deltas, rays_info = raymarching.march_rays_train(
            origin, rays.dests, self.bound, self.density_bitfield, self.cascade, self.grid_size,
            nears, fars, counter, self.mean_count, True, 128, True, 0., self.max_steps)

        rgbs, sigmas = self.model(xyzs, dirs)
        sigmas = sigmas * self.density_scale

        weights_sum, depth, image = raymarching.composite_rays_train(
            sigmas, rgbs, deltas, rays_info, self.t_thresh)
        image = image + (1 - weights_sum).unsqueeze(-1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)

        return image, depth

    def render_test(
        self,
        rays: RayBatch
    ) -> Tuple[Tensor, Tensor]:
        origin = torch.tile(rays.origin, (len(rays.dests), 1))
        nears, fars = raymarching.near_far_from_aabb(origin, rays.dests, self.aabb, self.min_near)

        N = len(rays)
        weights_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        depth = torch.zeros(N, dtype=torch.float32, device=self.device)
        image = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        n_alive = N
        rays_alive = torch.arange(n_alive, dtype=torch.int32, device=self.device)
        rays_t = nears.clone()

        step = 0
        while step < self.max_steps:
            n_alive = len(rays_alive)
            if n_alive <= 0:
                break

            n_step = max(min(N // n_alive, 8), 1)
            xyzs, dirs, deltas = raymarching.march_rays(
                n_alive, n_step, rays_alive, rays_t, origin, rays.dests,
                self.bound, self.density_bitfield, self.cascade, self.grid_size,
                nears, fars, 128, False, 0., self.max_steps)

            rgbs, sigmas = self.model(xyzs, dirs)
            sigmas = sigmas * self.density_scale

            raymarching.composite_rays(
                n_alive, n_step, rays_alive, rays_t,
                sigmas, rgbs, deltas, weights_sum, depth, image, self.t_thresh)

            rays_alive = rays_alive[rays_alive >= 0]
            step += n_step

        image = image + (1 - weights_sum).unsqueeze(-1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)

        return image, depth


class Renderer:
    def __init__(
        self: T,
        model: Nerf,
        net_cfg: NetworkConfig,
        intr: Intrinsics,
        near: float,
        far: float,
        bg_color: str,
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
            bg_color (str): Background color.
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

        self.bg_color = torch.tensor(utils.color_str2rgb(bg_color), device=self.device)

        self.logger.info('Renderer "{}" initialized'.format(name))
        self.clock = utils.Clock()

        bound = self.model.bound
        self.update_step = 16
        self.raymarcher = Raymarcher(self.model, bound, self.update_step).to(self.device)

    @property
    def use_precrop(self):
        return self._use_precrop

    @use_precrop.setter
    def use_precrop(self, value: bool):
        if value != self._use_precrop:
            msg = 'Training {} square cropping'.format('on' if value else 'off')
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

        output['rgb_map'] = rgb_buf + (1 - acc_buf) * self.bg_color
        return output

    def render_raymarching(
        self: T,
        pose: TensorType[4, 4],
        img: Optional[TensorType['H', 'W', 3]] = None,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        output = {}

        # TODO: tmp
        num_rays = self.num_rays if training else None

        precrop_frac = self.precrop_frac if self._use_precrop else 1.
        rays, output['target'] = nerf_lib.generate_rays(
            pose, self.intr, img, precrop=precrop_frac, bsize=num_rays)

        render_fn = self.raymarcher.render if training else self.raymarcher.render_test
        output['rgb_map'], output['trans_map'] = render_fn(rays)
        return output

    def update_raymarching(self):
        self.raymarcher.update_state()
