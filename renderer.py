import itertools
from math import *
from typing import Dict, Optional, Tuple, TypeVar
import torch
from torch import Tensor

from common import Box2D, Intrinsics, RayBatch, TensorModule
from config import RendererConfig
from nerf_lib import nerf_lib
from networks.style_nerf import StyleTCNerf
import raymarching
import utils

T = TypeVar('T', bound='Renderer')

STEP_CTR_SIZE = 16


class Renderer(TensorModule):
    def __init__(
        self: T,
        model: StyleTCNerf,
        cfg: RendererConfig,
        intr: Intrinsics,
        bound: float,
        name: str = 'Renderer',
        precrop_frac: float = 1.,
        raymarch_channels: int = 3
    ) -> None:
        """
        NeRF renderer.

        Args:
            model (TCNerf): Backbone model. Renderer is set to model device during init.
            net_cfg (NetworkConfig): Network configuration.
            intr (Intrinsics): Render camera intrinsics.
            name (str, optional): Logging name. Defaults to 'Renderer'.
            precrop_frac (float, optional): Cropping fraction. Defaults to 1.
            num_rays (Optional[int], optional): No. of rays to sample randomly. Defaults to None
                (render all rays in image).
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.logger = utils.create_logger(name)

        self.intr = intr
        self._use_precrop = False
        self.precrop_frac = precrop_frac
        self.raymarch_channels = raymarch_channels

        self.update_occ = True

        # Raymarching variables
        self.bound = bound
        # if self.cfg.use_ndc:
        #     self.aabb = torch.tensor([-bound, -bound, -1., bound, bound, 0.99])
        # else:
        self.aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound])

        self.cascade = 1 + ceil(log2(bound))
        grid_size = self.cfg.grid_size
        bitfield_size = self.cascade * (grid_size ** 3) // 8
        self.density_grid = torch.zeros((self.cascade, grid_size ** 3))
        self.density_bitfield = torch.zeros((bitfield_size, ), dtype=torch.uint8)
        self.step_counter = torch.zeros((STEP_CTR_SIZE, 2), dtype=torch.int32)
        self.local_step = 0
        self.mean_count = 0
        self.mean_density = 0

        self.logger.info('Renderer "{}" initialized'.format(name))
        self.clock = utils.Clock()

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def state_dict(self):
        state_dict = {'model': self.model.state_dict()}
        keys = [
            'intr', 'precrop_frac', 'raymarch_channels', 'bound',
            'density_grid', 'density_bitfield', 'step_counter',
            'local_step', 'mean_count', 'mean_density'
        ]
        for k in keys:
            v = getattr(self, k)
            if torch.is_tensor(v):
                v = v.detach()
            state_dict[k] = v

        return state_dict

    def load_state_dict(self, state_dict):
        validate_keys = ['intr', 'precrop_frac', 'raymarch_channels', 'bound']
        for k in validate_keys:
            if getattr(self, k) != state_dict[k]:
                self.logger.error('Values do not match when loading key "{}"'.format(k))

        self.model.load_state_dict(state_dict['model'])

        load_keys = ['density_grid', 'density_bitfield', 'step_counter',
                     'local_step', 'mean_count', 'mean_density']
        for k in load_keys:
            v = state_dict[k]
            if torch.is_tensor(v):
                v = v.to(self.device)
            setattr(self, k, v)

    @property
    def use_precrop(self):
        return self._use_precrop

    @use_precrop.setter
    def use_precrop(self, value: bool):
        if value != self._use_precrop:
            msg = 'Training {} square cropping'.format('on' if value else 'off')
            self.logger.info(msg)
            self._use_precrop = value

    def _compute_occ_sigmas(self, xyzs, cas):
        """Compute sigma values for raymarching occupancy grid.

        Args:
            xyzs (Tensor[N, 3]): Normalized 3D coordinates.
            cas (int): Cascade level.

        Returns:
            sigmas (Tensor[N, ]): Density values.
        """
        bound = min(2 ** cas, self.bound)
        half_grid_size = bound / self.cfg.grid_size
        cas_xyzs = xyzs * (bound - half_grid_size)
        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size

        sigmas = self.model(cas_xyzs).reshape(-1).detach() * self.cfg.density_scale
        return sigmas

    @torch.no_grad()
    def update_state(self) -> None:
        tmp_grid = -torch.ones_like(self.density_grid)

        if self.local_step < self.cfg.update_thres:
            # Full update: sample all CAS * (GRID_SIZE ** 3) cells
            bsize = self.cfg.grid_bsize
            if bsize is None:
                bsize = self.cfg.grid_size
            X, Y, Z = [torch.arange(
                self.cfg.grid_size, dtype=torch.int32, device=self.device).split(bsize)
                for _ in range(3)]

            for (xs, ys, zs) in itertools.product(X, Y, Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                coords = torch.cat([
                    xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                indices = raymarching.morton3D(coords).long()

                # Normalize to [-1, 1]
                xyzs = 2 * coords.float() / (self.cfg.grid_size - 1) - 1

                for cas in range(self.cascade):
                    tmp_grid[cas, indices] = self._compute_occ_sigmas(xyzs, cas)

        else:
            # Random sampling update
            self.tmp = True
            N = self.cfg.grid_size ** 3 // 4
            for cas in range(self.cascade):
                coords = torch.randint(0, self.cfg.grid_size, (N, 3), device=self.device)
                indices = raymarching.morton3D(coords).long()

                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)
                rand_mask = torch.randint(0, occ_indices.shape[0], [N],
                                          dtype=torch.long, device=self.device)
                occ_indices = occ_indices[rand_mask]
                occ_coords = raymarching.morton3D_invert(occ_indices)

                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)

                xyzs = 2 * coords.float() / (self.cfg.grid_size - 1) - 1
                tmp_grid[cas, indices] = self._compute_occ_sigmas(xyzs, cas)

        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(
            self.density_grid[valid_mask] * self.cfg.density_decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()

        density_thresh = min(self.mean_density, self.cfg.density_thresh)
        self.density_bitfield = raymarching.packbits(
            self.density_grid, density_thresh, self.density_bitfield)

        total_step = min(STEP_CTR_SIZE, self.cfg.update_iter)
        assert total_step > 0
        self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)

    def render_train(
        self,
        rays: RayBatch,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        z_hats = None
        # if self.cfg.use_ndc:
        #     z_hats = rays.dirs[:, 2]
        #     rays = rays.warp_ndc(1., self.intr)

        if self.update_occ and (self.local_step % self.cfg.update_iter == 0):
            self.update_state()

        nears, fars = raymarching.near_far_from_aabb(
            rays.origins, rays.dirs, self.aabb, self.cfg.min_near)

        if self.update_occ:
            counter = self.step_counter[self.local_step % STEP_CTR_SIZE]
            counter.zero_()
            self.local_step += 1
        else:
            counter = torch.zeros(2).to(self.step_counter)

        xyzs, dirs, deltas, rays_info = raymarching.march_rays_train(
            rays.origins, rays.dirs, z_hats, self.bound, self.density_bitfield,
            self.cascade, self.cfg.grid_size, nears, fars, counter, self.mean_count,
            True, 128, True, 0., self.cfg.max_steps, self.cfg.use_ndc)

        rgbs, sigmas = self.model(xyzs, dirs=dirs, **kwargs)
        sigmas = sigmas * self.cfg.density_scale

        weights_sum, depth, image = raymarching.composite_rays_train(
            sigmas, rgbs, deltas, rays_info, self.cfg.t_thresh, self.cfg.use_ndc)
        classes = image[:, 3:]
        image = image[:, :3]

        image = image + (1 - weights_sum).unsqueeze(-1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)

        return image, depth, classes

    def render_test(
        self,
        rays: RayBatch,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        z_hats = None
        # if self.cfg.use_ndc:
        #     z_hats = rays.dirs[:, 2]
        #     rays = rays.warp_ndc(1., self.intr)

        nears, fars = raymarching.near_far_from_aabb(
            rays.origins, rays.dirs, self.aabb, self.cfg.min_near)

        N = len(rays)
        weights_sum = torch.zeros(N, dtype=torch.float32, device=self.device)
        depth = torch.zeros(N, dtype=torch.float32, device=self.device)
        image = torch.zeros(N, self.raymarch_channels, dtype=torch.float32, device=self.device)

        n_alive = N
        rays_alive = torch.arange(n_alive, dtype=torch.int32, device=self.device)
        rays_t = nears.clone()
        # if self.cfg.use_ndc:
        #     nears_z = rays.lerp(nears)[:, 2]
        #     rays_t_phy = (2 / (nears_z - 1) + 1) / z_hats
        #     rays_t = torch.stack([rays_t, rays_t_phy], dim=-1)  # [N, 2]
        # else:
        rays_t = rays_t[:, None]  # [N, 1]

        step = 0
        while step < self.cfg.max_steps:
            n_alive = len(rays_alive)
            if n_alive <= 0:
                break

            n_step = max(min(N // n_alive, 8), 1)
            xyzs, dirs, deltas = raymarching.march_rays(
                n_alive, n_step, rays_alive, rays_t, rays.origins, rays.dirs, z_hats,
                self.bound, self.density_bitfield, self.cascade, self.cfg.grid_size,
                nears, fars, 128, False, 0., self.cfg.max_steps, self.cfg.use_ndc)

            rgbs, sigmas = self.model(xyzs, dirs=dirs, **kwargs)
            sigmas = sigmas * self.cfg.density_scale

            raymarching.composite_rays(
                n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, self.cfg.use_ndc,
                weights_sum, depth, image, self.cfg.t_thresh)

            rays_alive = rays_alive[rays_alive >= 0]
            step += n_step

        classes = image[:, 3:]
        image = image[:, :3]

        image = image + (1 - weights_sum).unsqueeze(-1)
        depth = torch.clamp(depth - nears, min=0) / (fars - nears)

        return image, depth, classes

    def render(
        self: T,
        pose,
        image = None,
        patch: Optional[Box2D] = None,
        num_rays: Optional[int] = None,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        output = {}

        precrop_frac = self.precrop_frac if self._use_precrop else 1.
        rays, output['target'] = nerf_lib.generate_rays(
            pose, self.intr, image, patch=patch, precrop=precrop_frac,
            bsize=num_rays, camera_flip=self.cfg.flip_camera)

        render_fn = self.render_train if training else self.render_test
        # output['rgb_map'], output['trans_map'], _ = render_fn(rays)
        output['rgb_map'], output['trans_map'], output['classes'] = render_fn(rays)
        return output


# class StyleRenderer(Renderer):
#     def __init__(
#         self,
#         model: StyleTCNerf,
#         base: Renderer,
#         new_cfg: RendererConfig
#     ):
#         super().__init__(
#             model, base.cfg, base.intr, base.bound, name='StyleRenderer'
#         )

#         # TODO: Need to overwrite some of the render configs
#         self.cfg.max_steps = new_cfg.max_steps

#         self.density_grid = base.density_grid
#         self.density_bitfield = base.density_bitfield
#         self.update_occ = False

#     def render(
#         self: T,
#         pose: TensorType[4, 4],
#         style_image: TensorType['H', 'W', 3],
#         image: Optional[TensorType['H', 'W', 3]] = None,
#         patch: Optional[Box2D] = None,
#         num_rays: Optional[int] = None,
#         training: bool = True
#     ) -> Dict[str, torch.Tensor]:
#         output = {}

#         precrop_frac = self.precrop_frac if self._use_precrop else 1.
#         rays, output['target'] = nerf_lib.generate_rays(
#             pose, self.intr, image, patch=patch, precrop=precrop_frac,
#             bsize=num_rays, camera_flip=self.cfg.flip_camera)

#         render_fn = self.render_train if training else self.render_test
#         output['rgb_map'], _ = render_fn(rays, style_input=style_image)
#         return output
