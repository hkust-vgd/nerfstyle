from __future__ import annotations
import numpy as np
import torch

from common import OccupancyGrid
from config import DatasetConfig, NetworkConfig
from data import load_bbox
from networks.embedder import Embedder, MultiEmbedder
from networks.linears import MultiLinear, StaticMultiLinear, DynamicMultiLinear
from nerf_lib import nerf_lib
from .nerf import Nerf
from .single_nerf import SingleNerf


class MultiNerf(Nerf):
    def __init__(
        self,
        net_cfg: NetworkConfig,
        num_nets: int
    ) -> None:
        """
        Composite of multiple NeRF MLP networks.

        Args:
            net_cfg (NetworkConfig): Network configutation.
            num_nets (int): No. of networks.
        """
        self.net_cfg = net_cfg
        self.num_nets = num_nets

        if net_cfg.network_seed is not None:
            MultiLinear.set_rng_cm(net_cfg.network_seed)

        super().__init__(net_cfg)

    @torch.no_grad()
    def extract(self, idx: int) -> Nerf:
        single_model = SingleNerf(self.net_cfg)
        for name, module in self.named_modules():
            if isinstance(module, MultiLinear):
                single_module = single_model.get_submodule(name)
                single_module.weight.data = module.weight.data[idx]
                single_module.bias.data = module.bias.data[idx, 0]

        return single_model


class StaticMultiNerf(MultiNerf):
    def __init__(
        self,
        net_cfg: NetworkConfig,
        num_nets: int
    ) -> None:
        """
        Accepts a 2D batch (N, B) of input, i.e. each sub-network receives the same no. of input
        samples to evaluate. Used during distillation stage.

        Args:
            net_cfg (NetworkConfig): Network configuration.
            num_nets (int): No. of networks.
        """
        super().__init__(net_cfg, num_nets)

    @staticmethod
    def get_embedder(enc_counts):
        return MultiEmbedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return StaticMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)


class DynamicMultiNerf(MultiNerf):
    def __init__(
        self,
        net_cfg: NetworkConfig,
        dataset_cfg: DatasetConfig
    ) -> None:
        """
        Accepts a 1D batch (B, ) of input. Each input sample is delegated to corresponding
        sub-network based on its position. Each sub-network receives an unequal no. of input
        samples. Used during finetuning stage and inference.

        Args:
            net_cfg (NetworkConfig): Network configuration.
            dataset_cfg (DatasetConfig): Dataset configuration.
        """
        super().__init__(net_cfg, np.prod(dataset_cfg.net_res))
        self.occ_map = None

        # For dynamic evaluation
        bbox = load_bbox(dataset_cfg)
        if dataset_cfg.replica_cfg is not None:
            bbox.scale(dataset_cfg.replica_cfg.scale_factor)
        self.global_min_pt = bbox.min_pt.clone().detach()
        self.global_max_pt = bbox.max_pt.clone().detach()
        self.net_res = torch.tensor(dataset_cfg.net_res)
        self.mid_pts = torch.empty((self.num_nets, 3))

        self.voxel_size = (self.global_max_pt - self.global_min_pt) / self.net_res
        self.basis = torch.tensor(
            [self.net_res[2] * self.net_res[1], self.net_res[2], 1],
            dtype=torch.long, device=self.device)

        # Last element (always False) is for index == -1
        self.is_active = torch.zeros(self.num_nets + 1, dtype=torch.bool, device=self.device)
        self._ready = False

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return DynamicMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)

    def load_ckpt(self, ckpt):
        super().load_ckpt(ckpt)

        self.mid_pts = ckpt['mid_pts'].to(self.device)
        self.is_active = ckpt['active'].to(self.device)

        # Model is ready if can succesfully from existing state dict
        self._ready = True

    def save_ckpt(self, ckpt):
        ckpt = super().save_ckpt(ckpt)
        ckpt['mid_pts'] = self.mid_pts.cpu()
        ckpt['active'] = self.is_active.cpu()
        return ckpt

    def load_nodes(self, nodes):
        # assert len(nodes) == self.num_nets
        nodes.sort(key=(lambda node: node['idx']))

        min_pt = np.ones(3) * np.inf
        max_pt = np.ones(3) * -np.inf

        for idx, node in enumerate(nodes):
            min_pt = np.minimum(min_pt, node['min_pt'])
            max_pt = np.maximum(max_pt, node['max_pt'])
            mid_pt = (node['min_pt'] + node['max_pt']) / 2
            self.mid_pts[idx] = torch.from_numpy(mid_pt).to(self.device)
            if not node['started']:
                continue

            self.is_active[idx] = True
            single_std = node['model']
            for name, module in self.named_modules():
                if isinstance(module, MultiLinear):
                    module.weight.data[idx] = single_std[name + '.weight']
                    module.bias.data[idx, 0] = single_std[name + '.bias']

        # Model is ready if sub-networks span the global domain
        self._ready = (
            np.allclose(self.global_min_pt.cpu().numpy(), min_pt) and
            np.allclose(self.global_max_pt.cpu().numpy(), max_pt)
        )

    def load_occ_map(self, map_path):
        self.occ_map = OccupancyGrid.load(map_path, self.logger).to(self.device)
        self.logger.info('Loaded occupancy map "{}"'.format(map_path))

    # TODO: reduce overlap with OccupancyGrid
    def map_to_nets_indices(self, pts, masks):
        epsilon = 1e-5
        invalid = [(pts >= self.global_max_pt - epsilon), (pts < self.global_min_pt + epsilon)]
        invalid = torch.any(torch.cat(invalid, dim=-1), dim=-1)  # (N, )
        indices = (pts - self.global_min_pt) / self.voxel_size
        indices = torch.sum(indices.to(self.basis) * self.basis, dim=-1)
        indices[invalid] = -1

        active_mask = self.is_active[indices]
        for mask in masks:
            active_mask = torch.logical_and(active_mask, mask)

        inactive_mask = torch.logical_not(active_mask)
        indices[inactive_mask] = -1
        return indices, torch.sum(inactive_mask).item()

    def forward(self, pts, dirs=None, ert_mask=None):
        assert self._ready

        masks = []
        if self.occ_map is not None:
            masks.append(self.occ_map(pts))
        if ert_mask is not None:
            masks.append(ert_mask)
        net_indices, valid = self.map_to_nets_indices(pts, masks)

        # No points to evaluate
        if (valid == len(pts)):
            rgbs = torch.zeros((len(pts), 3), device=self.device)
            densities = torch.zeros((len(pts), 1), device=self.device)
            return rgbs, densities

        net_indices, order = torch.sort(net_indices)
        counts = torch.zeros(self.num_nets, dtype=torch.int64, device=self.device)
        active_nets, active_counts = torch.unique_consecutive(
            net_indices[valid:], return_counts=True)
        counts[active_nets] = active_counts
        sorted_pts, sorted_dirs = pts[order], dirs[order]

        # Assert all indices assigned to valid nets
        assert torch.all(self.is_active[net_indices[valid:]])

        # Perform global-to-local mapping
        nerf_lib.global_to_local(sorted_pts[valid:], self.mid_pts, self.voxel_size, counts)

        sorted_rgbs = torch.zeros((len(pts), 3), device=self.device)
        sorted_densities = torch.zeros((len(pts), 1), device=self.device)
        sorted_rgbs[valid:], sorted_densities[valid:] = \
            super().forward(sorted_pts[valid:], sorted_dirs[valid:], counts)

        rgbs = torch.empty_like(sorted_rgbs)
        densities = torch.empty_like(sorted_densities)
        rgbs[order], densities[order] = sorted_rgbs, sorted_densities

        return rgbs, densities