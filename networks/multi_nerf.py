from __future__ import annotations
from re import A
from typing import Optional
import numpy as np
import torch

from config import DatasetConfig, NetworkConfig
from data import load_bbox
from networks.embedder import Embedder, MultiEmbedder
from networks.linears import MultiLinear, StaticMultiLinear, DynamicMultiLinear
from nerf_lib import nerf_lib
from occ_map import OccupancyGrid
import utils
from .nerf import Nerf


class MultiNerf(Nerf):
    def __init__(
        self,
        num_nets: int,
        network_seed: Optional[int],
        **nerf_params
    ) -> None:
        """Composite of multiple NeRF MLP networks.

        Args:
            num_nets (int): No. of networks.
            nerf_params: refer to parent class documentation.
        """
        self._nerf_params = nerf_params
        self.num_nets = num_nets

        if network_seed is not None:
            MultiLinear.set_rng_cm(network_seed)

        super().__init__(**nerf_params)

    @classmethod
    def _get_default_config(
        cls,
        net_cfg: NetworkConfig
    ) -> dict:
        nerf_config = {
            'x_enc_counts': net_cfg.x_enc_count,
            'd_enc_counts': net_cfg.d_enc_count,
            'x_layers': 2,
            'x_width': 32,
            'd_widths': [32, 32],
            'activation': net_cfg.activation
        }
        return nerf_config

    @torch.no_grad()
    def extract(self, idx: int) -> Nerf:
        single_model = Nerf(**self._nerf_params)
        for name, module in self.named_modules():
            if isinstance(module, MultiLinear):
                single_module = single_model.get_submodule(name)
                single_module.weight.data = module.weight.data[idx]
                single_module.bias.data = module.bias.data[idx, 0]

        return single_model


class StaticMultiNerf(MultiNerf):
    def __init__(self, *params, **nerf_params) -> None:
        """
        Accepts a 2D batch (N, B) of input, i.e. each sub-network receives the same no. of input
        samples to evaluate. Used during distillation stage.
        """
        super().__init__(*params, **nerf_params)

    @classmethod
    def create_nerf(
        cls,
        num_nets: int,
        net_cfg: NetworkConfig
    ) -> StaticMultiNerf:
        nerf_config = super()._get_default_config(net_cfg)
        model = cls(num_nets, net_cfg.network_seed, **nerf_config)
        return model

    @staticmethod
    def get_embedder(enc_counts):
        return MultiEmbedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return StaticMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)


class DynamicMultiNerf(MultiNerf):
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        *params,
        **nerf_params
    ) -> None:
        """
        Accepts a 1D batch (B, ) of input. Each input sample is delegated to corresponding
        sub-network based on its position. Each sub-network receives an unequal no. of input
        samples. Used during finetuning stage and inference.
        """

        super().__init__(*params, **nerf_params)
        self.occ_map = None

        # For dynamic evaluation
        self.global_min_pt, self.global_max_pt, _ = load_bbox(dataset_cfg)
        self.net_res = dataset_cfg.net_res
        self.mid_pts = np.empty((self.num_nets, 3))
        self.voxel_size, self.basis = None, None
        self.is_active = None

        assert self.num_nets == np.prod(dataset_cfg.net_res)
        self._ready = False

    @classmethod
    def create_nerf(
        cls,
        num_nets: int,
        net_cfg: NetworkConfig,
        dataset_cfg: DatasetConfig
    ) -> DynamicMultiNerf:
        nerf_config = super()._get_default_config(net_cfg)
        model = cls(dataset_cfg, num_nets, net_cfg.network_seed, **nerf_config)
        return model

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return DynamicMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)

    def load_nodes(
        self,
        nodes: list,
        device: torch.device
    ) -> None:
        assert len(nodes) == self.num_nets
        nodes.sort(key=(lambda node: node['idx']))

        # Last element (always False) is for index == -1
        self.is_active = torch.zeros(self.num_nets + 1, dtype=torch.bool)

        min_pt = np.ones(3) * np.inf
        max_pt = np.ones(3) * -np.inf

        for idx, node in enumerate(nodes):
            min_pt = np.minimum(min_pt, node['min_pt'])
            max_pt = np.maximum(max_pt, node['max_pt'])
            self.mid_pts[idx] = (node['min_pt'] + node['max_pt']) / 2
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
            np.allclose(self.global_min_pt, min_pt) and
            np.allclose(self.global_max_pt, max_pt)
        )

        # Move dataset metadata onto device for later computation
        for k in ['global_min_pt', 'global_max_pt', 'net_res', 'mid_pts']:
            v = getattr(self, k)
            setattr(self, k, torch.FloatTensor(v).to(device))

        self.voxel_size = (self.global_max_pt - self.global_min_pt) / self.net_res
        self.basis = torch.LongTensor([self.net_res[2] * self.net_res[1], self.net_res[2], 1])
        self.basis = self.basis.to(device)
        self.is_active = self.is_active.to(device)

    def load_occ_map(self, map_path, device):
        self.occ_map = OccupancyGrid.load(map_path, self.logger).to(device)
        self.logger.info('Loaded occupancy map "{}"'.format(map_path))

    def map_to_nets_indices(self, pts):
        epsilon = 1e-5
        invalid = [(pts >= self.global_max_pt - epsilon), (pts < self.global_min_pt + epsilon)]
        invalid = torch.any(torch.cat(invalid, dim=-1), dim=-1)  # (N, )
        indices = (pts - self.global_min_pt) / self.voxel_size
        indices = torch.sum(indices.to(self.basis) * self.basis, dim=-1)
        indices[invalid] = -1

        inactive_mask = torch.logical_not(self.is_active[indices])
        indices[inactive_mask] = -1
        return indices, torch.sum(inactive_mask).item()

    def forward(self, pts, dirs=None, *_):
        assert self._ready

        net_indices, valid = self.map_to_nets_indices(pts)
        if self.occ_map is not None:
            net_indices = torch.where(self.occ_map(pts), net_indices, -1)
            valid = torch.sum(net_indices < 0).item()

        if (valid == len(pts)):
            rgbs = torch.zeros((len(pts), 3)).to(pts)
            densities = torch.zeros((len(pts), 1)).to(pts)
            return rgbs, densities

        net_indices, order = torch.sort(net_indices)
        counts = torch.bincount(net_indices[valid:], minlength=self.num_nets)
        sorted_pts, sorted_dirs = pts[order], dirs[order]

        # TODO: Remove assertion if working
        assert torch.all(self.is_active[net_indices[valid:]])

        # Perform global-to-local mapping
        nerf_lib.global_to_local(sorted_pts[valid:], self.mid_pts, self.voxel_size, counts)

        sorted_rgbs = torch.zeros((len(pts), 3)).to(sorted_pts)
        sorted_densities = torch.zeros((len(pts), 1)).to(sorted_pts)
        sorted_rgbs[valid:], sorted_densities[valid:] = \
            super().forward(sorted_pts[valid:], sorted_dirs[valid:], counts)

        rgbs = torch.empty_like(sorted_rgbs)
        densities = torch.empty_like(sorted_densities)
        rgbs[order], densities[order] = sorted_rgbs, sorted_densities

        return rgbs, densities
