from __future__ import annotations
from re import A
from typing import Optional
import numpy as np
import torch

from config import DatasetConfig, NetworkConfig
from networks.embedder import Embedder, MultiEmbedder
from networks.linears import MultiLinear, StaticMultiLinear, DynamicMultiLinear
from nerf_lib import NerfLib
from occ_map import OccupancyGrid
import utils
from .nerf import Nerf


class MultiNerf(Nerf):
    def __init__(
        self,
        num_nets: int,
        dataset_cfg: DatasetConfig,
        network_seed: Optional[int],
        **nerf_params
    ) -> None:
        """Composite of multiple NeRF MLP networks.

        Args:
            num_nets (int): No. of networks.
            dataset_cfg (DatasetConfig): Dataset config file.
            nerf_params: refer to parent class documentation.
        """
        self._nerf_params = nerf_params
        self.num_nets = num_nets

        if network_seed is not None:
            MultiLinear.set_rng_cm(network_seed)

        super().__init__(**nerf_params)

        # Register buffers for dynamic evaluation
        bbox_path = dataset_cfg.root_path / 'bbox.txt'
        self.global_min_pt, self.global_max_pt = \
            utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)
        self.net_res = dataset_cfg.net_res
        self.mid_pts = np.empty((num_nets, 3))
        self.voxel_size, self.basis = None, None

        self._full = (self.num_nets == np.prod(dataset_cfg.net_res))
        self._ready = False

    @classmethod
    def create_nerf(
        cls,
        num_nets: int,
        net_cfg: NetworkConfig,
        dataset_cfg: DatasetConfig
    ) -> MultiNerf:
        nerf_config = {
            'x_enc_counts': net_cfg.x_enc_count,
            'd_enc_counts': net_cfg.d_enc_count,
            'x_layers': 2,
            'x_width': 32,
            'd_widths': [32, 32],
            'activation': net_cfg.activation
        }
        model = cls(num_nets, dataset_cfg,
                    net_cfg.network_seed, **nerf_config)
        return model

    @torch.no_grad()
    def extract(self, idx: int) -> Nerf:
        single_model = Nerf(**self._nerf_params)
        for name, module in self.named_modules():
            if isinstance(module, MultiLinear):
                single_module = single_model.get_submodule(name)
                single_module.weight.data = module.weight.data[idx]
                single_module.bias.data = module.bias.data[idx, 0]

        return single_model

    def load_nodes(self, nodes, device):
        assert len(nodes) == self.num_nets

        # Assuming checkpoint is already in grid order
        min_pt = np.ones(3) * np.inf
        max_pt = np.ones(3) * -np.inf

        for idx, node in enumerate(nodes):
            single_std = node['model']
            for name, module in self.named_modules():
                if isinstance(module, MultiLinear):
                    module.weight.data[idx] = single_std[name + '.weight']
                    module.bias.data[idx, 0] = single_std[name + '.bias']
            min_pt = np.minimum(min_pt, node['min_pt'])
            max_pt = np.maximum(max_pt, node['max_pt'])
            self.mid_pts[idx] = (node['min_pt'] + node['max_pt']) / 2

        # Model is ready if (1) all sub-networks are loaded and
        # (2) sub-networks span the global domain
        self._ready = (
            self._full and
            np.allclose(self.global_min_pt, min_pt) and
            np.allclose(self.global_max_pt, max_pt)
        )

        # Move dataset metadata onto device for later computation
        for k in ['global_min_pt', 'global_max_pt', 'net_res', 'mid_pts']:
            v = getattr(self, k)
            setattr(self, k, torch.FloatTensor(v).to(device))

        self.voxel_size = (self.global_max_pt - self.global_min_pt) \
            / self.net_res
        self.basis = torch.LongTensor([
            self.net_res[2] * self.net_res[1], self.net_res[2], 1])
        self.basis = self.basis.to(device)


class StaticMultiNerf(MultiNerf):
    def __init__(self, *params, **nerf_params) -> None:
        """
        Accepts a 2D batch (N, B) of input, i.e. each sub-network receives the
        same no. of input samples to evaluate. Used during distillation stage.
        """
        super().__init__(*params, **nerf_params)

    @staticmethod
    def get_embedder(enc_counts):
        return MultiEmbedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return StaticMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)


class DynamicMultiNerf(MultiNerf):
    def __init__(self, *params, **nerf_params) -> None:
        """
        Accepts a 1D batch (B, ) of input. Each input sample is delegated to
        corresponding sub-network based on its position. Each sub-network
        receives an unequal no. of input samples. Used during finetuning stage
        and inference.
        """
        super().__init__(*params, **nerf_params)
        self.occ_map = None
        self.clock = utils.Clock()

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    def get_linear(self, in_channels, out_channels):
        return DynamicMultiLinear(
            self.num_nets, in_channels, out_channels, self.activation)

    def load_occ_map(self, map_path, device):
        self.occ_map = OccupancyGrid.load(map_path, self.logger).to(device)
        self.logger.info('Loaded occupancy map "{}"'.format(map_path))

    def map_to_nets_indices(self, pts):
        invalid = [(pts >= self.global_max_pt), (pts < self.global_min_pt)]
        invalid = torch.any(torch.cat(invalid, dim=-1), dim=-1)  # (N, )
        indices = (pts - self.global_min_pt) / self.voxel_size
        indices = torch.sum(indices.to(self.basis) * self.basis, dim=-1)
        indices[invalid] = -1
        return indices, torch.sum(invalid).item()

    def map_to_local(self, global_pts, counts):
        local_pts = torch.empty_like(global_pts)
        ptr = 0
        for mid_pt, count in zip(self.mid_pts, counts):
            local_pts[ptr:ptr+count] = global_pts[ptr:ptr+count] - mid_pt
            ptr += count
        local_pts /= (self.voxel_size / 2)
        return local_pts

    def forward(self, pts, dirs=None, *_):
        assert self._ready
        self.clock.reset()

        net_indices, valid = self.map_to_nets_indices(pts)
        if self.occ_map is not None:
            net_indices = torch.where(self.occ_map(pts), net_indices, -1)
            valid = torch.sum(net_indices < 0).item()

        net_indices, order = torch.sort(net_indices)
        counts = torch.bincount(net_indices[valid:], minlength=self.num_nets)
        sorted_pts, sorted_dirs = pts[order], dirs[order]
        self.clock.click('sort + filter dirs')

        # Perform global-to-local mapping
        NerfLib.global_to_local(
            sorted_pts[valid:], self.mid_pts, self.voxel_size, counts)
        self.clock.click('global to local')

        sorted_rgbs = torch.zeros((len(pts), 3)).to(sorted_pts)
        sorted_densities = torch.zeros((len(pts), 1)).to(sorted_pts)
        sorted_rgbs[valid:], sorted_densities[valid:] = \
            super().forward(sorted_pts[valid:], sorted_dirs[valid:], counts)
        self.clock.click('evaluate')

        rgbs = torch.empty_like(sorted_rgbs)
        densities = torch.empty_like(sorted_densities)
        rgbs[order], densities[order] = sorted_rgbs, sorted_densities

        return rgbs, densities
