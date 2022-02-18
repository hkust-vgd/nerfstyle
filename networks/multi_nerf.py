from __future__ import annotations
from turtle import forward
from typing import List, Optional
import torch

from config import NetworkConfig
from networks.embedder import Embedder, MultiEmbedder
from networks.linears import MultiLinear, StaticMultiLinear, DynamicMultiLinear
from .nerf import Nerf


class MultiNerf(Nerf):
    def __init__(
        self,
        num_networks: int,
        network_seed: Optional[int],
        **nerf_params
    ) -> None:
        """Composite of multiple NeRF MLP networks.

        Args:
            num_networks (int): No. of networks.
            nerf_params: refer to parent class documentation.
        """
        self._nerf_params = nerf_params
        self.num_networks = num_networks
        if network_seed is not None:
            MultiLinear.set_rng_cm(network_seed)

        super().__init__(**nerf_params)

    @classmethod
    def create_nerf(
        cls,
        num_networks: int,
        net_cfg: NetworkConfig
    ) -> MultiNerf:
        nerf_config = {
            'x_enc_counts': net_cfg.x_enc_count,
            'd_enc_counts': net_cfg.d_enc_count,
            'x_layers': 2,
            'x_width': 32,
            'd_widths': [32, 32],
            'activation': net_cfg.activation
        }
        model = cls(num_networks, net_cfg.network_seed, **nerf_config)
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

    def load_state_dict(self, state_dict, strict):
        return super().load_state_dict(state_dict, strict)


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
            self.num_networks, in_channels, out_channels, self.activation)


class DynamicMultiNerf(MultiNerf):
    def __init__(self, *params, **nerf_params) -> None:
        """
        Accepts a 1D batch (B, ) of input. Each input sample is delegated to
        corresponding sub-network based on its position. Each sub-network
        receives an unequal no. of input samples. Used during finetuning stage
        and inference.
        """
        super().__init__(*params, **nerf_params)

    def get_linear(self, in_channels, out_channels):
        return DynamicMultiLinear(
            self.num_networks, in_channels, out_channels, self.activation)

    def forward(self, pt, dir=None):
        # TODO: map global to local coordinates
        super().forward(pt, dir)
