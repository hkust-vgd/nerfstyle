from contextlib import nullcontext
from math import sqrt
from typing import Optional
import torch
from torch import nn
from torchtyping import TensorType

from config import NetworkConfig
from networks.embedder import MultiEmbedder
from utils import RNGContextManager
from .nerf import Nerf


# Patch function with version generalized for multi-model networks.
# NOTE: Explicit references to this function in modules loaded prior
# to this point will not be affected by this patch.
nn.init._calculate_fan_in_and_fan_out = lambda t: (t.size(-1), t.size(-2))


def standard_uniform_(tensor):
    bound = 1. / sqrt(tensor.size(-1))
    nn.init.uniform_(tensor, -bound, bound)


class MultiLinear(nn.Module):
    rng_cm = nullcontext()

    @classmethod
    def set_rng_cm(cls, seed: int):
        cls.rng_cm = RNGContextManager(seed)

    def __init__(
        self,
        num_networks: int,
        in_features: int,
        out_features: int,
        activation: str,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(
            num_networks, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(
            num_networks, 1, out_features))

        with MultiLinear.rng_cm:
            nn.init.kaiming_uniform_(self.weight, a=sqrt(5),
                                     nonlinearity=activation)
            standard_uniform_(self.bias)

    def forward(
        self,
        x: TensorType['num_networks', 'batch_size', 'in_channels']
    ) -> TensorType['num_networks', 'batch_size', 'out_channels']:
        weight_transpose = self.weight.permute(0, 2, 1)
        product = torch.bmm(x, weight_transpose)
        result = product + self.bias
        return result


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
        self.num_networks = num_networks
        if network_seed is not None:
            MultiLinear.set_rng_cm(network_seed)

        super().__init__(**nerf_params)

    def _create_embedders(self, x_enc_counts, d_enc_counts):
        self.x_embedder = MultiEmbedder(x_enc_counts)
        self.d_embedder = MultiEmbedder(d_enc_counts)

    def get_linear(self, in_channels, out_channels):
        return MultiLinear(
            self.num_networks, in_channels, out_channels, self.activation)


def create_multi_nerf(
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
    model = MultiNerf(num_networks, net_cfg.network_seed, **nerf_config)
    return model
