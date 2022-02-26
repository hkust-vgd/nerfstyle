from contextlib import nullcontext
from math import sqrt

import torch
from torch import nn
from torchtyping import TensorType

from nerf_lib import nerf_lib
import utils


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
        cls.rng_cm = utils.RNGContextManager(seed)

    def __init__(
        self,
        num_networks: int,
        in_features: int,
        out_features: int,
        activation: str,
    ) -> None:
        super().__init__()

        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(
            num_networks, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(
            num_networks, 1, out_features))

        with MultiLinear.rng_cm:
            nn.init.kaiming_uniform_(self.weight, a=sqrt(5),
                                     nonlinearity=activation)
            standard_uniform_(self.bias)

    def __repr__(self) -> str:
        attrs = ['num_networks', 'in_features', 'out_features']
        return utils.get_repr(self, attrs)


class StaticMultiLinear(MultiLinear):
    def __init__(self, *multilinear_args) -> None:
        super().__init__(*multilinear_args)

    def forward(
        self,
        x: TensorType['num_networks', 'batch_size', 'in_channels']
    ) -> TensorType['num_networks', 'batch_size', 'out_channels']:
        weight_transpose = self.weight.permute(0, 2, 1)
        product = torch.bmm(x, weight_transpose)
        result = product + self.bias
        return result


class DynamicMultiLinear(MultiLinear):
    def __init__(self, *multilinear_args) -> None:
        super().__init__(*multilinear_args)

        self.group_limits = [2048, 1024]
        self.aux_index = nerf_lib.init_multimatmul_aux_data(
            self.num_networks, self.out_features,
            self.in_features, self.group_limits)

    def forward(
        self,
        x: TensorType['batch_size', 'in_channels'],
        counts: TensorType['num_networks']
    ) -> TensorType['batch_size', 'out_channels']:
        nerf_lib.multimatmul(
            self.weight, self.bias, x, counts.cpu(),
            self.group_limits, self.aux_index)

        weight_transpose = self.weight.permute(0, 2, 1)
        result = torch.empty((len(x), self.out_features)).to(self.weight)
        ptr = 0
        for idx, count in enumerate(counts):
            # (O, ) + (B, I) @ (I, O) = (B, O)
            result[ptr:ptr+count] = torch.addmm(
                self.bias[idx, 0], x[ptr:ptr+count], weight_transpose[idx])
            ptr += count
        return result
