import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn

from common import TensorModule


pos_encoding_config = {
    'otype': 'HashGrid',  # TODO: replace with 'DenseGrid'
    'n_levels': 16,
    'n_features_per_level': 2,
    'log2_hashmap_size': 19,
    'base_resolution': 16,
    'per_level_scale': 2.0,
    'interpolation': 'Linear'
}

dir_encoding_config = {
    'otype': 'SphericalHarmonics',
    'degree': 4
}

density_net_config = {
    'otype': 'FullyFusedMLP',
    'activation': 'ReLU',
    'output_activation': 'None',
    'n_neurons': 64,
    'n_hidden_layers': 1
}

rgb_net_config = {
    'otype': 'FullyFusedMLP',
    'activation': 'ReLU',
    'output_activation': 'Sigmoid',
    'n_neurons': 64,
    'n_hidden_layers': 2
}


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


class TCNerf(TensorModule):
    def __init__(self) -> None:
        super(TCNerf, self).__init__()

        self.n_output_dims = 16

        self.x_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=pos_encoding_config
        )

        self.d_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=dir_encoding_config
        )

        self.density_net = tcnn.Network(
            n_input_dims=self.x_embedder.n_output_dims,
            n_output_dims=self.n_output_dims,
            network_config=density_net_config
        )

        rgb_net_input_dims = self.density_net.n_output_dims + self.d_embedder.n_output_dims
        self.rgb_net = tcnn.Network(
            n_input_dims=rgb_net_input_dims,
            n_output_dims=3,
            network_config=rgb_net_config
        )

        self.bound = 3.5

    def save_ckpt(self, ckpt):
        ckpt['model'] = self.state_dict()
        return ckpt

    def _forward(self, pts, dirs=None):
        pts = (pts + self.bound) / (2 * self.bound)
        x_embedded = self.x_embedder(pts)
        density_output = self.density_net(x_embedded)
        sigmas = trunc_exp(density_output[:, 0:1])

        if dirs is None:
            return sigmas

        dirs = (dirs + 1) / 2
        d_embedded = self.d_embedder(dirs)
        rgb_input = torch.concat((density_output, d_embedded), axis=-1)
        rgbs = self.rgb_net(rgb_input)
        return rgbs, sigmas

    def forward(self, pts, dirs=None, ert_mask=None):
        if ert_mask is None or torch.sum(ert_mask) == len(pts):
            return self._forward(pts, dirs)

        assert dirs is not None
        rgbs = torch.zeros((len(pts), 3), device=self.device, dtype=torch.half)
        densities = torch.zeros((len(pts), 1), device=self.device, dtype=torch.half)

        if torch.sum(ert_mask) > 0:
            rgbs[ert_mask], densities[ert_mask] = self._forward(
                pts[ert_mask], dirs[ert_mask])
        return rgbs, densities
