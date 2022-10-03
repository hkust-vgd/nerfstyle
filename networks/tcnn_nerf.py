import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn

from common import TensorModule, BBox
from config import NetworkConfig


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
    def __init__(
        self,
        cfg: NetworkConfig,
        bbox: BBox
    ) -> None:
        super(TCNerf, self).__init__()

        self.cfg = cfg
        self.bounds_bbox = bbox
        # TODO: parametrize

        self.n_output_dims = 15

        self.x_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',  # TODO: replace with 'DenseGrid'
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 16,
                'per_level_scale': 1.4062964748768472
            }
        )

        self.d_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4
            }
        )

        self.density_net = tcnn.Network(
            n_input_dims=self.x_embedder.n_output_dims,
            n_output_dims=self.n_output_dims + 1,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 64,
                'n_hidden_layers': 1
            }
        )

        rgb_net_input_dims = self.density_net.n_output_dims + self.n_output_dims
        self.rgb_net = tcnn.Network(
            n_input_dims=rgb_net_input_dims,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': 64,
                'n_hidden_layers': 2
            }
        )

    def save_ckpt(self, ckpt):
        ckpt['model'] = self.state_dict()
        return ckpt

    def forward(self, pts, dirs=None):
        pts = self.bounds_bbox.normalize(pts)
        x_embedded = self.x_embedder(pts)
        density_output = self.density_net(x_embedded)
        sigmas = trunc_exp(density_output[:, 0:1])

        if dirs is None:
            return sigmas

        dirs = (dirs + 1) / 2
        d_embedded = self.d_embedder(dirs)
        rgb_input = torch.cat((density_output[:, 1:], d_embedded), dim=-1)
        rgbs = self.rgb_net(rgb_input)
        return rgbs, sigmas
