import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn
from typing import Optional

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
        bbox: BBox,
        enc_dtype: Optional[torch.dtype] = None
    ) -> None:
        super(TCNerf, self).__init__()

        self.cfg = cfg
        self.bounds_bbox = bbox

        pos_enc_cfg = self.cfg.pos_enc
        max_res = pos_enc_cfg.max_res_coeff * torch.max(self.bounds_bbox.size).item()
        per_lvl_scale = np.exp2(np.log2(max_res / pos_enc_cfg.min_res) / (pos_enc_cfg.n_lvls - 1))

        self.x_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',  # TODO: replace with 'DenseGrid'
                'n_levels': pos_enc_cfg.n_lvls,
                'n_features_per_level': pos_enc_cfg.n_feats_per_lvl,
                'log2_hashmap_size': pos_enc_cfg.hashmap_size,
                'base_resolution': pos_enc_cfg.min_res,
                'per_level_scale': per_lvl_scale
            },
            seed=self.cfg.network_seed,
            dtype=enc_dtype
        )

        self.d_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': self.cfg.dir_enc_sh_deg
            },
            seed=self.cfg.network_seed,
            dtype=enc_dtype
        )

        self.density_net = tcnn.Network(
            n_input_dims=self.x_embedder.n_output_dims,
            n_output_dims=self.cfg.density_out_dims,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.cfg.density_hidden_dims,
                'n_hidden_layers': self.cfg.density_hidden_layers
            },
            seed=self.cfg.network_seed
        )

        rgb_net_input_dims = self.density_net.n_output_dims + self.cfg.density_out_dims - 1
        self.rgb_net = tcnn.Network(
            n_input_dims=rgb_net_input_dims,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': self.cfg.rgb_hidden_dims,
                'n_hidden_layers': self.cfg.rgb_hidden_layers
            },
            seed=self.cfg.network_seed
        )

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
