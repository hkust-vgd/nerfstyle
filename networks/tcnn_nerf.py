import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn
from typing import Optional

from common import TensorModule, BBox
from config import NetworkConfig
from networks.fx_vae import StyleVAEExtractor


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
                'otype': 'HashGrid',
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

    def forward(self, pts, dirs=None, **kwargs):
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


class StyleTCNerf(TensorModule):
    def __init__(
        self,
        cfg: NetworkConfig,
        bbox: BBox,
        enc_dtype: Optional[torch.dtype] = None,
        use_dir: bool = True,
    ) -> None:
        super(StyleTCNerf, self).__init__()

        self.cfg = cfg
        self.bounds_bbox = bbox
        self.use_dir = use_dir

        pos_enc_cfg = self.cfg.pos_enc
        max_res = pos_enc_cfg.max_res_coeff * torch.max(self.bounds_bbox.size).item()
        per_lvl_scale = np.exp2(np.log2(max_res / pos_enc_cfg.min_res) / (pos_enc_cfg.n_lvls - 1))

        self.x_density_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',
                'n_levels': pos_enc_cfg.n_lvls,
                'n_features_per_level': pos_enc_cfg.n_feats_per_lvl,
                'log2_hashmap_size': pos_enc_cfg.hashmap_size,
                'base_resolution': pos_enc_cfg.min_res,
                'per_level_scale': per_lvl_scale
            },
            seed=self.cfg.network_seed,
            dtype=enc_dtype
        )

        self.x_color_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',
                'n_levels': pos_enc_cfg.n_lvls,
                'n_features_per_level': pos_enc_cfg.n_feats_per_lvl,
                'log2_hashmap_size': pos_enc_cfg.hashmap_size,
                'base_resolution': pos_enc_cfg.min_res,
                'per_level_scale': per_lvl_scale
            },
            seed=self.cfg.network_seed,
            dtype=enc_dtype
        )

        self.x_style_embedders = torch.nn.ModuleList([
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': pos_enc_cfg.n_lvls,
                    'n_features_per_level': pos_enc_cfg.n_feats_per_lvl,
                    'log2_hashmap_size': pos_enc_cfg.hashmap_size,
                    'base_resolution': pos_enc_cfg.min_res,
                    'per_level_scale': per_lvl_scale
                },
                seed=self.cfg.network_seed,
                dtype=enc_dtype
            ) for _ in range(8)
        ])

        self.d_embedder = None
        if use_dir:
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
            n_input_dims=self.x_density_embedder.n_output_dims,
            n_output_dims=1,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.cfg.density_hidden_dims,
                'n_hidden_layers': self.cfg.density_hidden_layers
            },
            seed=self.cfg.network_seed
        )

        self.color1_net = tcnn.Network(
            n_input_dims=self.x_color_embedder.n_output_dims,
            n_output_dims=16,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.cfg.density_hidden_dims,
                'n_hidden_layers': self.cfg.density_hidden_layers
            },
            seed=self.cfg.network_seed
        )

        rgb_net_input_dims = self.color1_net.n_output_dims
        if use_dir:
            rgb_net_input_dims += self.d_embedder.n_output_dims

        self.color2_net = tcnn.Network(
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

        # style_in_dims = 512
        # style_out_dims = 32

        # self.style_fx = StyleVAEExtractor().cuda()
        # self.style_net = tcnn.Network(
        #     n_input_dims=style_in_dims,
        #     n_output_dims=style_out_dims,
        #     network_config={
        #         'otype': 'FullyFusedMLP',
        #         'activation': 'ReLU',
        #         'output_activation': 'None',
        #         'n_neurons': 64,
        #         'n_hidden_layers': 3
        #     },
        #     seed=self.cfg.network_seed
        # )

        # self.fixed_style_vectors = torch.load('style_vectors.pth').cuda()

    def forward(self, pts, dirs=None, style_input=None):
        pts = self.bounds_bbox.normalize(pts)
        x_embedded = self.x_density_embedder(pts)
        density_output = self.density_net(x_embedded)
        sigmas = trunc_exp(density_output)

        if dirs is None and style_input is None:
            return sigmas

        if style_input is None:
            # random_id = np.random.randint(400)
            # style_vectors = self.style_net(self.fixed_style_vectors[random_id].unsqueeze(0))
            x_color_embedded = self.x_color_embedder(pts)
        else:
            style_images, style_ids = style_input
            # style_vectors = self.style_net(self.style_fx(style_images))
            x_color_embedded = self.x_style_embedders[style_ids.item()](pts)

        # style_vectors = torch.tile(style_vectors, (len(pts), 1))
        # x_color_embedded = torch.cat((x_color_embedded, style_vectors), dim=-1)
        color1_output = self.color1_net(x_color_embedded)

        if self.use_dir:
            dirs = (dirs + 1) / 2
            d_embedded = self.d_embedder(dirs)
            rgb_input = torch.cat((color1_output, d_embedded), dim=-1)
        else:
            rgb_input = color1_output

        rgbs = self.color2_net(rgb_input)
        return rgbs, sigmas
