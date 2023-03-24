from functools import partial
import torch
import tinycudann as tcnn
from typing import Callable, Optional, Tuple

from common import TensorModule, BBox
from config import NetworkConfig
import utils
from .tcnn_nerf import get_grid_encoder, trunc_exp


class StyleTCNerf(TensorModule):
    def __init__(
        self,
        cfg: NetworkConfig,
        bbox: BBox,
        class_dim: int,
        enc_dtype: Optional[torch.dtype] = None,
        use_dir: bool = True
    ) -> None:
        super(StyleTCNerf, self).__init__()

        self.cfg = cfg
        self.bounds_bbox = bbox
        self.use_dir = use_dir
        self.class_dim = class_dim

        max_bound = torch.max(self.bounds_bbox.size).item()
        self.x_density_embedder = get_grid_encoder(self.cfg, max_bound, enc_dtype)
        self.x_color_embedder = get_grid_encoder(self.cfg, max_bound, enc_dtype)

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

        self.class_net = tcnn.Network(
            n_input_dims=self.x_color_embedder.n_output_dims,
            n_output_dims=class_dim,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.cfg.density_hidden_dims,
                'n_hidden_layers': self.cfg.density_hidden_layers
            },
            seed=self.cfg.network_seed
        )

    # def init_style(self, num_styles):
    #     self.x_style_embedder = GridEncoder(
    #         input_dim=3,
    #         num_levels=16,
    #         level_dim=2,
    #         per_level_scale=self.x_color_embedder.per_level_scale,
    #         base_resolution=16,
    #         log2_hashmap_size=20,
    #         gridtype='hash',
    #         align_corners=True
    #     ).cuda()

    #     self.x_style_embedder.initialize(
    #         self.x_color_embedder.embeddings,
    #         self.x_color_embedder.offsets,
    #         num_styles=num_styles
    #     )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def _forward(self, pts, dirs=None):
        pts = self.bounds_bbox.normalize(pts)
        x_embedded = self.x_density_embedder(pts)
        density_output = self.density_net(x_embedded)
        sigmas = trunc_exp(density_output)

        if dirs is None:
            return sigmas

        x_color_embedded = self.x_color_embedder(pts)
        classes = self.class_net(x_color_embedded)
        color1_output = self.color1_net(x_color_embedded)

        if self.use_dir:
            dirs = (dirs + 1) / 2
            d_embedded = self.d_embedder(dirs)
            rgb_input = torch.cat((color1_output, d_embedded), dim=-1)
        else:
            rgb_input = color1_output

        rgbs = self.color2_net(rgb_input)
        rgbs = torch.cat((rgbs, classes), dim=1)
        return rgbs, sigmas

    def forward(self, pts, dirs=None, bsize=1000000):
        N = len(pts)
        if N < bsize:
            return self._forward(pts, dirs)
        else:
            sigmas = torch.empty((N, 1), device=self.device)

            if dirs is None:
                batch_fn = partial(self._forward, dirs=None)
                utils.batch_exec(batch_fn, sigmas, bsize=bsize)(pts)
                return sigmas

            batch_fn = partial(self._forward)
            rgbs = torch.empty((N, 3 + self.class_dim), device=self.device)
            utils.batch_exec(batch_fn, rgbs, sigmas, bsize=bsize)(pts, dirs)
            return rgbs, sigmas
