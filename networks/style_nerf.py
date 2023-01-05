
from copy import deepcopy
from einops import rearrange
import torch
import torch.nn.functional as F
import tinycudann as tcnn

from common import TensorModule
# from networks.fx import VGG16FeatureExtractor
from networks.fx_vae import StyleVAEExtractor
# from networks.attn import MultiHeadAttention
from networks.tcnn_nerf import TCNerf, trunc_exp
# import utils


class StyleNerf(TensorModule):
    def __init__(
        self,
        model: TCNerf
    ) -> None:
        super(StyleNerf, self).__init__()
        self.cfg = model.cfg
        self.bounds_bbox = model.bounds_bbox

        self.d_embedder = model.x_embedder
        self.x_embedder = deepcopy(model.x_embedder)
        self.density_net = model.density_net

        rgb_net_input_dims = self.density_net.n_output_dims + self.x_embedder.n_output_dims - 1
        self.rgb_net = tcnn.Network(
            n_input_dims=rgb_net_input_dims,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': self.cfg.rgb_hidden_dims,
                'n_hidden_layers': self.cfg.rgb_hidden_layers
            },
            seed=self.cfg.network_seed
        )

        self.color_net = tcnn.Network(
            n_input_dims=67,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 64,
                'n_hidden_layers': 3
            },
            seed=self.cfg.network_seed
        )

        # self.style_fx = VGG16FeatureExtractor('relu5_1')
        self.style_fx = StyleVAEExtractor().cuda()

        # source_dim = 3  # dimension of encoding
        # cross_dim = 512  # dimension of style vector
        # self.style_attn = MultiHeadAttention(4, 32, 32, source_dim, cross_dim, actv='layer')
        self.style_net = tcnn.Network(
            n_input_dims=512,
            n_output_dims=32,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'None',
                'n_neurons': 64,
                'n_hidden_layers': 3
            },
            seed=self.cfg.network_seed
        )

    def forward(self, pts, style_images, tmp=False):
        pts = self.bounds_bbox.normalize(pts)
        d_embedded = self.d_embedder(pts)
        density_output = self.density_net(d_embedded)
        sigmas = trunc_exp(density_output[:, 0:1])

        x_embedded = self.x_embedder(pts)
        rgb_input = torch.cat((density_output[:, 1:], x_embedded), dim=-1)
        rgbs = self.rgb_net(rgb_input)

        if not tmp:
            rgbs = torch.sigmoid(rgbs)
            return rgbs, sigmas

        style_vectors = self.style_fx(style_images)
        style_vectors = self.style_net(style_vectors).repeat((len(x_embedded), 1))

        color_input = torch.cat((rgbs, x_embedded, style_vectors), dim=-1)
        rgbs = torch.sigmoid(self.color_net(color_input))
        return rgbs, sigmas
