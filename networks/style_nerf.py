
from copy import deepcopy
from einops import rearrange
import torch
import torch.nn.functional as F
import tinycudann as tcnn

from common import TensorModule
from networks.fx import VGG16FeatureExtractor
from networks.attn import MultiHeadAttention
from networks.tcnn_nerf import TCNerf, trunc_exp
import utils


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

        self.style_fx = VGG16FeatureExtractor('relu5_1')

        source_dim = 3  # dimension of encoding
        cross_dim = 512  # dimension of style vector
        self.style_attn = MultiHeadAttention(4, 32, 32, source_dim, cross_dim, actv='layer')

    def forward(self, pts, style_images):
        pts = self.bounds_bbox.normalize(pts)
        d_embedded = self.d_embedder(pts)
        density_output = self.density_net(d_embedded)
        sigmas = trunc_exp(density_output[:, 0:1])

        x_embedded = self.x_embedder(pts)
        style_images = F.interpolate(style_images, size=(256, 256))
        style_vectors = self.style_fx(style_images)['relu5_1']
        style_vectors = rearrange(style_vectors, 'b c h w -> b (h w) c')

        rgb_input = torch.cat((density_output[:, 1:], x_embedded), dim=-1)
        rgbs = self.rgb_net(rgb_input)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            def attn(feat_vectors_batch):
                feat_vectors_batch = feat_vectors_batch.unsqueeze(1)
                attn_feats_batch = self.style_attn(feat_vectors_batch, style_vectors, style_vectors)
                return attn_feats_batch.squeeze(1)

            attn_rgbs = torch.empty_like(rgbs, device=self.device)
            utils.batch_exec(attn, attn_rgbs, bsize=16384)(rgbs)
            attn_rgbs = torch.sigmoid(attn_rgbs)

        return attn_rgbs, sigmas
