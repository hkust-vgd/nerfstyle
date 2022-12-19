
from copy import deepcopy
import torch
import tinycudann as tcnn

from common import TensorModule
from networks.tcnn_nerf import TCNerf, trunc_exp


class StyleNerf(TensorModule):
    def __init__(
        self,
        model: TCNerf
    ) -> None:
        super(StyleNerf, self).__init__()
        self.cfg = model.cfg
        self.bounds_bbox = model.bounds_bbox

        self.x_embedder = model.x_embedder
        self.s_embedder = deepcopy(model.x_embedder)
        self.density_net = model.density_net

        rgb_net_input_dims = self.density_net.n_output_dims + self.s_embedder.n_output_dims - 1
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

    # def encode(self, pts):
    #     pts = self.bounds_bbox.normalize(pts)
    #     embedded = self.s_embedder(pts)
    #     return embedded.float()

    # def density(self, pts):
    #     pts = self.bounds_bbox.normalize(pts)
    #     embedded = self.x_embedder(pts)
    #     density_output = self.density_net(embedded)
    #     sigmas = trunc_exp(density_output[:, 0:1])
    #     return sigmas.float()

    # def color(self, feats):
    #     out = self.final_fc(feats)
    #     return torch.sigmoid(out)

    def forward(self, pts, dirs):
        pts = self.bounds_bbox.normalize(pts)
        x_embedded = self.x_embedder(pts)
        density_output = self.density_net(x_embedded)
        sigmas = trunc_exp(density_output[:, 0:1])

        s_embedded = self.s_embedder(pts)
        rgb_input = torch.cat((density_output[:, 1:], s_embedded), dim=-1)
        rgbs = self.rgb_net(rgb_input)
        return rgbs, sigmas
