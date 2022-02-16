import torch
from torch import nn
from typing import List, Optional

from config import NetworkConfig
from networks.embedder import Embedder


class Nerf(nn.Module):
    def __init__(
        self,
        x_enc_counts: int,
        d_enc_counts: int,
        x_layers: int,
        x_width: int,
        d_widths: List[int],
        activation: str = 'relu',
        skip: List[int] = ()
    ) -> None:
        """NeRF MLP network.

        Args:
            x_enc_counts (int): No. of terms for positional embedder.
            d_enc_counts (int): No. of terms for direction embedder.
            x_layers (int): No. of MLP layers before density output.
            x_width (int): MLP layer common width before density output.
            d_widths (list): MLP layer widths after density output.
            activation (str): Activation after each linear layer.
            skip (tuple, optional): Layers indices with input skip connection.
        """

        super(Nerf, self).__init__()
        self.skip = skip
        self.activation = activation

        self.x_embedder, self.d_embedder = None, None
        self._create_embedders(x_enc_counts, d_enc_counts)
        x_channels = self.x_embedder.out_channels
        d_channels = self.d_embedder.out_channels

        channels = [x_channels] + [x_width] * x_layers
        in_channels, out_channels = channels[:-1], channels[1:]
        for i in skip:
            in_channels[i] += x_channels
        self.x_layers = nn.ModuleList(
            [self.get_linear(i, j) for i, j in zip(in_channels, out_channels)])

        in_channels, out_channels = d_widths[:-1], d_widths[1:]
        in_channels[0] += d_channels
        self.d_layers = nn.ModuleList(
            [self.get_linear(i, j) for i, j in zip(in_channels, out_channels)])

        self.x2d_layer = self.get_linear(x_width, d_widths[0])
        self.a_layer = self.get_linear(x_width, 1)
        self.c_layer = self.get_linear(d_widths[-1], 3)

        activations_dict = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }

        self.activation = activations_dict[activation]

    def _create_embedders(self, x_enc_counts, d_enc_counts):
        self.x_embedder = Embedder(x_enc_counts)
        self.d_embedder = Embedder(d_enc_counts)

    @staticmethod
    def get_linear(in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def forward(self, pt, dir=None):
        x = self.x_embedder(pt)
        out = x
        for i, layer in enumerate(self.x_layers):
            if i in self.skip:
                out = torch.cat([x, out], dim=-1)
            out = self.activation(layer(out))

        a = self.a_layer(out)
        if dir is None:
            return a

        d = self.d_embedder(dir)
        out = torch.cat([self.x2d_layer(out), d], dim=-1)
        for layer in self.d_layers:
            out = self.activation(layer(out))

        c = torch.sigmoid(self.c_layer(out))
        return c, a


def create_single_nerf(
    net_cfg: NetworkConfig
) -> Nerf:
    nerf_config = {
        'x_enc_counts': net_cfg.x_enc_count,
        'd_enc_counts': net_cfg.d_enc_count,
        'x_layers': 8,
        'x_width': 256,
        'd_widths': [256, 128],
        'activation': net_cfg.activation,
        'skip': [5]
    }
    model = Nerf(**nerf_config)
    return model
