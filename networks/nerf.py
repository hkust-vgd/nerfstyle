from __future__ import annotations
import torch
from torch import nn
from typing import List, TypeVar

from common import TensorModule
from config import NetworkConfig
from networks.embedder import Embedder
import utils

T = TypeVar('T', bound='Nerf')


class Nerf(TensorModule):
    def __init__(
        self: T,
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
        self.logger = utils.create_logger(__name__)
        self.device = torch.device('cpu')

        self.x_embedder = self.get_embedder(x_enc_counts)
        self.d_embedder = self.get_embedder(d_enc_counts)
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

        self.actv = activations_dict[activation]

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    @staticmethod
    def get_linear(in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def to(
        self: T,
        device: torch.device
    ) -> T:
        self.device = device
        return super().to(device)

    def load_ckpt(self, ckpt):
        self.load_state_dict(ckpt['model'])

    def save_ckpt(self, ckpt):
        ckpt['model'] = self.state_dict()
        return ckpt

    def forward(self, pts, dirs, *args):
        # If additional arguments are provided, bind them to every linear layer
        def bind(layer):
            if not args:
                return layer
            return lambda x: layer(x, *args)

        x = self.x_embedder(pts)
        out = x
        for i, layer in enumerate(self.x_layers):
            if i in self.skip:
                out = torch.cat([x, out], dim=-1)
            out = self.actv(bind(layer)(out))

        a = bind(self.a_layer)(out)
        if dirs is None:
            return a

        d = self.d_embedder(dirs)
        out = torch.cat([bind(self.x2d_layer)(out), d], dim=-1)
        for layer in self.d_layers:
            out = self.actv(bind(layer)(out))

        c = torch.sigmoid(bind(self.c_layer)(out))
        return c, a


class SingleNerf(Nerf):
    def __init__(self, **nerf_params) -> None:
        super().__init__(**nerf_params)

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    @staticmethod
    def get_linear(in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    @classmethod
    def create_nerf(
        cls,
        net_cfg: NetworkConfig
    ) -> SingleNerf:
        nerf_config = {
            'x_enc_counts': net_cfg.x_enc_count,
            'd_enc_counts': net_cfg.d_enc_count,
            'x_layers': 8,
            'x_width': 256,
            'd_widths': [256, 128],
            'activation': net_cfg.activation,
            'skip': [5]
        }
        model = cls(**nerf_config)
        return model
