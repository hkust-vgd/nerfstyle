from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import nn

from common import TensorModule
from config import NetworkConfig
import utils


class Nerf(TensorModule, ABC):
    def __init__(
        self,
        net_cfg: NetworkConfig
    ) -> None:
        """
        Abstract base class for all NeRF networks.

        Args:
            net_cfg (NetworkConfig): Network configuration.
        """
        super(Nerf, self).__init__()
        self.skip = net_cfg.x_skips
        self.activation = net_cfg.activation
        self.logger = utils.create_logger(__name__)
        self.device = torch.device('cpu')

        x_widths, d_widths = net_cfg.x_widths, net_cfg.d_widths
        if isinstance(x_widths, int):
            x_widths = [x_widths] * net_cfg.x_layers
        if isinstance(d_widths, int):
            d_widths = [d_widths] * net_cfg.d_layers
        assert len(x_widths) == net_cfg.x_layers
        assert len(d_widths) == net_cfg.d_layers

        self.x_embedder = self.get_embedder(net_cfg.x_enc_count)
        self.d_embedder = self.get_embedder(net_cfg.d_enc_count)
        x_channels = self.x_embedder.out_channels
        d_channels = self.d_embedder.out_channels

        channels = [x_channels] + x_widths
        in_channels, out_channels = channels[:-1], channels[1:]
        for i in self.skip:
            in_channels[i] += x_channels
        self.x_layers = nn.ModuleList(
            [self.get_linear(i, j) for i, j in zip(in_channels, out_channels)])

        in_channels, out_channels = d_widths[:-1], d_widths[1:]
        in_channels[0] += d_channels
        self.d_layers = nn.ModuleList(
            [self.get_linear(i, j) for i, j in zip(in_channels, out_channels)])

        self.x2d_layer = self.get_linear(x_widths[-1], d_widths[0])
        self.a_layer = self.get_linear(x_widths[-1], 1)
        self.c_layer = self.get_linear(d_widths[-1], 3)

        activations_dict = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }

        self.actv = activations_dict[net_cfg.activation]

    @staticmethod
    @abstractmethod
    def get_embedder(enc_counts):
        pass

    @staticmethod
    @abstractmethod
    def get_linear(in_channels, out_channels):
        pass

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
