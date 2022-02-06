import torch
from torch import nn
import torch.nn.functional as F


class Nerf(nn.Module):
    def __init__(self, x_channels, d_channels, x_layers,
                 x_width, d_widths, skip=()):
        """NeRF MLP network.

        Args:
            x_channels (int): No. of position channels, after encoding.
            d_channels (int): No. of direction channels, after encoding.
            x_layers (int): No. of MLP layers before density output.
            x_width (int): MLP layer common width before density output.
            d_widths (list): MLP layer widths after density output.
            skip (tuple, optional): Layers indices with input skip connection.
        """

        super(Nerf, self).__init__()
        self.skip = skip

        channels = [x_channels] + [x_width] * x_layers
        in_channels, out_channels = channels[:-1], channels[1:]
        for i in skip:
            in_channels[i] += x_channels
        self.x_layers = nn.ModuleList(
            [nn.Linear(i, j) for i, j in zip(in_channels, out_channels)])

        in_channels, out_channels = d_widths[:-1], d_widths[1:]
        in_channels[0] += d_channels
        self.d_layers = nn.ModuleList(
            [nn.Linear(i, j) for i, j in zip(in_channels, out_channels)])

        self.x2d_layer = nn.Linear(x_width, d_widths[0])
        self.a_layer = nn.Linear(x_width, 1)
        self.c_layer = nn.Linear(d_widths[-1], 3)

    def forward(self, x, d):
        out = x
        for i, layer in enumerate(self.x_layers):
            if i in self.skip:
                out = torch.cat([x, out], dim=-1)
            out = F.relu(layer(out))

        a = self.a_layer(out)
        out = torch.cat([self.x2d_layer(out), d], dim=-1)
        for layer in self.d_layers:
            out = F.relu(layer(out))

        c = torch.sigmoid(self.c_layer(out))
        return c, a
