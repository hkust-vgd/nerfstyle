import torch
from torch import nn

from config import NetworkConfig
from networks.embedder import Embedder
from .nerf import Nerf


class SingleNerf(Nerf):
    def __init__(
        self,
        net_cfg: NetworkConfig
    ) -> None:
        """
        A single NeRF network.

        Args:
            net_cfg (NetworkConfig): Network configuration.
        """
        super().__init__(net_cfg)

    @staticmethod
    def get_embedder(enc_counts):
        return Embedder(enc_counts)

    @staticmethod
    def get_linear(in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def forward(self, pts, dirs=None, ert_mask=None):
        if ert_mask is None or torch.sum(ert_mask) == len(pts):
            return super().forward(pts, dirs)

        rgbs = torch.zeros((len(pts), 3), device=self.device)
        densities = torch.zeros((len(pts), 1), device=self.device)

        if torch.sum(ert_mask) > 0:
            rgbs[ert_mask], densities[ert_mask] = super().forward(
                pts[ert_mask], dirs[ert_mask])
        return rgbs, densities
