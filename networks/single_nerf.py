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
        # TODO: Perform ERT for single nerf as well
        return super().forward(pts, dirs)
