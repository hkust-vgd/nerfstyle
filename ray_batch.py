from dataclasses import dataclass
from numpy import ndarray
import torch


@dataclass
class RayBatch:
    """A batch of N rays sharing a common origin point."""

    origin: ndarray
    """(3,) array. Origin of ray batch."""

    dests: ndarray
    """(N, 3) array. Direction vectors of rays relative to origin."""

    near: float
    """Closest distance of object box from any ray origin."""

    far: float
    """Furthest distance of object box from any ray origin."""

    def __len__(self):
        return len(self.dests)

    def viewdirs(self):
        norms = torch.norm(self.dests, dim=-1, keepdim=True)
        return self.dests / norms

    def lerp(self, coeffs):
        """Interpolate ray batch.

        Args:
            coeffs (ndarray[N, K]): Array of K coefficients for each ray.

        Returns:
            ndarray[N, K, 3]: [description]
        """
        assert len(coeffs) == len(self)
        out = torch.einsum('nc, nk -> nkc', self.dests, coeffs) + self.origin
        return out
