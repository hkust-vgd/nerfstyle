from dataclasses import dataclass, field
from numpy import ndarray
import torch


@dataclass(frozen=True)
class Intrinsics:
    """Camera specifications."""

    h: int
    """Height of each frame."""

    w: int
    """Width of each frame."""

    fx: float
    """Focal length in X-axis."""

    fy: float
    """Focal length in Y-axis."""

    cx: float
    """Camera center offset in X-axis."""

    cy: float
    """Camera center offset in Y-axis."""


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

    def __post_init__(self):
        # Normalize the rays to unit vectors
        assert len(self.dests.shape) == 2
        self.dests = self.dests / torch.norm(self.dests, dim=-1, keepdim=True)

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
