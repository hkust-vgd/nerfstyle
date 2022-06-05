from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar, Union

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import utils

T = TypeVar('T')
patch_typeguard()


class TrainMode(Enum):
    PRETRAIN = 0
    DISTILL = 1
    FINETUNE = 2


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

    def scale(self, w: int, h: int) -> Intrinsics:
        """
        Rescales intrinsic matrix to new dimensions. If aspect ratio is different, focal
        length is rescaled to shorter edge.

        Args:
            w (int): New width.
            h (int): New height.

        Returns:
            Intrinsics: New intrinsic matrix object.
        """
        cx = w / 2.
        cy = h / 2.

        old_ar = self.w / self.h
        new_ar = w / h
        ratio = h / self.h if new_ar >= old_ar else w / self.w
        fx = self.fx * ratio
        fy = self.fy * ratio

        intr = Intrinsics(h, w, fx, fy, cx, cy)
        return intr


@dataclass
class LossValue:
    print_name: str
    """Identifier when logging on console."""

    log_name: str
    """Identifier when logging on TensorBoard."""

    value: torch.Tensor
    """Loss tensor."""


@dataclass
class RayBatch:
    """A batch of N rays sharing a common origin point."""

    origin: np.ndarray
    """(3,) array. Origin of ray batch."""

    dests: np.ndarray
    """(N, 3) array. Direction vectors of rays relative to origin."""

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
            coeffs (np.ndarray[N, K]): Array of K coefficients for each ray.

        Returns:
            np.ndarray[N, K, 3]: Array of points at interpolated positions for each ray.
        """
        assert len(coeffs) == len(self)
        out = torch.einsum('nc, nk -> nkc', self.dests, coeffs) + self.origin
        return out


class TensorModule(torch.nn.Module):
    def __init__(self):
        """
        A PyTorch module where all stored tensor attributes will work like buffers, i.e. one can
        call '.to()' or '.cuda()' to move them to a GPU device.
        """
        super().__init__()
        self.device = torch.device('cpu')

    def _apply(self, fn):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                self.__setattr__(k, fn(v))

        super()._apply(fn)
        return self

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        if device is None:
            self.device = torch.device('cuda')
        elif isinstance(device, int):
            self.device = torch.device('cuda', device)
        else:
            self.device = device

        return super().cuda(device)

    def cpu(self: T) -> T:
        self.device = torch.device('cpu')
        return super().cpu()

    def to(self: T, *args, **kwargs) -> T:
        self.device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BBox(TensorModule):
    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> None:
        super().__init__()
        self._min_pt = bbox_min
        self._max_pt = bbox_max

    @property
    def min_pt(self) -> torch.Tensor:
        return torch.tensor(self._min_pt, device=self.device)

    @property
    def max_pt(self) -> torch.Tensor:
        return torch.tensor(self._max_pt, device=self.device)

    def size(self) -> torch.Tensor:
        return self.max_pt - self.min_pt

    def scale(self, factor: float):
        mid_pt = (self._min_pt + self._max_pt) / 2
        self._min_pt = (self._min_pt - mid_pt) * factor + mid_pt
        self._max_pt = (self._max_pt - mid_pt) * factor + mid_pt

    def forward(self):
        raise NotImplementedError


class RotatedBBox(BBox):
    def __init__(
        self,
        pts: np.ndarray,
        scale_factor: float = 1.0
    ) -> None:
        """
        A 3D dimensional bounding box.

        Args:
            pts (np.ndarray): The 8 coordinates of the bounding box.
        """
        assert pts.shape == (8, 3)

        # Indeixing convention:
        # Top face clockwise: v0 - v3, Bottom face clockwise: v4 - v7
        # v3 is on top of v4
        self.pts = pts
        super().__init__(np.min(self.pts, axis=0), np.max(self.pts, axis=0))

        if scale_factor > 1.0:
            midpt = (self._min_pt + self._max_pt) / 2
            self.pts = (self.pts - midpt) * scale_factor + midpt
            self._min_pt = np.min(self.pts, axis=0)
            self._max_pt = np.max(self.pts, axis=0)

        # Identify 6 triangular reference faces on each side of bbox
        faces = np.array([
            [0, 1, 2], [4, 3, 2], [5, 2, 1],
            [6, 1, 0], [7, 0, 3], [4, 5, 6]
        ])
        pts0, pts1, pts2 = self.pts[faces.T]
        vecs1, vecs2 = pts1 - pts0, pts2 - pts0
        normals = np.stack([np.cross(u, v) for u, v in zip(vecs1, vecs2)], axis=0)

        self.origins = torch.tensor(pts0)
        self.normals = torch.tensor(normals)

    @typechecked
    def forward(
        self,
        pts: TensorType['batch_size', 3],
        outside: bool = False
    ) -> TensorType['batch_size']:
        # Point is inside bbox if all 6 reference faces are facing it
        vecs = pts.unsqueeze(1) - self.origins  # (N, 6, 3)
        dot_prods = torch.einsum('nfc, fc -> nf', vecs, self.normals)

        if outside:
            return torch.any(dot_prods <= 0, dim=-1)
        return torch.all(dot_prods > 0, dim=-1)


class OccupancyGrid(TensorModule):
    def __init__(
        self,
        map: np.ndarray,
        global_min_pt: np.ndarray,
        global_max_pt: np.ndarray,
        res: np.ndarray
    ) -> None:
        assert global_min_pt.shape == global_max_pt.shape == res.shape == (3, )
        assert np.all(map.shape == res)
        super().__init__()
        self.grid = map

        self.grid_flat = torch.BoolTensor(np.append(map.reshape(-1), 0))
        self.global_min_pt = torch.FloatTensor(global_min_pt)
        self.global_max_pt = torch.FloatTensor(global_max_pt)
        self.res = torch.FloatTensor(res)
        self.voxel_size = (self.global_max_pt - self.global_min_pt) / self.res
        self.basis = torch.LongTensor([res[2] * res[1], res[2], 1])

    @classmethod
    def load(cls, path, logger=None):
        @utils.loader(logger)
        def _load(grid_path):
            grid_np = np.load(grid_path)
            obj = cls(grid_np['map'], grid_np['global_min_pt'],
                      grid_np['global_max_pt'], grid_np['res'])
            return obj

        grid_obj = _load(path)
        return grid_obj

    def pts_to_indices(
        self,
        pts: TensorType['batch_size', 3]
    ) -> TensorType['batch_size', 3]:
        indices = (pts - self.global_min_pt) / self.voxel_size
        indices_long = torch.floor(indices).to(torch.long)
        return indices_long

    def forward(
        self,
        pts: TensorType['batch_size', 3]
    ) -> TensorType['batch_size']:
        epsilon = 1e-5
        invalid = [
            (pts >= self.global_max_pt - epsilon),
            (pts < self.global_min_pt + epsilon)
        ]
        invalid = torch.any(torch.cat(invalid, dim=-1), dim=-1)  # (N, )
        indices = torch.sum(self.pts_to_indices(pts) * self.basis, dim=-1)
        indices[invalid] = -1

        out = self.grid_flat[indices]
        return out
