import sys

import numpy as np
from numpy import ndarray
import torch
from torch import is_tensor, nn
from torchtyping import TensorType

import utils


class OccupancyGrid(nn.Module):
    def __init__(
        self,
        map: ndarray,
        global_min_pt: ndarray,
        global_max_pt: ndarray,
        res: ndarray
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

    # Allow '.to()', '.cuda()', etc. work on all stored tensor attributes
    def _apply(self, fn):
        for k, v in self.__dict__.items():
            if is_tensor(v):
                self.__setattr__(k, fn(v))
        return super()._apply(fn)

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
        indices = (pts - self.global_min_pt) / self.voxel_size
        indices = torch.sum(indices.to(torch.long) * self.basis, dim=-1)
        indices[invalid] = -1

        out = self.grid_flat[indices]
        return out


if __name__ == '__main__':
    grid = OccupancyGrid.load(sys.argv[1]).cuda()
