import numpy as np
from pathlib import Path
from typing import List

from common import Intrinsics
from config import DatasetConfig
from data.base_dataset import BaseDataset
import utils


class NSVFDataset(BaseDataset):
    """NSVF (Neural Sparse Voxel Fields) dataset."""

    def __init__(self, cfg: DatasetConfig, *args) -> None:
        self.root = cfg.root_path
        super().__init__(cfg, *args)

    def _get_split_idx(self) -> int:
        return self.split.value

    def _get_image_paths(self) -> List[Path]:
        rgb_dir = self.root / 'rgb'
        return sorted(rgb_dir.glob('{:d}_*.png'.format(self._get_split_idx())))

    def _get_poses(self) -> np.ndarray:
        pose_dir = self.root / 'pose'
        pose_paths = sorted(pose_dir.glob('{}_*.txt'.format(self._get_split_idx())))
        return np.stack([utils.load_matrix(path) for path in pose_paths], axis=0)

    def _get_intr(self) -> Intrinsics:
        intr_path = self.root / 'intrinsics.txt'
        _, _, H, W = self.images.shape

        try:
            # 4x4 matrix format
            intr_mtx = np.loadtxt(intr_path)
            intr = Intrinsics.from_np(intr_mtx, dims=(H, W))
        except ValueError:
            # parameters stored in top row
            with open(intr_path, 'r') as file:
                f, cx, cy, _ = map(float, file.readline().split())
            intr = Intrinsics(H, W, f, f, cx, cy)

        return intr
