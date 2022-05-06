from abc import ABC
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from common import Intrinsics
from config import DatasetConfig


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        cfg: DatasetConfig,
        split: str,
        skip: int = 1
    ):
        super().__init__()

        self.cfg = cfg
        self.split = split
        self.skip = skip

        root = self.cfg.root_path
        assert root.exists(), 'Root path "{}" does not exist'.format(root)

        # Common dataset interface
        self.imgs: np.ndarray
        self.poses: np.ndarray
        self.intrinsics: Intrinsics
        self.near, self.far = 0., 0.

    def _alpha2white(self):
        assert self.imgs.shape[-1] == 4
        rgb, alpha = self.imgs[..., :3], self.imgs[..., 3:]
        self.imgs = rgb * alpha + (1 - alpha)

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]

    def __len__(self):
        return len(self.rgb_paths)
