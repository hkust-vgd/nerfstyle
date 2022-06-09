from abc import ABC
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset

from common import Intrinsics
from config import DatasetConfig


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        cfg: DatasetConfig,
        split: str,
        max_count: Optional[int] = None
    ):
        super().__init__()

        self.cfg = cfg
        self.split = split
        self.max_count = max_count

        root = self.cfg.root_path
        assert root.exists(), 'Root path "{}" does not exist'.format(root)

        # Common dataset interface
        self.imgs: np.ndarray
        self.poses: np.ndarray
        self.frame_ids: List[int]
        self.frame_str_ids: List[str]
        self.intrinsics: Intrinsics
        self.near, self.far = 0., 0.

    def _alpha2white(self):
        assert self.imgs.shape[1] == 4
        rgb, alpha = self.imgs[:, :3], self.imgs[:, 3:]
        self.imgs = rgb * alpha + (1 - alpha)

    def _set_frame_ids(self, frame_count: int):
        if self.max_count is not None:
            assert self.max_count > 0, 'Invalid value for "max_count"'
            frame_ids = np.linspace(0, frame_count, self.max_count + 1)[:-1]
            frame_ids = np.round(frame_ids).astype(int)
        else:
            frame_ids = np.arange(frame_count)

        self.frame_ids = list(frame_ids)
        width = len(str(frame_count))
        self.frame_str_ids = ['{:0{width}d}'.format(i, width=width) for i in frame_ids]

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]

    def __len__(self):
        return len(self.rgb_paths)
