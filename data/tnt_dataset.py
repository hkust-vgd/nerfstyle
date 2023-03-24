from typing import List, Optional
import numpy as np

from common import DatasetSplit
from config import DatasetConfig
from data.nsvf_dataset import NSVFDataset


class TnTDataset(NSVFDataset):
    """Tanks and Temples dataset."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: DatasetSplit,
        max_count: Optional[int] = None
    ) -> None:
        super().__init__(cfg, split, max_count)

    def _get_split_idx(self) -> int:
        # use same splits for validation and test
        return max(self.split.value, 1)

    def _get_poses(self) -> np.ndarray:
        poses = super()._get_poses()
        raise NotImplementedError(poses.shape)

    def _get_seg_groups(self):
        return None
