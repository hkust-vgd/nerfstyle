import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from common import DatasetSplit, Intrinsics
from config import DatasetConfig
from data.base_dataset import BaseDataset


# Data is first preprocessed by torch-ngp
class LLFFDataset(BaseDataset):
    """LLFF (Local Light Field Fusion) dataset."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: DatasetSplit,
        max_count: Optional[int] = None
    ) -> None:
        self.root = cfg.root_path
        split_path = self.root / 'transforms_{}.json'.format(split.name.lower())
        with open(split_path, 'r') as f:
            self.split_json = json.load(f)
        super().__init__(cfg, split, max_count)

    def _get_image_paths(self) -> List[Path]:
        if self.split == DatasetSplit.TEST:
            return None
        return [self.root / f['file_path'] for f in self.split_json['frames']]

    def _get_seg_groups(self) -> np.ndarray:
        # segs_np = np.load(self.root / 'seg_bkp' / 'train_seg_groups.npy')
        seg_paths = [self.root / 'seg' / '{}_seg.npz'.format(fn) for fn in self.fns]
        segs_np = np.stack([np.load(p)['seg_map'] for p in seg_paths])
        return segs_np.astype(np.float32)

    def _get_poses(self) -> np.ndarray:
        poses = [f['transform_matrix'] for f in self.split_json['frames']]
        poses = np.array(poses, dtype=np.float32)
        return poses

    def _get_intr(self) -> Intrinsics:
        intr = Intrinsics(
            h=int(self.split_json['h']),
            w=int(self.split_json['w']),
            fx=self.split_json['fl_x'],
            fy=self.split_json['fl_y'],
            cx=self.split_json['cx'],
            cy=self.split_json['cy']
        )
        return intr
