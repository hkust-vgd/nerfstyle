import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from common import DatasetSplit, Intrinsics
from config import DatasetConfig
from data.base_dataset import BaseDataset
from utils.matrix import convert_poses


class ReplicaDataset(BaseDataset):
    """Replica dataset."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: DatasetSplit,
        max_count: Optional[int] = None
    ) -> None:
        self.root = cfg.root_path
        self.split_traj_ids = cfg.replica_cfg.traj_ids[:3]
        super().__init__(cfg, split, max_count)

    def _get_image_paths(self) -> List[Path]:
        image_paths = []

        for traj in self.split_traj_ids:
            subdir = self.root / 'train' / '{:02d}'.format(traj)
            image_paths += sorted(subdir.glob('*.png'))

        return image_paths

    def _get_seg_groups(self) -> np.ndarray:
        groups_np = np.load(self.root / 'seg_ids' / '{}.npy'.format(self.cfg.replica_cfg.name))
        return groups_np.astype(np.float32)

    def _get_poses(self) -> np.ndarray:
        poses = []

        for traj in self.split_traj_ids:
            subdir = self.root / 'train' / '{:02d}'.format(traj)
            with open(subdir / 'cameras.json', 'r') as f:
                traj_cameras = json.load(f)
            poses += [np.linalg.inv(c['Rt']) for c in traj_cameras]

        poses = np.stack(poses).astype(np.float32)
        poses = convert_poses(poses, w_coord='rdf', c_coord='rub')

        # Center poses
        poses_t = poses[:, :3, 3]
        poses_center = (np.amax(poses_t, axis=0) + np.amin(poses_t, axis=0)) / 2.0
        poses_t -= poses_center

        return poses

    def _get_intr(self) -> Intrinsics:
        _, _, H, W = self.images.shape
        cx, cy = W // 2, H // 2
        f = self.cfg.replica_cfg.focal_ratio * max(H, W)
        intr = Intrinsics(H, W, f, f, cx, cy)
        return intr
