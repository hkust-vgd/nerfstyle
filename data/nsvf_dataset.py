import numpy as np
from common import BBox, Intrinsics
from data.base_dataset import BaseDataset
import utils


class NSVFDataset(BaseDataset):
    """NSVF (Neural Sparse Voxel Fields) dataset."""

    def __init__(self, *args):
        super().__init__(*args)

        root = self.cfg.root_path
        rgb_dir = root / 'rgb'
        pose_dir = root / 'pose'
        intrinsics_path = root / 'intrinsics.txt'

        split_prefix = 0 if self.is_train else 2
        self.rgb_paths = sorted(rgb_dir.glob('{}_*.png'.format(split_prefix)))
        self.pose_paths = sorted(pose_dir.glob('{}_*.txt'.format(split_prefix)))
        assert len(self.rgb_paths) == len(self.pose_paths)
        assert all([fn1.stem == fn2.stem for fn1, fn2 in zip(self.rgb_paths, self.pose_paths)])

        frame_ids = self._init_frame_ids(len(self.rgb_paths))
        if self.max_count is not None:
            self.rgb_paths = [self.rgb_paths[i] for i in frame_ids]
            self.pose_paths = [self.pose_paths[i] for i in frame_ids]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([utils.load_matrix(path) for path in self.pose_paths], axis=0)
        self._alpha2white()
        self.bbox = BBox.from_radius(self.cfg.bound)

        _, _, H, W = self.imgs.shape
        with open(intrinsics_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
        self.intr = Intrinsics(H, W, f, f, cx, cy)
