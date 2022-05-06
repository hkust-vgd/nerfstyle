from pathlib import Path
from typing import Union
import numpy as np

from data.base_dataset import BaseDataset
from common import Intrinsics
import utils


class NSVFDataset(BaseDataset):
    def __init__(
        self,
        dataroot: Union[str, Path],
        split: str,
        skip: int = 1
    ):
        super().__init__(dataroot, split, skip)

        root = self.cfg.root_path
        rgb_dir = root / 'rgb'
        pose_dir = root / 'pose'
        intrinsics_path = root / 'intrinsics.txt'
        bbox_path = root / 'bbox.txt'
        nf_path = root / 'near_and_far.txt'

        split_prefix = {'train': 0, 'val': 1, 'test': 2}
        self.rgb_paths = sorted(rgb_dir.glob('{}_*.png'.format(split_prefix[split])))
        self.pose_paths = sorted(pose_dir.glob('{}_*.txt'.format(split_prefix[split])))
        assert len(self.rgb_paths) == len(self.pose_paths)
        assert all([fn1.stem == fn2.stem for fn1, fn2 in zip(self.rgb_paths, self.pose_paths)])

        if skip > 1:
            self.rgb_paths = self.rgb_paths[::skip]
            self.pose_paths = self.pose_paths[::skip]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([utils.load_matrix(path) for path in self.pose_paths], axis=0)
        self._alpha2white()

        H, W = self.imgs.shape[1:3]
        with open(intrinsics_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
        self.intrinsics = Intrinsics(H, W, f, f, cx, cy)

        self.bbox_min, self.bbox_max = load_bbox(bbox_path)

        # bbox_center = (bbox_min + bbox_max) / 2
        # pts = self.poses[:, :3, -1]
        # closest_pts = np.clip(pts, bbox_min, bbox_max)
        # furthest_pts = np.where(pts < bbox_center, bbox_max, bbox_min)
        # self.near = np.amin(np.linalg.norm(pts - closest_pts, axis=1))
        # self.far = np.amax(np.linalg.norm(pts - furthest_pts, axis=1))

        self.near, self.far = utils.load_matrix(nf_path)[0]

        self.bg_color = np.ones(3, dtype=np.float32)

    def __str__(self):
        name = self.cfg.root_path.stem
        desc = 'NSVF dataset \"{}\" with {:d} entries'
        return desc.format(name, len(self))


def load_bbox(bbox_path):
    return utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)
