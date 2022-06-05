from pathlib import Path
from typing import Union
import numpy as np

from common import Intrinsics, BBox
from config import DatasetConfig
from data.base_dataset import BaseDataset
import utils


class NSVFDataset(BaseDataset):
    def __init__(self, *args):
        super().__init__(*args)

        root = self.cfg.root_path
        rgb_dir = root / 'rgb'
        pose_dir = root / 'pose'
        intrinsics_path = root / 'intrinsics.txt'
        nf_path = root / 'near_and_far.txt'

        split_prefix = {'train': 0, 'val': 1, 'test': 2}
        self.rgb_paths = sorted(rgb_dir.glob('{}_*.png'.format(split_prefix[self.split])))
        self.pose_paths = sorted(pose_dir.glob('{}_*.txt'.format(split_prefix[self.split])))
        assert len(self.rgb_paths) == len(self.pose_paths)
        assert all([fn1.stem == fn2.stem for fn1, fn2 in zip(self.rgb_paths, self.pose_paths)])

        if self.skip > 1:
            self.rgb_paths = self.rgb_paths[::self.skip]
            self.pose_paths = self.pose_paths[::self.skip]
        if self.max_count >= 0:
            self.rgb_paths = self.rgb_paths[:self.max_count]
            self.pose_paths = self.pose_paths[:self.max_count]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([utils.load_matrix(path) for path in self.pose_paths], axis=0)
        self._alpha2white()

        _, _, H, W = self.imgs.shape
        with open(intrinsics_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
        self.intrinsics = Intrinsics(H, W, f, f, cx, cy)

        # bbox_center = (bbox_min + bbox_max) / 2
        # pts = self.poses[:, :3, -1]
        # closest_pts = np.clip(pts, bbox_min, bbox_max)
        # furthest_pts = np.where(pts < bbox_center, bbox_max, bbox_min)
        # self.near = np.amin(np.linalg.norm(pts - closest_pts, axis=1))
        # self.far = np.amax(np.linalg.norm(pts - furthest_pts, axis=1))

        self.near, self.far = utils.load_matrix(nf_path)[0]

    def __str__(self):
        name = self.cfg.root_path.stem
        desc = 'NSVF dataset \"{}\" with {:d} entries'
        return desc.format(name, len(self))


def load_bbox(
    dataset_cfg: DatasetConfig,
    _
) -> BBox:
    bbox_path = dataset_cfg.root_path / 'bbox.txt'
    bbox_min, bbox_max = utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)
    bbox = BBox(bbox_min, bbox_max)
    return bbox
