import json
import numpy as np

from common import Intrinsics, RotatedBBox
from config import DatasetConfig
from data.base_dataset import BaseDataset
import utils


class ReplicaDataset(BaseDataset):
    def __init__(self, *args):
        super().__init__(*args)
        assert self.cfg.replica_cfg is not None

        self.rgb_paths = []
        self.cameras = []

        for traj in self.cfg.replica_cfg.traj_ids:
            subdir = self.cfg.root_path / 'train' / '{:02d}'.format(traj)
            self.rgb_paths += sorted(subdir.glob('*_rgb.png'))

            with open(subdir / 'cameras.json', 'r') as f:
                traj_cameras = json.load(f)
            self.cameras += traj_cameras

        self.camera_t = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        self.pose_t = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        assert len(self.rgb_paths) == len(self.cameras)
        frame_ids = self._init_frame_ids(len(self.rgb_paths))
        if self.max_count is not None:
            self.rgb_paths = [self.rgb_paths[i] for i in frame_ids]
            self.cameras = [self.cameras[i] for i in frame_ids]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([camera['Rt'] for camera in self.cameras])
        self._alpha2white()

        if self.cfg.replica_cfg.black2white:
            mask = np.all(self.imgs <= 0., axis=-1, keepdims=True)
            self.imgs = np.where(mask, 1., self.imgs)

        for i in range(len(self)):
            R = np.copy(self.poses[i, :3, :3])
            t = np.copy(self.poses[i, :3, 3])
            self.poses[i, :3, 3] = -np.matmul(R.T, t)
            self.poses[i, :3, :3] = np.matmul(self.camera_t, R).T

        self.poses = np.einsum('ij, njk -> nik', self.pose_t, self.poses)
        self.poses = self.poses.astype(np.float32)

        _, _, H, W = self.imgs.shape
        cx, cy = W // 2, H // 2
        f = self.cfg.replica_cfg.focal_ratio * max(H, W)
        self.intr = Intrinsics(H, W, f, f, cx, cy)

    def __str__(self):
        return super().__str__(name=self.cfg.replica_cfg.name)


def load_bbox(
    dataset_cfg: DatasetConfig
) -> RotatedBBox:
    assert dataset_cfg.replica_cfg is not None
    bbox_path = dataset_cfg.root_path / 'bboxes' / '{}.txt'.format(dataset_cfg.replica_cfg.name)
    bbox_coords = utils.load_matrix(bbox_path)

    bbox = RotatedBBox(bbox_coords)
    return bbox
