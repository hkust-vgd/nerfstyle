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

        if self.skip > 1:
            self.rgb_paths = self.rgb_paths[::self.skip]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([camera['Rt'] for camera in self.cameras])
        self._alpha2white()

        if self.cfg.replica_cfg.black2white:
            mask = np.all(self.imgs <= 0., axis=-1, keepdims=True)
            self.imgs = np.where(mask, 1., self.imgs)

        if self.skip > 1:
            self.poses = self.poses[::self.skip]

        for i in range(len(self)):
            R = np.copy(self.poses[i, :3, :3])
            t = np.copy(self.poses[i, :3, 3])
            self.poses[i, :3, 3] = -np.matmul(R.T, t)
            self.poses[i, :3, :3] = np.matmul(self.camera_t, R).T

        self.poses = np.einsum('ij, njk -> nik', self.pose_t, self.poses)
        self.poses = self.poses.astype(np.float32)

        _, H, W, _ = self.imgs.shape
        cx, cy = W // 2, H // 2
        f = self.cfg.replica_cfg.focal_ratio * max(H, W)
        self.intrinsics = Intrinsics(H, W, f, f, cx, cy)

        self.near = self.cfg.replica_cfg.near
        self.far = self.cfg.replica_cfg.far

    def __str__(self):
        desc = 'Replica dataset \"{}\" with {:d} entries'
        return desc.format(self.cfg.replica_cfg.name, len(self))


def load_bbox(
    dataset_cfg: DatasetConfig,
    scale_box: bool = True
) -> RotatedBBox:
    assert dataset_cfg.replica_cfg is not None
    bbox_path = dataset_cfg.root_path / 'bboxes' / '{}.txt'.format(dataset_cfg.replica_cfg.name)
    bbox_coords = utils.load_matrix(bbox_path)

    scale_factor = dataset_cfg.replica_cfg.scale_factor if scale_box else 1.0
    bbox = RotatedBBox(bbox_coords, scale_factor)
    return bbox
