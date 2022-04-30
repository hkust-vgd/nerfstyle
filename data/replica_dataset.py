import json
import numpy as np

from data.base_dataset import BaseDataset
from common import Intrinsics
import utils


class ReplicaDataset(BaseDataset):
    def __init__(self, *args):
        super().__init__(*args)

        self.rgb_paths = sorted(self.root.glob('*_rgb.png'))
        self.camera_path = self.root / 'cameras.json'
        focal_ratio = 0.5

        self.camera_t = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        self.pose_t = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        with open(self.camera_path) as f:
            self.cameras = json.load(f)

        assert len(self.rgb_paths) == len(self.cameras)

        if self.skip > 1:
            self.rgb_paths = self.rgb_paths[::self.skip]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([camera['Rt'] for camera in self.cameras])
        self._alpha2white()

        if self.skip > 1:
            self.poses = self.poses[::self.skip]

        for i in range(len(self)):
            R = np.copy(self.poses[i, :3, :3])
            t = np.copy(self.poses[i, :3, 3])
            self.poses[i, :3, 3] = -np.matmul(R.T, t)
            self.poses[i, :3, :3] = np.matmul(self.camera_t, R).T

        self.poses = np.einsum('ij, njk -> nik', self.pose_t, self.poses)
        self.poses = self.poses.astype(np.float32)

        _, _, H, W = self.imgs.shape
        cx, cy = W // 2, H // 2
        f = focal_ratio * max(H, W)
        self.intrinsics = Intrinsics(H, W, f, f, cx, cy)

        # Use hard coded values for now
        self.near = 0.5
        self.far = 8.0
        self.bg_color = np.ones(3, dtype=np.float32)

    def __str__(self):
        desc = 'Replica dataset \"{}\" with {:d} entries'
        return desc.format(self.root.stem, len(self))
