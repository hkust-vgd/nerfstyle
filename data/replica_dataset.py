import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ReplicaDataset(Dataset):
    def __init__(
        self,
        dataroot: Path
    ):
        self.root = dataroot

        self.rgb_paths = sorted(self.root.glob('*_rgb.png'))
        self.camera_path = self.root / 'cameras.json'
        self.focal_length = 800

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
        assert np.all([self.cameras[0]['K'] == camera['K'] for camera in self.cameras])

        intrinsics_matrix = np.array(self.cameras[0]['K'])
        print(intrinsics_matrix)

        self.poses = [camera['Rt'] for camera in self.cameras]
        for i in range(len(self)):
            R = np.copy(self.poses[i, :3, :3])
            t = np.copy(self.poses[i, :3, 3])
            self.poses[i, :3, 3] = -np.matmul(R.T, t)
            self.poses[i, :3, :3] = np.matmul(self.camera_t, R).T

        self.poses = np.einsum('ij, njk -> nik', self.pose_t, self.poses)[:, :3]

        # BBoxes
        # Near, far

    def __len__(self):
        return len(self.rgb_paths)


if __name__ == '__main__':
    root_path = Path('/home/hwpang/datasets/replica_all/train/00/')
    dataset = ReplicaDataset(root_path)
