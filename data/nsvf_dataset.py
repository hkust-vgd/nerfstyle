from pathlib import Path
import numpy as np
import imageio
from data.common import Intrinsics, load_matrix


class NSVFDataset:
    def __init__(self, dataroot, split='train'):
        self.root = Path(dataroot)
        rgb_dir = self.root / 'rgb'
        pose_dir = self.root / 'pose'
        intrinsics_path = self.root / 'intrinsics.txt'
        bbox_path = self.root / 'bbox.txt'

        split_prefix = {'train': 0, 'val': 1, 'test': 2}
        self.rgb_paths = sorted(rgb_dir.glob('{}_*.png'.format(split_prefix[split])))
        self.pose_paths = sorted(pose_dir.glob('{}_*.txt'.format(split_prefix[split])))
        assert len(self.rgb_paths) == len(self.pose_paths)
        assert all([fn1.stem == fn2.stem for fn1, fn2 in zip(self.rgb_paths, self.pose_paths)])

        def _parse_rgb(path):
            return np.array(imageio.imread(path), dtype=np.float32) / 255.0

        self.imgs = np.stack([_parse_rgb(path) for path in self.rgb_paths])
        self.poses = np.stack([load_matrix(path) for path in self.pose_paths], axis=0)

        H, W = self.imgs.shape[1:3]
        with open(intrinsics_path, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
        self.intrinsics = Intrinsics(H, W, f, f, cx, cy)

        self.bbox = load_matrix(bbox_path)[0, :-1].reshape(2, 3)

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]

    def __len__(self):
        return len(self.rgb_paths)

    def __str__(self):
        return "NSVF Dataset \"{}\" with {:d} entries".format(self.root.stem, len(self))
