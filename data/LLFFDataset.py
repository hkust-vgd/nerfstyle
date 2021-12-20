import os
import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)


def getMatrix(z, up, pos):
    """
    Get viewing matrix
    :param z: Unnormalized front (z) vector
    :param up: Reference upwards vector
    :param pos: Camera position
    :return: 3x4 transformtation matrix
    """
    z = normalize(z)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
    M = np.stack([x, y, z, pos], axis=1)
    return M


class LLFFDataset:
    POSES_FN = 'poses_bounds.npy'
    EXTS = ['.png', '.jpg', '.jpeg']

    def __init__(self, base_dir, factor=8):
        poses_arr = np.load(os.path.join(base_dir, self.POSES_FN))  # (N, 17)
        poses, bounds = np.split(poses_arr, [15, ], axis=1)
        poses = poses.reshape(-1, 3, 5)
        h, w, f = poses[0, :, 4]
        self.poses = poses[:, :, :4]
        self.bounds = bounds
        print(h, w, f)

        images_dir = 'images' if factor is None else 'images_{:d}'.format(factor)
        images_dir = os.path.join(base_dir, images_dir)
        assert os.path.exists(images_dir)

        all_files = [os.path.join(images_dir, fn) for fn in sorted(os.listdir(images_dir))]
        self.paths = [fn for fn in all_files if os.path.splitext(fn)[-1] in self.EXTS]

        self._recenter()

    def _recenter(self):
        y_mean = self.poses[:, :, 1].mean(0)
        z_mean = self.poses[:, :, 2].mean(0)
        center = self.poses[:, :, 3].mean(0)
        c2w = getMatrix(z_mean, y_mean, center)
        c2w = np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0)
        


if __name__ == '__main__':
    path = '/home/hwpang/datasets/nerf_llff_data/fern'
    tmp = LLFFDataset(path)
    print(tmp.poses.shape)
    print(tmp.bounds.shape)
