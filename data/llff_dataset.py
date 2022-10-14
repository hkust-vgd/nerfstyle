from typing import Optional
import numpy as np
from common import BBox, Intrinsics
from data.base_dataset import BaseDataset
import utils


class LLFFDataset(BaseDataset):
    """LLFF (Local Light Field Fusion) dataset."""

    def __init__(
        self, *args,
        factor: int = 8,
        eval_every: int = 8,
        bd_factor: Optional[float] = 4/3
    ) -> None:
        """
        Initialize dataset.

        Args:
            *args: refer to `BaseDataset.__init__`.
            factor (int): Scale factor for LLFF images.
            bd_factor (Optional[float], optional): Scale pose origin coords, such that the \
                smallest near value is equal to this value.
        """

        super().__init__(*args)

        root = self.cfg.root_path

        if factor == 1:
            images_dir = root / 'images'
        else:
            images_dir = root / 'images_{:d}'.format(factor)
        assert images_dir.exists(), 'Images for chosen factor do not exist'

        self.rgb_paths = sorted(images_dir.glob('*.png'))
        poses_bds = np.load(root / 'poses_bounds.npy').astype(np.float32)
        assert len(self.rgb_paths) == len(poses_bds), 'No. of images ({:d}) ' \
            'and poses ({:d}) do not match'.format(len(self.rgb_paths), len(poses_bds))

        frame_ids = self._init_frame_ids(len(self.rgb_paths))
        if self.max_count is not None:
            self.rgb_paths = [self.rgb_paths[i] for i in frame_ids]
            poses_bds = poses_bds[frame_ids]

        split = utils.train_test_split(len(self.rgb_paths), eval_every, not self.is_train)
        self.rgb_paths = [self.rgb_paths[i] for i in split]
        poses_bds = poses_bds[split]
        self.frame_str_ids = [self.frame_str_ids[i] for i in split]

        self.imgs = np.stack([utils.parse_rgb(path) for path in self.rgb_paths])

        poses_hwf = poses_bds[:, :-2].reshape([-1, 3, 5])
        bds = poses_bds[:, -2:]

        # Setup factor
        assert np.all(poses_hwf[:, :, 4] == poses_hwf[0, :, 4])
        H, W, K = poses_hwf[0, :, 4] / factor
        assert self.imgs.shape[-2:] == (H, W)

        self.poses = utils.full_mtx(poses_hwf[:, :, :4])  # (N, 4, 4)
        self.intr = Intrinsics(H, W, K, K, H / 2, W / 2)

        trans_mtx = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.poses = np.einsum('nij,jk->nik', self.poses, trans_mtx)  # i=j=k=4

        # Scale poses origins
        sc = bd_factor / bds.min() if bd_factor is not None else 1.
        self.poses[:, :3, 3] *= sc

        # Poses are expressed relative to ref. pose instead of world origin.
        # Reference pose is averaged over all poses.
        ref_pose = utils.full_mtx(utils.poses_avg(self.poses))
        self.poses = np.einsum('ij,njk->nik', np.linalg.inv(ref_pose), self.poses)

        self.bbox = BBox.from_radius(self.cfg.bound)
