from abc import ABC
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset

from common import BBox, DatasetSplit, DatasetCoordSystem, Intrinsics
from config import DatasetConfig


class BaseDataset(Dataset, ABC):
    """NeRF dataset base class."""

    # Common interface of datasets:

    imgs: np.ndarray
    """Array of images. Size should be [N, 3, H, W]."""

    poses: np.ndarray
    """Array of poses. Size should be [N, 4, 4]."""

    bbox: BBox
    """Bounding box that bounds the entire scene volume (but not \
        necessarily the camera origins)."""

    frame_str_ids: List[str]
    """String version of `frame_ids`. Used in output filenames."""

    intr: Intrinsics
    """Default camera intrinsic parameters."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: DatasetSplit,
        max_count: Optional[int] = None
    ):
        """
        Initialize dataset.

        Args:
            cfg (DatasetConfig): Config file.
            split (DatasetSplit): Selects train / test split to be loaded.
            max_count (Optional[int], optional): Select a subset of size N, picked uniformly \
                over the list of images. If None, uses all images. Defaults to None.
        """
        super().__init__()

        self.cfg = cfg
        self.split = split
        self.max_count = max_count

        assert self.cfg.root_path.exists(), 'Root path "{}" does not exist'.format(
            self.cfg.root_path)

    def _alpha2white(self):
        assert self.imgs.shape[1] == 4
        rgb, alpha = self.imgs[:, :3], self.imgs[:, 3:]
        self.imgs = rgb * alpha + (1 - alpha)

    def _preprocess_poses(self):
        assert len(self.poses.shape) == 3
        assert self.poses.shape[1] == self.poses.shape[2] == 4

        # Correct coordinate system
        if self.cfg.coord_type == DatasetCoordSystem.RDF:
            self.poses = self.poses[:, [0, 2, 1, 3]]
            self.poses[:, 2] *= -1

        # Scale poses
        self.poses[:, :3, 3] *= self.cfg.scale

    def _init_frame_ids(self, frame_count: int) -> List[int]:
        if self.max_count is None or self.max_count >= frame_count:
            # Use all frames
            frame_ids = np.arange(frame_count)
        else:
            assert self.max_count > 0, 'Invalid value for "max_count"'
            # Pick frames uniformly
            frame_ids = np.linspace(0, frame_count, self.max_count + 1)[:-1]
            frame_ids = np.round(frame_ids).astype(int)

        width = len(str(frame_count))
        self.frame_str_ids = ['{:0{width}d}'.format(i, width=width) for i in frame_ids]

        return list(frame_ids)

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index]

    def __len__(self):
        assert len(self.imgs) == len(self.poses)
        return len(self.imgs)

    def __str__(self, name: Optional[str] = None) -> str:
        cls_name = self.__class__.__name__
        if name is None:
            name = self.cfg.root_path.stem
        desc = '{} \"{}\" {} split with {:d} entries'
        split_str = ['train', 'validation', 'test'][self.split.value]
        return desc.format(cls_name, name, split_str, len(self))
