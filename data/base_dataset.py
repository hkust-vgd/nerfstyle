from abc import ABC
from pathlib import Path
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset

from common import BBox, DatasetSplit, Intrinsics
from config import DatasetConfig
import utils

import einops
import torch


class BaseDataset(Dataset, ABC):
    """NeRF dataset base class."""

    # Interface properties

    fns: List[str]
    """Array of image filenames (without extension)."""

    images: np.ndarray
    """Array of images. Size should be [N, 3, H, W]."""

    poses: np.ndarray
    """Array of poses. Size should be [N, 4, 4]."""

    bbox: BBox
    """Bounding box that bounds the entire scene volume (but not \
        necessarily the camera origins)."""

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

        # Load poses
        self.poses = self._get_poses()
        assert len(self.poses.shape) == 3
        assert self.poses.shape[1] == self.poses.shape[2] == 4
        self.poses[:, :3, 3] *= self.cfg.scale

        # Load images
        image_paths = self._get_image_paths()
        self.has_gt = (image_paths is not None)
        if self.has_gt:
            self.fns = [path.stem for path in image_paths]
            if len(set(self.fns)) != len(self.fns):
                # Include parent directory name
                self.fns = [path.parent.stem + '_' + path.stem for path in image_paths]

            self.images = np.stack([utils.parse_rgb(path) for path in image_paths])
            if self.images.shape[1] == 4:
                rgb, alpha = self.images[:, :3], self.images[:, 3:]
                self.images = rgb * alpha + (1 - alpha)
            assert len(self.images) == len(self.poses)
        else:
            self.images = None
            w = len(str(len(self)))
            self.fns = ['frame_{:0{w}d}'.format(i, w=w) for i in range(len(self))]

        # Load segment groups
        self.seg_groups, self.num_classes = None, 0
        if self.split == DatasetSplit.TRAIN:
            self.seg_groups = self._get_seg_groups()
            unique_groups = np.unique(self.seg_groups)
            self.num_classes = len(unique_groups)
            assert self.seg_groups.shape[-2:] == self.images.shape[-2:]
            assert np.all(unique_groups == np.arange(self.num_classes))

        # Color transform
        if self.cfg.ct_image is not None and self.images is not None:
            gt_images = einops.rearrange(self.images, 'n c h w -> n h w c')
            gt_images = torch.from_numpy(gt_images).cuda()
            style_image = utils.parse_rgb(self.cfg.ct_image)
            style_image = einops.rearrange(style_image, 'c h w -> h w c')
            style_image = torch.from_numpy(style_image).cuda()
            ct_result, _ = utils.match_colors_for_image_set(gt_images, style_image)
            self.images = einops.rearrange(ct_result, 'n h w c -> n c h w').cpu().numpy()

        # Set frames
        if self.max_count is None or self.max_count >= len(self):
            # Use all frames
            frame_ids = np.arange(len(self))
        else:
            assert self.max_count > 0, 'Invalid value for "max_count"'
            # Pick frames uniformly
            frame_ids = np.linspace(0, len(self), self.max_count + 1)[:-1]
            frame_ids = np.round(frame_ids).astype(int)

            self.fns = [self.fns[i] for i in frame_ids]
            self.poses = self.poses[frame_ids]
            if self.has_gt:
                self.images = self.images[frame_ids]

        # Load intrinsic matrix(s)
        self.intr = self._get_intr()

        # Set bounding box
        self.bbox = BBox.from_radius(self.cfg.bound)

    def _get_image_paths(self) -> List[Path]:
        pass

    def _get_poses(self) -> np.ndarray:
        pass

    def _get_seg_groups(self) -> np.ndarray:
        pass

    def _get_intr(self) -> Intrinsics:
        pass

    def __getitem__(self, index):
        if self.seg_groups is not None:
            seg_groups = (self.seg_groups[index]).astype(np.float32)
            image = np.concatenate((self.images[index], seg_groups[None]), axis=0)
            return image, self.poses[index]

        if self.has_gt:
            return self.images[index], self.poses[index]
        return None, self.poses[index]

    def __len__(self):
        return len(self.poses)

    def __str__(self, name: Optional[str] = None) -> str:
        cls_name = self.__class__.__name__
        if name is None:
            name = self.cfg.root_path.stem
        desc = '{} \"{}\" {} split with {:d} entries'
        split_str = ['train', 'validation', 'test'][self.split.value]
        return desc.format(cls_name, name, split_str, len(self))
