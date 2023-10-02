from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor
from common import DatasetSplit
import utils


class SingleImage(Dataset):
    def __init__(
        self,
        image_path: Path,
        longer_edge: Optional[int] = None
    ):
        self.image_name = image_path.name
        style_image_np = utils.parse_rgb(image_path, size=longer_edge)
        self.style_image = torch.tensor(style_image_np)

    def __getitem__(self, _):
        return self.style_image

    def __len__(self):
        return 1

    def __str__(self) -> str:
        desc = 'single style image "{}"'.format(self.image_name)
        return desc


class WikiartDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        split: DatasetSplit,
        max_images: Optional[int] = 100,
        fix_id: Optional[int] = None
    ):
        super().__init__()

        self.root_dir = Path(root_path)
        self.split = split
        img_dir = self.root_dir / split.name.lower()

        self.paths = sorted(img_dir.glob('*.jpg'))
        if max_images is not None:
            self.paths = self.paths[:max_images]

        transforms = [
            RandomResizedCrop(256, scale=(0.4, 0.6), ratio=(1.0, 1.0)),
            ToTensor()
        ]

        self.transform = Compose(transforms)
        self.fix_id = fix_id

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __getitem__(self, index):
        if self.fix_id is not None:
            index = self.fix_id

        img = Image.open(self.paths[index])
        img = img.convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        if self.fix_id:
            return 1
        return len(self.paths)

    def __str__(self) -> str:
        desc = 'WikiartDataset {} split with {:d} entries'
        split_str = ['train', 'validation', 'test'][self.split.value]
        return desc.format(split_str, len(self))
