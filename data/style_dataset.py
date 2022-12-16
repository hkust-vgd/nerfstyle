from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor
from common import DatasetSplit


class WikiartDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        split: DatasetSplit,
        fix_id: Optional[int] = None
    ):
        super().__init__()

        self.root_dir = Path(root_path)
        self.split = split
        img_dir = self.root_dir / split.name.lower()

        self.paths = sorted(img_dir.glob('*.jpg'))

        transforms = [
            Resize(800),
            RandomCrop(800),
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
        return len(self.paths)
