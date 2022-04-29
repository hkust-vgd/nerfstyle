import importlib
from config import DatasetConfig
from data.base_dataset import BaseDataset


def get_dataset(
    dataset_cfg: DatasetConfig,
    split: str,
    skip: int = 1
) -> BaseDataset:
    dataroot = dataset_cfg.root_path
    dataset_type = dataset_cfg.type

    module_name = 'data.{}_dataset'.format(dataset_type.lower())
    class_name = '{}Dataset'.format(dataset_type)

    module = importlib.import_module(module_name)
    module_ctor = getattr(module, class_name)
    dataset = module_ctor(dataroot, split, skip)

    return dataset
