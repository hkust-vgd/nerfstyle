import importlib
from config import DatasetConfig
from data.base_dataset import BaseDataset


def get_dataset(
    dataset_cfg: DatasetConfig,
    split: str,
    skip: int = 1
) -> BaseDataset:
    dataset_type = dataset_cfg.type

    module_name = 'data.{}_dataset'.format(dataset_type.lower())
    class_name = '{}Dataset'.format(dataset_type)

    module = importlib.import_module(module_name)
    module_ctor = getattr(module, class_name)
    dataset = module_ctor(dataset_cfg, split, skip)

    return dataset


def load_bbox(
    dataset_cfg: DatasetConfig
):
    dataset_type = dataset_cfg.type
    module_name = 'data.{}_dataset'.format(dataset_type.lower())
    loader_fn_name = 'load_bbox'

    module = importlib.import_module(module_name)
    module_loader = getattr(module, loader_fn_name)
    bbox = module_loader(dataset_cfg)

    return bbox
