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
    dataset_cfg: DatasetConfig,
    bbox_fn: str = 'bbox.txt'
):
    dataset_type = dataset_cfg.type
    bbox_path = dataset_cfg.root_path / bbox_fn

    module_name = 'data.{}_dataset'.format(dataset_type.lower())
    loader_fn_name = 'load_bbox'

    module = importlib.import_module(module_name)
    module_loader = getattr(module, loader_fn_name)
    bbox = module_loader(bbox_path)

    return bbox
