import importlib
from typing import List

import torch

from config import BaseConfig
from trainers.base import Trainer


def get_trainer(
    cfg: BaseConfig,
    nargs: List[str]
) -> Trainer:
    """
    Returns trainer object dynamically.

    Args:
        cfg (BaseConfig): Command line arguments.
        nargs (List[str]): Overwritten config parameters.

    Returns:
        Trainer: Trainer object.
    """
    if cfg.style_image is None:
        module_name = 'trainers.base'
        class_name = 'Trainer'
    else:
        module_name = 'trainers.style'
        class_name = 'StyleTrainer'

    module = importlib.import_module(module_name)
    module_ctor: Trainer = getattr(module, class_name)

    trainer = module_ctor(cfg, nargs)
    return trainer
