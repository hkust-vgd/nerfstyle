import importlib
import pickle
from typing import List

import torch

from config import BaseConfig
from trainers.base import Trainer


def unpickle_trainer(ckpt_path: str) -> Trainer:
    with open(ckpt_path, 'rb') as f:
        trainer = pickle.load(f)

    assert isinstance(trainer, Trainer)
    return trainer


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

    if cfg.ckpt is None:
        # (1) Initalize new trainer and train from scratch
        trainer = module_ctor(cfg, nargs)
        trainer.logger.info('Initialized new {} from scratch'.format(class_name))
        return trainer

    # checkpoint is provided
    ckpt_trainer = unpickle_trainer(cfg.ckpt)
    if cfg.style_image is not None:
        # (2) Initialize new trainer and load model / renderer
        trainer = module_ctor(cfg, nargs, ckpt_trainer)
        trainer.logger.info('Initialized new {} from checkpoint \"{}\"'.format(
            class_name, cfg.ckpt))
        return trainer

    # (3) Continue training from existing checkpoint
    ckpt_trainer.device = torch.device('cuda:0')  # hard code for now
    ckpt_trainer.model.to(ckpt_trainer.device)
    ckpt_trainer.renderer.to(ckpt_trainer.device)
    ckpt_trainer.logger.info('Loaded checkpoint \"{}\"'.format(cfg.ckpt))
    return ckpt_trainer
