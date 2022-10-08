import importlib
from typing import List

from common import TrainMode
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

    Raises:
        NotImplementedError: invalid value for `args.mode`.

    Returns:
        Trainer: Trainer object.
    """
    if cfg.mode is TrainMode.DISTILL:
        module_name = 'trainers.distill'
        class_name = 'DistillTrainer'
    else:
        module_name = 'trainers.volumetric'
        class_name = 'VolumetricTrainer' if cfg.style_image is None else 'StyleTrainer'

    module = importlib.import_module(module_name)
    module_ctor: Trainer = getattr(module, class_name)

    if cfg.ckpt_path is None:
        # train from scratch, initalize new trainer
        trainer = module_ctor(cfg, nargs)
    else:
        # unpickle trianer from checkpoint
        trainer = module_ctor.load_ckpt(cfg.ckpt_path)

    return trainer
