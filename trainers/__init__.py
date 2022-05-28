from argparse import Namespace
import importlib
from typing import List

from trainers.base import Trainer


def get_trainer(
    args: Namespace,
    nargs: List[str]
) -> Trainer:
    """
    Returns trainer object dynamically.

    Args:
        args (Namespace): Command line arguments.
        nargs (List[str]): Overwritten config parameters.

    Raises:
        NotImplementedError: invalid value for `args.mode`.

    Returns:
        Trainer: Trainer object.
    """
    if args.mode in ['pretrain', 'finetune']:
        module_name = 'trainers.volumetric'
        class_name = 'VolumetricTrainer' if args.style_image is None else 'StyleTrainer'
    elif args.mode == 'distill':
        module_name = 'trainers.distill'
        class_name = 'DistillTrainer'
    else:
        raise NotImplementedError

    module = importlib.import_module(module_name)
    module_ctor = getattr(module, class_name)
    trainer = module_ctor(args, nargs)

    return trainer
