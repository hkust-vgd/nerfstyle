import importlib
from trainers.base import Trainer


def get_trainer(args, nargs) -> Trainer:
    if args.mode in ['pretrain', 'finetune']:
        module_name = 'trainers.end2end'
        class_name = 'End2EndTrainer'
    elif args.mode == 'distill':
        module_name = 'trainers.distill'
        class_name = 'DistillTrainer'
    else:
        raise NotImplementedError

    module = importlib.import_module(module_name)
    module_ctor = getattr(module, class_name)
    trainer = module_ctor(args, nargs)

    return trainer
