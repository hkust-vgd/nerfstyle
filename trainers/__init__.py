from .base import Trainer


def get_trainer(args, nargs):
    if args.mode == 'pretrain':
        from .pretrained import PretrainTrainer
        return PretrainTrainer(args, nargs)
    elif args.mode == 'distill':
        from .distill import DistillTrainer
        return DistillTrainer(args, nargs)
    else:
        raise NotImplementedError
