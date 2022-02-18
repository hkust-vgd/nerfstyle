from .base import Trainer


# TODO: use importlib instead
def get_trainer(args, nargs) -> Trainer:
    if args.mode == 'pretrain':
        from .end2end import End2EndTrainer
        return End2EndTrainer(args, nargs)
    elif args.mode == 'distill':
        from .distill import DistillTrainer
        return DistillTrainer(args, nargs)
    else:
        raise NotImplementedError
