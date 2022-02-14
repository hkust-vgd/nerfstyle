import torch
from .base import Trainer
from config import DatasetConfig, NetworkConfig, TrainConfig
from networks.multi_nerf import create_multi_nerf


class DistillTrainer(Trainer):
    def __init__(self, args, nargs):
        super().__init__(__name__, args, nargs)

        self.dataset_cfg, nargs = DatasetConfig.load(
            args.dataset_cfg, nargs=nargs)
        self.net_cfg, nargs = NetworkConfig.load(nargs=nargs)
        # self.train_cfg, nargs = TrainConfig.load(
        #     'cfgs/training/{}.yaml'.format(args.mode), nargs=nargs)
        if len(nargs) > 0:
            self.logger.error('Unrecognized arguments: ' + ' '.join(nargs))

        model = create_multi_nerf(100, self.net_cfg).to(self.device)
        x_tmp = torch.FloatTensor(100, 32, 63).to(self.device)
        d_tmp = torch.FloatTensor(100, 32, 27).to(self.device)
        out_c, out_a = model(x_tmp, d_tmp)
        print(out_c.shape, out_a.shape)

    # TODO: tmp placeholder
    def run(self):
        pass
