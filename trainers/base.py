from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from nerf_lib import NerfLib
from utils import create_logger


class Trainer:
    def __init__(self, name, args, nargs):
        self.logger = create_logger(name)
        self.iter_ctr = 0
        self.time0 = 0

        self.name = args.name
        self.log_path: Path = Path('./runs') / self.name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Parse args
        self.dataset_cfg, nargs = config.DatasetConfig.load(
            args.dataset_cfg, nargs=nargs)
        self.net_cfg, nargs = config.NetworkConfig.load(nargs=nargs)
        self.train_cfg, nargs = config.TrainConfig.load(
            'cfgs/training/{}.yaml'.format(args.mode), nargs=nargs)
        if len(nargs) > 0:
            self.logger.error('Unrecognized arguments: ' + ' '.join(nargs))

        self.device = torch.device('cuda:0')
        self.lib = NerfLib(self.net_cfg, self.train_cfg, self.device)
        self.writer = SummaryWriter(log_dir=self.log_path)

        np.random.seed(self.train_cfg.rng_seed)
        torch.manual_seed(self.train_cfg.rng_seed)
        torch.cuda.manual_seed(self.train_cfg.rng_seed)

    def check_interval(self, interval, after=0):
        return (self.iter_ctr % interval == 0) and (self.iter_ctr > after)

    def print_status(self, status_dict):
        log_items = [k + ': ' + str(v) for k, v in status_dict.items()]
        log_str = '[TRAIN] Iter: {:d}, '.format(self.iter_ctr) + \
            ', '.join(log_items)
        self.logger.info(log_str)

    def run_iter(self):
        pass

    def run(self):
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()
