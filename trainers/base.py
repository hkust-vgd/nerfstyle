from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from data.nsvf_dataset import NSVFDataset
from nerf_lib import NerfLib
from utils import create_logger, cycle


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

        # Initialize dataset
        self.train_set = NSVFDataset(self.dataset_cfg.root_path, 'train')
        self.train_loader = cycle(DataLoader(self.train_set, batch_size=None,
                                             shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

    def run_iter(self):
        pass

    def run(self):
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()
