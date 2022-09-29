from abc import ABC, abstractmethod
from pathlib import Path
import sys
from typing import List
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from common import TrainMode
from config import BaseConfig, DatasetConfig, NetworkConfig, RendererConfig, TrainConfig
from nerf_lib import nerf_lib
import utils


class Trainer(ABC):
    def __init__(
        self,
        name: str,
        cfg: BaseConfig,
        nargs: List[str]
    ) -> None:
        self.logger = utils.create_logger(name)
        self.iter_ctr = 0
        self.time0 = 0
        self.time1 = 0

        self.name = cfg.name
        self.log_dir: Path = Path(cfg.run_dir) / self.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if next(self.log_dir.iterdir(), None) is not None:
            # log dir is not empty
            if (cfg.ckpt_path is not None) and (Path(cfg.ckpt_path).parent == self.log_dir):
                # loading from existing checkpoint in log dir
                proceed = utils.prompt_bool(
                    'Checkpoints beyond "{}" will be overwritten. Proceed?'.format(cfg.ckpt_path))
            else:
                proceed = utils.prompt_bool('Log directory not empty. Clean directory?')
                if proceed:
                    utils.rmtree(self.log_dir)
                    self.log_dir.mkdir()

            if not proceed:
                sys.exit(1)

        # Parse arguments
        net_cfg_path = None if cfg.mode == TrainMode.PRETRAIN else 'cfgs/network/kilonerf.yaml'
        # train_cfg_path = 'cfgs/training/{}.yaml'.format(cfg.mode.name.lower())

        self.dataset_cfg, nargs = DatasetConfig.load_nargs(cfg.data_cfg_path, nargs=nargs)
        self.train_cfg, nargs = TrainConfig.load_nargs(nargs=nargs)
        self.net_cfg, nargs = NetworkConfig.load_nargs(net_cfg_path, nargs=nargs)
        self.render_cfg, nargs = RendererConfig.load_nargs(nargs=nargs)
        if len(nargs) > 0:
            self.logger.error('Unrecognized arguments: ' + ' '.join(nargs))

        self.device = torch.device('cuda:0')

        nerf_lib.device = self.device
        nerf_lib.load_cuda_ext()
        nerf_lib.init_stream_pool(16)
        nerf_lib.init_magma()

        self.writer = None
        if self.train_cfg.intervals.log > 0:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        np.random.seed(self.train_cfg.rng_seed)
        torch.manual_seed(self.train_cfg.rng_seed)
        torch.cuda.manual_seed(self.train_cfg.rng_seed)

    def check_interval(self, interval, after=0, final=False):
        if interval <= 0:
            return False

        is_final = (self.iter_ctr == self.train_cfg.num_iterations) and final
        flag = (((self.iter_ctr % interval == 0) or is_final) and (self.iter_ctr > after))
        return flag

    def print_status(self, status_dict, phase='TRAIN', out_fn=None):
        if out_fn is None:
            out_fn = self.logger.info
        log_items = [k + ': ' + str(v) for k, v in status_dict.items()]
        log_str = '[{}] Iter: {:d}, {}'.format(phase, self.iter_ctr, ', '.join(log_items))
        out_fn(log_str)

    @abstractmethod
    def run_iter(self):
        pass

    def run(self):
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()

    def close(self):
        nerf_lib.destroy_stream_pool()
        nerf_lib.deinit_multimatmul_aux_data()
        self.logger.info('Closed')
