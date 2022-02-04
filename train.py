import argparse
from itertools import cycle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import DatasetConfig, NetworkConfig, TrainConfig
from nerf_lib import NerfLib
from ray_batch import RayBatch
from networks.nerf import Nerf
from data.nsvf_dataset import NSVFDataset
from utils import batch, compute_psnr, create_logger


class PretrainTrainer:
    def __init__(self, args):
        self.logger = create_logger(__name__)
        self.iter_ctr = 0
        self.time0 = 0

        self.dataset_cfg = DatasetConfig.load(args.dataset_cfg)
        self.net_cfg = NetworkConfig.load()
        self.train_cfg = TrainConfig.load(
            'cfgs/training/{}.yaml'.format(args.mode))

        self.name = args.name
        self.log_path: Path = Path('./runs') / self.name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda:0')
        self.lib = NerfLib(self.net_cfg, self.train_cfg, self.device)
        self.writer = SummaryWriter(log_dir=self.log_path)

        np.random.seed(self.train_cfg.rng_seed)
        torch.manual_seed(self.train_cfg.rng_seed)
        torch.cuda.manual_seed(self.train_cfg.rng_seed)

        # Initialize model
        x_channels, d_channels = 3, 3
        x_enc_channels = 2 * x_channels * self.net_cfg.x_enc_count + x_channels
        d_enc_channels = 2 * d_channels * self.net_cfg.d_enc_count + d_channels
        self.model = Nerf(x_enc_channels, d_enc_channels,
                          8, 256, [256, 128], [5]).to(self.device)
        self.logger.info('Created model ' + str(self.model))

        # Initialize optimizer
        self.optim = torch.optim.Adam(params=self.model.parameters(),
                                      lr=self.train_cfg.initial_learning_rate,
                                      betas=(0.9, 0.999))

        # Load checkpoint if provided
        if args.ckpt_path is None:
            self.logger.info('Training model from scratch')
        else:
            self.load_ckpt(args.ckpt_path)

        # Initialize dataset
        self.train_set = NSVFDataset(self.dataset_cfg.root_path, 'train')
        self.train_loader = cycle(DataLoader(self.train_set, batch_size=None,
                                             shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

    @staticmethod
    def calc_loss(rendered, target):
        mse_loss = torch.mean((rendered - target) ** 2)
        return mse_loss

    def check_interval(self, interval, after=0):
        return (self.iter_ctr % interval == 0) and (self.iter_ctr > after)

    def print_status(self, loss, psnr):
        log_str = '[TRAIN] Iter: {:d}, Loss: {:.5f}, PSNR: {:.5f}'
        self.logger.info(log_str.format(
            self.iter_ctr, loss.item(), psnr.item()))

    def log_status(self, loss, psnr, cur_lr):
        self.writer.add_scalar('train/loss', loss.item(), self.iter_ctr)
        self.writer.add_scalar('train/psnr', psnr.item(), self.iter_ctr)
        self.writer.add_scalar('misc/iter_time', time.time() - self.time0,
                               self.iter_ctr)
        self.writer.add_scalar('misc/cur_lr', cur_lr, self.iter_ctr)

    def load_ckpt(self, ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
            self.iter_ctr = ckpt['iter']
            self.model.load_state_dict(ckpt['model'])
            self.optim.load_state_dict(ckpt['optim'])

            rng_states = ckpt['rng_states']
            np.random.set_state(rng_states['np'])
            torch.set_rng_state(rng_states['torch'])
            torch.cuda.set_rng_state(rng_states['torch_cuda'])

        except FileNotFoundError:
            self.logger.error(
                'Checkpoint file \"{}\" not found'.format(ckpt_path))
        except KeyError:
            self.logger.error(
                'Checkpoint file \"{}\" has invalid format'.format(ckpt_path))

        self.logger.info('Loaded checkpoint \"{}\"'.format(ckpt_path))
        self.logger.info('Model now at iteration #{:d}'.format(self.iter_ctr))

    def save_ckpt(self):
        ckpt_dict = {
            'iter': self.iter_ctr,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'rng_states': {
                'np': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state()
            }
        }

        ckpt_fn = 'iter_{:0{width}d}.pth'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        ckpt_path = self.log_path / ckpt_fn

        torch.save(ckpt_dict, ckpt_path)
        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)

        # Generate rays
        precrop = None
        if self.iter_ctr < self.train_cfg.precrop_iterations:
            precrop = self.train_cfg.precrop_fraction
        rays_o, rays_d, coords = self.lib.generate_rays(
            self.train_set.intrinsics, pose, precrop=precrop)
        target = torch.FloatTensor(img[coords]).to(self.device)
        rays = RayBatch(rays_o, rays_d, self.train_set.near,
                        self.train_set.far)

        # Render rays
        pts, dists = self.lib.sample_points(rays)
        dirs = rays.viewdirs()

        pts_flat = pts.reshape(-1, 3)
        pts_embedded = self.lib.embed_x(pts_flat)
        dirs_embedded = self.lib.embed_d(dirs)
        dirs_embedded = torch.repeat_interleave(
            dirs_embedded, repeats=self.net_cfg.num_samples_per_ray, dim=0)

        rgbs, densities = [], []
        for pts_batch, dirs_batch in batch(pts_embedded, dirs_embedded,
                                           bsize=self.net_cfg.pts_bsize):
            out_c, out_a = self.model(pts_batch, dirs_batch)
            rgbs.append(out_c)
            densities.append(out_a)

        # Compute loss and update weights
        rgbs = torch.concat(rgbs, dim=0).reshape(pts.shape)
        densities = torch.concat(densities, dim=0).reshape(pts.shape[:-1])
        rgb_map = self.lib.integrate_points(dists, rgbs, densities)
        loss = self.calc_loss(rendered=rgb_map, target=target)
        psnr = compute_psnr(loss)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update counter after backprop
        self.iter_ctr += 1

        new_lr = self.train_cfg.initial_learning_rate
        if self.train_cfg.learning_rate_decay:
            new_lr = self.train_cfg.initial_learning_rate * \
                (0.1 ** (self.iter_ctr / self.train_cfg.learning_rate_decay))
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lr

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(loss, psnr)
        if self.check_interval(self.train_cfg.intervals.log):
            self.log_status(loss, psnr, new_lr)
        if self.check_interval(self.train_cfg.intervals.ckpt):
            self.save_ckpt()

    def run(self):
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('--mode', choices=['pretrain', 'distill', 'finetune'],
                        default='pretrain')
    parser.add_argument('--name', default='tmp')
    parser.add_argument('--ckpt-path')

    args = parser.parse_args()
    trainer = PretrainTrainer(args)
    trainer.run()


if __name__ == '__main__':
    train()
