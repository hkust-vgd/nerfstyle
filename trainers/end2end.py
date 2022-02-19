import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.nsvf_dataset import NSVFDataset
from networks.nerf import SingleNerf
from networks.multi_nerf import DynamicMultiNerf
import utils
from .base import Trainer


class End2EndTrainer(Trainer):
    def __init__(self, args, nargs):
        super().__init__(__name__, args, nargs)

        # Initialize model
        if args.mode == 'pretrain':
            self.model = SingleNerf.create_nerf(self.net_cfg)
        elif args.mode == 'finetune':
            num_nets = np.prod(self.dataset_cfg.net_res)
            self.model = DynamicMultiNerf.create_nerf(
                num_nets, self.net_cfg, self.dataset_cfg)
        self.model = self.model.to(self.device)
        self.logger.info('Created model ' + str(self.model))

        # Initialize optimizer
        self.optim = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.train_cfg.initial_learning_rate,
            betas=(0.9, 0.999)
        )

        # Load checkpoint if provided
        if args.ckpt_path is None:
            self.logger.info('Training model from scratch')
        else:
            self.load_ckpt(args.ckpt_path)
            if args.occ_map is not None:
                self.model.load_occ_map(args.occ_map, self.device)

        # Initialize dataset
        self.train_set = NSVFDataset(self.dataset_cfg.root_path, 'train')
        self.train_loader = utils.cycle(DataLoader(
            self.train_set, batch_size=None, shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

    @staticmethod
    def calc_loss(rendered, target):
        mse_loss = torch.mean((rendered - target) ** 2)
        return mse_loss

    def print_status(self, loss, psnr):
        status_dict = {
            'Loss': '{:.5f}'.format(loss.item()),
            'PSNR': '{:.5f}'.format(psnr.item())
        }
        super().print_status(status_dict)

    def log_status(self, loss, psnr, cur_lr):
        self.writer.add_scalar('train/loss', loss.item(), self.iter_ctr)
        self.writer.add_scalar('train/psnr', psnr.item(), self.iter_ctr)
        self.writer.add_scalar('misc/iter_time', time.time() - self.time0,
                               self.iter_ctr)
        self.writer.add_scalar('misc/cur_lr', cur_lr, self.iter_ctr)

    def load_ckpt(self, ckpt_path):
        @utils.loader(self.logger)
        def _load(ckpt_path):
            ckpt = torch.load(ckpt_path)
            if 'model' not in ckpt.keys():
                self.model.load_nodes(ckpt['trained'], self.device)
                self.logger.info('Loaded distill checkpoint \"{}\"'.format(
                    ckpt_path))
                return

            self.iter_ctr = ckpt['iter']
            self.model.load_state_dict(ckpt['model'])
            self.optim.load_state_dict(ckpt['optim'])

            rng_states = ckpt['rng_states']
            np.random.set_state(rng_states['np'])
            torch.set_rng_state(rng_states['torch'])
            torch.cuda.set_rng_state(rng_states['torch_cuda'])

        _load(ckpt_path)
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
        ckpt_path = self.log_dir / ckpt_fn

        torch.save(ckpt_dict, ckpt_path)
        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)

        # Generate rays
        precrop = None
        if self.iter_ctr < self.train_cfg.precrop_iterations:
            precrop = self.train_cfg.precrop_fraction
        target, rays = self.lib.generate_rays(
            img, pose, self.train_set, precrop=precrop)

        # Render rays
        pts, dists = self.lib.sample_points(rays)
        dirs = rays.viewdirs()

        pts_flat = pts.reshape(-1, 3)
        dirs_flat = torch.repeat_interleave(
            dirs, repeats=self.net_cfg.num_samples_per_ray, dim=0)

        rgbs = torch.empty((len(pts_flat), 3)).to(self.device)
        densities = torch.empty((len(pts_flat), 1)).to(self.device)
        utils.batch_exec(self.model, rgbs, densities,
                         bsize=self.net_cfg.pts_bsize)(pts_flat, dirs_flat)
        rgbs = rgbs.reshape(*dists.shape, 3)
        densities = densities.reshape(dists.shape)

        # Compute loss and update weights
        bg_color = torch.tensor(self.train_set.bg_color).to(self.device)
        rgb_map = self.lib.integrate_points(dists, rgbs, densities, bg_color)
        loss = self.calc_loss(rendered=rgb_map, target=target)
        psnr = utils.compute_psnr(loss)

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