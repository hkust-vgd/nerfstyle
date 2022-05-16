import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import einops
from tqdm import tqdm

from common import LossValue
from data import get_dataset, load_bbox
from networks.nerf import SingleNerf
from networks.multi_nerf import DynamicMultiNerf
from renderer import Renderer
from trainers.base import Trainer
import utils


class End2EndTrainer(Trainer):
    def __init__(self, args, nargs):
        super().__init__(__name__, args, nargs)

        # Initialize model
        if args.mode == 'pretrain':
            self.model = SingleNerf.create_nerf(self.net_cfg)
        elif args.mode == 'finetune':
            num_nets = np.prod(self.dataset_cfg.net_res)
            self.model = DynamicMultiNerf.create_nerf(num_nets, self.net_cfg, self.dataset_cfg)
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
                self.model.load_occ_map(args.occ_map)

        # Initialize dataset
        self.train_set = get_dataset(self.dataset_cfg, 'train')
        self.train_loader = utils.cycle(DataLoader(self.train_set, batch_size=None, shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

        self.test_set = get_dataset(self.dataset_cfg, 'test', skip=self.train_cfg.test_skip)
        self.test_loader = DataLoader(self.test_set, batch_size=None, shuffle=False)
        self.logger.info('Loaded ' + str(self.test_set))

        # Initialize renderers
        self.train_renderer = Renderer(
            self.model, self.train_set, self.net_cfg, self.train_cfg, all_rays=False)
        self.test_renderer = Renderer(
            self.model, self.test_set, self.net_cfg, self.train_cfg, all_rays=True)

        # Load bbox if needed
        self.bbox = None
        if self.train_cfg.bbox_lambda > 0.0:
            self.bbox = load_bbox(self.dataset_cfg, scale_box=False).to(self.device)

    def calc_loss(self, output: Dict[str, torch.Tensor]):
        rendered = output['rgb_map']
        target = output['target']
        assert target is not None

        mse_loss = torch.mean((rendered - target) ** 2)
        losses = {
            'mse': LossValue('MSE', 'mse_loss', mse_loss),
            'psnr': LossValue('PSNR', 'psnr', utils.compute_psnr(mse_loss))
        }

        # Penalize positive densities outside bbox
        bbox_lambda = self.train_cfg.bbox_lambda
        if bbox_lambda > 0.:
            bbox_mask = self.bbox(output['pts'], outside=True)
            pos_densities = F.relu(output['densities'].reshape(-1))
            bbox_loss = torch.mean(bbox_mask * pos_densities) * bbox_lambda
            losses['bbox'] = LossValue('BBox', 'bbox_loss', bbox_loss)

            total_loss = mse_loss + bbox_loss
            losses['total'] = LossValue('Total', 'total_loss', total_loss)

        return losses

    def print_status(
        self,
        losses: Dict[str, LossValue]
    ) -> None:
        status_dict = {lv.print_name: '{:.5f}'.format(lv.value.item()) for lv in losses.values()}
        super().print_status(status_dict)

    def log_status(
        self,
        losses: Dict[str, LossValue],
        cur_lr: float
    ) -> None:
        for lv in losses.values():
            self.writer.add_scalar('train/{}'.format(lv.log_name), lv.value.item(), self.iter_ctr)

        self.writer.add_scalar('misc/iter_time', self.time1 - self.time0, self.iter_ctr)
        self.writer.add_scalar('misc/cur_lr', cur_lr, self.iter_ctr)

    def load_ckpt(self, ckpt_path):
        @utils.loader(self.logger)
        def _load(ckpt_path):
            ckpt = torch.load(ckpt_path)
            if 'model' not in ckpt.keys():
                self.model.load_nodes(ckpt['trained'])
                self.logger.info('Loaded distill checkpoint \"{}\"'.format(ckpt_path))
                return

            # TODO: Remove if all old models are fixed
            if 'x_embedder.basis' not in ckpt['model'].keys():
                ckpt['model']['x_embedder.basis'] = 2 ** torch.range(0, 9, device=self.device)
            if 'd_embedder.basis' not in ckpt['model'].keys():
                ckpt['model']['d_embedder.basis'] = 2 ** torch.range(0, 3, device=self.device)

            self.iter_ctr = ckpt['iter']
            self.model.load_ckpt(ckpt)
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
            'optim': self.optim.state_dict(),
            'rng_states': {
                'np': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state()
            }
        }
        ckpt_dict = self.model.save_ckpt(ckpt_dict)

        ckpt_fn = 'iter_{:0{width}d}.pth'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        ckpt_path = self.log_dir / ckpt_fn

        torch.save(ckpt_dict, ckpt_path)
        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    @torch.no_grad()
    def test_networks(self):
        img_dir = self.log_dir / 'epoch_{:0{width}d}'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        img_dir.mkdir()

        for i, (img, pose) in tqdm(enumerate(self.test_loader), total=len(self.test_set)):
            img, pose = img.to(self.device), pose.to(self.device)
            rgb_map = self.test_renderer.render(pose)['rgb_map']

            rgb_output = einops.rearrange(rgb_map.reshape(img.shape), 'h w c -> c h w')
            save_path = img_dir / 'frame_{:03d}.png'.format(i)
            torchvision.utils.save_image(rgb_output, save_path)

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)

        self.train_renderer.precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        ret_flags = ['densities', 'pts']
        output = self.train_renderer.render(pose, img, ret_flags)

        losses = self.calc_loss(output)
        self.optim.zero_grad()
        back_key = 'total' if 'total' in losses.keys() else 'mse'
        losses[back_key].value.backward()
        self.optim.step()

        # Update counter after backprop
        self.iter_ctr += 1
        self.time1 = time.time()

        new_lr = self.train_cfg.initial_learning_rate
        if self.train_cfg.learning_rate_decay:
            new_lr = self.train_cfg.initial_learning_rate * \
                (0.1 ** (self.iter_ctr / self.train_cfg.learning_rate_decay))
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lr

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(losses)
        if self.check_interval(self.train_cfg.intervals.test):
            self.test_networks()
        if self.check_interval(self.train_cfg.intervals.log):
            self.log_status(losses, new_lr)
        if self.check_interval(self.train_cfg.intervals.ckpt, final=True):
            self.save_ckpt()
