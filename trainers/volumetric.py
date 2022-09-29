from argparse import Namespace
from functools import partial
import time
from typing import Dict, List

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from common import LossValue, TrainMode
from config import BaseConfig
from data import get_dataset
from loss import FeatureExtractor, MattingLaplacian, StyleLoss
from networks.tcnn_nerf import TCNerf
from renderer import Renderer
from trainers.base import Trainer
import utils


class VolumetricTrainer(Trainer):
    def __init__(
        self,
        cfg: BaseConfig,
        nargs: List[str]
    ) -> None:
        """
        Default volumetric rendering trainer.

        Args:
            cfg (BaseConfig): Command line arguments.
            nargs (List[str]): Overwritten config parameters.
        """
        super().__init__(__name__, cfg, nargs)

        # Initialize dataset
        self.train_set = get_dataset(self.dataset_cfg, 'train')
        self.train_set.bbox = self.train_set.bbox.to(self.device)
        self.train_loader = utils.cycle(DataLoader(self.train_set, batch_size=None, shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

        self.test_set = get_dataset(self.dataset_cfg, 'test',
                                    max_count=self.train_cfg.max_eval_count)
        self.test_loader = DataLoader(self.test_set, batch_size=None, shuffle=False)
        self.logger.info('Loaded ' + str(self.test_set))

        # Initialize model
        if cfg.mode == TrainMode.PRETRAIN:
            self.model = TCNerf(self.net_cfg, self.train_set.bbox)
        elif cfg.mode == TrainMode.FINETUNE:
            self.logger.error('Not implemented now')
        else:
            self.logger.error('Wrong training mode: {}'.format(cfg.mode.name))

        self.model = self.model.to(self.device)
        self.logger.info('Created model ' + str(self.model))

        # Initialize optimizer
        train_keys = None
        # train_keys = ['x_layers', 'd_layers', 'x2d_layer', 'c_layer']

        def trainable(key: str) -> bool:
            if train_keys is None:
                return True
            return any([(k in key) for k in train_keys])

        train_params = [p for n, p in self.model.named_parameters() if trainable(n)]

        self.optim = torch.optim.Adam(
            params=train_params,
            lr=self.train_cfg.initial_learning_rate,
            betas=(0.9, 0.999)
        )

        # Load checkpoint if provided
        if cfg.ckpt_path is None:
            self.logger.info('Training model from scratch')
        else:
            self.load_ckpt(cfg.ckpt_path)
            if cfg.occ_map is not None:
                self.model.load_occ_map(cfg.occ_map)

        # Initialize renderer
        self.renderer = Renderer(
            self.model, self.render_cfg, self.train_set.intrinsics, self.dataset_cfg.bound,
            bg_color=self.dataset_cfg.bg_color,
            precrop_frac=self.train_cfg.precrop_fraction
        )

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor]
    ) -> Dict[str, LossValue]:
        assert output['target'] is not None

        rendered = output['rgb_map']
        target = output['target']
        assert target is not None

        mse_loss = torch.mean((rendered - target) ** 2)
        losses = {
            'mse': LossValue('MSE', 'mse_loss', mse_loss),
            'psnr': LossValue('PSNR', 'psnr', utils.compute_psnr(mse_loss))
        }

        sparsity_loss = 0
        sparsity_lambda = self.train_cfg.sparsity_lambda
        if sparsity_lambda > 0.:
            coeff = self.train_cfg.sparsity_exp_coeff
            sparsity_losses = torch.abs(1 - torch.exp(-coeff * output['sparsity']))
            sparsity_loss = torch.mean(sparsity_losses) * sparsity_lambda
            losses['sparsity'] = LossValue('Sparsity', 'sparsity_loss', sparsity_loss)

        weight_reg_loss = 0
        weight_reg_lambda = self.train_cfg.weight_reg_lambda
        if weight_reg_lambda > 0.:
            view_params = [p for n, p in self.model.named_parameters()
                           if ('c_layer' in n) or ('d_layers' in n)]
            norm_sum = torch.sum(torch.stack([p.norm(2) for p in view_params]))
            weight_reg_loss = norm_sum * weight_reg_lambda
            losses['weight_reg'] = LossValue('Weight Reg.', 'weight_reg_loss', weight_reg_loss)

        if sparsity_loss + weight_reg_loss > 0:
            total_loss = mse_loss + sparsity_loss + weight_reg_loss
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
            frame_id = self.test_set.frame_str_ids[i]
            img, pose = img.to(self.device), pose.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.renderer.render(pose, training=False)

            _, h, w = img.shape
            rgb_output = einops.rearrange(output['rgb_map'], '(h w) c -> c h w', h=h, w=w)
            save_path = img_dir / 'frame_{}.png'.format(frame_id)
            torchvision.utils.save_image(rgb_output, save_path)

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)

        self.renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        with torch.cuda.amp.autocast(enabled=True):
            num_rays = self.train_cfg.num_rays_per_batch
            output = self.renderer.render(pose, img, num_rays=num_rays, training=True)

        if self.train_cfg.sparsity_lambda > 0.:
            bbox = self.train_set.bbox
            sparsity_pts = torch.rand((self.train_cfg.sparsity_samples, 3), device=self.device)
            sparsity_pts = sparsity_pts * bbox.size() + bbox.min_pt
            output['sparsity'] = self.model(sparsity_pts)

        losses = self.calc_loss(output)
        self.optim.zero_grad()
        back_key = 'total' if 'total' in losses.keys() else 'mse'
        losses[back_key].value.backward()
        self.optim.step()

        # Update counter after backprop
        self.iter_ctr += 1
        self.time1 = time.time()

        # TODO: Use torch.optim.lr_scheduler.LambdaLR
        lr = self.train_cfg.initial_learning_rate
        decay = self.train_cfg.learning_rate_decay
        if decay < 0:
            decay = self.train_cfg.num_iterations

        if decay != 0:
            lr = self.train_cfg.initial_learning_rate * \
                (0.1 ** (self.iter_ctr / decay))
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(losses)
        if self.check_interval(self.train_cfg.intervals.test):
            self.test_networks()
        if self.check_interval(self.train_cfg.intervals.log):
            self.log_status(losses, lr)
        if self.check_interval(self.train_cfg.intervals.ckpt, final=True):
            self.save_ckpt()

    def run(self):
        if self.train_cfg.test_before_train:
            self.test_networks()
        super().run()


class StyleTrainer(VolumetricTrainer):
    def __init__(
        self,
        args: Namespace,
        nargs: List[str]
    ) -> None:
        """
        Volumetric rendering trainer for style transfer.

        Args:
            args (Namespace): Command line arguments.
            nargs (List[str]): Overwritten config parameters.
        """
        self.style_dims = (256, 256)
        super().__init__(args, nargs)

        # Intialize losses and style image
        self.fe = FeatureExtractor().to(self.device)
        style_image_np = utils.parse_rgb(args.style_image, size=self.style_dims)
        self.style_image = torch.tensor(style_image_np, device=self.device)
        self.style_loss = StyleLoss(self.fe(self.style_image, detach=True))
        self.photo_loss = MattingLaplacian(device=self.device)

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor]
    ) -> Dict[str, LossValue]:
        assert output['target'] is not None

        nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=256, w=256)
        rgb_map_chw = nc2chw(output['rgb_map'])
        target_chw = nc2chw(output['target'])

        rgb_feats = self.fe(rgb_map_chw)
        target_feats = self.fe(target_chw)

        content_loss = F.mse_loss(rgb_feats['layer3'], target_feats['layer3'])
        style_loss = self.style_loss(rgb_feats)
        photo_loss = self.photo_loss(target_chw, rgb_map_chw)

        content_loss *= self.train_cfg.content_lambda
        style_loss *= self.train_cfg.style_lambda
        photo_loss *= self.train_cfg.photo_lambda

        losses = {
            'content': LossValue('Content', 'content_loss', content_loss),
            'style': LossValue('Style', 'style_loss', style_loss),
            'photo': LossValue('Photo', 'photo_loss', photo_loss),
        }
        total_loss = content_loss + style_loss + photo_loss

        losses['total'] = LossValue('Total', 'total_loss', total_loss)
        return losses
