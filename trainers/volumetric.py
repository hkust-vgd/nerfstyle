from argparse import Namespace
from functools import partial
import time
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from common import LossValue, TrainMode
from config import BaseConfig
from data import get_dataset, load_bbox
from loss import FeatureExtractor, MattingLaplacian, StyleLoss
from networks.multi_nerf import DynamicMultiNerf
from networks.single_nerf import SingleNerf
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

        # Initialize model
        if cfg.mode == TrainMode.PRETRAIN:
            self.model = SingleNerf(self.net_cfg)
        elif cfg.mode == TrainMode.FINETUNE:
            self.model = DynamicMultiNerf(self.net_cfg, self.dataset_cfg)
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

        # Initialize dataset
        self.train_set = get_dataset(self.dataset_cfg, 'train')
        self.train_loader = utils.cycle(DataLoader(self.train_set, batch_size=None, shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

        self.test_set = get_dataset(self.dataset_cfg, 'test', skip=self.train_cfg.test_skip,
                                    max_count=30)
        self.test_loader = DataLoader(self.test_set, batch_size=None, shuffle=False)
        self.logger.info('Loaded ' + str(self.test_set))

        # Initialize renderers
        self.train_renderer, self.test_renderer = self._get_renderers()

        # Load bbox if needed
        self.bbox = None
        if self.train_cfg.bbox_lambda > 0.0:
            self.bbox = load_bbox(self.dataset_cfg, scale_box=False).to(self.device)

    def _get_renderers(self) -> Tuple[Renderer, Renderer]:
        intr = self.train_set.intrinsics
        near, far = self.train_set.near, self.train_set.far

        train_renderer = Renderer(
            self.model, self.net_cfg, intr, near, far,
            precrop_frac=self.train_cfg.precrop_fraction,
            num_rays=self.train_cfg.num_rays_per_batch, name='trainRenderer')
        test_renderer = Renderer(
            self.model, self.net_cfg, intr, near, far, name='testRenderer', use_ert=True)

        return train_renderer, test_renderer

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
            output = self.test_renderer.render(pose)

            _, h, w = img.shape
            rgb_output = einops.rearrange(output['rgb_map'], '(h w) c -> c h w', h=h, w=w)
            save_path = img_dir / 'frame_{:03d}.png'.format(i)
            torchvision.utils.save_image(rgb_output, save_path)

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)

        self.train_renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
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

    def _get_renderers(self) -> Tuple[Renderer, Renderer]:
        intr = self.train_set.intrinsics
        sparse_intr = intr.scale(*self.style_dims)
        near, far = self.train_set.near, self.train_set.far

        train_renderer = Renderer(
            self.model, self.net_cfg, sparse_intr, near, far,
            precrop_frac=self.train_cfg.precrop_fraction, name='trainRenderer')
        test_renderer = Renderer(
            self.model, self.net_cfg, intr, near, far, name='testRenderer')

        return train_renderer, test_renderer

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
