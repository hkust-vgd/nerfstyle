from argparse import Namespace
from copy import copy
from functools import partial
import pickle
import time
from typing import Dict, List

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from common import LossValue, TrainMode
from config import BaseConfig
from data import get_dataset
from loss import FeatureExtractor, MattingLaplacian, StyleLoss
from nerf_lib import nerf_lib
from networks.tcnn_nerf import TCNerf
from renderer import Renderer
from trainers.base import Trainer
import utils


class VolumetricTrainer(Trainer):
    SAVE_KEYS = [
        'version', 'name', 'model', 'renderer',
        'dataset_cfg', 'train_cfg'
    ]

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
            enc_dtype = None if self.train_cfg.enable_amp else torch.float32
            self.model = TCNerf(self.net_cfg, self.train_set.bbox, enc_dtype)
        elif cfg.mode == TrainMode.FINETUNE:
            self.logger.error('Not implemented now')
        else:
            self.logger.error('Wrong training mode: {}'.format(cfg.mode.name))

        self.model = self.model.to(self.device)
        self.logger.info('Created model ' + str(type(self.model)))

        # Initialize optimizer and miscellaneous components
        # train_keys = None
        # def trainable(key: str) -> bool:
        #     if train_keys is None:
        #         return True
        #     return any([(k in key) for k in train_keys])

        # train_params = [p for n, p in self.model.named_parameters() if trainable(n)]
        train_params = list(self.model.parameters())

        self.optim = torch.optim.Adam(
            params=train_params,
            lr=self.train_cfg.initial_learning_rate,
            betas=(0.9, 0.999)
        )

        def lr_lambda(_): return 1.
        if self.train_cfg.learning_rate_decay > 0:
            def lr_lambda(iter: int): return 0.1 ** (iter / self.train_cfg.learning_rate_decay)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.enable_amp)
        self.ema = utils.EMA(train_params, decay=self.train_cfg.ema_decay)

        # Initialize renderer
        self.renderer = Renderer(
            self.model, self.render_cfg, self.train_set.intrinsics, self.dataset_cfg.bound,
            bg_color=self.dataset_cfg.bg_color,
            precrop_frac=self.train_cfg.precrop_fraction
        )

    def __getstate__(self):
        state_dict = {k: v for k, v in self.__dict__.items() if k in self.SAVE_KEYS}

        # TODO: list of items using state_dict
        state_dict['ema'] = self.ema.state_dict()

        return state_dict

    def __setstate__(self, state_dict: Dict):
        self.logger = utils.create_logger(state_dict['name'])

        # Version check
        cur_ver = utils.get_git_sha()
        pkl_ver = state_dict['version']
        if cur_ver != pkl_ver:
            self.logger.warn(
                'Checkpoint version "{}" differs from current repo version "{}". '
                'Errors may occur during loading.'.format(pkl_ver[:7], cur_ver[:7])
            )

        for k in self.SAVE_KEYS:
            if k not in state_dict.keys():
                self.logger.error('Key "{}" not found in checkpoint'.format(k))
            setattr(self, k, state_dict[k])

        # TODO: recover training components / RNG states

        # TODO: infer params from saved optimizer
        self.ema = utils.EMA(self.model.parameters(), self.train_cfg.ema_decay)
        self.ema.load_state_dict(state_dict['ema'])

    @classmethod
    def load_ckpt(
        cls,
        ckpt_path: str
    ) -> 'VolumetricTrainer':
        with open(ckpt_path, 'rb') as f:
            trainer: VolumetricTrainer = pickle.load(f)

        trainer.device = torch.device('cuda:0')  # hard code for now
        trainer.model.to(trainer.device)
        trainer.renderer.to(trainer.device)
        trainer.logger.info('Loaded checkpoint \"{}\"'.format(ckpt_path))

        return trainer

    def save_ckpt(self) -> None:
        ckpt_fn = 'iter_{:0{width}d}.pkl'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        ckpt_path = self.log_dir / ckpt_fn

        with open(ckpt_path, 'wb') as f:
            pickle.dump(self, f)

        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor]
    ) -> Dict[str, LossValue]:
        assert output['target'] is not None
        rendered = output['rgb_map']
        target = output['target']

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
        losses: Dict[str, LossValue],
        **kwargs
    ) -> None:
        status_dict = {lv.print_name: '{:.5f}'.format(lv.value.item()) for lv in losses.values()}
        super().print_status(status_dict, **kwargs)

    def log_status(
        self,
        losses: Dict[str, LossValue],
        cur_lr: float
    ) -> None:
        for lv in losses.values():
            self.writer.add_scalar('train/{}'.format(lv.log_name), lv.value.item(), self.iter_ctr)

        self.writer.add_scalar('misc/iter_time', self.time1 - self.time0, self.iter_ctr)
        self.writer.add_scalar('misc/cur_lr', cur_lr, self.iter_ctr)

    @torch.no_grad()
    def test_networks(self):
        img_dir = self.log_dir / 'epoch_{:0{width}d}'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        img_dir.mkdir()

        eval_losses: List[Dict[str, LossValue]] = []

        for i, (img, pose) in tqdm(enumerate(self.test_loader), total=len(self.test_set)):
            frame_id = self.test_set.frame_str_ids[i]
            img, pose = img.to(self.device), pose.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                with self.ema.average_parameters():
                    output = self.renderer.render(pose, img, training=False)

            _, h, w = img.shape
            rgb_output = einops.rearrange(output['rgb_map'], '(h w) c -> c h w', h=h, w=w)
            save_path = img_dir / 'frame_{}.png'.format(frame_id)
            torchvision.utils.save_image(rgb_output, save_path)

            eval_losses.append(self.calc_loss(output))

        avg_loss = copy(eval_losses[0])
        avg_loss['mse'].value = torch.mean(torch.stack([el['mse'].value for el in eval_losses]))
        avg_loss['psnr'].value = utils.compute_psnr(avg_loss['mse'].value)
        self.print_status(avg_loss, phase='TEST')

    def run_iter(self):
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)

        self.renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            num_rays = self.train_cfg.num_rays_per_batch
            output = self.renderer.render(pose, img, num_rays=num_rays, training=True)

            if self.train_cfg.sparsity_lambda > 0.:
                bbox = self.train_set.bbox
                sparsity_pts = torch.rand((self.train_cfg.sparsity_samples, 3), device=self.device)
                sparsity_pts = sparsity_pts * bbox.size() + bbox.min_pt
                output['sparsity'] = self.model(sparsity_pts)

            losses = self.calc_loss(output)

        self.optim.zero_grad()

        back_loss = losses['total' if 'total' in losses.keys() else 'mse'].value
        self.scaler.scale(back_loss).backward()
        self.scaler.step(self.optim)
        old_scale = self.scaler.get_scale()
        self.scaler.update()
        if old_scale <= self.scaler.get_scale():
            self.scheduler.step()
        self.ema.update()

        # Update counter after backprop
        self.iter_ctr += 1
        self.time1 = time.time()

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(losses)
        if self.check_interval(self.train_cfg.intervals.test):
            self.test_networks()
        if self.check_interval(self.train_cfg.intervals.log):
            self.log_status(losses, self.scheduler.get_last_lr()[0])
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
