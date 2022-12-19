from functools import partial
from itertools import product
from typing import Dict, List, Optional
import time

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import Box2D, DatasetSplit, LossValue
from config import BaseConfig
from data.style_dataset import WikiartDataset
from loss import AdaINStyleLoss, MattingLaplacian
from networks.style_nerf import StyleNerf
from networks.fx import VGG16FeatureExtractor
from trainers.base import Trainer
import utils


class StyleTrainer(Trainer):
    def __init__(
        self,
        cfg: BaseConfig,
        nargs: List[str],
        trainer: Optional[Trainer] = None
    ) -> None:
        """
        Volumetric rendering trainer for style transfer.
        """
        assert cfg.style_image is not None
        super().__init__(cfg, nargs, trainer)

        # Intialize losses and style image
        fx_keys = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.fx = VGG16FeatureExtractor(fx_keys).to(self.device)
        self.style_loss = AdaINStyleLoss(fx_keys)
        self.photo_loss = MattingLaplacian(device=self.device)

        root_path = 'datasets/wikiart'
        test_id = 12345
        self.style_train_set = WikiartDataset(root_path, DatasetSplit.TRAIN, fix_id=test_id)
        self.style_train_loader = utils.cycle(DataLoader(
            self.style_train_set, batch_size=1, shuffle=True))

        # New model
        self.model = StyleNerf(self.model)
        self.model.cuda()
        self.renderer.model = self.model
        self._reset_optim(['s_embedder'])

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor],
        style_img: torch.Tensor
    ) -> Dict[str, LossValue]:
        assert output['target'] is not None

        W, H = self.train_set.intr.size()
        nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=H, w=W)
        rgb_map_chw = nc2chw(output['rgb_map'])
        target_chw = nc2chw(output['target'])

        rgb_feats = self.fx(rgb_map_chw)
        target_feats = self.fx(target_chw)
        style_feats = self.fx(style_img)

        content_loss = F.mse_loss(rgb_feats['relu3_1'], target_feats['relu3_1'])
        style_loss = self.style_loss(rgb_feats, style_feats)
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

    def run_iter(self):
        """
        Run one training iteration.
        """
        self.time0 = time.time()
        img, pose = next(self.train_loader)
        img, pose = img.to(self.device), pose.to(self.device)
        style_img = next(self.style_train_loader).to(self.device)
        W, H = self.train_set.intr.size()

        self.renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        self.optim.zero_grad()

        # First pass: render all pixels without gradients
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            with torch.no_grad():
                output = self.renderer.render(pose, img, style_img)

        # Compute d_loss / d_pixels and cache
        output['rgb_map'].requires_grad = True
        losses = self.calc_loss(output, style_img)
        back_loss = losses['total' if 'total' in losses.keys() else 'mse'].value
        self.scaler.scale(back_loss).backward()
        grad_map = einops.rearrange(output['rgb_map'].grad, '(h w) c -> h w c', h=H, w=W)

        # Second pass: render pixels patch-wise with gradients
        ps = self.train_cfg.defer_patch_size
        for x, y in product(range(0, W, ps), range(0, H, ps)):
            patch = Box2D(x=x, y=y, w=ps, h=ps)
            with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                patch_output = self.renderer.render(pose, img, style_img, patch=patch)

            # Backprop cached grads to network
            patch_grad = grad_map[patch.hrange(), patch.wrange()].reshape(-1, 3)
            patch_output['rgb_map'].backward(patch_grad)

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
        if self._check_interval(self.train_cfg.intervals.print):
            self.print_status(losses)
        if self._check_interval(self.train_cfg.intervals.test):
            self.test_networks()
        if self._check_interval(self.train_cfg.intervals.log):
            self.log_status(losses)
        if self._check_interval(self.train_cfg.intervals.ckpt, final=True):
            self.save_ckpt()
