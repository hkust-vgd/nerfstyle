from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional
import time

import einops
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import Box2D, DatasetSplit, LossValue
from config import BaseConfig, ConfigValue
from data.style_dataset import WikiartDataset, SingleImage
from loss import AdaINStyleLoss, GramStyleLoss, MattingLaplacian
from networks.fx import VGG16FeatureExtractor
from renderer import StyleRenderer
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

        self.tmp_dir = Path('/tmp/nerfRenderer')
        self.tmp_dir.mkdir(exist_ok=True)

        # Intialize losses and style image
        fx_keys = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.fx = VGG16FeatureExtractor(fx_keys).to(self.device)
        self.style_loss = AdaINStyleLoss(fx_keys)
        # self.style_loss = GramStyleLoss(fx_keys)
        self.photo_loss = MattingLaplacian(device=self.device)

        self.num_styles = 64

        if cfg.style_image is ConfigValue.EmptyPassed:
            root_path = 'datasets/wikiart'
            self.style_train_set = WikiartDataset(
                root_path, DatasetSplit.TRAIN, max_images=self.num_styles)
            self.style_train_loader = utils.cycle(DataLoader(
                self.style_train_set, batch_size=1, shuffle=True))
        else:
            self.style_train_set = SingleImage(cfg.style_image, size=(256, 256))
            self.style_train_loader = utils.cycle(DataLoader(self.style_train_set, batch_size=1))
        self.style_test_loader = DataLoader(self.style_train_set, batch_size=1, shuffle=False)
        self.logger.info('Loaded ' + str(self.style_train_set))

        self.model.cuda()
        self._reset_optim(['x_style_embedder', 'color1_net', 'color2_net', 'style_net'])
        self.renderer = StyleRenderer(self.model, self.renderer, self.render_cfg)
        # self.model.x_style_embedder.initialize(
        #     self.model.x_color_embedder.embeddings, num_styles=1)
        # self.model.x_style_embedder.embeddings = self.model.x_color_embedder.embeddings

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
        rgb_resized_feats = self.fx(F.interpolate(rgb_map_chw.unsqueeze(0), (256, 256)))
        target_feats = self.fx(target_chw)
        style_feats = self.fx(style_img)

        content_loss = F.mse_loss(rgb_feats['relu3_1'], target_feats['relu3_1'])
        style_loss = self.style_loss(rgb_resized_feats, style_feats)
        photo_loss = self.photo_loss(target_chw, rgb_map_chw)

        content_loss *= self.train_cfg.content_lambda
        style_loss *= self.train_cfg.style_lambda
        photo_loss *= self.train_cfg.photo_lambda

        losses = {
            'content': LossValue('Content', 'content_loss', content_loss),
            'style': LossValue('Style', 'style_loss', style_loss),
            'photo': LossValue('Photo', 'photo_loss', photo_loss),
        }
        # total_loss = content_loss + style_loss + photo_loss
        # total_loss = content_loss + photo_loss
        total_loss = content_loss + style_loss

        # if self.iter_ctr < self.mse_end:
        #     mse_loss = F.mse_loss(rgb_map_chw, target_chw)
        #     losses['mse'] = LossValue('MSE', 'mse_loss', mse_loss)
        #     total_loss = mse_loss

        losses['total'] = LossValue('Total', 'total_loss', total_loss)
        return losses

    @torch.no_grad()
    def test_networks(self):
        """
        Render and evaluate images from test set.
        """
        image_dir = self.log_dir / 'epoch_{:0{width}d}'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        image_dir.mkdir()

        eval_once = (self.iter_ctr == 0)

        for (style_image, style_id) in tqdm(self.style_test_loader, disable=eval_once):
            style_image = style_image.to(self.device)

            frames = []
            for _, pose in self.test_loader:
                pose = pose.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                    with self.ema.average_parameters():
                        output = self.renderer.render(
                            pose, (style_image, style_id), None, training=False)

                h, w = self.test_set.intr.h, self.test_set.intr.w
                rgb_output = einops.rearrange(output['rgb_map'], '(h w) c -> c h w', h=h, w=w)
                collage = utils.collage_h(rgb_output, style_image.squeeze(0))
                frame = (collage.cpu().numpy() * 255).astype(np.uint8)
                frames.append(einops.rearrange(frame, 'c h w -> h w c'))
                # save_path = tmp_dir / '{}_style{:d}.png'.format(frame_id, style_id.item())
                # torchvision.utils.save_image(collage, save_path)

                out_path = image_dir / 'style{:d}.gif'.format(style_id.item())
                imageio.mimsave(out_path, frames, fps=3.75)

            if eval_once:
                break

    def run_iter(self):
        """
        Run one training iteration.
        """
        self.time0 = time.time()
        image, pose = next(self.train_loader)
        style_images, style_ids = next(self.style_train_loader)
        image, pose = image.to(self.device), pose.to(self.device)
        style_images = style_images.to(self.device)
        W, H = self.train_set.intr.size()

        self.renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        self.optim.zero_grad()

        # First pass: render all pixels without gradients
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            with torch.no_grad():
                output = self.renderer.render(pose, (style_images, style_ids), image)

        # Compute d_loss / d_pixels and cache
        output['rgb_map'].requires_grad = True
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            losses = self.calc_loss(output, style_images)
            back_loss = losses['total' if 'total' in losses.keys() else 'mse'].value
        self.scaler.scale(back_loss).backward()
        grad_map = einops.rearrange(output['rgb_map'].grad, '(h w) c -> h w c', h=H, w=W)

        # Second pass: render pixels patch-wise with gradients
        ps = self.train_cfg.defer_patch_size
        for x, y in product(range(0, W, ps), range(0, H, ps)):
            patch = Box2D(x=x, y=y, w=ps, h=ps)
            with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                patch_output = self.renderer.render(
                    pose, (style_images, style_ids), image, patch=patch)

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
