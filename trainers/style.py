from functools import partial
from itertools import product
from typing import Dict, List
import time

import einops
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import Box2D, LossValue
from config import BaseConfig, ConfigValue
from data.style_dataset import SingleImage
from loss import get_style_loss, MattingLaplacian
from networks.fx import VGG16FeatureExtractor
from trainers.base import Trainer
import utils


class StyleTrainer(Trainer):
    OPTIM_KEYS = ['x_color_embedder']

    def __init__(
        self,
        cfg: BaseConfig,
        nargs: List[str]
    ) -> None:
        """
        Volumetric rendering trainer for style transfer.
        """
        assert cfg.style_image is not None
        super().__init__(cfg, nargs, load_model_only=True)

        # self.tmp_dir = Path('/tmp/nerfRenderer')
        # self.tmp_dir.mkdir(exist_ok=True)

        # Intialize losses and style image
        # fx_keys = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        fx_keys = ['relu3']
        self.content_feat = 'relu3'
        self.fx = VGG16FeatureExtractor(fx_keys).to(self.device)
        # self.style_loss = get_style_loss('AdaINStyleLoss', fx_keys)
        # self.style_loss = get_style_loss('NNFMStyleLoss', fx_keys)
        matching = None
        if self.train_cfg.style_matching is not None:
            matching = [int(c) for c in self.train_cfg.style_matching.split(',')]
        self.style_loss = get_style_loss(
            'SemanticStyleLoss', fx_keys,
            clusters_path=self.train_cfg.style_seg_path, matching=matching)
        self.photo_loss = MattingLaplacian(device=self.device)

        if cfg.style_image is ConfigValue.EmptyPassed:
            raise NotImplementedError
            # root_path = 'datasets/wikiart'
            # self.style_train_set = WikiartDataset(
            #     root_path, DatasetSplit.TRAIN, max_images=self.num_styles)
            # self.style_train_loader = utils.cycle(DataLoader(
            #     self.style_train_set, batch_size=1, shuffle=True))
        else:
            longer_edge = max(self.train_set.intr.w, self.train_set.intr.h)
            self.style_train_set = SingleImage(cfg.style_image, longer_edge)
            self.style_train_loader = utils.cycle(DataLoader(self.style_train_set, batch_size=1))
        self.style_test_loader = DataLoader(self.style_train_set, batch_size=1, shuffle=False)
        self.logger.info('Loaded ' + str(self.style_train_set))

        style_image = next(self.style_train_loader).cuda()
        style_feats = self.fx(style_image)
        self.style_loss.init_feats(style_feats, num_classes=self.train_set.num_classes)

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor],
        style_img: torch.Tensor
    ) -> Dict[str, LossValue]:
        assert output['target'] is not None
        if output['target'].shape[-1] == 4:
            target, classes = output['target'][:, :3], output['target'][:, 3].to(torch.long)
        else:
            raise NotImplementedError
            # target, classes = output['target'], None

        W, H = self.train_set.intr.size()
        nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=H, w=W)
        rgb_map_chw = nc2chw(output['rgb_map'])
        target_chw = nc2chw(target)
        preds = torch.argmax(nc2chw(output['classes']), dim=0)

        rgb_feats = self.fx(rgb_map_chw)
        target_feats = self.fx(target_chw)
        style_feats = self.fx(style_img)

        content_loss = F.mse_loss(rgb_feats[self.content_feat], target_feats[self.content_feat])
        style_loss = self.style_loss(rgb_feats, style_feats, preds, self.iter_ctr)
        # photo_loss = self.photo_loss(target_chw, rgb_map_chw)

        content_loss *= self.train_cfg.content_lambda
        style_loss *= self.train_cfg.style_lambda
        # photo_loss *= self.train_cfg.photo_lambda

        losses = {
            'content': LossValue('Content', 'content_loss', content_loss),
            'style': LossValue('Style', 'style_loss', style_loss),
            # 'photo': LossValue('Photo', 'photo_loss', photo_loss),
        }
        total_loss = content_loss + style_loss

        # if self.iter_ctr < self.mse_end:
        #     mse_loss = F.mse_loss(rgb_map_chw, target_chw)
        #     losses = {'mse': LossValue('MSE', 'mse_loss', mse_loss)}
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

        style_image = next(self.style_train_loader)
        style_image = style_image.to(self.device)

        frames = []
        for i, (_, pose) in tqdm(enumerate(self.test_loader), total=len(self.test_set)):
            pose = pose.to(self.device)
            frame_id = self.test_set.fns[i]

            with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                # with self.ema.average_parameters():
                output = self.renderer.render(pose, None, training=False)

            h, w = self.test_set.intr.h, self.test_set.intr.w
            rgb_output = einops.rearrange(output['rgb_map'], '(h w) c -> c h w', h=h, w=w)
            collage = utils.collage_h(rgb_output, style_image.squeeze(0))
            frame = (collage.cpu().numpy() * 255).astype(np.uint8)
            frames.append(einops.rearrange(frame, 'c h w -> h w c'))
            # save_path = tmp_dir / '{}_style{:d}.png'.format(frame_id, style_id.item())
            # torchvision.utils.save_image(collage, save_path)
            save_path = image_dir / '{}.png'.format(frame_id)
            torchvision.utils.save_image(rgb_output, save_path)

        out_path = image_dir / 'video.gif'
        imageio.mimsave(out_path, frames, fps=3.75)

        # Temp save features
        # rgb_feats = self.fx(rgb_output)[self.content_feat].squeeze(0)
        # with open(image_dir / '{}_feats.npy'.format(frame_id), 'wb') as f:
        #     np.save(f, rgb_feats.cpu().numpy())
        # if self.iter_ctr == 0:
        #     style_feats = self.fx(style_image)[self.content_feat].squeeze(0)
        #     with open(self.log_dir / 'style_feats.npy', 'wb') as f:
        #         np.save(f, style_feats.cpu().numpy())

    def run_iter(self):
        """
        Run one training iteration.
        """
        self.time0 = time.time()
        image, pose = next(self.train_loader)
        style_image = next(self.style_train_loader)
        image, pose = image.to(self.device), pose.to(self.device)
        style_image = style_image.to(self.device)
        W, H = self.train_set.intr.size()

        self.renderer.use_precrop = (self.iter_ctr < self.train_cfg.precrop_iterations)
        self.optim.zero_grad()

        # First pass: render all pixels without gradients
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            with torch.no_grad():
                output = self.renderer.render(pose, image, training=True)

        # Compute d_loss / d_pixels and cache
        output['rgb_map'].requires_grad = True
        with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
            losses = self.calc_loss(output, style_image)
            back_loss = losses['total' if 'total' in losses.keys() else 'mse'].value
        self.scaler.scale(back_loss).backward()
        grad_map = einops.rearrange(output['rgb_map'].grad, '(h w) c -> h w c', h=H, w=W)

        # Second pass: render pixels patch-wise with gradients
        ps = self.train_cfg.defer_patch_size
        for x, y in product(range(0, W, ps), range(0, H, ps)):
            patch = Box2D(x=x, y=y, w=ps, h=ps)
            with torch.cuda.amp.autocast(enabled=self.train_cfg.enable_amp):
                patch_output = self.renderer.render(pose, image, patch=patch, training=True)

            # Backprop cached grads to network
            patch_grad = grad_map[patch.hrange(), patch.wrange()].reshape(-1, 3)
            patch_output['rgb_map'].backward(patch_grad)

        self.scaler.step(self.optim)
        old_scale = self.scaler.get_scale()
        self.scaler.update()
        if old_scale <= self.scaler.get_scale():
            self.scheduler.step()
        # self.ema.update()

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
