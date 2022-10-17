from functools import partial
from typing import Dict, List, Optional

import einops
import torch
import torch.nn.functional as F

from common import LossValue
from config import BaseConfig
from loss import FeatureExtractor, MattingLaplacian, StyleLoss
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
        self.style_dims = (256, 256)
        super().__init__(cfg, nargs, trainer)

        # Intialize losses and style image
        self.fe = FeatureExtractor().to(self.device)
        style_image_np = utils.parse_rgb(cfg.style_image, size=self.style_dims)
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
