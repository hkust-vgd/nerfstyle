from argparse import Namespace
from functools import partial
from typing import Dict, List
import pickle

import einops
import torch
import torch.nn.functional as F

from common import LossValue
from loss import FeatureExtractor, MattingLaplacian, StyleLoss
from trainers.base import Trainer
import utils


class StyleTrainer(Trainer):
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

    @classmethod
    def load_ckpt(
        cls,
        ckpt_path: str
    ) -> 'StyleTrainer':
        with open(ckpt_path, 'rb') as f:
            trainer: Trainer = pickle.load(f)
        
        if isinstance(trainer, StyleTrainer):
            # TODO: continue training
            raise NotImplementedError('not implemented yet')
        elif isinstance(trainer, Trainer):
            raise NotImplementedError('sfsg')
        else:
            raise ValueError('Unpickled object is not trainer!')

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
