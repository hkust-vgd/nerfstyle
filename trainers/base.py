from pathlib import Path
import pickle
import sys
import time
from typing import Callable, Dict, List, Optional

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from common import DatasetSplit, LossValue
from config import BaseConfig, DatasetConfig, NetworkConfig, RendererConfig, TrainConfig
from data import get_dataset
from nerf_lib import nerf_lib
from networks.tcnn_nerf import TCNerf, StyleTCNerf
from renderer import Renderer
import utils


class Trainer:
    SAVE_KEYS = [
        'version', 'name', 'model', 'renderer',
        'dataset_cfg', 'train_cfg', 'net_cfg', 'render_cfg'
    ]

    def __init__(
        self,
        cfg: BaseConfig,
        nargs: List[str],
        trainer: Optional['Trainer'] = None
    ) -> None:
        """
        Default volumetric rendering trainer.

        Args:
            cfg (BaseConfig): Command line arguments.
            nargs (List[str]): Overwritten config parameters.
            model (Optional[TCNerf], optional): Existing model. Defaults to None.
            renderer (Optional[Renderer], optional): Existing renderer. Defaults to None.
        """
        self.logger = utils.create_logger(__name__)
        if (cfg.data_cfg is None) and (cfg.ckpt is None):
            self.logger.error('Data config must be provided if training from scratch')

        self.iter_ctr = 0
        self.time0 = 0
        self.time1 = 0

        self.name = cfg.name
        self.version = utils.get_git_sha()
        self.log_dir: Path = Path(cfg.run_dir) / self.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if next(self.log_dir.iterdir(), None) is not None:
            # log dir is not empty
            if (cfg.ckpt is not None) and (Path(cfg.ckpt).parent == self.log_dir):
                # loading from existing checkpoint in log dir
                proceed = utils.prompt_bool(
                    'Checkpoints beyond "{}" will be overwritten. Proceed?'.format(cfg.ckpt))
            else:
                proceed = utils.prompt_bool('Log directory not empty. Clean directory?')
                if proceed:
                    utils.rmtree(self.log_dir)
                    self.log_dir.mkdir()

            if not proceed:
                sys.exit(1)

        # Parse arguments
        if trainer is None:
            self.dataset_cfg, nargs = DatasetConfig.load_nargs(cfg.data_cfg, nargs=nargs)
        else:
            self.dataset_cfg = trainer.dataset_cfg

            # TODO: Overwrite with new parameters
            if cfg.data_cfg is not None:
                self.logger.warning('Overwriting functionality not yet implemented')

        train_cfg_path = 'cfgs/training/style.yaml' if cfg.style_image is not None else None
        render_cfg_path = Path('cfgs/renderer/{}.yaml'.format(self.dataset_cfg.type.lower()))
        if not render_cfg_path.exists():
            render_cfg_path = None

        self.train_cfg, nargs = TrainConfig.load_nargs(train_cfg_path, nargs=nargs)
        self.net_cfg, nargs = NetworkConfig.load_nargs(nargs=nargs)
        self.render_cfg, nargs = RendererConfig.load_nargs(render_cfg_path, nargs=nargs)

        if len(nargs) > 0:
            self.logger.error('Unrecognized arguments: ' + ' '.join(nargs))

        self.device = torch.device('cuda:0')
        nerf_lib.device = self.device

        self.writer = None
        if self.train_cfg.intervals.log > 0:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        np.random.seed(self.train_cfg.rng_seed)
        torch.manual_seed(self.train_cfg.rng_seed)
        torch.cuda.manual_seed(self.train_cfg.rng_seed)

        # Initialize dataset
        self.train_set = get_dataset(self.dataset_cfg, split=DatasetSplit.TRAIN)
        self.train_set.bbox = self.train_set.bbox.to(self.device)
        self.train_loader = utils.cycle(DataLoader(self.train_set, batch_size=None, shuffle=True))
        self.logger.info('Loaded ' + str(self.train_set))

        self.test_set = get_dataset(self.dataset_cfg, split=DatasetSplit.VAL,
                                    max_count=self.train_cfg.max_eval_count)
        self.test_loader = DataLoader(self.test_set, batch_size=None, shuffle=False)
        self.logger.info('Loaded ' + str(self.test_set))

        # Initialize model and renderer
        if trainer is None:
            enc_dtype = None if self.train_cfg.enable_amp else torch.float32
            self.model = StyleTCNerf(self.net_cfg, self.train_set.bbox, enc_dtype)

            self.model = self.model.to(self.device)
            self.logger.info('Created model ' + str(type(self.model)))

            self.renderer = Renderer(
                self.model, self.render_cfg, self.train_set.intr, self.dataset_cfg.bound,
                precrop_frac=self.train_cfg.precrop_fraction
            )
        else:
            self.model = trainer.model.to(self.device)
            self.renderer = trainer.renderer.to(self.device)

        # Initialize optimizer and miscellaneous components
        self._reset_optim(['embedder', 'net'])
        # self._reset_optim(['x_density_embedder', 'x_color_embedder', 'net'])

    def _reset_optim(self, keywords=None):
        all_keys = [n for n, _ in self.model.named_parameters()]
        train_keys, train_params = [], []
        for n, p in self.model.named_parameters():
            if keywords is None or any([(kw in n) for kw in keywords]):
                train_keys.append(n)
                train_params.append(p)
        if len(train_keys) == 0:
            self.logger.error('Keywords {} not found in keys {}'.format(
                keywords, all_keys))

        param_count = np.sum(np.prod(p.size()) for p in train_params)
        msg = 'Optimizing {:d} parameters from '.format(param_count)
        msg += ('all components' if keywords is None else 'components ' + str(keywords))
        self.logger.info(msg)

        self.optim = torch.optim.Adam(
            params=train_params,
            lr=self.train_cfg.initial_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-15
        )

        def lr_lambda(_): return 1.
        if self.train_cfg.learning_rate_decay > 0:
            def lr_lambda(iter: int): return 0.1 ** (iter / self.train_cfg.learning_rate_decay)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.enable_amp)
        self.ema = utils.EMA(train_params, decay=self.train_cfg.ema_decay)

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

    def save_ckpt(self) -> None:
        """
        Save to new checkpoint.
        """
        ckpt_fn = 'iter_{:0{width}d}.pkl'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        ckpt_path = self.log_dir / ckpt_fn

        with open(ckpt_path, 'wb') as f:
            pickle.dump(self, f)

        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    def calc_loss(
        self,
        output: Dict[str, torch.Tensor],
        mse_only: bool = False
    ) -> Dict[str, LossValue]:
        """
        Compute losses from render output.

        Args:
            output (Dict[str, torch.Tensor]): Render result from `Renderer.render`.

        Returns:
            Dict[str, LossValue]: Dict of all loss values.
        """
        assert output['target'] is not None
        rendered = output['rgb_map']
        target = output['target']

        mse_loss = torch.mean((rendered - target) ** 2)
        losses = {
            'mse': LossValue('MSE', 'mse_loss', mse_loss),
            'psnr': LossValue('PSNR', 'psnr', utils.compute_psnr(mse_loss))
        }

        if mse_only:
            return losses

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
            net_params = [p for n, p in self.model.named_parameters() if 'rgb_net' in n]
            norm_sum = torch.sum(torch.stack([p.norm(2) for p in net_params]))
            weight_reg_loss = norm_sum * weight_reg_lambda
            losses['weight_reg'] = LossValue('Weight Reg.', 'weight_reg_loss', weight_reg_loss)

        if sparsity_loss + weight_reg_loss > 0:
            total_loss = mse_loss + sparsity_loss + weight_reg_loss
            losses['total'] = LossValue('Total', 'total_loss', total_loss)

        return losses

    def print_status(
        self,
        losses: Dict[str, LossValue],
        phase: str = 'TRAIN',
        out_fn: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Print losses and metrics.

        Args:
            losses (Dict[str, LossValue]): Dict of losses / metrics.
            phase (str, optional): Short description text. Defaults to 'TRAIN'.
            out_fn (Optional[Callable[[str], None]], optional): Function for printing. \
                If None, uses `self.logger.info` for output. Defaults to None.
        """
        status_dict = {lv.print_name: '{:.5f}'.format(lv.value.item()) for lv in losses.values()}
        if out_fn is None:
            out_fn = self.logger.info
        log_items = [k + ': ' + str(v) for k, v in status_dict.items()]
        log_str = '[{}] Iter: {:d}, {}'.format(phase, self.iter_ctr, ', '.join(log_items))
        out_fn(log_str)

    def log_status(
        self,
        losses: Dict[str, LossValue]
    ) -> None:
        """
        Log losses, metrics and other statistics to TensorBoard.

        Args:
            losses (Dict[str, LossValue]): Dict of losses / metrics.
        """
        for lv in losses.values():
            self.writer.add_scalar('train/{}'.format(lv.log_name), lv.value.item(), self.iter_ctr)

        self.writer.add_scalar('misc/iter_time', self.time1 - self.time0, self.iter_ctr)
        self.writer.add_scalar('misc/cur_lr', self.scheduler.get_last_lr()[0], self.iter_ctr)

    @torch.no_grad()
    def test_networks(self):
        """
        Render and evaluate images from test set.
        """
        img_dir = self.log_dir / 'epoch_{:0{width}d}'.format(
            self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
        img_dir.mkdir()

        # eval_losses: List[Dict[str, LossValue]] = []

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

            # eval_losses.append(self.calc_loss(output))

        # avg_loss = copy(eval_losses[0])
        # avg_loss['mse'].value = torch.mean(torch.stack([el['mse'].value for el in eval_losses]))
        # avg_loss['psnr'].value = utils.compute_psnr(avg_loss['mse'].value)
        # self.print_status(avg_loss, phase='TEST')

    def _check_interval(self, interval, after=0, final=False):
        if interval <= 0:
            return False

        is_final = (self.iter_ctr == self.train_cfg.num_iterations) and final
        flag = (((self.iter_ctr % interval == 0) or is_final) and (self.iter_ctr > after))
        return flag

    def run_iter(self):
        """
        Run one training iteration.
        """
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
                sparsity_pts = sparsity_pts * bbox.size + bbox.min_pt
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
        if self._check_interval(self.train_cfg.intervals.print):
            self.print_status(losses)
        if self._check_interval(self.train_cfg.intervals.test):
            self.test_networks()
        if self._check_interval(self.train_cfg.intervals.log):
            self.log_status(losses)
        if self._check_interval(self.train_cfg.intervals.ckpt, final=True):
            self.save_ckpt()

    def run(self):
        """
        Start training loop.
        """
        if self.train_cfg.test_before_train:
            self.test_networks()
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()

    def close(self):
        """
        Cleanup function at end of training, or when training is interrupted.
        """
        self.logger.info('Closed')
