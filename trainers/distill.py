from collections import deque
from functools import partial
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked

from .base import Trainer
from networks.nerf import Nerf, SingleNerf
from networks.multi_nerf import StaticMultiNerf
import utils


patch_typeguard()


class Node:
    def __init__(
        self,
        idx: str,
        min_pt: List[float],
        max_pt: List[float],
        log_dir: Path
    ) -> None:
        self.idx = idx
        self.min_pt, self.max_pt = np.array(min_pt), np.array(max_pt)
        self.net: Optional[Nerf] = None
        # self.leq_child: Optional[Node] = None
        # self.gt_child: Optional[Node] = None
        self.log_path = log_dir / 'node_{}.log'.format(idx)

    def log_init(self):
        with open(self.log_path, 'w') as f:
            f.write('Node "{}"\n'.format(self.idx))
            f.write('Min coords: ' + str(self.min_pt) + '\n')
            f.write('Max coords: ' + str(self.max_pt) + '\n')

    def log_append(self, msg: str):
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')

    def export_ckpt(self):
        ckpt = {
            'idx': self.idx,
            'min_pt': self.min_pt,
            'max_pt': self.max_pt,
            'log_path': str(self.log_path),
            'started': (self.net is not None)
        }
        if ckpt['started']:
            ckpt['model'] = self.net.state_dict()
        return ckpt


class NodeQueue(deque):
    def __init__(
        self,
        iterable: Iterable[Node] = ()
    ) -> None:
        super().__init__(iterable)

    def batch_popleft(self, count: int) -> List[Node]:
        batch = [self.popleft() for _ in range(min(len(self), count))]
        return batch

    def batch_push(self, nodes: List[Node]) -> None:
        for node in nodes:
            self.append(node)


class DistillDataset(Dataset):
    @typechecked
    def __init__(
        self,
        pts: TensorType['num_nets', 'dataset_size', 3],
        dirs: TensorType['num_nets', 'dataset_size', 3],
        colors: TensorType['num_nets', 'dataset_size', 3],
        alphas: TensorType['num_nets', 'dataset_size', 1]
    ) -> None:
        self._pts = pts
        self._dirs = dirs
        self._colors = colors
        self._alphas = alphas

        self._num_nets, self._size = pts.shape[:2]

    def __getitem__(self, index):
        item_dict = {
            'pts': self._pts[:, index],
            'dirs': self._dirs[:, index],
            'colors_gt': self._colors[:, index],
            'alphas_gt': self._alphas[:, index]
        }
        return item_dict

    def get_gt(self):
        return self._colors, self._alphas

    def __len__(self) -> int:
        return self._size

    def __str__(self):
        desc = 'distillation dataset over {:d} networks with {:d} entries'
        return desc.format(self._num_nets, self._size)


class DistillTrainer(Trainer):
    losses = {
        'mse': partial(F.mse_loss, reduction='none'),
        'mae': partial(F.l1_loss, reduction='none'),
        'mape': lambda out, tar: F.l1_loss(
            out, tar, reduction='none') / (torch.abs(tar) + 0.1)
    }

    def __init__(self, args, nargs):
        super().__init__(__name__, args, nargs)

        self.test_log_dir = self.log_dir / 'logs'
        self.test_log_dir.mkdir(parents=True, exist_ok=True)
        self.root_nodes = self._generate_nodes()
        self.nodes_queue = NodeQueue(self.root_nodes)
        self.trained_nodes = NodeQueue()

        # Load teacher model
        if args.teacher_ckpt_path is None:
            self.logger.error('Please provide path to teacher model')

        ckpt = utils.load_ckpt_path(args.teacher_ckpt_path, self.logger)
        self.teacher = SingleNerf.create_nerf(self.net_cfg).to(self.device)
        self.teacher.load_state_dict(ckpt, strict=False)
        self.logger.info('Loaded teacher model ' + str(self.teacher))

        self.losses = ['mse', 'mae', 'mape']
        self.metrics = self.losses + ['se_q99']
        self.metric_items = ['all', 'color', 'alpha']
        self.round_ctr = 0

        # Entities initialized when training new batch of nodes
        self.num_nets = 0
        self.cur_nodes = []
        self.model, self.optim = None, None
        self.train_set, self.test_set = None, None
        self.train_loader = None
        self.best_losses = {}

    def _generate_nodes(self) -> List[Node]:
        bbox_path = self.dataset_cfg.root_path / 'bbox.txt'
        min_pt, max_pt = utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)

        net_res = self.dataset_cfg.net_res
        log_dir = self.test_log_dir

        intervals = [np.linspace(start, end, num+1) for start, end, num in
                     zip(min_pt, max_pt, net_res)]

        node_idx_fmt = '{{:0{}d}}_{{:0{}d}}_{{:0{}d}}'.format(
            *[len(str(dim)) for dim in net_res])

        nodes = []
        for pt in itertools.product(*[range(ax) for ax in net_res]):
            node_id = node_idx_fmt.format(*pt)
            node_min_pt = [intervals[i][x] for i, x in enumerate(pt)]
            node_max_pt = [intervals[i][x+1] for i, x in enumerate(pt)]
            nodes.append(Node(node_id, node_min_pt, node_max_pt, log_dir))

        return nodes

    def _build_dataset(
        self,
        node_batch: List[Node],
        samples_per_net: int
    ) -> DistillDataset:
        # Randomly collect points and directions
        pts = torch.empty((self.num_nets * samples_per_net, 3))
        pts_norm = torch.empty((self.num_nets * samples_per_net, 3))
        dirs = torch.empty((self.num_nets * samples_per_net, 3))

        tsize = utils.compute_tensor_size(pts, pts_norm, dirs, unit='GB')
        self.logger.info('Creating {:d} input samples ({})...'.format(
            samples_per_net, tsize))

        def gen_one_node(node):
            node = node[0]
            node_pts, node_pts_norm = utils.get_random_pts(
                samples_per_net, node.min_pt, node.max_pt)
            node_dirs = utils.get_random_dirs(samples_per_net)
            return node_pts, node_pts_norm, node_dirs

        utils.batch_exec(gen_one_node, pts, pts_norm, dirs,
                         bsize=1, progress=True)(node_batch)

        # Compute embedded points / dirs
        # Evaluate ground truth from teacher model
        colors = torch.empty((self.num_nets * samples_per_net, 3))
        alphas = torch.empty((self.num_nets * samples_per_net, 1))

        tsize = utils.compute_tensor_size(colors, alphas, unit='GB')
        self.logger.info('Computing inputs and targets ({})...'.format(tsize))

        def gen_data_batch(pts_batch, dirs_batch):
            pts_batch = pts_batch.to(self.device)
            dirs_batch = dirs_batch.to(self.device)
            color, density = self.teacher(pts_batch, dirs_batch)
            alpha = utils.density2alpha(
                density, self.train_cfg.distill.alpha_dist)
            return color.cpu(), alpha.cpu()

        with torch.no_grad():
            utils.batch_exec(gen_data_batch, colors, alphas,
                             bsize=self.train_cfg.distill.init_data_bsize,
                             progress=True)(pts, dirs)
        pts_norm, dirs, colors, alphas = utils.reshape(
            pts_norm, dirs, colors, alphas,
            shape=(len(node_batch), samples_per_net, -1))

        dataset = DistillDataset(pts_norm, dirs, colors, alphas)
        return dataset

    def _init_round(self) -> None:
        """ Initializes a new round of training. """
        self.round_ctr += 1
        self.iter_ctr = 0
        self.logger.info('Starting round #' + str(self.round_ctr))

        self.cur_nodes = self.nodes_queue.batch_popleft(
            self.train_cfg.distill.nets_bsize)
        for node in self.cur_nodes:
            node.log_init()
        self.num_nets = len(self.cur_nodes)

        self.model = StaticMultiNerf.create_nerf(
            self.num_nets, self.net_cfg).to(self.device)
        self.logger.info('Created student model ' + str(self.model))
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.initial_learning_rate)

        self.train_set = self._build_dataset(
            self.cur_nodes, self.train_cfg.distill.train_samples_pnet)
        self.logger.info('Loaded ' + str(self.train_set))
        self.test_set = self._build_dataset(
            self.cur_nodes, self.train_cfg.distill.test_samples_pnet)
        self.logger.info('Loaded ' + str(self.test_set))

        # Collate dataset items by 2nd (batch) dimension
        def collate_fn(elems):
            keys = elems[0].keys()
            collate_dict = {k: torch.stack(
                [e[k] for e in elems], dim=1).to(self.device)
                for k in keys
            }
            return collate_dict

        self.train_loader = utils.cycle(DataLoader(
            self.train_set, batch_size=self.train_cfg.distill.train_bsize,
            shuffle=True, drop_last=True, collate_fn=collate_fn))
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.train_cfg.distill.test_bsize,
            shuffle=False, drop_last=False, collate_fn=collate_fn)

        # Containers for best losses per network
        self.best_losses = {
            mk: {
                k: float('inf') * torch.ones(self.num_nets)
                for k in self.metric_items
            } for mk in self.metrics
        }

    @staticmethod
    @typechecked
    def calc_loss(
        colors: TensorType['num_nets', 'bsize', 3],
        alphas: TensorType['num_nets', 'bsize', 1],
        colors_gt: TensorType['num_nets', 'bsize', 3],
        alphas_gt: TensorType['num_nets', 'bsize', 1],
        loss_type: str
    ) -> Dict[str, TensorType]:
        loss_fn = DistillTrainer.losses[loss_type]

        color_loss = loss_fn(colors, colors_gt)  # (N,B,3)
        alpha_loss = loss_fn(alphas, alphas_gt)  # (N,B,1)
        all_loss = torch.cat((color_loss, alpha_loss), dim=-1)  # (N,B,4)

        mean_per_net = partial(
            einops.reduce, pattern='n b c -> n', reduction='mean')

        losses = utils.to_device({
            'all': mean_per_net(all_loss),
            'color': mean_per_net(color_loss),
            'alpha': mean_per_net(alpha_loss)
        }, 'cpu')
        return losses

    @typechecked
    def calc_quantile_loss(
        self,
        colors: TensorType['num_nets', 'bsize', 3],
        alphas: TensorType['num_nets', 'bsize', 1],
        colors_gt: TensorType['num_nets', 'bsize', 3],
        alphas_gt: TensorType['num_nets', 'bsize', 1]
    ) -> Dict[str, TensorType]:
        bsize = colors.shape[1]

        mse_fn = DistillTrainer.losses['mse']
        color_loss = mse_fn(colors, colors_gt)  # (N,B,3)
        alpha_loss = mse_fn(alphas, alphas_gt)  # (N,B,1)
        all_loss = torch.cat((color_loss, alpha_loss), dim=-1)  # (N,B,4)

        mean_per_pt = partial(
            einops.reduce, pattern='n b c -> n b', reduction='mean')

        def quantile(error_mtx):
            """
            Given (N,B) error matrix, get the K-th quantile point error for
            each network, where K is predefined in config file.
            """
            quantile_idx = int(bsize * self.train_cfg.distill.quantile)
            sorted_error_mtx, _ = torch.sort(error_mtx, dim=1)
            return sorted_error_mtx[:, quantile_idx]

        losses = utils.to_device({
            'all': quantile(mean_per_pt(all_loss)),
            'color': quantile(mean_per_pt(color_loss)),
            'alpha': quantile(mean_per_pt(alpha_loss))
        }, 'cpu')
        return losses

    def print_status(self, loss):
        status_dict = {
            'Sum': '{:.5f}'.format(loss.item()),
            'Avg': '{:.5f}'.format(loss.item() / self.num_nets)
        }
        super().print_status(status_dict)

    def save_ckpt(self):
        self.logger.info('Extracting networks...')
        for i in tqdm(range(self.num_nets)):
            self.cur_nodes[i].net = self.model.extract(i)

        is_final = (self.iter_ctr == self.train_cfg.num_iterations)
        if is_final:
            self.trained_nodes.batch_push(self.cur_nodes)
            self.cur_nodes = []

        cur_nodes = [node.export_ckpt() for node in self.cur_nodes]
        empty_nodes = [node.export_ckpt() for node in self.nodes_queue]
        trained_nodes = [node.export_ckpt() for node in self.trained_nodes]

        ckpt_dict = {
            'round': self.round_ctr,
            'iter': self.iter_ctr,
            'current': cur_nodes,
            'empty': empty_nodes,
            'trained': trained_nodes,
            'optim': self.optim.state_dict(),
            'rng_states': {
                'np': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state()
            }
        }
        if is_final:
            ckpt_fn = 'round_{:d}_final.pth'.format(self.round_ctr)
        else:
            iter_str = 'iter_{:0{width}d}'.format(
                self.iter_ctr, width=len(str(self.train_cfg.num_iterations)))
            ckpt_fn = 'round_{:d}_{}.pth'.format(self.round_ctr, iter_str)
        ckpt_path = self.log_dir / ckpt_fn

        torch.save(ckpt_dict, ckpt_path)
        self.logger.info('Saved checkpoint at {}'.format(ckpt_path))

    @torch.no_grad()
    def test_networks(self):
        self.logger.info('Running evaluation...')
        colors, alphas = [], []
        for batch in tqdm(self.test_loader):
            color, density = self.model(batch['pts'], batch['dirs'])
            colors.append(color)
            alphas.append(utils.density2alpha(
                density, self.train_cfg.distill.alpha_dist))
        colors, alphas = utils.batch_cat(colors, alphas, dim=1)
        colors_gt, alphas_gt = \
            [t.to(self.device) for t in self.test_set.get_gt()]

        # Calculate losses for all metrics
        losses = {
            loss: self.calc_loss(colors, alphas, colors_gt, alphas_gt, loss)
            for loss in self.losses
        }
        losses['se_q99'] = self.calc_quantile_loss(
            colors, alphas, colors_gt, alphas_gt)

        # Update best losses for all metrics
        for mk, k in itertools.product(self.metrics, self.metric_items):
            self.best_losses[mk][k] = torch.minimum(
                    self.best_losses[mk][k], losses[mk][k])

        # Log metrics
        for i in range(self.num_nets):
            log_dict = {
                mk.upper(): '{:.5f} (c: {:.5f}, a: {:.5f})'.format(
                    md['all'][i], md['color'][i], md['alpha'][i])
                for mk, md in self.best_losses.items()}
            log_fn = self.cur_nodes[i].log_append
            super().print_status(log_dict, phase='TEST', out_fn=log_fn)

    def run_iter(self):
        # Optimize
        self.optim.zero_grad()
        batch = next(self.train_loader)
        colors, densities = self.model(batch['pts'], batch['dirs'])
        alphas = utils.density2alpha(
            densities, self.train_cfg.distill.alpha_dist)

        mse_losses = self.calc_loss(
            colors, alphas, batch['colors_gt'], batch['alphas_gt'], 'mse')
        mse_loss = mse_losses['all'].sum()
        mse_loss.backward()
        self.optim.step()

        # Update counter after backprop
        self.iter_ctr += 1

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(mse_loss)
        if self.check_interval(self.train_cfg.intervals.test, after=-1):
            self.test_networks()
        if self.check_interval(self.train_cfg.intervals.ckpt, final=True):
            self.save_ckpt()

    def run(self):
        while self.nodes_queue:
            self._init_round()
            super().run()  # Run loop
            self.logger.info('{:d} / {:d} networks completed'.format(
                len(self.trained_nodes), len(self.root_nodes)))
