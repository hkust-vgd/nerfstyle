from collections import deque
import itertools
from typing import Iterable, List, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked

from .base import Trainer
from networks.nerf import Nerf, create_single_nerf
from networks.multi_nerf import create_multi_nerf
import utils


patch_typeguard()


class Node:
    def __init__(
        self,
        min_pt: List[float],
        max_pt: List[float]
    ) -> None:
        self.min_pt, self.max_pt = np.array(min_pt), np.array(max_pt)
        self.net: Optional[Nerf] = None
        self.leq_child: Optional[Node] = None
        self.gt_child: Optional[Node] = None


class NodeQueue(deque):
    def __init__(
        self,
        iterable: Iterable[Node] = ...,
        maxlen: Optional[int] = None
    ) -> None:
        super().__init__(iterable, maxlen)

    def batch_popleft(self, count: int) -> List[Node]:
        batch = [self.popleft() for _ in range(min(len(self), count))]
        return batch


class DistillDataset(Dataset):
    @typechecked
    def __init__(
        self,
        x_codes: TensorType['num_nets', 'dataset_size', -1],
        d_codes: TensorType['num_nets', 'dataset_size', -1],
        colors: TensorType['num_nets', 'dataset_size', 3],
        alphas: TensorType['num_nets', 'dataset_size', 1]
    ) -> None:
        self._x_codes = x_codes
        self._d_codes = d_codes
        self._colors = colors
        self._alphas = alphas

        self._num_nets, self._size = x_codes.shape[:2]

    def __getitem__(self, index):
        item_dict = {
            'x_codes': self._x_codes[:, index],
            'd_codes': self._d_codes[:, index],
            'colors_gt': self._colors[:, index],
            'alphas_gt': self._alphas[:, index]
        }
        return item_dict

    def __len__(self) -> int:
        return self._size

    def __str__(self):
        desc = 'distillation dataset over {:d} networks with {:d} entries'
        return desc.format(self._num_nets, self._size)


class DistillTrainer(Trainer):
    def __init__(self, args, nargs):
        super().__init__(__name__, args, nargs)

        self.root_nodes = self._generate_nodes()
        self.nodes_queue = NodeQueue(self.root_nodes)

        # Load teacher model
        if args.teacher_ckpt_path is None:
            self.logger.error('Please provide path to teacher model')

        ckpt = utils.load_ckpt_path(args.teacher_ckpt_path, self.logger)
        self.teacher = create_single_nerf(self.net_cfg).to(self.device)
        self.teacher.load_state_dict(ckpt)
        self.teacher.eval()
        self.logger.info('Loaded teacher model ' + str(self.teacher))

        # Entities reset in each search step
        self.num_nets = 0
        self.model, self.optim = None, None
        self.train_set, self.test_set = None, None
        self.train_loader = None

    def _generate_nodes(self) -> List[Node]:
        bbox_path = self.dataset_cfg.root_path / 'bbox.txt'
        min_pt, max_pt = utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)

        network_res = self.dataset_cfg.network_res
        intervals = [np.linspace(start, end, num+1) for start, end, num in
                     zip(min_pt, max_pt, network_res)]

        nodes = []
        for pt in itertools.product(*[range(ax) for ax in network_res]):
            node_min_pt = [intervals[i][x] for i, x in enumerate(pt)]
            node_max_pt = [intervals[i][x+1] for i, x in enumerate(pt)]
            nodes.append(Node(node_min_pt, node_max_pt))

        return nodes

    def _build_dataset(
        self,
        node_batch: List[Node],
        samples_per_net: int
    ) -> Dataset:
        build_data_bsize = 160000
        alpha_dist = 0.0211

        # Randomly collect points and directions
        pts, pts_norm, dirs = [], [], []
        self.logger.info('Creating {:d} input samples...'.format(
            samples_per_net))
        for node in tqdm(node_batch):
            node_pts, node_pts_norm = utils.get_random_pts(
                samples_per_net, node.min_pt, node.max_pt)
            node_dirs = utils.get_random_dirs(samples_per_net)
            pts.append(torch.FloatTensor(node_pts))
            pts_norm.append(torch.FloatTensor(node_pts_norm))
            dirs.append(torch.FloatTensor(node_dirs))
        pts, pts_norm, dirs = utils.batch_cat(pts, pts_norm, dirs)

        # Compute embedded points / dirs
        # Evaluate ground truth from teacher model
        x_codes, d_codes, colors, alphas = [], [], [], []
        self.logger.info('Computing input and target data...')
        with torch.no_grad():
            for batch in utils.batch(pts, pts_norm, dirs,
                                     bsize=build_data_bsize, progress=True):
                pts_batch, pts_norm_batch, dirs_batch = batch
                x_code = self.lib.embed_x(pts_batch.to(self.device))
                x_norm_code = self.lib.embed_x(pts_norm_batch.to(self.device))
                d_code = self.lib.embed_d(dirs_batch.to(self.device))
                color, density = self.teacher(x_code, d_code)

                x_codes.append(x_norm_code.cpu())
                d_codes.append(d_code.cpu())
                colors.append(color.cpu())
                alphas.append(utils.density2alpha(density, alpha_dist).cpu())

        x_codes, d_codes, colors, alphas = utils.batch_cat(
            x_codes, d_codes, colors, alphas,
            reshape=(len(node_batch), samples_per_net, -1))

        dataset = DistillDataset(x_codes, d_codes, colors, alphas)
        return dataset

    def _reset_trainer(self) -> None:
        """
        Resets model, optimizer and datasets. Performed once before training a
        new batch of nodes.
        """
        max_num_networks = 512
        train_samples_per_net = 100000
        test_samples_per_net = 20000
        train_bsize = 512

        node_batch = self.nodes_queue.batch_popleft(max_num_networks)
        self.num_nets = len(node_batch)
        self.model = create_multi_nerf(self.num_nets, self.net_cfg).to(
            self.device)
        # self.logger.info('Created model ' + str(self.model))
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.initial_learning_rate)

        self.train_set = self._build_dataset(
            node_batch, train_samples_per_net)
        self.logger.info('Loaded ' + str(self.train_set))
        self.test_set = self._build_dataset(
            node_batch, test_samples_per_net)
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
            self.train_set, batch_size=train_bsize,
            shuffle=True, drop_last=True, collate_fn=collate_fn))

    @staticmethod
    def calc_loss(rendered, target):
        mse_loss = F.mse_loss(rendered, target, reduction='none')
        mse_loss = einops.reduce(mse_loss, 'n b c -> n', 'mean')
        return mse_loss.sum()

    def print_status(self, loss):
        status_dict = {
            'Sum': '{:.5f}'.format(loss.item()),
            'Avg': '{:.5f}'.format(loss.item() / self.num_nets)
        }
        super().print_status(status_dict)

    def run_iter(self):
        alpha_dist = 0.0211

        if self.iter_ctr == 0:
            self._reset_trainer()

        self.optim.zero_grad()
        batch = next(self.train_loader)
        colors, densities = self.model(batch['x_codes'], batch['d_codes'])
        alphas = utils.density2alpha(densities, alpha_dist)

        # Concat (N,B,3) + (N,B,1) -> (N,B,4) for loss computation
        loss = self.calc_loss(
            torch.cat((colors, alphas), dim=-1),
            torch.cat((batch['colors_gt'], batch['alphas_gt']), dim=-1)
        )
        loss.backward()
        self.optim.step()

        # Update counter after backprop
        self.iter_ctr += 1

        # Misc. tasks at different intervals
        if self.check_interval(self.train_cfg.intervals.print):
            self.print_status(loss)
