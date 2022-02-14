from collections import deque
import itertools
from typing import Iterable, List, Optional
import numpy as np
import torch
from tqdm import tqdm

from .base import Trainer
from networks.nerf import Nerf, create_single_nerf
from networks.multi_nerf import create_multi_nerf
import utils


class Node:
    def __init__(
        self,
        min_pt: List[int],
        max_pt: List[int]
    ) -> None:
        self.min_pt, self.max_pt = min_pt, max_pt
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

    def _generate_nodes(self) -> List[Node]:
        min_pt, max_pt = self.train_set.bbox_min, self.train_set.bbox_max
        network_res = self.dataset_cfg.network_res
        intervals = [np.linspace(start, end, num+1) for start, end, num in
                     zip(min_pt, max_pt, network_res)]

        nodes = []
        for pt in itertools.product(*[range(ax) for ax in network_res]):
            node_min_pt = [intervals[i][x] for i, x in enumerate(pt)]
            node_max_pt = [intervals[i][x+1] for i, x in enumerate(pt)]
            nodes.append(Node(node_min_pt, node_max_pt))

        return nodes

    def _build_dataset(self, node_batch: List[Node]):
        samples_per_net = 100000
        build_data_bsize = 640000
        alpha_dist = 0.0211

        # Randomly collect points and directions
        pts, dirs = [], []
        self.logger.info('Creating {:d} input samples...'.format(
            samples_per_net))
        for node in tqdm(node_batch):
            node_pts = utils.get_random_pts(
                samples_per_net, node.min_pt, node.max_pt)
            node_dirs = utils.get_random_dirs(samples_per_net)
            pts.append(torch.FloatTensor(node_pts))
            dirs.append(torch.FloatTensor(node_dirs))
        pts, dirs = utils.batch_cat(pts, dirs)

        # Compute ground truth from teacher model
        x_codes, d_codes = [], []
        colors, alphas = [], []
        self.logger.info('Computing ground truth from teacher model...')
        with torch.no_grad():
            for pts_batch, dirs_batch in utils.batch(pts, dirs,
                                                     bsize=build_data_bsize,
                                                     progress=True):
                x_code = self.lib.embed_x(pts_batch.to(self.device))
                d_code = self.lib.embed_d(dirs_batch.to(self.device))
                color, density = self.teacher(x_code, d_code)
                x_codes.append(x_code.cpu())
                d_codes.append(d_code.cpu())
                colors.append(color.cpu())
                alphas.append(utils.density2alpha(density, alpha_dist).cpu())

        x_codes, d_codes, colors, alphas = utils.batch_cat(
            x_codes, d_codes, colors, alphas,
            reshape=(len(node_batch), samples_per_net, -1))

        print(x_codes.shape, d_codes.shape, colors.shape, alphas.shape)

    def run_iter(self):
        max_num_networks = 512

        node_batch = self.nodes_queue.batch_popleft(max_num_networks)

        num_nets = len(node_batch)
        model = create_multi_nerf(num_nets, self.net_cfg).to(self.device)

        self._build_dataset(node_batch)

        raise NotImplementedError
