from collections import namedtuple
import logging
import sys
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

Intrinsics = namedtuple('Intrinsics', ['h', 'w', 'fx', 'fy', 'cx', 'cy'])


def batch(*tensors, bsize=1, progress=False):
    batch_range = range(0, len(tensors[0]), bsize)
    if progress:
        batch_range = tqdm(batch_range)

    for i in batch_range:
        out = tuple(t[i:i+bsize] for t in tensors)
        if len(tensors) == 1:
            out = out[0]
        yield out


def batch_cat(*tensor_lists, dim=0, reshape=None) -> List[torch.Tensor]:
    if reshape is None:
        return [torch.cat(tl, dim=dim) for tl in tensor_lists]
    return [torch.cat(tl, dim=dim).reshape(reshape) for tl in tensor_lists]


def compute_psnr(loss):
    psnr = -10. * torch.log(loss) / torch.log(
        torch.FloatTensor([10.]).to(loss.device))
    return psnr


def create_logger(name, level='info'):
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    handler = ExitHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def density2alpha(densities, dists) -> torch.Tensor:
    return 1. - torch.exp(-F.relu(densities) * dists)


def get_random_pts(n, min_pt, max_pt):
    pts = np.stack([
        np.random.uniform(min_pt[i], max_pt[i], size=(n,)) for i in range(3)
    ], axis=1)
    pts_norm = 2 * (pts - min_pt) / (max_pt - min_pt) - 1
    return pts, pts_norm


def get_random_dirs(n):
    random_dirs = np.random.randn(n, 3)
    random_dirs /= np.linalg.norm(random_dirs, axis=1).reshape(-1, 1)
    return random_dirs


def load_matrix(path):
    vals = [[float(w) for w in line.strip().split()] for line in open(path)]
    return np.array(vals).astype(np.float32)


def load_ckpt_path(path, logger):
    try:
        ckpt = torch.load(path)['model']
    except FileNotFoundError:
        logger.error(
            'Checkpoint file \"{}\" not found'.format(path))
    return ckpt


class ExitHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super(ExitHandler, self).__init__(stream)

    def emit(self, record):
        super(ExitHandler, self).emit(record)
        if record.levelno >= logging.ERROR:
            sys.exit(1)


class RNGContextManager:
    """Reusable context manager that switches to a separately managed
    PyTorch RNG during the block it is wrapped with.
    """
    def __init__(self, seed: int) -> None:
        """ Initializes the RNG with seed.

        Args:
            seed (Optional[int]): Initializing seed.
        """
        self.seed = seed
        self.rng_state = None
        self.cached_state = None

    def __enter__(self) -> None:
        self.cached_state = torch.random.get_rng_state()
        if self.rng_state is not None:
            torch.random.set_rng_state(self.rng_state)
        else:
            torch.random.manual_seed(self.seed)

    def __exit__(self, *_) -> None:
        self.rng_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.cached_state)
