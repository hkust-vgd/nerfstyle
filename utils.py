from collections import namedtuple
import logging
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
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


def batch_exec(
    func: Callable,
    *dest: Iterable[Any],
    bsize: int = 1,
    in_dim: int = 0,
    out_dim: Optional[int] = None,
    progress: bool = False
) -> Callable:
    """ Batch execution of function.

    Args:
        func (Callable): Function to be executed.
        *dest (Iterable[Any]): Objects to store execution result.
        bsize (int, optional): Batch size. Defaults to 1.
        in_dim (int, optional): Generate batches of all arguments by iterating
            over this dimension. Defaults to 0.
        out_dim (Optional[int], optional): Accumulates results over this
            dimension. Defaults to None (i.e. same as in_dim).
        progress (bool, optional): Displays progress meter. Defaults to False.

    Returns:
        Callable: Batched function.
    """
    if out_dim is None:
        out_dim = in_dim

    def create_slice(dim, s, e):
        if dim == 0:
            return slice(s, e)
        return tuple([slice(None) for _ in range(dim)] + slice(s, e))

    def get_size(obj, dim):
        if hasattr(obj, 'shape'):
            return obj.shape[dim]
        assert dim == 0
        return len(obj)

    def batch_func(*args):
        size = get_size(args[0], in_dim)
        main_loop = range(0, size, bsize)
        test = tqdm(main_loop, total=size, disable=(not progress))
        out_s = 0
        for in_s in main_loop:
            in_e = min(size, in_s + bsize)
            in_slice = create_slice(in_dim, in_s, in_e)
            bargs = [a[in_slice] for a in args]

            bout = func(*bargs)
            if not isinstance(bout, tuple):
                bout = (bout, )

            out_bsize = get_size(bout[0], out_dim)
            out_slice = create_slice(out_dim, out_s, out_s + out_bsize)
            for d, bo in zip(dest, bout):
                d[out_slice] = bo
            out_s += out_bsize
            test.update(in_e - in_s)
        test.close()

    return batch_func


def compute_psnr(loss):
    psnr = -10. * torch.log(loss) / torch.log(
        torch.FloatTensor([10.]).to(loss.device))
    return psnr


def compute_tensor_size(
    *tensors: torch.Tensor,
    unit: str = 'B',
    prec: int = 3
) -> str:
    bytes_count = sum([t.nelement() * t.element_size() for t in tensors])
    unit = unit.upper()
    if unit == 'B':
        return '{:d} B'.format(bytes_count)
    elif unit == 'KB':
        return '{:.{prec}f} KB'.format(bytes_count / (2 ** 10), prec=prec)
    elif unit == 'MB':
        return '{:.{prec}f} MB'.format(bytes_count / (2 ** 20), prec=prec)
    elif unit == 'GB':
        return '{:.{prec}f} GB'.format(bytes_count / (2 ** 30), prec=prec)
    else:
        raise ValueError('Unrecognized unit ' + unit)


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
    return torch.FloatTensor(pts), torch.FloatTensor(pts_norm)


def get_random_dirs(n):
    random_dirs = np.random.randn(n, 3)
    random_dirs /= np.linalg.norm(random_dirs, axis=1).reshape(-1, 1)
    return torch.FloatTensor(random_dirs)


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


def reshape(*tensors: torch.Tensor, shape: Tuple[int]) -> Tuple[torch.Tensor]:
    return tuple([t.reshape(shape) for t in tensors])


def to_device(old_dict: Dict[str, torch.Tensor], device: str):
    new_dict = {k: v.to(device) for k, v in old_dict.items()}
    return new_dict


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
