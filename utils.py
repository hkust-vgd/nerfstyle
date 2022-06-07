from collections import defaultdict
import functools
import logging
from pathlib import Path
import sys
from tabulate import tabulate
from time import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    *dest: Any,
    bsize: int = 1,
    in_dim: int = 0,
    out_dim: Optional[int] = None,
    progress: bool = False,
    is_iter: bool = False
) -> Callable:
    """ Batch execution of function.

    Args:
        func (Callable): Function to be executed.
        *dest (Any): Objects to store execution result.
        bsize (int, optional): Batch size. Defaults to 1.
        in_dim (int, optional): Generate batches of all arguments by iterating
            over this dimension. Defaults to 0.
        out_dim (Optional[int], optional): Accumulates results over this
            dimension. Defaults to None (i.e. same as in_dim).
        progress (bool, optional): Displays progress meter. Defaults to False.
        is_iter (bool, optional): If True, input argument is a single iterator object. Parameters
            'bsize' and 'in_dim' would be ignored. Defaults to False.

    Returns:
        Callable: Batched function.
    """
    if out_dim is None:
        out_dim = in_dim

    def create_slice(dim, s, e):
        if dim == 0:
            return slice(s, e)
        return tuple([slice(None) for _ in range(dim)] + [slice(s, e)])

    def get_size(obj, dim):
        if hasattr(obj, 'shape'):
            return obj.shape[dim]
        assert dim == 0
        return len(obj)

    def wrap_tuple(obj):
        if isinstance(obj, tuple):
            return obj
        return (obj, )

    def batch_func(*args):
        if is_iter:
            size = len(args[0])
            main_loop = args[0]
        else:
            size = get_size(args[0], in_dim)
            main_loop = range(0, size, bsize)
        prog_bar = tqdm(main_loop, total=size, disable=(not progress))

        out_s = 0
        for loop_obj in main_loop:
            if is_iter:
                # "loop_obj" = batch args
                bargs = wrap_tuple(loop_obj)
            else:
                # "loop_obj" = start position
                in_e = min(size, loop_obj + bsize)
                in_slice = create_slice(in_dim, loop_obj, in_e)
                bargs = [a[in_slice] for a in args]

            bout = wrap_tuple(func(*bargs))
            out_bsize = get_size(bout[0], out_dim)
            out_slice = create_slice(out_dim, out_s, out_s + out_bsize)
            for d, bo in zip(dest, bout):
                d[out_slice] = bo
            out_s += out_bsize

            if is_iter:
                prog_bar.update()
            else:
                prog_bar.update(in_e - loop_obj)
        prog_bar.close()

    return batch_func


def compute_psnr(loss):
    psnr = -10. * torch.log(loss) / torch.log(torch.FloatTensor([10.]).to(loss.device))
    return psnr


def compute_tensor_size(
    *tensors: torch.Tensor,
    unit: str = 'B',
    prec: int = 3
) -> str:
    bytes_count = sum([t.nelement() * t.element_size() for t in tensors])
    return format_bytes(bytes_count, unit, prec)


def create_logger(name, level='info'):
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    handler = ExitHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def density2alpha(densities, dists) -> torch.Tensor:
    return 1. - torch.exp(-F.relu(densities) * dists)


def format_bytes(
    bytes_count: int,
    unit: str = 'B',
    prec: int = 3
) -> str:
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


def get_random_pts(n, min_pt, max_pt):
    pts = np.stack([np.random.uniform(min_pt[i], max_pt[i], size=(n,)) for i in range(3)], axis=1)
    pts_norm = 2 * (pts - min_pt) / (max_pt - min_pt) - 1
    return torch.FloatTensor(pts), torch.FloatTensor(pts_norm)


def get_random_dirs(n):
    random_dirs = np.random.randn(n, 3)
    random_dirs /= np.linalg.norm(random_dirs, axis=1).reshape(-1, 1)
    return torch.FloatTensor(random_dirs)


def get_repr(obj, attrs):
    attrs_str = ', '.join(['{}={}'.format(attr, getattr(obj, attr)) for attr in attrs])
    obj_repr = '{}({})'.format(type(obj).__name__, attrs_str)
    return obj_repr


def load_matrix(path):
    vals = [[float(w) for w in line.strip().split()] for line in open(path)]
    return np.array(vals).astype(np.float32)


def loader(logger=None):
    def decorate(load_fn):
        error_emitter = logger.error if logger is not None else sys.exit

        @functools.wraps(load_fn)
        def inner_loader(path):
            try:
                return load_fn(path)
            except FileNotFoundError:
                error_emitter('File \"{}\" not found'.format(path))
            except KeyError:
                traceback.print_exception(*sys.exc_info())
                error_emitter('File \"{}\" has invalid format'.format(path))

        return inner_loader
    return decorate


def reshape(*tensors: torch.Tensor, shape: Tuple[int]) -> Tuple[torch.Tensor]:
    return tuple([t.reshape(shape) for t in tensors])


def parse_rgb(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    img = Image.open(path)
    if size is not None:
        img = img.resize(size)

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = einops.rearrange(img_np, 'h w c -> c h w')
    return img_np


def print_memory_usage(
    desc: str,
    device: Optional[torch.device] = None,
    unit: str = 'MB'
) -> None:
    """
    Prints current allocated and cached memory in a GPU device.
    - Use `del` to remove unneeded variables if too much memory is allocated.
    - Use `torch.cuda.empty_cache()` to free up cache space.

    Args:
        desc (str): Description of location where this function is called.
        device (Optional[torch.device], optional): Device for checking. Defaults to None (uses
            current device).
        unit (str, optional): Unit for formatting, must be in [B, KB, MB, GB]. Defaults to 'MB'.
    """
    mem_allocated = format_bytes(torch.cuda.memory_allocated(device), unit=unit)
    mem_cached = format_bytes(torch.cuda.memory_reserved(device), unit=unit)

    msg = '{}: Allocated - {}, Cached - {}'.format(desc, mem_allocated, mem_cached)
    print(msg)


def prompt_bool(msg):
    result = None
    prompt_msg = msg + ' (Y/N) '

    while result not in ['y', 'n']:
        result = input(prompt_msg).lower()

    return result == 'y'


def rmtree(path: Path):
    """Removes directory and all child paths recursively.

    Args:
        path (Path): Path to directory to be removed.
    """
    if path.is_file():
        path.unlink()
    else:
        for child in path.iterdir():
            rmtree(child)
        path.rmdir()


def to_device(old_dict: Dict[str, torch.Tensor], device: str):
    new_dict = {k: v.to(device) for k, v in old_dict.items()}
    return new_dict


class Clock:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.prev = None
        self.reset()
        self.record = defaultdict(list)
        self.cur_record = defaultdict(list)
        self.stats = {
            'Min': np.min,
            'Max': np.max,
            'Avg': np.mean
        }

    def reset(self):
        self.prev = time()

    def aggregate(self):
        for k, v in self.cur_record.items():
            self.record[k].append(np.sum(v))
        self.cur_record.clear()

    def click(self, msg, reset=True, click_verbose=False):
        delta = time() - self.prev
        if reset:
            self.reset()

        self.cur_record[msg].append(delta)
        if self.verbose or click_verbose:
            print('Event "{}": {:.3f}s'.format(msg, delta))

    def print_stats(self):
        self.aggregate()

        stats_table = []
        for k in self.record.keys():
            stats_row = [k]
            for stat_fn in self.stats.values():
                stat = stat_fn(self.record[k])
                stats_row.append('{:.5f} s'.format(stat))
            stats_table.append(stats_row)

        headers = ['Event'] + list(self.stats.keys())
        stats_table = tabulate(stats_table, headers=headers)
        print(stats_table)


global_clock = Clock()


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
