from collections import defaultdict
import functools
import logging
from pathlib import Path
import random
import shutil
import string
import sys
from tabulate import tabulate
from time import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import einops
import git
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


# TODO: Put these in smaller classes and delete useless ones


class BufferDir:
    """ Context manager for creating / cleaning a temporary buffer directory. """
    def __init__(self, root_dir: Path, k: int = 20) -> None:
        assert root_dir.exists()
        dir_name = ''.join(random.choices(string.ascii_letters, k=k))
        self.path = root_dir / dir_name

    def __enter__(self):
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(exist_ok=False)
        return self.path

    def __exit__(self, *_):
        shutil.rmtree(self.path)


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


class CustomFormatter(logging.Formatter):
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: format,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class EMA(ExponentialMovingAverage):
    """ Extended EMA class with enable/disable parameter.
    When enabled, EMA works as usual; when disabled, calls to EMA methods are ignored.
    Initally, EMA is disabled if `decay` is `None`, else enabled.
    This can be modified afterwards via setting `self.enabled`.
    """
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: Optional[float]
    ):
        self.enabled = (decay is not None)
        if decay is None:
            decay = 1.

        super(EMA, self).__init__(parameters, decay)

        def wrap(fn):
            def new_fn(*args, **kwargs):
                return fn(*args, **kwargs) if self.enabled else lambda _: None
            return new_fn

        ema_methods = [m for m in dir(ExponentialMovingAverage) if not m.startswith('__')]
        for m in ema_methods:
            old_fn = getattr(self, m)
            if callable(old_fn):
                setattr(self, m, wrap(old_fn))


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


def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def color_str2rgb(color: str) -> Tuple[float]:
    color_map = mcolors.get_named_colors_mapping()
    assert color in color_map.keys(), 'Invalid color "{}"'.format(color)
    return mcolors.to_rgb(color_map[color])


# Input: [C, H, W] or [1, C, H, W]
def collage_h(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    assert len(img1.shape) == len(img2.shape)
    assert img1.shape[-3] == img2.shape[-3]
    assert (len(img1.shape) == 3) or (len(img1.shape) == 4 and img1.shape[0] == 1)

    h1, h2 = img1.shape[-2], img2.shape[-2]
    h_out = max(h1, h2)
    # pad parameter -> (left, right, top, bottom)
    if h1 < h_out:
        img1_pad = F.pad(img1, pad=(0, 0, 0, h_out - h1), value=0)
        collage = torch.cat((img1_pad, img2), dim=-1)
    else:
        img2_pad = F.pad(img2, pad=(0, 0, 0, h_out - h2), value=0)
        collage = torch.cat((img1, img2_pad), dim=-1)

    return collage


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
    handler.setFormatter(CustomFormatter())
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


def get_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


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
    size: Optional[Union[Tuple[int, int], int]] = None
) -> np.ndarray:
    img = Image.open(path)
    if size is not None:
        if isinstance(size, int):
            img_w, img_h = img.size
            if img_w > img_h:
                size = (size, int(size * img_h / img_w))
            else:
                size = (int(size * img_w / img_h), size)
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


def train_test_split(total: int, split_every: int, is_train: bool) -> List[int]:
    ids = [i for i in np.arange(total) if (i % split_every == 0) != is_train]
    return ids


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply


# TODO: type check ndarray sizes


def full_mtx(mtx: np.ndarray):
    assert mtx.shape[-1] == 4 and mtx.shape[-2] <= 4, 'Wrong input shape'
    rows = mtx.shape[-2]
    if rows == 4:  # already full
        return mtx

    base = np.tile(np.eye(4), mtx.shape[:-2] + (1, 1))
    base[..., :rows, :] = mtx[..., :, :]
    return base.astype(mtx.dtype)


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def build_view_mtx(pos: np.ndarray, up: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    up, vec2 = normalize(up), normalize(vec2)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    mtx = np.stack([vec0, vec1, vec2, pos], axis=1)
    return mtx


def poses_avg(poses: np.ndarray) -> np.ndarray:
    up = np.sum(poses[:, :3, 1], axis=0)
    vec2 = np.sum(poses[:, :3, 2], axis=0)
    pos = np.mean(poses[:, :3, 3], axis=0)
    return build_view_mtx(pos, up, vec2)
