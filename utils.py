from collections import namedtuple
import logging
import sys
import numpy as np
import torch
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


def compute_psnr(loss):
    psnr = -10. * torch.log(loss) / torch.log(
        torch.FloatTensor([10.]).to(loss.device))
    return psnr


def load_matrix(path):
    vals = [[float(w) for w in line.strip().split()] for line in open(path)]
    return np.array(vals).astype(np.float32)


class ExitHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super(ExitHandler, self).__init__(stream)

    def emit(self, record):
        super(ExitHandler, self).emit(record)
        if record.levelno >= logging.ERROR:
            sys.exit(1)


def create_logger(name, level='info'):
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    handler = ExitHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
