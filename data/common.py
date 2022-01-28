import numpy as np
from collections import namedtuple

Intrinsics = namedtuple('Intrinsics', ['h', 'w', 'fx', 'fy', 'cx', 'cy'])


def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)
