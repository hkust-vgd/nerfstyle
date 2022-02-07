from pathlib import Path
from dataclasses import dataclass, asdict
from dacite import from_dict
from dacite import Config as DaciteConfig
from typing import Optional
import yaml
from utils import create_logger


logger = create_logger(__name__)


def flatten(d: dict, delim: str = '.'):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            subitems = {k + delim + sk: sv for sk, sv in flatten(v).items()}
            items.update(subitems)
        else:
            items[k] = v
    return items


class Config:
    default_path: Optional[str] = None
    print_col_width: int = 30

    @classmethod
    def load(cls, config_path=None):
        has_default = cls.default_path is not None
        assert has_default or config_path is not None, \
            "No default path to use, provide a specific config path"

        cfg_dict = {}
        if has_default:
            with open(cls.default_path, 'r') as f:
                cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        if config_path is not None:
            with open(config_path, 'r') as f:
                new_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_dict.update(new_cfg_dict)

        types_cfg = DaciteConfig(type_hooks={
            Path: lambda p: Path(p).expanduser(),
            tuple: tuple
        })

        obj = from_dict(data_class=cls, data=cfg_dict, config=types_cfg)
        logger.info('Loaded the following {} options:'.format(cls.__name__))
        obj.print()

        return obj

    def print(self):
        for k, v in flatten(asdict(self)).items():
            print('{: <{width}}| {}'.format(
                k, str(v), width=self.print_col_width))


@dataclass
class DatasetConfig(Config):
    root_path: Path
    """Root path of dataset."""

    type: str
    """Type of dataset."""

    grid_res: tuple
    """Occupancy grid resolution for each dimension."""


@dataclass
class NetworkConfig(Config):
    x_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the
        input position."""

    d_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the
        view direction."""

    num_samples_per_ray: int
    """No. of samples per ray."""

    pts_bsize: int
    """No. of points to be parsed by the network at the same time."""

    default_path = 'cfgs/network/default.yaml'


@dataclass
class TrainConfig(Config):
    num_rays_per_batch: int
    """No. of rays to randomly generate per image."""

    precrop_iterations: int
    """Perform cropping for this number of iterations."""

    precrop_fraction: float
    """Ratio for pre-cropping."""

    initial_learning_rate: float
    """Initial learning rate."""

    learning_rate_decay: int
    """No. of iterations when learning rate drops to 10% of initial value.
        Set to zero to use constant rate."""

    num_iterations: int
    """No. of total iterations for training."""

    @dataclass
    class TrainIntervalConfig:
        print: int
        log: int
        ckpt: int

    intervals: TrainIntervalConfig
    """Intervals to be used during training."""

    rng_seed: int
    """Seed for NumPy / PyTorch randomized number generators."""

    default_path = 'cfgs/training/default.yaml'


@dataclass
class OccupancyGridConfig(Config):
    subgrid_size: int
    """No. of cells to subdivide each grid cell during testing for
        occupancy."""

    threshold: float
    """Threshold value determining if cell is occupied."""

    voxel_bsize: int
    """No. of voxels to handle at the same time."""

    default_path = 'cfgs/occupancy_grid.yaml'