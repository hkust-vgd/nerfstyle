from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from dacite import from_dict
from dacite import Config as DaciteConfig
from simple_parsing.docstring import get_attribute_docstring
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import yaml
from utils import create_logger


logger = create_logger(__name__)
T = TypeVar('T')


def flatten(
    d: Dict[str, Any],
    delim: str = '.',
    append_root: bool = True,
    show_root: bool = False
) -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            def new_key(sk):
                return k + delim + sk if append_root else delim + sk
            subitems = {new_key(sk): sv for sk, sv in flatten(v).items()}
            if show_root:
                items[k] = ''
            items.update(subitems)
        else:
            items[k] = v
    return items


def unflatten(
    d: Dict[str, Any],
    delim: str = '.'
) -> Dict[str, Any]:
    items = {}

    def setval(d: Dict[str, Any], k: str, v: Any):
        if delim not in k:
            d[k] = v
        else:
            rk, sk = k.split(delim)
            if rk not in d.keys():
                d[rk] = {}
            setval(d[rk], sk, v)

    for k, v in d.items():
        setval(items, k, v)

    return items


class Config:
    default_path: Optional[str] = None
    print_col_width: int = 30

    @classmethod
    def load_nargs(
        cls: T,
        config_path: Optional[Path] = None,
        nargs: List[str] = ()
    ) -> Tuple[T, List[str]]:
        nargs = list(nargs)
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

        # Overwrite arguments on-the-fly
        def _argnames(k: str):
            names = ['--' + k]
            if '_' in k:
                names += ['--' + k.replace('_', '-')]
            return names

        if len(nargs) > 0:
            parser = ArgumentParser(add_help=False)
            cfg_dict_flat = flatten(cfg_dict)

            for k, v in cfg_dict_flat.items():
                docstr = get_attribute_docstring(cls, k).docstring_below.replace('%', '%%')
                if v is None:
                    continue
                elif isinstance(v, bool):
                    action = 'store_false' if v else 'store_true'
                    parser.add_argument(*_argnames(k), action=action, help=docstr)
                else:
                    parser.add_argument(*_argnames(k), type=type(v), default=v, help=docstr)
            parser.print_help()
            args, nargs = parser.parse_known_args(nargs)
            cfg_dict_flat.update(vars(args))
            cfg_dict = unflatten(cfg_dict_flat)

        obj = from_dict(data_class=cls, data=cfg_dict, config=types_cfg)
        logger.info('Loaded the following {} options:'.format(cls.__name__))
        obj.print()

        return obj, nargs

    @classmethod
    def load(
        cls: T,
        config_path: Optional[Path] = None
    ) -> T:
        obj, _ = cls.load_nargs(config_path)
        return obj

    def print(self):
        disp_dict = flatten(asdict(self), delim='  ', append_root=False, show_root=True)
        for k, v in disp_dict.items():
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

    net_res: tuple
    """Local NeRF grid resolution for each dimension."""

    @dataclass
    class ReplicaConfig:
        name: str
        """Name of scene."""

        near: float
        """Near plane distance for sampling."""

        far: float
        """Far plane distance for sampling."""

        focal_ratio: float
        """Set focal length to frame side length times this value."""

        traj_ids: List[int]
        """Trajectory ids that belong to this scene."""

        black2white: bool
        """Convert black (0, 0, 0) pixels into white."""

        scale_factor: float
        """Scale the bounding box by this value to allow greater tolerance."""

    replica_cfg: Optional[ReplicaConfig]
    """Additional config settings for Replica dataset."""


@dataclass
class NetworkConfig(Config):
    x_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the input position."""

    d_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the view direction."""

    activation: str
    """Activation function after each linear layer."""

    network_seed: Optional[int]
    """Separate RNG seed for initializing networks."""

    num_samples_per_ray: int
    """No. of samples per ray."""

    pts_bsize: int
    """Batch size of point samples for evaluating the MLP network."""

    pixels_bsize: int
    """Batch size of pixels for integrating the volumetric rendering equation."""

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

    test_skip: int
    """Render the test images every N frames."""

    test_before_train: bool
    """Render the test images once before the first iteration."""

    @dataclass
    class TrainIntervalConfig:
        print: int
        log: int
        ckpt: int
        test: int

    intervals: TrainIntervalConfig
    """Intervals to be used during training."""

    @dataclass
    class DistillConfig:
        alpha_dist: float
        """Alpha distance."""

        init_data_bsize: int
        """Batch size of points when initializing dataset, at the beginning of training each new
           batch of nodes."""

        nets_bsize: int
        """No. of subnetworks to simultaneously train during one round of training."""

        quantile: float
        """Quantile to use during metric evaluation."""

        sparsity_check: float
        """Nodes with an occupied volume less than this percentage will be treated as empty, i.e.
           no subnetwork will be trained. Used in conjunction with an occupancy map only."""

        test_bsize: int
        """Batch size of points when evaluating all subnetworks."""

        test_samples_pnet: int
        """No. of points for evaluating each subnetwork."""

        train_bsize: int
        """Batch size of points when training all subnetworks."""

        train_samples_pnet: int
        """No. of points for training each subnetwork."""

    distill: Optional[DistillConfig]

    rng_seed: int
    """Seed for NumPy / PyTorch randomized number generators."""

    content_lambda: float
    """Content loss multiplier."""

    style_lambda: float
    """Style loss multiplier."""

    photo_lambda: float
    """Photorealistic loss multiplier."""

    bbox_lambda: float
    """Bounding box loss multiplier."""

    default_path = 'cfgs/training/default.yaml'


@dataclass
class OccupancyGridConfig(Config):
    subgrid_size: int
    """No. of cells to subdivide each grid cell during testing for occupancy."""

    threshold: float
    """Threshold value determining if cell is occupied."""

    voxel_bsize: int
    """No. of voxels to handle at the same time."""

    default_path = 'cfgs/occupancy_grid.yaml'
