from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import dataclasses
from dataclasses import dataclass
from dacite import from_dict
from dacite import Config as DaciteConfig
from enum import Enum
from pathlib import Path
from simple_parsing.docstring import get_attribute_docstring
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
import yaml

from common import TrainMode
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


def is_opt(field_opt_type):
    # TODO: Use get_origin / get_args (Python 3.8)
    if getattr(field_opt_type, '__origin__', None) is not Union:
        return False
    return type(None) == field_opt_type.__args__[1]


def extract_opt(field_opt_type):
    assert is_opt(field_opt_type)
    return field_opt_type.__args__[0]


class Config:
    default_path: Optional[str] = None
    print_col_width: int = 30

    types_cfg = DaciteConfig(type_hooks={
        Path: lambda p: Path(p).expanduser(),
        TrainMode: lambda m: TrainMode[m.upper()],
        tuple: tuple
    })

    @classmethod
    def read_nargs(
        cls: T,
    ) -> Tuple[T, List[str]]:
        parser = cls.create_parser()
        args, nargs = parser.parse_known_args()
        cfg_dict = vars(args)

        obj = from_dict(data_class=cls, data=cfg_dict, config=cls.types_cfg)
        logger.info('Loaded the following {} options:'.format(cls.__name__))
        obj.print()

        return obj, nargs

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

        # Overwrite arguments on-the-fly
        if len(nargs) > 0:
            parser = cls.create_parser(cfg_dict)
            args, nargs = parser.parse_known_args(nargs)
            cfg_dict = unflatten(vars(args))

        obj = from_dict(data_class=cls, data=cfg_dict, config=cls.types_cfg)
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

    @classmethod
    def create_parser(
        cls,
        loaded_values: Optional[Dict[str, Any]] = None
    ) -> ArgumentParser:
        def _argnames(k: str):
            names = ['--' + k]
            if '_' in k:
                names += ['--' + k.replace('_', '-')]
            return names

        cfg_dict = {f.name: f.default for f in dataclasses.fields(cls)}
        if loaded_values is not None:
            for k, v in loaded_values.items():
                assert k in cfg_dict
                cfg_dict[k] = v
        cfg_dict_flat = flatten(cfg_dict)
        parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)

        for k, v in cfg_dict_flat.items():
            field_type = cls
            docstr = ''

            # Get (nested) field type and docstring
            for k_part in k.split('.'):
                docstr = get_attribute_docstring(cls, k_part).docstring_below
                docstr = docstr.replace('%', '%%')

                cfg_types = {f.name: f.type for f in dataclasses.fields(field_type)}
                field_type = cfg_types[k_part]
                if is_opt(field_type):
                    field_type = extract_opt(field_type)

            if v is dataclasses.MISSING:
                # Required argument
                parser.add_argument(k, help=docstr)
            elif v is None:
                # Optional argument, no default value
                if not dataclasses.is_dataclass(field_type):
                    parser.add_argument(*_argnames(k), type=field_type, help=docstr)

            # Optional argument, has loaded default value
            elif isinstance(v, bool):
                action = 'store_false' if v else 'store_true'
                parser.add_argument(*_argnames(k), action=action, help=docstr)
            elif isinstance(v, Enum):
                choices = [n.lower() for n in type(v)._member_names_]
                default_choice = v.name.lower()
                parser.add_argument(*_argnames(k), choices=choices,
                                    default=default_choice, help=docstr)
            else:
                parser.add_argument(*_argnames(k), type=type(v), default=v, help=docstr)

        return parser

    def print(self):
        disp_dict = flatten(dataclasses.asdict(self), delim='  ', append_root=False, show_root=True)
        for k, v in disp_dict.items():
            print('{: <{width}}| {}'.format(k, str(v), width=self.print_col_width))


@dataclass
class BaseConfig(Config):
    data_cfg_path: Path
    """Path of dataset configuration file."""

    name: str
    """Name of experiment."""

    run_dir: Path = './runs'
    """Root path of log folder. Logs will be stored at <run_dir>/<name>."""

    mode: TrainMode = TrainMode.PRETRAIN
    """Training mode."""

    ckpt_path: Optional[Path] = None
    """Path of checkpoint to load from."""

    teacher_ckpt_path: Optional[Path] = None
    """Path of teacher checkpoint used to training the distillation stage."""

    occ_map: Optional[Path] = None
    """Path of occupancy map, used for speeding up inference."""

    style_image: Optional[Path] = None
    """If provided, model will perform style transfer on this image."""


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

    bg_color: str
    """Background color. Any matplotlib.colors compatible string is acceptable."""

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
        """Scale bounding box by this value before subdivision into smaller networks."""

    replica_cfg: Optional[ReplicaConfig]
    """Additional config settings for Replica dataset."""

    default_path = 'cfgs/dataset/default.yaml'


@dataclass
class NetworkConfig(Config):
    x_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the input position."""

    d_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the view direction."""

    x_layers: int
    """No. of linear layers in position MLP (i.e. before color / density split)."""

    d_layers: int
    """No. of linear layers in density MLP (i.e. after position / direction merge)."""

    x_widths: Union[int, List[int]]
    """No. of channels for each layer of position MLP."""

    d_widths: Union[int, List[int]]
    """No. of channels for each layer of density MLP."""

    x_skips: List[int]
    """Indices of hidden layers to insert skip connection from input position."""

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

    ert_bsize: int
    """Max no. of points to evaluate per ray, when using ERT."""

    ert_trans_thres: float
    """Terminate rays with transmittance lower than this threshold, when using ERT."""

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

        converge_thres: float
        """Retrain sub-network if quantile loss is not smaller than this threshold."""

        init_data_bsize: int
        """Batch size of points when initializing dataset, at the beginning of training each new
           batch of nodes."""

        max_retries: int
        """Retrain a network for at most this no. of times to prevent infinite loop."""

        nets_bsize: int
        """No. of subnetworks to simultaneously train during one round of training."""

        quantile: float
        """Quantile to use during metric evaluation."""

        retrain: Optional[Path]
        """List of nodes to retrain for distillation stage."""

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

    sparsity_lambda: float
    """Sparsity loss multiplier."""

    sparsity_bbox_scale: float
    """Scale BBox by this factor, then sample points inside to compute sparsity loss."""

    sparsity_exp_coeff: float
    """Exponential coefficient in sparsity loss computation."""

    sparsity_samples: int
    """No. of point samples per iteration for calculating sparsity loss."""

    weight_reg_lambda: float
    """Weight regularization multiplier."""

    content_lambda: float
    """Content loss multiplier."""

    style_lambda: float
    """Style loss multiplier."""

    photo_lambda: float
    """Photorealistic loss multiplier."""

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
