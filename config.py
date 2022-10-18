from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from dacite import from_dict
from dacite import Config as DaciteConfig
from dacite.exceptions import UnexpectedDataError
from simple_parsing.docstring import get_attribute_docstring
import yaml

from utils import create_logger


logger = create_logger(__name__)
T = TypeVar('T', bound='Config')


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

    types_cfg = DaciteConfig(strict=True, type_hooks={
        Path: lambda p: Path(p).expanduser(),
        tuple: tuple
    })

    @classmethod
    def _get_cfg(
        cls: Type[T],
        cfg_dict: Dict
    ) -> 'Config':
        obj = None
        try:
            obj = from_dict(data_class=cls, data=cfg_dict, config=cls.types_cfg)
            logger.info('Loaded {} options:'.format(cls.__name__))
            obj.print()
        except UnexpectedDataError as e:
            print(cls.__name__)
            logger.error('Unrecognized parameters found while parsing {}: {}'.format(
                cls.__name__, ', '.join(list(e.keys))))

        return obj

    @classmethod
    def read_nargs(
        cls: Type[T]
    ) -> Tuple[T, List[str]]:
        parser = cls.create_parser()
        args, nargs = parser.parse_known_args()

        obj = cls._get_cfg(vars(args))
        return obj, nargs

    @classmethod
    def load_nargs(
        cls: Type[T],
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

        obj = cls._get_cfg(cfg_dict)
        return obj, nargs

    @classmethod
    def load(
        cls: Type[T],
        config_path: Optional[Path] = None
    ) -> T:
        obj, _ = cls.load_nargs(config_path)
        return obj

    @classmethod
    def create_parser(
        cls: Type[T],
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
    name: str
    """Name of experiment."""

    data_cfg: Optional[Path] = None
    """Path of dataset configuration file."""

    ckpt: Optional[Path] = None
    """Path of checkpoint to load from."""

    style_image: Optional[Path] = None
    """If provided, model will perform style transfer on this image."""

    run_dir: Path = './runs'
    """Root path of log folder. Logs will be stored at <run_dir>/<name>."""


@dataclass
class DatasetConfig(Config):
    root_path: Path
    """Root path of dataset."""

    type: str
    """Type of dataset."""

    bound: float
    """Radius of bounding box for sampling. Should contain entire scene."""

    scale: float
    """Scale all poses (w.r.t origin) by a factor."""

    bg_color: str
    """Background color. Any matplotlib.colors compatible string is acceptable."""

    @dataclass
    class ReplicaConfig:
        name: str
        """Name of scene."""

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
    network_seed: Optional[int]
    """Separate RNG seed for initializing networks."""

    density_out_dims: int
    """No. of dimensions for density network output."""

    density_hidden_dims: int
    """No. of dimensions for density network hidden layers."""

    density_hidden_layers: int
    """No. of hidden layers for density network."""

    rgb_hidden_dims: int
    """No. of dimensions for RGB network hidden layers."""

    rgb_hidden_layers: int
    """No. of hidden layers for RGB network."""

    @dataclass
    class HashGridConfig:
        n_lvls: int
        """No. of levels."""

        n_feats_per_lvl: int
        """No. of feature dimensions per level."""

        hashmap_size: int
        """Log2 base of hash table size of each level."""

        min_res: int
        """Resolution for coarsest level."""

        max_res_coeff: float
        """Maximum resolution coefficient. Multiply with bounding box diameter\
            to obtain resolution for finest level."""

    pos_enc: HashGridConfig
    """Config settings for positional encoding."""

    dir_enc_sh_deg: int
    """No. of basis degrees for SH encoding of view direction."""

    default_path = 'cfgs/network/default.yaml'


@dataclass
class RendererConfig(Config):
    grid_size: int
    """Side length of occupancy grid."""

    grid_bsize: Optional[int]
    """Side length of subgrid for batching. Default is same as grid_size (no batch)."""

    update_iter: int
    """No. of training iterations before updating occupancy grid once."""

    min_near: float
    """Minimum distance for near point."""

    t_thresh: float
    """Transmittance threshold during ray accumulation."""

    use_ndc: bool
    """Use NDC for rendering."""

    flip_camera: int
    """Bitwise value (0-7) for flipping X/Y/Z axes of camera frame. If no flipping, the axes \
        point to the right (X) / down (Y) / front (Z)."""

    max_steps: int
    """Maximum no. of sampled points along each ray."""

    update_thres: int
    """No. of inital steps for sampling all grid cells."""

    density_scale: float
    """Scaling factor for density value."""

    density_thresh: float
    """Threshold value for determining occupancy."""

    density_decay: float
    """Multiply densities by this value for each update."""

    default_path = 'cfgs/renderer/default.yaml'


@dataclass
class TrainConfig(Config):
    num_rays_per_batch: int
    """No. of rays to sample for each training iteration."""

    defer_patch_size: int
    """Patch side length to use during deferred backpropagation for full-image losses."""

    precrop_iterations: int
    """Perform cropping for this number of iterations."""

    precrop_fraction: float
    """Ratio for pre-cropping."""

    initial_learning_rate: float
    """Initial learning rate."""

    learning_rate_decay: int
    """No. of iterations when learning rate drops to 10% of initial value.
       Set to zero to use constant rate."""

    max_eval_count: Optional[int]
    """During evaluation, only render N frames. The frames are evenly spaced out and span the
       entire test set. If None, render all frames."""

    num_iterations: int
    """No. of total iterations for training."""

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

    rng_seed: int
    """Seed for NumPy / PyTorch randomized number generators."""

    enable_amp: bool
    """Enable FP16 AMP for training and testing."""

    ema_decay: Optional[float]
    """EMA decay rate. Leave blank if not using EMA."""

    sparsity_lambda: float
    """Sparsity loss multiplier."""

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
