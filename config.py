from ast import Load
from dataclasses import dataclass
from dacite import from_dict
import yaml


class Config:
    default_path: str

    @classmethod
    def load(cls, config_path=None):
        with open(cls.default_path, 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                new_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_dict.update(new_cfg_dict)

        return from_dict(data_class=cls, data=cfg_dict)


@dataclass
class NetworkConfig(Config):
    x_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the input position."""

    d_enc_count: int
    """No. of frequencies (pairs of sines / cosines) to encode the view direction."""

    num_samples_per_ray: int
    """No. of samples per ray."""

    network_chunk_size: int
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
