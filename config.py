from dataclasses import dataclass
import yaml


class Config:
    default_path: str

    @classmethod
    def load(cls, config_path=None):
        with open(cls.default_path, 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        if config_path is not None:
            # TODO: Overwrite default parameters
            raise NotImplementedError

        return cls(**cfg_dict)


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

    learning_rate_decay_rate: int
    """No. of iterations for a full exponential decay."""

    num_iterations: int
    """No. of total iterations for training."""

    default_path = 'cfgs/training/default.yaml'
