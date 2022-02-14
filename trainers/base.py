from pathlib import Path
import torch
from utils import create_logger


class Trainer:
    def __init__(self, name, args, _):
        self.logger = create_logger(name)
        self.iter_ctr = 0
        self.time0 = 0

        self.name = args.name
        self.log_path: Path = Path('./runs') / self.name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda:0')

    def run_iter(self):
        pass

    def run(self):
        while self.iter_ctr < self.train_cfg.num_iterations:
            self.run_iter()
