import argparse
from pathlib import Path
import torch
from config import DatasetConfig, NetworkConfig
from networks.nerf import Nerf
from utils import load_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('weights_path')
    args = parser.parse_args()
    device = torch.device('cuda:0')

    dataset_cfg = DatasetConfig.load(args.dataset_cfg)
    net_cfg = NetworkConfig.load()

    bbox_path = dataset_cfg.root_path / 'bbox.txt'
    bbox_min, bbox_max = load_matrix(bbox_path)[0, :-1].reshape(2, 3)

    ckpt = torch.load(args.weights_path)['model']
    x_channels, d_channels = 3, 3
    x_enc_channels = 2 * x_channels * net_cfg.x_enc_count + x_channels
    d_enc_channels = 2 * d_channels * net_cfg.d_enc_count + d_channels
    model = Nerf(x_enc_channels, d_enc_channels,
                 8, 256, [256, 128], [5]).to(device)
    model.load_state_dict(ckpt)
    model.eval()


if __name__ == '__main__':
    main()
