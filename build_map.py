import argparse
from pathlib import Path
import numpy as np
import torch

from config import DatasetConfig, NetworkConfig, OccupancyGridConfig
from networks.nerf import SingleNerf
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('weights_path')
    parser.add_argument('--save_name', default='occupancy_grid.npz')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    logger = utils.create_logger(__name__)

    net_cfg = NetworkConfig.load()
    dataset_cfg = DatasetConfig.load(args.dataset_cfg)
    grid_cfg = OccupancyGridConfig.load()

    bbox_path = dataset_cfg.root_path / 'bbox.txt'
    bbox_min, bbox_max = utils.load_matrix(bbox_path)[0, :-1].reshape(2, 3)
    save_path = Path(args.weights_path).parent / args.save_name

    # Compute top left sample coords (H*W*D, 1, 3)
    top_left_samples = [torch.linspace(
        bbox_min[d], bbox_max[d], dataset_cfg.grid_res[d]+1
    )[:-1] for d in range(3)]
    top_left_samples = torch.stack(
        torch.meshgrid(*top_left_samples, indexing='ij'), dim=-1
    ).reshape(-1, 1, 3)

    # Compute offset coords (1, K^3, 3)
    voxel_size = (bbox_max - bbox_min) / dataset_cfg.grid_res
    offset_samples = [torch.linspace(
        0, voxel_size[d], grid_cfg.subgrid_size) for d in range(3)]
    offset_samples = torch.stack(
        torch.meshgrid(*offset_samples, indexing='ij'), dim=-1
    ).reshape(1, -1, 3)

    # Combine the two
    points_per_voxel = grid_cfg.subgrid_size ** 3
    all_samples = top_left_samples.expand(-1, points_per_voxel, -1) \
        + offset_samples
    all_samples = all_samples.to(device)

    # Load embedders and model
    model = SingleNerf.create_nerf(net_cfg).to(device)

    @utils.loader(logger)
    def _load(ckpt_path):
        ckpt = torch.load(ckpt_path)['model']
        model.load_state_dict(ckpt, strict=False)

    _load(args.weights_path)
    logger.info('Loaded model from "{}"'.format(args.weights_path))

    # Compute occupancy grid
    logger.info('Computing occupancy grid...')
    vals = torch.empty(np.prod(dataset_cfg.grid_res))

    def compute_occupancy_batch(voxels_batch):
        out = model(voxels_batch.reshape(-1, 3), None)
        out = out.reshape(grid_cfg.voxel_bsize, points_per_voxel)
        return torch.any(out > grid_cfg.threshold, dim=1)
    utils.batch_exec(compute_occupancy_batch, vals,
                     bsize=grid_cfg.voxel_bsize, progress=True)(all_samples)

    occ_map = vals.reshape(dataset_cfg.grid_res).cpu()
    count = torch.sum(occ_map).item()
    logger.info('{} out of {} voxels ({:.2f}%) are occupied'.format(
        count, len(all_samples), count * 100 / len(all_samples)))

    save_dict = {
        'map': occ_map.numpy(),
        'global_min_pt': bbox_min,
        'global_max_pt': bbox_max,
        'res': dataset_cfg.grid_res
    }
    np.savez_compressed(save_path, **save_dict)
    logger.info('Saved to "{}".'.format(save_path))


if __name__ == '__main__':
    main()
