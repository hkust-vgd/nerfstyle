import argparse
from pathlib import Path
import numpy as np
import torch

from config import DatasetConfig, NetworkConfig, OccupancyGridConfig
from data import load_bbox
from networks.nerf import SingleNerf
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('weights_path')
    parser.add_argument('--save_name', default='occupancy_grid.npz')
    parser.add_argument('--bbox_filter', action='store_true')
    args, nargs = parser.parse_known_args()

    device = torch.device('cuda:0')
    logger = utils.create_logger(__name__)

    net_cfg, nargs = NetworkConfig.load_nargs(nargs=nargs)
    dataset_cfg, nargs = DatasetConfig.load_nargs(args.dataset_cfg, nargs=nargs)
    grid_cfg, nargs = OccupancyGridConfig.load_nargs(nargs=nargs)
    if len(nargs) > 0:
        logger.error('Unrecognized arguments: ' + ' '.join(nargs))

    save_path = Path(args.weights_path).parent / args.save_name
    bbox = load_bbox(dataset_cfg)

    # Compute top left sample coords (H*W*D, 1, 3)
    top_left_samples = [torch.linspace(
        bbox.min_pt[d], bbox.max_pt[d], dataset_cfg.grid_res[d]+1
    )[:-1] for d in range(3)]
    top_left_samples = torch.stack(
        torch.meshgrid(*top_left_samples, indexing='ij'), dim=-1
    ).reshape(-1, 1, 3)

    # Compute offset coords (1, K^3, 3)
    voxel_size = (bbox.max_pt - bbox.min_pt) / dataset_cfg.grid_res
    offset_samples = [torch.linspace(0, voxel_size[d], grid_cfg.subgrid_size) for d in range(3)]
    offset_samples = torch.stack(
        torch.meshgrid(*offset_samples, indexing='ij'), dim=-1).reshape(1, -1, 3)

    # Filter positions outside bbox
    in_bbox_grid = None
    if args.bbox_filter:
        midpts = (top_left_samples + (voxel_size / 2)).float().squeeze(1)
        in_bbox_grid = torch.empty(np.prod(dataset_cfg.grid_res))

        logger.info('Filtering points outside bbox...')
        utils.batch_exec(bbox, in_bbox_grid,
                         bsize=grid_cfg.voxel_bsize, progress=True)(midpts)
        in_bbox_grid = in_bbox_grid.reshape(dataset_cfg.grid_res)

    # Combine the two
    points_per_voxel = grid_cfg.subgrid_size ** 3
    all_samples = top_left_samples.expand(-1, points_per_voxel, -1) + offset_samples

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
        voxels_batch = voxels_batch.to(device)
        out = model(voxels_batch.reshape(-1, 3), None)
        out = out.reshape(-1, points_per_voxel)
        return torch.any(out > grid_cfg.threshold, dim=1)
    utils.batch_exec(compute_occupancy_batch, vals,
                     bsize=grid_cfg.voxel_bsize, progress=True)(all_samples)

    occ_map = vals.reshape(dataset_cfg.grid_res).cpu()
    if in_bbox_grid is not None:
        occ_map = np.logical_and(occ_map, in_bbox_grid)

    count = torch.sum(occ_map).item()
    logger.info('{} out of {} voxels ({:.2f}%) are occupied'.format(
        count, len(all_samples), count * 100 / len(all_samples)))

    save_dict = {
        'map': occ_map.numpy(),
        'global_min_pt': bbox.min_pt,
        'global_max_pt': bbox.max_pt,
        'res': dataset_cfg.grid_res
    }
    np.savez_compressed(save_path, **save_dict)
    if save_path.suffix != '.npz':
        save_path = str(save_path) + ".npz"
    logger.info('Saved to "{}".'.format(save_path))


if __name__ == '__main__':
    main()
