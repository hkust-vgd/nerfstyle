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
    args = parser.parse_args()
    device = torch.device('cuda:0')
    logger = utils.create_logger(__name__)

    net_cfg = NetworkConfig.load()
    dataset_cfg = DatasetConfig.load(args.dataset_cfg)
    grid_cfg = OccupancyGridConfig.load()

    bbox_min, bbox_max, bbox_coords = load_bbox(dataset_cfg)

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
    offset_samples = [torch.linspace(0, voxel_size[d], grid_cfg.subgrid_size) for d in range(3)]
    offset_samples = torch.stack(
        torch.meshgrid(*offset_samples, indexing='ij'), dim=-1).reshape(1, -1, 3)

    # Filter positions outside tilted bbox
    in_bbox_grid = None
    if bbox_coords is not None:
        midpoints = (top_left_samples + (voxel_size / 2)).float()

        # Top face clockwise: 0-3, Bottom face clockwise: 4-7
        # 3 is connected with 4
        origins = bbox_coords[[0, 4, 5, 6, 7, 4]]
        face_pt1 = bbox_coords[[1, 3, 2, 1, 0, 5]]
        face_pt2 = bbox_coords[[2, 2, 1, 0, 3, 6]]
        vecs1, vecs2 = face_pt1 - origins, face_pt2 - origins
        normals = np.stack([np.cross(u, v) for u, v in zip(vecs1, vecs2)], axis=0)

        origins = torch.tensor(origins, device=device)
        normals = torch.tensor(normals, device=device)

        def test_bbox_batch(points):
            rel_vecs = points.to(device) - origins
            dot_prods = torch.einsum('nfc, fc -> nf', rel_vecs, normals)
            in_bbox = torch.all(dot_prods >= 0, dim=1)
            return in_bbox

        in_bbox_grid = torch.empty(len(midpoints))
        logger.info('Filtering points outside 3D bounding box...')
        utils.batch_exec(test_bbox_batch, in_bbox_grid,
                         bsize=grid_cfg.voxel_bsize, progress=True)(midpoints)
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
        out = out.reshape(grid_cfg.voxel_bsize, points_per_voxel)
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
        'global_min_pt': bbox_min,
        'global_max_pt': bbox_max,
        'res': dataset_cfg.grid_res
    }
    np.savez_compressed(save_path, **save_dict)
    logger.info('Saved to "{}".'.format(save_path))


if __name__ == '__main__':
    main()
