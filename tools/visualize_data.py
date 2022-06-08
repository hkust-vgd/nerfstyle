import argparse
import numpy as np
import json
import torch
from tqdm import tqdm

import __init__
from common import Intrinsics
from config import DatasetConfig
from data import load_bbox
from nerf_lib import nerf_lib


def load_replica(cfg: DatasetConfig):
    camera_t = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pose_t = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    flen = 256.

    poses_list = []
    for i in cfg.replica_cfg.traj_ids:
        pose_path = cfg.root_path / 'train' / '{:02d}'.format(i) / 'cameras.json'
        with open(pose_path) as f:
            poses_list += json.load(f)

    poses = np.stack([p['Rt'] for p in poses_list], axis=0)
    for i in range(len(poses)):
        R, t = np.copy(poses[i, :3, :3]), np.copy(poses[i, :3, 3])
        poses[i, :3, 3] = -np.matmul(R.T, t)
        poses[i, :3, :3] = np.matmul(camera_t, R).T

    poses = np.einsum('ij, njk -> nik', pose_t, poses)[:, :3]
    intr = Intrinsics(512, 512, flen, flen, 256.0, 256.0)

    return poses[:, :3], intr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    nerf_lib.device = device
    cfg = DatasetConfig.load(args.dataset_cfg)
    poses, intr = load_replica(cfg)
    bbox = load_bbox(cfg).to(device)

    max_scale = 0.
    mid_pt = bbox.mid_pt()
    box_size = bbox.size() / 2

    for pose in tqdm(poses):
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
        rays, _ = nerf_lib.generate_rays(pose_tensor, intr)
        coeffs = torch.ones((len(rays), 1), device=device) * cfg.replica_cfg.far
        far_pts = rays.lerp(coeffs).squeeze(1)
        _max_scale = torch.max((far_pts - mid_pt) / box_size)
        max_scale = max(max_scale, _max_scale.item())

    print(max_scale)


if __name__ == '__main__':
    main()
