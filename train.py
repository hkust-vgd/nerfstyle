import torch

from nerf_lib import NerfLib
from ray_batch import RayBatch
from networks.nerf import Nerf
from data.nsvf_dataset import NSVFDataset


def batch(*tensors, bsize=1):
    for i in range(0, len(tensors[0]), bsize):
        yield (t[i:i+bsize] for t in tensors)


def train():
    conf = {
        'x_enc_count': 10,
        'd_enc_count': 4,
        'num_rays_per_batch': 1024,
        'num_samples_per_ray': 384,
        'network_chunk_size': 65536
    }

    nerf_lib = NerfLib(conf)

    # Randomly sample image / pose from dataset
    dataset = NSVFDataset('/home/hwpang/datasets/nsvf/Synthetic_NeRF/Chair', 'train')
    tmp_img, tmp_pose = dataset[0]

    # Generate rays (filter center only?)
    target, rays_o, rays_d = nerf_lib.generate_rays(dataset.intrinsics, tmp_img, tmp_pose)
    rays = RayBatch(rays_o, rays_d, dataset.near, dataset.far)

    # Render rays
    pts = nerf_lib.sample_points(rays)
    dirs = rays.viewdirs()

    pts_flat = pts.reshape(-1, 3)
    pts_embedded = nerf_lib.embed_x(pts_flat)
    dirs_embedded = nerf_lib.embed_d(dirs)
    dirs_embedded = torch.repeat_interleave(dirs_embedded, repeats=conf['num_samples_per_ray'], dim=0)

    model = Nerf(63, 27, 8, 256, [256, 128], [5])

    rgbs, densities = [], []
    for pts_batch, dirs_batch in batch(pts_embedded, dirs_embedded, bsize=conf['network_chunk_size']):
        out_c, out_a = model(pts_batch, dirs_batch)
        rgbs.append(out_c)
        densities.append(out_a)

    rgbs = torch.concat(rgbs, dim=0).reshape(pts.shape)
    densities = torch.concat(densities, dim=0).reshape(pts.shape[:-1])
    print(rgbs.shape, densities.shape)


if __name__ == '__main__':
    train()
