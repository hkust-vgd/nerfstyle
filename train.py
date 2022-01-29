import torch

from nerf_lib import NerfLib
from networks.nerf import Nerf
from data.nsvf_dataset import NSVFDataset

if __name__ == '__main__':
    conf = {
        'x_enc_count': 10,
        'd_enc_count': 4,
        'num_rays_per_batch': 1024,
    }

    nerf_lib = NerfLib(conf)

    dataset = NSVFDataset('/home/hwpang/datasets/nsvf/Synthetic_NeRF/Chair', 'train')
    tmp_img, tmp_pose = dataset[0]
    target, rays_o, rays_d = nerf_lib.generate_rays(dataset.intrinsics, tmp_img, tmp_pose)

    # model = Nerf(63, 27, 8, 256, [256, 128], [5])
    # model = Nerf(63, 27, 2, 32, [32, 32])
    # tmp_x, tmp_d = torch.rand(8, 63), torch.rand(8, 27)
    # tmp_c, tmp_a = model(tmp_x, tmp_d)
    # print(tmp_c.shape)
