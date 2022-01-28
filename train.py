import torch

from nerf_lib import NerfLib
from networks.nerf import Nerf

if __name__ == '__main__':
    conf = {
        'x_enc_count': 10,
        'd_enc_count': 4
    }

    nerf_lib = NerfLib(conf)
    tmp_x, tmp_d = torch.rand(8, 3), torch.rand(8, 3)
    enc_x = nerf_lib.embed_x(tmp_x)
    enc_d = nerf_lib.embed_d(tmp_d)
    print(enc_x.shape, enc_d.shape)

    # model = Nerf(63, 27, 8, 256, [256, 128], [5])
    # model = Nerf(63, 27, 2, 32, [32, 32])
    # tmp_x, tmp_d = torch.rand(8, 63), torch.rand(8, 27)
    # tmp_c, tmp_a = model(tmp_x, tmp_d)
    # print(tmp_c.shape)
