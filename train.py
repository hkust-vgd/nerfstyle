import torch
from torch.utils.data import DataLoader

from nerf_lib import NerfLib
from ray_batch import RayBatch
from networks.nerf import Nerf
from data.nsvf_dataset import NSVFDataset


def batch(*tensors, bsize=1):
    for i in range(0, len(tensors[0]), bsize):
        yield (t[i:i+bsize] for t in tensors)


def compute_psnr(loss):
    psnr = -10. * torch.log(loss) / torch.log(torch.FloatTensor([10.]).to(loss.device))
    return psnr


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


class PretrainTrainer:
    def __init__(self):
        self.conf = {
            'x_enc_count': 10,
            'd_enc_count': 4,
            'num_rays_per_batch': 1024,
            'num_samples_per_ray': 384,
            'network_chunk_size': 65536,
            'precrop_fraction': 0.5,
            'precrop_iterations': 10000,
            'initial_learning_rate': 0.0005,
            'learning_rate_decay_rate': 500
        }

        self.device = torch.device('cuda:0')
        self.lib = NerfLib(self.conf, self.device)

        # Initialize model
        x_enc_channels = 6 * self.conf['x_enc_count'] + 3
        d_enc_channels = 6 * self.conf['d_enc_count'] + 3
        self.model = Nerf(x_enc_channels, d_enc_channels, 8, 256, [256, 128], [5]).to(self.device)

        # Initialize dataset
        self.train_set = NSVFDataset('/home/hwpang/datasets/nsvf/Synthetic_NeRF/Chair', 'train')
        self.train_loader = cycle(DataLoader(self.train_set, batch_size=None, shuffle=True))

        # Initialize optimizer
        self.optim = torch.optim.Adam(params=self.model.parameters(),
                                      lr=self.conf['initial_learning_rate'], betas=(0.9, 0.999))

        self.iter_ctr = 0

    @staticmethod
    def calc_loss(rendered, target):
        mse_loss = torch.mean((rendered - target) ** 2)
        return mse_loss

    def check_interval(self, interval):
        return (self.iter_ctr % interval) == 0

    def print_status(self, loss):
        # TODO: Use logging class
        log_str = '[TRAIN] Iter: {:d}, Loss: {:.5f}, PSNR: {:.5f}'
        print(log_str.format(self.iter_ctr, loss.item(), compute_psnr(loss).item()))

    def run_iter(self):
        img, pose = next(self.train_loader)

        # Generate rays
        precrop = self.conf['precrop_fraction'] if self.iter_ctr < self.conf['precrop_iterations'] else None
        rays_o, rays_d, coords = self.lib.generate_rays(self.train_set.intrinsics, pose, precrop=precrop)
        target = torch.FloatTensor(img[coords]).to(self.device)
        rays = RayBatch(rays_o, rays_d, self.train_set.near, self.train_set.far)

        # Render rays
        pts, dists = self.lib.sample_points(rays)
        dirs = rays.viewdirs()

        pts_flat = pts.reshape(-1, 3)
        pts_embedded = self.lib.embed_x(pts_flat)
        dirs_embedded = self.lib.embed_d(dirs)
        dirs_embedded = torch.repeat_interleave(dirs_embedded, repeats=self.conf['num_samples_per_ray'], dim=0)

        rgbs, densities = [], []
        for pts_batch, dirs_batch in batch(pts_embedded, dirs_embedded, bsize=self.conf['network_chunk_size']):
            out_c, out_a = self.model(pts_batch, dirs_batch)
            rgbs.append(out_c)
            densities.append(out_a)

        rgbs = torch.concat(rgbs, dim=0).reshape(pts.shape)
        densities = torch.concat(densities, dim=0).reshape(pts.shape[:-1])
        rgb_map = self.lib.integrate_points(dists, rgbs, densities)
        loss = self.calc_loss(rendered=rgb_map, target=target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.check_interval(100):
            self.print_status(loss)

        self.iter_ctr += 1

    def run(self, niters=200000):
        while self.iter_ctr < niters:
            self.run_iter()


if __name__ == '__main__':
    trainer = PretrainTrainer()
    trainer.run()
