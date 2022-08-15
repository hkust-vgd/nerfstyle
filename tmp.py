import torch

from networks.tcnn_nerf import TCNerf


def main():
    device = torch.device('cuda')
    test_net = TCNerf().to(device)

    dummy_pts = torch.zeros((1024, 3), device=device)
    dummy_dirs = torch.zeros((1024, 3), device=device)
    outs = test_net(dummy_pts, dummy_dirs)

    for foo in outs:
        print(foo.shape)


if __name__ == '__main__':
    main()
