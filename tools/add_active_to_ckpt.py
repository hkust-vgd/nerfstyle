import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('distill_ckpt_path')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path)
    distill_ckpt = torch.load(args.distill_ckpt_path)

    nodes = distill_ckpt['trained']
    is_active = torch.zeros(len(nodes) + 1, dtype=torch.bool)
    for idx, node in enumerate(nodes):
        is_active[idx] = node['started']

    ckpt['active'] = is_active
    torch.save(ckpt, args.ckpt_path)


if __name__ == '__main__':
    main()
