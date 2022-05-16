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
    nodes.sort(key=(lambda node: node['idx']))

    mid_pts = torch.empty((len(nodes), 3))
    is_active = torch.zeros(len(nodes) + 1, dtype=torch.bool)

    for idx, node in enumerate(nodes):
        mid_pts[idx] = torch.from_numpy((node['min_pt'] + node['max_pt']) / 2)
        is_active[idx] = node['started']

    ckpt['mid_pts'] = mid_pts
    ckpt['active'] = is_active
    torch.save(ckpt, args.ckpt_path)


if __name__ == '__main__':
    main()
