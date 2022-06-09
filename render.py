import argparse
from functools import partial
from pathlib import Path
import sys

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from config import DatasetConfig, NetworkConfig
from data import get_dataset
from nerf_lib import nerf_lib
from networks.multi_nerf import DynamicMultiNerf
from networks.single_nerf import SingleNerf
from renderer import Renderer
import utils


def main():
    """
    Load model + test set and render.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('name')
    parser.add_argument('ckpt_path')
    parser.add_argument('--mode', choices=['pretrain', 'distill', 'finetune'],
                        default='pretrain')
    parser.add_argument('--out-dir', default='./outputs')
    parser.add_argument('--occ-map')
    parser.add_argument('--out-dims', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--max-count', type=int)
    parser.add_argument('--render-trans', action='store_true')
    args, nargs = parser.parse_known_args()

    logger = utils.create_logger(__name__)
    out_dir: Path = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if next(out_dir.iterdir(), None) is not None:
        proceed = utils.prompt_bool('Output directory not empty. Clean directory?')
        if proceed:
            utils.rmtree(out_dir)
            out_dir.mkdir()
        else:
            sys.exit(1)

    net_cfg_path = None if args.mode == 'pretrain' else 'cfgs/network/kilonerf.yaml'
    net_cfg, nargs = NetworkConfig.load_nargs(net_cfg_path, nargs=nargs)
    dataset_cfg, nargs = DatasetConfig.load_nargs(args.dataset_cfg, nargs=nargs)
    if len(nargs) > 0:
        logger.error('Unrecognized arguments: ' + ' '.join(nargs))

    device = torch.device('cuda:0')
    nerf_lib.device = device
    nerf_lib.load_cuda_ext()
    nerf_lib.init_stream_pool(16)
    nerf_lib.init_magma()

    if args.mode == 'pretrain':
        model = SingleNerf(net_cfg)
    else:
        model = DynamicMultiNerf(net_cfg, dataset_cfg)
    model = model.to(device)
    logger.info('Created model ' + str(model))

    @utils.loader(logger)
    def _load(ckpt_path):
        ckpt = torch.load(ckpt_path)
        if args.mode == 'distill':
            model.load_nodes(ckpt['trained'])
            logger.info('Loaded distill checkpoint \"{}\"'.format(ckpt_path))
            return
        model.load_ckpt(ckpt)

        rng_states = ckpt['rng_states']
        np.random.set_state(rng_states['np'])
        torch.set_rng_state(rng_states['torch'])
        torch.cuda.set_rng_state(rng_states['torch_cuda'])

    _load(args.ckpt_path)
    logger.info('Loaded checkpoint \"{}\"'.format(args.ckpt_path))

    if args.occ_map is not None:
        model.load_occ_map(args.occ_map)

    test_set = get_dataset(dataset_cfg, 'test', max_count=args.max_count)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
    logger.info('Loaded ' + str(test_set))

    if args.out_dims is None:
        intr = test_set.intrinsics
        W, H = intr.w, intr.h
    else:
        W, H = args.out_dims
        intr = test_set.intrinsics.scale(W, H)

    near, far = test_set.near, test_set.far
    bg_color = dataset_cfg.bg_color
    renderer = Renderer(model, net_cfg, intr, near, far, bg_color, use_ert=True)

    @torch.no_grad()
    def render():
        for i, (_, pose) in tqdm(enumerate(test_loader), total=len(test_set)):
            frame_id = test_set.frame_str_ids[i]
            pose = pose.to(device)
            ret_flags = ['trans_map'] if args.render_trans else None
            output = renderer.render(pose, ret_flags=ret_flags)
            nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=H, w=W)
            c_map = nc2chw(output['rgb_map'])
            c_save_path = out_dir / 'frame_{}.png'.format(frame_id)
            torchvision.utils.save_image(c_map, c_save_path)

            if args.render_trans:
                t_map = nc2chw(output['trans_map'])
                t_save_path = out_dir / 'trans_{}.png'.format(frame_id)
                torchvision.utils.save_image(t_map, t_save_path)

    try:
        render()
    except KeyboardInterrupt:
        logger.info('Rendering interrupted')
    finally:
        nerf_lib.destroy_stream_pool()
        nerf_lib.deinit_multimatmul_aux_data()
        logger.info('Closed')


if __name__ == '__main__':
    main()
