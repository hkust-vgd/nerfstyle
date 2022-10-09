import argparse
from functools import partial
from pathlib import Path
import sys

import einops
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from data import get_dataset
from nerf_lib import nerf_lib
from trainers.base import Trainer
import utils


def main():
    """
    Load model + test set and render.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('--name')
    parser.add_argument('--out-dir', default='./outputs')
    parser.add_argument('--out-dims', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--max-count', type=int)
    # parser.add_argument('--render-trans', action='store_true')
    args = parser.parse_args()

    logger = utils.create_logger(__name__)
    device = torch.device('cuda:0')
    nerf_lib.device = device

    # Load renderer (with model)
    ckpt_trainer = Trainer.load_ckpt(args.ckpt_path)
    renderer = ckpt_trainer.renderer
    ema = ckpt_trainer.ema

    # Setup output environment
    if args.name is None:
        # default experiment name
        args.name = ckpt_trainer.name + '_' + Path(args.ckpt_path).stem
        if args.out_dims is not None:
            args.name += '_{:d}x{:d}'.format(*args.out_dims)
    out_dir: Path = Path(args.out_dir) / args.name
    logger.info('Writing to directory "{}"'.format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if next(out_dir.iterdir(), None) is not None:
        proceed = utils.prompt_bool('Output directory not empty. Clean directory?')
        if proceed:
            utils.rmtree(out_dir)
            out_dir.mkdir()
        else:
            sys.exit(1)

    # Setup dataset
    test_set = get_dataset(ckpt_trainer.dataset_cfg, 'test', max_count=args.max_count)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)
    logger.info('Loaded ' + str(test_set))

    # Modify camera parameters if needed
    if args.out_dims is None:
        W, H = renderer.intr.w, renderer.intr.h
    else:
        W, H = args.out_dims
        renderer.intr = renderer.intr.scale(W, H)

    @torch.no_grad()
    def render():
        for i, (_, pose) in tqdm(enumerate(test_loader), total=len(test_set)):
            frame_id = test_set.frame_str_ids[i]
            pose = pose.to(device)
            # ret_flags = ['trans_map'] if args.render_trans else None

            with torch.cuda.amp.autocast(enabled=ckpt_trainer.train_cfg.enable_amp):
                with ema.average_parameters():
                    output = renderer.render(pose)

            nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=H, w=W)
            c_map = nc2chw(output['rgb_map'])
            c_save_path = out_dir / 'frame_{}.png'.format(frame_id)
            torchvision.utils.save_image(c_map, c_save_path)

            # if args.render_trans:
            #     t_map = nc2chw(output['trans_map'])
            #     t_save_path = out_dir / 'trans_{}.png'.format(frame_id)
            #     torchvision.utils.save_image(t_map, t_save_path)

    try:
        render()
    except KeyboardInterrupt:
        logger.info('Rendering interrupted')
    finally:
        logger.info('Closed')


if __name__ == '__main__':
    main()
