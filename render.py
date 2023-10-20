import argparse
from functools import partial
from pathlib import Path
import sys

import einops
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from common import DatasetSplit
from data import get_dataset
from nerf_lib import nerf_lib
from networks.style_nerf import StyleTCNerf
from renderer import Renderer
import utils


def main():
    """
    Load model + test set and render.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('--name')
    parser.add_argument('--out-dir', default='./outputs')
    parser.add_argument('--out-dims', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--max-count', type=int)
    args = parser.parse_args()

    logger = utils.create_logger(__name__)
    device = torch.device('cuda:0')
    nerf_lib.device = device

    ckpt_state_dict = torch.load(args.ckpt)

    # Setup output environment
    if args.name is None:
        # default experiment name
        ckpt_path = Path(args.ckpt)
        args.name = '_'.join(ckpt_path.parts[-2:]).replace(ckpt_path.suffix, '')

        if args.out_dims is not None:
            args.name += '_{:d}x{:d}'.format(*args.out_dims)

    out_dir = Path(args.out_dir) / args.name
    logger.info('Writing to directory "{}"'.format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if next(out_dir.iterdir(), None) is not None:
        proceed = utils.prompt_bool('Output directory not empty. Clean directory?')
        if proceed:
            utils.rmtree(out_dir)
            out_dir.mkdir()
        else:
            sys.exit(1)

    # Load dataset
    _tmp = get_dataset(ckpt_state_dict['dataset_cfg'], split=DatasetSplit.TRAIN)
    num_classes = _tmp.num_classes
    del _tmp

    test_set = get_dataset(ckpt_state_dict['dataset_cfg'], split=DatasetSplit.TEST)
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)

    # Initialize model and renderer
    model = StyleTCNerf(ckpt_state_dict['net_cfg'], test_set.bbox,
                        num_classes, torch.float16, use_dir=False)
    logger.info('Created model ' + str(type(model)))

    renderer = Renderer(
        model, ckpt_state_dict['render_cfg'], test_set.intr,
        ckpt_state_dict['dataset_cfg'].bound,
        precrop_frac=ckpt_state_dict['train_cfg'].precrop_fraction,
        raymarch_channels=(3 + num_classes)
    ).to(device)

    # Modify camera parameters if needed
    if args.out_dims is None:
        W, H = renderer.intr.size()
    else:
        W, H = args.out_dims
        renderer.intr = renderer.intr.scale(W, H)

    # Load state dict from ckpt
    renderer.load_state_dict(ckpt_state_dict['renderer'])
    logger.info('Loaded checkpoint \"{}\"'.format(args.ckpt))

    @torch.no_grad()
    def render():
        for i, (_, pose) in tqdm(enumerate(test_loader), total=len(test_set)):
            frame_id = test_set.fns[i]
            pose = pose.to(device)

            with torch.cuda.amp.autocast(enabled=ckpt_state_dict['train_cfg'].enable_amp):
                output = renderer.render(pose)

            nc2chw = partial(einops.rearrange, pattern='(h w) c -> c h w', h=H, w=W)
            c_map = nc2chw(output['rgb_map'])
            c_save_path = out_dir / '{}.png'.format(frame_id)
            torchvision.utils.save_image(c_map, c_save_path)

    try:
        render()
    except KeyboardInterrupt:
        logger.info('Rendering interrupted')
    finally:
        logger.info('Closed')


if __name__ == '__main__':
    main()
