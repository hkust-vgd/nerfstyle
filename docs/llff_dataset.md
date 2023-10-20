# LLFF Dataset preparation

By default, the code looks for LLFF data in the directory `./datasets/nerf_llff_data`. Ideally, it should have the structure below:

```
datasets
├ nerf_llff_data
│  ├ room
│  │  ├ transforms_train.json
│  │  ├ transforms_val.json
│  │  ├ transforms_test.json
│  │  ├ images_8 (*)
│  │  ├ seg (*)
│  │  └ seg_cops (**)
│  ├ fern
│  │  └ ...
│  └ ...
└ ...

(*) Needs to be downloaded externally
(**) Optional
```

### Images

`images_8` are training images in the original LLFF dataset. Please download the LLFF dataset, and copy the `images_8` folder to `./datasets/nerf_llff_data/<scene>/`.

### Segmentation

`seg` are numpy segmentation maps used to train the classification network. Each training image in `image_8` has a corresponding NPZ segmentation map and a visualization in PNG.

Please download them via the links below and move them to the correct location:

**Clustering unsupervised segmentation**

Segmentation based on the paper *Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering* by Kim et. al. ([Github code](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip))

Code is slightly altered to perform segmentation on multiple images. This is the approach used in the paper.

- [`room`](https://www.dropbox.com/scl/fi/bi80ocawmxxf9g65e09fg/llff_room_seg.zip?rlkey=vzknloidkaer1lbwo6o5qjfk0&dl=0)
- [`fern`](https://www.dropbox.com/scl/fi/6m2v9glmi5h1rs3r7dw0y/llff_fern_seg.zip?rlkey=irllez5ecb736tjx3nots94s8&dl=0)
- [`trex`](https://www.dropbox.com/scl/fi/rqyc7ljhu4wjjk8w1xm8f/llff_trex_seg.zip?rlkey=fe1qx1c1y7igqdo155jvvip3b&dl=0)

**Segmentation via COPS**

Segmentation based on the paper *Combinatiorial Optimization for Panoptic Segmentation* (COPS) by Abbas et. al. ([Github code](https://github.com/aabbas90/COPS))

Scene is divided into finer regions based on the scene objects.

- [`room`](https://www.dropbox.com/scl/fi/pbcajd5rirvh1x04gn2h8/llff_room_seg_cops.zip?rlkey=tg17egxu5vcln3h2yj3k3cvj3&dl=0)
- `fern` (to be updated)
- `trex` (to be updated)

To use these segmentation maps instead, specify `--seg-name seg_cops`:
```bash
python train.py --log-dir runs/room_base_cops --data-cfg cfgs/dataset/llff_room.yaml --seg-name seg_cops
```
