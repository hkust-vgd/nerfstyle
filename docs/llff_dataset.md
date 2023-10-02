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
│  │  └ seg (*)
│  ├ fern
│  │  └ ...
│  └ ...
└ ...

(*) Needs to be downloaded externally
```

### Images

`images_8` are training images in the original LLFF dataset. Please download the LLFF dataset, and copy the `images_8` folder to `./datasets/nerf_llff_data/<scene>/`.

### Segmentation

`seg` are numpy segmentation maps used to train the classification network. Each training image in `image_8` has a corresponding NPZ segmentation map and a visualization in PNG.

Please download them via the links below and move them to the correct location:

- [`room`](https://www.dropbox.com/scl/fi/5viu1eghd9y8am7y1mokv/llff_room_seg.zip?rlkey=bif24y476wfelmoyi5gueb7kj&dl=0)
- `fern` (to be updated)
- `trex` (to be updated)
