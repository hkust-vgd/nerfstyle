# LLFF Dataset preparation

The LLFF dataset folder should follow the structure below:

```
nerf_llff_data
├ room
│  ├ images_8
│  ├ transforms_train.json
│  ├ transforms_val.json
│  ├ transforms_test.json
│  └ seg
├ fern
│  └ ...
└ ...
```

`images_8` are training images in the original LLFF dataset. The other files are additionally required for stylization training, which is explained below:

- `transforms_***.json` are files that describe the camera poses of LLFF datasets in Euclidean
space instead of NDC space.
- `seg` are numpy segmentation maps used to train the classification network.
  - Each training image in `image_8` has a corresponding NPZ segmentation map and a 
  visualization in PNG.

Please download them via the links below and move them to the correct location:
- `room`
- `fern` (TBU)
- `trex` (TBU)

After preparation, modify the `root_path` in the corresponding config file in `cfgs/dataset`.
