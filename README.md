# NeRF Renderer



## Description

A NeRF renderer written from scratch.
- Implementation based from [KiloNeRF](https://github.com/creiser/kilonerf) (Reiser et al.)
  - Uses a composition of multiple NeRF models to speed up inference
  - Helps break down large scenes into smaller, local radiance fields
- Optimized with CUDA extensions
- Supports style transfer training; can render alternate appearances based on a style image

## Roadmap

- [x] Pretrained model on indoor scenes from [Replica](https://github.com/facebookresearch/Replica-Dataset)
- [x] Multi-model rendering via KiloNeRF
- [x] Ensure model compactness with sparsity loss
- [ ] Pretrained models on more datasets (e.g. Tanks & Temples)
- [ ] Accelerate training by parametric encoding
- [ ] Distill to PlenOctrees for real-time rendering

## Setup

**Code and CUDA extension**

1. Clone repository.
2. Edit PyTorch / CUDA version in `environment.yml`, if necessary.
   - The code has been tested on Python 3.7 + PyTorch 1.11 + CUDA 11.3 only.
3. Install new conda environment: `conda env create -f environment.yml`.
4. Install MAGMA:
   - Head to the [relevant section](https://github.com/creiser/kilonerf#option-b-build-cuda-extension-yourself) in the KiloNeRF repository.
   - Follow the instructions up to `sudo -E make install`.
   - The extension will be compiled at runtime with Ninja.

**Replica dataset**

1. Download frames / trajectories for Replica at the repository for [Generative Scene Networks](https://github.com/apple/ml-gsn#datasets) (DeVries et al.).
2. Extract to desired location.
3. Copy the `data/bboxes` directory to the Replica root directory. The folder structure should be as follows:

```
replica_all/
├─ bboxes/
├─ test/
└─ train/
```

4. Change `root_path` in the config files (`./cfgs/dataset/replica_<scene_name>.yaml`)

## Trained models

Trained models (KiloNeRF) + occupancy grids are provided below for testing.

|Dataset|Scene|DL link|Size|
|---|---|---|---|
|Synthetic_NeRF|`chair`|[Dropbox](https://www.dropbox.com/s/ye6joiw5n55wdqb/nerf_chair.tar.gz?dl=0)|78.83 MB|
|Synthetic_NeRF|`lego`|||
|Replica|`office4`|[Dropbox](https://www.dropbox.com/s/7817p9eg8u2v2y0/replica_office4.tar.gz?dl=0)|498.36 MB|
|Replica|`room1`|[Dropbox](https://www.dropbox.com/s/2lj420du7voqzlp/replica_room1.tar.gz?dl=0)|388.89 MB|
|Replica|`room2`|||

## Usage

**Render a finetuned KiloNeRF model**

> ```bash
> python render.py <cfg_path> <out_folder_name> <ckpt_path> --mode finetune --occ-map <occ_map_path> --max-count <max_count> --out-dims <W> <H>
> ```

- `cfg_path`: Path to config file, located in `cfgs`
- `out_folder_name`: Name of output folder, generated in `outputs`
- `ckpt_path`: Checkpoint path (pretrained DL link above)
- `--occ-map`: Occupancy map path (pretrained DL link above)
- `--max-count`: No. of images to render
  - Exclude flag to render all images in test set
- `--out-dims`: Output dimension
  - Exclude flag to render at default dataset dimension

> ```bash
> # Example: Synthetic-NeRF "lego" scene
> python render.py cfgs/dataset/nerf_lego.yaml lego <ckpt_path> --mode finetune --occ-map <occ_map_path> --max-count 100
> 
> # Example: Replica "room1" scene in 1024x768 resolution
> python render.py cfgs/dataset/replica_room1.yaml room1 <ckpt_path> --mode finetune --occ-map <occ_map_path> --max-count 100 --out-dims 1024 768
> ```

**Training models on datasets**

Instructions to be provided later.
