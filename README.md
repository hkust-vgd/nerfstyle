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

## Usage

**Render a pretrained model**

```bash
python render.py <cfg_path> <out_folder_name> <ckpt_path>

# Example: scene `room1` from Replics, render at 
python render.py cfg
```

**Training models on datasets**

To be updated later
