# RIS-PiDiNet: Rotation Invariant Symmetry-Aware Pixel Difference Network

<div align="center">

**[CVPR 2026 Main Track]**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://github.com/yuhua666/RIS-PiDiNet)
[![GitHub Stars](https://img.shields.io/github/stars/yuhua666/RIS-PiDiNet?style=social)](https://github.com/yuhua666/RIS-PiDiNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

> **RIS-PiDiNet: Geometry-Consistent Detection via Rotation Invariant Symmetry-Aware Pixel Difference Convolution**
>
> *Accepted at CVPR 2026 (Main Track)*

---

## Introduction

Remote sensing objects exhibit **arbitrary rotations** (due to overhead viewpoints) and **intrinsic structural symmetry** (e.g., bilateral aircraft, radial roundabouts). Standard convolutions lack mechanisms for either property.

**RIS-PiDiNet** is a geometry-consistent detector that explicitly integrates geometric priors into feature learning through two complementary modules:

- 🔄 **S-PDC (Symmetry-Aware Pixel Difference Convolution)** — modulates Pixel Difference Convolution with **Polar Harmonic Transform (PHT)** harmonic kernels:

$$H_i^{(n,l)} = \cos(2\pi n r_i^2 + l\theta_i)$$

combined with trainable harmonic coefficients $\alpha_{n,l}$:

$$y = \sum_{(n,l)\in\mathcal{O}} \alpha_{n,l} \sum_{i \neq c}^{N^2} w_i \cos(2\pi n r_i^2 + l\theta_i)(x_i - q)$$

- 🔁 **RIS-PDC (Rotation Invariant S-PDC)** — applies **SO(2) group averaging** over 8 discrete rotation angles to achieve full rotation invariance:

$$y_{\text{final}} = \frac{1}{8} \sum_{j=1}^{8} (R_{\theta_j}\boldsymbol{K}) * \boldsymbol{x}$$

The backbone follows a stacked **RIS-block** design inspired by [LSKNet](https://github.com/zcablii/LSKNet), augmented with lightweight **LBP** local descriptors for fine-grained structural representation.

<div align="center">
<img src="fig/Network0514.pdf" alt="RIS-PiDiNet Architecture" width="90%"/>
<br>
<em>Figure: (a) RIS-PiDiNet overview. (b) Backbone block. (c) RIS-PDC with SO(2) rotations and PHT harmonic kernels. (d) Eight-direction rotation on weights. (e) Harmonic kernel + LBP application.</em>
</div>

---

## Main Results on DOTA-v1.0

### Single-Scale Training

| Method | Pre. | RI | mAP | #Params | FLOPs |
|:-------|:----:|:--:|:---:|:-------:|:-----:|
| LSKNet-T | IN | ✗ | 74.83 | 21.0M | 124G |
| LSKNet-S | IN | ✗ | 77.49 | 31.0M | 161G |
| PIKNet-S | IN | ✗ | 78.39 | 30.8M | 190G |
| **RIS-PiDiNet-T (ours)** | IN | ✔ | 76.92 | 21.0M | 159G |
| **RIS-PiDiNet-S (ours)** | IN | ✔ | **78.53** | 31.0M | 206G |

### Multi-Scale Training

| Method | Pre. | RI | mAP | #Params | FLOPs |
|:-------|:----:|:--:|:---:|:-------:|:-----:|
| LSKNet-T | IN | ✗ | 81.37 | 21.0M | 124G |
| LSKNet-S | IN | ✗ | 81.64 | 31.0M | 161G |
| **RIS-PiDiNet-T (ours)** | IN | ✔ | 81.52 | 21.0M | 159G |
| **RIS-PiDiNet-S (ours)** | IN | ✔ | **81.81** | 31.0M | 206G |

> Pre.: IN = ImageNet-1K pretrained. RI: rotation invariant design.

---

## Environment Setup

### Requirements

- Python 3.8+
- PyTorch ≥ 1.9.0
- CUDA ≥ 10.2
- mmcv-full == 1.7.0
- mmdet == 2.28.2
- mmrotate == 0.3.4

### Step 1: Create conda environment

```bash
conda create -n ris-pidiNet python=3.8 -y
conda activate ris-pidiNet
```

### Step 2: Install PyTorch

```bash
# CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Or CUDA 11.6
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Step 3: Install mmcv-full

```bash
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

> Adjust the URL to match your CUDA/PyTorch version. See [mmcv installation docs](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

### Step 4: Install mmdet and mmrotate dependencies

```bash
pip install mmdet==2.28.2
```

### Step 5: Install this repo

```bash
git clone https://github.com/yuhua666/RIS-PiDiNet.git
cd RIS-PiDiNet
pip install -v -e .
```

### Step 6: Install other dependencies

```bash
pip install timm einops
```

---

## Dataset Preparation

### DOTA-v1.0 / DOTA-v1.5

1. Download DOTA from the [official website](https://captain-whu.github.io/DOTA/dataset.html).
2. Split the dataset using the provided scripts:

```bash
# Single-scale split
python tools/data/dota/split/img_split.py \
    --base-json tools/data/dota/split/split_configs/ss_trainval.json

# Multi-scale split
python tools/data/dota/split/img_split.py \
    --base-json tools/data/dota/split/split_configs/ms_trainval.json

# Test set split
python tools/data/dota/split/img_split.py \
    --base-json tools/data/dota/split/split_configs/ms_test.json
```

3. Organize the data as:

```
data/
└── split_ss_dota/          # or split_ms_dota for multi-scale
    ├── train/
    │   ├── images/
    │   └── annfiles/
    ├── val/
    │   ├── images/
    │   └── annfiles/
    └── test/
        ├── images/
        └── annfiles/
```

---

## Pretrained Weights

Download ImageNet-1K pretrained weights for the backbone:

| Model | Download |
|:------|:--------:|
| LSKNet-T (backbone) |  |
| LSKNet-S (backbone) |  |

Place the downloaded weights in `pretrained/`.

---

## Training

### Single GPU

```bash
python tools/train.py \
    configs/lsknet/lsk_t_fpn_1x_dota_le90-ours.py \
    --work-dir work_dirs/ris_pidiNet_t_ss
```

### Multi-GPU (recommended)

```bash
# RIS-PiDiNet-T, single-scale
bash tools/dist_train.sh \
    configs/lsknet/lsk_t_fpn_1x_dota_le90-ours.py \
    8                               # number of GPUs

# RIS-PiDiNet-S, single-scale
bash tools/dist_train.sh \
    configs/lsknet/lsk_s_fpn_1x_dota_le90-ours.py \
    8

# RIS-PiDiNet-S, with EMA (for best results)
bash tools/dist_train.sh \
    configs/lsknet/lsk_s_ema_fpn_1x_dota_le90-ours.py \
    8
```

---

## Testing & Evaluation

### Local Evaluation

```bash
python tools/test.py \
    configs/lsknet/lsk_t_fpn_1x_dota_le90-ours.py \
    work_dirs/ris_pidiNet_t_ss/latest.pth \
    --eval mAP
```

### Generate Submission Files (for DOTA test server)

```bash
python tools/test.py \
    configs/lsknet/lsk_s_fpn_1x_dota_le90-ours.py \
    work_dirs/ris_pidiNet_s_ss/latest.pth \
    --format-only \
    --eval-options submission_dir=work_dirs/submission/
```

Then zip and submit the `submission/` folder to the [DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).

---

## Key Configurations

| Config | Backbone | Scale | Description |
|:-------|:--------:|:-----:|:------------|
| `lsk_t_fpn_1x_dota_le90-ours.py` | LSKNet-T | SS | RIS-PiDiNet-T, single-scale |
| `lsk_s_fpn_1x_dota_le90-ours.py` | LSKNet-S | SS | RIS-PiDiNet-S, single-scale |
| `lsk_s_ema_fpn_1x_dota_le90-ours.py` | LSKNet-S | SS | RIS-PiDiNet-S + EMA |

---

## Method Overview

### S-PDC: Symmetry-Aware Pixel Difference Convolution

S-PDC modulates the standard PDC operator with PHT harmonic kernels $H_i^{(n,l)} = \cos(2\pi n r_i^2 + l\theta_i)$, where polar coordinates are discretized as:

$$r_i^2 = \frac{u_i^2 + v_i^2}{k^2 + \epsilon}, \qquad \theta_i = \arctan2(v_i,\ u_i + \epsilon)$$

and $k = \lfloor N/2 \rfloor$. Multiple harmonic orders are combined via trainable $\alpha_{n,l}$ coefficients, enabling the network to learn a sparse, symmetry-selective basis.

### RIS-PDC: Rotation Invariant S-PDC

RIS-PDC applies the SO(2) rotation group by generating 8 equi-spaced rotated copies of the convolution kernel and averaging the responses:

$$y_{\text{final}} = \frac{1}{8}\sum_{j=1}^{8}(R_{\theta_j}\boldsymbol{K}) * \boldsymbol{x}$$

This kernel-domain averaging ensures $\text{RIS-PDC}(R_\phi \boldsymbol{x}) \approx \text{RIS-PDC}(\boldsymbol{x})$ for arbitrary rotation $R_\phi$.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rispidiNet_cvpr2026,
  title     = {RIS-PiDiNet: Rotation Invariant Symmetry-Aware Pixel Difference Network for Remote Sensing Object Detection},
  author    = {Yuhua and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon [LSKNet](https://github.com/zcablii/LSKNet) and [MMRotate](https://github.com/open-mmlab/mmrotate). We sincerely thank the authors for their excellent open-source contributions.

- [MMRotate](https://github.com/open-mmlab/mmrotate) — the base detection framework
- [LSKNet](https://github.com/zcablii/LSKNet) — backbone architecture inspiration
- [PiDiNet](https://github.com/hellozhuo/pidinet) — Pixel Difference Convolution

---

## License

This project is released under the [MIT License](LICENSE).
