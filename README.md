# MMAHH-Mamba

This repository contains a single-file research implementation of the method described in:

**"MMAHH-Mamba: Modality-Missing-Aware Heterogeneous Hypergraph Mamba for Robust Brain Tumor Segmentation with Incomplete Multimodal MRI"**

## Overview

Multimodal MRI provides complementary evidence for brain tumor segmentation. Missing sequences change not only the available input channels but also the reliability of cross-modal relationships and the spatial propagation of tumor information. MMAHH-Mamba implements three coupled components:

1. **Missing-aware Frequency–Spatial Hypergraph Alignment (MFH-Align).** A reconstructible multiscale Laplacian decomposition preserves low-frequency anatomy and high-frequency residuals. Modality-specific local attention with 3D rotary position encoding estimates reliability and builds a shared frequency–spatial prior. Directed heterogeneous hyperedges aggregate observed evidence and compensate for missing virtual nodes without letting virtual nodes generate the observed-source consensus.
2. **Anisotropic Topology-Guided Mamba (ATG-Mamba).** Hypergraph co-membership, directional weights and anisotropic scales modulate four selective state-space scans. Missingness-aware direction and feature gates select useful responses, followed by token retention with a threshold that decreases as more modalities are missing.
3. **Sparse-to-Dense Hypergraph Refinement (SDH-Refine).** Retained tokens are pooled into regional representations over the existing hypergraph. Prior-guided semantic backflow restores dense context at low-confidence and pruned positions before segmentation decoding.

The implementation jointly optimizes segmentation, modality-contribution calibration, scan geometry and a missingness-dependent computation budget.

## Requirements

- Python 3.10+ recommended; Python 3.11 is a suitable starting point.
- PyTorch 2.3+.
- NumPy, SciPy and NiBabel.
- Linux or Windows; an NVIDIA GPU is recommended for training.

## Quick start

Run a synthetic shape demonstration:

```bash
python MMAHH-Mamba.py demo --image-size 16 --batch-size 1 --device cuda
```

Check backward propagation and all 15 nonempty modality subsets:

```bash
python MMAHH-Mamba.py demo --device cuda --backward --all-modalities
```

Run a complete synthetic pipeline without downloading BraTS:

```bash
python MMAHH-Mamba.py smoke --device cpu --output runs/smoke
```

## Architecture configuration

| Component | Default setting in `MMAHH-Mamba.py` |
|---|---|
| Input | Four-channel 3D MRI; order T1, T1ce, T2, FLAIR |
| Output | Four-class voxel logits at the original input size |
| Shallow encoder | Four modality-specific two-layer stride-2 3D CNNs |
| Latent grid | Approximately one-quarter of input size along each axis |
| Feature width / attention heads | 24 / 2 |
| Frequency scales | 2 |
| Local candidates | Center plus six 3D face neighbors |
| Hypergraph anchors | One shared-prior anchor per latent position |
| Scan directions | Row forward/reverse and column forward/reverse per axial plane |
| SSM state dimension | 8 |
| Base retention threshold | 0.55, reduced by `0.2 × missingness` |
| Deep execution | Dense with straight-through masking in training; packed retained tokens at inference |
| Decoder | Shared 3D convolutional decoder with a masked high-resolution bypass |
| Default parameter count | 122,908 |

## Datasets

The data interface targets **BraTS 2019, BraTS 2020 and BraTS 2021** with co-registered T1, T1ce, T2 and FLAIR images and raw labels `{0, 1, 2, 4}`. Obtain the datasets from their providers; data are not redistributed here.

A suggested local organization is:

```text
data/
└── BraTS2021/
    └── BraTS2021_00000/
        ├── BraTS2021_00000_t1.nii.gz
        ├── BraTS2021_00000_t1ce.nii.gz
        ├── BraTS2021_00000_t2.nii.gz
        ├── BraTS2021_00000_flair.nii.gz
        └── BraTS2021_00000_seg.nii.gz
```
