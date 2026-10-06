# MMAHH-Mamba

This repository contains a single-file research implementation of the method described in:

**"MMAHH-Mamba: Modality-Missing-Aware Heterogeneous Hypergraph Mamba for Robust Brain Tumor Segmentation with Incomplete Multimodal MRI"**

This is an executable initial implementation based on the manuscript's method description. It is not a verified reproduction of the reported experiments, and no pretrained BraTS weights are included.

## Overview

Multimodal MRI provides complementary evidence for brain tumor segmentation. Missing sequences change not only the available input channels but also the reliability of cross-modal relationships and the spatial propagation of tumor information. MMAHH-Mamba implements three coupled components:

1. **Missing-aware Frequency–Spatial Hypergraph Alignment (MFH-Align).** A reconstructible multiscale Laplacian decomposition preserves low-frequency anatomy and high-frequency residuals. Modality-specific local attention with 3D rotary position encoding estimates reliability and builds a shared frequency–spatial prior. Directed heterogeneous hyperedges aggregate observed evidence and compensate for missing virtual nodes without letting virtual nodes generate the observed-source consensus.
2. **Anisotropic Topology-Guided Mamba (ATG-Mamba).** Hypergraph co-membership, directional weights and anisotropic scales modulate four selective state-space scans. Missingness-aware direction and feature gates select useful responses, followed by token retention with a threshold that decreases as more modalities are missing.
3. **Sparse-to-Dense Hypergraph Refinement (SDH-Refine).** Retained tokens are pooled into regional representations over the existing hypergraph. Prior-guided semantic backflow restores dense context at low-confidence and pruned positions before segmentation decoding.

The implementation jointly optimizes segmentation, modality-contribution calibration, scan geometry and a missingness-dependent computation budget.

## Repository structure

```text
MMAHH-Mamba/
├── MMAHH-Mamba.py
├── README.md
└── requirements.txt
```

All model components, data loading, training, evaluation and inference commands are contained in `MMAHH-Mamba.py`. It does not require the earlier multi-file package or a separate configuration directory. Running training creates local `runs/` outputs; these generated data and checkpoints are not part of the three-file source release.

## Requirements

- Python 3.10+ recommended; Python 3.11 is a suitable starting point.
- PyTorch 2.3+.
- NumPy, SciPy and NiBabel.
- Linux or Windows; an NVIDIA GPU is recommended for training.

Create an environment and install the dependencies from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU training, install a PyTorch wheel compatible with your NVIDIA driver in this environment before installing the remaining requirements. Check the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The default `torch` scan backend runs on CPU or CUDA and requires no custom extension. The optional `--backend cuda` uses the compiled selective-scan extension from `mamba-ssm`; install that extension separately using its upstream instructions. It is not needed to run the examples below. The portable backend is a readable reference recurrence, not an optimized inference kernel.

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

The smoke command creates 10 tiny synthetic volumes, generates five patient folds, runs two training updates, saves and reloads the model, and evaluates two held-out cases under all 15 subsets. These outputs test execution only; synthetic Dice scores are not paper results.

Use `--device cpu` where CUDA is unavailable. Run `python MMAHH-Mamba.py --help` for commands, or append `--help` after a command for its options.

Because the requested filename contains a hyphen, load it from another Python script with `importlib`:

```python
import sys
import torch
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("mmahh_mamba_single", "MMAHH-Mamba.py")
mmahh = module_from_spec(spec)
sys.modules[spec.name] = mmahh
spec.loader.exec_module(mmahh)

model = mmahh.build_mmahh_mamba(width=24, heads=2, state_size=8)
images = torch.randn(1, 4, 16, 16, 16)
# Fixed channel order: T1, T1ce, T2, FLAIR. T1ce is absent here.
available = torch.tensor([[1.0, 0.0, 1.0, 1.0]])

model.eval()
with torch.no_grad():
    logits, auxiliary = model(images, available, return_aux=True)
    probabilities = logits.softmax(dim=1)
    segmentation = probabilities.argmax(dim=1)
    reliability = auxiliary["reliability"]
    retained_tokens = auxiliary["hard_keep"]
```

Without `return_aux=True`, the forward method returns only the logits tensor. The logits have shape `[B, 4, D, H, W]`, where the four output classes are background, non-enhancing/necrotic core, edema and enhancing tumor after internal BraTS label remapping. At least one modality must be observed per case. Absent channels are explicitly masked before any image encoder or decoder bypass uses them.

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

These are explicit reference settings. The manuscript does not provide a complete layer-by-layer architecture specification. This release's parameter count is therefore not the manuscript's reported 9.18 M, and its FLOPs and latency must be measured independently.

## Frequency and missing-modality design

For each level of the shallow feature pyramid:

\[
L^{s}=\operatorname{Down}(L^{s-1}),\qquad
H^{s}=L^{s-1}-\operatorname{Up}(L^{s}).
\]

The preceding feature is recoverable as:

\[
L^{s-1}=\operatorname{Up}(L^{s})+H^{s}.
\]

The model keeps all residual branches and learns their contribution. Reconstructibility refers to this decomposition, not to invertibility of the later nonlinear feature fusion or final segmentation network.

During training, global random modality dropping and local feature-grid masking regularize the network. At inference, only real modality availability is used. A missing virtual node can receive regional evidence but cannot become a source for the observed-node hyperedge consensus.

Four-direction scan order is preserved; hypergraph topology changes the state-update strength through direction weights, co-membership and spatial scales. This follows the detailed Section 3.3 formulation rather than the alternative global-reordering wording in the overview.

Token pruning only reduces the subsequent deep SSM/feed-forward branch. The first four scans remain dense. At inference, retained tokens are physically gathered and processed before being scattered back; zeroing a dense tensor alone would not save computation. Python-level scheduling may offset arithmetic savings, so this release does not claim a measured speedup.

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

Nonzero voxels are normalized per modality. Image and label shapes/affines are checked; the program does not silently register or resample mismatched volumes. NIfTI axes and affine are preserved. Label 4 is mapped internally to class 3 and mapped back to 4 for exported segmentations.

Prepare a manifest and fixed folds:

```bash
python MMAHH-Mamba.py prepare --root data/BraTS2021 --output data/brats2021.json
python MMAHH-Mamba.py splits --manifest data/brats2021.json --output data/splits2021.json --seed 2026
```

Each outer fold has disjoint training, validation and test patients. Checkpoint selection uses only the validation subset. If combining years, resolve repeat patients before generating folds; patient identity across years cannot be inferred from filename alone.

## Training protocol

Built-in configurations are named `reference` and `smoke`. No extra config file is required:

```bash
python MMAHH-Mamba.py train --config reference \
  --manifest data/brats2021.json --splits data/splits2021.json \
  --fold 0 --device cuda --output runs/fold0
```

To customize the configuration:

```bash
python MMAHH-Mamba.py config --preset reference --output my_config.json
# Edit my_config.json, then pass --config my_config.json to train.
```

| Item | Reference implementation setting |
|---|---|
| Framework | PyTorch; no MONAI dependency |
| Patch size | `96 × 96 × 96` |
| Batch size | 1; manuscript states 4, which requires separate profiling |
| Epochs | 300 |
| Optimizer | AdamW |
| Initial / minimum learning rate | `2e-4 / 1e-6` |
| Weight decay | `0.01` |
| Schedule | 5% linear warmup, then cosine decay |
| Augmentation | Foreground-aware crops, flips, small 3D rotations, intensity scaling/shifts and weak noise |
| Modality / local feature drop | 0.25 / 0.1 |
| Auxiliary weights | Contribution 0.1, geometry 0.01, budget 0.05 |
| Auxiliary warmup | First 10 epochs |
| Target retention range | 0.45–0.85 according to missingness |
| Validation / inference | Sliding windows with Gaussian blending and 0.5 overlap |

Unspecified values above are implementation choices, not recovered experimental settings. The objective is:

\[
\mathcal{L}=\mathcal{L}_{\mathrm{Dice}}+\mathcal{L}_{\mathrm{CE}}
+\lambda_c\mathcal{L}_{\mathrm{con}}
+\lambda_g\mathcal{L}_{\mathrm{geo}}
+\lambda_b\mathcal{L}_{\mathrm{budget}}.
\]

Counterfactual contribution calibration computes all removable observed modalities, costing up to four additional no-gradient forward passes. A singleton case is excluded from this calibration because removing its only observation would create an invalid empty acquisition. The manuscript's one-sampled-modality wording does not specify how to estimate the full contribution distribution; this implementation uses explicit full calculations rather than assigning invented values to unmeasured modalities.

The budget loss is a squared retention error. Geometry targets are detached. Empty retained hyperedges use the prior and existing consensus instead of dividing by zero. Training uses a dense straight-through approximation for the deep branch, whereas inference uses packed sequences; the resulting context difference requires real-data validation.

Resume your own trusted checkpoint with the same partition and configuration:

```bash
python MMAHH-Mamba.py train --config reference \
  --manifest data/brats2021.json --splits data/splits2021.json \
  --fold 0 --device cuda --output runs/fold0 --resume runs/fold0/last.pt
```

Checkpoints include optimizer/scaler and random-generator state. Load only trusted checkpoint files. Mixed-precision overflow skips are logged as `amp_skipped`, and an epoch without a successful optimizer update fails explicitly.

## Evaluation and inference

Evaluate one held-out fold under all 15 nonempty modality subsets with the same checkpoint:

```bash
python MMAHH-Mamba.py evaluate --checkpoint runs/fold0/best.pt \
  --manifest data/brats2021.json --splits data/splits2021.json \
  --fold 0 --device cuda --output runs/fold0/eval
```

Repeat training and evaluation for folds 0–4. `cases.csv` stores patient/combination/region scores; `summary.json` stores per-combination and per-available-modality-count summaries. Requested combinations requiring a genuinely unavailable sequence are reported as skipped, not silently relabeled as complete observations.

For unlabeled inference, generate a manifest using `prepare --allow-missing-label`, then:

```bash
python MMAHH-Mamba.py predict --checkpoint runs/fold0/best.pt \
  --manifest data/inference.json --modalities t1 t2 flair \
  --device cuda --output runs/predictions
```

Omit `--modalities` to use all actually observed sequences. Predictions are saved as NIfTI volumes with the original shape and affine.

Metrics use ET = class 3, TC = classes 1+3, and WT = all foreground after internal label mapping:

- Dice: both masks empty → 1; only one empty → 0.
- Sensitivity: empty ground truth → undefined.
- HD95: maximum of the two directed 95th-percentile surface distances in physical millimeters. Both masks empty → 0; only one empty → infinity. Finite averages are accompanied by counts of infinite cases.

## Validation status

The underlying implementation passed CPU/GPU synthetic training, checkpoint reload, all-subset evaluation and NIfTI export checks. A 96³ synthetic CUDA float16 forward/backward optimizer step was also exercised. These checks establish execution, not medical accuracy or convergence.

The standalone release was additionally checked with `demo --backward --all-modalities` on CUDA and `smoke` on both CPU and CUDA, without importing the earlier package. Each smoke run produced 90 patient/combination/region rows. CUDA scatter reductions may differ slightly between runs; the synthetic invariance check uses an explicit floating-point tolerance.

Local validation used Windows, Python 3.9.13 and PyTorch 2.6.0+cu124 on an NVIDIA RTX 4060 Laptop GPU. Linux commands are provided, but no Linux run was performed locally. The pre-existing local SciPy/NumPy pairing emitted a compatibility warning; use a clean environment satisfying `requirements.txt` for training rather than copying that old pairing.

Real BraTS training, the manuscript's reported Dice/HD95/Sensitivity, its 9.18 M parameter configuration, FLOPs and latency have **not** been reproduced by this release. The optional compiled Mamba backend has not been locally validated. No manuscript result table is presented as a measured result of this code.

## Mapping from the paper to the code

| Paper operation | Implementation |
|---|---|
| Reconstructible Laplacian decomposition | `laplacian_pyramid`, `reconstruct_pyramid` |
| Local heterogeneous reliability and shared prior | `FrequencyPrior` |
| Directed high-order alignment and virtual compensation | `MFHAlign` |
| Sparse hypergraph incidence and co-membership | `incoming`, `scatter_regions`, `shared_membership` |
| Input-selective state recurrence | `SelectiveSSM` |
| Four-way topology modulation and pruning | `ATGMamba` |
| Sparse-to-dense regional backflow | `SDHRefine` |
| Joint losses | `segmentation_loss`, `contribution_loss`, `auxiliary_losses` |
| Complete network | `MMAHHMamba`, `build_mmahh_mamba` |
| Training / evaluation / prediction | `train_main`, `evaluate_main`, `predict_main` |

The selective state update uses `h = exp(ΔA)h + ΔB(x)x`, matching the selective-scan reference convention rather than claiming an exact zero-order-hold input discretization.

## Acknowledgments

This implementation is conceptually related to the selective state-space modeling formulation of [Mamba](https://github.com/state-spaces/mamba). Its default recurrence is implemented directly in PyTorch; the optional compiled backend is supplied by the upstream Mamba project.

The repository uses PyTorch, NumPy, SciPy and NiBabel. Dataset acknowledgments and citations should follow the applicable BraTS release.

Before publishing, add the actual manuscript citation, author information and a license chosen by the copyright holder. No paper DOI, final repository URL or license is fabricated in this initial release. Upload the three source files only; synthetic data, patient images and test checkpoints generated locally are not publication artifacts.
