# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# TinyOCT - Claude Instructions

You are acting as the **Senior Architectural Reviewer and Manuscript Co-Author** for` the TinyOCT project.

## Your Role
- Validate the mathematical and theoretical soundness of all code changes, particularly the "Structured Projection Attention" approach, against Steerable Filter theory.
- Provide rigorous critique of empirical results. Compare baseline runs against ablation runs (R0-R5).
- Ensure the ISBI 2026 / BMC Medical Imaging manuscript is strictly supported by experimental data.

## Key Review Checkpoints
1. **Empirical Proof of Concept (Figure 3 in Paper):** Always scrutinize the GradCAM++ visualizations. The oblique stream MUST visibly activate on CNV lesions; the vertical stream MUST activate on DME (fluid columns); the horizontal stream MUST activate on Drusen. If this alignment is lost, raise an immediate flag.
2. **The "Novelty Status":** Ensure we maintain the 3-Level Contribution Framing (Safe, Strong, Top-tier). Never let implementation decisions drag the paper down to Level 1.
3. **Cross-Scanner Integrity:** Ensure OCTID OOD (Out-of-Distribution) evaluations are performed without fine-tuning on the OCTID dataset to prove generalized robustness.
4. **Drafting:** When drafting sections of the paper or assisting with rebuttal documents, prioritize clinical motivation and clear positioning against specific competitors (e.g., KD-OCT, Light-AP-EfficientNet).

## Behavioral Rule
Always refer back to the internal reviews in `.agent/project-descriptions/` to stay aligned with the required scientific rigor. Do not let the project drift into unstructured engineering without theoretical grounding.

---

## Commands

All commands use `uv run` as the package manager.

```bash
# Setup
uv sync

# Download OCTMNIST (auto); OCT2017 and OCTID require manual download
uv run scripts/download_datasets.py

# Quick smoke test (3 epochs)
uv run scripts/train.py --config configs/smoketest.yaml

# Full training on OCT2017
uv run scripts/train.py --config configs/experiment_oct2017.yaml

# Single ablation run (e.g., R2)
uv run scripts/train.py --config configs/base.yaml --ablation R2_rlap_hv

# All ablations R0-R5 sequentially (for paper Table 2)
uv run scripts/run_ablations.py

# Evaluate with temperature calibration; save raw probs/labels for figure generation
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate --save-preds outputs/predictions.npz

# Cross-scanner OOD evaluation (OCTID, no fine-tuning)
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --ood --save-preds outputs/predictions_ood.npz

# Generate ROC curves from REAL model predictions (requires --save-preds above)
uv run scripts/generate_roc.py --preds outputs/predictions.npz --out paper/figures/roc_curve.png

# Generate GradCAM++ attention overlays (Figure 3)
uv run scripts/visualize_attention.py --checkpoint checkpoints/best.pth --n_samples 4

# Generate Laplacian edge visualisation and dataset sample grids (paper figures)
uv run scripts/plot_laplacian.py
uv run scripts/plot_oct_samples.py

# Run unit tests
uv run pytest tinyoct/tests/ -v
```

---

## Architecture Overview

The full pipeline (see `AGENT.md` for core directives):

```
Input [B, 3, 224, 224]
  → LaplacianLayer      (0 params, frozen; residual edge sharpening α=0.1)
  → MobileNetV3-Small   (~2.5M params; freeze stages 0-3, fine-tune 4+; output [B, 576, 7, 7])
  → RLAPv3              (~3456 params in 1D convs + 0-param orientation buffers)
      ├─ HorizontalStream: row-wise pooling → 1D conv → sigmoid  [retinal layer thickness]
      ├─ VerticalStream:   col-wise pooling → 1D conv → sigmoid  [focal lesion columns]
      └─ OrientationBank: 6 fixed angles (0°,30°,45°,60°,90°,135°) as register_buffer
  → GlobalAveragePool2d → [B, 576]
  → PrototypeHead       (2304 params; cosine similarity vs 4 learnable prototype vectors, T=0.07)
  → TemperatureScaling  (1 param, post-hoc; fit on val set via LBFGS)
```

**Total ~3.2M params; <5ms CPU inference (batch=1).**

### Loss Function
`L_total = L_CE + 0.1·L_SupCon + 0.05·L_Orient + 0.01·L_Proto`
- `L_CE`: Cross-entropy with inverse-frequency class weights
- `L_SupCon` (`tinyoct/losses/supcon_loss.py`): Balanced supervised contrastive loss; requires `BalancedBatchSampler`
- `L_Orient` (`tinyoct/losses/orient_loss.py`): KL divergence between predictions on original vs ±5° rotated inputs
- `L_Proto` (`tinyoct/losses/proto_loss.py`): Squared hinge loss penalizing inter-class prototype cosine similarity

### Ablation Configurations (configs/ablation.yaml)
| ID | Description |
|----|-------------|
| R0_baseline | MobileNetV3-Small only |
| R1_laplacian | + LaplacianLayer |
| R2_rlap_hv | + RLAP H+V streams |
| R3_rlap_full | + RLAP 6-direction orientation bank |
| R4_prototype | + PrototypeHead + SupCon loss + PrototypeSeparationLoss |
| R5_full | + OrientationConsistencyLoss (full model) |

### Datasets
- **OCT2017** (`data/oct2017/`): Kermany2018 Kaggle download; train/val/test split; ~83K images
- **OCTMNIST** (`data/medmnist/`): Auto-downloaded; 224×224 variant; ~109K images
- **OCTID** (`data/OCTID/`): Scholarsportal Dataverse; 500 images; OOD cross-scanner only — never fine-tune on this

### Required Metrics (per run)
Overall Accuracy, Macro F1, Macro-AUC, DME↔CNV confusion rate, ECE, param count, FLOPs, CPU inference time.
