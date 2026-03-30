# TinyOCT — Operational Guide

Comprehensive setup, training, evaluation, and figure-generation instructions for the TinyOCT project.
Target: ISBI 2026 / BMC Medical Imaging submission.

---

## 1. Environment Setup

### 1.1 Prerequisites

- **Python 3.13** via `uv` (handles virtualenv and dependencies automatically)
  - Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
- **Weights & Biases** (optional but recommended for experiment tracking)
  - Free account at https://wandb.ai

### 1.2 Initialize Environment

```bash
git clone https://github.com/mazleon/tinyoct.git
cd tinyoct
uv sync
```

### 1.3 Configure Secrets

Create a `.env` file in the project root:

```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=tinyoct
WANDB_ENTITY=your_username_or_team
```

Set `logging.use_wandb: false` in your config to disable W&B entirely (e.g., for offline runs).

---

## 2. Dataset Management

### 2.1 Supported Datasets

| Dataset | Images | Role |
|---------|--------|------|
| **OCT2017** (Kermany 2018) | ~84K | Primary train/val/test |
| **OCTMNIST** (MedMNIST v2) | ~109K | 224×224 robustness validation |
| **OCTID** | 500 | Cross-scanner OOD — never train on this |

### 2.2 Download Instructions

**OCTMNIST** — auto-downloaded:
```bash
uv run scripts/download_datasets.py
```

**OCT2017** — manual download required:
1. Download the `~2GB` archive from https://www.kaggle.com/datasets/paultimothymooney/kermany2018
2. Extract and place under `data/oct2017/` following the structure below.

**OCTID** — manual download required:
1. Download from https://dataverse.scholarsportal.info/dataverse/OCTID
2. Place under `data/OCTID/`.

### 2.3 Required Data Directory Structure

```
data/
├── oct2017/
│   ├── train/
│   │   ├── CNV/
│   │   ├── DME/
│   │   ├── DRUSEN/
│   │   └── NORMAL/
│   ├── val/
│   │   └── (same four subdirs — 125 images per class, stratified)
│   └── test/
│       └── (same four subdirs)
├── medmnist/
│   └── octmnist_224.npz
└── OCTID/
    ├── NORMAL/
    ├── DR/        (→ mapped to DME)
    ├── AMD/       (→ mapped to CNV)
    └── CSR/       (→ mapped to DRUSEN; MH class excluded)
```

---

## 3. Configuration

All hyperparameters live in `configs/`. Never hardcode values in scripts.

| File | Purpose |
|------|---------|
| `configs/base.yaml` | Master config — all default hyperparameters |
| `configs/ablation.yaml` | R0–R5 ablation overrides (paper Table 2) |
| `configs/experiment_oct2017.yaml` | Full 30-epoch run on OCT2017 |
| `configs/experiment_crossscanner.yaml` | OCTID OOD evaluation config |
| `configs/smoketest.yaml` | 3-epoch quick pipeline check (OCT2017) |
| `configs/smoketest_octmnist.yaml` | 3-epoch quick pipeline check (OCTMNIST) |

### Key Parameters to Adjust

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.dataset` | `oct2017` | Switch to `octmnist` for MedMNIST runs |
| `train.epochs` | `30` | Full training epochs |
| `train.batch_size` | `64` | Reduce if OOM; BalancedBatchSampler requires ≥4 per class |
| `train.lr` | `1.0e-3` | AdamW learning rate |
| `train.loss.supcon_weight` | `0.1` | SupCon loss weight λ₁ |
| `train.loss.orient_weight` | `0.05` | Orientation consistency loss weight λ₂ |
| `train.loss.proto_weight` | `0.01` | Prototype separation loss weight λ₃ |
| `train.loss.proto_margin` | `-0.1` | Prototype separation cosine similarity margin |
| `logging.use_wandb` | `true` | Disable for offline runs |
| `train.seed` | `42` | Fixed for all research runs — do not change |

---

## 4. Training

### 4.1 Smoke Test — Run First

Always run a smoke test before committing to a full training run:

```bash
# OCT2017
uv run scripts/train.py --config configs/smoketest.yaml

# OCTMNIST
uv run scripts/train.py --config configs/smoketest_octmnist.yaml
```

Expected output: 3 epochs complete, checkpoint saved under `checkpoints/smoketest/`, no shape errors.

### 4.2 Full Training

```bash
# Full model (R5) on OCT2017
uv run scripts/train.py --config configs/experiment_oct2017.yaml
```

Checkpoints saved to `checkpoints/` with naming `epoch_{N:03d}_{run_name}.pth`. Best checkpoint tracked by `val_macro_f1`.

### 4.3 Single Ablation Run

```bash
# Example: R2 (RLAP horizontal + vertical only)
uv run scripts/train.py --config configs/base.yaml --ablation R2_rlap_hv
```

### 4.4 Full Ablation Study (Paper Table 2)

```bash
uv run scripts/run_ablations.py
```

Runs R0 → R5 sequentially. Results saved to `outputs/ablation_results.json`. Each run uses seed=42.

---

## 5. Evaluation

### 5.1 Standard Test-Set Evaluation

```bash
# Evaluate + apply temperature calibration; save predictions for figure generation
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate \
    --save-preds outputs/predictions.npz
```

The `--save-preds` flag writes `labels`, `probs`, `preds`, and `class_names` to a `.npz` file. This file is required for generating real ROC curves (Section 6.2).

### 5.2 Cross-Scanner OOD Evaluation (OCTID)

```bash
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --ood \
    --save-preds outputs/predictions_ood.npz
```

**Rule:** OCTID must never be used for training or fine-tuning. Only load it via `dm.setup_ood()` and `dm.ood_dataloader()`.

### 5.3 Reported Metrics

Every evaluation run must report:
- Overall Accuracy, Macro F1, Per-class F1
- Macro-AUC (one-vs-rest)
- **DME↔CNV confusion rate** — the primary clinical validity metric
- ECE (Expected Calibration Error) — required for clinical deployment framing
- Parameter count, FLOPs, CPU inference time (<5ms target)

---

## 6. Figure Generation

### 6.1 GradCAM++ Attention Overlays (Paper Figure 3 — Acceptance-Critical)

```bash
uv run scripts/visualize_attention.py --checkpoint checkpoints/best.pth --n_samples 4
```

Output saved to `outputs/test_figures/`. This figure must show:
- **Oblique/orientation stream** activating on CNV lesion regions
- **Vertical stream** activating on DME fluid columns
- **Horizontal stream** activating on Drusen layer patterns

If this alignment is not visible, the anatomical motivation of RLAP is not empirically supported. Raise this as a blocker before writing the results section.

### 6.2 ROC Curves (Paper Figure 4)

ROC curves must be generated from real model predictions. First save predictions during evaluation (Section 5.1), then:

```bash
uv run scripts/generate_roc.py \
    --preds outputs/predictions.npz \
    --out paper/figures/roc_curve.png
```

For OOD ROC:
```bash
uv run scripts/generate_roc.py \
    --preds outputs/predictions_ood.npz \
    --out paper/figures/roc_curve_ood.png
```

### 6.3 Laplacian Preprocessing Visualisation

```bash
uv run scripts/plot_laplacian.py
```

Shows: original image → Laplacian edges → residual output. Use for the Appendix or supplementary material to justify LaplacianLayer design.

### 6.4 Dataset Sample Grid

```bash
uv run scripts/plot_oct_samples.py
```

Generates a grid of sample images per class. Useful for the Data section of the paper.

---

## 7. Unit Tests

Run after any change to model architecture, data loading, or loss functions:

```bash
uv run pytest tinyoct/tests/ -v
```

### Critical Assertions Enforced by Tests

- `LaplacianLayer` has **0 trainable parameters**
- `OrientationBank` has **0 trainable parameters** (all angles are `register_buffer`)
- RLAP H+V streams have ~3456 parameters total (1D convs only)
- `PrototypeHead` output range is in `[-1, 1]` (cosine similarity)
- Full model has `<5M` total parameters
- Output shape is `[B, num_classes]` with `num_classes=4`

---

## 8. Implementation Rules

These rules define the scientific validity of the project. Violating them weakens the paper's claims.

1. **Zero-Parameter Rule:** All RLAP directional masks must use `register_buffer`. Do not introduce learnable layers (`nn.Linear`, `nn.Conv2d`) into the orientation bank. The zero-parameter claim is a core novelty.
2. **No Isotropic Attention:** Do not use SE blocks, CBAM, or global average pooling as attention mechanisms inside RLAP. Isotropic mechanisms directly undermine the directional prior claim.
3. **Parameter Budget:** Total model must remain ~3.2M parameters. CPU inference must remain <5ms at batch size 1.
4. **Seed Discipline:** All research runs use `seed: 42`. Never run ablation comparisons with different seeds.
5. **Tensor Shape Comments:** Document `[B, C, H, W]` shapes in inline comments on every significant tensor operation in model modules.
6. **Loss Monitoring:** Log `L_CE`, `L_SupCon`, `L_Orient`, and `L_Proto` as separate W&B scalars. If any loss term diverges, the combined loss cannot be trusted.
7. **OCTID is OOD-only:** No fine-tuning, no validation on OCTID. Its value is as a held-out cross-scanner test.
