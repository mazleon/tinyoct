# TinyOCT — Anatomy-Guided Structured Projection Attention

Ultra-lightweight retinal OCT classification using zero-parameter anatomically-guided directional attention.

**Target venues:** ISBI 2026 / BMC Medical Imaging

---

## Core Idea

RLAP (Retinal Layer-Aware Pooling) projects feature maps onto three anatomically motivated subspaces — horizontal (retinal layer thickness), vertical (focal fluid columns), and oblique (Bruch's membrane curvature) — without adding any learnable parameters. This is the first direction-aware, zero-parameter attention mechanism for retinal OCT classification.

### Three-Level Contribution (use in abstract and introduction)

| Level | Claim |
|-------|-------|
| Level 1 — Safe | Directional pooling improves OCT accuracy and reduces the clinically critical DME↔CNV confusion rate |
| Level 2 — Strong | Anatomical priors improve interpretability and cross-scanner out-of-distribution robustness |
| Level 3 — Top-tier | Structured projections onto anatomical subspaces provide a new paradigm for parameter-free geometry-aware attention in medical imaging |

---

## Architecture

```
Input (224×224)
  → LaplacianLayer        [0 params — frozen boundary sharpening, α=0.1]
  → MobileNetV3-Small     [~2.5M params — ImageNet pre-trained, output: 576 × 7 × 7]
  → RLAP v3               [~3456 params — structured projection attention]
       ├─ HorizontalStream [row pooling → 1D conv → sigmoid — retinal layer thickness]
       ├─ VerticalStream   [col pooling → 1D conv → sigmoid — focal lesion columns]
       └─ OrientationBank  [0 params — 6 fixed angles: 0°,30°,45°,60°,90°,135°]
  → PrototypeHead          [2304 params — cosine similarity + SupCon loss, T=0.07]
  → TemperatureScaling     [1 param — post-hoc ECE calibration via LBFGS]
```

**~3.2M total parameters. <5ms CPU inference (batch=1).**

Loss: `L_total = L_CE + 0.1·L_SupCon + 0.05·L_Orient`

---

## Quickstart

```bash
# 1. Setup
uv sync

# 2. Download datasets (OCTMNIST auto-downloads; see instructions.md for OCT2017 / OCTID)
uv run scripts/download_datasets.py

# 3. Smoke test — verify pipeline in 3 epochs
uv run scripts/train.py --config configs/smoketest.yaml

# 4. Full training on OCT2017
uv run scripts/train.py --config configs/experiment_oct2017.yaml

# 5. Run ablation study (R0–R5, paper Table 2)
uv run scripts/run_ablations.py

# 6. Evaluate + calibrate; save predictions for figure generation
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate \
    --save-preds outputs/predictions.npz

# 7. Cross-scanner OOD evaluation (OCTID — no fine-tuning)
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --ood \
    --save-preds outputs/predictions_ood.npz

# 8. Generate ROC curves from real predictions (paper Figure 4)
uv run scripts/generate_roc.py --preds outputs/predictions.npz \
    --out paper/figures/roc_curve.png

# 9. Generate GradCAM++ attention overlays (paper Figure 3)
uv run scripts/visualize_attention.py --checkpoint checkpoints/best.pth --n_samples 4

# 10. Unit tests
uv run pytest tinyoct/tests/ -v
```

---

## Datasets

| Dataset | Images | Role | Download |
|---------|--------|------|----------|
| OCT2017 (Kermany 2018) | 84,495 | Primary train/val/test | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| OCTMNIST (MedMNIST v2) | 109,309 | 224×224 robustness validation | `uv run scripts/download_datasets.py` (auto) |
| OCTID | 500 | Cross-scanner OOD — Cirrus HD-OCT | [Scholarsportal Dataverse](https://dataverse.scholarsportal.info/dataverse/OCTID) |

OCTID is used **only** for OOD evaluation. Never fine-tune on it.

---

## Ablation Configurations (paper Table 2)

| ID | Description |
|----|-------------|
| R0_baseline | MobileNetV3-Small only — no RLAP, no Laplacian, no prototype |
| R1_laplacian | + LaplacianLayer |
| R2_rlap_hv | + RLAP horizontal + vertical streams |
| R3_rlap_full | + RLAP 6-direction orientation bank |
| R4_prototype | + PrototypeHead + BalancedSupConLoss |
| R5_full | + OrientationConsistencyLoss (complete model) |

---

## Competitive Positioning

| Model | Params | Speed | Mechanism | Our differentiator |
|-------|--------|-------|-----------|-------------------|
| CNN-Transformer + CBAM (Pan, BSPC 2025) | 1.28M | 2.5ms | Isotropic global/local attention | CBAM has no anatomical direction awareness; we beat on interpretability + domain priors |
| Light-AP-EfficientNet (Cao, WWW 2025) | ~5M | ~5ms | Isotropic adaptive pooling | No directional awareness; RLAP adds H/V/oblique structure |
| SE-Enhanced Hybrid (PMC 2025) | >10M | >20ms | Channel-only SE + Xception | 3× larger; channel-only; fails point-of-care constraint |
| KD-OCT (Nourbakhsh, arXiv Dec 2025) | ~8M | ~10ms | ConvNeXtV2 distillation | Requires heavy teacher; 3-class only; no structural prior |

---

## Required Metrics (all runs)

Overall Accuracy, Macro F1, Per-class F1, Macro-AUC, **DME↔CNV confusion rate**, ECE, parameter count, FLOPs, CPU inference time.

The DME↔CNV confusion rate is the primary clinical validity metric. A measurable reduction vs. R0 baseline is the central empirical proof for the paper.

---

## Paper Figure Checklist

- **Figure 3 (acceptance-critical):** Side-by-side GradCAM++ overlays. Oblique stream activates on CNV lesions; vertical stream activates on DME fluid columns; horizontal stream activates on Drusen layer patterns. If this alignment is absent, the anatomical motivation is not empirically supported.
- **Figure 4:** Per-class ROC curves generated from `scripts/generate_roc.py` using real saved predictions.
- **Table 2:** Ablation R0–R5 showing incremental metric gains.
- **Table 3:** Cross-scanner OCTID OOD results vs. baselines.
