# TinyOCT — General Agent Guidelines

This project implements anatomy-guided structured projection attention for ultra-lightweight retinal OCT classification.
Target: ISBI 2026 / BMC Medical Imaging.

---

## Core Directives

1. **The Zero-Parameter Rule:** The central novelty of this paper is that RLAP adds zero parameters for its spatial directionality logic. All anatomical orientation masks (horizontal, vertical, oblique) **must** be implemented via `register_buffer`. Never introduce learnable layers (`nn.Linear`, `nn.Conv2d`) into the orientation bank.

2. **No Isotropic Attention:** Do not use SE blocks, CBAM, global average pooling, or any channel-only recalibration mechanism inside RLAP. These are explicitly the competitor mechanisms we are differentiating from.

3. **Parameter Budget:** Total model must stay ~3.2M parameters. CPU inference must be <5ms at batch size 1 to support the point-of-care deployment claim.

4. **Tensor Shape Discipline:** Document `[B, C, H, W]` shapes in inline comments on every significant tensor operation when modifying model modules.

---

## Architecture Pipeline

| Stage | Module | File | Params |
|-------|--------|------|--------|
| 1 (frozen) | LaplacianLayer — boundary alignment, residual α=0.1 | `tinyoct/models/laplacian.py` | 0 |
| 2 | MobileNetV3-Small — ImageNet pre-trained, freeze stages 0–3 | `tinyoct/models/tinyoct.py` | ~2.5M |
| 3 | RLAPv3 — H/V streams (1D conv) + 6-direction orientation bank (buffers) | `tinyoct/models/rlap.py` | ~3456 (H+V) + 0 (bank) |
| 4 | PrototypeHead — cosine similarity, T=0.07, SupCon loss | `tinyoct/models/prototype_head.py` | 2304 |
| 5 (post-hoc) | TemperatureScaling — ECE calibration via LBFGS | `tinyoct/training/calibration.py` | 1 |

Backbone output: `[B, 576, 7, 7]`. RLAP preserves this shape. After GAP: `[B, 576]`.

---

## Loss Function

```
L_total = L_CE + 0.1 · L_SupCon + 0.05 · L_Orient
```

- `L_CE` — Cross-entropy with inverse-frequency class weights
- `L_SupCon` — Balanced supervised contrastive loss (requires `BalancedBatchSampler`)
- `L_Orient` — KL divergence between predictions on original vs ±5° rotated inputs

Log all three terms separately in W&B. If any diverges, the combined loss is unreliable.

---

## Ablation Configurations

Defined in `configs/ablation.yaml`. Run all six in order via `uv run scripts/run_ablations.py`.

| ID | Adds |
|----|------|
| R0_baseline | Nothing — MobileNetV3-Small only |
| R1_laplacian | LaplacianLayer |
| R2_rlap_hv | RLAP horizontal + vertical streams |
| R3_rlap_full | RLAP 6-direction orientation bank |
| R4_prototype | PrototypeHead + BalancedSupConLoss |
| R5_full | OrientationConsistencyLoss (complete model) |

---

## Required Metrics (every run)

Overall Accuracy, Macro F1, Per-class F1, Macro-AUC, **DME↔CNV confusion rate**, ECE, param count, FLOPs, CPU inference time.

The DME↔CNV confusion rate is the primary clinical validity metric. A measurable reduction from R0 to R5 is the central empirical proof.

---

## Unit Test Parameter Assertions

```python
# LaplacianLayer
assert sum(p.numel() for p in model.laplacian.parameters()) == 0

# OrientationBank — all masks are register_buffer, not parameters
assert sum(p.numel() for p in model.rlap.orientation_bank.parameters()) == 0

# RLAP H+V 1D conv streams combined
rlap_params = sum(p.numel() for p in model.rlap.parameters())
assert rlap_params <= 4000  # ~3456 for two 576-channel kernel_size=3 1D convs

# Full model parameter budget
total = sum(p.numel() for p in model.parameters())
assert total < 5_000_000
```

---

## Scientific Integrity Rules

- **Figure 3 is the acceptance figure.** GradCAM++ must show: oblique stream → CNV lesions, vertical stream → DME fluid columns, horizontal stream → Drusen layer bands. If this alignment is absent, do not submit.
- **OCTID is OOD-only.** Never fine-tune or validate hyperparameters on OCTID.
- **ROC curves must come from real predictions.** Use `scripts/evaluate.py --save-preds` + `scripts/generate_roc.py`. Never use synthetic or placeholder data for paper figures.
- **All research runs use `seed: 42`.** Do not compare ablation runs that used different seeds.
