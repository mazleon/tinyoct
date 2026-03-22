# TinyOCT — Anatomy-Guided Structured Projection Attention

Ultra-lightweight retinal OCT classification with zero-parameter anatomical attention.

## Quickstart

```bash
# 1. Setup Environment
# Ensure 'uv' is installed: https://docs.astral.sh/uv/getting-started/installation/
uv sync

# 2. Download datasets
uv run tinyoct/scripts/download_datasets.py      # auto-downloads OCTMNIST
# Follow printed instructions for OCT2017 (Kaggle) and OCTID

# 3. Train full model (R5)
uv run tinyoct/scripts/train.py --config tinyoct/configs/experiment_oct2017.yaml

# 4. Run complete ablation study (R0–R5, paper Table 2)
uv run tinyoct/scripts/run_ablations.py

# 5. Evaluate + calibrate
uv run tinyoct/scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate

# 6. Cross-scanner OOD evaluation
uv run tinyoct/scripts/evaluate.py --checkpoint checkpoints/best.pth --ood

# 7. Generate attention figures (paper Figure 3)
uv run tinyoct/scripts/visualize_attention.py --checkpoint checkpoints/epoch_030_R3_rlap_full.pth

# 8. Run unit tests
uv run pytest tinyoct/tests/ -v
```

## Architecture

```
Input (224×224)
  → LaplacianLayer        [0 params — frozen boundary sharpening]
  → MobileNetV3-Small     [~2.5M params — ImageNet pre-trained]
  → RLAP v3               [~576 params — structured projection attention]
       ├─ HorizontalStream [row space → retinal layer thickness]
       ├─ VerticalStream   [column space → focal lesion columns]
       └─ OrientationBank  [0 params — 6 oblique bases for Bruch's membrane]
  → PrototypeHead          [cosine similarity + SupCon loss]
  → TemperatureScaling     [post-hoc ECE calibration]
```

## Paper

Key claim: RLAP performs structured projection onto anatomically-motivated
subspaces — the first zero-parameter directional attention mechanism for
retinal OCT classification.

## Datasets

| Dataset | Images | Role | Download |
|---------|--------|------|----------|
| OCT2017 | 84,495 | Primary train/test | kaggle.com/datasets/paultimothymooney/kermany2018 |
| OCTMNIST | 109,309 | Low-res robustness | `pip install medmnist` (auto) |
| OCTID | 500 | Cross-scanner OOD | dataverse.scholarsportal.info/dataverse/OCTID |
