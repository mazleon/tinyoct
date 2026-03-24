# TinyOCT — Operational Guidelines & Instructions

This document provides a comprehensive guide for developers and researchers to set up, configure, and execute the TinyOCT project for retinal OCT classification.

---

## 🚀 1. Environment Setup

### 1.1 Prerequisites

- **Python Manager**: We use `uv` for lightning-fast, reproducible dependency management.
  - Install `uv`: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
- **Weights & Biases (W&B)**: Used for experiment tracking and logging.
  - Create a free account at [wandb.ai](https://wandb.ai).

### 1.2 Initialize Environment

```bash
# Clone the repository
git clone https://github.com/mazleon/tinyoct.git
cd tinyoct

# Sync dependencies (creates .venv automatically)
uv sync
```

### 1.3 Configure Secrets

Create a `.env` file in the project root to store your W&B credentials:

```bash
# .env
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=tinyoct
```

---

## 📊 2. Dataset Management

### 2.1 Supported Datasets

The project supports three primary datasets:
1. **OCT2017 (Kermany 2018)**: Primary large-scale dataset (~84K images).
2. **OCTMNIST (MedMNIST v2)**: 224×224 normalized version (~109K images).
3. **OCTID**: Small clinical dataset (500 images) used for cross-scanner OOD validation.

### 2.2 Download & Preparation

```bash
# Auto-download OCTMNIST (224x224 version)
uv run scripts/download_datasets.py
```

For **OCT2017**, download the 2GB archive from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) and place it in `data/oct2017/`.

### 2.3 Directory Structure

Ensure your `data/` directory follows this structure:

```text
data/
├── oct2017/
│   ├── train/  (CNV, DME, DRUSEN, NORMAL)
│   ├── val/    (125 images per class, stratified from test)
│   └── test/   (remaining 125 images per class)
├── medmnist/
│   └── octmnist_224.npz
└── OCTID/
    └── (NORMAL, DR, AMD)
```

---

## ⚙️ 3. Configuration & Parameters

All experiment settings are controlled via YAML files in the `configs/` directory.

### 3.1 Key Parameters to Adjust

Open `configs/base.yaml` or `configs/smoketest.yaml` to modify:
- **`data.dataset`**: Set to `oct2017` or `octmnist`.
- **`train.epochs`**: Number of training iterations (default: 30).
- **`train.batch_size`**: Image count per batch (default: 64).
- **`train.lr`**: Learning rate (default: 1.0e-3).
- **`train.loss`**: Relative weights for the three loss terms (`ce_weight`, `supcon_weight`, `orient_weight`).
- **`logging.use_wandb`**: Set `true` to enable online tracking.

---

## 🏗️ 4. Training & Validation

### 4.1 Running a Smoke Test

Always run a 3-epoch smoke test first to verify the pipeline:

```bash
# For OCT2017
uv run scripts/train.py --config configs/smoketest.yaml

# For OCTMNIST
uv run scripts/train.py --config configs/smoketest_octmnist.yaml
```

### 4.2 Full Training

Execute the main training script with the base configuration:

```bash
uv run scripts/train.py --config configs/base.yaml
```

### 4.3 Validation Split Logic

The project uses a stratified split. If you update the dataset, use the integrated logic in `scripts/download_datasets.py` or the manual split tool I used to ensure a balanced 500-sample validation set.

---

## 📈 5. Evaluation & Visualization

### 5.1 Model Evaluation

Evaluate a trained checkpoint for accuracy, F1-score, and calibration (ECE):

```bash
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate
```

### 5.2 OOD Evaluation (Cross-Scanner)

Test the model's robustness on the unseen OCTID dataset:

```bash
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --ood
```

### 5.3 Attention Visualization (Week 7 Key Figure)

Generate the paper's "Acceptance Figure" showing where each RLAP stream focuses:

```bash
uv run scripts/visualize_attention.py --checkpoint checkpoints/best.pth
```

Figures will be saved in `outputs/figures/`.

---

## 🧪 6. Testing & Maintenance

Run the unit test suite after any architectural changes to ensure RLAP stream dimensions and parameter counts remain correct:

```bash
uv run pytest tinyoct/tests/ -v
```

### Critical Rules for Model Implementation

1. **Explicit Shapes**: Always document tensor dimensions in code comments.
2. **Determinism**: Keep `seed: 42` for all research runs.
3. **Loss Monitoring**: Monitor all three loss terms (`L_CE`, `L_sc`, `L_or`) separately in W&B.
