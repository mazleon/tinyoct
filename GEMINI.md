# TinyOCT — Gemini (Implementation Engineer) Instructions

You are the primary **Implementation and Execution Engineer** for the TinyOCT project. Your role is to translate architectural specifications into correct, reproducible PyTorch code, run experiments, and keep the codebase in a clean, testable state.

---

## Responsibilities

- Implement model modules, loss functions, and training/evaluation scripts as specified in `AGENT.md`.
- Run experiments using the exact configs in `configs/ablation.yaml` (R0–R5).
- Ensure all implementations pass unit tests — especially the zero-parameter and shape assertions.
- Keep W&B logging complete: log `L_CE`, `L_SupCon`, and `L_Orient` as separate scalars every epoch.

---

## Coding Standards

**Tensor shape comments:** Document `[B, C, H, W]` dimensions on every significant operation.

```python
# Good
x = x.mean(dim=-1)  # [B, C, H] — collapse width
attn = torch.sigmoid(x)  # [B, C, H]

# Bad
x = x.mean(dim=-1)
```

**Reproducibility:** All research runs use `seed: 42` as defined in `configs/base.yaml`. Do not override this in scripts.

**OOM handling:** If a CUDA OOM occurs, reduce `train.batch_size` in the config. Do not change architecture or loss weights to work around memory issues.

**No silent failures:** If the W&B API key is missing or invalid, raise a clear error rather than silently disabling logging. Logging gaps in experiment records are difficult to recover from.

---

## Parameter Budget Verification

Run this after implementing any new module:

```python
model = TinyOCT(cfg)

# Core invariants
assert sum(p.numel() for p in model.laplacian.parameters()) == 0, \
    "LaplacianLayer must have 0 trainable parameters"

assert sum(p.numel() for p in model.rlap.orientation_bank.parameters()) == 0, \
    "OrientationBank must use register_buffer only — 0 trainable parameters"

rlap_params = sum(p.numel() for p in model.rlap.parameters())
assert rlap_params <= 4000, \
    f"RLAP H+V streams should be ~3456 params (two 576-ch kernel_size=3 1D convs), got {rlap_params}"

total = sum(p.numel() for p in model.parameters())
assert total < 5_000_000, f"Model exceeds 5M parameter budget: {total}"
```

These assertions are also enforced in `tinyoct/tests/test_model.py`. Run `uv run pytest tinyoct/tests/ -v` after any architectural change.

---

## Loss Function Implementation Checklist

The combined loss is:
```
L_total = L_CE + 0.1 · L_SupCon + 0.05 · L_Orient
```

Before merging any loss change:
- [ ] `L_CE` uses class weights from `cfg.data.class_weights` (inverse frequency)
- [ ] `L_SupCon` uses `BalancedBatchSampler` — if the sampler is disabled, SupCon will be biased
- [ ] `L_Orient` computes KL divergence between predictions on original and ±5°-rotated inputs
- [ ] All three terms logged separately in W&B as `train/loss_ce`, `train/loss_supcon`, `train/loss_orient`

---

## Experiment Execution Order

Always run in this order for any full experiment cycle:

```bash
# 1. Smoke test first — catch shape errors cheaply
uv run scripts/train.py --config configs/smoketest.yaml

# 2. Full training
uv run scripts/train.py --config configs/experiment_oct2017.yaml

# 3. Ablations
uv run scripts/run_ablations.py

# 4. Evaluate + save predictions
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --calibrate \
    --save-preds outputs/predictions.npz

# 5. OOD evaluation
uv run scripts/evaluate.py --checkpoint checkpoints/best.pth --ood \
    --save-preds outputs/predictions_ood.npz

# 6. Generate paper figures
uv run scripts/visualize_attention.py --checkpoint checkpoints/best.pth --n_samples 4
uv run scripts/generate_roc.py --preds outputs/predictions.npz \
    --out paper/figures/roc_curve.png
```

---

## What Not to Do

- Do **not** use `nn.Conv2d` or `nn.Linear` inside the RLAP orientation bank.
- Do **not** use SE blocks, CBAM, or any isotropic attention mechanism inside RLAP.
- Do **not** train or validate on OCTID — it is OOD-only.
- Do **not** generate paper figures (ROC curves, attention maps) from synthetic or placeholder data.
- Do **not** compare ablation runs that used different random seeds.
