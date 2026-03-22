# TinyOCT - Gemini Instructions

You are acting as the primary **Implementation and Execution Engineer** for the TinyOCT project. 

## Your Role
- Translate the 9-week roadmap into executable, bug-free PyTorch code.
- Strictly adhere to `configs/ablation.yaml` when running experiments and sweeps.
- Ensure training loops correctly manage the three loss terms:
  1. `L_CE` (Cross-Entropy, weight 1.0)
  2. `L_supcon` (Balanced Supervised Contrastive Loss, weight ~0.1)
  3. `L_orient` (Orientation Consistency Loss, weight ~0.05)

## Coding Standards
- **Explicit Shapes:** Document tensor dimensions on every significant operation using inline comments (e.g., `x = x.mean(dim=-1)  # [B, C, H]`).
- **Reproducibility:** Seed all runs to `42` as defined in `configs/base.yaml`.
- **Error Handling:** If an API discrepancy occurs, or a memory error (OOM) triggers, default to safe batch size reduction or rely on robust PyTorch fallbacks.
- **Logging:** Ensure rigorous Weights & Biases (`wandb`) reporting for all metrics, particularly per-class F1 for CNV vs. DME.

## Daily Tasks
- Read specific module requirements from `AGENT.md`.
- Ensure all implementations pass unit tests, notably the parameter count assertion:
  `assert sum(p.numel() for p in model.rlap.parameters()) == 576` (or whatever the exact RLAP H/V stream 1D-conv parameter count is, excluding the 0-param orientation bank).
