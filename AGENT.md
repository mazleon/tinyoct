# TinyOCT - General Agent Guidelines

Welcome to the TinyOCT codebase. This project implements an anatomy-guided structured projection attention mechanism for ultra-lightweight retinal OCT classification.

## Core Directives
1. **The Zero-Parameter Rule:** The core contribution of this paper rests on RLAP (Retinal Layer-Aware Pooling) adding ZERO parameters for its spatial directionality logic. All anatomical masks (horizontal, vertical, oblique) MUST be implemented using `register_buffer`. Do not introduce learnable layers (like `nn.Linear` or `nn.Conv2d`) into the orientation bank.
2. **Directional Prior Only:** Our novelty relies on anatomically motivated directional priors. Do NOT use isotropic pooling or global attention mechanisms (like SE or CBAM) anywhere in the RLAP module.
3. **Parameter Budget:** The overall model must remain around 3.2M parameters. It must support <5ms inference on CPU to prove point-of-care deployment readiness.
4. **PyTorch Idioms:** Ensure exact tensor shape tracking (`[B, C, H, W]`) in all comments when modifying modules to prevent shape mismatches during complex tensor broadcasting.

## Architecture Pipeline
- **Stage 1 [Frozen]:** LaplacianLayer for boundary alignment (0 params). Add residually with `alpha=0.1`.
- **Stage 2:** MobileNetV3-Small backbone (`timm`, 96 out channels, 7x7 spatial size).
- **Stage 3:** RLAP v3 - Structured Projection Attention. Features are projected onto Row Space (Retinal Layers), Column Space (Lesions), and Oblique Space (Bruch's membrane).
- **Stage 4:** Prototype Head with Cosine Similarity scoring and SupCon balanced loss.
- **Stage 5 [Post-Hoc]:** Temperature Scaling for ECE calibration.

## Evaluation & Ablation
You will be required to run the exact ablation table configurations defined in `configs/ablation.yaml` (R0 through R5). Ensure metrics logged include Overall Accuracy, Macro F1, Macro-AUC, DME-CNV Confusion Rate, ECE, Param count, FLOPs, and CPU inference time.
