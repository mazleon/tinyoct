#!/usr/bin/env python3
"""
Main training script for TinyOCT v3.

Usage:
    # Full TinyOCT model (R5):
    uv run scripts/train.py --config configs/experiment_oct2017.yaml

    # ResNet18 baseline (fair comparison):
    uv run scripts/train.py --config configs/smoketest_resnet.yaml --model resnet18

    # Single ablation run:
    uv run scripts/train.py --config configs/base.yaml --ablation R2_rlap_hv

    # All ablation rows (R0–R5):
    uv run scripts/run_ablations.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tinyoct.utils.seed import set_seed
from tinyoct.utils.config import load_config, merge_ablation
from tinyoct.models import TinyOCT, ResNet18Baseline
from tinyoct.data import OCTDataModule
from tinyoct.training import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/base.yaml")
    p.add_argument("--ablation", default=None, help="Ablation key from configs/ablation.yaml")
    p.add_argument("--model",    default="tinyoct",
                   choices=["tinyoct", "resnet18"],
                   help="Model architecture: 'tinyoct' (default) or 'resnet18' baseline")
    p.add_argument("--device",   default="auto")
    return p.parse_args()


def build_model(model_name: str, cfg):
    """Instantiate the selected model architecture."""
    if model_name == "resnet18":
        print(f"\n  Architecture: ResNet18 Baseline (vanilla CNN)")
        print(f"  Pretrained:   {cfg.model.pretrained}")
        return ResNet18Baseline(cfg)
    else:
        print(f"\n  Architecture: TinyOCT (RLAP + Laplacian + PrototypeHead)")
        print(f"  Pretrained:   {cfg.model.pretrained}")
        return TinyOCT(cfg)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply ablation overrides if specified
    if args.ablation:
        abl_cfg = load_config("configs/ablation.yaml")
        overrides = dict(getattr(abl_cfg.ablations, args.ablation))
        cfg = merge_ablation(cfg, overrides)
        print(f"\nRunning ablation: {args.ablation} — {overrides.get('description', '')}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    set_seed(cfg.project.seed)

    # Data
    dm = OCTDataModule(cfg)
    dm.setup("fit")

    # Model — select architecture based on --model flag
    model = build_model(args.model, cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,} total | {params['trainable']:,} trainable")
    print(f"RLAP parameters:  {params['rlap']:,}")

    # Train
    trainer = Trainer(model, cfg, dm, device, model_name=args.model)
    trainer.fit()


if __name__ == "__main__":
    main()
