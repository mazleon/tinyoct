#!/usr/bin/env python3
"""
Main training script for TinyOCT v3.

Usage:
    # Full model (R5):
    python scripts/train.py --config configs/experiment_oct2017.yaml

    # Single ablation run:
    python scripts/train.py --config configs/base.yaml --ablation R2_rlap_hv

    # All ablation rows (R0–R5):
    python scripts/run_ablations.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.seed import set_seed
from src.utils.config import load_config, merge_ablation
from src.models import TinyOCT
from src.data import OCTDataModule
from src.training import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/base.yaml")
    p.add_argument("--ablation", default=None, help="Ablation key from configs/ablation.yaml")
    p.add_argument("--device",   default="auto")
    return p.parse_args()


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

    # Model
    model = TinyOCT(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params['total']:,} total | {params['trainable']:,} trainable")
    print(f"RLAP parameters:  {params['rlap']:,}")

    # Train
    trainer = Trainer(model, cfg, dm, device)
    trainer.fit()


if __name__ == "__main__":
    main()
