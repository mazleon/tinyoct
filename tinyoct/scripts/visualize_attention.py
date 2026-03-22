#!/usr/bin/env python3
"""
Generate the paper's key acceptance figure (Figure 3):
  Side-by-side RLAP attention stream overlays per pathology.

IMPORTANT: Use the Week 3 checkpoint (R3_rlap_full) — this is the
first checkpoint with the orientation bank. The oblique stream
activating on CNV is the empirical proof of your core claim.

Usage:
    python scripts/visualize_attention.py \
        --checkpoint checkpoints/epoch_030_R3_rlap_full.pth \
        --n_samples 4
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.config import load_config
from src.models import TinyOCT
from src.data import OCT2017Dataset, get_val_transforms
from src.evaluation import AttentionVisualizer

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--n_samples", type=int, default=2,
                   help="Number of samples per class to visualise")
    p.add_argument("--output_dir", default="outputs/figures")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")  # attention viz always on CPU

    model = TinyOCT(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])

    viz = AttentionVisualizer(model, device, args.output_dir)

    ds = OCT2017Dataset(
        root=cfg.data.oct2017_path,
        split="test",
        transform=get_val_transforms(cfg.data.image_size),
    )

    # Collect n_samples per class
    class_samples = {i: [] for i in range(4)}
    for img, label in ds:
        if len(class_samples[label]) < args.n_samples:
            class_samples[label].append((img, label))
        if all(len(v) >= args.n_samples for v in class_samples.values()):
            break

    for class_id, samples in class_samples.items():
        for i, (img, label) in enumerate(samples):
            viz.visualize_rlap_streams(
                image=img,
                label=label,
                save_name=f"attn_{CLASS_NAMES[class_id]}_{i+1}.png"
            )
    print(f"\nFigures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
