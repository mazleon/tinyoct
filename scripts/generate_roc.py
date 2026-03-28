#!/usr/bin/env python3
"""
Generate per-class ROC curves for the paper (Figure 4 / supplementary).

Requires real model predictions saved via:
    uv run scripts/evaluate.py --checkpoint <ckpt> --save-preds outputs/predictions.npz

Usage:
    uv run scripts/generate_roc.py --preds outputs/predictions.npz
    uv run scripts/generate_roc.py --preds outputs/predictions.npz --out paper/figures/roc_curve.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True,
                   help="Path to .npz file produced by evaluate.py --save-preds")
    p.add_argument("--out", default="paper/figures/roc_curve.png")
    args = p.parse_args()

    data = np.load(args.preds, allow_pickle=True)
    labels = data["labels"]        # [N]
    probs  = data["probs"]         # [N, num_classes]
    class_names = list(data["class_names"])
    num_classes = len(class_names)

    # Binarise labels for one-vs-rest ROC
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"TinyOCT {cls} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) on OCT2017 Dataset",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved ROC curve to {args.out}")


if __name__ == "__main__":
    main()
