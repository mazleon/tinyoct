#!/usr/bin/env python3
"""
Comprehensive local evaluation script for TinyOCT and ResNet18 models.

Computes full medical-grade evaluation metrics on the local OCT2017 test split:
  - Overall Accuracy, Macro F1, Weighted F1
  - Per-class Precision, Recall, F1, Specificity
  - Macro / per-class ROC-AUC
  - Expected Calibration Error (ECE)
  - Confusion matrix (absolute + normalised)
  - Confidence histogram
  - Full per-sample predictions CSV
  - LaTeX-formatted summary table

Usage:
    # Evaluate all available checkpoints
    uv run scripts/evaluate_local.py --data-root ./data/OCT2017

    # Evaluate specific checkpoints
    uv run scripts/evaluate_local.py \
        --tinyoct-ckpts checkpoints/smoketest/best.pth checkpoints/tinyoct_30ep/best.pth \
        --resnet-ckpts checkpoints/resnet18_smoketest/best.pth checkpoints/resnet18_30ep/best.pth \
        --data-root ./data/OCT2017

Outputs (all saved to outputs/evaluation/):
    metrics_<model>_<tag>.json
    confusion_<model>_<tag>.png
    roc_<model>_<tag>.png
    report_all_models.txt
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, auc as sklearn_auc,
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("scikit-learn not installed: pip install scikit-learn")
    sys.exit(1)

from tinyoct.models import TinyOCT, ResNet18Baseline
from tinyoct.data import OCTDataModule
from tinyoct.utils.config import load_config
from tinyoct.utils.seed import set_seed

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
COLORS     = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Local comprehensive evaluation")
    p.add_argument("--data-root", default="./data/OCT2017",
                   help="Path to OCT2017 root (contains train/ val/ test/)")
    p.add_argument("--split", default="test", choices=["test", "val"],
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--tinyoct-ckpts", nargs="*", default=[],
                   help="TinyOCT checkpoint paths")
    p.add_argument("--resnet-ckpts", nargs="*", default=[],
                   help="ResNet18 checkpoint paths")
    p.add_argument("--config", default="configs/smoketest.yaml")
    p.add_argument("--output-dir", default="outputs/evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_test_loader(data_root: str, split: str, batch_size: int, image_size: int = 224):
    """Build a DataLoader directly from the folder structure."""
    from tinyoct.data.dataset import OCT2017Dataset
    from tinyoct.data.transforms import get_val_transforms

    tf = get_val_transforms(image_size)
    ds = OCT2017Dataset(root=data_root, split=split, transform=tf)
    print(f"  {split} set: {len(ds)} images")
    counts = ds.class_counts()
    for cls, cnt in counts.items():
        print(f"    {cls}: {cnt}")

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False
    )
    return loader, ds

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tinyoct(ckpt_path: str, cfg, device):
    model = TinyOCT(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"  Loaded TinyOCT from epoch {epoch} | val_f1={metrics.get('macro_f1', '?'):.4f}")
    return model

def load_resnet18(ckpt_path: str, cfg, device):
    model = ResNet18Baseline(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"  Loaded ResNet18 from epoch {epoch} | val_f1={metrics.get('macro_f1', '?'):.4f}")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on dataloader, return labels, preds, probs."""
    all_labels, all_preds, all_probs = [], [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_labels.extend(y.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    elapsed = time.time() - t0
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        elapsed,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — lower is better, 0 is perfect."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += mask.sum() * abs(acc_bin - conf_bin)
    return float(ece / len(labels))


def compute_specificity(cm: np.ndarray) -> dict:
    """Per-class specificity = TN / (TN + FP)."""
    specs = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specs[cls] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return specs


def full_metrics(labels, preds, probs, elapsed_s, n_test) -> dict:
    """Compute all paper-grade metrics."""
    acc   = accuracy_score(labels, preds)
    mf1   = f1_score(labels, preds, average="macro",    zero_division=0)
    wf1   = f1_score(labels, preds, average="weighted", zero_division=0)
    prec  = precision_score(labels, preds, average=None, zero_division=0)
    rec   = recall_score(labels, preds, average=None,    zero_division=0)
    pf1   = f1_score(labels, preds, average=None,        zero_division=0)
    cm    = confusion_matrix(labels, preds)
    ece   = compute_ece(probs, labels)
    specs = compute_specificity(cm)

    # AUC
    labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
    try:
        macro_auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
        per_auc   = roc_auc_score(labels_bin, probs, multi_class="ovr", average=None)
    except Exception:
        macro_auc = 0.0
        per_auc   = [0.0] * 4

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        per_class[cls] = {
            "precision":   float(prec[i]),
            "recall":      float(rec[i]),
            "f1":          float(pf1[i]),
            "specificity": specs[cls],
            "auc":         float(per_auc[i]),
        }

    return {
        "accuracy":      float(acc),
        "macro_f1":      float(mf1),
        "weighted_f1":   float(wf1),
        "macro_auc":     float(macro_auc),
        "ece":           float(ece),
        "n_samples":     int(n_test),
        "inference_s":   float(elapsed_s),
        "ms_per_sample": float(elapsed_s * 1000 / n_test),
        "per_class":     per_class,
        "confusion_matrix": cm.tolist(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    """Save normalised confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for ax, (data, label) in zip(axes, [
        (cm, "Absolute"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1), "Normalised"),
    ]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=(1 if label == "Normalised" else None))
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
        fmt = ".2f" if label == "Normalised" else "d"
        for i in range(4):
            for j in range(4):
                val = data[i, j]
                txt = f"{val:.2f}" if label == "Normalised" else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > (0.5 if label == "Normalised" else cm.max()/2) else "black",
                        fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curves(labels: np.ndarray, probs: np.ndarray, title: str, out_path: Path):
    """Save per-class ROC curves."""
    labels_bin = label_binarize(labels, classes=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class_bars(results_dict: dict, metric: str, out_path: Path):
    """Grouped bar chart comparing models on per-class metric."""
    model_names = list(results_dict.keys())
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(model_names)
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, res) in enumerate(results_dict.items()):
        vals = [res["per_class"][cls][metric] for cls in CLASS_NAMES]
        bars = ax.bar(x + i * width - (len(model_names) - 1) * width / 2,
                      vals, width * 0.9, label=model_name, color=palette[i % len(palette)],
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"Per-Class {metric.capitalize()} Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics_table(name: str, m: dict):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  Macro F1       : {m['macro_f1']:.4f}")
    print(f"  Weighted F1    : {m['weighted_f1']:.4f}")
    print(f"  Macro AUC      : {m['macro_auc']:.4f}")
    print(f"  ECE (↓better)  : {m['ece']:.4f}")
    print(f"  Inference      : {m['ms_per_sample']:.1f} ms/sample  ({m['n_samples']} samples)")
    print(f"\n  {'Class':<10} {'Prec':>6} {'Recall':>8} {'F1':>6} {'Spec':>8} {'AUC':>6}")
    print(f"  {'─'*50}")
    for cls in CLASS_NAMES:
        pc = m["per_class"][cls]
        print(f"  {cls:<10} {pc['precision']:>6.3f} {pc['recall']:>8.3f} "
              f"{pc['f1']:>6.3f} {pc['specificity']:>8.3f} {pc['auc']:>6.3f}")
    print()


def generate_latex_table(all_results: dict) -> str:
    """Generate a full LaTeX comparison table."""
    lines = [
        "% ── TABLE: Comprehensive Model Comparison on OCT2017 Test Set ──",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of TinyOCT and ResNet18 baseline on OCT2017 test set.}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "Model & Params & Acc & MacF1 & AUC & ECE & CNV & DME & DRUSEN & NORMAL \\\\",
        "\\midrule",
    ]
    for name, m in all_results.items():
        p_str = name  # use display name
        f1s = [m["per_class"][c]["f1"] for c in CLASS_NAMES]
        row = (
            f"{name} & "
            f"{m.get('params_m', '?')}M & "
            f"{m['accuracy']:.3f} & "
            f"{m['macro_f1']:.3f} & "
            f"{m['macro_auc']:.3f} & "
            f"{m['ece']:.3f} & "
            + " & ".join(f"{v:.3f}" for v in f1s)
            + " \\\\"
        )
        lines.append(row)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build test DataLoader ────────────────────────────────────
    print(f"\nLoading {args.split} split from: {args.data_root}")
    loader, ds = build_test_loader(args.data_root, args.split, args.batch_size)

    # ── Auto-discover checkpoints if none specified ──────────────
    tinyoct_ckpts = args.tinyoct_ckpts or []
    resnet_ckpts  = args.resnet_ckpts  or []

    auto_search = [
        ("tinyoct",  "checkpoints/smoketest/best.pth"),
        ("tinyoct",  "checkpoints/tinyoct_30ep/best.pth"),
        ("resnet18", "checkpoints/resnet18_smoketest/best.pth"),
        ("resnet18", "checkpoints/resnet18_30ep/best.pth"),
    ]
    if not tinyoct_ckpts and not resnet_ckpts:
        print("\nAuto-discovering checkpoints...")
        for arch, path in auto_search:
            if Path(path).exists():
                (tinyoct_ckpts if arch == "tinyoct" else resnet_ckpts).append(path)
                print(f"  Found: {path}")

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Run evaluations ───────────────────────────────────────────
    all_results = {}

    def evaluate_model(model, name, labels=None):
        print(f"\n[{name}] Running inference on {len(ds)} samples...")
        lbl, preds, probs, elapsed = run_inference(model, loader, device)
        m = full_metrics(lbl, preds, probs, elapsed, len(ds))
        print_metrics_table(name, m)

        # Save JSON
        json_path = out_dir / f"metrics_{name.replace(' ','_')}.json"
        with open(json_path, "w") as f:
            json.dump(m, f, indent=2)

        # Confusion matrix plots
        cm = np.array(m["confusion_matrix"])
        plot_confusion_matrix(cm, f"Confusion Matrix — {name}",
                               out_dir / f"confusion_{name.replace(' ','_')}.png")

        # ROC curves
        plot_roc_curves(lbl, probs, f"ROC Curves — {name}",
                        out_dir / f"roc_{name.replace(' ','_')}.png")

        return m, lbl, probs

    # Evaluate TinyOCT checkpoints
    for ckpt_path in tinyoct_ckpts:
        tag = Path(ckpt_path).parent.name
        name = f"TinyOCT ({tag})"
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        try:
            model = load_tinyoct(ckpt_path, cfg, device)
            m, lbl, probs = evaluate_model(model, name)
            m["params_m"] = "0.9"
            all_results[name] = m
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            del model if 'model' in dir() else None

    # Evaluate ResNet18 checkpoints
    for ckpt_path in resnet_ckpts:
        tag = Path(ckpt_path).parent.name
        name = f"ResNet18 ({tag})"
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {ckpt_path}")
        try:
            model = load_resnet18(ckpt_path, cfg, device)
            m, lbl, probs = evaluate_model(model, name)
            m["params_m"] = "11.2"
            all_results[name] = m
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            del model if 'model' in dir() else None

    if not all_results:
        print("\nNo models evaluated. Check checkpoint paths.")
        return

    # ── Cross-model comparison plots ─────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating cross-model comparison plots...")
    for metric in ["f1", "recall", "precision", "specificity", "auc"]:
        plot_per_class_bars(all_results, metric,
                            out_dir / f"compare_perclass_{metric}.png")

    # ── Aggregate summary table ───────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY TABLE")
    print(f"{'='*60}")
    header = f"{'Model':<30} {'Acc':>6} {'MacF1':>7} {'WgtF1':>7} {'AUC':>7} {'ECE':>7} {'ms/s':>7}"
    print(header)
    print("─" * len(header))
    for name, m in all_results.items():
        print(f"{name:<30} {m['accuracy']:>6.3f} {m['macro_f1']:>7.3f} "
              f"{m['weighted_f1']:>7.3f} {m['macro_auc']:>7.3f} "
              f"{m['ece']:>7.4f} {m['ms_per_sample']:>7.1f}")

    # ── LaTeX table ───────────────────────────────────────────────
    latex = generate_latex_table(all_results)
    latex_path = out_dir / "table_comparison.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved: {latex_path}")
    print("\n" + latex)

    # ── Save all results ──────────────────────────────────────────
    combined_path = out_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {combined_path}")
    print(f"All plots saved to:   {out_dir}/")


if __name__ == "__main__":
    main()
