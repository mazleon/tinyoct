#!/usr/bin/env python3
"""
Full paper evaluation pipeline for TinyOCT.

Produces all figures and metrics needed for the ISBI / BMC paper:
  - Classification report (per-class precision, recall, F1)
  - Normalised confusion matrix  → paper/figures/confusion_matrix.png
  - Per-class + macro ROC curves → paper/figures/roc_curves.png
  - Calibration reliability plot → paper/figures/calibration_curve.png
  - Summary JSON                 → outputs/eval_results.json
  - Raw predictions              → outputs/predictions_test.npz

Usage:
    uv run scripts/run_evaluation.py --checkpoint checkpoints/best.pth
    uv run scripts/run_evaluation.py --checkpoint checkpoints/best.pth --calibrate
"""

import sys, argparse, json, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from tinyoct.utils.config import load_config
from tinyoct.utils.seed import set_seed
from tinyoct.models import TinyOCT
from tinyoct.data import OCTDataModule
from tinyoct.training import TemperatureScaling

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#264653"]

# ── matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── helpers ──────────────────────────────────────────────────────────────────

def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(y.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── figure 1: confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max() if fmt == "d" else 1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        thresh = data.max() / 2.0
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = f"{data[i,j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=10)

    fig.suptitle("TinyOCT — OCT2017 Test Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return cm, cm_norm


# ── figure 2: ROC curves ─────────────────────────────────────────────────────

def plot_roc_curves(labels, probs, out_path):
    y_bin = label_binarize(labels, classes=range(len(CLASS_NAMES)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-class OvR curves
    ax = axes[0]
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls}  (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: macro + micro average
    ax = axes[1]
    # micro
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="navy", lw=2.5, linestyle=":",
            label=f"Micro-avg  (AUC={auc_micro:.4f})")
    # macro
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], probs[:, i])[0] for i in range(len(CLASS_NAMES))]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(CLASS_NAMES)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(CLASS_NAMES)
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="crimson", lw=2.5,
            label=f"Macro-avg  (AUC={auc_macro:.4f})")
    # individual faint
    for i, col in enumerate(COLORS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=col, lw=1, alpha=0.35)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro / Micro Average ROC")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("TinyOCT — ROC Analysis, OCT2017 Test Set",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return auc_macro, auc_micro


# ── figure 3: calibration curve ──────────────────────────────────────────────

def plot_calibration(labels, probs, out_path):
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        y_true_bin = (labels == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_true_bin, probs[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", color=col, lw=1.8, ms=5, label=cls)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram (per class)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Confidence histogram
    ax2 = axes[1]
    max_probs = probs.max(axis=1)
    correct = (np.array(probs).argmax(axis=1) == labels)
    ax2.hist(max_probs[correct], bins=20, alpha=0.6, color="#2A9D8F",
             label="Correct", density=True)
    ax2.hist(max_probs[~correct], bins=20, alpha=0.6, color="#E63946",
             label="Incorrect", density=True)
    ax2.set_xlabel("Max predicted probability (confidence)")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("TinyOCT — Calibration Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Paper evaluation pipeline")
    p.add_argument("--checkpoint", required=True, help="Path to best.pth")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--calibrate", action="store_true",
                   help="Apply temperature scaling before evaluation")
    p.add_argument("--figures-dir", default="paper/figures")
    p.add_argument("--outputs-dir", default="outputs")
    args = p.parse_args()

    figures_dir = Path(args.figures_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.project.seed)
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = TinyOCT(cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dm = OCTDataModule(cfg)
    dm.setup("test")

    # ── Temperature calibration ───────────────────────────────────────────────
    if args.calibrate:
        dm.setup("fit")
        ts = TemperatureScaling(model)
        ts.fit(dm.val_dataloader(), device)
        T_val = model.log_temperature.exp().item()
        print(f"Temperature scaling applied (T={T_val:.3f})")

    # ── Collect predictions ───────────────────────────────────────────────────
    print("\nRunning inference on test set...")
    test_loader = dm.test_dataloader()
    labels, preds, probs = collect_predictions(model, test_loader, device)

    # Save raw predictions for downstream use
    preds_path = outputs_dir / "predictions_test.npz"
    np.savez(preds_path, labels=labels, preds=preds, probs=probs,
             class_names=np.array(CLASS_NAMES))
    print(f"Saved raw predictions → {preds_path}")

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    report_path = outputs_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"\n── Classification Report ────────────────────────────────\n{report}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    cm, cm_norm = plot_confusion_matrix(
        labels, preds, figures_dir / "confusion_matrix.png"
    )
    auc_macro, auc_micro = plot_roc_curves(
        labels, probs, figures_dir / "roc_curves.png"
    )
    plot_calibration(labels, probs, figures_dir / "calibration_curve.png")

    # ── Summary metrics → JSON ────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score
    per_class_auc = {}
    y_bin = label_binarize(labels, classes=range(len(CLASS_NAMES)))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        per_class_auc[cls] = round(auc(fpr, tpr), 6)

    per_cls_report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=6, output_dict=True
    )

    # ECE
    from tinyoct.training import TemperatureScaling as TS
    probs_t = torch.tensor(probs)
    labels_t = torch.tensor(labels)
    ece = float(TS.compute_ece(probs_t, labels_t))

    # DME<->CNV confusion
    dme_idx = np.where(labels == 1)[0]
    cnv_idx = np.where(labels == 0)[0]
    dme_as_cnv = (preds[dme_idx] == 0).sum()
    cnv_as_dme = (preds[cnv_idx] == 1).sum()
    dmc_confusion = float(dme_as_cnv + cnv_as_dme) / max(1, len(dme_idx) + len(cnv_idx))

    # param count (before moving to CPU)
    param_counts = model.count_parameters()

    # inference speed (CPU, batch=1)
    model.cpu().eval()
    dummy = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    import time
    for _ in range(50):  # warmup
        model(dummy)
    t0 = time.perf_counter()
    for _ in range(500):
        model(dummy)
    cpu_ms = (time.perf_counter() - t0) * 1000 / 500

    results = {
        "checkpoint": str(args.checkpoint),
        "calibrated": args.calibrate,
        "n_test_samples": int(len(labels)),
        "overall": {
            "accuracy":    round(float(accuracy_score(labels, preds)), 6),
            "macro_f1":    round(float(f1_score(labels, preds, average="macro")), 6),
            "macro_auc":   round(float(auc_macro), 6),
            "micro_auc":   round(float(auc_micro), 6),
            "ece":         round(ece, 6),
            "dme_cnv_confusion_rate": round(dmc_confusion, 6),
        },
        "per_class_f1": {
            cls: round(per_cls_report[cls]["f1-score"], 6)
            for cls in CLASS_NAMES
        },
        "per_class_auc": per_class_auc,
        "per_class_precision": {
            cls: round(per_cls_report[cls]["precision"], 6)
            for cls in CLASS_NAMES
        },
        "per_class_recall": {
            cls: round(per_cls_report[cls]["recall"], 6)
            for cls in CLASS_NAMES
        },
        "model": {
            "total_params":     param_counts.get("total"),
            "trainable_params": param_counts.get("trainable"),
            "rlap_params":      param_counts.get("rlap"),
            "head_params":      param_counts.get("head"),
            "backbone_params":  param_counts.get("backbone"),
            "cpu_ms_per_image": round(cpu_ms, 3),
        },
    }

    json_path = outputs_dir / "eval_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved summary → {json_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    o = results["overall"]
    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          TinyOCT — Test Set Results Summary              ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Accuracy        {o['accuracy']:.4f}                           ║
    ║  Macro F1        {o['macro_f1']:.4f}                           ║
    ║  Macro AUC       {o['macro_auc']:.4f}                           ║
    ║  Micro AUC       {o['micro_auc']:.4f}                           ║
    ║  ECE             {o['ece']:.4f}                           ║
    ║  DME↔CNV conf.  {o['dme_cnv_confusion_rate']:.4f}                           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Per-class AUC:                                          ║
    ║    CNV    {results['per_class_auc']['CNV']:.4f}    DME    {results['per_class_auc']['DME']:.4f}              ║
    ║    DRUSEN {results['per_class_auc']['DRUSEN']:.4f}    NORMAL {results['per_class_auc']['NORMAL']:.4f}              ║
    ╠══════════════════════════════════════════════════════════╣
    ║  CPU latency: {cpu_ms:.2f}ms/image                           ║
    ║  Params: {param_counts.get('total', 0):,} total                        ║
    ╚══════════════════════════════════════════════════════════╝
    """))


if __name__ == "__main__":
    main()
