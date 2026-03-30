#!/usr/bin/env python3
"""
Model comparison script for TinyOCT paper.

Loads best checkpoints from TinyOCT and ResNet18, evaluates both on the same
test set, and outputs a side-by-side comparison table for the paper.

Usage:
    python scripts/compare_models.py \
        --tinyoct-ckpt checkpoints/smoketest/best.pth \
        --resnet-ckpt checkpoints/resnet18_smoketest/best.pth \
        --config configs/smoketest.yaml

Output:
    - Console: formatted comparison table
    - outputs/model_comparison.json: structured results
    - LaTeX table row snippet for paper inclusion
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from tinyoct.utils.seed import set_seed
from tinyoct.utils.config import load_config
from tinyoct.utils.metrics import compute_metrics, CLASS_NAMES
from tinyoct.models import TinyOCT, ResNet18Baseline
from tinyoct.data import OCTDataModule


def parse_args():
    p = argparse.ArgumentParser(description="Compare TinyOCT vs ResNet18 baseline")
    p.add_argument("--tinyoct-ckpt", type=str, required=True,
                   help="Path to TinyOCT best checkpoint")
    p.add_argument("--resnet-ckpt", type=str, required=True,
                   help="Path to ResNet18 best checkpoint")
    p.add_argument("--config", type=str, default="configs/smoketest.yaml",
                   help="Config for data loading (only data section is used)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="outputs/model_comparison.json")
    return p.parse_args()


def load_model(model_class, cfg, ckpt_path, device):
    """Load a model from checkpoint."""
    model = model_class(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Run evaluation and collect all metrics."""
    all_preds, all_labels, all_probs = [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return compute_metrics(all_labels, all_preds, all_probs)


@torch.no_grad()
def measure_inference_time(model, device, input_size=(1, 3, 224, 224), n_runs=100):
    """Measure average inference time on CPU (batch=1)."""
    model_cpu = model.cpu()
    model_cpu.eval()
    dummy = torch.randn(*input_size)

    # Warmup
    for _ in range(10):
        _ = model_cpu(dummy)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model_cpu(dummy)
        times.append((time.perf_counter() - t0) * 1000)  # milliseconds

    model.to(device)  # move back
    return {
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "median_ms": round(np.median(times), 2),
    }


def format_comparison_table(tinyoct_results, resnet_results):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 75)
    print("  MODEL COMPARISON: TinyOCT vs ResNet18 Baseline")
    print("=" * 75)

    rows = [
        ("Total Parameters", f"{tinyoct_results['params']['total']:,}",
         f"{resnet_results['params']['total']:,}"),
        ("Trainable Parameters", f"{tinyoct_results['params']['trainable']:,}",
         f"{resnet_results['params']['trainable']:,}"),
        ("RLAP Parameters", f"{tinyoct_results['params']['rlap']:,}",
         f"{resnet_results['params']['rlap']:,}"),
        ("", "", ""),
        ("Accuracy", f"{tinyoct_results['metrics']['accuracy']:.4f}",
         f"{resnet_results['metrics']['accuracy']:.4f}"),
        ("Macro F1", f"{tinyoct_results['metrics']['macro_f1']:.4f}",
         f"{resnet_results['metrics']['macro_f1']:.4f}"),
        ("Macro AUC", f"{tinyoct_results['metrics']['macro_auc']:.4f}",
         f"{resnet_results['metrics']['macro_auc']:.4f}"),
    ]

    # Per-class F1
    for cls in CLASS_NAMES:
        t_f1 = tinyoct_results['metrics'].get('per_class_f1', {}).get(cls, 0)
        r_f1 = resnet_results['metrics'].get('per_class_f1', {}).get(cls, 0)
        rows.append((f"  F1 ({cls})", f"{t_f1:.4f}", f"{r_f1:.4f}"))

    # Inference time
    if 'inference' in tinyoct_results:
        rows.append(("", "", ""))
        rows.append(("CPU Inference (ms)", f"{tinyoct_results['inference']['mean_ms']:.1f}",
                      f"{resnet_results['inference']['mean_ms']:.1f}"))

    print(f"\n{'Metric':<25} {'TinyOCT':<20} {'ResNet18':<20}")
    print("-" * 65)
    for label, t_val, r_val in rows:
        if label == "":
            print()
        else:
            print(f"{label:<25} {t_val:<20} {r_val:<20}")
    print("=" * 75)


def generate_latex_rows(tinyoct_results, resnet_results):
    """Generate LaTeX table rows for the paper."""
    t = tinyoct_results
    r = resnet_results

    t_f1 = t['metrics'].get('per_class_f1', {})
    r_f1 = r['metrics'].get('per_class_f1', {})

    print("\n% LaTeX table rows for paper (Table X: Model Comparison)")
    print("% Model & Params & Acc & F1 & AUC & CNV & DME & DRUSEN & NORMAL \\\\")
    print(f"ResNet18 & {r['params']['total']/1e6:.1f}M & "
          f"{r['metrics']['accuracy']:.3f} & {r['metrics']['macro_f1']:.3f} & "
          f"{r['metrics']['macro_auc']:.3f} & "
          f"{r_f1.get('CNV',0):.3f} & {r_f1.get('DME',0):.3f} & "
          f"{r_f1.get('DRUSEN',0):.3f} & {r_f1.get('NORMAL',0):.3f} \\\\")
    print(f"TinyOCT (Ours) & {t['params']['total']/1e6:.1f}M & "
          f"{t['metrics']['accuracy']:.3f} & {t['metrics']['macro_f1']:.3f} & "
          f"{t['metrics']['macro_auc']:.3f} & "
          f"{t_f1.get('CNV',0):.3f} & {t_f1.get('DME',0):.3f} & "
          f"{t_f1.get('DRUSEN',0):.3f} & {t_f1.get('NORMAL',0):.3f} \\\\")
    print()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.project.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data — use val set for comparison
    dm = OCTDataModule(cfg)
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    print(f"Validation set: {len(dm.val_ds)} images")

    # ── Load TinyOCT ─────────────────────────────────────────────
    print(f"\nLoading TinyOCT from: {args.tinyoct_ckpt}")
    tinyoct_model = load_model(TinyOCT, cfg, args.tinyoct_ckpt, device)
    tinyoct_params = tinyoct_model.count_parameters()
    tinyoct_metrics = evaluate_model(tinyoct_model, val_loader, device)
    tinyoct_inference = measure_inference_time(tinyoct_model, device)
    print(f"  TinyOCT: acc={tinyoct_metrics['accuracy']:.4f}, f1={tinyoct_metrics['macro_f1']:.4f}")

    # ── Load ResNet18 ────────────────────────────────────────────
    # Create a config with ResNet18 settings for model instantiation
    print(f"\nLoading ResNet18 from: {args.resnet_ckpt}")
    resnet_model = load_model(ResNet18Baseline, cfg, args.resnet_ckpt, device)
    resnet_params = resnet_model.count_parameters()
    resnet_metrics = evaluate_model(resnet_model, val_loader, device)
    resnet_inference = measure_inference_time(resnet_model, device)
    print(f"  ResNet18: acc={resnet_metrics['accuracy']:.4f}, f1={resnet_metrics['macro_f1']:.4f}")

    # ── Compile results ──────────────────────────────────────────
    tinyoct_results = {
        "model": "TinyOCT",
        "params": tinyoct_params,
        "metrics": tinyoct_metrics,
        "inference": tinyoct_inference,
    }
    resnet_results = {
        "model": "ResNet18",
        "params": resnet_params,
        "metrics": resnet_metrics,
        "inference": resnet_inference,
    }

    # ── Output ───────────────────────────────────────────────────
    format_comparison_table(tinyoct_results, resnet_results)
    generate_latex_rows(tinyoct_results, resnet_results)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "tinyoct": tinyoct_results,
        "resnet18": resnet_results,
        "comparison": {
            "param_reduction": f"{resnet_params['total'] / tinyoct_params['total']:.1f}x",
            "accuracy_delta": round(tinyoct_metrics['accuracy'] - resnet_metrics['accuracy'], 4),
            "f1_delta": round(tinyoct_metrics['macro_f1'] - resnet_metrics['macro_f1'], 4),
        }
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
