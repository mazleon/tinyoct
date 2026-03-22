"""
Comprehensive evaluation for TinyOCT.
Generates: accuracy, per-class F1, macro-AUC, ECE, DME↔CNV confusion rate,
           cross-scanner OOD metrics, parameter count, FLOPs, CPU inference time.
"""

import time
import torch
import numpy as np
from pathlib import Path

from ..utils.metrics import compute_metrics
from ..training.calibration import TemperatureScaling

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


class Evaluator:
    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader, desc: str = "test") -> dict:
        """Full evaluation on a dataloader."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

        metrics = compute_metrics(all_labels, all_preds, all_probs)

        # DME↔CNV confusion rate (clinically critical pair)
        # DME = class 1, CNV = class 0
        dme_indices = [i for i, l in enumerate(all_labels) if l == 1]
        cnv_indices = [i for i, l in enumerate(all_labels) if l == 0]
        dme_as_cnv = sum(1 for i in dme_indices if all_preds[i] == 0)
        cnv_as_dme = sum(1 for i in cnv_indices if all_preds[i] == 1)
        metrics["dme_cnv_confusion"] = (dme_as_cnv + cnv_as_dme) / max(1, len(dme_indices) + len(cnv_indices))

        # ECE
        probs_t = torch.tensor(all_probs)
        labels_t = torch.tensor(all_labels)
        metrics["ece"] = TemperatureScaling.compute_ece(probs_t, labels_t)

        print(f"\n── {desc} results ──────────────────────────────")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
        return metrics

    def measure_inference_speed(self, n_runs: int = 1000, warmup: int = 100) -> dict:
        """Measure CPU inference time (batch size 1)."""
        self.model.eval().cpu()
        dummy = torch.randn(1, 3, self.cfg.data.image_size, self.cfg.data.image_size)

        # Warm-up
        for _ in range(warmup):
            _ = self.model(dummy)

        # Measure
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model(dummy)
        elapsed = (time.perf_counter() - t0) * 1000 / n_runs  # ms per inference

        print(f"\nCPU inference time: {elapsed:.2f}ms per image (batch=1, {n_runs} runs)")
        return {"cpu_ms_per_image": elapsed}

    def count_params_flops(self) -> dict:
        """Count parameters and FLOPs."""
        param_counts = self.model.count_parameters()

        try:
            from thop import profile as thop_profile
            dummy = torch.randn(1, 3, self.cfg.data.image_size, self.cfg.data.image_size)
            self.model.cpu().eval()
            macs, _ = thop_profile(self.model, inputs=(dummy,), verbose=False)
            param_counts["flops_M"] = macs / 1e6
        except ImportError:
            print("thop not installed — skipping FLOPs count. pip install thop")
            param_counts["flops_M"] = None

        print("\nParameter breakdown:")
        for k, v in param_counts.items():
            if v is not None:
                print(f"  {k:<20} {v:,}" if isinstance(v, int) else f"  {k:<20} {v:.1f}M")
        return param_counts
