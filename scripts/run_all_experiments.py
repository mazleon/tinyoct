#!/usr/bin/env python3
"""
Master experiment runner for TinyOCT paper.

Runs every experiment needed for the paper in correct order:
  1. ResNet18 baseline         (30 epochs, OCT2017)
  2. TinyOCT R9_full_v2        (30 epochs, OCT2017) — main result
  3. Ablations R0–R9           (30 epochs each)    — Table 2

All jobs run sequentially on a single GPU.
Each job logs to a separate file under /workspace/logs/.
Total wall-clock time on A100 40GB @ batch=256: ~3-4 hours.

Usage (on pod):
    nohup uv run python3 -u scripts/run_all_experiments.py \\
        > /workspace/logs/master.log 2>&1 &
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path("/workspace/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path("outputs/experiment_results.json")

# ── Experiment definitions ─────────────────────────────────────────────────
# (name, config, extra_args, log_suffix)
EXPERIMENTS = [
    # --- Baseline -----------------------------------------------------------
    {
        "name":   "resnet18_full",
        "desc":   "ResNet18 vanilla CNN baseline (30 epochs, OCT2017)",
        "config": "configs/experiment_resnet.yaml",
        "extra":  ["--model", "resnet18"],
        "log":    "resnet18_full.log",
    },
    # --- Main TinyOCT v2 result ---------------------------------------------
    {
        "name":   "tinyoct_v2_full",
        "desc":   "TinyOCT v2 R9 — all improvements, full training (30 epochs)",
        "config": "configs/experiment_oct2017.yaml",
        "extra":  ["--ablation", "R9_full_v2"],
        "log":    "tinyoct_v2_full.log",
    },
    # --- Original ablation rows (paper Table 2 columns 1-6) -----------------
    {
        "name":   "R0_baseline",
        "desc":   "R0: MobileNetV3-Small baseline, no RLAP",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R0_baseline"],
        "log":    "ablation_R0.log",
    },
    {
        "name":   "R1_laplacian",
        "desc":   "R1: + Frozen LaplacianLayer",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R1_laplacian"],
        "log":    "ablation_R1.log",
    },
    {
        "name":   "R2_rlap_hv",
        "desc":   "R2: + RLAP H+V streams",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R2_rlap_hv"],
        "log":    "ablation_R2.log",
    },
    {
        "name":   "R3_rlap_full",
        "desc":   "R3: + RLAP 6-direction orientation bank",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R3_rlap_full"],
        "log":    "ablation_R3.log",
    },
    {
        "name":   "R4_prototype",
        "desc":   "R4: + PrototypeHead + SupCon",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R4_prototype"],
        "log":    "ablation_R4.log",
    },
    {
        "name":   "R5_full",
        "desc":   "R5: TinyOCT v1 — all components + L_orient",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R5_full"],
        "log":    "ablation_R5.log",
    },
    # --- Improvement ablation rows (paper Table 2 columns 7-10) -------------
    {
        "name":   "R6_focal_stream",
        "desc":   "R6: R5 + FocalSpotStream only",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R6_focal_stream"],
        "log":    "ablation_R6.log",
    },
    {
        "name":   "R7_focal_loss",
        "desc":   "R7: R5 + FocalLoss gamma=2.0 only",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R7_focal_loss"],
        "log":    "ablation_R7.log",
    },
    {
        "name":   "R8_margin_supcon",
        "desc":   "R8: R5 + MarginSupCon margin=0.3 only",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R8_margin_supcon"],
        "log":    "ablation_R8.log",
    },
    {
        "name":   "R9_full_v2",
        "desc":   "R9: Full TinyOCT v2 — ablation-config version (base.yaml)",
        "config": "configs/base.yaml",
        "extra":  ["--ablation", "R9_full_v2"],
        "log":    "ablation_R9.log",
    },
]


def run_experiment(exp: dict, results: dict) -> bool:
    name = exp["name"]
    log_path = LOG_DIR / exp["log"]

    cmd = [
        sys.executable, "scripts/train.py",
        "--config", exp["config"],
        *exp.get("extra", []),
    ]

    print(f"\n{'='*70}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Starting: {name}")
    print(f"  {exp['desc']}")
    print(f"  Log: {log_path}")
    print(f"{'='*70}")

    with open(log_path, "w") as log_f:
        ret = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    status = "done" if ret.returncode == 0 else f"error (code {ret.returncode})"
    results[name] = {
        "status": status,
        "log":    str(log_path),
        "finished_at": datetime.now().isoformat(),
    }

    if ret.returncode == 0:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] DONE: {name}")
    else:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] ERROR in {name} — see {log_path}")
        # Print last 20 lines of log for quick diagnosis
        try:
            lines = log_path.read_text().splitlines()
            print("  Last 20 lines of log:")
            for line in lines[-20:]:
                print(f"    {line}")
        except Exception:
            pass

    return ret.returncode == 0


def main():
    print(f"\nTinyOCT Master Experiment Runner")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Logs directory: {LOG_DIR}")

    results = {}
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['name']}")
        run_experiment(exp, results)

        # Save results after each experiment (so partial results survive crashes)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    done  = sum(1 for v in results.values() if v["status"] == "done")
    errors = sum(1 for v in results.values() if "error" in v["status"])
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Done: {done}  |  Errors: {errors}  |  Total: {len(EXPERIMENTS)}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
