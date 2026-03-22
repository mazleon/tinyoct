#!/usr/bin/env python3
"""
Run all 6 ablation rows (R0–R5) sequentially.
Saves results to outputs/ablation_results.json for paper Table 2.

Usage:
    python scripts/run_ablations.py
"""

import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

ABLATIONS = ["R0_baseline", "R1_laplacian", "R2_rlap_hv",
             "R3_rlap_full", "R4_prototype", "R5_full"]

def main():
    results = {}
    for abl in ABLATIONS:
        print(f"\n{'='*60}")
        print(f"  Running ablation: {abl}")
        print(f"{'='*60}")
        ret = subprocess.run([
            sys.executable, "scripts/train.py",
            "--config", "configs/base.yaml",
            "--ablation", abl,
        ], capture_output=False)
        if ret.returncode != 0:
            print(f"  ERROR in {abl} — check logs above")
        results[abl] = {"status": "done" if ret.returncode == 0 else "error"}

    # Save summary
    out = Path("outputs/ablation_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation runs complete. Summary saved to {out}")

if __name__ == "__main__":
    main()
