#!/usr/bin/env python3
"""
Evaluation script — generates all paper metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pth
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --ood   # cross-scanner
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tinyoct.utils.config import load_config
from tinyoct.utils.seed import set_seed
from tinyoct.models import TinyOCT
from tinyoct.data import OCTDataModule
from tinyoct.evaluation import Evaluator
from tinyoct.training import TemperatureScaling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--ood", action="store_true", help="Cross-scanner OOD evaluation on OCTID")
    p.add_argument("--calibrate", action="store_true", help="Apply temperature scaling first")
    p.add_argument("--save-preds", metavar="PATH", default=None,
                   help="Save labels/probs/preds to a .npz file for ROC figure generation")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.project.seed)

    # Load model
    model = TinyOCT(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    dm = OCTDataModule(cfg)
    dm.setup("test")

    evaluator = Evaluator(model, cfg, device)

    if args.calibrate:
        dm.setup("fit")  # need val loader
        ts = TemperatureScaling(model)
        ts.fit(dm.val_dataloader(), device)

    if args.ood:
        dm.setup_ood()
        evaluator.evaluate(dm.ood_dataloader(), desc="OCTID cross-scanner OOD",
                           save_preds=args.save_preds)
    else:
        evaluator.evaluate(dm.test_dataloader(), desc="OCT2017 test set",
                           save_preds=args.save_preds)

    evaluator.measure_inference_speed()
    evaluator.count_params_flops()


if __name__ == "__main__":
    main()
