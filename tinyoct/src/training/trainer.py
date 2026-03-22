"""
Training loop for TinyOCT.
Handles: training, validation, checkpointing, logging, LR scheduling.
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..losses import CombinedLoss
from ..utils.metrics import compute_metrics


class Trainer:
    def __init__(self, model, cfg, datamodule, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.dm = datamodule
        self.device = device
        self.best_metric = -float("inf")
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Optimiser
        tc = cfg.train
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tc.epochs, eta_min=tc.lr * 0.01
        )

        # Loss
        class_weights = getattr(cfg.data, "class_weights", None)
        self.criterion = CombinedLoss(cfg, class_weights)

        # Optional W&B
        self.use_wandb = cfg.logging.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=cfg.logging.wandb_project, config=dict(cfg))
            except ImportError:
                print("wandb not installed — logging disabled")
                self.use_wandb = False

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        loader = self.dm.train_dataloader()
        total_loss = ce_loss = sc_loss = or_loss = 0.0
        correct = total = 0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits, features = self.model(x, return_features=True)

            losses = self.criterion(self.model, x, logits, features, y)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses["total"].item()
            ce_loss    += losses["ce"].item()
            sc_loss    += losses["supcon"].item()
            or_loss    += losses["orient"].item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        n = len(loader)
        return {
            "loss": total_loss / n,
            "ce":   ce_loss / n,
            "supcon": sc_loss / n,
            "orient": or_loss / n,
            "acc": correct / total,
        }

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        loader = self.dm.val_dataloader()
        all_preds, all_labels, all_probs = [], [], []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        return compute_metrics(all_labels, all_preds, all_probs)

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = ""):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        # Save every epoch (needed for Week 7 attention visualisation)
        if self.cfg.checkpoint.save_every_epoch:
            path = self.ckpt_dir / f"epoch_{epoch:03d}{tag}.pth"
            torch.save(state, path)

        # Save best
        monitor = metrics.get(self.cfg.checkpoint.monitor, 0)
        if self.cfg.checkpoint.mode == "max" and monitor > self.best_metric:
            self.best_metric = monitor
            torch.save(state, self.ckpt_dir / "best.pth")

    def fit(self):
        tc = self.cfg.train
        print(f"\nStarting training for {tc.epochs} epochs on {self.device}")

        for epoch in range(1, tc.epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m   = self.val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{tc.epochs}  "
                f"loss={train_m['loss']:.4f}  "
                f"acc={train_m['acc']:.4f}  "
                f"val_acc={val_m['accuracy']:.4f}  "
                f"val_f1={val_m['macro_f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            self.save_checkpoint(epoch, val_m)

            if self.use_wandb:
                import wandb
                wandb.log({"epoch": epoch, **train_m, **{f"val_{k}": v for k, v in val_m.items()}})

        print("\nTraining complete.")
