"""
Training loop for TinyOCT.
Handles: training, validation, checkpointing, LR scheduling, and full
Weights & Biases logging (dataset stats, training metrics, eval metrics,
per-class F1, confusion matrices, checkpoint artifacts).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..losses import CombinedLoss
from ..utils.metrics import compute_metrics, CLASS_NAMES


def _load_wandb_key():
    """Load W&B API key from .env file or environment."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("WANDB_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip()
                if key:
                    os.environ.setdefault("WANDB_API_KEY", key)
                    return key
    return os.environ.get("WANDB_API_KEY", "")


class Trainer:
    def __init__(self, model, cfg, datamodule, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.dm = datamodule
        self.device = device
        self.best_metric = -float("inf")
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Optimiser — only trainable params
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

        # W&B -----------------------------------------------------------
        self.use_wandb = cfg.logging.use_wandb
        self.wandb = None
        if self.use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------

    def _init_wandb(self):
        """Initialise W&B with full config, dataset stats, and model summary."""
        try:
            _load_wandb_key()
            import wandb
            self.wandb = wandb

            # Flatten OmegaConf to plain dict for W&B config panel
            try:
                from omegaconf import OmegaConf
                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            except Exception:
                import copy
                cfg_dict = {}

            run = wandb.init(
                project=self.cfg.logging.wandb_project,
                name=getattr(self.cfg.project, "name", "tinyoct"),
                config=cfg_dict,
                tags=["oct2017", "tinyoct", "rlap"],
            )

            # ── Dataset statistics table ─────────────────────────────
            self._log_dataset_stats()

            # ── Model summary table ──────────────────────────────────
            self._log_model_summary()

            print(f"W&B run started: {run.url}")

        except ImportError:
            print("wandb not installed — logging disabled. Run: uv add wandb")
            self.use_wandb = False
        except Exception as e:
            print(f"W&B init failed: {e}")
            self.use_wandb = False

    def _log_dataset_stats(self):
        """Log class distribution for train, val, test splits as W&B tables."""
        if self.wandb is None:
            return
        for split_name, ds in [
            ("train", self.dm.train_ds),
            ("val",   self.dm.val_ds),
            ("test",  getattr(self.dm, "test_ds", None)),
        ]:
            if ds is None:
                continue
            counts = ds.class_counts() if hasattr(ds, "class_counts") else {}
            total  = len(ds)
            table  = self.wandb.Table(
                columns=["split", "class", "count", "fraction"],
                data=[
                    [split_name, cls, cnt, round(cnt / max(total, 1), 4)]
                    for cls, cnt in counts.items()
                ]
            )
            self.wandb.log({f"dataset/{split_name}_distribution": table})
            self.wandb.log({f"dataset/{split_name}_total": total})

    def _log_model_summary(self):
        """Log parameter counts as W&B summary metrics."""
        if self.wandb is None:
            return
        p = self.model.count_parameters()
        self.wandb.summary.update({
            "model/total_params":     p["total"],
            "model/trainable_params": p["trainable"],
            "model/rlap_params":      p["rlap"],
            "model/backbone_params":  p["backbone"],
            "model/head_params":      p["head"],
            "model/rlap_overhead_pct": round(p["rlap"] / max(p["total"], 1) * 100, 4),
        })

    def _log_checkpoint_artifact(self, ckpt_path: Path, tag: str):
        """Upload a checkpoint file as a W&B artifact."""
        if self.wandb is None or not ckpt_path.exists():
            return
        artifact = self.wandb.Artifact(
            name=f"checkpoint-{tag}",
            type="model",
            description=f"TinyOCT checkpoint: {tag}",
        )
        artifact.add_file(str(ckpt_path))
        self.wandb.log_artifact(artifact)

    def _log_confusion_matrix(self, cm: list, epoch: int, prefix: str = "val"):
        """Log confusion matrix as a W&B heatmap table."""
        if self.wandb is None:
            return
        cm_np = np.array(cm)
        # Normalised version
        cm_norm = (cm_np.astype(float) / cm_np.sum(axis=1, keepdims=True).clip(min=1)).round(3)
        table = self.wandb.Table(
            columns=["True\\Pred"] + CLASS_NAMES,
            data=[[CLASS_NAMES[i]] + row.tolist() for i, row in enumerate(cm_norm)]
        )
        self.wandb.log({f"{prefix}/confusion_matrix_epoch_{epoch:03d}": table})

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        loader = self.dm.train_dataloader()
        total_loss = ce_loss = sc_loss = or_loss = 0.0
        correct = total = 0
        step_log_every = self.cfg.logging.log_every_n_steps

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

            preds    = logits.argmax(dim=1)
            correct  += (preds == y).sum().item()
            total    += y.size(0)

            # Per-step W&B log
            if self.use_wandb and self.wandb and (batch_idx + 1) % step_log_every == 0:
                self.wandb.log({
                    "train/step_loss":           losses["total"].item(),
                    "train/step_ce_loss":        losses["ce"].item(),
                    "train/step_supcon_loss":    losses["supcon"].item(),
                    "train/step_orient_loss":    losses["orient"].item(),
                    "train/lr":                  self.optimizer.param_groups[0]["lr"],
                })

        n = len(loader)
        return {
            "loss":   total_loss / n,
            "ce":     ce_loss / n,
            "supcon": sc_loss / n,
            "orient": or_loss / n,
            "acc":    correct / total,
        }

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        loader = self.dm.val_dataloader()
        all_preds, all_labels, all_probs = [], [], []

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        return compute_metrics(all_labels, all_preds, all_probs)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "") -> Path:
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics":         metrics,
            "cfg":             dict(self.cfg) if hasattr(self.cfg, "__iter__") else str(self.cfg),
        }

        saved_path = None
        if self.cfg.checkpoint.save_every_epoch:
            path = self.ckpt_dir / f"epoch_{epoch:03d}{tag}.pth"
            torch.save(state, path)
            saved_path = path

        # Save best
        monitor = metrics.get(self.cfg.checkpoint.monitor, 0)
        if self.cfg.checkpoint.mode == "max" and monitor > self.best_metric:
            self.best_metric = monitor
            best_path = self.ckpt_dir / "best.pth"
            torch.save(state, best_path)
            # Upload best checkpoint artifact to W&B
            self._log_checkpoint_artifact(best_path, f"best_epoch_{epoch:03d}")
            print(f"  ✓ New best ({self.cfg.checkpoint.monitor}={monitor:.4f}) — saved {best_path}")
            saved_path = best_path

        return saved_path

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self):
        tc = self.cfg.train
        print(f"\nStarting training for {tc.epochs} epochs on {self.device}")
        print(f"  Dataset: OCT2017  |  Train: {len(self.dm.train_ds):,}  |  Val: {len(self.dm.val_ds):,}")
        if hasattr(self.dm.train_ds, "class_counts"):
            print(f"  Class distribution (train): {self.dm.train_ds.class_counts()}")

        for epoch in range(1, tc.epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m   = self.val_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0

            # Console output
            per_cls = val_m.get("per_class_f1", {})
            print(
                f"Epoch {epoch:3d}/{tc.epochs}  "
                f"loss={train_m['loss']:.4f}  acc={train_m['acc']:.3f}  "
                f"val_acc={val_m['accuracy']:.3f}  val_f1={val_m['macro_f1']:.3f}  "
                f"CNV={per_cls.get('CNV', 0):.3f}  DME={per_cls.get('DME', 0):.3f}  "
                f"({elapsed:.1f}s)"
            )

            # Save checkpoint
            self.save_checkpoint(epoch, {**val_m, **{f"train_{k}": v for k, v in train_m.items()}})

            # W&B epoch log
            if self.use_wandb and self.wandb:
                per_cls_log = {f"val/f1_{cls}": v for cls, v in per_cls.items()}
                self.wandb.log({
                    "epoch": epoch,
                    # Train metrics
                    "train/loss":         train_m["loss"],
                    "train/ce_loss":      train_m["ce"],
                    "train/supcon_loss":  train_m["supcon"],
                    "train/orient_loss":  train_m["orient"],
                    "train/accuracy":     train_m["acc"],
                    "train/lr":           self.optimizer.param_groups[0]["lr"],
                    # Val metrics
                    "val/accuracy":       val_m["accuracy"],
                    "val/macro_f1":       val_m["macro_f1"],
                    "val/macro_auc":      val_m.get("macro_auc", 0.0),
                    "val/epoch_time_s":   elapsed,
                    # Best so far
                    "val/best_macro_f1":  self.best_metric,
                    **per_cls_log,
                })

                # Log confusion matrix every 5 epochs or last epoch
                if epoch % 5 == 0 or epoch == tc.epochs:
                    cm = val_m.get("confusion_matrix")
                    if cm:
                        self._log_confusion_matrix(cm, epoch)

        print("\nTraining complete.")
        if self.use_wandb and self.wandb:
            self.wandb.summary.update({
                "final/val_accuracy": val_m["accuracy"],
                "final/val_macro_f1": val_m["macro_f1"],
                "final/val_macro_auc": val_m.get("macro_auc", 0.0),
                "final/best_macro_f1": self.best_metric,
            })
            self.wandb.finish()
            print(f"W&B run finished. Results at: https://wandb.ai/{self.cfg.logging.wandb_project}")
