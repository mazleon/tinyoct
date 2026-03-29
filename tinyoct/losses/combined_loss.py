"""
Combined loss function for TinyOCT.

L_total = L_CE + λ₁ · L_supcon + λ₂ · L_orient

  L_CE:      FocalLoss (gamma > 0) or weighted CrossEntropy (gamma = 0)
             Focal Loss: down-weights easy CNV/NORMAL, focuses on hard
             DRUSEN/DME via (1-p_t)^gamma modulation. gamma=0 → standard CE.
  L_supcon:  Balanced Supervised Contrastive with optional margin.
             margin > 0 enforces inter-class separation in embedding space,
             addressing observed DRUSEN/DME overlap at 0.63 confidence.
  L_orient:  Orientation Consistency (robustness to acquisition variation)

Config keys consumed (all optional with backward-compatible defaults):
  train.loss.focal_gamma    float  (default 0.0 → standard CE)
  train.supcon.margin       float  (default 0.0 → standard SupCon)
"""

import torch
import torch.nn as nn

from .focal_loss import FocalLoss
from .supcon_loss import BalancedSupConLoss
from .orient_loss import OrientationConsistencyLoss


class CombinedLoss(nn.Module):
    """
    Args:
        cfg:           Config object (cfg.train.loss)
        class_weights: Optional list [num_classes] for CE/Focal weighting
    """

    def __init__(self, cfg, class_weights=None):
        super().__init__()
        lc = cfg.train.loss

        self.lambda_supcon = lc.supcon_weight   # λ₁ (default 0.1)
        self.lambda_orient = lc.orient_weight   # λ₂ (default 0.05)

        # ── CE / Focal Loss ──────────────────────────────────────────
        # getattr with defaults preserves backward compat for old configs
        focal_gamma = getattr(lc, "focal_gamma", 0.0)
        self.use_focal = focal_gamma > 0.0

        if self.use_focal:
            # FocalLoss owns class_weights internally
            self.ce = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
            self.ce_weights = None
        else:
            # Standard weighted CE — weights applied dynamically in forward()
            if class_weights is not None:
                self.register_buffer(
                    "ce_weights",
                    torch.tensor(class_weights, dtype=torch.float32),
                )
            else:
                self.ce_weights = None
            self.ce = nn.CrossEntropyLoss(weight=None)

        # ── Supervised Contrastive ───────────────────────────────────
        supcon_margin = getattr(cfg.train.supcon, "margin", 0.0)
        self.supcon = BalancedSupConLoss(
            temperature=cfg.train.supcon.temperature,
            margin=supcon_margin,
        )

        # ── Orientation Consistency ──────────────────────────────────
        self.orient = OrientationConsistencyLoss(
            angle_range=lc.orient_angle_range,
            temperature=lc.orient_temperature,
        )

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            model:    TinyOCT model (needed for L_orient forward pass)
            x:        Raw input images [B, 3, H, W]
            logits:   Model predictions [B, num_classes]
            features: Pre-head feature vectors [B, feature_dim]
            labels:   Ground-truth labels [B]
        Returns:
            dict with keys: total, ce, supcon, orient
        """
        device = logits.device

        # ── L_CE / L_Focal ───────────────────────────────────────────
        if self.use_focal:
            loss_ce = self.ce(logits, labels)       # FocalLoss handles weights
        elif self.ce_weights is not None:
            loss_ce = nn.functional.cross_entropy(
                logits, labels, weight=self.ce_weights.to(device)
            )
        else:
            loss_ce = self.ce(logits, labels)

        # ── L_supcon ─────────────────────────────────────────────────
        loss_sc = torch.tensor(0.0, device=device)
        if self.lambda_supcon > 0:
            loss_sc = self.supcon(features, labels)

        # ── L_orient ─────────────────────────────────────────────────
        loss_or = torch.tensor(0.0, device=device)
        if self.lambda_orient > 0:
            loss_or = self.orient(model, x)

        # ── Total ─────────────────────────────────────────────────────
        total = (
            loss_ce
            + self.lambda_supcon * loss_sc
            + self.lambda_orient * loss_or
        )

        return {
            "total":  total,
            "ce":     loss_ce.detach(),
            "supcon": loss_sc.detach(),
            "orient": loss_or.detach(),
        }
