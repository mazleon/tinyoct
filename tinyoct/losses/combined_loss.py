"""
Combined loss function for TinyOCT.

L_total = L_CE + λ₁ · L_supcon + λ₂ · L_orient

  L_CE:      Cross-entropy with class weights (handles OCT2017 imbalance)
  L_supcon:  Balanced Supervised Contrastive (tightens prototype clusters)
  L_orient:  Orientation Consistency (robustness to acquisition variation)
"""

import torch
import torch.nn as nn

from .supcon_loss import BalancedSupConLoss
from .orient_loss import OrientationConsistencyLoss


class CombinedLoss(nn.Module):
    """
    Args:
        cfg:           Config object (cfg.train.loss)
        class_weights: Optional tensor [num_classes] for CE weighting
    """

    def __init__(self, cfg, class_weights=None):
        super().__init__()
        lc = cfg.train.loss

        self.lambda_supcon = lc.supcon_weight   # λ₁ (default 0.1)
        self.lambda_orient = lc.orient_weight   # λ₂ (default 0.05)

        # Cross-entropy (weighted for class imbalance)
        device_weights = None
        if class_weights is not None:
            device_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=device_weights)

        # Supervised contrastive
        self.supcon = BalancedSupConLoss(temperature=cfg.train.supcon.temperature)

        # Orientation consistency
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

        # ── L_CE ────────────────────────────────────────────────────
        loss_ce = self.ce(logits, labels)

        # ── L_supcon ────────────────────────────────────────────────
        loss_sc = torch.tensor(0.0, device=device)
        if self.lambda_supcon > 0:
            loss_sc = self.supcon(features, labels)

        # ── L_orient ────────────────────────────────────────────────
        loss_or = torch.tensor(0.0, device=device)
        if self.lambda_orient > 0:
            loss_or = self.orient(model, x)

        # ── Total ────────────────────────────────────────────────────
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
