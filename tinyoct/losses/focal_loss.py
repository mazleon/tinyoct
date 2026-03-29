"""
Focal Loss for hard-example mining in class-imbalanced OCT classification.

Standard cross-entropy treats all samples equally, so the majority class
(CNV: 37,205 samples) dominates the gradient signal. Focal Loss down-weights
easy, confidently-classified examples via a (1-p_t)^gamma modulation factor,
concentrating learning on the hard minority cases (DRUSEN: 8,616 samples).

Formula:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

When gamma=0, reduces exactly to standard weighted cross-entropy.
When gamma=2 (Lin et al. default), a sample predicted at 80% confidence
contributes only (1-0.8)^2 = 0.04x the gradient of an uncertain sample.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
OCT application: CNV/NORMAL are easy; DRUSEN/DME are the hard cases requiring
focused gradient signal. gamma=2 empirically yields best DRUSEN F1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-class weighting.

    Args:
        gamma:         Focusing exponent (default 2.0).
                       gamma=0 → standard CE; higher gamma → more focus on hard samples.
        class_weights: Optional 1D list or tensor [num_classes].
                       Applied as per-class alpha_t in the focal formula.
                       Combine with gamma > 0 for simultaneous frequency balancing
                       and hard-example mining.
    """

    def __init__(self, gamma: float = 2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw class scores [B, num_classes] (before softmax)
            labels: Ground-truth class indices [B]
        Returns:
            Scalar focal loss (mean reduction)
        """
        # Numerically stable log-probabilities and probabilities in one pass
        log_p = F.log_softmax(logits, dim=1)      # [B, num_classes]
        p = torch.exp(log_p)                       # [B, num_classes]

        # Gather log(p_t) and p_t for the true class at each sample
        idx = labels.view(-1, 1)                   # [B, 1]
        log_pt = log_p.gather(1, idx).squeeze(1)   # [B]
        p_t = p.gather(1, idx).squeeze(1)          # [B]

        # Focal modulation: down-weight easy (high-confidence) examples
        focal_weight = (1.0 - p_t) ** self.gamma   # [B]; 1.0 when gamma=0

        # Per-sample loss (positive: log_pt is negative, so negate)
        loss = -focal_weight * log_pt              # [B]

        # Per-class alpha weighting (addresses class imbalance)
        # Normalization mirrors PyTorch CrossEntropyLoss(weight=...):
        #   divide by sum(alpha_t) so that gamma=0 reduces exactly to weighted CE.
        if self.class_weights is not None:
            # Move weights to same device as labels (handles CPU→CUDA transfer)
            alpha_t = self.class_weights.to(labels.device)[labels]  # [B]
            loss = alpha_t * loss
            return loss.sum() / alpha_t.sum()
        else:
            return loss.mean()

    def extra_repr(self) -> str:
        has_weights = self.class_weights is not None
        return f"gamma={self.gamma}, class_weights={'yes' if has_weights else 'none'}"
