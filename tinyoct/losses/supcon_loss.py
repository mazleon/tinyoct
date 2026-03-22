"""
Balanced Supervised Contrastive Loss for prototype head training.

Based on: Khosla et al., NeurIPS 2020 — "Supervised Contrastive Learning"
Modified for: class-balanced sampling to handle OCT2017 imbalance.

OCT2017 class distribution (approximate):
  CNV: 37,206  Normal: 26,315  DME: 11,349  Drusen: 8,617
Without balancing, SupCon is dominated by CNV/Normal pairs.
BalancedBatchSampler (in dataset.py) ensures equal class counts per batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with temperature scaling.

    Args:
        temperature: Contrastive temperature (default 0.07)
        contrast_mode: 'all' uses all samples as anchors (default)
    """

    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all"):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalised feature vectors [B, D]
            labels:   Class labels [B]
        Returns:
            Scalar loss value
        """
        device = features.device
        B = features.shape[0]

        # L2-normalise
        features = F.normalize(features, dim=1)

        # Compute similarity matrix: [B, B]
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask: diagonal = self-similarity (exclude)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)

        # Positive mask: same class, different sample
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        # Log-sum-exp denominator (all non-self pairs)
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(diag_mask, 0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Per-sample contrastive loss
        log_prob = sim_matrix - log_denom

        # Mean over positive pairs
        num_positives = pos_mask.sum(dim=1)
        # Avoid division by zero for classes with only 1 sample in batch
        loss = -(pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-9)

        # Only average over samples that have at least one positive
        valid = (num_positives > 0).float()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return (loss * valid).sum() / valid.sum()
