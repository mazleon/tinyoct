"""
Prototype Separation Loss (L_proto).

Penalises high cosine similarity between class prototype vectors,
preventing DME/DRUSEN prototype drift that causes confidence-0.63
misclassifications.

For K=4 classes, there are C(4,2) = 6 unique pairs. The loss computes:

    L_proto = (1/P) * Σ_{i<j} max(cos(p_i, p_j) - margin, 0)²

where P = K(K-1)/2 = 6 pairs.

The squared hinge formulation:
  - Ignores pairs already separated beyond `margin` (no wasted gradient)
  - Penalises violating pairs quadratically (smooth, stable gradients)
  - margin=0.0 penalises ALL positive cosine similarity

Default margin=-0.1 allows very slight overlap (cos_sim up to -0.1)
which is biologically realistic — DME and DRUSEN DO share some OCT
features (both are macular pathologies), so enforcing strict
orthogonality would fight the data distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeSeparationLoss(nn.Module):
    """
    Penalises cosine similarity between prototype vectors above a margin.

    Args:
        margin: Similarity threshold below which no penalty applies.
                Default -0.1 (allows slight negative correlation).
    """

    def __init__(self, margin: float = -0.1):
        super().__init__()
        self.margin = margin

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prototypes: Prototype weight matrix [K, D] from PrototypeHead

        Returns:
            Scalar loss penalising inter-prototype similarity
        """
        K = prototypes.shape[0]

        # L2-normalise prototypes for cosine similarity
        proto_norm = F.normalize(prototypes, dim=1)   # [K, D]

        # Full cosine similarity matrix: [K, K]
        sim_matrix = torch.matmul(proto_norm, proto_norm.T)  # [K, K]

        # Extract upper triangle (unique pairs, excluding diagonal)
        mask = torch.triu(torch.ones(K, K, device=prototypes.device), diagonal=1).bool()
        pair_sims = sim_matrix[mask]  # [K*(K-1)/2]

        # Squared hinge: penalise similarities above margin
        violations = torch.clamp(pair_sims - self.margin, min=0.0)  # [P]
        loss = (violations ** 2).mean()  # scalar

        return loss
