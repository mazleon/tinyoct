"""
Pathology-Specific Prototype Head.

Replaces FC + softmax with cosine-similarity classification.
Each class has a learnable prototype vector in feature space.
Classification = cosine similarity to nearest prototype.

Benefits:
  1. Interpretable: prototypes can be visualised and inspected
  2. Handles class imbalance naturally (no hard margin tuning)
  3. Similarity scores give directly interpretable confidence
     e.g. "87% similar to DME prototype"

Paper justification: "We replace the standard classification head with
a prototype-based cosine similarity scoring mechanism, enabling
interpretable per-class confidence estimates aligned with clinical
decision-making requirements."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """
    Cosine similarity-based prototype classification head.

    Args:
        feature_dim:  Input feature dimensionality
        num_classes:  Number of classes (4 for OCT2017)
        temperature:  Scaling factor for cosine similarities (default 0.07)
    """

    def __init__(
        self,
        feature_dim: int = 96,
        num_classes: int = 4,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

        # Learnable prototype vectors: one per class
        # Orthogonal init ensures maximum initial separation — critical for
        # minority classes like DRUSEN that receive fewer gradient updates
        # in early epochs. With K=4 in D=576 space, orthogonality is exact.
        proto_init = torch.empty(num_classes, feature_dim)
        nn.init.orthogonal_(proto_init)
        self.prototypes = nn.Parameter(
            F.normalize(proto_init, dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, feature_dim] (after global average pool)
        Returns:
            logits: [B, num_classes] — scaled cosine similarities
        """
        # L2-normalise both features and prototypes for cosine similarity
        x_norm = F.normalize(x, dim=1)                              # [B, D]
        proto_norm = F.normalize(self.prototypes, dim=1)            # [K, D]
        # Cosine similarity: [B, K]
        similarities = torch.matmul(x_norm, proto_norm.T)           # [B, K]
        # Scale by temperature (learnable via loss, fixed at training time)
        logits = similarities / self.temperature                     # [B, K]
        return logits

    def get_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw cosine similarities (0–1 range) for interpretability."""
        x_norm = F.normalize(x, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        return torch.matmul(x_norm, proto_norm.T)  # [B, K], values in [-1, 1]

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.prototypes.shape[1]}, "
            f"num_classes={self.num_classes}, "
            f"temperature={self.temperature}"
        )
