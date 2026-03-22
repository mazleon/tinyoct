"""
Orientation Consistency Loss (L_orient).

Enforces prediction stability under small rotations (±5°),
modelling realistic OCT acquisition variation (patient head tilt).

IMPORTANT: This loss operates at PREDICTION level, not attention-map level.
Rotating an OCT image should change the attention maps (retinal layers are
now at a different angle). What must be stable is the CLASSIFICATION DECISION.

Paper claim: "We introduce an orientation consistency regularizer that
enforces prediction stability under realistic OCT acquisition variation
(±5° patient head tilt), improving cross-scanner robustness without
requiring additional data augmentation at test time."
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class OrientationConsistencyLoss(nn.Module):
    """
    KL divergence between predictions on original and slightly-rotated input.

    Args:
        angle_range:  Max rotation in degrees (default ±5°)
        temperature:  Softmax temperature for KL computation (default 2.0)
                      Higher temperature = softer distributions = gentler loss
    """

    def __init__(self, angle_range: float = 5.0, temperature: float = 2.0):
        super().__init__()
        self.angle_range = angle_range
        self.temperature = temperature

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model: The TinyOCT model (called twice)
            x:     Input images [B, 3, H, W]
        Returns:
            Scalar KL divergence loss
        """
        # Random rotation angle in ±angle_range degrees
        angle = random.uniform(-self.angle_range, self.angle_range)

        # Rotate input (bilinear interpolation to avoid aliasing)
        x_rot = TF.rotate(
            x, angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,
        )

        # Forward pass on original (detached — we don't backprop through this)
        with torch.no_grad():
            logits_orig = model(x)

        # Forward pass on rotated (backprop through this)
        logits_rot = model(x_rot)

        # Soft probability distributions
        p_orig = F.softmax(logits_orig / self.temperature, dim=-1).detach()
        p_rot  = F.softmax(logits_rot  / self.temperature, dim=-1)

        # KL divergence: KL(p_orig || p_rot)
        loss = F.kl_div(
            p_rot.log(),
            p_orig,
            reduction="batchmean",
        )
        return loss
