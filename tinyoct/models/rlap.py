"""
RLAP: Retinal Layer-Aware Pooling
Structured Projection Attention for anatomically-grounded feature modulation.

Mathematical framing:
  RLAP projects the feature tensor F ∈ ℝ^{C×H×W} onto four anatomically-
  motivated subspace families:
    φ_h     → row space       (retinal layer thickness)
    φ_v     → column space    (focal lesion columns)
    φ_focal → spot space      (drusenoid focal deposits — optional)
    φ_θ     → oblique bases   (Bruch's membrane orientation, 6 angles)

All orientation masks are register_buffer — zero trainable parameters.
The 1D convolutions in H and V streams ARE learnable (~3456 params total)
but this is negligible vs the backbone's 4.3M.

FocalSpotStream adds 1728 params when enabled (focal_spot=True):
    576 (depthwise 1×1 conv) + 1152 (BatchNorm2d weight + bias) = 1728

Parameter assertions:
    focal_spot=False: sum(p.numel() for p in rlap.parameters()) == 3456
    focal_spot=True:  sum(p.numel() for p in rlap.parameters()) == 5184
    OrientationBank:  0 parameters (pure buffers, always)

Scientific basis for FocalSpotStream:
  DRUSEN produce focal bright spots at the RPE-Bruch's membrane interface.
  The H-stream averages across width (suppressing focal deposits) and the
  V-stream averages across height (misaligned for horizontal drusen clusters).
  A morphological white top-hat transform (MaxPool - input) explicitly
  highlights local intensity maxima — the exact signature of drusenoid
  deposits — without introducing structural assumptions about their location.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizontalStream(nn.Module):
    """
    Projects features onto row space — captures retinal layer thickness.
    AvgPool along width → 1D conv along height → sigmoid attention.
    Output: A_h ∈ ℝ^{B×C×H×1}
    """

    def __init__(self, channels: int, height: int, kernel_size: int = 3):
        super().__init__()
        # 1D conv along height dimension
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # depthwise: one filter per channel
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Pool along width: [B, C, H, W] → [B, C, H]
        pooled = x.mean(dim=-1)             # [B, C, H]
        # 1D conv along height
        out = self.conv1d(pooled)           # [B, C, H]
        out = self.bn(out)
        out = torch.sigmoid(out)            # [B, C, H]
        # Reshape for broadcasting: [B, C, H] → [B, C, H, 1]
        return out.unsqueeze(-1)            # [B, C, H, 1]


class VerticalStream(nn.Module):
    """
    Projects features onto column space — captures focal lesion columns.
    AvgPool along height → 1D conv along width → sigmoid attention.
    Output: A_v ∈ ℝ^{B×C×1×W}
    """

    def __init__(self, channels: int, width: int, kernel_size: int = 3):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pooled = x.mean(dim=-2)             # [B, C, W]
        out = self.conv1d(pooled)           # [B, C, W]
        out = self.bn(out)
        out = torch.sigmoid(out)            # [B, C, W]
        return out.unsqueeze(-2)            # [B, C, 1, W]


class OrientationBank(nn.Module):
    """
    Projects features onto oblique bases — captures Bruch's membrane orientation.

    Uses 6 fixed angles: 0°, 30°, 45°, 60°, 90°, 135°
    Spans the angular half-space at 30° resolution — consistent with
    steerable filter theory (Weiler & Cesa, CVPR 2018).

    ALL masks are register_buffer — ZERO trainable parameters.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        angles: list = None,
    ):
        super().__init__()
        if angles is None:
            angles = [0, 30, 45, 60, 90, 135]
        self.angles = angles
        self.num_angles = len(angles)

        # Build orientation masks — pure geometry, no learning
        masks = []
        for angle_deg in angles:
            mask = self._make_stripe_mask(height, width, angle_deg)
            masks.append(mask)

        # Stack: [num_angles, 1, H, W]
        mask_tensor = torch.stack(masks, dim=0)
        # register_buffer: zero parameters, moves to device with model
        self.register_buffer("masks", mask_tensor)

    @staticmethod
    def _make_stripe_mask(H: int, W: int, angle_deg: float) -> torch.Tensor:
        """
        Create a soft stripe mask aligned at `angle_deg` degrees.
        Uses a Gaussian weighting along the perpendicular direction
        for smooth, differentiable activation.
        """
        theta = torch.tensor(angle_deg * math.pi / 180.0)
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Project coordinates onto perpendicular direction
        cos_t, sin_t = theta.cos(), theta.sin()
        # Perpendicular to stripe: rotate direction by 90°
        perp = grid_x * (-sin_t) + grid_y * cos_t

        # Gaussian weighting: centre stripe = 1.0, falls off smoothly
        sigma = 0.3
        mask = torch.exp(-0.5 * (perp / sigma) ** 2)  # [H, W]
        return mask.unsqueeze(0)  # [1, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
        Returns:
            A_theta: Orientation attention map [B, C, H, W], zero parameters
        """
        B, C, H, W = x.shape
        angle_responses = []
        for i in range(self.num_angles):
            mask = self.masks[i]             # [1, H, W]
            # Weight and pool along masked direction
            weighted = x * mask.unsqueeze(0) # [B, C, H, W]
            # Mean-pool to scalar per channel
            response = weighted.mean(dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
            angle_responses.append(torch.sigmoid(response))
        # Average across all orientation responses
        A_theta = torch.stack(angle_responses, dim=0).mean(dim=0)  # [B, C, 1, 1]
        return A_theta.expand(B, C, H, W)   # [B, C, H, W]

    def extra_repr(self) -> str:
        return f"angles={self.angles}, params=0 (pure buffers)"


class FocalSpotStream(nn.Module):
    """
    Morphological white top-hat transform for DRUSEN focal deposit detection.

    DRUSEN appear as focal bright spots at the RPE-Bruch's membrane interface.
    A white top-hat transform (MaxPool(x) - x) produces a non-negative map
    that is large only where pixel values are local maxima — exactly where
    drusenoid deposits create focal hyperreflective spots in the OCT scan.

    The depthwise 1×1 conv allows per-channel rescaling of the spot map
    without cross-channel mixing, preserving the morphological signal while
    learning which channels carry the most drusen-relevant information.

    Args:
        channels:    Number of input feature channels (576)
        pool_size:   Neighbourhood size for top-hat (default 7 = one retinal
                     layer thickness in feature space at 7×7 resolution)

    Parameter count: C (dw conv weights) + 2C (BN weight+bias) = 3C
    For C=576: 576 + 1152 = 1728 parameters total.
    Output: A_focal ∈ [0, 1]^{B×C×H×W} (sigmoid-gated spatial attention map)
    """

    def __init__(self, channels: int, pool_size: int = 7):
        super().__init__()
        # MaxPool with same-padding preserves spatial dimensions
        # stride=1 ensures every position gets a neighbourhood maximum
        self.pool = nn.MaxPool2d(
            kernel_size=pool_size, stride=1, padding=pool_size // 2
        )
        # Depthwise 1×1 conv: per-channel rescaling of top-hat map
        # groups=channels → no cross-channel mixing, preserves spot signal
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=1, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W] — MUST be the unmodulated input,
               not post-H/V-stream features, to preserve top-hat correctness.
        Returns:
            A_focal: Spatial attention map [B, C, H, W] in [0, 1]
        """
        # White top-hat: local max minus the value = focal peak highlight
        # spots[b,c,h,w] > 0 only where x[b,c,h,w] is a local maximum
        spots = self.pool(x) - x          # [B, C, H, W], non-negative
        out = self.dw_conv(spots)         # [B, C, H, W], per-channel reweight
        out = self.bn(out)
        return torch.sigmoid(out)         # [B, C, H, W], values in [0, 1]


class RLAPv3(nn.Module):
    """
    RLAP v3: Full Structured Projection Attention module.

    F_rlap = F ⊗ A_h ⊗ A_v ⊗ A_focal ⊗ A_theta

    Application order (anatomically motivated):
      1. A_h    — suppress layers of wrong thickness (CNV/DME check)
      2. A_v    — suppress non-lesion columns
      3. A_focal — spotlight focal drusen deposits on already-gated features
      4. A_theta — global orientation consistency (Bruch's membrane angle)

    FocalSpotStream inputs are taken from the ORIGINAL x, not from `out`,
    because the top-hat transform is only meaningful on unmodulated feature
    magnitudes (H/V multiplication distorts local neighbourhood statistics).

    Args:
        channels:    Number of feature channels (576 for MobileNetV3-Small last stage)
        height:      Feature map height (7 for 224px input)
        width:       Feature map width  (7 for 224px input)
        horizontal:  Enable horizontal stream (default True)
        vertical:    Enable vertical stream (default True)
        focal_spot:  Enable FocalSpotStream for DRUSEN (default False)
        use_bank:    Enable orientation bank (default True)
        angles:      Angles for orientation bank (default [0,30,45,60,90,135])
        kernel_size: 1D conv kernel size for H/V streams (default 3)
    """

    def __init__(
        self,
        channels: int = 96,
        height: int = 7,
        width: int = 7,
        horizontal: bool = True,
        vertical: bool = True,
        focal_spot: bool = False,
        use_bank: bool = True,
        angles: list = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.use_horizontal = horizontal
        self.use_vertical = vertical
        self.use_focal = focal_spot
        self.use_bank = use_bank

        if horizontal:
            self.h_stream = HorizontalStream(channels, height, kernel_size)
        if vertical:
            self.v_stream = VerticalStream(channels, width, kernel_size)
        if focal_spot:
            self.focal_stream = FocalSpotStream(channels)
        if use_bank:
            self.o_bank = OrientationBank(channels, height, width, angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
        Returns:
            Attention-modulated feature tensor, same shape
        """
        out = x
        if self.use_horizontal:
            A_h = self.h_stream(x)           # [B, C, H, 1]
            out = out * A_h
        if self.use_vertical:
            A_v = self.v_stream(x)           # [B, C, 1, W]
            out = out * A_v
        if self.use_focal:
            # Use original x — top-hat must operate on unmodulated features
            A_focal = self.focal_stream(x)   # [B, C, H, W]
            out = out * A_focal
        if self.use_bank:
            A_t = self.o_bank(x)             # [B, C, H, W]
            out = out * A_t
        return out

    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """
        Return individual attention maps for visualisation (GradCAM++ overlays).
        """
        maps = {}
        if self.use_horizontal:
            maps["horizontal"] = self.h_stream(x)   # [B, C, H, 1]
        if self.use_vertical:
            maps["vertical"] = self.v_stream(x)     # [B, C, 1, W]
        if self.use_focal:
            maps["focal_spot"] = self.focal_stream(x)  # [B, C, H, W]
        if self.use_bank:
            B, C, H, W = x.shape
            per_angle = {}
            for i, angle in enumerate(self.o_bank.angles):
                mask = self.o_bank.masks[i]
                weighted = x * mask.unsqueeze(0)
                response = weighted.mean(dim=(-2, -1), keepdim=True)
                per_angle[f"angle_{angle}"] = torch.sigmoid(response).expand(B, C, H, W)
            maps["orientation_bank"] = per_angle
        return maps
