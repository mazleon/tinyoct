"""
LaplacianLayer: Frozen frequency-domain preprocessing.

Sharpens retinal layer boundaries before the backbone sees the image.
This amplifies the exact signal that RLAP's horizontal stream detects.

Key properties:
  - ZERO trainable parameters (register_buffer only)
  - Multi-scale residual addition:
      output = input + alpha * edges_fine + alpha_coarse * edges_coarse
  - Fine (3×3): sharp retinal layer boundaries for CNV/DME
  - Coarse (5×5): diffuse drusenoid deposit boundaries for DRUSEN
  - alpha=0.1, alpha_coarse=0.05 are the recommended starting values

Scientific basis for dual-scale design:
  DRUSEN are sub-RPE drusenoid deposits with diffuse, low-contrast spatial
  extent (~50-200μm). The 3×3 kernel responds to high-spatial-frequency edges
  (CNV/DME layer boundaries) but suppresses the broad, shallow contrast
  gradients characteristic of drusen. The 5×5 coarse kernel captures the
  intermediate spatial scale where drusen deposits are most discriminative.
  Both kernels sum to zero (no DC response), ensuring pure edge detection.

Paper claim: "multi-scale frequency bias alignment with anatomical boundary
detection — fine scale for sharp retinal layer boundaries, coarse scale for
diffuse drusenoid deposits"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLayer(nn.Module):
    """
    Zero-parameter multi-scale Laplacian edge enhancement layer.

    Args:
        alpha:        Fine (3×3) residual blend strength (default 0.1).
        alpha_coarse: Coarse (5×5) residual blend strength (default 0.05).
                      Set to 0.0 to disable coarse scale (ablation).
    """

    def __init__(self, alpha: float = 0.1, alpha_coarse: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.alpha_coarse = alpha_coarse

        # Fine 3×3 Laplacian kernel — detects sharp retinal layer boundaries
        # (CNV subretinal fluid margins, DME focal fluid edges)
        kernel_fine = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]], dtype=torch.float32
        )
        self.register_buffer("kernel_fine", kernel_fine.view(1, 1, 3, 3))

        # Coarse 5×5 Laplacian-of-Gaussian approximation — detects diffuse
        # drusenoid deposits at Bruch's membrane. Kernel sums to zero (no DC).
        # Derived from LoG at σ=1.4px: captures ~3× broader spatial scale.
        kernel_coarse = torch.tensor(
            [[ 0.,  0., -1.,  0.,  0.],
             [ 0., -1., -2., -1.,  0.],
             [-1., -2., 16., -2., -1.],
             [ 0., -1., -2., -1.,  0.],
             [ 0.,  0., -1.,  0.,  0.]], dtype=torch.float32
        )
        self.register_buffer("kernel_coarse", kernel_coarse.view(1, 1, 5, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W], values in ~[-2, 2] (ImageNet normalised)
        Returns:
            Multi-scale boundary-enhanced tensor, same shape as input
        """
        # Grayscale-equivalent for edge detection (mean over colour channels)
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Fine-scale edges: 3×3, padding=1 preserves spatial size
        edges_fine = F.conv2d(gray, self.kernel_fine, padding=1)    # [B, 1, H, W]

        # Coarse-scale edges: 5×5, padding=2 preserves spatial size
        edges_coarse = F.conv2d(gray, self.kernel_coarse, padding=2) # [B, 1, H, W]

        # Residual blend: add both edge responses to all colour channels
        # expand_as broadcasts [B, 1, H, W] → [B, C, H, W] without copy
        out = (x
               + self.alpha * edges_fine.expand_as(x)
               + self.alpha_coarse * edges_coarse.expand_as(x))
        return out

    def extra_repr(self) -> str:
        return (f"alpha_fine={self.alpha}, alpha_coarse={self.alpha_coarse}, "
                f"params=0 (frozen buffers)")
