"""
LaplacianLayer: Frozen frequency-domain preprocessing.

Sharpens retinal layer boundaries before the backbone sees the image.
This amplifies the exact signal that RLAP's horizontal stream detects.

Key properties:
  - ZERO trainable parameters (register_buffer only)
  - Residual addition: output = input + alpha * laplacian_response
  - alpha=0.1 is the recommended starting value

Paper claim: "frequency bias alignment with anatomical boundary detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLayer(nn.Module):
    """
    Zero-parameter Laplacian edge enhancement layer.

    Args:
        alpha: Residual blend strength (default 0.1).
               Higher alpha = stronger edge emphasis.
               Ablate: set alpha=0 to disable without removing the module.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

        # 3×3 Laplacian kernel — detects layer boundaries
        kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]], dtype=torch.float32
        )
        # Shape: [out_channels=1, in_channels=1, H=3, W=3]
        kernel = kernel.view(1, 1, 3, 3)
        # register_buffer: moves to GPU with model, NOT counted as parameters
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W], values in ~[-2, 2] (ImageNet normalised)
        Returns:
            Boundary-enhanced tensor, same shape as input
        """
        # Convert to grayscale-equivalent for edge detection
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        # Apply Laplacian (padding=1 preserves spatial size)
        edges = F.conv2d(gray, self.kernel, padding=1)  # [B, 1, H, W]
        # Broadcast edge response to all channels and add residually
        edge_broadcast = edges.expand_as(x)  # [B, C, H, W]
        return x + self.alpha * edge_broadcast

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, params=0 (frozen buffer)"
