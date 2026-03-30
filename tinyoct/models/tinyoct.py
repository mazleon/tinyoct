"""
TinyOCT: Full model assembly.

Pipeline:
  Input (224×224)
    → LaplacianLayer        [frozen, 0 params]
    → MobileNetV3-Small     [~2.5M params, ImageNet pre-trained]
    → RLAPv3                [~3456 params in 1D convs, 0 in orientation bank]
    → GlobalAvgPool
    → PrototypeHead         [~4 × feature_dim params]
    → Temperature scaling   [1 param, post-hoc calibration]

Total trainable: ~4.5M params
RLAP structural overhead: ~3456 params (0.076% of total)
Inference: <5ms on CPU (batch size 1)
"""

import torch
import torch.nn as nn
import timm

from .laplacian import LaplacianLayer
from .rlap import RLAPv3
from .prototype_head import PrototypeHead


class TinyOCT(nn.Module):
    """
    TinyOCT: Anatomy-Guided Structured Projection Attention model.

    Args:
        cfg: OmegaConf / SimpleNamespace config object with model sub-config.
             See configs/base.yaml for all options.
    """

    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        self.num_classes = mc.num_classes
        self.feature_dim = mc.feature_dim
        self.use_prototype = mc.prototype.enabled

        # ── Dimension guard ─────────────────────────────────────────
        # MobileNetV3-Small outputs 576 channels from its last stage.
        # Catching misconfigurations here prevents silent matmul errors
        # downstream in PrototypeHead or nn.Linear.
        assert mc.feature_dim == 576, (
            f"feature_dim={mc.feature_dim} but MobileNetV3-Small outputs 576 "
            f"channels. Check cfg.model.feature_dim in your config."
        )

        # ── Stage 1: Frozen multi-scale Laplacian preprocessing ──────
        self.laplacian = LaplacianLayer(
            alpha=mc.laplacian.alpha,
            alpha_coarse=getattr(mc.laplacian, "alpha_coarse", 0.05),
        )

        # ── Stage 2: MobileNetV3-Small backbone ──────────────────────
        self.backbone = timm.create_model(
            mc.backbone,
            pretrained=mc.pretrained,
            features_only=True,  # return intermediate feature maps
            out_indices=[-1],    # only last feature map
        )
        # Freeze early layers (stages 0–3), fine-tune stages 4+
        self._freeze_early_layers()

        # ── Stage 3: RLAP v3 ─────────────────────────────────────────
        self.rlap = RLAPv3(
            channels=mc.feature_dim,
            height=mc.spatial_size,
            width=mc.spatial_size,
            horizontal=mc.rlap.horizontal,
            vertical=mc.rlap.vertical,
            focal_spot=getattr(mc.rlap, "focal_spot", False),
            use_bank=mc.rlap.orientation_bank,
            angles=mc.rlap.angles,
            kernel_size=mc.rlap.kernel_size,
        )

        # ── Stage 4: Global average pooling ──────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Stage 5: Classification head ─────────────────────────────
        if self.use_prototype:
            self.head = PrototypeHead(
                feature_dim=mc.feature_dim,
                num_classes=mc.num_classes,
                temperature=mc.prototype.temperature,
            )
        else:
            self.head = nn.Linear(mc.feature_dim, mc.num_classes)

        # ── Stage 6: Temperature scaling (post-hoc, set after training) ─
        self.log_temperature = nn.Parameter(
            torch.zeros(1), requires_grad=False  # enabled during calibration
        )

    def _freeze_early_layers(self):
        """Freeze early backbone layers. Fine-tune only last 2 blocks."""
        # Get all named children of the backbone
        children = list(self.backbone.named_children())
        # Freeze all but last 2
        for name, module in children[:-2]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, 224, 224]
            return_features: If True, also return pre-head features [B, feature_dim]
        Returns:
            logits: [B, num_classes]
            features (optional): [B, feature_dim]
        """
        # Stage 1: Laplacian
        x = self.laplacian(x)               # [B, 3, 224, 224]

        # Stage 2: Backbone
        features = self.backbone(x)          # list; take last element
        F_map = features[-1]                 # [B, 576, 7, 7]

        # Stage 3: RLAP
        F_rlap = self.rlap(F_map)           # [B, 576, 7, 7]

        # Stage 4: Pooling — focal-aware when FocalSpotStream is active
        # Standard GAP destroys the spatial locality that FocalSpotStream
        # was designed to preserve (DRUSEN focal deposit signals averaged
        # away). Attention-weighted pooling retains focal spot emphasis.
        if self.use_prototype and self.rlap.use_focal:
            A_focal = self.rlap.focal_stream(F_map)  # [B, 576, 7, 7]
            weights = A_focal / (A_focal.sum(dim=(-2, -1), keepdim=True) + 1e-8)  # [B, 576, 7, 7]
            flat = (F_rlap * weights).sum(dim=(-2, -1))  # [B, 576]
        else:
            pooled = self.gap(F_rlap)        # [B, 576, 1, 1]
            flat = pooled.flatten(1)         # [B, 576]

        # Stage 5: Head
        logits = self.head(flat)             # [B, num_classes]

        # Stage 6: Temperature scaling (active ONLY during post-hoc calibration)
        # During training, PrototypeHead already applies its own temperature
        # (T=0.07). Applying a second temperature here would double-scale
        # logits, distorting gradients and inflating confidence. The post-hoc
        # temperature is only unlocked during calibration (requires_grad=True).
        if self.log_temperature.requires_grad:
            T = self.log_temperature.exp()   # [1]
            logits = logits / T              # [B, num_classes]

        if return_features:
            return logits, flat
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """Extract RLAP attention maps for visualisation (Week 7)."""
        with torch.no_grad():
            x = self.laplacian(x)
            features = self.backbone(x)
            F_map = features[-1]
            return self.rlap.get_attention_maps(F_map)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rlap_params = sum(p.numel() for p in self.rlap.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "rlap": rlap_params,
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }
