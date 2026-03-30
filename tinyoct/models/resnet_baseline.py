"""
ResNet18 Baseline for TinyOCT comparison experiments.

A standard ResNet18 (ImageNet-pretrained) with the same training API as TinyOCT,
providing a fair CNN baseline for ablation and comparison studies.

Key design decisions:
  - Same forward(x, return_features=True) signature as TinyOCT
  - Same count_parameters() interface
  - Same freeze strategy (freeze early layers, fine-tune last block)
  - No RLAP, no Laplacian, no PrototypeHead — pure ResNet18 + FC
  - ~11.2M total params vs TinyOCT's ~4.5M (2.5× larger)

This baseline answers the reviewer question: "Does the performance come from
the architecture (RLAP) or just from transfer learning?"
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Baseline(nn.Module):
    """
    ResNet18 baseline model with identical training API to TinyOCT.

    Args:
        cfg: Config object with model sub-config. Only uses:
             - cfg.model.num_classes
             - cfg.model.pretrained
    """

    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        self.num_classes = mc.num_classes
        self.feature_dim = 512  # ResNet18 final feature channels

        # ── ResNet18 backbone ────────────────────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if mc.pretrained else None
        backbone = models.resnet18(weights=weights)

        # Extract everything except the final FC layer
        # ResNet18 layers: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.conv1 = backbone.conv1        # [B, 64, 112, 112]
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool    # [B, 64, 56, 56]
        self.layer1 = backbone.layer1      # [B, 64, 56, 56]
        self.layer2 = backbone.layer2      # [B, 128, 28, 28]
        self.layer3 = backbone.layer3      # [B, 256, 14, 14]
        self.layer4 = backbone.layer4      # [B, 512, 7, 7]
        self.avgpool = backbone.avgpool    # [B, 512, 1, 1]

        # ── Classification head ──────────────────────────────────────
        self.fc = nn.Linear(512, mc.num_classes)  # [B, num_classes]

        # ── Freeze early layers (same strategy as TinyOCT) ───────────
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze conv1 + layer1-2, fine-tune layer3-4 + FC.

        This parallels TinyOCT's strategy of freezing early backbone
        stages while fine-tuning later ones. Keeps comparison fair.
        """
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
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
            return_features: If True, also return pre-FC features [B, 512]
        Returns:
            logits: [B, num_classes]
            features (optional): [B, 512]
        """
        # Stem
        x = self.conv1(x)       # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 56, 56]

        # Residual blocks
        x = self.layer1(x)      # [B, 64, 56, 56]
        x = self.layer2(x)      # [B, 128, 28, 28]
        x = self.layer3(x)      # [B, 256, 14, 14]
        x = self.layer4(x)      # [B, 512, 7, 7]

        # Global average pool
        x = self.avgpool(x)     # [B, 512, 1, 1]
        flat = torch.flatten(x, 1)  # [B, 512]

        # Classification head
        logits = self.fc(flat)  # [B, num_classes]

        if return_features:
            return logits, flat
        return logits

    def count_parameters(self) -> dict:
        """Same interface as TinyOCT.count_parameters()."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Backbone = everything except FC
        backbone_params = sum(
            p.numel() for name, p in self.named_parameters()
            if not name.startswith("fc.")
        )

        return {
            "total": total,
            "trainable": trainable,
            "rlap": 0,  # No RLAP in baseline
            "backbone": backbone_params,
            "head": sum(p.numel() for p in self.fc.parameters()),
        }
