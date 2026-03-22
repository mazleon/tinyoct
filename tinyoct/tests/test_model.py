"""
Unit tests for TinyOCT.
Run: pytest tests/ -v

Key assertions:
  - Zero-parameter claim for RLAP orientation bank
  - Output shape correctness
  - Attention map shapes
  - Loss function forward pass
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from types import SimpleNamespace


# ── Minimal config fixture ────────────────────────────────────────────────────
def make_cfg():
    return SimpleNamespace(
        model=SimpleNamespace(
            backbone="mobilenetv3_small_100",
            pretrained=False,           # no download in tests
            feature_dim=576,            # true last-stage channels from features_only=True
            spatial_size=7,
            num_classes=4,
            rlap=SimpleNamespace(
                horizontal=True,
                vertical=True,
                orientation_bank=True,
                angles=[0, 30, 45, 60, 90, 135],
                kernel_size=3,
            ),
            prototype=SimpleNamespace(enabled=True, temperature=0.07),
            laplacian=SimpleNamespace(enabled=True, alpha=0.1),
        ),
        train=SimpleNamespace(
            loss=SimpleNamespace(
                ce_weight=1.0, supcon_weight=0.1, orient_weight=0.05,
                orient_angle_range=5, orient_temperature=2.0,
            ),
            supcon=SimpleNamespace(temperature=0.07),
        ),
        data=SimpleNamespace(class_weights=None),
    )


# ── LaplacianLayer tests ──────────────────────────────────────────────────────
class TestLaplacianLayer:
    def test_zero_parameters(self):
        from src.models.laplacian import LaplacianLayer
        layer = LaplacianLayer(alpha=0.1)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params == 0, f"LaplacianLayer should have 0 params, got {n_params}"

    def test_output_shape(self):
        from src.models.laplacian import LaplacianLayer
        layer = LaplacianLayer()
        x = torch.randn(2, 3, 224, 224)
        out = layer(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"


# ── RLAP tests ────────────────────────────────────────────────────────────────
class TestRLAPv3:
    def test_orientation_bank_zero_params(self):
        """CRITICAL: OrientationBank must have zero trainable parameters."""
        from src.models.rlap import OrientationBank
        bank = OrientationBank(channels=96, height=7, width=7)
        n_params = sum(p.numel() for p in bank.parameters())
        assert n_params == 0, (
            f"OrientationBank must have 0 parameters for the zero-param claim. "
            f"Got {n_params}. Check register_buffer usage."
        )

    def test_rlap_output_shape(self):
        from src.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, height=7, width=7)
        x = torch.randn(4, 96, 7, 7)
        out = rlap(x)
        assert out.shape == x.shape, f"RLAP output shape mismatch: {out.shape}"

    def test_horizontal_only(self):
        from src.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, horizontal=True, vertical=False, use_bank=False)
        x = torch.randn(2, 96, 7, 7)
        out = rlap(x)
        assert out.shape == x.shape

    def test_attention_maps_returned(self):
        from src.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, height=7, width=7)
        x = torch.randn(1, 96, 7, 7)
        maps = rlap.get_attention_maps(x)
        assert "horizontal" in maps
        assert "vertical" in maps
        assert "orientation_bank" in maps
        assert len(maps["orientation_bank"]) == 6  # 6 angles


# ── PrototypeHead tests ───────────────────────────────────────────────────────
class TestPrototypeHead:
    def test_output_shape(self):
        from src.models.prototype_head import PrototypeHead
        head = PrototypeHead(feature_dim=96, num_classes=4)
        x = torch.randn(8, 96)
        logits = head(x)
        assert logits.shape == (8, 4), f"Head output shape: {logits.shape}"

    def test_similarities_range(self):
        from src.models.prototype_head import PrototypeHead
        head = PrototypeHead(feature_dim=96, num_classes=4)
        x = torch.randn(4, 96)
        sims = head.get_similarities(x)
        assert sims.min() >= -1.01 and sims.max() <= 1.01, "Cosine similarity out of [-1, 1]"


# ── Full model tests ──────────────────────────────────────────────────────────
class TestTinyOCT:
    @pytest.fixture
    def model(self):
        from src.models.tinyoct import TinyOCT
        cfg = make_cfg()
        return TinyOCT(cfg)

    def test_forward_pass(self, model):
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 4), f"Output shape: {logits.shape}"

    def test_forward_with_features(self, model):
        x = torch.randn(2, 3, 224, 224)
        logits, features = model(x, return_features=True)
        assert logits.shape == (2, 4)
        assert features.shape == (2, 576)  # 576 = true last-stage backbone channels

    def test_param_count_reasonable(self, model):
        params = model.count_parameters()
        assert params["total"] < 5_000_000, f"Model too large: {params['total']:,} params"
        print(f"\nTotal params: {params['total']:,}")


# ── Loss tests ────────────────────────────────────────────────────────────────
class TestLosses:
    def test_supcon_loss(self):
        from src.losses.supcon_loss import BalancedSupConLoss
        loss_fn = BalancedSupConLoss()
        features = torch.randn(16, 96)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.item() >= 0, "SupCon loss must be non-negative"
        assert not torch.isnan(loss), "SupCon loss is NaN"

    def test_orient_loss(self):
        from src.models.tinyoct import TinyOCT
        from src.losses.orient_loss import OrientationConsistencyLoss
        cfg = make_cfg()
        model = TinyOCT(cfg)
        loss_fn = OrientationConsistencyLoss(angle_range=5.0)
        x = torch.randn(4, 3, 224, 224)
        loss = loss_fn(model, x)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
