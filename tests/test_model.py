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
                focal_spot=False,       # disabled by default; R6/R9 tests enable this
                angles=[0, 30, 45, 60, 90, 135],
                kernel_size=3,
            ),
            prototype=SimpleNamespace(enabled=True, temperature=0.07),
            laplacian=SimpleNamespace(enabled=True, alpha=0.1, alpha_coarse=0.05),
        ),
        train=SimpleNamespace(
            loss=SimpleNamespace(
                ce_weight=1.0, supcon_weight=0.1, orient_weight=0.05,
                orient_angle_range=5, orient_temperature=2.0,
                focal_gamma=0.0,        # 0.0 = standard CE (backward compat)
            ),
            supcon=SimpleNamespace(temperature=0.07, margin=0.0),
        ),
        data=SimpleNamespace(class_weights=None),
    )


# ── LaplacianLayer tests ──────────────────────────────────────────────────────
class TestLaplacianLayer:
    def test_zero_parameters(self):
        from tinyoct.models.laplacian import LaplacianLayer
        layer = LaplacianLayer(alpha=0.1)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params == 0, f"LaplacianLayer should have 0 params, got {n_params}"

    def test_output_shape(self):
        from tinyoct.models.laplacian import LaplacianLayer
        layer = LaplacianLayer()
        x = torch.randn(2, 3, 224, 224)
        out = layer(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"


# ── RLAP tests ────────────────────────────────────────────────────────────────
class TestRLAPv3:
    def test_orientation_bank_zero_params(self):
        """CRITICAL: OrientationBank must have zero trainable parameters."""
        from tinyoct.models.rlap import OrientationBank
        bank = OrientationBank(channels=96, height=7, width=7)
        n_params = sum(p.numel() for p in bank.parameters())
        assert n_params == 0, (
            f"OrientationBank must have 0 parameters for the zero-param claim. "
            f"Got {n_params}. Check register_buffer usage."
        )

    def test_rlap_output_shape(self):
        from tinyoct.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, height=7, width=7)
        x = torch.randn(4, 96, 7, 7)
        out = rlap(x)
        assert out.shape == x.shape, f"RLAP output shape mismatch: {out.shape}"

    def test_horizontal_only(self):
        from tinyoct.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, horizontal=True, vertical=False, use_bank=False)
        x = torch.randn(2, 96, 7, 7)
        out = rlap(x)
        assert out.shape == x.shape

    def test_attention_maps_returned(self):
        from tinyoct.models.rlap import RLAPv3
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
        from tinyoct.models.prototype_head import PrototypeHead
        head = PrototypeHead(feature_dim=96, num_classes=4)
        x = torch.randn(8, 96)
        logits = head(x)
        assert logits.shape == (8, 4), f"Head output shape: {logits.shape}"

    def test_similarities_range(self):
        from tinyoct.models.prototype_head import PrototypeHead
        head = PrototypeHead(feature_dim=96, num_classes=4)
        x = torch.randn(4, 96)
        sims = head.get_similarities(x)
        assert sims.min() >= -1.01 and sims.max() <= 1.01, "Cosine similarity out of [-1, 1]"


# ── Full model tests ──────────────────────────────────────────────────────────
class TestTinyOCT:
    @pytest.fixture
    def model(self):
        from tinyoct.models.tinyoct import TinyOCT
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
        from tinyoct.losses.supcon_loss import BalancedSupConLoss
        loss_fn = BalancedSupConLoss()
        features = torch.randn(16, 96)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.item() >= 0, "SupCon loss must be non-negative"
        assert not torch.isnan(loss), "SupCon loss is NaN"

    def test_orient_loss(self):
        from tinyoct.models.tinyoct import TinyOCT
        from tinyoct.losses.orient_loss import OrientationConsistencyLoss
        cfg = make_cfg()
        model = TinyOCT(cfg)
        loss_fn = OrientationConsistencyLoss(angle_range=5.0)
        x = torch.randn(4, 3, 224, 224)
        loss = loss_fn(model, x)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


# ── FocalLoss tests ───────────────────────────────────────────────────────────
class TestFocalLoss:
    def test_gamma_zero_equals_ce(self):
        """FL(gamma=0) must equal standard weighted CE numerically."""
        from tinyoct.losses.focal_loss import FocalLoss
        import torch.nn as nn
        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        labels = torch.randint(0, 4, (32,))
        weights = [0.48, 1.60, 2.20, 0.72]

        fl = FocalLoss(gamma=0.0, class_weights=weights)
        ce = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32)
        )
        fl_val = fl(logits, labels).item()
        ce_val = ce(logits, labels).item()
        assert abs(fl_val - ce_val) < 1e-4, (
            f"FL(gamma=0) = {fl_val:.6f} should equal CE = {ce_val:.6f}"
        )

    def test_output_scalar(self):
        from tinyoct.losses.focal_loss import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_no_nan(self):
        from tinyoct.losses.focal_loss import FocalLoss
        loss_fn = FocalLoss(gamma=2.0, class_weights=[0.48, 1.60, 2.20, 0.72])
        logits = torch.randn(16, 4)
        labels = torch.randint(0, 4, (16,))
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "FocalLoss produced NaN"
        assert loss.item() >= 0, "FocalLoss must be non-negative"

    def test_gamma2_lower_than_gamma0_on_easy_samples(self):
        """Higher gamma should reduce loss contribution from easy (confident) samples."""
        from tinyoct.losses.focal_loss import FocalLoss
        # Construct logits where the model is very confident about the right class
        logits = torch.zeros(4, 4)
        logits[0, 0] = 10.0; logits[1, 1] = 10.0
        logits[2, 2] = 10.0; logits[3, 3] = 10.0
        labels = torch.tensor([0, 1, 2, 3])

        loss_gamma0 = FocalLoss(gamma=0.0)(logits, labels).item()
        loss_gamma2 = FocalLoss(gamma=2.0)(logits, labels).item()
        assert loss_gamma2 < loss_gamma0, (
            f"gamma=2 loss ({loss_gamma2:.6f}) should be < gamma=0 loss ({loss_gamma0:.6f}) "
            f"on easy samples (model is very confident and correct)"
        )


# ── FocalSpotStream tests ─────────────────────────────────────────────────────
class TestFocalSpotStream:
    def test_output_shape(self):
        from tinyoct.models.rlap import FocalSpotStream
        stream = FocalSpotStream(channels=96)
        x = torch.randn(4, 96, 7, 7)
        out = stream(x)
        assert out.shape == x.shape, f"FocalSpotStream shape: {out.shape} != {x.shape}"

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        from tinyoct.models.rlap import FocalSpotStream
        stream = FocalSpotStream(channels=96)
        x = torch.randn(4, 96, 7, 7)
        out = stream(x)
        assert out.min() >= 0.0 and out.max() <= 1.0, (
            f"FocalSpotStream output out of [0,1]: [{out.min():.4f}, {out.max():.4f}]"
        )

    def test_param_count(self):
        """1728 params for C=576: 576 (dw conv) + 1152 (BN weight+bias)."""
        from tinyoct.models.rlap import FocalSpotStream
        stream = FocalSpotStream(channels=576)
        n_params = sum(p.numel() for p in stream.parameters())
        assert n_params == 1728, f"Expected 1728 params for C=576, got {n_params}"

    def test_rlap_with_focal_spot_shape(self):
        from tinyoct.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, height=7, width=7, focal_spot=True)
        x = torch.randn(2, 96, 7, 7)
        out = rlap(x)
        assert out.shape == x.shape

    def test_rlap_attention_maps_include_focal(self):
        from tinyoct.models.rlap import RLAPv3
        rlap = RLAPv3(channels=96, height=7, width=7, focal_spot=True)
        x = torch.randn(1, 96, 7, 7)
        maps = rlap.get_attention_maps(x)
        assert "focal_spot" in maps, "focal_spot key missing from attention maps"
        assert maps["focal_spot"].shape == (1, 96, 7, 7)


# ── MarginSupCon tests ────────────────────────────────────────────────────────
class TestMarginSupCon:
    def test_margin_zero_matches_original(self):
        """margin=0.0 must produce the same loss as the base BalancedSupConLoss."""
        from tinyoct.losses.supcon_loss import BalancedSupConLoss
        torch.manual_seed(42)
        features = torch.randn(16, 96)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        loss_no_margin = BalancedSupConLoss(margin=0.0)(features, labels).item()
        loss_orig      = BalancedSupConLoss()(features, labels).item()
        assert abs(loss_no_margin - loss_orig) < 1e-5, (
            f"margin=0 loss ({loss_no_margin:.6f}) should equal base ({loss_orig:.6f})"
        )

    def test_margin_positive_increases_loss(self):
        """Positive margin makes the denominator harder → higher loss for separated clusters."""
        from tinyoct.losses.supcon_loss import BalancedSupConLoss
        torch.manual_seed(42)
        features = torch.randn(16, 96)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        loss_0   = BalancedSupConLoss(margin=0.0)(features, labels).item()
        loss_0_3 = BalancedSupConLoss(margin=0.3)(features, labels).item()
        assert loss_0_3 > loss_0, (
            f"margin=0.3 ({loss_0_3:.4f}) should be > margin=0 ({loss_0:.4f})"
        )

    def test_no_nan_with_margin(self):
        from tinyoct.losses.supcon_loss import BalancedSupConLoss
        loss_fn = BalancedSupConLoss(margin=0.3)
        features = torch.randn(16, 96)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        loss = loss_fn(features, labels)
        assert not torch.isnan(loss), "MarginSupCon produced NaN"
