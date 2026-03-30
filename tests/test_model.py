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
                proto_weight=0.01,      # prototype separation penalty
                proto_margin=-0.1,      # cosine similarity margin
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


# ── Feature Dim Assertion tests ───────────────────────────────────────────────
class TestFeatureDimAssertion:
    def test_correct_dim_passes(self):
        """feature_dim=576 should create model without error."""
        from tinyoct.models.tinyoct import TinyOCT
        cfg = make_cfg()
        model = TinyOCT(cfg)  # should not raise
        assert model.feature_dim == 576

    def test_wrong_dim_raises(self):
        """feature_dim != 576 must raise AssertionError at model init."""
        from tinyoct.models.tinyoct import TinyOCT
        cfg = make_cfg()
        cfg.model.feature_dim = 96  # wrong!
        with pytest.raises(AssertionError, match="576"):
            TinyOCT(cfg)


# ── Orthogonal Prototype Init tests ───────────────────────────────────────────
class TestOrthogonalPrototypeInit:
    def test_prototypes_are_orthogonal(self):
        """After init, all prototype pairs should have near-zero cosine similarity."""
        from tinyoct.models.prototype_head import PrototypeHead
        import torch.nn.functional as F
        head = PrototypeHead(feature_dim=576, num_classes=4)
        protos = F.normalize(head.prototypes.data, dim=1)  # [4, 576]
        sim = torch.matmul(protos, protos.T)  # [4, 4]
        # Off-diagonal should be near zero
        off_diag = sim - torch.eye(4)
        assert off_diag.abs().max() < 0.05, (
            f"Prototypes are not orthogonal: max off-diagonal cosine sim = {off_diag.abs().max():.4f}"
        )

    def test_prototypes_are_unit_norm(self):
        """All prototypes should be unit-normalised after init."""
        from tinyoct.models.prototype_head import PrototypeHead
        head = PrototypeHead(feature_dim=576, num_classes=4)
        norms = head.prototypes.data.norm(dim=1)  # [4]
        assert torch.allclose(norms, torch.ones(4), atol=1e-4), (
            f"Prototype norms: {norms.tolist()}"
        )


# ── Double Temperature Scaling tests ──────────────────────────────────────────
class TestTemperatureScaling:
    def test_no_double_scaling_during_training(self):
        """When log_temperature.requires_grad=False, logits must NOT be divided by T."""
        from tinyoct.models.tinyoct import TinyOCT
        cfg = make_cfg()
        model = TinyOCT(cfg)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        # At init, log_temperature=0 → T=1.0
        # If double scaling were active, logits would be divided twice
        # We verify by checking that log_temperature.requires_grad is False
        assert not model.log_temperature.requires_grad
        # Verify logits are reasonable (not blown up by 1/0.07 double scaling)
        assert logits.abs().max() < 200, (
            f"Logits suspiciously large ({logits.abs().max():.1f}), "
            f"possible double temperature scaling"
        )

    def test_calibration_mode_enables_scaling(self):
        """Enabling requires_grad on log_temperature should activate post-hoc scaling."""
        from tinyoct.models.tinyoct import TinyOCT
        cfg = make_cfg()
        model = TinyOCT(cfg)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits_before = model(x).clone()
        # Enable calibration
        model.log_temperature.requires_grad_(True)
        model.log_temperature.data.fill_(0.5)  # T = exp(0.5) ≈ 1.65
        with torch.no_grad():
            logits_after = model(x)
        # Logits should be different (divided by T ≈ 1.65)
        assert not torch.allclose(logits_before, logits_after, atol=1e-3), (
            "Enabling calibration temperature did not change logits"
        )


# ── Focal-Aware Pooling tests ─────────────────────────────────────────────────
class TestFocalAwarePooling:
    def test_focal_pooling_output_shape(self):
        """With focal_spot=True, model should still produce correct output shape."""
        from tinyoct.models.tinyoct import TinyOCT
        cfg = make_cfg()
        cfg.model.rlap.focal_spot = True  # enable focal stream
        model = TinyOCT(cfg)
        x = torch.randn(2, 3, 224, 224)
        logits, features = model(x, return_features=True)
        assert logits.shape == (2, 4), f"Logits shape: {logits.shape}"
        assert features.shape == (2, 576), f"Features shape: {features.shape}"


# ── Prototype Separation Loss tests ───────────────────────────────────────────
class TestPrototypeSeparationLoss:
    def test_output_non_negative(self):
        from tinyoct.losses.proto_loss import PrototypeSeparationLoss
        loss_fn = PrototypeSeparationLoss(margin=-0.1)
        protos = torch.randn(4, 576)
        loss = loss_fn(protos)
        assert loss.item() >= 0, f"Proto loss should be non-negative, got {loss.item()}"

    def test_no_nan(self):
        from tinyoct.losses.proto_loss import PrototypeSeparationLoss
        loss_fn = PrototypeSeparationLoss(margin=-0.1)
        protos = torch.randn(4, 576)
        loss = loss_fn(protos)
        assert not torch.isnan(loss), "Proto separation loss produced NaN"

    def test_orthogonal_prototypes_low_loss(self):
        """Orthogonal prototypes should have very low separation loss."""
        from tinyoct.losses.proto_loss import PrototypeSeparationLoss
        import torch.nn.functional as F
        loss_fn = PrototypeSeparationLoss(margin=-0.1)
        # Create orthogonal prototypes
        proto_init = torch.empty(4, 576)
        torch.nn.init.orthogonal_(proto_init)
        protos = F.normalize(proto_init, dim=1)
        loss = loss_fn(protos).item()
        assert loss < 0.02, f"Orthogonal protos should give near-zero loss, got {loss:.4f}"

    def test_identical_prototypes_high_loss(self):
        """Identical prototypes should produce high separation loss."""
        from tinyoct.losses.proto_loss import PrototypeSeparationLoss
        loss_fn = PrototypeSeparationLoss(margin=-0.1)
        # All prototypes are the same vector
        protos = torch.randn(1, 576).expand(4, -1).clone()
        loss = loss_fn(protos).item()
        assert loss > 1.0, f"Identical protos should give high loss, got {loss:.4f}"

    def test_combined_loss_includes_proto(self):
        """CombinedLoss should include proto term when proto_weight > 0."""
        from tinyoct.models.tinyoct import TinyOCT
        from tinyoct.losses.combined_loss import CombinedLoss
        cfg = make_cfg()
        cfg.train.loss.proto_weight = 0.01
        model = TinyOCT(cfg)
        loss_fn = CombinedLoss(cfg)
        x = torch.randn(4, 3, 224, 224)
        logits, features = model(x, return_features=True)
        labels = torch.tensor([0, 1, 2, 3])
        losses = loss_fn(model, x, logits, features, labels)
        assert "proto" in losses, "proto key missing from CombinedLoss output"
        assert losses["proto"].item() >= 0
