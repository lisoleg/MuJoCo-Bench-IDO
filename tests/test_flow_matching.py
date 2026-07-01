"""
Unit tests for FlowMatchingEtaPredictor (core/kappa_snap_mj.py)
=================================================================

Tests the FlowMatchingEtaPredictor:
  - η trajectory buffer management
  - Next η prediction via flow matching
  - Hesitation-RMSE computation
  - Retry-VOC computation
  - Stagnation detection
  - gauss_ex_residual backward compatibility with flow_predictor

Author: tomas-arc3-solver project · MuJoCo-Bench-IDO v0.2.0
"""
import numpy as np
import pytest

from core.kappa_snap_mj import (
    FlowMatchingEtaPredictor,
    gauss_ex_residual,
    IDO_KAPPA_SNAP_MJ_VERSION,
)
from core.goal_eml_mj import GoalEML


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_goal() -> GoalEML:
    """Create a simple GoalEML for testing."""
    return GoalEML(
        name='test_task',
        invariants=['ee_at_target'],
        target_pos=np.array([1.0, 0.0, 0.5]),
        delta_K=0.05,
        max_energy_inject=500.0,
    )


@pytest.fixture
def flow_predictor() -> FlowMatchingEtaPredictor:
    """Create a FlowMatchingEtaPredictor instance."""
    return FlowMatchingEtaPredictor(window_size=10)


# ── Version ───────────────────────────────────────────────────────


class TestFlowMatchingVersion:
    """Verify κ-Snap version upgrade."""

    def test_version_upgraded(self) -> None:
        """κ-Snap version should be v0.2.0 after upgrade."""
        assert IDO_KAPPA_SNAP_MJ_VERSION == "v0.2.0"


# ── Buffer Management ────────────────────────────────────────────


class TestFlowPredictorBuffer:
    """Test η trajectory buffer management."""

    def test_push_appends(self, flow_predictor) -> None:
        """Push adds η values to buffer."""
        flow_predictor.push(0.5)
        flow_predictor.push(0.4)
        flow_predictor.push(0.3)
        assert flow_predictor.eta_buffer == [0.5, 0.4, 0.3]

    def test_push_bounded(self, flow_predictor) -> None:
        """Buffer stays within window_size."""
        for i in range(20):
            flow_predictor.push(float(i))
        assert len(flow_predictor.eta_buffer) <= 10
        assert flow_predictor.eta_buffer[0] == 10.0  # last 10 values

    def test_clear(self, flow_predictor) -> None:
        """Clear resets the buffer."""
        flow_predictor.push(0.5)
        flow_predictor.push(0.4)
        flow_predictor.clear()
        assert flow_predictor.eta_buffer == []


# ── Prediction ────────────────────────────────────────────────────


class TestFlowPredictorPrediction:
    """Test η prediction via flow matching."""

    def test_predict_with_insufficient_data(self, flow_predictor) -> None:
        """Prediction returns last η with < 2 values."""
        flow_predictor.push(0.5)
        assert flow_predictor.predict_next_eta() == 0.5

    def test_predict_empty_buffer(self, flow_predictor) -> None:
        """Prediction returns 0.0 with empty buffer."""
        assert flow_predictor.predict_next_eta() == 0.0

    def test_predict_linear_descending(self, flow_predictor) -> None:
        """Prediction follows linear descending trend with correction."""
        # η descending: 1.0, 0.8, 0.6, 0.4, 0.2
        for eta in [1.0, 0.8, 0.6, 0.4, 0.2]:
            flow_predictor.push(eta)

        predicted: float = flow_predictor.predict_next_eta()

        # η(t) = 0.2, Δη(t) = 0.2 - 0.4 = -0.2
        # All deltas are -0.2, so residual correction = 0
        # η(t+1) ≈ 0.2 + (-0.2) + 0 = 0.0
        # But max(0, 0.0) = 0.0
        assert abs(predicted - 0.0) < 1e-4

    def test_predict_with_residual_correction(self, flow_predictor) -> None:
        """Prediction includes residual correction for non-linear trend."""
        # Non-linear: [1.0, 0.7, 0.5, 0.4, 0.35]
        for eta in [1.0, 0.7, 0.5, 0.4, 0.35]:
            flow_predictor.push(eta)

        predicted: float = flow_predictor.predict_next_eta()

        # η(t) = 0.35, Δη(t) = 0.35 - 0.4 = -0.05
        # Deltas: [-0.3, -0.2, -0.1, -0.05]
        # Residual errors: [-0.2-(-0.3)]=+0.1, [-0.1-(-0.2)]=+0.1, [-0.05-(-0.1)]=+0.05
        # residual_correction = mean(0.1, 0.1, 0.05) = 0.0833
        # η(t+1) = 0.35 + (-0.05) + 0.0833 = 0.3833
        # Non-negative check passes
        assert predicted >= 0.0

    def test_predict_non_negative(self, flow_predictor) -> None:
        """Predicted η is always non-negative."""
        # Fast descending: predictions might go negative
        for eta in [1.0, 0.8, 0.6, 0.4, 0.2]:
            flow_predictor.push(eta)
        predicted: float = flow_predictor.predict_next_eta()
        assert predicted >= 0.0


# ── Hesitation-RMSE ──────────────────────────────────────────────


class TestHesitationRMSE:
    """Test Hesitation-RMSE computation."""

    def test_empty_buffer(self, flow_predictor) -> None:
        """RMSE is 0.0 with empty buffer."""
        assert flow_predictor.compute_hesitation_rmse() == 0.0

    def test_constant_eta_zero_rmse(self, flow_predictor) -> None:
        """Constant η → RMSE = 0 (perfect plateau)."""
        for _ in range(5):
            flow_predictor.push(0.5)
        assert flow_predictor.compute_hesitation_rmse() == 0.0

    def test_varying_eta_positive_rmse(self, flow_predictor) -> None:
        """Varying η → RMSE > 0."""
        for eta in [0.3, 0.5, 0.7, 0.5, 0.3]:
            flow_predictor.push(eta)
        rmse: float = flow_predictor.compute_hesitation_rmse()
        assert rmse > 0.0

    def test_rmse_formula(self, flow_predictor) -> None:
        """RMSE = sqrt(mean((η_i - η_mean)^2))."""
        values: list = [0.3, 0.5, 0.7]
        for v in values:
            flow_predictor.push(v)
        mean: float = np.mean(values)
        expected: float = float(np.sqrt(np.mean([(v - mean) ** 2 for v in values])))
        assert abs(flow_predictor.compute_hesitation_rmse() - expected) < 1e-8


# ── Retry-VOC ─────────────────────────────────────────────────────


class TestRetryVOC:
    """Test Retry-VOC computation."""

    def test_empty_buffer(self, flow_predictor) -> None:
        """VOC is 0.0 with empty buffer."""
        assert flow_predictor.compute_retry_voc() == 0.0

    def test_single_value(self, flow_predictor) -> None:
        """VOC is 0.0 with only one η value."""
        flow_predictor.push(0.5)
        assert flow_predictor.compute_retry_voc() == 0.0

    def test_monotonic_descending_low_voc(self, flow_predictor) -> None:
        """Monotonic descending → all same signs → low VOC."""
        for eta in [1.0, 0.8, 0.6, 0.4, 0.2]:
            flow_predictor.push(eta)
        voc: float = flow_predictor.compute_retry_voc()
        # All signs are -1 → variance of [-1, -1, -1, -1] = 0
        assert abs(voc) < 1e-8

    def test_alternating_high_voc(self, flow_predictor) -> None:
        """Alternating up/down → high VOC."""
        for eta in [0.5, 0.7, 0.3, 0.9, 0.2]:
            flow_predictor.push(eta)
        voc: float = flow_predictor.compute_retry_voc()
        # Signs: [+1, -1, +1, -1] → variance > 0
        assert voc > 0.0


# ── Stagnation Detection ──────────────────────────────────────────


class TestStagnationDetection:
    """Test η stagnation detection."""

    def test_insufficient_data(self, flow_predictor) -> None:
        """No stagnation with < 2 values."""
        flow_predictor.push(0.5)
        assert not flow_predictor.detect_stagnation()

    def test_stagnation_detected(self, flow_predictor) -> None:
        """Stagnation detected when η barely changes."""
        for eta in [0.5000, 0.5001, 0.5002, 0.5003, 0.5004]:
            flow_predictor.push(eta)
        assert flow_predictor.detect_stagnation(threshold=0.01)

    def test_no_stagnation_with_progress(self, flow_predictor) -> None:
        """No stagnation when η is making meaningful progress."""
        for eta in [1.0, 0.5, 0.2, 0.1, 0.05]:
            flow_predictor.push(eta)
        assert not flow_predictor.detect_stagnation(threshold=0.01)


# ── gauss_ex_residual backward compatibility ──────────────────────


class TestGaussExResidualCompat:
    """Test gauss_ex_residual backward compatibility with flow_predictor."""

    def test_without_flow_predictor(self, simple_goal) -> None:
        """gauss_ex_residual works without flow_predictor (backward compat)."""
        z_i: dict = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'E_total': 100.0,
            'ee_vel': np.zeros(6),
        }
        # Without flow_predictor — should return pure η
        eta_base: float = gauss_ex_residual(z_i, simple_goal)
        assert eta_base > 0  # Not at target, η should be positive

    def test_with_flow_predictor(self, simple_goal) -> None:
        """gauss_ex_residual with flow_predictor returns trend-adjusted η."""
        z_i: dict = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'E_total': 100.0,
            'ee_vel': np.zeros(6),
        }
        predictor: FlowMatchingEtaPredictor = FlowMatchingEtaPredictor()

        # First call — pushes η, insufficient data for prediction → η unchanged
        eta1: float = gauss_ex_residual(z_i, simple_goal, flow_predictor=predictor)
        assert eta1 > 0

        # Second call with slightly different observation
        z_i2: dict = {
            'ee_pos': np.array([0.5, 0.0, 0.25]),  # closer to target
            'qpos': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'E_total': 100.0,
            'ee_vel': np.zeros(6),
        }
        eta2: float = gauss_ex_residual(z_i2, simple_goal, flow_predictor=predictor)
        assert eta2 > 0
        # η2 should be different from η1 due to different position
        assert eta2 != eta1

    def test_flow_predictor_none_same_as_no_predictor(self, simple_goal) -> None:
        """flow_predictor=None returns same result as no flow_predictor."""
        z_i: dict = {
            'ee_pos': np.array([0.5, 0.0, 0.25]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'E_total': 100.0,
            'ee_vel': np.zeros(6),
        }
        eta_no_fp: float = gauss_ex_residual(z_i, simple_goal)
        eta_fp_none: float = gauss_ex_residual(z_i, simple_goal, flow_predictor=None)
        assert abs(eta_no_fp - eta_fp_none) < 1e-12
