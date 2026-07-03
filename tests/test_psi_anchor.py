"""
Unit tests for ψ-Anchor (agent/psi_anchor.py)
===============================================

Tests the PsiAnchor meta-management layer:
  - η history update and trend analysis
  - Dynamic δ_K adjustment
  - Evolution policy decision
  - Noether conservation anchor injection
  - Epiplexity score computation
  - Self-evolution triggering
  - Macro evolution application

Author: tomas-arc3-solver project · MuJoCo-Bench-IDO v0.2.0
"""
import math
import numpy as np
import pytest

from core.goal_eml_mj import GoalEML
from agent.psi_anchor import (
    PsiAnchor, PsiAnchorState,
    IDO_PSI_ANCHOR_VERSION,
    ETA_TREND_WINDOW, PLATEAU_THRESHOLD,
    EPIPLEXITY_EVOLUTION_THRESH, PLATEAU_EVOLUTION_MIN_STEPS,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_goal() -> GoalEML:
    """Create a simple GoalEML for testing."""
    return GoalEML(
        name='test_task',
        invariants=['ee_at_target', 'torso_upright', 'no_self_collide'],
        target_pos=np.array([1.0, 0.0, 0.5]),
        delta_K=0.05,
        max_energy_inject=500.0,
        pos_tol=0.02,
        ori_tol=0.15,
    )


@pytest.fixture
def psi_anchor(simple_goal) -> PsiAnchor:
    """Create a PsiAnchor instance from simple_goal."""
    return PsiAnchor(simple_goal)


# ── Version ───────────────────────────────────────────────────────


class TestPsiAnchorVersion:
    """Verify ψ-Anchor version constant."""

    def test_version_string(self) -> None:
        """PsiAnchor version should be v0.3.1 (bug fix from v0.3.0)."""
        assert IDO_PSI_ANCHOR_VERSION == "v0.3.1"


# ── Initialization ────────────────────────────────────────────────


class TestPsiAnchorInit:
    """Test PsiAnchor initialization."""

    def test_init_basic(self, simple_goal) -> None:
        """PsiAnchor initializes correctly from GoalEML."""
        anchor = PsiAnchor(simple_goal)
        assert anchor.goal_eml is simple_goal
        assert anchor.eta_history == []
        assert anchor.evo_policy == 'light'
        assert anchor.conservation_anchors == simple_goal.invariants
        assert anchor.plateau_steps == 0
        assert anchor.adjusted_delta_K == simple_goal.delta_K

    def test_init_epiplexity_computed(self, simple_goal) -> None:
        """Epiplexity is computed on init."""
        anchor = PsiAnchor(simple_goal)
        expected: float = len(simple_goal.invariants) * (1.0 / simple_goal.delta_K) * math.log(simple_goal.max_energy_inject)
        assert abs(anchor.epiplexity_score - expected) < 1e-6


# ── η History Update ──────────────────────────────────────────────


class TestEtaHistory:
    """Test η history buffer management."""

    def test_update_eta_history_append(self, psi_anchor) -> None:
        """update_eta_history appends values to buffer."""
        psi_anchor.update_eta_history(0.5)
        psi_anchor.update_eta_history(0.4)
        psi_anchor.update_eta_history(0.3)
        assert psi_anchor.eta_history == [0.5, 0.4, 0.3]

    def test_update_eta_history_bounded(self, psi_anchor) -> None:
        """η history buffer stays within max length."""
        for i in range(150):
            psi_anchor.update_eta_history(float(i))
        assert len(psi_anchor.eta_history) <= 100
        # Should keep last 100 values
        assert psi_anchor.eta_history[0] == 50.0


# ── η Trend Analysis ──────────────────────────────────────────────


class TestEtaTrendAnalysis:
    """Test η trend classification."""

    def test_unknown_with_insufficient_data(self, psi_anchor) -> None:
        """Returns 'unknown' with < 3 η values."""
        psi_anchor.eta_history = [0.5]
        assert psi_anchor.analyze_eta_trend() == 'unknown'
        psi_anchor.eta_history = [0.5, 0.4]
        assert psi_anchor.analyze_eta_trend() == 'unknown'

    def test_descending_trend(self, psi_anchor) -> None:
        """Detects descending (converging) η trend."""
        psi_anchor.eta_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        trend: str = psi_anchor.analyze_eta_trend()
        assert trend == 'descending'

    def test_ascending_trend(self, psi_anchor) -> None:
        """Detects ascending (diverging) η trend."""
        psi_anchor.eta_history = [0.2, 0.4, 0.6, 0.8, 1.0]
        trend: str = psi_anchor.analyze_eta_trend()
        assert trend == 'ascending'

    def test_plateau_trend(self, psi_anchor) -> None:
        """Detects plateau (stalled) η trend."""
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003, 0.5004]
        trend: str = psi_anchor.analyze_eta_trend()
        assert trend == 'plateau'

    def test_plateau_counter_increment(self, psi_anchor) -> None:
        """Plateau counter increments on plateau detection."""
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003]
        psi_anchor.analyze_eta_trend()
        assert psi_anchor.plateau_steps >= 1

    def test_plateau_counter_reset_on_non_plateau(self, psi_anchor) -> None:
        """Plateau counter resets when trend is not plateau."""
        psi_anchor.plateau_steps = 5
        psi_anchor.eta_history = [1.0, 0.8, 0.6, 0.4]
        psi_anchor.analyze_eta_trend()
        assert psi_anchor.plateau_steps == 0


# ── Dynamic δ_K Adjustment ────────────────────────────────────────


class TestDeltaKAdjustment:
    """Test ψ-Anchor dynamic δ_K adjustment."""

    def test_tighten_on_descending(self, psi_anchor) -> None:
        """δ_K is tightened (×0.8) on descending trend."""
        psi_anchor.eta_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        adjusted: float = psi_anchor.adjust_delta_K(0.05)
        assert abs(adjusted - 0.05 * 0.8) < 1e-8

    def test_relax_on_plateau(self, psi_anchor) -> None:
        """δ_K is relaxed (×1.2) on plateau trend."""
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003, 0.5004]
        adjusted: float = psi_anchor.adjust_delta_K(0.05)
        assert abs(adjusted - 0.05 * 1.2) < 1e-8

    def test_freeze_on_ascending(self, psi_anchor) -> None:
        """δ_K is frozen on ascending trend."""
        psi_anchor.eta_history = [0.2, 0.4, 0.6, 0.8, 1.0]
        adjusted: float = psi_anchor.adjust_delta_K(0.05)
        assert abs(adjusted - 0.05) < 1e-8

    def test_freeze_on_unknown(self, psi_anchor) -> None:
        """δ_K is frozen on unknown trend."""
        psi_anchor.eta_history = [0.5]
        adjusted: float = psi_anchor.adjust_delta_K(0.05)
        assert abs(adjusted - 0.05) < 1e-8

    def test_clamped_bounds(self, psi_anchor) -> None:
        """Adjusted δ_K is clamped to [1e-4, 10.0]."""
        # Test lower bound
        psi_anchor.eta_history = [1.0, 0.8, 0.6]  # descending
        adjusted: float = psi_anchor.adjust_delta_K(1e-5)
        assert adjusted >= 1e-4

        # Test upper bound
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002]  # plateau
        adjusted = psi_anchor.adjust_delta_K(9.0)
        assert adjusted <= 10.0


# ── Evolution Policy ──────────────────────────────────────────────


class TestEvolutionPolicy:
    """Test ψ-Anchor evolution policy decision."""

    def test_plateau_low_epiplexity_freeze(self, psi_anchor) -> None:
        """Low epiplexity + plateau → freeze policy."""
        # Reduce epiplexity below threshold
        psi_anchor.epiplexity_score = 0.5
        psi_anchor.epiplexity_thresh = 2.0
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003, 0.5004]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'freeze'

    def test_plateau_high_epiplexity_light(self, psi_anchor) -> None:
        """High epiplexity + plateau → light evolution."""
        psi_anchor.epiplexity_score = 5.0
        psi_anchor.epiplexity_thresh = 2.0
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003, 0.5004]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'light'

    def test_descending_high_epiplexity_light(self, psi_anchor) -> None:
        """Descending + high epiplexity → light evolution."""
        psi_anchor.epiplexity_score = 5.0
        psi_anchor.epiplexity_thresh = 2.0
        psi_anchor.eta_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'light'

    def test_descending_low_epiplexity_freeze(self, psi_anchor) -> None:
        """Descending + low epiplexity → freeze."""
        psi_anchor.epiplexity_score = 0.5
        psi_anchor.epiplexity_thresh = 2.0
        psi_anchor.eta_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'freeze'

    def test_ascending_always_light(self, psi_anchor) -> None:
        """Ascending → always light evolution."""
        psi_anchor.eta_history = [0.2, 0.4, 0.6, 0.8, 1.0]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'light'

    def test_unknown_defaults_light(self, psi_anchor) -> None:
        """Unknown trend defaults to light evolution."""
        psi_anchor.eta_history = [0.5]
        policy: str = psi_anchor.decide_evolution_policy()
        assert policy == 'light'


# ── Conservation Anchor ──────────────────────────────────────────


class TestConservationAnchor:
    """Test Noether conservation anchor injection."""

    def test_noether_ok_full_score(self, psi_anchor) -> None:
        """Noether OK → conservation score = 1.0, no violations."""
        anchor_dict: dict = psi_anchor.inject_conservation_anchor(True, "")
        assert anchor_dict['ok'] is True
        assert anchor_dict['violations'] == []
        assert anchor_dict['conservation_score'] == 1.0

    def test_noether_violation_degraded_score(self, psi_anchor) -> None:
        """Noether violation → degraded conservation score."""
        msg: str = "Noether-E: energy increased by 5.0J exceeds budget"
        anchor_dict: dict = psi_anchor.inject_conservation_anchor(False, msg)
        assert anchor_dict['ok'] is False
        assert 'Noether-E' in anchor_dict['violations']
        # 1 violation → score = 1.0 - 0.3 = 0.7
        assert abs(anchor_dict['conservation_score'] - 0.7) < 1e-8

    def test_multiple_violations(self, psi_anchor) -> None:
        """Multiple violations extract all codes and heavily degrade score."""
        msg: str = "Noether-E: energy Noether-F: torque Noether-C: collision"
        anchor_dict: dict = psi_anchor.inject_conservation_anchor(False, msg)
        assert 'Noether-E' in anchor_dict['violations']
        assert 'Noether-F' in anchor_dict['violations']
        assert 'Noether-C' in anchor_dict['violations']
        # 3 violations → score = max(0.1, 1.0 - 0.3*3) = max(0.1, 0.1) = 0.1
        assert abs(anchor_dict['conservation_score'] - 0.1) < 1e-8


# ── Epiplexity ───────────────────────────────────────────────────


class TestEpiplexity:
    """Test epiplexity score computation."""

    def test_epiplexity_formula(self, simple_goal) -> None:
        """Epiplexity = len(invariants) * (1/delta_K) * log(max_energy)."""
        expected: float = len(simple_goal.invariants) * (1.0 / simple_goal.delta_K) * math.log(simple_goal.max_energy_inject)
        computed: float = PsiAnchor(simple_goal).compute_epiplexity(simple_goal)
        assert abs(computed - expected) < 1e-6

    def test_epiplexity_delta_K_safety(self) -> None:
        """Epiplexity handles delta_K ≈ 0 safely."""
        tiny_goal: GoalEML = GoalEML(
            name='tiny', invariants=['x'],
            delta_K=1e-8, max_energy_inject=500.0)
        epi: float = PsiAnchor(tiny_goal).compute_epiplexity(tiny_goal)
        # Should not be inf or overflow
        assert epi > 0 and epi < 1e8

    def test_epiplexity_energy_safety(self) -> None:
        """Epiplexity handles max_energy ≈ 0 safely."""
        zero_energy_goal: GoalEML = GoalEML(
            name='zero_energy', invariants=['x'],
            delta_K=0.05, max_energy_inject=1.0)  # log(1) = 0
        epi: float = PsiAnchor(zero_energy_goal).compute_epiplexity(zero_energy_goal)
        assert abs(epi) < 1e-6


# ── Evolution Triggering ──────────────────────────────────────────


class TestEvolutionTriggering:
    """Test self-evolution timing (When dimension)."""

    def test_no_trigger_insufficient_plateau(self, psi_anchor) -> None:
        """Evolution not triggered with insufficient plateau steps."""
        psi_anchor.plateau_steps = 2
        psi_anchor.plateau_evolution_min = 5
        psi_anchor.epiplexity_score = 5.0
        psi_anchor._conservation_score = 1.0
        # Force trend = plateau
        psi_anchor.eta_history = [0.5, 0.5001, 0.5002, 0.5003]
        assert not psi_anchor.should_trigger_evolution()

    def test_trigger_when_conditions_met(self, psi_anchor) -> None:
        """Evolution triggered when plateau + high epiplexity + conservation ok."""
        psi_anchor.plateau_evolution_min = 5
        psi_anchor.epiplexity_score = 5.0
        psi_anchor._conservation_score = 1.0
        # Simulate plateau for ≥ 5 steps
        psi_anchor.eta_history = [0.5] * 10
        psi_anchor.plateau_steps = 7
        assert psi_anchor.should_trigger_evolution()

    def test_no_trigger_low_epiplexity(self, psi_anchor) -> None:
        """Evolution not triggered when epiplexity is below threshold."""
        psi_anchor.plateau_steps = 10
        psi_anchor.epiplexity_score = 0.5
        psi_anchor.epiplexity_thresh = 2.0
        psi_anchor._conservation_score = 1.0
        psi_anchor.eta_history = [0.5] * 10
        assert not psi_anchor.should_trigger_evolution()

    def test_no_trigger_low_conservation(self, psi_anchor) -> None:
        """Evolution not triggered when conservation score is low."""
        psi_anchor.plateau_steps = 10
        psi_anchor.epiplexity_score = 5.0
        psi_anchor._conservation_score = 0.3  # below 0.5
        psi_anchor.eta_history = [0.5] * 10
        assert not psi_anchor.should_trigger_evolution()


# ── PsiAnchorState ───────────────────────────────────────────────


class TestPsiAnchorState:
    """Test PsiAnchorState snapshot retrieval."""

    def test_get_state_returns_dataclass(self, psi_anchor) -> None:
        """get_state returns a PsiAnchorState dataclass."""
        state: PsiAnchorState = psi_anchor.get_state()
        assert isinstance(state, PsiAnchorState)
        assert hasattr(state, 'eta_trend')
        assert hasattr(state, 'evo_policy')
        assert hasattr(state, 'adjusted_delta_K')
        assert hasattr(state, 'epiplexity_score')
        assert hasattr(state, 'conservation_score')
        assert hasattr(state, 'evolution_triggered')
        assert hasattr(state, 'plateau_steps')


# ── Macro Evolution ───────────────────────────────────────────────


class TestMacroEvolution:
    """Test MotorPrimitives macro evolution."""

    def test_freeze_policy_no_changes(self, psi_anchor) -> None:
        """Freeze policy returns macros unchanged."""
        macros: list = [(lambda x: None, 0.70), (lambda x: None, 0.50)]
        result: list = psi_anchor.apply_evolution_to_macros(macros, 'freeze')
        assert result[0][1] == 0.70
        assert result[1][1] == 0.50

    def test_light_policy_promote_demote(self, psi_anchor) -> None:
        """Light policy promotes best (+0.05) and demotes worst (-0.05)."""
        fn_a = lambda x: None  # noqa: E731
        fn_b = lambda x: None  # noqa: E731
        macros: list = [(fn_a, 0.70), (fn_b, 0.40)]
        result: list = psi_anchor.apply_evolution_to_macros(macros, 'light')
        # fn_a is best (0.70), promoted to 0.75
        # fn_b is worst (0.40), demoted to 0.35
        best_idx: int = 0  # 0.70 > 0.40
        worst_idx: int = 1
        assert abs(result[best_idx][1] - 0.75) < 1e-8
        assert abs(result[worst_idx][1] - 0.35) < 1e-8

    def test_light_policy_clamp_bounds(self, psi_anchor) -> None:
        """Light evolution IC-Values are clamped to [0.1, 1.0]."""
        fn_a = lambda x: None  # noqa: E731
        fn_b = lambda x: None  # noqa: E731
        macros: list = [(fn_a, 0.99), (fn_b, 0.11)]
        result: list = psi_anchor.apply_evolution_to_macros(macros, 'light')
        # Best promoted: 0.99 + 0.05 = 1.04 → clamped to 1.0
        assert abs(result[0][1] - 1.0) < 1e-8
        # Worst demoted: 0.11 - 0.05 = 0.06 → clamped to 0.1
        assert abs(result[1][1] - 0.1) < 1e-8

    def test_empty_macros_no_crash(self, psi_anchor) -> None:
        """Empty macro list doesn't crash."""
        result: list = psi_anchor.apply_evolution_to_macros([], 'light')
        assert result == []
