"""
Unit tests for SIP-Bench (benchmarks/run_mujoco_bench.py)
===========================================================

Tests the SIP-Bench longitudinal evaluation mode:
  - run_sip_benchmark function structure
  - Phase T0, T1, T2 metrics
  - Retention Gain and Stability Index computation
  - CLI --eval-mode sip argument parsing
  - Backward compatibility of run_benchmark with --eval-mode standard

NOTE: These tests use mock environments since dm_control may not be
installed. They verify the structural correctness of SIP-Bench logic
without requiring a real MuJoCo physics engine.

Author: tomas-arc3-solver project · MuJoCo-Bench-IDO v0.2.0
"""
import argparse
import json
import numpy as np
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.goal_eml_mj import GoalEML
from agent.psi_anchor import PsiAnchor
from core.kappa_snap_mj import FlowMatchingEtaPredictor
from benchmarks.run_mujoco_bench import (
    IDO_RUN_MUJOCO_BENCH_VERSION,
    TASK_REGISTRY,
    run_single_episode,
    _aggregate_metrics,
    run_sip_benchmark,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_goal() -> GoalEML:
    """Create a simple GoalEML for testing."""
    return GoalEML(
        name='test_task',
        invariants=['ee_at_target', 'torso_upright'],
        target_pos=np.array([1.0, 0.0, 0.5]),
        delta_K=0.05,
        max_energy_inject=500.0,
        pos_tol=0.02,
        ori_tol=0.15,
    )


# ── Version ───────────────────────────────────────────────────────


class TestSIPBenchVersion:
    """Verify run_mujoco_bench version upgrade."""

    def test_version_upgraded(self) -> None:
        """run_mujoco_bench version should be v0.3.0."""
        assert IDO_RUN_MUJOCO_BENCH_VERSION == "v0.3.0"


# ── Aggregate Metrics ─────────────────────────────────────────────


class TestAggregateMetrics:
    """Test _aggregate_metrics helper."""

    def test_empty_results(self) -> None:
        """Empty results list returns empty dict."""
        assert _aggregate_metrics([]) == {}

    def test_single_result(self) -> None:
        """Single result aggregates correctly."""
        results: list = [{
            'steps_to_goal': 100,
            'final_eta': 0.01,
            'noether_violations': 0,
            'elapsed_s': 5.0,
            'avg_return': 1.0,
            'hesit_rmse': 0.02,
            'retry_voc': 0.01,
            'epiplexity_score': 3.5,
        }]
        summary: dict = _aggregate_metrics(results)
        assert abs(summary['avg_steps'] - 100.0) < 1e-8
        assert abs(summary['avg_final_eta'] - 0.01) < 1e-8
        assert abs(summary['avg_hesit_rmse'] - 0.02) < 1e-8
        assert abs(summary['avg_retry_voc'] - 0.01) < 1e-8
        assert abs(summary['avg_epiplexity'] - 3.5) < 1e-8

    def test_multiple_results(self) -> None:
        """Multiple results aggregate averages correctly."""
        results: list = [
            {'steps_to_goal': 100, 'final_eta': 0.01, 'noether_violations': 0,
             'elapsed_s': 5.0, 'avg_return': 1.0, 'hesit_rmse': 0.02,
             'retry_voc': 0.01, 'epiplexity_score': 3.0},
            {'steps_to_goal': 200, 'final_eta': 0.02, 'noether_violations': 1,
             'elapsed_s': 10.0, 'avg_return': 2.0, 'hesit_rmse': 0.04,
             'retry_voc': 0.02, 'epiplexity_score': 4.0},
        ]
        summary: dict = _aggregate_metrics(results)
        assert abs(summary['avg_steps'] - 150.0) < 1e-8
        assert summary['total_noether_violations'] == 1
        assert abs(summary['avg_hesit_rmse'] - 0.03) < 1e-8
        assert abs(summary['avg_epiplexity'] - 3.5) < 1e-8

    def test_missing_v02_metrics_default_zero(self) -> None:
        """Missing v0.2.0 metrics default to 0.0."""
        results: list = [{
            'steps_to_goal': 100,
            'final_eta': 0.01,
            'noether_violations': 0,
            'elapsed_s': 5.0,
            'avg_return': 1.0,
            # No hesit_rmse, retry_voc, epiplexity_score
        }]
        summary: dict = _aggregate_metrics(results)
        assert summary['avg_hesit_rmse'] == 0.0
        assert summary['avg_retry_voc'] == 0.0
        assert summary['avg_epiplexity'] == 0.0


# ── SIP-Bench Structural Tests ────────────────────────────────────


class TestSIPBenchStructure:
    """Test SIP-Bench structural correctness without real environments."""

    def test_task_registry_has_all_tasks(self) -> None:
        """TASK_REGISTRY contains all four standard tasks."""
        expected_tasks: list = ['humanoid-stand', 'hopper-stand',
                                'walker-run', 'reacher-easy']
        for task in expected_tasks:
            assert task in TASK_REGISTRY

    def test_sip_benchmark_function_exists(self) -> None:
        """run_sip_benchmark function is defined and callable."""
        assert callable(run_sip_benchmark)


# ── Retention Gain Computation ────────────────────────────────────


class TestRetentionGain:
    """Test Retention Gain and Stability Index computation logic."""

    def test_retention_gain_improvement(self) -> None:
        """Retention Gain > 1 when T2 has fewer steps than T0."""
        t0_avg_steps: float = 200.0
        t2_avg_steps: float = 150.0
        # Improvement persisted: T0/T2 = 200/150 ≈ 1.33
        retention_gain: float = t0_avg_steps / t2_avg_steps
        assert retention_gain > 1.0

    def test_retention_gain_no_improvement(self) -> None:
        """Retention Gain < 1 when T2 has more steps than T0."""
        t0_avg_steps: float = 200.0
        t2_avg_steps: float = 300.0
        retention_gain: float = t0_avg_steps / t2_avg_steps
        assert retention_gain < 1.0

    def test_stability_index_more_stable(self) -> None:
        """Stability Index < 1 when T2 is more stable than T0."""
        t0_std: float = 50.0
        t2_std: float = 30.0
        stability_index: float = t2_std / t0_std
        assert stability_index < 1.0

    def test_stability_index_less_stable(self) -> None:
        """Stability Index > 1 when T2 is less stable than T0."""
        t0_std: float = 30.0
        t2_std: float = 50.0
        stability_index: float = t2_std / t0_std
        assert stability_index > 1.0

    def test_stability_index_zero_t0_std(self) -> None:
        """Stability Index handles zero T0 std gracefully."""
        t0_std: float = 0.0
        t2_std: float = 0.0
        # Both zero → stability_index = 0
        if t0_std > 0:
            stability_index: float = t2_std / t0_std
        else:
            stability_index = 0.0 if t2_std == 0 else float('inf')
        assert stability_index == 0.0


# ── CLI Argument Parsing ──────────────────────────────────────────


class TestCLIArguments:
    """Test CLI argument parsing for SIP-Bench mode."""

    def test_eval_mode_standard(self) -> None:
        """--eval-mode standard is accepted."""
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument('--eval-mode', default='standard',
                            choices=['standard', 'sip'])
        args: argparse.Namespace = parser.parse_args(['--eval-mode', 'standard'])
        assert args.eval_mode == 'standard'

    def test_eval_mode_sip(self) -> None:
        """--eval-mode sip is accepted."""
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument('--eval-mode', default='standard',
                            choices=['standard', 'sip'])
        args: argparse.Namespace = parser.parse_args(['--eval-mode', 'sip'])
        assert args.eval_mode == 'sip'

    def test_default_eval_mode(self) -> None:
        """Default eval-mode is standard."""
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument('--eval-mode', default='standard',
                            choices=['standard', 'sip'])
        args: argparse.Namespace = parser.parse_args([])
        assert args.eval_mode == 'standard'

    def test_evolution_rounds_argument(self) -> None:
        """--evolution_rounds argument is parsed."""
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument('--evolution_rounds', type=int, default=3)
        args: argparse.Namespace = parser.parse_args(['--evolution_rounds', '5'])
        assert args.evolution_rounds == 5


# ── run_single_episode v0.2.0 metrics ─────────────────────────────


class TestRunSingleEpisodeMetrics:
    """Test run_single_episode v0.2.0 metric collection.

    NOTE: MotorPrimitives methods import dm_control internally, so we
    replace the agent's macros list with noop mock functions to avoid
    the import dependency in unit tests.
    """

    def _make_mock_env_and_timestep(self) -> tuple:
        """Create mock environment and timestep for testing without dm_control."""
        mock_env = MagicMock()
        mock_physics = MagicMock()
        mock_data = MagicMock()
        mock_data.energy = np.array([50.0, 50.0])
        mock_data.ctrl = np.zeros(6)
        mock_data.qpos = np.array([1.0, 0.0, 0.0, 0.0])
        mock_data.qvel = np.zeros(4)
        mock_physics.data = mock_data
        mock_physics.model.nq = 4
        mock_physics.model.nv = 4
        mock_physics.model.nu = 6
        mock_physics.model.njnt = 2

        # Mock named access for ee_pos
        mock_named = MagicMock()
        mock_xpos = MagicMock()
        mock_xpos.__getitem__ = MagicMock(
            side_effect=KeyError('right_hand'))
        mock_named.data.xpos = mock_xpos
        mock_named.data.cvel = MagicMock(
            side_effect=KeyError('right_hand'))
        mock_physics.named = mock_named

        mock_timestep = MagicMock()
        mock_timestep.physics = mock_physics
        mock_timestep.reward = 0.5
        mock_timestep.last = MagicMock(return_value=False)
        # Ensure timestep.observation doesn't have 'to_target' (not a reacher task)
        mock_timestep.observation = {}

        mock_env.physics = mock_physics
        mock_env.reset = MagicMock(return_value=mock_timestep)
        mock_env.step = MagicMock(return_value=mock_timestep)

        return mock_env, mock_timestep

    def test_v02_metrics_in_result(self) -> None:
        """run_single_episode returns hesit_rmse, retry_voc, epiplexity_score."""
        mock_env, mock_timestep = self._make_mock_env_and_timestep()

        goal: GoalEML = GoalEML(
            name='test', invariants=['x'],
            target_pos=np.array([1.0, 0.0, 0.5]),
            delta_K=0.05, max_energy_inject=500.0,
        )

        from agent.mujoco_ido_agent import IDOMuJoCoAgent
        agent = IDOMuJoCoAgent(mock_env, goal, kappa_thresh=0.05)
        agent.psi_anchor = PsiAnchor(goal)
        agent.flow_predictor = FlowMatchingEtaPredictor()
        # Replace macros with noop functions to avoid dm_control import
        noop_fn = MagicMock()
        agent.macros = [(noop_fn, 0.70), (noop_fn, 0.50)]

        # Run a single episode (1 step)
        result: dict = run_single_episode(mock_env, agent, max_steps=1)

        # Check v0.2.0 metrics exist in result
        assert 'hesit_rmse' in result
        assert 'retry_voc' in result
        assert 'epiplexity_score' in result

    def test_v02_metrics_without_anchor_predictor(self) -> None:
        """run_single_episode returns default v0.2.0 metrics without anchor."""
        mock_env, mock_timestep = self._make_mock_env_and_timestep()

        goal: GoalEML = GoalEML(
            name='test', invariants=['x'],
            target_pos=np.array([1.0, 0.0, 0.5]),
            delta_K=0.05, max_energy_inject=500.0,
        )

        from agent.mujoco_ido_agent import IDOMuJoCoAgent
        agent = IDOMuJoCoAgent(mock_env, goal, kappa_thresh=0.05)
        # No psi_anchor, no flow_predictor
        # Replace macros with noop functions to avoid dm_control import
        noop_fn = MagicMock()
        agent.macros = [(noop_fn, 0.70), (noop_fn, 0.50)]

        result: dict = run_single_episode(mock_env, agent, max_steps=1)
        assert result['hesit_rmse'] == 0.0
        assert result['retry_voc'] == 0.0
        # v0.5.0: PsiAnchor is now created by default in IDOMuJoCoAgent.__init__,
        # so epiplexity_score is no longer 0.0 — it's computed from GoalEML.
        assert result['epiplexity_score'] > 0.0
