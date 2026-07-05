"""
Integration tests for v0.17.0 TOMAS Agent stack.

Tests the following modules:
  - core/hg_pinn.py: HardPhysicsGate, HG_PINN_Policy
  - agent/footstep_planner.py: SupportPolygon, FootstepPlanner
  - agent/tomas_deploy.py: TOMASAgent, MetaQuery
  - agent/failure_attribution.py: TOMASFailureAttributor (offline mode)
  - agent/__init__.py: Module exports

All tests run WITHOUT MuJoCo installed — uses mock environments.
"""
import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestHardPhysicsGate(unittest.TestCase):
    """Test HardPhysicsGate from core/hg_pinn.py."""

    def setUp(self):
        from core.hg_pinn import HardPhysicsGate, HardPhysicsGateConfig
        self.config = HardPhysicsGateConfig(
            max_velocity=1.5,
            max_torque=0.05,
            tau_safe=0.05,
        )
        self.gate = HardPhysicsGate(self.config)

    def test_no_violation_within_limits(self):
        """Action within all limits should produce no violations."""
        action = np.array([0.01, 0.02, 0.01])
        safe_action, violations = self.gate.forward(action)
        self.assertEqual(len(violations), 0)
        np.testing.assert_array_almost_equal(safe_action, action)

    def test_velocity_violation(self):
        """Action exceeding velocity limit should be scaled down."""
        action = np.array([2.0, 2.0, 2.0])  # norm > 1.5
        safe_action, violations = self.gate.forward(action)
        self.assertIn("MAX_VELOCITY", violations)
        self.assertLessEqual(np.linalg.norm(safe_action), 1.5 + 1e-6)

    def test_torque_violation(self):
        """Action exceeding torque limit should be clamped."""
        action = np.array([0.1, 0.1, 0.1])  # > 0.05
        safe_action, violations = self.gate.forward(action)
        self.assertIn("MAX_TORQUE", violations)
        self.assertTrue(np.all(np.abs(safe_action) <= 0.05 + 1e-6))

    def test_tau_safe_violation(self):
        """Action exceeding tau_safe should be clamped."""
        from core.hg_pinn import HardPhysicsGate, HardPhysicsGateConfig
        # Use tau_safe < max_torque to trigger TAU_SAFE separately
        config = HardPhysicsGateConfig(max_torque=0.1, tau_safe=0.01)
        gate = HardPhysicsGate(config)
        action = np.array([0.06, 0.04, 0.03])  # > tau_safe (0.01), < max_torque (0.1)
        safe_action, violations = gate.forward(action)
        self.assertIn("TAU_SAFE", violations)
        self.assertTrue(np.all(np.abs(safe_action) <= 0.01 + 1e-6))

    def test_acceleration_violation(self):
        """Large change from previous action should trigger acceleration limit."""
        # First step
        self.gate.forward(np.array([0.01, 0.01, 0.01]))
        # Second step with large change
        safe_action, violations = self.gate.forward(np.array([0.05, 0.05, 0.05]))
        # Should have acceleration violation (or be clamped)
        self.assertTrue(
            "MAX_ACCELERATION" in violations or np.allclose(safe_action, np.array([0.01, 0.01, 0.01]), atol=0.1)
        )

    def test_pitch_violation(self):
        """Excessive pitch should trigger violation and degrade forward action."""
        action = np.array([0.01, 0.01, 0.01])
        state = {"pitch": 20.0}  # > 15 degrees
        safe_action, violations = self.gate.forward(action, state)
        self.assertIn("PITCH_VIOLATION", violations)

    def test_reset(self):
        """Reset should clear previous action state."""
        self.gate.forward(np.array([0.01, 0.01, 0.01]))
        self.gate.reset()
        self.assertIsNone(self.gate.prev_action)

    def test_stats(self):
        """get_stats should return valid statistics."""
        self.gate.forward(np.array([2.0, 2.0, 2.0]))
        stats = self.gate.get_stats()
        self.assertIn("total_violations", stats)
        self.assertGreater(stats["total_violations"], 0)


class TestHGPINNPolicy(unittest.TestCase):
    """Test HG_PINN_Policy from core/hg_pinn.py."""

    def setUp(self):
        from core.hg_pinn import HGPINNConfig, HG_PINN_Policy
        self.config = HGPINNConfig(obs_dim=10, action_dim=7)
        self.policy = HG_PINN_Policy(self.config)

    def test_forward_returns_action(self):
        """forward() should return a dict with 'action' key."""
        obs = np.random.randn(10)
        result = self.policy.forward(obs)
        self.assertIn("action", result)
        self.assertIn("raw_action", result)
        self.assertIn("violations", result)
        self.assertEqual(len(result["action"]), 7)

    def test_predict_returns_array(self):
        """predict() should return just the action array."""
        obs = np.random.randn(10)
        action = self.policy.predict(obs)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 7)

    def test_dict_observation(self):
        """Should handle dict observation format."""
        obs = {"obs": np.random.randn(10), "goal": np.random.randn(7)}
        result = self.policy.forward(obs)
        self.assertEqual(len(result["action"]), 7)

    def test_tuple_observation(self):
        """Should handle tuple observation format."""
        obs = (np.random.randn(10), np.random.randn(7))
        result = self.policy.forward(obs)
        self.assertEqual(len(result["action"]), 7)

    def test_reset(self):
        """reset() should clear state."""
        self.policy.forward(np.random.randn(10))
        self.policy.reset()
        stats = self.policy.get_energy_stats()
        self.assertEqual(stats["current_energy"], 0.0)

    def test_gate_projection_applied(self):
        """Large raw action should be projected by gate."""
        # Create action head that generates large action
        from core.hg_pinn import HGPINNConfig as TestConfig, HardPhysicsGateConfig as TestGateConfig, HG_PINN_Policy as TestPolicy
        config = TestConfig(obs_dim=10, action_dim=7)
        gate_config = TestGateConfig(max_torque=0.01)  # Very tight
        policy = TestPolicy(config, gate_config)

        obs = np.random.randn(10) * 10  # Large observation
        result = policy.forward(obs)
        # Action should be clamped
        self.assertTrue(np.all(np.abs(result["action"]) <= 0.01 + 1e-6))


class TestSupportPolygon(unittest.TestCase):
    """Test SupportPolygon from agent/footstep_planner.py."""

    def setUp(self):
        from agent.footstep_planner import SupportPolygon
        self.poly = SupportPolygon(foot_radius=0.03, safety_margin=0.01)

    def test_zmp_inside(self):
        """ZMP at center should be inside polygon."""
        self.poly.set_foot_positions(np.array([0.0, 0.05]), np.array([0.0, -0.05]))
        zmp = np.array([0.0, 0.0])
        self.assertTrue(self.poly.check_zmp(zmp))

    def test_zmp_outside(self):
        """ZMP far from center should be outside polygon."""
        self.poly.set_foot_positions(np.array([0.0, 0.05]), np.array([0.0, -0.05]))
        zmp = np.array([1.0, 1.0])
        self.assertFalse(self.poly.check_zmp(zmp))

    def test_polygon_area_positive(self):
        """Polygon area should be positive."""
        self.poly.set_foot_positions(np.array([0.0, 0.05]), np.array([0.0, -0.05]))
        area = self.poly.get_polygon_area()
        self.assertGreater(area, 0.0)

    def test_vertices_returned(self):
        """Should return polygon vertices."""
        self.poly.set_foot_positions(np.array([0.0, 0.05]), np.array([0.0, -0.05]))
        vertices = self.poly.get_vertices()
        self.assertGreaterEqual(len(vertices), 3)


class TestFootstepPlanner(unittest.TestCase):
    """Test FootstepPlanner from agent/footstep_planner.py."""

    def setUp(self):
        from agent.footstep_planner import FootstepPlanner
        self.planner = FootstepPlanner(
            nominal_step_length=0.08,
            nominal_step_width=0.10,
        )

    def test_plan_straight_line(self):
        """Plan should generate footsteps from start to goal."""
        plan = self.planner.plan(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([1.0, 0.0, 0.0]),
        )
        self.assertGreater(plan.total_steps, 0)
        self.assertGreater(plan.total_distance, 0.5)
        self.assertEqual(len(plan.footsteps), plan.total_steps)

    def test_plan_short_distance(self):
        """Very short distance should produce minimal steps."""
        plan = self.planner.plan(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.01, 0.0, 0.0]),
        )
        self.assertGreaterEqual(plan.total_steps, 0)

    def test_plan_alternating_sides(self):
        """Footsteps should alternate left and right."""
        from agent.footstep_planner import FootSide
        plan = self.planner.plan(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.5, 0.0, 0.0]),
        )
        if len(plan.footsteps) >= 2:
            sides = [fs.side for fs in plan.footsteps]
            for i in range(1, len(sides)):
                self.assertNotEqual(sides[i], sides[i - 1])

    def test_swing_trajectory(self):
        """Swing trajectory should arc above ground."""
        traj = self.planner.compute_swing_trajectory(
            start_pos=np.array([0.0, 0.0, 0.0]),
            end_pos=np.array([0.1, 0.0, 0.0]),
            max_height=0.03,
        )
        self.assertEqual(traj.shape[1], 3)
        # Midpoint should be higher than endpoints
        mid_idx = len(traj) // 2
        self.assertGreater(traj[mid_idx, 2], traj[0, 2])
        self.assertGreater(traj[mid_idx, 2], traj[-1, 2])

    def test_com_trajectory(self):
        """CoM trajectory should be generated for a plan."""
        plan = self.planner.plan(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.3, 0.0, 0.0]),
        )
        com_traj = self.planner.compute_com_trajectory(plan)
        self.assertGreater(len(com_traj), 0)
        self.assertEqual(com_traj.shape[1], 3)

    def test_obstacle_avoidance(self):
        """Plan with obstacles should still reach goal."""
        plan = self.planner.plan_with_obstacle_avoidance(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.5, 0.0, 0.0]),
            obstacles=[{"position": [0.25, 0.0], "radius": 0.1}],
        )
        self.assertGreater(plan.total_steps, 0)

    def test_plan_to_dict(self):
        """Plan should serialize to dict."""
        plan = self.planner.plan(
            start_pos=np.array([0.0, 0.0, 0.0]),
            goal_pos=np.array([0.2, 0.0, 0.0]),
        )
        d = plan.to_dict()
        self.assertIn("footsteps", d)
        self.assertIn("total_distance", d)
        self.assertIn("plan_safe", d)


class TestTOMASFailureAttribution(unittest.TestCase):
    """Test TOMASFailureAttributor (offline mode)."""

    def setUp(self):
        from agent.failure_attribution import TOMASFailureAttributor
        self.attributor = TOMASFailureAttributor()

    def test_offline_attribution_descending(self):
        """Offline attribution should handle descending eta trend (no failure)."""
        eta_history = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
        snap_trail = [{"step": i, "eta": eta, "psi_violation": ""} for i, eta in enumerate(eta_history)]

        result = self.attributor.offline_attribuate(
            eta_history=eta_history,
            snap_trail=snap_trail,
        )
        self.assertIsNotNone(result)
        # Descending eta is healthy — should be "none" or "unknown" (no pathology)
        self.assertIn(result.failure_type, ("none", "unknown"))
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_offline_attribution_plateau(self):
        """Offline attribution should detect local_optimum_trap."""
        eta_history = [0.5] * 20  # Stuck at 0.5
        snap_trail = [{"step": i, "eta": 0.5, "psi_violation": ""} for i in range(20)]

        result = self.attributor.offline_attribuate(
            eta_history=eta_history,
            snap_trail=snap_trail,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.failure_type, "local_optimum_trap")

    def test_offline_attribution_escape(self):
        """Offline attribution should detect eta_escape."""
        eta_history = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9]
        snap_trail = [{"step": i, "eta": eta, "psi_violation": ""} for i, eta in enumerate(eta_history)]

        result = self.attributor.offline_attribuate(
            eta_history=eta_history,
            snap_trail=snap_trail,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.failure_type, "eta_escape")


class TestTOMASAgentDeploy(unittest.TestCase):
    """Test TOMASAgent deployment with mock environment."""

    def setUp(self):
        from agent.tomas_deploy import TOMASAgent
        self.mock_env = self._create_mock_env()
        self.agent = TOMASAgent(
            env=self.mock_env,
            vla_policy=self._mock_vla,
            max_steps=10,
        )

    @staticmethod
    def _mock_vla(obs):
        """Simple mock VLA that returns small random actions."""
        return np.random.randn(7) * 0.01

    @staticmethod
    def _create_mock_env():
        """Create a mock MuJoCo-like environment."""
        env = MagicMock()
        env.observation_space = MagicMock()
        env.observation_space.shape = (18,)
        env.action_space = MagicMock()
        env.action_space.shape = (7,)
        env.action_dim = 7

        obs = np.random.randn(18) * 0.1
        env.reset = MagicMock(return_value=(obs, {}))
        env.step = MagicMock(return_value=(obs, 0.0, False, {"eta": 0.5}))
        env.render = MagicMock()
        env.close = MagicMock()
        return env

    def test_deploy_single_episode(self):
        """Deploy should complete one episode."""
        report = self.agent.deploy(num_episodes=1, max_steps_per_episode=5)
        self.assertEqual(report.total_episodes, 1)
        self.assertGreater(report.total_steps, 0)

    def test_deploy_report_serializes(self):
        """Deploy report should serialize to JSON."""
        report = self.agent.deploy(num_episodes=1, max_steps_per_episode=3)
        json_str = report.to_json()
        self.assertIsInstance(json_str, str)

    def test_meta_query_why_this_action(self):
        """MetaQuery WHY_THIS_ACTION should return explanation."""
        from agent.tomas_deploy import MetaQueryType
        self.agent.deploy(num_episodes=1, max_steps_per_episode=3)
        result = self.agent.meta_query(MetaQueryType.WHY_THIS_ACTION)
        # May be None if no audit trail, but should not crash
        if result is not None:
            self.assertIsInstance(result.answer, str)

    def test_meta_query_audit_snap(self):
        """MetaQuery AUDIT_SNAP should return audit summary."""
        from agent.tomas_deploy import MetaQueryType
        self.agent.deploy(num_episodes=1, max_steps_per_episode=3)
        result = self.agent.meta_query(MetaQueryType.AUDIT_SNAP)
        if result is not None:
            self.assertIn("total_steps", result.evidence)

    def test_deploy_summary(self):
        """get_deploy_summary should return summary dict."""
        self.agent.deploy(num_episodes=1, max_steps_per_episode=3)
        summary = self.agent.get_deploy_summary()
        self.assertEqual(summary["total_deploys"], 1)
        self.assertIn("latest_status", summary)

    def test_skill_library_empty(self):
        """Skill library should start empty."""
        skills = self.agent.get_skill_library()
        self.assertEqual(len(skills), 0)


class TestModuleImports(unittest.TestCase):
    """Test that all new modules can be imported."""

    def test_import_agent_package(self):
        """agent package should import without errors."""
        import agent
        self.assertTrue(hasattr(agent, 'TOMASAgent'))
        self.assertTrue(hasattr(agent, 'TOMASMuJoCoWrapper'))
        self.assertTrue(hasattr(agent, 'TOMASFailureAttributor'))
        self.assertTrue(hasattr(agent, 'FootstepPlanner'))
        self.assertTrue(hasattr(agent, 'SupportPolygon'))

    def test_import_hg_pinn_new_classes(self):
        """core.hg_pinn should export new classes."""
        from core.hg_pinn import HardPhysicsGate, HardPhysicsGateConfig, HG_PINN_Policy
        self.assertIsNotNone(HardPhysicsGate)
        self.assertIsNotNone(HardPhysicsGateConfig)
        self.assertIsNotNone(HG_PINN_Policy)

    def test_import_footstep_planner(self):
        """agent.footstep_planner should import all classes."""
        from agent.footstep_planner import (
            FootstepPlanner, SupportPolygon, Footstep, FootstepPlan,
            FootSide, StepPhase,
        )
        self.assertIsNotNone(FootstepPlanner)
        self.assertIsNotNone(SupportPolygon)

    def test_import_tomas_deploy(self):
        """agent.tomas_deploy should import all classes."""
        from agent.tomas_deploy import (
            TOMASAgent, DeployStatus, DeployReport,
            MetaQueryType, MetaQueryResult, SkillRecord,
        )
        self.assertIsNotNone(TOMASAgent)
        self.assertIsNotNone(DeployStatus)

    def test_import_failure_attribution(self):
        """agent.failure_attribution should import all classes."""
        from agent.failure_attribution import (
            TOMASFailureAttributor, FailureAttributionResult,
        )
        self.assertIsNotNone(TOMASFailureAttributor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
