"""
Unit tests for agent/mujoco_ido_agent.py: MotorPrimitives, IDOMuJoCoAgent.

Tests are designed to run WITHOUT dm_control / MuJoCo installed.
All physics objects are mocked with simple attribute containers.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ── Ensure project root is importable ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.goal_eml_mj import GoalEML
from agent.mujoco_ido_agent import IDOMuJoCoAgent, MotorPrimitives


# ──────────────────────────────────────────────────────
# Helper: Mock physics objects for agent tests
# ──────────────────────────────────────────────────────
class MockPhysicsModel:
    """Simulates MuJoCo model attributes."""
    def __init__(self, njnt=6, nu=6, nq=7, nv=7):
        self.njnt = njnt
        self.nu = nu
        self.nq = nq
        self.nv = nv


class MockPhysicsData:
    """Simulates MuJoCo data attributes."""
    def __init__(self, ctrl=None, qpos=None, qvel=None, energy=None,
                 actuator_force=None, contact=None):
        if ctrl is not None:
            self.ctrl = ctrl
        else:
            self.ctrl = np.zeros(6)
        if qpos is not None:
            self.qpos = qpos
        else:
            self.qpos = np.zeros(7)
        if qvel is not None:
            self.qvel = qvel
        else:
            self.qvel = np.zeros(7)
        if energy is not None:
            self.energy = energy
        else:
            self.energy = [0.0, 0.0]
        if actuator_force is not None:
            self.actuator_force = actuator_force
        else:
            self.actuator_force = np.zeros(6)
        self.contact = contact or []


class MockPhysics:
    """Minimal mock dm_control Physics for MotorPrimitives."""
    def __init__(self, njnt=6, nu=6, ctrl=None, energy=None):
        self.model = MockPhysicsModel(njnt=njnt, nu=nu)
        self.data = MockPhysicsData(ctrl=ctrl, energy=energy)


class MockNamedData:
    """Simulates physics.named.data with xpos/cvel access."""
    def __init__(self, xpos_map=None):
        self._xpos_map = xpos_map or {}
        self.xpos = MockNamedAccess(self._xpos_map)
        self.cvel = MockNamedAccess(self._xpos_map)


class MockNamedAccess:
    """Simulates named-data subscript like ['right_hand', :]."""
    def __init__(self, data_map):
        self._data_map = data_map

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, _ = key
            return self._data_map.get(name, np.zeros(3))
        return self._data_map.get(key, np.zeros(3))


class MockPhysicsWithNamed(MockPhysics):
    """Physics mock with .named.data support for IDOMuJoCoAgent._extract_eml_obs."""
    def __init__(self, njnt=6, nu=6, ctrl=None, energy=None,
                 xpos_map=None, qpos=None, qvel=None,
                 actuator_force=None):
        super().__init__(njnt=njnt, nu=nu, ctrl=ctrl, energy=energy)
        self.named = MagicMock()
        self.named.data = MockNamedData(xpos_map)
        if qpos is not None:
            self.data.qpos = qpos
        if qvel is not None:
            self.data.qvel = qvel
        if actuator_force is not None:
            self.data.actuator_force = actuator_force


class MockTimeStep:
    """Simulates dm_control TimeStep."""
    def __init__(self, physics):
        self.physics = physics
        self._reward = 0.0

    def reward(self):
        return self._reward

    def last(self):
        return False


class MockEnv:
    """Simulates dm_control Environment."""
    def __init__(self, physics):
        self.physics = physics
        self._step_count = 0

    def reset(self):
        return MockTimeStep(self.physics)

    def step(self, action):
        self._step_count += 1
        return MockTimeStep(self.physics)


# ──────────────────────────────────────────────────────
# 1. MotorPrimitives tests
# ──────────────────────────────────────────────────────
class TestMotorPrimitives(unittest.TestCase):
    """Tests for MotorPrimitives.get_library() and structure validation."""

    def test_get_library_returns_list(self):
        """get_library() should return a list of (callable, float) tuples."""
        phys = MockPhysics()
        mp = MotorPrimitives(phys)
        library = mp.get_library()
        self.assertIsInstance(library, list)
        self.assertTrue(len(library) > 0)

    def test_get_library_tuple_structure(self):
        """Each entry should be a (callable, float) tuple."""
        phys = MockPhysics()
        mp = MotorPrimitives(phys)
        library = mp.get_library()
        for entry in library:
            self.assertEqual(len(entry), 2)
            self.assertTrue(callable(entry[0]))
            self.assertIsInstance(entry[1], float)

    def test_get_library_five_primitives(self):
        """get_library() should return exactly 5 primitives."""
        phys = MockPhysics()
        mp = MotorPrimitives(phys)
        library = mp.get_library()
        self.assertEqual(len(library), 5)

    def test_get_library_primitive_names(self):
        """Primitive functions should match expected names."""
        phys = MockPhysics()
        mp = MotorPrimitives(phys)
        library = mp.get_library()
        names = [fn.__name__ for fn, _ in library]
        self.assertIn('step_forward', names)
        self.assertIn('step_left', names)
        self.assertIn('step_right', names)
        self.assertIn('squat', names)
        self.assertIn('torque_explore', names)

    def test_get_library_ic_values(self):
        """IC-Value scores should match expected values."""
        phys = MockPhysics()
        mp = MotorPrimitives(phys)
        library = mp.get_library()
        scores = {fn.__name__: score for fn, score in library}
        self.assertAlmostEqual(scores['step_forward'], 0.70)
        self.assertAlmostEqual(scores['step_left'], 0.65)
        self.assertAlmostEqual(scores['step_right'], 0.65)
        self.assertAlmostEqual(scores['squat'], 0.50)
        self.assertAlmostEqual(scores['torque_explore'], 0.40)

    def test_motor_primitives_n_joints(self):
        """MotorPrimitives should store n_joints from physics.model."""
        phys = MockPhysics(njnt=10)
        mp = MotorPrimitives(phys)
        self.assertEqual(mp.n_joints, 10)

    def test_motor_primitives_zero_ctrl(self):
        """MotorPrimitives should create zero_ctrl of size physics.model.nu."""
        phys = MockPhysics(nu=8)
        mp = MotorPrimitives(phys)
        self.assertEqual(len(mp.zero_ctrl), 8)
        np.testing.assert_array_equal(mp.zero_ctrl, np.zeros(8))

    def test_pd_stabilize_returns_array(self):
        """pd_stabilize should return ctrl_delta array."""
        phys = MockPhysics(njnt=6, nu=6)
        mp = MotorPrimitives(phys)
        target = np.array([1.0, 0.0, 0.0])
        ee = np.array([0.0, 0.0, 0.0])
        delta = mp.pd_stabilize(phys, target, ee)
        self.assertIsInstance(delta, np.ndarray)
        self.assertEqual(delta.shape, phys.data.ctrl.shape)

    def test_pd_stabilize_direction(self):
        """pd_stabilize should produce ctrl pointing toward target."""
        phys = MockPhysics(njnt=6, nu=6, ctrl=np.zeros(6))
        mp = MotorPrimitives(phys)
        target = np.array([1.0, 0.0, 0.0])
        ee = np.array([0.0, 0.0, 0.0])
        delta = mp.pd_stabilize(phys, target, ee)
        # err = [1,0,0] → Kp*err = [30,0,0] → clipped[:2] = [0.5, 0]
        self.assertAlmostEqual(delta[0], 0.5)
        self.assertAlmostEqual(delta[1], 0.0)

    def test_pd_stabilize_clip(self):
        """pd_stabilize should clip ctrl_delta[:2] to [-0.5, 0.5]."""
        phys = MockPhysics(njnt=6, nu=6, ctrl=np.zeros(6))
        mp = MotorPrimitives(phys)
        # Very large error → should be clipped to 0.5
        target = np.array([100.0, 100.0, 100.0])
        ee = np.array([0.0, 0.0, 0.0])
        delta = mp.pd_stabilize(phys, target, ee)
        self.assertAlmostEqual(abs(delta[0]), 0.5)
        self.assertAlmostEqual(abs(delta[1]), 0.5)


# ──────────────────────────────────────────────────────
# 2. IDOMuJoCoAgent tests (with mock physics)
# ──────────────────────────────────────────────────────
class TestIDOMuJoCoAgentInit(unittest.TestCase):
    """Tests for IDOMuJoCoAgent initialization."""

    def test_agent_initialization(self):
        """IDOMuJoCoAgent should initialize with correct attributes."""
        phys = MockPhysicsWithNamed(njnt=6, nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test', target_pos=np.array([1.0, 0.0, 0.0]))

        agent = IDOMuJoCoAgent(env, goal, kappa_thresh=0.05)

        self.assertEqual(agent.env, env)
        self.assertEqual(agent.goal.name, 'test')
        self.assertAlmostEqual(agent.kappa_thresh, 0.05)
        self.assertEqual(agent.stall_count, 0)
        self.assertIsNone(agent.prev_data)
        self.assertIsNone(agent._last_eta)
        self.assertIsInstance(agent.mp, MotorPrimitives)
        self.assertIsInstance(agent.macros, list)
        self.assertEqual(len(agent.macros), 5)
        self.assertEqual(agent.oracle_buffer, [])

    def test_agent_custom_params(self):
        """IDOMuJoCoAgent should accept custom max_stall and enable_critique."""
        phys = MockPhysicsWithNamed(njnt=6, nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')

        agent = IDOMuJoCoAgent(env, goal, max_stall=50, enable_critique=False)
        self.assertEqual(agent.max_stall, 50)
        self.assertFalse(agent.enable_critique)


class TestOracleReplay(unittest.TestCase):
    """Tests for store_oracle_step / replay_oracle Oracle buffer."""

    def _make_agent(self):
        """Helper to create a minimal IDOMuJoCoAgent."""
        phys = MockPhysicsWithNamed(njnt=6, nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')
        agent = IDOMuJoCoAgent(env, goal)
        return agent

    def test_store_and_replay_single_step(self):
        """Store one action → replay should retrieve it."""
        agent = self._make_agent()
        action = np.array([0.5, -0.3, 0.1, 0.0, 0.2, 0.4])
        agent.store_oracle_step(action)
        retrieved = agent.replay_oracle(0)
        np.testing.assert_array_almost_equal(retrieved, action)

    def test_store_and_replay_multiple_steps(self):
        """Store multiple actions → replay should retrieve by index."""
        agent = self._make_agent()
        actions = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
        for a in actions:
            agent.store_oracle_step(a)
        for i, expected in enumerate(actions):
            retrieved = agent.replay_oracle(i)
            np.testing.assert_array_almost_equal(retrieved, expected)

    def test_replay_out_of_range(self):
        """replay_oracle with invalid index should return None."""
        agent = self._make_agent()
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        agent.store_oracle_step(action)
        self.assertIsNone(agent.replay_oracle(1))
        self.assertIsNone(agent.replay_oracle(99))

    def test_replay_empty_buffer(self):
        """replay_oracle on empty buffer should return None."""
        agent = self._make_agent()
        self.assertIsNone(agent.replay_oracle(0))

    def test_store_copies_action(self):
        """store_oracle_step should copy the action (not reference)."""
        agent = self._make_agent()
        action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        agent.store_oracle_step(action)
        # Modify original → buffer should be unaffected
        action[0] = 999.0
        retrieved = agent.replay_oracle(0)
        self.assertAlmostEqual(retrieved[0], 1.0)

    def test_oracle_buffer_length(self):
        """Oracle buffer should grow with each store_oracle_step call."""
        agent = self._make_agent()
        for i in range(5):
            agent.store_oracle_step(np.zeros(6))
        self.assertEqual(len(agent.oracle_buffer), 5)


class TestAgentKappaSnap(unittest.TestCase):
    """Tests for IDOMuJoCoAgent._compute_kappa_snap."""

    def _make_agent(self, goal=None):
        """Helper to create agent with a specific goal."""
        phys = MockPhysicsWithNamed(njnt=6, nu=6)
        env = MockEnv(phys)
        if goal is None:
            goal = GoalEML(name='test', target_pos=np.array([1.0, 0.0, 0.0]))
        return IDOMuJoCoAgent(env, goal)

    def test_compute_kappa_snap_zero(self):
        """When observation is at goal, kappa_snap should be near 0."""
        agent = self._make_agent(goal=GoalEML(
            name='test', target_pos=np.array([0.0, 0.0, 0.0])))
        z_i = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = agent._compute_kappa_snap(z_i)
        self.assertAlmostEqual(eta, 0.0, places=5)

    def test_compute_kappa_snap_positive(self):
        """When observation is far from goal, kappa_snap should be positive."""
        agent = self._make_agent(goal=GoalEML(
            name='test', target_pos=np.array([0.0, 0.0, 0.0])))
        z_i = {
            'ee_pos': np.array([5.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = agent._compute_kappa_snap(z_i)
        self.assertGreater(eta, 0.0)


class TestAgentNoetherCheck(unittest.TestCase):
    """Tests for IDOMuJoCoAgent._run_noether_check."""

    def _make_agent(self):
        """Helper to create agent for Noether check tests."""
        phys = MockPhysicsWithNamed(njnt=6, nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test', max_energy_inject=500.0)
        return IDOMuJoCoAgent(env, goal)

    def test_noether_check_no_prev_data(self):
        """With no prev_data, Noether check should pass (True, '')."""
        agent = self._make_agent()
        ok, msg = agent._run_noether_check()
        self.assertTrue(ok)
        self.assertEqual(msg, "")


class TestImportConsistency(unittest.TestCase):
    """Tests verifying agent module imports resolve correctly."""

    def test_agent_mujoco_ido_importable(self):
        """agent.mujoco_ido_agent should export IDOMuJoCoAgent."""
        from agent.mujoco_ido_agent import IDOMuJoCoAgent as A
        self.assertIs(A, IDOMuJoCoAgent)

    def test_motor_primitives_importable(self):
        """agent.mujoco_ido_agent should export MotorPrimitives."""
        from agent.mujoco_ido_agent import MotorPrimitives as M
        self.assertIs(M, MotorPrimitives)

    def test_agent_package_importable(self):
        """agent package (__init__.py) should be importable."""
        import agent
        self.assertTrue(hasattr(agent, '__file__'))

    def test_cross_module_core_imports_work(self):
        """Agent module should successfully use core module imports."""
        # Verify that the cross-module imports work
        from core.goal_eml_mj import GoalEML
        from core.kappa_snap_mj import gauss_ex_residual
        from core.noether_check_mj import noether_check_mj
        # These should all resolve correctly
        self.assertIsNotNone(GoalEML)
        self.assertIsNotNone(gauss_ex_residual)
        self.assertIsNotNone(noether_check_mj)


if __name__ == '__main__':
    unittest.main(verbosity=2)
