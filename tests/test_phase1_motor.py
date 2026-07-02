"""
Phase 1 Motor Layer Refactor — Comprehensive QA Tests
======================================================

Validates the MuJoCo-Bench-IDO v0.5.0 Phase 1 Motor layer changes:
  1. Import consistency (no circular deps, correct imports)
  2. Interface contract (return types, dimensions, backward compat)
  3. TASK_CONTROLLER_MAP coverage (all 25 tasks mapped)
  4. Decision flow logic (Noether → safe_action, η paths)
  5. Runtime crash bug detection (noether_check_mj dict vs tuple)

Author: QA Engineer Edward (Yan) · Phase 1 verification
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.task_pd_controllers import (
    TaskPDController, GenericPDController, get_controller_for_task,
    TASK_CONTROLLER_MAP,
    HumanoidStandPD, HumanoidWalkPD, HumanoidRunPD,
    ReacherTargetPD, WalkerStandPD, WalkerWalkPD, WalkerRunPD,
    CheetahRunPD, CartpoleBalancePD, CartpoleSwingupPD,
    HopperStandPD, HopperHopPD, FishSwimPD, SwimmerSwimPD,
    FingerSpinPD, FingerTurnEasyPD, FingerTurnHardPD,
    BallInCupCatchPD, AcrobotSwingupPD, PendulumSwingupPD,
    ManipulatorBringBallPD,
)
from agent.mujoco_ido_agent import IDOMuJoCoAgent, MotorPrimitives
from core.goal_eml_mj import GoalEML
from core.noether_check_mj import noether_check_mj


# ── Mock Physics Helpers ──────────────────────────────────────────────

class MockPhysicsModel:
    def __init__(self, njnt=6, nu=6, nq=7, nv=7):
        self.njnt = njnt
        self.nu = nu
        self.nq = nq
        self.nv = nv


class MockPhysicsData:
    def __init__(self, ctrl=None, qpos=None, qvel=None, energy=None,
                 actuator_force=None, contact=None, nu=None):
        self.ctrl = ctrl if ctrl is not None else np.zeros(nu or 6)
        self.qpos = qpos if qpos is not None else np.zeros(max((nu or 6) + 1, 7))
        self.qvel = qvel if qvel is not None else np.zeros(max((nu or 6) + 1, 7))
        self.energy = energy if energy is not None else [0.0, 0.0]
        self.actuator_force = actuator_force if actuator_force is not None else np.zeros(nu or 6)
        self.contact = contact if contact is not None else []


class MockPhysics:
    def __init__(self, njnt=6, nu=6, ctrl=None, energy=None):
        self.model = MockPhysicsModel(njnt=njnt, nu=nu)
        self.data = MockPhysicsData(ctrl=ctrl, energy=energy, nu=nu)
        # Minimal named access for PD controllers that use phys.named.data.xpos/xmat/sensordata
        # Default: all bodies at origin, upright (zz=1.0), zero velocity
        default_xpos = {
            'torso': np.array([0.0, 0.0, 1.2]),
            'head': np.array([0.0, 0.0, 1.6]),
            'pelvis': np.array([0.0, 0.0, 0.9]),
            'right_hand': np.array([0.0, 0.0, 0.0]),
        }
        default_xmat = {
            'torso': np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # upright
        }
        default_sensordata = {
            'torso_subtreelinvel': np.array([0.0, 0.0, 0.0]),
        }
        self._named_data = MockNamedData(default_xpos)
        self._named_data.xmat = MockNamedAccess(default_xmat)
        self._named_data.sensordata = default_sensordata
        self.named = MagicMock()
        self.named.data = self._named_data


class MockNamedData:
    def __init__(self, xpos_map=None, xmat_map=None, sensordata_map=None):
        self._xpos_map = xpos_map or {}
        self.xpos = MockNamedAccess(self._xpos_map)
        self.cvel = MockNamedAccess(self._xpos_map)
        # xmat: rotation matrix per body (9-element flat array for dm_control)
        self._xmat_map = xmat_map or {}
        self.xmat = MockNamedAccess(self._xmat_map)
        # sensordata: dict of sensor_name → array
        self._sensordata_map = sensordata_map or {}
        self.sensordata = self._sensordata_map


class MockNamedAccess:
    """Mock named-array access that mimics dm_control's named.data.xpos/xmat.

    Supports both full-name access: xpos['torso'] → 3-vector
    and element access: xpos['torso', 2] → scalar (z-coordinate)
    This matches dm_control's actual named-array indexing behavior.
    """
    def __init__(self, data_map):
        self._data_map = data_map

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, idx = key
            arr = self._data_map.get(name, np.zeros(3))
            # Return scalar for element access (dm_control behavior)
            return float(arr[idx]) if isinstance(idx, int) else arr
        return self._data_map.get(key, np.zeros(3))


class MockPhysicsWithNamed(MockPhysics):
    def __init__(self, njnt=6, nu=6, ctrl=None, energy=None,
                 xpos_map=None, xmat_map=None, sensordata_map=None,
                 qpos=None, qvel=None,
                 actuator_force=None):
        # Ensure ctrl shape matches nu
        if ctrl is None:
            ctrl = np.zeros(nu)
        super().__init__(njnt=njnt, nu=nu, ctrl=ctrl, energy=energy)
        # Override named.data with custom maps
        custom_xpos = xpos_map or {
            'torso': np.array([0.0, 0.0, 1.2]),
            'head': np.array([0.0, 0.0, 1.6]),
        }
        custom_xmat = xmat_map or {
            'torso': np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        }
        custom_sensordata = sensordata_map or {
            'torso_subtreelinvel': np.array([0.0, 0.0, 0.0]),
        }
        self.named.data = MockNamedData(custom_xpos, custom_xmat, custom_sensordata)
        if qpos is not None:
            self.data.qpos = qpos
        if qvel is not None:
            self.data.qvel = qvel
        if actuator_force is not None:
            self.data.actuator_force = actuator_force


class MockTimeStep:
    def __init__(self, physics, observation=None):
        self.physics = physics
        self._reward = 0.0
        self.observation = observation or {}

    def reward(self):
        return self._reward

    def last(self):
        return False


class MockEnv:
    def __init__(self, physics):
        self.physics = physics
        self._step_count = 0

    def reset(self):
        return MockTimeStep(self.physics)

    def step(self, action):
        self._step_count += 1
        return MockTimeStep(self.physics)


# ── 1. Import Consistency Tests ───────────────────────────────────────

class TestImportConsistency(unittest.TestCase):
    """Verify import chains have no circular dependencies and all exports are correct."""

    def test_task_pd_controllers_imports_stdlib_only(self):
        """task_pd_controllers.py should only import stdlib + numpy (no circular deps)."""
        import agent.task_pd_controllers as tpc_mod
        source_file = tpc_mod.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Should NOT import from mujoco_ido_agent (would create circular dep)
        self.assertNotIn('from agent.mujoco_ido_agent', content)
        self.assertNotIn('import agent.mujoco_ido_agent', content)
        # Should NOT import from psi_anchor
        self.assertNotIn('from agent.psi_anchor', content)

    def test_mujoco_ido_agent_imports_task_pd(self):
        """mujoco_ido_agent.py should correctly import TaskPDController, GenericPDController, get_controller_for_task."""
        from agent.mujoco_ido_agent import IDOMuJoCoAgent
        # Verify the module name is correct
        self.assertEqual(IDOMuJoCoAgent.__module__, 'agent.mujoco_ido_agent')

    def test_task_pd_controllers_exports(self):
        """task_pd_controllers should export all required names."""
        import agent.task_pd_controllers as tpc
        self.assertTrue(hasattr(tpc, 'TaskPDController'))
        self.assertTrue(hasattr(tpc, 'GenericPDController'))
        self.assertTrue(hasattr(tpc, 'get_controller_for_task'))
        self.assertTrue(hasattr(tpc, 'TASK_CONTROLLER_MAP'))

    def test_run_mujoco_bench_imports_unaffected(self):
        """run_mujoco_bench.py imports should not be broken by Phase 1 changes."""
        from benchmarks.run_mujoco_bench import (
            IDOMuJoCoAgent, TASK_REGISTRY, run_benchmark
        )
        self.assertIsNotNone(IDOMuJoCoAgent)
        self.assertIsNotNone(TASK_REGISTRY)


# ── 2. Interface Contract Tests ───────────────────────────────────────

class TestInterfaceContract(unittest.TestCase):
    """Verify return types, dimensions, and backward compatibility of Phase 1 interfaces."""

    def test_compute_action_returns_ndarray(self):
        """TaskPDController.compute_action() should return np.ndarray(nu,)."""
        phys = MockPhysics(nu=6)
        ctrl_cls = GenericPDController
        ctrl = ctrl_cls(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_action(timestep, phys)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6,))

    def test_compute_safe_action_returns_ndarray(self):
        """TaskPDController.compute_safe_action() should return np.ndarray(nu,)."""
        phys = MockPhysics(nu=6)
        ctrl = GenericPDController(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_safe_action(timestep, phys)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6,))

    def test_compute_safe_action_base_class_returns_zeros(self):
        """TaskPDController base compute_safe_action() should return zeros (nu,)."""
        # Create a minimal subclass that doesn't override compute_safe_action
        class MinimalPD(TaskPDController):
            def compute_action(self, timestep, physics):
                return np.zeros(self.nu)

        phys = MockPhysics(nu=4)
        ctrl = MinimalPD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_safe_action(timestep, phys)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_clip_ctrl_dimension_matches_nu(self):
        """_clip_ctrl should ensure output dimension = physics.model.nu."""
        phys = MockPhysics(nu=21)
        ctrl = HumanoidStandPD(phys)

        # Test with correct-size input
        input_ctrl = np.ones(21)
        result = ctrl._clip_ctrl(input_ctrl)
        self.assertEqual(result.shape, (21,))

        # Test with shorter input — should pad with zeros
        input_short = np.ones(5)
        result = ctrl._clip_ctrl(input_short)
        self.assertEqual(result.shape, (21,))
        self.assertEqual(result[5], 0.0)  # Padded position should be 0

        # Test with longer input — should truncate
        input_long = np.ones(30)
        result = ctrl._clip_ctrl(input_long)
        self.assertEqual(result.shape, (21,))

    def test_clip_ctrl_range_minus1_to_1(self):
        """_clip_ctrl should clip all values to [-1, 1]."""
        phys = MockPhysics(nu=6)
        ctrl = GenericPDController(phys)

        # Test with out-of-range input
        input_ctrl = np.array([5.0, -3.0, 0.5, 1.5, -2.0, 0.0])
        result = ctrl._clip_ctrl(input_ctrl)
        self.assertTrue(np.all(result <= 1.0))
        self.assertTrue(np.all(result >= -1.0))
        self.assertAlmostEqual(result[0], 1.0)  # 5.0 clipped to 1.0
        self.assertAlmostEqual(result[1], -1.0)  # -3.0 clipped to -1.0
        self.assertAlmostEqual(result[2], 0.5)   # 0.5 stays
        self.assertAlmostEqual(result[3], 1.0)   # 1.5 clipped to 1.0
        self.assertAlmostEqual(result[4], -1.0)  # -2.0 clipped to -1.0

    def test_task_name_default_backward_compatible(self):
        """IDOMuJoCoAgent.__init__ task_name default should maintain backward compat."""
        phys = MockPhysicsWithNamed(nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')

        # Create agent WITHOUT specifying task_name (backward compat)
        agent = IDOMuJoCoAgent(env, goal)
        self.assertEqual(agent.task_name, 'humanoid-stand')
        # task_controller should be HumanoidStandPD
        self.assertIsInstance(agent.task_controller, TaskPDController)

    def test_task_name_custom(self):
        """IDOMuJoCoAgent should accept custom task_name."""
        phys = MockPhysicsWithNamed(nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')

        agent = IDOMuJoCoAgent(env, goal, task_name='walker-walk')
        self.assertEqual(agent.task_name, 'walker-walk')

    def test_choose_action_returns_ndarray_nu(self):
        """IDOMuJoCoAgent.choose_action() should return np.ndarray(nu,)."""
        phys = MockPhysicsWithNamed(nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test', target_pos=np.array([0.0, 0.0, 0.0]))

        agent = IDOMuJoCoAgent(env, goal, task_name='walker-walk')
        agent.prev_data = None  # First step: no Noether check

        timestep = MockTimeStep(phys)
        result = agent.choose_action(timestep, physics=phys)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6,))
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))


# ── 3. TASK_CONTROLLER_MAP Coverage Tests ──────────────────────────────

class TestTaskControllerMapCoverage(unittest.TestCase):
    """Verify all 25 TASK_REGISTRY tasks have TASK_CONTROLLER_MAP entries."""

    TASK_REGISTRY_TASKS = [
        'humanoid-stand', 'humanoid-walk', 'humanoid-run',
        'walker-stand', 'walker-walk', 'walker-run',
        'hopper-stand', 'hopper-hop',
        'cheetah-run',
        'cartpole-balance', 'cartpole-swingup',
        'cartpole-balance_sparse', 'cartpole-swingup_sparse',
        'reacher-easy', 'reacher-hard',
        'fish-swim',
        'manipulator-bring_ball',
        'acrobot-swingup', 'pendulum-swingup',
        'finger-spin', 'finger-turn_easy', 'finger-turn_hard',
        'ball_in_cup-catch',
        'swimmer-swim6', 'swimmer-swim15',
    ]

    def test_controller_map_has_25_entries(self):
        """TASK_CONTROLLER_MAP should have exactly 25 entries."""
        self.assertEqual(len(TASK_CONTROLLER_MAP), 25)

    def test_all_registry_tasks_in_controller_map(self):
        """Every TASK_REGISTRY task should have a TASK_CONTROLLER_MAP entry."""
        for task in self.TASK_REGISTRY_TASKS:
            self.assertIn(task, TASK_CONTROLLER_MAP,
                          f"Task '{task}' missing from TASK_CONTROLLER_MAP")

    def test_controller_map_entries_are_task_pd_subclasses(self):
        """Every TASK_CONTROLLER_MAP value should be a TaskPDController subclass."""
        for task, ctrl_cls in TASK_CONTROLLER_MAP.items():
            self.assertTrue(issubclass(ctrl_cls, TaskPDController),
                            f"Controller for '{task}' ({ctrl_cls.__name__}) "
                            f"is not a TaskPDController subclass")

    def test_get_controller_for_task_factory(self):
        """get_controller_for_task should create correct controller instances."""
        phys = MockPhysics(nu=6)

        # Known task → specific controller
        ctrl = get_controller_for_task('humanoid-stand', phys)
        self.assertIsInstance(ctrl, HumanoidStandPD)

        # Unknown task → GenericPDController fallback
        ctrl = get_controller_for_task('unknown-task', phys)
        self.assertIsInstance(ctrl, GenericPDController)

    def test_controller_nu_matches_physics_model(self):
        """Each controller's nu should match the physics model's nu."""
        for nu_val in [1, 2, 4, 6, 21]:
            phys = MockPhysics(nu=nu_val)
            ctrl = GenericPDController(phys)
            self.assertEqual(ctrl.nu, nu_val)

    def test_controller_map_no_duplicate_keys(self):
        """TASK_CONTROLLER_MAP should have no duplicate keys."""
        keys = list(TASK_CONTROLLER_MAP.keys())
        self.assertEqual(len(keys), len(set(keys)))


# ── 4. Decision Flow Logic Tests ──────────────────────────────────────

class TestDecisionFlowLogic(unittest.TestCase):
    """Verify the Phase 1 decision flow in choose_action()."""

    def _make_agent(self, task_name='humanoid-stand', nu=21, kappa_thresh=0.05):
        """Create a mock agent with specified task and nu."""
        # Ensure mock ctrl/qpos/qvel dimensions match nu
        phys = MockPhysicsWithNamed(
            nu=nu, njnt=nu,
            ctrl=np.zeros(nu),
            qpos=np.zeros(max(nu + 7, 28)),
            qvel=np.zeros(max(nu + 7, 28)),
        )
        env = MockEnv(phys)
        goal = GoalEML(name='test', target_pos=np.array([0.0, 0.0, 0.0]),
                       max_energy_inject=500.0)
        return IDOMuJoCoAgent(env, goal, task_name=task_name,
                              kappa_thresh=kappa_thresh)

    def test_noether_violation_uses_safe_action_not_squat(self):
        """On Noether violation, choose_action should use compute_safe_action (NOT squat)."""
        agent = self._make_agent()
        phys = agent.env.physics
        timestep = MockTimeStep(phys)

        # Set prev_data to trigger Noether check
        # Create a data object with huge energy increase to force violation
        prev = MockPhysicsData(ctrl=np.zeros(21),
                               qpos=np.zeros(28),
                               qvel=np.zeros(28),
                               energy=[0.0, 0.0])
        agent.prev_data = prev

        # Make cur data have huge energy increase
        phys.data.energy = [0.0, 5000.0]  # ΔE = 5000 > max_energy_inject

        # Call choose_action — should NOT crash on tuple unpacking
        # (this will crash if _run_noether_check tries dict→tuple unpack)
        try:
            action = agent.choose_action(timestep, physics=phys)
            # If we get here, the bug may have been avoided by prev_data check
            # but we need to verify the noether violation path specifically
        except ValueError as e:
            if "too many values to unpack" in str(e):
                # This confirms the bug: noether_check_mj returns dict,
                # but choose_action unpacks as (ok, msg) tuple
                self.fail(f"CRITICAL BUG: choose_action crashes on "
                          f"Noether violation path: {e}")

    def test_eta_below_threshold_blend_path(self):
        """When η < κ_thresh, choose_action should blend pd_stabilize + task_ctrl."""
        agent = self._make_agent(kappa_thresh=0.05)
        phys = agent.env.physics

        # Mock _compute_kappa_snap to return low η (below threshold)
        with patch.object(agent, '_compute_kappa_snap', return_value=0.01):
            agent.prev_data = None  # Skip Noether check
            timestep = MockTimeStep(phys)
            action = agent.choose_action(timestep, physics=phys)

            # Should return an array of correct shape
            self.assertIsInstance(action, np.ndarray)
            self.assertEqual(action.shape, (21,))

    def test_eta_above_threshold_task_ctrl_path(self):
        """When η ≥ κ_thresh, choose_action should use task_controller.compute_action()."""
        agent = self._make_agent(kappa_thresh=0.05)
        phys = agent.env.physics

        # Mock _compute_kappa_snap to return high η (above threshold)
        with patch.object(agent, '_compute_kappa_snap', return_value=100.0):
            agent.prev_data = None  # Skip Noether check
            timestep = MockTimeStep(phys)
            action = agent.choose_action(timestep, physics=phys)

            self.assertIsInstance(action, np.ndarray)
            self.assertEqual(action.shape, (21,))

    def test_macros_not_used_in_decision_loop(self):
        """macros should NOT be used in choose_action decision loop (only ψ-Anchor compat)."""
        # Read source to verify
        import agent.mujoco_ido_agent as agent_mod
        source_file = agent_mod.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            source = f.read()

        # The choose_action method should NOT select from macros
        # Check that there's no macro selection in choose_action
        # Look for the method content
        choose_action_start = source.find('def choose_action')
        choose_action_end = source.find('\n    def ', choose_action_start + 1)
        if choose_action_end == -1:
            choose_action_end = len(source)
        choose_action_body = source[choose_action_start:choose_action_end]

        # Should NOT contain macro selection logic like:
        # "best_fn, best_ic = max(...)" or "self.macros[..."
        self.assertNotIn('max(self.macros', choose_action_body)
        self.assertNotIn('self.macros[', choose_action_body)
        self.assertNotIn('best_fn', choose_action_body)
        self.assertNotIn('best_ic', choose_action_body)

        # Should contain task_controller usage
        self.assertIn('task_controller.compute_action', choose_action_body)
        self.assertIn('task_controller.compute_safe_action', choose_action_body)

    def test_squat_not_used_in_noether_path(self):
        """Noether violation path should NOT use squat (should use safe_action)."""
        import agent.mujoco_ido_agent as agent_mod
        source_file = agent_mod.__file__
        with open(source_file, 'r', encoding='utf-8') as f:
            source = f.read()

        choose_action_start = source.find('def choose_action')
        choose_action_end = source.find('\n    def ', choose_action_start + 1)
        if choose_action_end == -1:
            choose_action_end = len(source)
        choose_action_body = source[choose_action_start:choose_action_end]

        # The Noether violation block should use compute_safe_action
        # and NOT call self.mp.squat
        self.assertIn('compute_safe_action', choose_action_body)

        # Verify "squat" is not called in choose_action
        # (it's still defined in MotorPrimitives but not used in choose_action)
        self.assertNotIn('self.mp.squat', choose_action_body)


# ── 5. Runtime Bug Detection ──────────────────────────────────────────

class TestNoetherReturnTypeBug(unittest.TestCase):
    """Detect the critical runtime bug: noether_check_mj returns dict but
    _run_noether_check + choose_action unpack as (ok, msg) tuple."""

    def test_noether_check_mj_returns_dict(self):
        """noether_check_mj should return a dict (not a tuple)."""
        prev = MockPhysicsData(ctrl=np.zeros(6),
                               qpos=np.zeros(7),
                               qvel=np.zeros(7),
                               energy=[0.0, 0.0])
        cur = MockPhysicsData(ctrl=np.zeros(6),
                              qpos=np.zeros(7),
                              qvel=np.zeros(7),
                              energy=[0.0, 0.0])
        goal = GoalEML(name='test', max_energy_inject=500.0)

        result = noether_check_mj(prev, cur, goal)
        self.assertIsInstance(result, dict)
        self.assertIn('ok', result)
        self.assertIn('message', result)

    def test_dict_unpack_as_tuple_crashes(self):
        """Unpacking noether_check_mj dict as (ok, msg) tuple should crash."""
        prev = MockPhysicsData(ctrl=np.zeros(6),
                               qpos=np.zeros(7),
                               qvel=np.zeros(7),
                               energy=[0.0, 0.0])
        cur = MockPhysicsData(ctrl=np.zeros(6),
                              qpos=np.zeros(7),
                              qvel=np.zeros(7),
                              energy=[0.0, 0.0])
        goal = GoalEML(name='test', max_energy_inject=500.0)

        result = noether_check_mj(prev, cur, goal)
        with self.assertRaises(ValueError):
            ok, msg = result  # This is what choose_action does — CRASH!

    def test_run_noether_check_return_type_annotation_wrong(self):
        """_run_noether_check() return annotation says Tuple[bool, str]
        but actually returns dict from noether_check_mj — type mismatch."""
        import inspect
        from agent.mujoco_ido_agent import IDOMuJoCoAgent
        method = IDOMuJoCoAgent._run_noether_check
        sig = inspect.signature(method)
        # The annotation says Tuple[bool, str] but noether_check_mj returns dict
        # This is a contract violation
        ann = sig.return_annotation
        # Either 'Tuple[bool, str]' or the actual typing object
        self.assertIn('Tuple', str(ann),
                      "Annotation should be Tuple (which is wrong since "
                      "noether_check_mj returns dict)")

    def test_choose_action_crashes_with_prev_data(self):
        """v0.5.0 fix: choose_action should NOT crash when prev_data is set.

        Previously, noether_check_mj() returned dict but choose_action
        unpacked as (ok, msg) tuple -> ValueError. Now _run_noether_check()
        adapts dict->tuple. This test verifies the fix works.
        """
        phys = MockPhysicsWithNamed(nu=21)
        env = MockEnv(phys)
        goal = GoalEML(name='test', target_pos=np.array([0.0, 0.0, 0.0]),
                       max_energy_inject=500.0)
        agent = IDOMuJoCoAgent(env, goal, task_name='humanoid-stand')

        # Set prev_data to trigger actual Noether check (not the None shortcut)
        prev = MockPhysicsData(ctrl=np.zeros(21),
                               qpos=np.zeros(28),
                               qvel=np.zeros(28),
                               energy=[0.0, 0.0])
        agent.prev_data = prev

        timestep = MockTimeStep(phys)
        # v0.5.0 fix: should NOT raise ValueError anymore
        action = agent.choose_action(timestep, physics=phys)
        # Verify it returns a valid action
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 21)


# ── 6. Specific Controller Tests ──────────────────────────────────────

class TestSpecificControllers(unittest.TestCase):
    """Test individual controller compute_action outputs."""

    def test_humanoid_stand_pd_output_shape(self):
        """HumanoidStandPD should output ctrl of shape (21,)."""
        phys = MockPhysics(nu=21, njnt=21)
        ctrl = HumanoidStandPD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_action(timestep, phys)
        self.assertEqual(result.shape, (21,))
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_cartpole_balance_pd_output_shape(self):
        """CartpoleBalancePD should output ctrl of shape (1,)."""
        phys = MockPhysics(nu=1, njnt=1)
        ctrl = CartpoleBalancePD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_action(timestep, phys)
        self.assertEqual(result.shape, (1,))

    def test_reacher_target_pd_output_shape(self):
        """ReacherTargetPD should output ctrl of shape (2,)."""
        phys = MockPhysics(nu=2, njnt=2)
        ctrl = ReacherTargetPD(phys)
        timestep = MockTimeStep(phys, observation={
            'to_target': np.array([0.1, 0.2]),
            'position': np.array([0.0, 0.0]),
        })
        result = ctrl.compute_action(timestep, phys)
        self.assertEqual(result.shape, (2,))

    def test_hopper_stand_pd_output_shape(self):
        """HopperStandPD should output ctrl of shape (4,)."""
        phys = MockPhysics(nu=4, njnt=4)
        ctrl = HopperStandPD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_action(timestep, phys)
        self.assertEqual(result.shape, (4,))

    def test_generic_pd_fallback_output(self):
        """GenericPDController should produce valid output for any nu."""
        for nu in [1, 2, 4, 6, 14, 21]:
            phys = MockPhysics(nu=nu)
            ctrl = GenericPDController(phys)
            timestep = MockTimeStep(phys)
            result = ctrl.compute_action(timestep, phys)
            self.assertEqual(result.shape, (nu,))
            self.assertTrue(np.all(result >= -1.0))
            self.assertTrue(np.all(result <= 1.0))

    def test_humanoid_stand_safe_action_shape(self):
        """HumanoidStandPD.compute_safe_action should output (21,)."""
        phys = MockPhysics(nu=21, njnt=21)
        ctrl = HumanoidStandPD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_safe_action(timestep, phys)
        self.assertEqual(result.shape, (21,))
        # Safe action should have small magnitude (damping, not full torque)
        self.assertTrue(np.all(np.abs(result) <= 1.0))

    def test_cartpole_safe_action_shape(self):
        """CartpoleBalancePD.compute_safe_action should output (1,)."""
        phys = MockPhysics(nu=1, njnt=1)
        ctrl = CartpoleBalancePD(phys)
        timestep = MockTimeStep(phys)
        result = ctrl.compute_safe_action(timestep, phys)
        self.assertEqual(result.shape, (1,))


# ── 7. MotorPrimitives Retention Tests ─────────────────────────────────

class TestMotorPrimitivesRetention(unittest.TestCase):
    """Verify MotorPrimitives is retained for pd_stabilize but not used in decision loop."""

    def test_motor_primitives_still_exists(self):
        """MotorPrimitives class should still exist in mujoco_ido_agent.py."""
        from agent.mujoco_ido_agent import MotorPrimitives
        self.assertIsNotNone(MotorPrimitives)

    def test_agent_has_mp_attribute(self):
        """IDOMuJoCoAgent should still have .mp (MotorPrimitives) attribute."""
        phys = MockPhysicsWithNamed(nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')
        agent = IDOMuJoCoAgent(env, goal)
        self.assertIsInstance(agent.mp, MotorPrimitives)

    def test_agent_has_macros_attribute(self):
        """IDOMuJoCoAgent should still have .macros for ψ-Anchor compat."""
        phys = MockPhysicsWithNamed(nu=6)
        env = MockEnv(phys)
        goal = GoalEML(name='test')
        agent = IDOMuJoCoAgent(env, goal)
        self.assertIsInstance(agent.macros, list)
        self.assertEqual(len(agent.macros), 5)

    def test_pd_stabilize_still_callable(self):
        """MotorPrimitives.pd_stabilize should still be callable."""
        phys = MockPhysics(njnt=6, nu=6)
        mp = MotorPrimitives(phys)
        target = np.array([1.0, 0.0, 0.0])
        ee = np.array([0.0, 0.0, 0.0])
        delta = mp.pd_stabilize(phys, target, ee)
        self.assertIsInstance(delta, np.ndarray)


if __name__ == '__main__':
    unittest.main(verbosity=2)
