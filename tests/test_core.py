"""
Unit tests for core modules: goal_eml_mj, kappa_snap_mj, noether_check_mj.

Tests are designed to run WITHOUT dm_control / MuJoCo installed.
All physics objects are mocked with simple attribute containers.
"""
import sys
import os
import math
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields

import numpy as np

# ── Ensure project root is importable ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.goal_eml_mj import (
    GoalEML, IDO_GOAL_EML_MJ_VERSION,
    make_humanoid_stand_eml,
    make_hopper_stand_eml,
    make_walker_run_eml,
    make_reacher_easy_eml,
)
from core.kappa_snap_mj import (
    gauss_ex_residual, _quat_to_z_axis, IDO_KAPPA_SNAP_MJ_VERSION,
)
from core.noether_check_mj import (
    NoetherViolation, noether_check_mj, _min_geom_distance,
    MAX_TORQUE, TORQUE_MARGIN, SELF_COLLIDE_THRESH, ENERGY_DRIFT_EPS,
    IDO_NOETHER_MJ_VERSION,
)


# ──────────────────────────────────────────────────────
# Helper: Minimal mock physics objects
# ──────────────────────────────────────────────────────
class MockNamedData:
    """Simulates physics.named.data with xpos access."""
    def __init__(self, xpos_map=None):
        self._xpos_map = xpos_map or {}
        self.xpos = MockNamedAccess(self._xpos_map)
        self.cvel = MockNamedAccess(self._xpos_map)


class MockNamedAccess:
    """Simulates named-data subscript access like ['torso', :].

    Raises KeyError for missing keys to match dm_control behavior.
    """
    def __init__(self, data_map):
        self._data_map = data_map

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, _ = key
            if name not in self._data_map:
                raise KeyError(name)
            return self._data_map[name]
        if key not in self._data_map:
            raise KeyError(key)
        return self._data_map[key]


class MockPhysics:
    """Minimal mock dm_control Physics object for factory functions."""
    def __init__(self, xpos_map=None, nq=7, nv=7, nu=6, njnt=6):
        self.named = MagicMock()
        self.named.data = MockNamedData(xpos_map)
        self.model = MagicMock()
        self.model.nq = nq
        self.model.nv = nv
        self.model.nu = nu
        self.model.njnt = njnt
        self.data = MagicMock()


class MockContact:
    """Simulates a single MuJoCo contact entry."""
    def __init__(self, geom1, geom2, dist):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class MockMjData:
    """Minimal mock mjData object for Noether-check tests."""
    def __init__(self, energy=None, actuator_force=None, contacts=None):
        if energy is not None:
            self.energy = energy
        else:
            self.energy = [0.0, 0.0]
        if actuator_force is not None:
            self.actuator_force = actuator_force
        else:
            self.actuator_force = np.zeros(0)
        if contacts is not None:
            self.contact = contacts
        else:
            self.contact = []


# ──────────────────────────────────────────────────────
# 1. GoalEML dataclass tests
# ──────────────────────────────────────────────────────
class TestGoalEMLDataclass(unittest.TestCase):
    """Tests for GoalEML dataclass creation and field validation."""

    def test_version_defined(self):
        """IDO_GOAL_EML_MJ_VERSION should be a version string."""
        self.assertIsInstance(IDO_GOAL_EML_MJ_VERSION, str)
        self.assertTrue(IDO_GOAL_EML_MJ_VERSION.startswith('v'))

    def test_goal_eml_default_creation(self):
        """GoalEML should be creatable with just a name (defaults fill others)."""
        g = GoalEML(name='test_task')
        self.assertEqual(g.name, 'test_task')
        self.assertEqual(g.invariants, [])
        np.testing.assert_array_equal(g.target_pos, np.zeros(3))
        self.assertAlmostEqual(g.delta_K, 0.05)
        self.assertAlmostEqual(g.max_energy_inject, 500.0)
        self.assertAlmostEqual(g.pos_tol, 0.02)
        self.assertAlmostEqual(g.ori_tol, 0.15)

    def test_goal_eml_full_creation(self):
        """GoalEML should accept all fields explicitly."""
        g = GoalEML(
            name='custom',
            invariants=['inv_a', 'inv_b'],
            target_pos=np.array([1.0, 2.0, 3.0]),
            delta_K=0.1,
            max_energy_inject=300.0,
            pos_tol=0.03,
            ori_tol=0.2,
        )
        self.assertEqual(g.name, 'custom')
        self.assertEqual(g.invariants, ['inv_a', 'inv_b'])
        np.testing.assert_array_equal(g.target_pos, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(g.delta_K, 0.1)
        self.assertAlmostEqual(g.max_energy_inject, 300.0)
        self.assertAlmostEqual(g.pos_tol, 0.03)
        self.assertAlmostEqual(g.ori_tol, 0.2)

    def test_goal_eml_fields_count(self):
        """GoalEML should have exactly 7 fields."""
        self.assertEqual(len(fields(GoalEML)), 7)

    def test_goal_eml_invariants_default_factory(self):
        """Each GoalEML instance should have an independent invariants list."""
        g1 = GoalEML(name='a')
        g2 = GoalEML(name='b')
        g1.invariants.append('x')
        self.assertNotIn('x', g2.invariants)
        self.assertEqual(g2.invariants, [])

    def test_goal_eml_target_pos_default_factory(self):
        """Each GoalEML instance should have an independent target_pos array."""
        g1 = GoalEML(name='a')
        g2 = GoalEML(name='b')
        g1.target_pos[0] = 999.0
        self.assertAlmostEqual(g2.target_pos[0], 0.0)


# ──────────────────────────────────────────────────────
# 2. Factory function tests
# ──────────────────────────────────────────────────────
class TestGoalEMLFactories(unittest.TestCase):
    """Tests for 4 GoalEML factory functions with mock physics."""

    def _make_mock_physics(self, torso_pos=None):
        """Helper to create MockPhysics with optional torso position."""
        xpos_map = {}
        if torso_pos is not None:
            xpos_map['torso'] = torso_pos
        return MockPhysics(xpos_map=xpos_map)

    def test_make_humanoid_stand_default(self):
        """make_humanoid_stand_eml with mock physics should return correct GoalEML."""
        phys = self._make_mock_physics()
        g = make_humanoid_stand_eml(phys)
        self.assertEqual(g.name, 'humanoid_stand')
        self.assertEqual(g.invariants,
                         ['torso_upright', 'feet_on_ground', 'no_self_collide'])
        np.testing.assert_array_almost_equal(g.target_pos, [0.0, 0.0, 1.4])
        self.assertAlmostEqual(g.max_energy_inject, 500.0)
        self.assertAlmostEqual(g.delta_K, 0.05)

    def test_make_humanoid_stand_custom_delta_K(self):
        """make_humanoid_stand_eml should use provided delta_K."""
        phys = self._make_mock_physics()
        g = make_humanoid_stand_eml(phys, delta_K=0.1)
        self.assertAlmostEqual(g.delta_K, 0.1)

    def test_make_hopper_stand(self):
        """make_hopper_stand_eml should return correct GoalEML."""
        phys = self._make_mock_physics()
        g = make_hopper_stand_eml(phys)
        self.assertEqual(g.name, 'hopper_stand')
        self.assertEqual(g.invariants,
                         ['torso_upright', 'foot_on_ground', 'no_self_collide'])
        np.testing.assert_array_almost_equal(g.target_pos, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(g.max_energy_inject, 200.0)
        self.assertAlmostEqual(g.delta_K, 0.03)
        self.assertAlmostEqual(g.pos_tol, 0.05)
        self.assertAlmostEqual(g.ori_tol, 0.20)

    def test_make_hopper_stand_custom_delta_K(self):
        """make_hopper_stand_eml should use provided delta_K."""
        phys = self._make_mock_physics()
        g = make_hopper_stand_eml(phys, delta_K=0.15)
        self.assertAlmostEqual(g.delta_K, 0.15)

    def test_make_walker_run(self):
        """make_walker_run_eml should return correct GoalEML."""
        phys = self._make_mock_physics()
        g = make_walker_run_eml(phys)
        self.assertEqual(g.name, 'walker_run')
        self.assertEqual(g.invariants,
                         ['com_x_advancing', 'not_fallen', 'no_self_collide'])
        np.testing.assert_array_almost_equal(g.target_pos, [10.0, 0.0, 0.0])
        self.assertAlmostEqual(g.max_energy_inject, 600.0)
        self.assertAlmostEqual(g.delta_K, 0.05)
        self.assertAlmostEqual(g.pos_tol, 0.10)
        self.assertAlmostEqual(g.ori_tol, 0.25)

    def test_make_reacher_easy(self):
        """make_reacher_easy_eml should return correct GoalEML."""
        phys = self._make_mock_physics()
        g = make_reacher_easy_eml(phys)
        self.assertEqual(g.name, 'reacher_easy')
        self.assertEqual(g.invariants, ['ee_at_target'])
        np.testing.assert_array_almost_equal(g.target_pos, [0.1, 0.1, 0.0])
        self.assertAlmostEqual(g.max_energy_inject, 50.0)
        self.assertAlmostEqual(g.delta_K, 0.02)
        self.assertAlmostEqual(g.pos_tol, 0.01)
        self.assertAlmostEqual(g.ori_tol, 0.0)

    def test_reacher_easy_ori_tol_zero(self):
        """Reacher-Easy should have ori_tol=0 (no orientation constraint)."""
        phys = self._make_mock_physics()
        g = make_reacher_easy_eml(phys)
        self.assertEqual(g.ori_tol, 0.0)


# ──────────────────────────────────────────────────────
# 3. kappa_snap_mj tests
# ──────────────────────────────────────────────────────
class TestQuatToZAxis(unittest.TestCase):
    """Tests for _quat_to_z_axis quaternion-to-z-axis conversion."""

    def test_identity_quat(self):
        """Identity quaternion [1,0,0,0] → z-axis = [0,0,1]."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        result = _quat_to_z_axis(quat)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_90deg_rotation_around_x(self):
        """90° rotation around x-axis: z-axis rotates to [0,-1,0]."""
        # Quaternion for 90° around x: qw=cos(45°), qx=sin(45°), qy=0, qz=0
        angle = math.pi / 4  # half-angle
        quat = np.array([math.cos(angle), math.sin(angle), 0.0, 0.0])
        result = _quat_to_z_axis(quat)
        # Expected: zx=0, zy=-sin(90°)=-1, zz=0
        np.testing.assert_array_almost_equal(result, [0.0, -1.0, 0.0], decimal=5)

    def test_none_input(self):
        """None quaternion should return default z-axis [0,0,1]."""
        result = _quat_to_z_axis(None)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_short_quat(self):
        """Quaternion shorter than 4 elements should return [0,0,1]."""
        result = _quat_to_z_axis(np.array([1.0, 0.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_empty_quat(self):
        """Empty quaternion array should return [0,0,1]."""
        result = _quat_to_z_axis(np.array([]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_180deg_rotation_around_y(self):
        """180° rotation around y-axis: z-axis flips to [0,0,-1]."""
        # qw=cos(90°)=0, qx=0, qy=sin(90°)=1, qz=0
        angle = math.pi / 2
        quat = np.array([math.cos(angle), 0.0, math.sin(angle), 0.0])
        result = _quat_to_z_axis(quat)
        # zx=0, zy=0, zz=-1 (z-axis flipped)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, -1.0], decimal=5)


class TestGaussExResidual(unittest.TestCase):
    """Tests for gauss_ex_residual η computation."""

    def _make_goal(self, target_pos=None, max_energy=500.0, delta_K=0.05):
        """Helper to create a GoalEML for residual tests."""
        return GoalEML(
            name='test',
            target_pos=target_pos if target_pos is not None else np.array([1.0, 0.0, 0.0]),
            delta_K=delta_K,
            max_energy_inject=max_energy,
        )

    def test_zero_residual_at_target(self):
        """When ee_pos == target_pos, E <= budget, vel=0 → η ≈ 0 (only tilt from quat)."""
        goal = self._make_goal(target_pos=np.array([1.0, 0.0, 0.0]))
        z_i = {
            'ee_pos': np.array([1.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),  # identity quat
            'E_total': 100.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # Position error = 0, tilt = 0 (identity quat → z=[0,0,1]), energy excess = 0
        self.assertAlmostEqual(eta, 0.0, places=5)

    def test_position_error_only(self):
        """η should increase with position error (squared, weighted by w_pos)."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]))
        z_i = {
            'ee_pos': np.array([2.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # pos_err = 2.0, η = w_pos * 4.0 = 4.0
        self.assertAlmostEqual(eta, 4.0, places=5)

    def test_position_error_with_custom_weight(self):
        """η should reflect custom w_pos weight."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]))
        z_i = {
            'ee_pos': np.array([1.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal, w_pos=2.0, w_ori=0.0, w_eng=0.0, w_vel=0.0)
        # pos_err = 1.0, η = 2.0 * 1.0 = 2.0
        self.assertAlmostEqual(eta, 2.0, places=5)

    def test_energy_excess_contribution(self):
        """η should increase when E_total exceeds max_energy_inject."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]), max_energy=100.0)
        z_i = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),  # matches target → pos_err = 0
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 200.0,  # 100 over budget
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # energy_excess = 100, η = w_eng * 100^2 = 0.01 * 10000 = 100
        self.assertAlmostEqual(eta, 100.0, places=2)

    def test_velocity_contribution(self):
        """η should increase with ee_vel magnitude."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]))
        z_i = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        eta = gauss_ex_residual(z_i, goal)
        # vel_mag = 10.0 (from first 3 elements), η += w_vel * 100 = 0.05 * 100 = 5.0
        self.assertAlmostEqual(eta, 5.0, places=5)

    def test_no_ee_pos_key(self):
        """When z_i lacks 'ee_pos', default to zeros → large pos_err."""
        goal = self._make_goal(target_pos=np.array([5.0, 0.0, 0.0]))
        z_i = {
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # ee_pos defaults to zeros, pos_err = 5.0, η = w_pos * 25 = 25
        self.assertAlmostEqual(eta, 25.0, places=5)

    def test_no_qpos_key(self):
        """When z_i lacks 'qpos', _quat_to_z_axis gets None → default tilt=0."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]))
        z_i = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),
            'E_total': 0.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # All components ~0
        self.assertAlmostEqual(eta, 0.0, places=5)

    def test_combined_residual(self):
        """η should sum all weighted components correctly."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]), max_energy=100.0)
        z_i = {
            'ee_pos': np.array([1.0, 0.0, 0.0]),  # pos_err = 1.0
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),  # identity → tilt=0
            'E_total': 150.0,  # excess = 50
            'ee_vel': np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # vel_mag = 2
        }
        eta = gauss_ex_residual(z_i, goal)
        # η = w_pos*1 + w_ori*0 + w_eng*2500 + w_vel*4 = 1.0 + 0 + 25.0 + 0.2 = 26.2
        expected = 1.0 * 1.0 + 0.0 + 0.01 * 50.0 ** 2 + 0.05 * 2.0 ** 2
        self.assertAlmostEqual(eta, expected, places=5)

    def test_eta_return_type(self):
        """gauss_ex_residual should return a Python float, not numpy scalar."""
        goal = self._make_goal()
        z_i = {'ee_pos': np.zeros(3), 'E_total': 0.0}
        eta = gauss_ex_residual(z_i, goal)
        self.assertIsInstance(eta, float)

    def test_energy_within_budget(self):
        """When E_total < max_energy_inject, energy excess should be 0."""
        goal = self._make_goal(target_pos=np.array([0.0, 0.0, 0.0]), max_energy=500.0)
        z_i = {
            'ee_pos': np.array([0.0, 0.0, 0.0]),  # matches target → pos_err = 0
            'qpos': np.array([1.0, 0.0, 0.0, 0.0]),
            'E_total': 400.0,
            'ee_vel': np.zeros(6),
        }
        eta = gauss_ex_residual(z_i, goal)
        # energy_excess = max(0, 400-500) = 0, so no energy contribution
        # pos_err=0, tilt=0, vel=0 → η = 0
        self.assertAlmostEqual(eta, 0.0, places=5)

    def test_kappa_snap_version(self):
        """IDO_KAPPA_SNAP_MJ_VERSION should be a version string."""
        self.assertIsInstance(IDO_KAPPA_SNAP_MJ_VERSION, str)


# ──────────────────────────────────────────────────────
# 4. noether_check_mj tests
# ──────────────────────────────────────────────────────
class TestNoetherViolation(unittest.TestCase):
    """Tests for NoetherViolation dataclass."""

    def test_default_creation(self):
        """NoetherViolation default should have ok=True and empty strings."""
        v = NoetherViolation(ok=True)
        self.assertTrue(v.ok)
        self.assertEqual(v.code, "")
        self.assertEqual(v.message, "")

    def test_violation_creation(self):
        """NoetherViolation should store violation details."""
        v = NoetherViolation(ok=False, code="E", message="energy violation")
        self.assertFalse(v.ok)
        self.assertEqual(v.code, "E")
        self.assertEqual(v.message, "energy violation")

    def test_fields_count(self):
        """NoetherViolation should have exactly 3 fields."""
        from dataclasses import fields as dc_fields
        self.assertEqual(len(dc_fields(NoetherViolation)), 3)


class TestMinGeomDistance(unittest.TestCase):
    """Tests for _min_geom_distance helper."""

    def test_no_contacts(self):
        """No contacts → default distance 1.0."""
        data = MockMjData(contacts=[])
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 1.0)

    def test_no_contact_attribute(self):
        """Missing 'contact' attribute → default distance 1.0."""
        data = object()  # no 'contact' attribute
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 1.0)

    def test_single_contact_distinct_geoms(self):
        """Single contact with distinct geoms → return its distance."""
        contacts = [MockContact(0, 1, 0.02)]
        data = MockMjData(contacts=contacts)
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 0.02)

    def test_identical_geoms_filtered(self):
        """Contacts with geom1 == geom2 should be ignored."""
        contacts = [MockContact(0, 0, -0.01), MockContact(1, 2, 0.03)]
        data = MockMjData(contacts=contacts)
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 0.03)

    def test_only_identical_geoms(self):
        """If all contacts have geom1 == geom2, return default 1.0."""
        contacts = [MockContact(0, 0, -0.01), MockContact(1, 1, -0.02)]
        data = MockMjData(contacts=contacts)
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 1.0)

    def test_multiple_contacts_min_distance(self):
        """Should return the minimum distance among distinct-geom contacts."""
        contacts = [
            MockContact(0, 1, 0.05),
            MockContact(2, 3, 0.01),
            MockContact(4, 5, 0.10),
        ]
        data = MockMjData(contacts=contacts)
        result = _min_geom_distance(data)
        self.assertAlmostEqual(result, 0.01)


class TestNoetherCheck(unittest.TestCase):
    """Tests for noether_check_mj conservation gate."""

    def _make_goal(self, max_energy=500.0):
        """Helper to create GoalEML with specified energy budget."""
        return GoalEML(name='test', max_energy_inject=max_energy)

    def test_all_pass_no_contacts(self):
        """All gates pass: energy conserved, low torque, no collisions."""
        goal = self._make_goal(max_energy=500.0)
        prev = MockMjData(energy=[100.0, 200.0])  # E_prev = 300
        cur = MockMjData(energy=[150.0, 200.0],  # E_cur = 350
                         actuator_force=np.array([100.0, 200.0]),
                         contacts=[])
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_energy_violation(self):
        """Energy increase exceeding budget → fail with energy message."""
        goal = self._make_goal(max_energy=50.0)
        prev = MockMjData(energy=[100.0, 100.0])  # E_prev = 200
        cur = MockMjData(energy=[200.0, 200.0],   # E_cur = 400, dE = 200
                         actuator_force=np.array([10.0]),
                         contacts=[])
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertFalse(ok)
        self.assertIn("Noether-E", msg)
        self.assertIn("energy", msg)

    def test_energy_within_budget(self):
        """Energy increase within budget → pass."""
        goal = self._make_goal(max_energy=500.0)
        prev = MockMjData(energy=[100.0, 200.0])  # E_prev = 300
        cur = MockMjData(energy=[200.0, 200.0],   # E_cur = 400, dE = 100
                         actuator_force=np.zeros(2),
                         contacts=[])
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertTrue(ok)

    def test_torque_violation(self):
        """Torque exceeding limit * margin → fail with torque message."""
        goal = self._make_goal()
        prev = MockMjData(energy=[0.0, 0.0])
        cur = MockMjData(energy=[0.0, 0.0],
                         actuator_force=np.array([600.0]),
                         contacts=[])
        # max_torque=500, margin=1.05 → limit = 525, force=600 exceeds
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertFalse(ok)
        self.assertIn("Noether-F", msg)

    def test_torque_within_limit(self):
        """Torque within limit * margin → pass."""
        goal = self._make_goal()
        prev = MockMjData(energy=[0.0, 0.0])
        cur = MockMjData(energy=[0.0, 0.0],
                         actuator_force=np.array([400.0]),
                         contacts=[])
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertTrue(ok)

    def test_self_collision(self):
        """Min geom distance < threshold → fail with collision message."""
        goal = self._make_goal()
        prev = MockMjData(energy=[0.0, 0.0])
        contacts = [MockContact(0, 1, 0.001)]  # < SELF_COLLIDE_THRESH (0.005)
        cur = MockMjData(energy=[0.0, 0.0],
                         actuator_force=np.array([10.0]),
                         contacts=contacts)
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertFalse(ok)
        self.assertIn("Noether-C", msg)

    def test_custom_thresholds(self):
        """Custom max_torque, torque_margin, collide_thresh should override defaults."""
        goal = self._make_goal()
        prev = MockMjData(energy=[0.0, 0.0])
        cur = MockMjData(energy=[0.0, 0.0],
                         actuator_force=np.array([800.0]),  # would fail at default
                         contacts=[])
        # With custom max_torque=1000, margin=1.0 → limit=1000
        ok, msg = noether_check_mj(prev, cur, goal,
                                    max_torque=1000.0,
                                    torque_margin=1.0)
        self.assertTrue(ok)

    def test_energy_drift_eps_borderline(self):
        """Energy increase exactly = budget + ε should NOT violate."""
        goal = self._make_goal(max_energy=100.0)
        prev = MockMjData(energy=[0.0, 0.0])      # E_prev = 0
        cur = MockMjData(energy=[100.0 + ENERGY_DRIFT_EPS, 0.0],  # dE = budget + ε
                         actuator_force=np.zeros(2),
                         contacts=[])
        # dE = 100 + eps, budget + eps = 100 + eps → NOT strictly greater → pass
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertTrue(ok)

    def test_no_actuator_force_attribute(self):
        """Missing actuator_force → skip torque check (pass)."""
        goal = self._make_goal()
        prev = MockMjData(energy=[0.0, 0.0])
        cur = MockMjData(energy=[0.0, 0.0], contacts=[])
        # Remove actuator_force
        if hasattr(cur, 'actuator_force'):
            delattr(cur, 'actuator_force')
        ok, msg = noether_check_mj(prev, cur, goal)
        self.assertTrue(ok)

    def test_default_constants(self):
        """Module-level constants should have expected values."""
        self.assertAlmostEqual(MAX_TORQUE, 500.0)
        self.assertAlmostEqual(TORQUE_MARGIN, 1.05)
        self.assertAlmostEqual(SELF_COLLIDE_THRESH, 0.005)
        self.assertAlmostEqual(ENERGY_DRIFT_EPS, 1e-3)

    def test_noether_version(self):
        """IDO_NOETHER_MJ_VERSION should be a version string."""
        self.assertIsInstance(IDO_NOETHER_MJ_VERSION, str)


# ──────────────────────────────────────────────────────
# 5. Import consistency tests
# ──────────────────────────────────────────────────────
class TestImportConsistency(unittest.TestCase):
    """Tests verifying that module imports resolve correctly via submodule paths."""

    def test_core_goal_eml_mj_importable(self):
        """core.goal_eml_mj should be importable and export GoalEML."""
        from core.goal_eml_mj import GoalEML as G
        self.assertIs(G, GoalEML)

    def test_core_kappa_snap_mj_importable(self):
        """core.kappa_snap_mj should be importable and export gauss_ex_residual."""
        from core.kappa_snap_mj import gauss_ex_residual as F
        self.assertIs(F, gauss_ex_residual)

    def test_core_noether_check_mj_importable(self):
        """core.noether_check_mj should be importable and export noether_check_mj."""
        from core.noether_check_mj import noether_check_mj as N
        self.assertIs(N, noether_check_mj)

    def test_core_noether_violation_importable(self):
        """core.noether_check_mj should be importable and export NoetherViolation."""
        from core.noether_check_mj import NoetherViolation as V
        self.assertIs(V, NoetherViolation)

    def test_goal_eml_version_string(self):
        """GoalEML module version should be a non-empty string."""
        self.assertTrue(len(IDO_GOAL_EML_MJ_VERSION) > 0)

    def test_core_package_is_importable(self):
        """core package (__init__.py) should be importable."""
        import core
        self.assertTrue(hasattr(core, '__file__'))

    def test_agent_mujoco_ido_agent_importable(self):
        """agent.mujoco_ido_agent should be importable."""
        from agent.mujoco_ido_agent import IDOMuJoCoAgent, MotorPrimitives
        self.assertIsNotNone(IDOMuJoCoAgent)
        self.assertIsNotNone(MotorPrimitives)

    def test_cross_module_dependency(self):
        """agent module should successfully import from core modules."""
        # This tests that mujoco_ido_agent.py can import from core.*
        # The import already happened at module level; verify it worked
        from agent.mujoco_ido_agent import IDOMuJoCoAgent
        agent_module = sys.modules.get('agent.mujoco_ido_agent')
        self.assertIsNotNone(agent_module)


if __name__ == '__main__':
    unittest.main(verbosity=2)
