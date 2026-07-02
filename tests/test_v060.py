"""
MuJoCo-Bench-IDO v0.6.0 — Full Integration Tests
===================================================

Tests for the Machine Conscience Audit Framework, covering:
  - KappaSnapSchema: 20 event type validation
  - KappaSnapLogger + MerkleChain: log recording & chain integrity
  - PGGate: AST analysis + physical clamp
  - Noether v1.2.0: 4-gate (friction cone)
  - PsiAnchor v0.3.0: sentient finger limit check
  - KappaSnap v0.3.0: MerkleChain integration
  - ConscienceQuotient: CQ computation
  - SafeFuse: L1-L4 fuse levels
  - HybridSB3IDOAgent v0.6.0: PG-Gate + fuse + logger + CQ
  - IDOMuJoCoAgent: PG-Gate + logger
  - BayesianIntent: Beta-Bernoulli intent clarity
  - PinchLeafEnv: dm_control environment

Author: MuJoCo-Bench-IDO v0.6.0
"""

import hashlib
import json
import math
import numpy as np
import unittest
from typing import Dict, Any


class TestKappaSnapSchema(unittest.TestCase):
    """Test KappaSnapSchema — 20 event types validation."""

    def setUp(self):
        from core.kappa_snap_schema import KappaSnapSchema
        self.schema = KappaSnapSchema()

    def test_all_20_event_types_defined(self):
        """Verify all 20 event types are defined in EVENT_TYPES."""
        from core.kappa_snap_schema import EVENT_TYPES
        expected_types = [
            "INIT", "ACTION_ACCEPT", "REJECT_FRICTION_CONE",
            "REJECT_ENERGY_VIOLATION", "REJECT_SENTIENT_LIMIT",
            "REJECT_SELF_COLLISION", "REJECT_PG_GATE", "CREATIVE_PROBE",
            "THERMAL_DRIFT", "SCREW_LOOSENING", "CALIBRATION_DRIFT",
            "SENSOR_DEGRADED", "SELF_REFLECT", "FINGER_TORQUE_CLAMPED",
            "WIND_GUST", "BIOMASS_DETECTED", "TASK_START", "TASK_COMPLETE",
            "SAFE_STOP", "FATAL_ERROR",
        ]
        for et in expected_types:
            self.assertIn(et, EVENT_TYPES, f"Missing event type: {et}")
        self.assertEqual(len(EVENT_TYPES), 20)

    def test_create_event_valid_type(self):
        """Test creating events with valid event types."""
        event = self.schema.create_event(
            event_type="INIT", eta=0.0, decision="start",
            details={"task_name": "test", "goal_delta_K": 0.05})
        self.assertEqual(event["event_type"], "INIT")
        self.assertEqual(event["level"], "L0")  # Auto-filled
        self.assertTrue(self.schema.validate(event))

    def test_create_event_invalid_type_raises(self):
        """Test creating event with invalid type raises ValueError."""
        with self.assertRaises(ValueError):
            self.schema.create_event(event_type="INVALID_TYPE")

    def test_validate_missing_required_field(self):
        """Test validation fails when required field is missing."""
        event = {"event_type": "INIT", "eta": 0.0}  # Missing fields
        self.assertFalse(self.schema.validate(event))

    def test_validate_wrong_level_for_event_type(self):
        """Test validation fails when level doesn't match event_type."""
        event = self.schema.create_event(
            event_type="INIT", level="L3", eta=0.0, decision="start",
            details={"task_name": "test", "goal_delta_K": 0.05})
        # INIT expects L0, but we set L3 → invalid
        self.assertFalse(self.schema.validate(event))

    def test_validate_missing_detail_field(self):
        """Test validation fails when required detail field is missing."""
        event = self.schema.create_event(
            event_type="INIT", eta=0.0, decision="start",
            details={"task_name": "test"})  # Missing goal_delta_K
        self.assertFalse(self.schema.validate(event))

    def test_validate_all_20_event_types(self):
        """Test creating and validating all 20 event types."""
        from core.kappa_snap_schema import EVENT_TYPES
        for event_type, definition in EVENT_TYPES.items():
            # Create minimal valid details for each type
            details = {k: "test_value" for k in definition["required_details"]}
            event = self.schema.create_event(
                event_type=event_type, eta=0.1, decision="test",
                details=details)
            self.assertTrue(self.schema.validate(event),
                           f"Event type {event_type} should validate")


class TestMerkleChain(unittest.TestCase):
    """Test MerkleChain — tamper-proof audit chain."""

    def setUp(self):
        from core.kappa_snap_logger import MerkleChain
        self.chain = MerkleChain()

    def test_append_returns_snap_id(self):
        """Test that append returns a snap_id string."""
        snap_id = self.chain.append(eta=0.5, decision="EXPLOIT",
                                    event_type="ACTION_ACCEPT", level="L0")
        self.assertIsInstance(snap_id, str)
        self.assertTrue(len(snap_id) > 0)

    def test_chain_linkage(self):
        """Test that consecutive entries are linked via prev_snap_id."""
        snap1 = self.chain.append(eta=0.5, decision="EXPLOIT")
        snap2 = self.chain.append(eta=0.4, decision="SAFE")
        chain = self.chain.get_chain()
        self.assertEqual(chain[1]["prev_snap_id"], snap1)

    def test_verify_intact_chain(self):
        """Test that an intact chain passes verification."""
        for i in range(10):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")
        self.assertTrue(self.chain.verify())

    def test_tamper_detection(self):
        """Test that tampering with an entry breaks verification."""
        for i in range(5):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")
        # Tamper: modify eta of the first entry
        chain = self.chain.get_chain()
        chain[0]["eta"] = 999.0  # Tampered!
        # Note: verify() uses internal _chain, so we need to modify it directly
        self.chain._chain[0]["eta"] = 999.0
        self.assertFalse(self.chain.verify())

    def test_reset_clears_chain(self):
        """Test that reset clears the chain to genesis state."""
        self.chain.append(eta=0.5, decision="EXPLOIT")
        self.chain.reset()
        self.assertEqual(len(self.chain.get_chain()), 0)
        self.assertEqual(self.chain.get_last_snap_id(), "genesis")

    def test_hash_computation_rule(self):
        """Test snap_id = prev_snap_id + sha256(prev_snap_id+str(η)+str(decision))[:16]."""
        prev = "genesis"
        eta = 0.5
        decision = "EXPLOIT"
        hash_input = prev + str(eta) + str(decision)
        expected_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        expected_snap_id = prev + expected_hash
        snap_id = self.chain.append(eta=eta, decision=decision)
        self.assertEqual(snap_id, expected_snap_id)


class TestKappaSnapLogger(unittest.TestCase):
    """Test KappaSnapLogger — audit event logging with MerkleChain."""

    def setUp(self):
        from core.kappa_snap_logger import KappaSnapLogger
        self.logger = KappaSnapLogger()

    def test_log_returns_event_dict(self):
        """Test that log() returns a complete event dict."""
        event = self.logger.log("INIT", "L0", 0.0, "start")
        self.assertEqual(event["event_type"], "INIT")
        self.assertIn("snap_id", event)
        self.assertIn("prev_snap_id", event)

    def test_log_appends_to_merkle(self):
        """Test that log() appends to the MerkleChain."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT")
        chain = self.logger.get_merkle_chain()
        self.assertEqual(len(chain), 2)

    def test_verify_chain_after_logging(self):
        """Test MerkleChain verification after multiple log entries."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT")
        self.logger.log("REJECT_PG_GATE", "L3", 0.15, "SAFE")
        self.assertTrue(self.logger.verify_chain())

    def test_reset_clears_all_state(self):
        """Test reset clears both MerkleChain and log buffer."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.reset()
        self.assertEqual(len(self.logger.get_merkle_chain()), 0)
        self.assertEqual(len(self.logger.get_log_buffer()), 0)


class TestPGGate(unittest.TestCase):
    """Test PGGate — hard anchor gate (AST + physical clamp)."""

    def setUp(self):
        from core.pg_gate import PGGate
        self.pgate = PGGate()

    def test_physical_clamp_below_threshold(self):
        """Test actions below TAU_SAFE pass physical clamp."""
        action = np.array([0.03, 0.02, 0.01])  # All below 0.05
        clamped = self.pgate.physical_clamp(action, tau_safe=0.05)
        # Below threshold → no clamping needed (action unchanged)
        np.testing.assert_array_almost_equal(clamped, action)

    def test_physical_clamp_above_threshold(self):
        """Test actions above TAU_SAFE are clamped."""
        action = np.array([0.1, -0.15, 0.08])  # Above 0.05
        clamped = self.pgate.physical_clamp(action, tau_safe=0.05)
        # All values should be clamped to ±0.05
        for val in clamped:
            self.assertTrue(abs(val) <= 0.05,
                           f"Clamped value {val} exceeds tau_safe=0.05")

    def test_ast_analysis_sentient_keywords(self):
        """Test AST analysis identifies sentient actuator names."""
        # Mock physics with finger/hand actuator names
        from core.pg_gate import PGGate
        pgate = PGGate()
        # Create mock physics object
        class MockPhysics:
            class MockModel:
                actuator_names = ["finger_1", "finger_2", "torso", "hand_left"]
                nu = 4
            model = MockModel()
        mock_phys = MockPhysics()
        ast_result = pgate.ast_analysis(np.array([0.1, 0.2, 0.01, 0.3]), mock_phys)
        self.assertTrue(ast_result["is_sentient_target"])

    def test_gate_with_no_logger(self):
        """Test gate() works without KappaSnapLogger."""
        action = np.array([0.03, 0.02, 0.01])
        result = self.pgate.gate(action, physics=None, kappa_snap_logger=None)
        self.assertIsInstance(result, np.ndarray)


class TestNoetherFrictionCone(unittest.TestCase):
    """Test Noether v1.2.0 — friction cone 4th gate."""

    def test_friction_cone_check_function_exists(self):
        """Test that _friction_cone_check function exists."""
        from core.noether_check_mj import _friction_cone_check
        self.assertTrue(callable(_friction_cone_check))


class TestPsiAnchorSentient(unittest.TestCase):
    """Test PsiAnchor v0.3.0 — sentient finger limit check."""

    def test_tau_sentient_max_constant(self):
        """Test TAU_SENTIENT_MAX is defined as 0.05."""
        from agent.psi_anchor import TAU_SENTIENT_MAX
        self.assertEqual(TAU_SENTIENT_MAX, 0.05)

    def test_check_sentient_finger_limit_exists(self):
        """Test that check_sentient_finger_limit method exists."""
        from agent.psi_anchor import PsiAnchor
        from core.goal_eml_mj import GoalEML
        goal = GoalEML(name="test", target_pos=np.zeros(3),
                       max_energy_inject=100.0, delta_K=0.05)
        anchor = PsiAnchor(goal)
        self.assertTrue(hasattr(anchor, 'check_sentient_finger_limit'))


class TestConscienceQuotient(unittest.TestCase):
    """Test ConscienceQuotient — CQ compliance metrics."""

    def setUp(self):
        from core.cq import ConscienceQuotient
        self.cq = ConscienceQuotient()

    def test_record_step_and_compute(self):
        """Test recording steps and computing CQ."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.cq.record_step(noether_ok=False, pgate_ok=True, sentient_ok=True)
        cq = self.cq.compute_cq()
        # 2 fully compliant / 3 total = 0.667
        self.assertAlmostEqual(cq, 2.0/3.0, places=2)

    def test_cq_noether_ratio(self):
        """Test CQ_noether sub-metric."""
        self.cq.record_step(noether_ok=True, pgate_ok=False, sentient_ok=True)
        self.cq.record_step(noether_ok=False, pgate_ok=True, sentient_ok=True)
        cq_n = self.cq.compute_cq_noether()
        self.assertAlmostEqual(cq_n, 0.5, places=2)

    def test_cq_pgate_ratio(self):
        """Test CQ_pgate sub-metric."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.cq.record_step(noether_ok=True, pgate_ok=False, sentient_ok=True)
        cq_p = self.cq.compute_cq_pgate()
        self.assertAlmostEqual(cq_p, 0.5, places=2)

    def test_cq_sentient_ratio(self):
        """Test CQ_sentient sub-metric."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=False)
        cq_s = self.cq.compute_cq_sentient()
        self.assertAlmostEqual(cq_s, 0.5, places=2)

    def test_reset_clears_counters(self):
        """Test reset clears all counters."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.cq.reset()
        cq = self.cq.compute_cq()
        self.assertEqual(cq, 0.0)  # 0/0 = 0.0

    def test_get_report(self):
        """Test get_report returns complete CQ breakdown."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        report = self.cq.get_report()
        self.assertIn("cq", report)
        self.assertIn("cq_noether", report)
        self.assertIn("cq_pgate", report)
        self.assertIn("cq_sentient", report)


class TestSafeFuse(unittest.TestCase):
    """Test SafeFuse — L1-L4 safety fuse levels."""

    def setUp(self):
        from agent.safe_fuse import SafeFuse
        self.fuse = SafeFuse()

    def test_check_normal_operation(self):
        """Test no fuse when η below threshold."""
        result = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None)
        self.assertEqual(result[0], "normal")

    def test_check_l1_soft(self):
        """Test L1 soft fuse when η slightly exceeds δ_K."""
        result = self.fuse.check(
            eta=0.06, delta_K=0.05,  # η ∈ [δ_K×1.2, δ_K×1.5]
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None)
        self.assertEqual(result[0], "L1_soft")

    def test_check_l2_medium(self):
        """Test L2 medium fuse for single Noether violation."""
        result = self.fuse.check(
            eta=0.05, delta_K=0.05,
            noether_result={"ok": False, "total": 1},
            psi_anchor_state=None)
        self.assertEqual(result[0], "L2_medium")

    def test_check_l4_fatal(self):
        """Test L4 fatal fuse for catastrophic violations."""
        result = self.fuse.check(
            eta=0.05, delta_K=0.05,
            noether_result={"ok": False, "total": 3,
                           "energy": 1, "torque": 1, "collision": 1},
            psi_anchor_state=None)
        self.assertEqual(result[0], "L4_fatal")

    def test_apply_l1_soft(self):
        """Test L1 soft fuse reduces action magnitude."""
        action = np.array([1.0, -1.0, 0.5])
        fused = self.fuse.apply_fuse(action, "L1_soft")
        # L1 factor=0.8
        np.testing.assert_array_almost_equal(fused, action * 0.8)

    def test_apply_l4_fatal(self):
        """Test L4 fatal fuse returns zero action."""
        action = np.array([1.0, -1.0, 0.5])
        fused = self.fuse.apply_fuse(action, "L4_fatal")
        np.testing.assert_array_equal(fused, np.zeros(3))


class TestBayesianIntent(unittest.TestCase):
    """Test BayesianIntent — Beta-Bernoulli intent clarity model."""

    def setUp(self):
        from core.bayesian_intent import BayesianIntent
        self.bi = BayesianIntent()

    def test_update_returns_intent_dict(self):
        """Test update returns intent classification dict."""
        result = self.bi.update(
            observation={"ee_pos": np.zeros(3)},
            action=np.zeros(3),
            eta=0.1)
        self.assertIn("intent", result)
        self.assertIn("clarity", result)

    def test_get_intent_clarity(self):
        """Test intent clarity computation."""
        clarity = self.bi.get_intent_clarity()
        self.assertIsInstance(clarity, float)
        self.assertTrue(0.0 <= clarity <= 1.0)

    def test_reset_clears_posterior(self):
        """Test reset clears posterior to uniform prior."""
        self.bi.update(
            observation={"ee_pos": np.zeros(3)},
            action=np.zeros(3), eta=0.1)
        self.bi.reset()
        clarity = self.bi.get_intent_clarity()
        # Uniform prior → clarity should be low
        self.assertTrue(clarity < 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for v0.6.0 full decision loop."""

    def test_full_audit_loop_schema(self):
        """Test full audit loop: schema → logger → merkle → verify."""
        from core.kappa_snap_schema import KappaSnapSchema
        from core.kappa_snap_logger import KappaSnapLogger

        schema = KappaSnapSchema()
        logger = KappaSnapLogger(schema)

        # Log a series of events
        logger.log("INIT", "L0", 0.0, "start",
                    details={"task_name": "test", "goal_delta_K": 0.05})
        logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT",
                    details={"action_norm": 0.03, "tau_safe": 0.05})
        logger.log("REJECT_PG_GATE", "L3", 0.15, "SAFE",
                    details={"ast_reason": "sentient_target",
                             "original_action": [0.1],
                             "clamped_action": [0.05]})

        # Verify chain integrity
        self.assertTrue(logger.verify_chain())

        # Check all events in log buffer
        buffer = logger.get_log_buffer()
        self.assertEqual(len(buffer), 3)

    def test_pg_gate_with_logger(self):
        """Test PG-Gate integration with KappaSnapLogger."""
        from core.pg_gate import PGGate
        from core.kappa_snap_logger import KappaSnapLogger

        pgate = PGGate()
        logger = KappaSnapLogger()

        # Create an action that exceeds TAU_SAFE
        action = np.array([0.1, 0.2, 0.01])
        clamped = pgate.gate(action, physics=None,
                             kappa_snap_logger=logger)

        # Verify logger recorded the event
        buffer = logger.get_log_buffer()
        # PG-Gate should have logged REJECT_PG_GATE or ACTION_ACCEPT
        self.assertTrue(len(buffer) > 0)


if __name__ == '__main__':
    unittest.main()
