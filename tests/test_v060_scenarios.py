"""
MuJoCo-Bench-IDO v0.6.0 — Scenario-Level QA Tests (a-f + Integration)
======================================================================

Comprehensive tests for specified scenarios:
  a) PG-Gate clamping (action > TAU_SAFE, verify all <= TAU_SAFE)
  b) Merkle chain integrity (10 appends, verify True; tamper eta, verify False)
  c) SafeFuse L3 test (3 consecutive Noether violations, L1→L2→L3 progression)
  d) CQ compliance rate (100 steps: 80/90/95 → CQ=min=0.80)
  e) Friction cone test (f_t=0.5, f_n=0.6, mu=0.8 → ||f_t||/f_n=0.833 > mu → violation)
  f) BayesianIntent test (20 goal_directed updates → clarity increases)

Plus:
  - Integration test of HybridSB3IDOAgent v0.6.0 decision loop
  - PsiAnchor sentient finger limit edge cases
  - SafeFuse apply_fuse for L2 and L3

Author: QA Engineer Yan (严过关) — MuJoCo-Bench-IDO v0.6.0
"""

import hashlib
import numpy as np
import unittest
from typing import Dict, Any, List


# ── Scenario a: PG-Gate clamping ──────────────────────────────────────────

class TestPGGateClamping(unittest.TestCase):
    """Scenario a: PG-Gate clamping — action > TAU_SAFE → all elements <= TAU_SAFE."""

    def setUp(self):
        from core.pg_gate import PGGate, TAU_SAFE
        self.pgate = PGGate()
        self.TAU_SAFE = TAU_SAFE

    def test_large_positive_action_clamped(self):
        """Action with values well above TAU_SAFE should be fully clamped."""
        action = np.array([1.0, 0.5, 0.3, -1.0, -0.5])
        clamped = self.pgate.physical_clamp(action, tau_safe=self.TAU_SAFE)
        for val in clamped:
            self.assertTrue(abs(val) <= self.TAU_SAFE,
                           f"Value {val} exceeds TAU_SAFE={self.TAU_SAFE}")

    def test_action_with_single_large_element(self):
        """Single large element should be clamped while small ones stay."""
        action = np.array([0.01, 0.02, 10.0, 0.03])
        clamped = self.pgate.physical_clamp(action, tau_safe=self.TAU_SAFE)
        # Small values unchanged
        self.assertAlmostEqual(clamped[0], 0.01)
        self.assertAlmostEqual(clamped[1], 0.02)
        # Large value clamped
        self.assertAlmostEqual(clamped[2], self.TAU_SAFE)
        self.assertAlmostEqual(clamped[3], 0.03)

    def test_gate_full_pipeline_clamps_all(self):
        """gate() with no physics (fallback) should apply global sanity clip [-1, 1].

        v0.7.0 fix: When no physics is provided, PGGate skips AST analysis
        entirely and only applies _global_sanity_clip([-1, 1]). Previously,
        the no-actuator-names fallback clamped ALL actions to ±TAU_SAFE=0.05,
        which destroyed locomotion task performance.
        """
        action = np.array([0.3, -0.4, 0.2, -0.1])
        clamped = self.pgate.gate(action, physics=None, kappa_snap_logger=None)
        # All values should be within [-1, 1] (global sanity clip)
        for val in clamped:
            self.assertTrue(abs(val) <= 1.0 + 1e-6,
                           f"gate() output {val} exceeds sanity clip range [-1, 1]")
        # Values should be preserved since they're all within [-1, 1]
        np.testing.assert_array_almost_equal(clamped, action)

    def test_gate_with_sentient_physics_selective_clamp(self):
        """gate() with physics containing sentient actuators — selective + global clamp."""
        from core.pg_gate import PGGate

        class MockPhysics:
            class MockModel:
                actuator_names = ["finger_1", "finger_2", "arm_1", "arm_2"]
                nu = 4
            model = MockModel()

        pgate = PGGate()
        action = np.array([0.3, -0.4, 0.01, 0.02])  # Sentient actuators exceed TAU_SAFE
        clamped = pgate.gate(action, physics=MockPhysics(), kappa_snap_logger=None)
        # ALL values should be <= TAU_SAFE due to global physical_clamp
        for val in clamped:
            self.assertTrue(abs(val) <= 0.05 + 1e-6,
                           f"Clamped value {val} exceeds TAU_SAFE")

    def test_zero_action_passes_gate(self):
        """Zero action should pass PG-Gate unchanged."""
        action = np.zeros(5)
        clamped = self.pgate.gate(action, physics=None, kappa_snap_logger=None)
        np.testing.assert_array_almost_equal(clamped, action)

    def test_action_at_exact_threshold(self):
        """Action at exactly ±TAU_SAFE should pass unchanged."""
        action = np.array([0.05, -0.05, 0.05])
        clamped = self.pgate.physical_clamp(action, tau_safe=self.TAU_SAFE)
        np.testing.assert_array_almost_equal(clamped, action)


# ── Scenario b: Merkle chain integrity ───────────────────────────────────

class TestMerkleChainIntegrity(unittest.TestCase):
    """Scenario b: Merkle chain integrity — 10 appends, verify True; tamper, verify False."""

    def setUp(self):
        from core.kappa_snap_logger import MerkleChain
        self.chain = MerkleChain()

    def test_10_appends_verify_true(self):
        """Append 10 entries and verify chain integrity is True."""
        for i in range(10):
            snap_id = self.chain.append(
                eta=i * 0.1,
                decision=f"step_{i}",
                event_type="ACTION_ACCEPT",
                level="L0",
            )
            self.assertIsInstance(snap_id, str)
            self.assertTrue(len(snap_id) > 0)
        self.assertTrue(self.chain.verify(), "Intact chain of 10 entries should verify True")

    def test_tamper_eta_verify_false(self):
        """Tamper eta in an entry — chain should fail verification."""
        for i in range(10):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")

        # Tamper: modify eta of entry 5
        self.chain._chain[5]["eta"] = 999.0
        self.assertFalse(self.chain.verify(),
                        "Tampered chain should fail verification")

    def test_tamper_decision_verify_false(self):
        """Tamper decision string — chain should fail verification."""
        for i in range(10):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")

        # Tamper: modify decision of entry 3
        self.chain._chain[3]["decision"] = "TAMPERED"
        self.assertFalse(self.chain.verify(),
                        "Tampered decision should fail verification")

    def test_tamper_prev_snap_id_verify_false(self):
        """Tamper prev_snap_id linkage — chain should fail verification."""
        for i in range(10):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")

        # Tamper: break prev_snap_id linkage of entry 4
        self.chain._chain[4]["prev_snap_id"] = "broken_link"
        self.assertFalse(self.chain.verify(),
                        "Broken prev_snap_id linkage should fail verification")

    def test_chain_length_matches_appends(self):
        """Chain length should match number of appends."""
        for i in range(10):
            self.chain.append(eta=i * 0.1, decision=f"step_{i}")
        chain = self.chain.get_chain()
        self.assertEqual(len(chain), 10)

    def test_snap_id_hash_rule_matches_manual(self):
        """Verify snap_id hash computation matches the formula manually."""
        prev = "genesis"
        eta = 0.3
        decision = "EXPLOIT"
        hash_input = prev + str(eta) + str(decision)
        expected_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        expected_snap_id = prev + expected_hash

        snap_id = self.chain.append(eta=eta, decision=decision,
                                    event_type="ACTION_ACCEPT", level="L0")
        self.assertEqual(snap_id, expected_snap_id)

    def test_second_entry_prev_matches_first_snap_id(self):
        """Second entry's prev_snap_id should match first entry's snap_id."""
        snap1 = self.chain.append(eta=0.1, decision="step_0")
        snap2 = self.chain.append(eta=0.2, decision="step_1")
        chain = self.chain.get_chain()
        self.assertEqual(chain[1]["prev_snap_id"], snap1)


# ── Scenario c: SafeFuse L3 progression ──────────────────────────────────

class TestSafeFuseL3Progression(unittest.TestCase):
    """Scenario c: SafeFuse L1→L2→L3 progression via consecutive Noether violations."""

    def setUp(self):
        from agent.safe_fuse import SafeFuse
        self.fuse = SafeFuse(consecutive_noether_thresh=3)

    def test_l1_then_l2_then_l3_progression(self):
        """Test fuse level progression: normal→L1→L2→L3."""
        # Step 1: normal (η well below threshold, Noether ok)
        level, action = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
        )
        self.assertEqual(level, "normal")

        # Step 2: L1 soft (η in [δ_K×1.2, δ_K×1.5])
        level, _ = self.fuse.check(
            eta=0.06, delta_K=0.05,  # 0.06/0.05 = 1.2 → L1
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
        )
        self.assertEqual(level, "L1_soft")

        # Step 3: L2 medium (single Noether violation)
        level, _ = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": False, "total": 1},
            psi_anchor_state=None,
        )
        self.assertEqual(level, "L2_medium")

        # Step 4: L3 hard (3× consecutive Noether violations)
        # Need 3 consecutive violations — already have 1 from step 3
        # Step 4: 2nd consecutive violation
        level, _ = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": False, "total": 1},
            psi_anchor_state=None,
        )
        # Still L2 because only 2 consecutive
        self.assertEqual(level, "L2_medium")

        # Step 5: 3rd consecutive violation → L3
        level, _ = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": False, "total": 1},
            psi_anchor_state=None,
        )
        self.assertEqual(level, "L3_hard")

    def test_l3_via_psi_anchor_trigger(self):
        """L3 triggered by ψ-Anchor evolution (psi_trigger=True)."""
        level, _ = self.fuse.check(
            eta=0.03, delta_K=0.05,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state={"evolution_triggered": True},
        )
        self.assertEqual(level, "L3_hard")

    def test_consecutive_count_resets_on_ok(self):
        """Consecutive Noether count resets to 0 when Noether ok."""
        # 2 consecutive violations
        self.fuse.check(eta=0.03, delta_K=0.05,
                        noether_result={"ok": False, "total": 1},
                        psi_anchor_state=None)
        self.fuse.check(eta=0.03, delta_K=0.05,
                        noether_result={"ok": False, "total": 1},
                        psi_anchor_state=None)
        # Then ok → resets
        level, _ = self.fuse.check(eta=0.03, delta_K=0.05,
                                   noether_result={"ok": True, "total": 0},
                                   psi_anchor_state=None)
        self.assertEqual(level, "normal")
        # Next violation should start count from 1 again (L2, not L3)
        level, _ = self.fuse.check(eta=0.03, delta_K=0.05,
                                   noether_result={"ok": False, "total": 1},
                                   psi_anchor_state=None)
        self.assertEqual(level, "L2_medium")

    def test_apply_l2_medium(self):
        """L2 medium clips action to ±0.5."""
        action = np.array([1.0, -0.8, 0.3])
        fused = self.fuse.apply_fuse(action, "L2_medium")
        np.testing.assert_array_almost_equal(fused, np.clip(action, -0.5, 0.5))

    def test_apply_l3_hard_with_safe_action(self):
        """L3 hard uses provided safe_action (PD fallback)."""
        action = np.array([1.0, -0.8, 0.3])
        safe_action = np.array([0.1, -0.2, 0.05])
        fused = self.fuse.apply_fuse(action, "L3_hard", safe_action=safe_action)
        np.testing.assert_array_almost_equal(fused, np.clip(safe_action, -1.0, 1.0))

    def test_apply_l3_hard_without_safe_action(self):
        """L3 hard without safe_action uses emergency ×0.1 fallback."""
        action = np.array([1.0, -0.8, 0.3])
        fused = self.fuse.apply_fuse(action, "L3_hard", safe_action=None)
        np.testing.assert_array_almost_equal(fused, np.clip(action * 0.1, -0.1, 0.1))

    def test_apply_l1_soft_factor(self):
        """L1 soft reduces action by factor 0.8."""
        action = np.array([1.0, -0.5, 0.3])
        fused = self.fuse.apply_fuse(action, "L1_soft")
        np.testing.assert_array_almost_equal(fused, action * 0.8)


# ── Scenario d: CQ compliance rate ────────────────────────────────────────

class TestCQComplianceRate(unittest.TestCase):
    """Scenario d: CQ = min(CQ_noether, CQ_pgate, CQ_sentient) — 100 steps test."""

    def setUp(self):
        from core.cq import ConscienceQuotient
        self.cq = ConscienceQuotient()

    def test_cq_100_steps_min_formula(self):
        """100 steps: 80 noether_ok, 90 pgate_ok, 95 sentient_ok → CQ=0.80.

        CQ = min(80/100, 90/100, 95/100) = min(0.80, 0.90, 0.95) = 0.80
        """
        # Record 100 steps with specified compliance
        for i in range(100):
            noether_ok = i < 80   # 80 steps ok
            pgate_ok = i < 90      # 90 steps ok
            sentient_ok = i < 95   # 95 steps ok
            self.cq.record_step(noether_ok=noether_ok,
                                pgate_ok=pgate_ok,
                                sentient_ok=sentient_ok)

        cq = self.cq.compute_cq()
        self.assertAlmostEqual(cq, 0.80, places=2,
                              msg=f"CQ={cq} should be 0.80 (min of 0.80, 0.90, 0.95)")

    def test_cq_sub_dimensions(self):
        """Verify each CQ sub-dimension separately."""
        for i in range(100):
            noether_ok = i < 80
            pgate_ok = i < 90
            sentient_ok = i < 95
            self.cq.record_step(noether_ok=noether_ok,
                                pgate_ok=pgate_ok,
                                sentient_ok=sentient_ok)

        self.assertAlmostEqual(self.cq.compute_cq_noether(), 0.80, places=2)
        self.assertAlmostEqual(self.cq.compute_cq_pgate(), 0.90, places=2)
        self.assertAlmostEqual(self.cq.compute_cq_sentient(), 0.95, places=2)

    def test_cq_all_compliant_equals_1(self):
        """All steps compliant → CQ = 1.0."""
        for _ in range(50):
            self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        cq = self.cq.compute_cq()
        self.assertAlmostEqual(cq, 1.0, places=2)

    def test_cq_zero_steps_returns_zero(self):
        """No steps recorded → CQ = 0.0 (no data)."""
        cq = self.cq.compute_cq()
        self.assertEqual(cq, 0.0)

    def test_cq_report_includes_all_fields(self):
        """get_report() returns all expected fields."""
        self.cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        report = self.cq.get_report()
        expected_keys = ["cq", "cq_noether", "cq_pgate", "cq_sentient",
                         "total_steps", "noether_ok_steps", "pgate_ok_steps",
                         "sentient_ok_steps"]
        for key in expected_keys:
            self.assertIn(key, report, f"Missing key '{key}' in CQ report")


# ── Scenario e: Friction cone ────────────────────────────────────────────

class TestFrictionCone(unittest.TestCase):
    """Scenario e: ||f_t|| ≤ μ·f_n — f_t=0.5, f_n=0.6, mu=0.8 → violation."""

    def setUp(self):
        from core.noether_check_mj import _friction_cone_check, FRICTION_CONE_MU
        self._friction_cone_check = _friction_cone_check
        self.FRICTION_CONE_MU = FRICTION_CONE_MU

    def test_friction_cone_violation_scenario(self):
        """f_t=0.5, f_n=0.6, mu=0.8 → ||f_t||/f_n=0.833 > mu → violation.

        ||f_t|| = 0.5
        μ·f_n = 0.8 × 0.6 = 0.48
        0.5 > 0.48 → violation detected
        """
        from core.noether_check_mj import _friction_cone_check

        class MockContact:
            def __init__(self, geom1, geom2, force):
                self.geom1 = geom1
                self.geom2 = geom2
                self.force = force
                self.dist = 0.001

        class MockModel:
            geom_bodyid = np.array([1, 2])  # Both dynamic (not ground)
            body_parentid = np.array([0, 0])

        class MockData:
            contact = [MockContact(geom1=0, geom2=1,
                                   force=[0.6, 0.5, 0.0, 0.0, 0.0, 0.0])]
            model = MockModel()

        result = _friction_cone_check(MockData(), mu=0.8)

        # f_n = 0.6, f_t = sqrt(0.5^2) = 0.5
        # ||f_t||/f_n = 0.5/0.6 = 0.833 > 0.8 → violation
        self.assertEqual(result["friction_cone"], 1,
                        "Friction cone violation should be detected when ||f_t|| > mu*f_n")

    def test_friction_cone_no_violation(self):
        """f_t within cone → no violation detected."""
        from core.noether_check_mj import _friction_cone_check

        class MockContact:
            def __init__(self, geom1, geom2, force):
                self.geom1 = geom1
                self.geom2 = geom2
                self.force = force
                self.dist = 0.001

        class MockModel:
            geom_bodyid = np.array([1, 2])
            body_parentid = np.array([0, 0])

        class MockData:
            # f_t=0.3, f_n=0.6, mu=0.8 → 0.3 < 0.48 → no violation
            contact = [MockContact(geom1=0, geom2=1,
                                   force=[0.6, 0.3, 0.0, 0.0, 0.0, 0.0])]
            model = MockModel()

        result = _friction_cone_check(MockData(), mu=0.8)
        self.assertEqual(result["friction_cone"], 0,
                        "No violation when ||f_t|| ≤ mu*f_n")

    def test_friction_cone_ground_contact_excluded(self):
        """Contacts with ground (body_id=0) should be excluded."""
        from core.noether_check_mj import _friction_cone_check

        class MockContact:
            def __init__(self, geom1, geom2, force):
                self.geom1 = geom1
                self.geom2 = geom2
                self.force = force
                self.dist = 0.001

        class MockModel:
            geom_bodyid = np.array([0, 1])  # geom1 is ground (bodyid=0)
            body_parentid = np.array([0, 0])

        class MockData:
            # Even with large f_t, ground contact should be excluded
            contact = [MockContact(geom1=0, geom2=1,
                                   force=[0.6, 0.5, 0.0, 0.0, 0.0, 0.0])]
            model = MockModel()

        result = _friction_cone_check(MockData(), mu=0.8)
        self.assertEqual(result["friction_cone"], 0,
                        "Ground contacts should be excluded from friction cone check")

    def test_friction_cone_same_body_excluded(self):
        """Contacts between same body geoms should be excluded."""
        from core.noether_check_mj import _friction_cone_check

        class MockContact:
            def __init__(self, geom1, geom2, force):
                self.geom1 = geom1
                self.geom2 = geom2
                self.force = force
                self.dist = 0.001

        class MockModel:
            geom_bodyid = np.array([1, 1])  # Same body
            body_parentid = np.array([0, 0])

        class MockData:
            contact = [MockContact(geom1=0, geom2=1,
                                   force=[0.6, 0.5, 0.0, 0.0, 0.0, 0.0])]
            model = MockModel()

        result = _friction_cone_check(MockData(), mu=0.8)
        self.assertEqual(result["friction_cone"], 0,
                        "Same-body contacts should be excluded")

    def test_friction_cone_no_contacts(self):
        """Empty contacts → no violations."""
        from core.noether_check_mj import _friction_cone_check

        class MockData:
            contact = []
            model = None

        result = _friction_cone_check(MockData(), mu=0.8)
        self.assertEqual(result["friction_cone"], 0)
        self.assertEqual(len(result["friction_details"]), 0)


# ── Scenario f: BayesianIntent ────────────────────────────────────────────

class TestBayesianIntentScenario(unittest.TestCase):
    """Scenario f: 20 goal_directed updates → intent clarity increases."""

    def setUp(self):
        from core.bayesian_intent import BayesianIntent
        self.bi = BayesianIntent()

    def test_20_goal_directed_updates_clarity_increases(self):
        """After 20 goal_directed updates, intent clarity should increase."""
        initial_clarity = self.bi.get_intent_clarity()

        # Simulate 20 steps where η consistently decreases (goal_directed)
        eta_values = [0.5 - i * 0.02 for i in range(21)]  # Decreasing η
        for eta in eta_values:
            self.bi.update(eta=eta)

        final_clarity = self.bi.get_intent_clarity()
        self.assertTrue(final_clarity > initial_clarity,
                       f"Clarity should increase: initial={initial_clarity}, "
                       f"final={final_clarity}")

    def test_goal_directed_intent_classification(self):
        """Decreasing η should classify intent as goal_directed."""
        # Start with neutral eta, then decrease
        self.bi.update(eta=0.5)
        self.bi.update(eta=0.3)
        result = self.bi.update(eta=0.1)
        self.assertEqual(result["intent"], "goal_directed")

    def test_divergent_intent_classification(self):
        """Increasing η should classify intent as divergent."""
        self.bi.update(eta=0.1)
        self.bi.update(eta=0.3)
        result = self.bi.update(eta=0.5)
        self.assertEqual(result["intent"], "divergent")

    def test_clarity_range_valid(self):
        """Intent clarity should always be in [0, 1]."""
        for eta in [0.5, 0.3, 0.1, 0.4, 0.6, 0.2]:
            self.bi.update(eta=eta)
        clarity = self.bi.get_intent_clarity()
        self.assertTrue(0.0 <= clarity <= 1.0)

    def test_reset_restores_uniform_prior(self):
        """Reset should restore Beta(1,1) uniform prior for all labels."""
        from core.bayesian_intent import INTENT_LABELS
        self.bi.update(eta=0.1)
        self.bi.update(eta=0.05)
        self.bi.reset()
        for label in INTENT_LABELS:
            posterior = self.bi._intent_posterior[label]
            self.assertAlmostEqual(posterior[0], 1.0, places=2)
            self.assertAlmostEqual(posterior[1], 1.0, places=2)

    def test_bayesian_update_modifies_posterior(self):
        """Bayesian update should modify posterior alpha and beta."""
        from core.bayesian_intent import BayesianIntent
        bi = BayesianIntent()
        # goal_directed update with likelihood=1.0
        bi._bayesian_update("goal_directed", likelihood=1.0)
        posterior = bi._intent_posterior["goal_directed"]
        # alpha = 1+1 = 2, beta = 1+0 = 1
        self.assertEqual(posterior[0], 2.0)
        self.assertEqual(posterior[1], 1.0)


# ── PsiAnchor Sentient Finger Limit ──────────────────────────────────────

class TestPsiAnchorSentientLimit(unittest.TestCase):
    """Additional PsiAnchor sentient finger limit tests."""

    def setUp(self):
        from agent.psi_anchor import PsiAnchor, TAU_SENTIENT_MAX
        from core.goal_eml_mj import GoalEML
        self.TAU_SENTIENT_MAX = TAU_SENTIENT_MAX
        goal = GoalEML(name="test", target_pos=np.zeros(3),
                       max_energy_inject=100.0, delta_K=0.05)
        self.anchor = PsiAnchor(goal)

    def test_sentient_finger_limit_clamps_excessive(self):
        """Actions exceeding TAU_SENTIENT_MAX should be clamped when physics has sentient actuators.

        v0.3.1 fix: When physics=None (no actuator names available), the
        check now returns ok=True and passes action through unchanged.
        Previously, it clamped ALL actions to ±0.05, destroying locomotion.
        This test uses mock physics with sentient actuator names to verify
        clamping still works for manipulation tasks.
        """
        # Use mock physics with sentient actuator names to trigger clamping
        class MockPhysics:
            class MockModel:
                actuator_names = ["finger_1", "finger_2", "arm_1"]
                nu = 3
            model = MockModel()

        action = np.array([0.3, -0.2, 0.1])  # First two exceed 0.05
        result = self.anchor.check_sentient_finger_limit(action, physics=MockPhysics())
        self.assertFalse(result["ok"])
        # Sentient actuators (finger_1, finger_2) should be clamped to TAU_SENTIENT_MAX
        self.assertTrue(abs(result["clamped_action"][0]) <= self.TAU_SENTIENT_MAX + 1e-6)
        self.assertTrue(abs(result["clamped_action"][1]) <= self.TAU_SENTIENT_MAX + 1e-6)
        # Non-sentient actuator (arm_1) should pass through unchanged
        self.assertAlmostEqual(result["clamped_action"][2], 0.1)

    def test_sentient_finger_limit_passes_safe(self):
        """Actions within TAU_SENTIENT_MAX should pass."""
        action = np.array([0.03, -0.04, 0.02])  # All within 0.05
        result = self.anchor.check_sentient_finger_limit(action, physics=None)
        self.assertTrue(result["ok"])
        np.testing.assert_array_almost_equal(result["clamped_action"], action)

    def test_tau_sentient_max_value(self):
        """TAU_SENTIENT_MAX should be exactly 0.05."""
        self.assertEqual(self.TAU_SENTIENT_MAX, 0.05)

    def test_no_physics_checks_all_actuators(self):
        """When physics=None, no sentient actuators can be identified, so action passes through.

        v0.3.1 fix: When no sentient actuators can be identified (no physics
        or no "finger/hand/thumb" actuator names), the check returns ok=True
        and passes the action through unchanged. Previously, it treated ALL
        actuators as sentient and clamped everything to ±0.05, which was
        catastrophic for locomotion tasks.
        """
        action = np.array([0.1, 0.03, -0.08])
        result = self.anchor.check_sentient_finger_limit(action, physics=None)
        # No physics → no sentient identification → ok=True, action unchanged
        self.assertTrue(result["ok"])
        np.testing.assert_array_almost_equal(result["clamped_action"], action)
        # violated_indices should be empty (no clamping occurred)
        self.assertEqual(result["violated_indices"], [])


# ── KappaSnapSchema edge cases ───────────────────────────────────────────

class TestKappaSnapSchemaEdgeCases(unittest.TestCase):
    """Additional KappaSnapSchema validation edge cases."""

    def setUp(self):
        from core.kappa_snap_schema import KappaSnapSchema, EVENT_TYPES
        self.schema = KappaSnapSchema()
        self.EVENT_TYPES = EVENT_TYPES

    def test_all_event_types_have_level(self):
        """Each event type should have an associated audit level."""
        for event_type, definition in self.EVENT_TYPES.items():
            self.assertIn("level", definition,
                          f"Event type {event_type} missing 'level' key")

    def test_all_event_types_have_required_details(self):
        """Each event type should have required_details defined."""
        for event_type, definition in self.EVENT_TYPES.items():
            self.assertIn("required_details", definition,
                          f"Event type {event_type} missing 'required_details'")

    def test_create_event_auto_fills_level(self):
        """create_event should auto-fill level from event type definition."""
        event = self.schema.create_event(
            event_type="REJECT_PG_GATE", eta=0.1, decision="SAFE",
            details={"ast_reason": "sentient", "original_action": [0.1],
                     "clamped_action": [0.05]})
        # REJECT_PG_GATE → L3
        self.assertEqual(event["level"], "L3")

    def test_validate_extra_details_passes(self):
        """Extra detail fields (beyond required) should still pass validation."""
        event = self.schema.create_event(
            event_type="INIT", eta=0.0, decision="start",
            details={"task_name": "test", "goal_delta_K": 0.05,
                     "extra_field": "extra_value"})
        self.assertTrue(self.schema.validate(event))


# ── Integration: Full Decision Loop ──────────────────────────────────────

class TestIntegrationDecisionLoop(unittest.TestCase):
    """Integration test for HybridSB3IDOAgent v0.6.0 decision loop.

    Simulates a full decision loop step:
    η→ψ-Anchor→mode→Noether→SafeFuse→sentient→PG-Gate→Merkle→CQ→ctrl
    """

    def test_full_decision_loop_components(self):
        """Test the full decision loop with all v0.6.0 components."""
        from core.pg_gate import PGGate
        from core.kappa_snap_logger import KappaSnapLogger
        from core.cq import ConscienceQuotient
        from agent.safe_fuse import SafeFuse
        from core.bayesian_intent import BayesianIntent

        # Initialize all components
        pgate = PGGate(tau_safe=0.05)
        logger = KappaSnapLogger()
        cq = ConscienceQuotient()
        fuse = SafeFuse(consecutive_noether_thresh=3)
        bayesian = BayesianIntent()

        # Simulate decision loop step 1: Normal operation
        eta = 0.03
        delta_K = 0.05

        # Noether check (mock: all ok)
        noether_result = {"ok": True, "total": 0, "energy": 0,
                          "torque": 0, "collision": 0, "friction_cone": 0,
                          "friction_details": [], "message": ""}

        # SafeFuse check
        fuse_level, _ = fuse.check(eta=eta, delta_K=delta_K,
                                   noether_result=noether_result,
                                   psi_anchor_state=None)
        self.assertEqual(fuse_level, "normal")

        # Action generation (mock)
        action = np.array([0.03, 0.02, 0.01, 0.04])

        # Apply fuse (normal → unchanged)
        fused_action = fuse.apply_fuse(action, fuse_level)

        # Sentient check (PsiAnchor-style)
        from agent.psi_anchor import PsiAnchor, TAU_SENTIENT_MAX
        from core.goal_eml_mj import GoalEML
        goal = GoalEML(name="test", target_pos=np.zeros(3),
                       max_energy_inject=100.0, delta_K=0.05)
        anchor = PsiAnchor(goal)
        sentient_result = anchor.check_sentient_finger_limit(fused_action)
        self.assertTrue(sentient_result["ok"])

        # PG-Gate
        clamped_action = pgate.gate(fused_action, physics=None,
                                    kappa_snap_logger=logger)
        for val in clamped_action:
            self.assertTrue(abs(val) <= 0.05 + 1e-6)

        # MerkleChain verify
        self.assertTrue(logger.verify_chain())

        # CQ record
        cq.record_step(noether_ok=True, pgate_ok=True, sentient_ok=True)
        self.assertAlmostEqual(cq.compute_cq(), 1.0, places=2)

        # BayesianIntent update
        intent_result = bayesian.update(eta=eta)
        self.assertIn("intent", intent_result)
        self.assertIn("clarity", intent_result)

    def test_decision_loop_with_violation(self):
        """Test decision loop when Noether violation occurs."""
        from core.pg_gate import PGGate
        from core.kappa_snap_logger import KappaSnapLogger
        from core.cq import ConscienceQuotient
        from agent.safe_fuse import SafeFuse

        pgate = PGGate(tau_safe=0.05)
        logger = KappaSnapLogger()
        cq = ConscienceQuotient()
        fuse = SafeFuse()

        # Simulate Noether violation
        noether_result = {"ok": False, "total": 1, "energy": 0,
                          "torque": 1, "collision": 0, "friction_cone": 0,
                          "friction_details": [], "message": "Noether-F"}

        # SafeFuse: single violation → L2
        fuse_level, _ = fuse.check(eta=0.03, delta_K=0.05,
                                   noether_result=noether_result)
        self.assertEqual(fuse_level, "L2_medium")

        # Apply L2 fuse (clip to ±0.5)
        action = np.array([1.0, -0.8, 0.3])
        fused = fuse.apply_fuse(action, "L2_medium")
        np.testing.assert_array_almost_equal(fused, np.clip(action, -0.5, 0.5))

        # PG-Gate on fused action (no physics → global sanity clip only)
        # v0.7.0 fix: Without physics, PGGate skips AST analysis and only
        # applies _global_sanity_clip([-1, 1]). Previously it clamped ALL
        # actions to ±0.05 N·m, destroying locomotion-scale actions.
        clamped = pgate.gate(fused, physics=None, kappa_snap_logger=logger)
        # Values within [-1, 1] should pass through unchanged by sanity clip
        np.testing.assert_array_almost_equal(clamped, fused)

        # CQ: record violation
        cq.record_step(noether_ok=False, pgate_ok=False, sentient_ok=True)
        cq_noether = cq.compute_cq_noether()
        self.assertAlmostEqual(cq_noether, 0.0, places=2)

        # Verify logger recorded events
        self.assertTrue(logger.verify_chain())
        buffer = logger.get_log_buffer()
        self.assertTrue(len(buffer) > 0)

    def test_decision_loop_l4_fatal(self):
        """Test full decision loop with catastrophic L4 violation."""
        from core.pg_gate import PGGate
        from core.kappa_snap_logger import KappaSnapLogger
        from core.cq import ConscienceQuotient
        from agent.safe_fuse import SafeFuse

        pgate = PGGate()
        logger = KappaSnapLogger()
        cq = ConscienceQuotient()
        fuse = SafeFuse()

        # Catastrophic violation (energy+torque+collision all > 0)
        noether_result = {"ok": False, "total": 3, "energy": 1,
                          "torque": 1, "collision": 1, "friction_cone": 0,
                          "friction_details": [], "message": "catastrophic"}

        fuse_level, _ = fuse.check(eta=0.05, delta_K=0.05,
                                   noether_result=noether_result)
        self.assertEqual(fuse_level, "L4_fatal")

        # L4 → zero action
        action = np.array([1.0, -0.5, 0.3])
        fused = fuse.apply_fuse(action, "L4_fatal")
        np.testing.assert_array_equal(fused, np.zeros_like(action))

        # PG-Gate on zero action → should pass unchanged
        clamped = pgate.gate(fused, physics=None, kappa_snap_logger=logger)
        np.testing.assert_array_almost_equal(clamped, np.zeros(3))

        # CQ: all dimensions fail
        cq.record_step(noether_ok=False, pgate_ok=True, sentient_ok=True)
        self.assertAlmostEqual(cq.compute_cq_noether(), 0.0, places=2)


# ── KappaSnapLogger edge cases ───────────────────────────────────────────

class TestKappaSnapLoggerEdgeCases(unittest.TestCase):
    """Additional KappaSnapLogger edge cases."""

    def setUp(self):
        from core.kappa_snap_logger import KappaSnapLogger
        self.logger = KappaSnapLogger()

    def test_multiple_event_types_logged(self):
        """Log events at different audit levels."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT")
        self.logger.log("REJECT_PG_GATE", "L3", 0.15, "SAFE")
        self.logger.log("FINGER_TORQUE_CLAMPED", "L1", 0.08, "CLAMP")
        self.assertTrue(self.logger.verify_chain())

    def test_log_buffer_preserves_order(self):
        """Log buffer should preserve event order."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT")
        buffer = self.logger.get_log_buffer()
        self.assertEqual(buffer[0]["event_type"], "INIT")
        self.assertEqual(buffer[1]["event_type"], "ACTION_ACCEPT")

    def test_snap_id_prev_snap_id_consistency(self):
        """Each event's prev_snap_id should match the previous event's snap_id."""
        self.logger.log("INIT", "L0", 0.0, "start")
        self.logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT")
        buffer = self.logger.get_log_buffer()
        self.assertEqual(buffer[1]["prev_snap_id"], buffer[0]["snap_id"])


# ── compute_merkle_snap_id helper ────────────────────────────────────────

class TestComputeMerkleSnapId(unittest.TestCase):
    """Test compute_merkle_snap_id() helper from kappa_snap_mj v0.3."""

    def test_compute_merkle_snap_id_formula(self):
        """Verify compute_merkle_snap_id matches the expected formula."""
        from core.kappa_snap_mj import compute_merkle_snap_id

        prev_snap_id = "genesis"
        eta = 0.5
        decision = "EXPLOIT"

        # Expected: prev_snap_id + sha256(prev_snap_id + str(eta) + str(decision))[:16]
        hash_input = prev_snap_id + str(eta) + str(decision)
        expected_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        expected_snap_id = prev_snap_id + expected_hash

        result = compute_merkle_snap_id(prev_snap_id, eta, decision)
        self.assertEqual(result, expected_snap_id)

    def test_compute_merkle_snap_id_chained(self):
        """Verify chained snap_id computation."""
        from core.kappa_snap_mj import compute_merkle_snap_id

        snap1 = compute_merkle_snap_id("genesis", 0.5, "EXPLOIT")
        snap2 = compute_merkle_snap_id(snap1, 0.4, "SAFE")

        # Verify snap2 depends on snap1
        hash_input = snap1 + str(0.4) + str("SAFE")
        expected_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        expected_snap_id = snap1 + expected_hash
        self.assertEqual(snap2, expected_snap_id)


if __name__ == '__main__':
    unittest.main()
