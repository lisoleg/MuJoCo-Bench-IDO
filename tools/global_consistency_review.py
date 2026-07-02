#!/usr/bin/env python
"""Global Consistency Review for MuJoCo-Bench-IDO v0.6.0."""

import sys
import os
import inspect
import numpy as np

# Add project root to sys.path so core/agent/envs can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Global Consistency Review: MuJoCo-Bench-IDO v0.6.0 ===")

# 1. Module import test
modules_ok = []
modules_fail = []
test_modules = [
    ("core.kappa_snap_schema", "KappaSnapSchema"),
    ("core.kappa_snap_logger", "KappaSnapLogger"),
    ("core.pg_gate", "PGGate"),
    ("core.cq", "ConscienceQuotient"),
    ("core.bayesian_intent", "BayesianIntent"),
    ("core.kappa_snap_mj", "compute_merkle_snap_id"),
    ("core.noether_check_mj", "_friction_cone_check"),
    ("agent.psi_anchor", "TAU_SENTIENT_MAX"),
    ("agent.safe_fuse", "SafeFuse"),
    ("agent.hybrid_sb3_ido_agent", "HybridSB3IDOAgent"),
    ("core", "exports"),
    ("agent", "exports"),
    ("envs", "PinchLeafEnv"),
]
for mod_path, expected in test_modules:
    try:
        mod = __import__(mod_path, fromlist=[expected])
        modules_ok.append(mod_path)
    except Exception as e:
        modules_fail.append((mod_path, str(e)))
print(f"[1] Module Import: {len(modules_ok)}/13 OK")
for m in modules_ok:
    print(f"  OK: {m}")
for m, e in modules_fail:
    print(f"  FAIL: {m} -> {e}")
print()

# 2. Functional tests
from core.kappa_snap_schema import KappaSnapSchema
from core.kappa_snap_logger import KappaSnapLogger, MerkleChain
from core.pg_gate import PGGate
from core.cq import ConscienceQuotient
from core.bayesian_intent import BayesianIntent
from core.kappa_snap_mj import compute_merkle_snap_id
from core.noether_check_mj import _friction_cone_check
from agent.psi_anchor import TAU_SENTIENT_MAX
from agent.safe_fuse import SafeFuse

tp = 0
tt = 0
def test(name, cond):
    global tp, tt
    tt += 1
    if cond:
        tp += 1
        print(f"  OK: {name}")
    else:
        print(f"  FAIL: {name}")

print("[2] Functional Tests:")

# KappaSnapSchema
schema = KappaSnapSchema()
evt = schema.create_event("INIT", "L0", eta=0.0, decision="test",
                          details={"task_name": "test_task", "goal_delta_K": 0.1})
test("INIT event creates", evt["event_type"] == "INIT")
test("INIT event validates", schema.validate(evt))
tt += 1
try:
    schema.create_event("INVALID_TYPE", "L0", eta=0.0, decision="test", details={})
    print("  FAIL: Invalid event type should raise ValueError")
except ValueError:
    tp += 1
    print("  OK: Invalid event type raises ValueError")

# MerkleChain
mc = MerkleChain()
sid1 = mc.append(0.1, "accept", "ACTION_ACCEPT", "L5")
sid2 = mc.append(0.2, "reject", "REJECT_NOETHER", "L1")
test("MerkleChain append", sid1 is not None and sid2 is not None)
test("MerkleChain verify", mc.verify())
test("MerkleChain hash rule", compute_merkle_snap_id("", 0.1, "accept") is not None)

# KappaSnapLogger
logger = KappaSnapLogger()
evt = logger.log("INIT", "L0", 0.0, "test", {}, {"agent_id": "test"})
test("KappaSnapLogger log", evt is not None)
test("KappaSnapLogger verify_chain", logger.verify_chain())

# PGGate
pg = PGGate()
clamped = pg.physical_clamp(np.array([0.1, -0.2, 0.3]), 0.05)
expected_clamp = np.clip([0.1, -0.2, 0.3], -0.05, 0.05)
test("PGGate.physical_clamp", np.allclose(clamped, expected_clamp))

# CQ
cq = ConscienceQuotient()
cq.record_step(True, True, False)
cq.record_step(True, True, True)
test("CQ compute_cq", cq.compute_cq() == min(1.0, 1.0, 0.5))

# SafeFuse - correct test with proper keys matching SafeFuse.check()
sf = SafeFuse()
result_normal = sf.check(0.01, 0.1,
    {"ok": True, "total": 0, "energy": 0, "torque": 0, "collision": 0},
    {"mode": "normal"})
test("SafeFuse normal level", result_normal[0] == "normal")

result_l4 = sf.check(10.0, 0.1,
    {"ok": False, "total": 3, "energy": 1, "torque": 1, "collision": 1},
    {"mode": "normal"})
test("SafeFuse L4 detected", result_l4[0] == "L4_fatal")

result_l1 = sf.check(0.15, 0.1,
    {"ok": True, "total": 0, "energy": 0, "torque": 0, "collision": 0},
    {"mode": "normal"})
test("SafeFuse L1 detected", result_l1[0] == "L1_soft")

# SafeFuse apply_fuse
action = np.array([1.0, -1.0, 0.5])
fused_l1 = sf.apply_fuse(action, "L1_soft")
test("SafeFuse.apply_fuse L1_soft", np.allclose(fused_l1, action * 0.8))
fused_l4 = sf.apply_fuse(action, "L4_fatal")
test("SafeFuse.apply_fuse L4_fatal", np.allclose(fused_l4, np.zeros_like(action)))

# PsiAnchor
test("TAU_SENTIENT_MAX == 0.05", TAU_SENTIENT_MAX == 0.05)

# BayesianIntent
bi = BayesianIntent()
result = bi.update(0.5, 0.1, 0.01)
test("BayesianIntent update", result is not None)

# _friction_cone_check
test("_friction_cone_check callable", callable(_friction_cone_check))

print(f"\n[2] Functional Tests: {tp}/{tt} passed\n")

# 3. Cross-file consistency
print("[3] Cross-File Consistency:")
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent
import agent.hybrid_sb3_ido_agent as hmod
with open(hmod.__file__, encoding="utf-8") as f:
    h_src = f.read()
checks = [
    ("PGGate import", "PGGate" in h_src),
    ("KappaSnapLogger import", "KappaSnapLogger" in h_src),
    ("ConscienceQuotient import", "ConscienceQuotient" in h_src),
    ("SafeFuse import", "SafeFuse" in h_src),
    ("_pg_gate attr", "_pg_gate" in h_src),
    ("_cq attr", "_cq" in h_src),
    ("_safe_fuse attr", "_safe_fuse" in h_src),
    ("_logger attr", "_logger" in h_src),
    ("get_cq_report method", "get_cq_report" in h_src),
    ("get_merkle_chain method", "get_merkle_chain" in h_src),
    ("verify_merkle_chain method", "verify_merkle_chain" in h_src),
]
cc_ok = all(c for _, c in checks)
for name, cond in checks:
    status = "OK" if cond else "FAIL"
    print(f"  {status}: {name}")
print()

# 4. Interface contracts
print("[4] Interface Contract Compliance:")
pg_sig = inspect.signature(PGGate.gate)
pg_params = list(pg_sig.parameters.keys())
pg_ok = pg_params == ["self", "action", "physics", "kappa_snap_logger"]
print(f"  PGGate.gate params: {pg_params} {'OK' if pg_ok else 'FAIL'}")

sf_sig = inspect.signature(SafeFuse.check)
sf_params = list(sf_sig.parameters.keys())
sf_ok = sf_params == ["self", "eta", "delta_K", "noether_result", "psi_anchor_state"]
print(f"  SafeFuse.check params: {sf_params} {'OK' if sf_ok else 'FAIL'}")

cq_sig = inspect.signature(ConscienceQuotient.record_step)
cq_params = list(cq_sig.parameters.keys())
cq_ok = cq_params == ["self", "noether_ok", "pgate_ok", "sentient_ok"]
print(f"  CQ.record_step params: {cq_params} {'OK' if cq_ok else 'FAIL'}")

bi_sig = inspect.signature(BayesianIntent.update)
bi_params = list(bi_sig.parameters.keys())
bi_ok = bi_params == ["self", "observation", "action", "eta"]
print(f"  BayesianIntent.update params: {bi_params} {'OK' if bi_ok else 'FAIL'}")

contracts_ok = pg_ok and sf_ok and cq_ok and bi_ok
print()

# 5. Verdict
all_ok = (len(modules_fail) == 0 and tp == tt and cc_ok and contracts_ok)
print("=== FINAL VERDICT ===")
if all_ok:
    print("IS_PASS: YES")
    print("All module imports, functional tests, cross-file consistency, and interface contracts passed.")
else:
    print("IS_PASS: NO")
    if len(modules_fail) > 0:
        print(f"  Import failures: {modules_fail}")
    if tp < tt:
        print(f"  Functional failures: {tt - tp}")
    if not cc_ok:
        print("  Consistency failures")
    if not contracts_ok:
        print("  Contract failures")
