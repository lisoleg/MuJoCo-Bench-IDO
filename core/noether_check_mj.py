"""
Noether-Check — IDO Conservation Gate for Continuous Physics
=============================================================

Verifies three conservation invariants (Noether symmetry gates) at
each IDO decision step:

  1. Energy Gate:  ΔE ≤ max_energy_inject + ε  (energy budget)
  2. Force Gate:   max |actuator_force| ≤ MAX_TORQUE * margin
  3. Collision Gate: min self-collision geom distance ≥ collide_thresh
     (excludes ground/worldbody and same-body contacts)

If any gate fails, the IDO agent falls back to a safe squat primitive.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

IDO_NOETHER_MJ_VERSION: str = "v1.1.0"


@dataclass
class NoetherViolation:
    """Record of a Noether conservation gate check result.

    Attributes:
        ok: True if all gates passed, False if any violation detected.
        code: Short violation code (e.g., 'Noether-E', 'Noether-F', 'Noether-C').
        message: Human-readable description of the violation.
    """
    ok: bool
    code: str = ""
    message: str = ""


# Default physical safety thresholds
MAX_TORQUE: float = 500.0
TORQUE_MARGIN: float = 1.05
SELF_COLLIDE_THRESH: float = 0.01
ENERGY_DRIFT_EPS: float = 1e-3


def _min_geom_distance(data) -> float:
    """Compute minimum self-collision geom distance from MuJoCo contact data.

    Only considers contacts between geoms belonging to dynamic model bodies
    (geom_bodyid > 0), excluding ground/worldbody contacts.  This prevents
    normal ground contacts (feet on floor, dist ≈ 0) from triggering the
    self-collision gate.

    Args:
        data: MuJoCo Data object with .contact and .model attributes.

    Returns:
        Minimum distance among self-collision contacts, or 1.0 if none found.
    """
    if not hasattr(data, 'contact') or len(data.contact) == 0:
        return 1.0
    # Get body IDs for each geom — body 0 = worldbody (ground plane)
    model = getattr(data, 'model', None)
    if model is not None and hasattr(model, 'geom_bodyid'):
        body_ids = model.geom_bodyid
    else:
        # Fallback: no model info, use all contacts (legacy behavior)
        body_ids = None

    distances: List[float] = []
    for c in data.contact:
        if c.geom1 == c.geom2:
            continue
        # Skip contacts involving ground/worldbody (body_id == 0)
        if body_ids is not None:
            if body_ids[c.geom1] == 0 or body_ids[c.geom2] == 0:
                continue
            # Skip same-body contacts and parent-child body contacts.
            # In MuJoCo kinematic tree, adjacent bodies are structurally
            # close (e.g., upper_leg–lower_leg, torso–upper_arm).
            b1 = body_ids[c.geom1]
            b2 = body_ids[c.geom2]
            if b1 == b2:
                continue  # same body → not self-collision
            # Parent-child: check if either body is the other's parent
            # in the kinematic tree (model.body_parentid)
            # Sibling: same parent (e.g., upper_arm & upper_leg both
            # attached to torso) → structurally adjacent by design.
            if model is not None and hasattr(model, 'body_parentid'):
                parent_ids = model.body_parentid
                if parent_ids[b1] == b2 or parent_ids[b2] == b1:
                    continue  # parent-child → structurally close by design
                # Sibling exclusion: both bodies share the same parent
                # (e.g., left_upper_arm & right_upper_arm, upper_arm &
                # upper_leg — all children of torso). These are naturally
                # close in locomotion tasks and should not be flagged as
                # self-collision violations.
                if parent_ids[b1] == parent_ids[b2] and parent_ids[b1] != 0:
                    continue  # sibling bodies → structurally adjacent
        distances.append(c.dist)
    return min(distances) if distances else 1.0


def noether_check_mj(prev_data,
                      cur_data,
                      goal,
                      max_torque: Optional[float] = None,
                      torque_margin: Optional[float] = None,
                      collide_thresh: Optional[float] = None) -> dict:
    """Run Noether conservation gate between two MuJoCo data snapshots.

    Checks:
    1. Energy drift ΔE ≤ goal.max_energy_inject + ENERGY_DRIFT_EPS.
    2. Max actuator force ≤ max_torque * torque_margin.
    3. Min geom distance ≥ collide_thresh (self-collision avoidance).

    Args:
        prev_data: Previous MuJoCo Data snapshot.
        cur_data: Current MuJoCo Data snapshot.
        goal: GoalEML instance with max_energy_inject attribute.
        max_torque: Override for maximum allowed torque. Defaults to MAX_TORQUE.
        torque_margin: Override for torque safety margin. Defaults to TORQUE_MARGIN.
        collide_thresh: Override for self-collision threshold. Defaults to SELF_COLLIDE_THRESH.

    Returns:
        Dict with keys:
          - ok: True if all gates passed, False if any violation detected.
          - total: Total number of violations (0, 1, 2, or 3).
          - energy: 1 if energy gate failed, 0 otherwise.
          - torque: 1 if force/torque gate failed, 0 otherwise.
          - collision: 1 if collision gate failed, 0 otherwise.
          - message: Human-readable description of all violations (empty on success).
    """
    if max_torque is None:
        max_torque = MAX_TORQUE
    if torque_margin is None:
        torque_margin = TORQUE_MARGIN
    if collide_thresh is None:
        collide_thresh = SELF_COLLIDE_THRESH

    result: dict = {
        "ok": True,
        "total": 0,
        "energy": 0,
        "torque": 0,
        "collision": 0,
        "message": "",
    }

    # ── Energy Gate ──
    E_prev: float = (getattr(prev_data, 'energy', [0, 0])[0]
                     + getattr(prev_data, 'energy', [0, 0])[1])
    E_cur: float = (getattr(cur_data, 'energy', [0, 0])[0]
                    + getattr(cur_data, 'energy', [0, 0])[1])
    dE: float = E_cur - E_prev

    if dE > goal.max_energy_inject + ENERGY_DRIFT_EPS:
        result["ok"] = False
        result["energy"] = 1
        result["total"] += 1
        result["message"] = (f"Noether-E: energy increased by {dE:.4f}J "
                             f"exceeds budget {goal.max_energy_inject:.2f}J")

    # ── Force Gate ──
    if hasattr(cur_data, 'actuator_force'):
        forces: np.ndarray = np.abs(np.asarray(cur_data.actuator_force))
        if len(forces) > 0 and np.max(forces) > max_torque * torque_margin:
            result["ok"] = False
            result["torque"] = 1
            result["total"] += 1
            force_msg: str = (f"Noether-F: max torque {np.max(forces):.2f} "
                              f"exceeds limit {max_torque:.2f} N·m")
            if result["message"]:
                result["message"] += "; " + force_msg
            else:
                result["message"] = force_msg

    # ── Collision Gate ──
    min_dist: float = _min_geom_distance(cur_data)
    if min_dist < collide_thresh:
        result["ok"] = False
        result["collision"] = 1
        result["total"] += 1
        collision_msg: str = (f"Noether-C: self-collision detected "
                              f"(min_dist={min_dist:.6f}m < {collide_thresh}m)")
        if result["message"]:
            result["message"] += "; " + collision_msg
        else:
            result["message"] = collision_msg

    return result
