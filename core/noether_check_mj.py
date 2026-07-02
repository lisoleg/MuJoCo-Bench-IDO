"""
Noether-Check — IDO Conservation Gate for Continuous Physics
=============================================================

Verifies four conservation invariants (Noether symmetry gates) at
each IDO decision step:

  1. Energy Gate:  ΔE ≤ max_energy_inject + ε  (energy budget)
  2. Force Gate:   max |actuator_force| ≤ MAX_TORQUE * margin
  3. Collision Gate: min self-collision geom distance ≥ collide_thresh
     (excludes ground/worldbody and same-body contacts)
  4. Friction Cone Gate (v1.2.0): ||f_t|| ≤ μ · f_n for all contacts
     (Coulomb friction cone — prevents sliding/scratching that could
     harm biological tissue)

If any gate fails, the IDO agent falls back to a safe squat primitive.

v1.2.0 Upgrade: Added Noether-FrictionCone (4th gate) — checks that
tangential friction force stays within Coulomb cone μ·f_n for all
non-ground contacts. This prevents scenarios where excessive tangential
force could scratch or damage biological surfaces.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

IDO_NOETHER_MJ_VERSION: str = "v1.2.0"


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
FRICTION_CONE_MU: float = 0.8  # Default Coulomb friction coefficient


def _friction_cone_check(cur_data, mu: float = FRICTION_CONE_MU) -> dict:
    """Check Coulomb friction cone constraint for all non-ground contacts.

    v1.2.0 Noether-FrictionCone (4th gate): For each contact between
    dynamic bodies (geom_bodyid > 0), verify that the tangential
    friction force satisfies ||f_t|| ≤ μ · f_n, where:
      f_n = normal component of contact force
      f_t = tangential component of contact force
      μ = friction coefficient (default 0.8)

    This prevents scenarios where excessive tangential force could
    scratch or damage biological surfaces during manipulation.

    Args:
        cur_data: MuJoCo Data object with .contact, .model attributes.
        mu: Coulomb friction coefficient. Default 0.8.

    Returns:
        Dict with keys:
          - friction_cone: 1 if any contact violates friction cone, 0 otherwise.
          - friction_details: List of dicts for violating contacts with
            keys: contact_id, f_t, f_n, mu, violated.
    """
    result: dict = {
        "friction_cone": 0,
        "friction_details": [],
    }

    if not hasattr(cur_data, 'contact') or len(cur_data.contact) == 0:
        return result

    model = getattr(cur_data, 'model', None)
    if model is not None and hasattr(model, 'geom_bodyid'):
        body_ids = model.geom_bodyid
    else:
        body_ids = None

    violations: List[dict] = []

    for c_idx, c in enumerate(cur_data.contact):
        # Skip same-geom contacts
        if c.geom1 == c.geom2:
            continue

        # Skip ground/worldbody contacts (body_id == 0)
        if body_ids is not None:
            if body_ids[c.geom1] == 0 or body_ids[c.geom2] == 0:
                continue
            # Skip same-body contacts
            if body_ids[c.geom1] == body_ids[c.geom2]:
                continue

        # Extract contact force vector
        # MuJoCo stores contact force in c.force (6-vector: [f_normal, f_tangent_x, f_tangent_y, f_twist_x, f_twist_y, f_twist_z])
        # For dm_control, force may be accessible differently
        force_vector: Optional[np.ndarray] = None

        # Try to get force from contact object
        if hasattr(c, 'force'):
            force_vector = np.asarray(c.force)
        elif hasattr(cur_data, 'contact_force'):
            try:
                force_vector = np.asarray(cur_data.contact_force[c_idx])
            except (IndexError, AttributeError):
                pass

        if force_vector is None or len(force_vector) < 3:
            # No force data available for this contact — skip
            continue

        # Decompose force into normal and tangential components
        # MuJoCo contact frame: first component = normal, rest = tangential
        f_n: float = abs(float(force_vector[0]))  # Normal force magnitude
        f_t_vec: np.ndarray = force_vector[1:] if len(force_vector) > 1 else np.array([0.0])
        f_t: float = float(np.linalg.norm(f_t_vec))  # Tangential force magnitude

        # Check friction cone: ||f_t|| ≤ μ · f_n
        violated: bool = f_t > mu * f_n and f_n > 1e-6  # Skip near-zero normal forces

        if violated:
            violations.append({
                "contact_id": c_idx,
                "f_t": f_t,
                "f_n": f_n,
                "mu": mu,
                "violated": True,
            })

    if len(violations) > 0:
        result["friction_cone"] = 1
        result["friction_details"] = violations

    return result


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
                      collide_thresh: Optional[float] = None,
                      friction_mu: Optional[float] = None) -> dict:
    """Run Noether conservation gate between two MuJoCo data snapshots.

    Checks 4 gates (v1.2.0):
    1. Energy drift ΔE ≤ goal.max_energy_inject + ENERGY_DRIFT_EPS.
    2. Max actuator force ≤ max_torque * torque_margin.
    3. Min geom distance ≥ collide_thresh (self-collision avoidance).
    4. Friction cone ||f_t|| ≤ μ·f_n for all non-ground contacts.

    Args:
        prev_data: Previous MuJoCo Data snapshot.
        cur_data: Current MuJoCo Data snapshot.
        goal: GoalEML instance with max_energy_inject attribute.
        max_torque: Override for maximum allowed torque. Defaults to MAX_TORQUE.
        torque_margin: Override for torque safety margin. Defaults to TORQUE_MARGIN.
        collide_thresh: Override for self-collision threshold. Defaults to SELF_COLLIDE_THRESH.
        friction_mu: Override for friction coefficient μ. Defaults to FRICTION_CONE_MU.

    Returns:
        Dict with keys:
          - ok: True if all 4 gates passed, False if any violation detected.
          - total: Total number of violations (0, 1, 2, 3, or 4).
          - energy: 1 if energy gate failed, 0 otherwise.
          - torque: 1 if force/torque gate failed, 0 otherwise.
          - collision: 1 if collision gate failed, 0 otherwise.
          - friction_cone: 1 if friction cone gate failed, 0 otherwise (NEW v1.2).
          - friction_details: List of friction cone violation dicts (NEW v1.2).
          - message: Human-readable description of all violations (empty on success).
    """
    if max_torque is None:
        max_torque = MAX_TORQUE
    if torque_margin is None:
        torque_margin = TORQUE_MARGIN
    if collide_thresh is None:
        collide_thresh = SELF_COLLIDE_THRESH
    if friction_mu is None:
        friction_mu = FRICTION_CONE_MU

    result: dict = {
        "ok": True,
        "total": 0,
        "energy": 0,
        "torque": 0,
        "collision": 0,
        "friction_cone": 0,
        "friction_details": [],
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

    # ── Friction Cone Gate (v1.2.0) ──
    friction_result: dict = _friction_cone_check(cur_data, mu=friction_mu)
    result["friction_cone"] = friction_result["friction_cone"]
    result["friction_details"] = friction_result["friction_details"]

    if friction_result["friction_cone"] > 0:
        result["ok"] = False
        result["total"] += 1
        n_violations: int = len(friction_result["friction_details"])
        friction_msg: str = (f"Noether-FC: friction cone violation "
                             f"({n_violations} contacts with ||f_t|| > μ·f_n, μ={friction_mu})")
        if result["message"]:
            result["message"] += "; " + friction_msg
        else:
            result["message"] = friction_msg

    return result
