"""
Noether-Check — IDO Conservation Gate for Continuous Physics
=============================================================

Verifies three conservation invariants (Noether symmetry gates) at
each IDO decision step:

  1. Energy Gate:  ΔE ≤ max_energy_inject + ε  (energy budget)
  2. Force Gate:   max |actuator_force| ≤ MAX_TORQUE * margin
  3. Collision Gate: min geom distance ≥ SELF_COLLIDE_THRESH

If any gate fails, the IDO agent falls back to a safe squat primitive.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

IDO_NOETHER_MJ_VERSION: str = "v1.0.0"


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
SELF_COLLIDE_THRESH: float = 0.005
ENERGY_DRIFT_EPS: float = 1e-3


def _min_geom_distance(data) -> float:
    """Compute minimum pairwise geom distance from MuJoCo contact data.

    Args:
        data: MuJoCo Data object with .contact attribute.

    Returns:
        Minimum distance among all non-self contacts, or 1.0 if no contacts.
    """
    if not hasattr(data, 'contact') or len(data.contact) == 0:
        return 1.0
    distances: List[float] = []
    for c in data.contact:
        if c.geom1 != c.geom2:
            distances.append(c.dist)
    return min(distances) if distances else 1.0


def noether_check_mj(prev_data,
                      cur_data,
                      goal,
                      max_torque: Optional[float] = None,
                      torque_margin: Optional[float] = None,
                      collide_thresh: Optional[float] = None) -> Tuple[bool, str]:
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
        Tuple of (ok: bool, message: str).
        ok=True means all conservation invariants hold.
        message is empty on success, or describes the violation on failure.
    """
    if max_torque is None:
        max_torque = MAX_TORQUE
    if torque_margin is None:
        torque_margin = TORQUE_MARGIN
    if collide_thresh is None:
        collide_thresh = SELF_COLLIDE_THRESH

    # ── Energy Gate ──
    E_prev: float = (getattr(prev_data, 'energy', [0, 0])[0]
                     + getattr(prev_data, 'energy', [0, 0])[1])
    E_cur: float = (getattr(cur_data, 'energy', [0, 0])[0]
                    + getattr(cur_data, 'energy', [0, 0])[1])
    dE: float = E_cur - E_prev

    if dE > goal.max_energy_inject + ENERGY_DRIFT_EPS:
        return False, (f"Noether-E: energy increased by {dE:.4f}J "
                       f"exceeds budget {goal.max_energy_inject:.2f}J")

    # ── Force Gate ──
    if hasattr(cur_data, 'actuator_force'):
        forces: np.ndarray = np.abs(np.asarray(cur_data.actuator_force))
        if len(forces) > 0 and np.max(forces) > max_torque * torque_margin:
            return False, (f"Noether-F: max torque {np.max(forces):.2f} "
                           f"exceeds limit {max_torque:.2f} N·m")

    # ── Collision Gate ──
    min_dist: float = _min_geom_distance(cur_data)
    if min_dist < collide_thresh:
        return False, (f"Noether-C: self-collision detected "
                       f"(min_dist={min_dist:.6f}m < {collide_thresh}m)")

    return True, ""
