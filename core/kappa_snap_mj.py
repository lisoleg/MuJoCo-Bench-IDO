"""
κ-Snap (GaussEx Residual) — Continuous-state IDO version
=========================================================

Computes the GaussEx residual η = Σ w_i · f_i(z_i, Goal-EML)^2 that
measures how far the current observation z_i deviates from the IDO
goal manifold defined by GoalEML invariants.

Components:
  - Position error:   ||ee_pos − target_pos||^2
  - Orientation error: tilt angle from upright z-axis ^2
  - Energy excess:    max(0, E_total − max_energy_inject)^2
  - Velocity error:   ||ee_vel[:3]||^2

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from typing import Optional

IDO_KAPPA_SNAP_MJ_VERSION: str = "v1.0.0"


def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to the body's local z-axis vector.

    Uses the rotation matrix formula for the third column (z-axis).

    Args:
        quat: Quaternion array of length ≥4 in [w, x, y, z] convention.
              If None or shorter than 4, returns default [0, 0, 1].

    Returns:
        3-vector representing the body's z-axis in world coordinates.
    """
    if quat is None or len(quat) < 4:
        return np.array([0.0, 0.0, 1.0])
    qw: float = quat[0]
    qx: float = quat[1]
    qy: float = quat[2]
    qz: float = quat[3]
    zx: float = 2.0 * (qx * qz + qw * qy)
    zy: float = 2.0 * (qy * qz - qw * qx)
    zz: float = qw * qw - qx * qx - qy * qy + qz * qz
    return np.array([zx, zy, zz])


def gauss_ex_residual(z_i: dict,
                      goal,
                      w_pos: float = 1.0,
                      w_ori: float = 0.3,
                      w_eng: float = 0.01,
                      w_vel: float = 0.05) -> float:
    """Compute GaussEx residual η for continuous-state IDO κ-Snap.

    η = w_pos * pos_err^2 + w_ori * tilt_err^2
        + w_eng * energy_excess^2 + w_vel * vel_mag^2

    Args:
        z_i: EML observation dict with keys:
             'ee_pos' (3-vector), 'qpos' (nq-vector), 'E_total' (float),
             'ee_vel' (6-vector or 3-vector).
        goal: GoalEML instance with target_pos, max_energy_inject, etc.
        w_pos: Weight for position error component.
        w_ori: Weight for orientation (tilt) error component.
        w_eng: Weight for energy excess component.
        w_vel: Weight for velocity magnitude component.

    Returns:
        Scalar residual η (float). Lower η means closer to goal manifold.
    """
    # Position error
    ee: np.ndarray = z_i.get('ee_pos', np.zeros(3))
    target: np.ndarray = goal.target_pos
    if ee is None or target is None:
        pos_err: float = 0.0
    else:
        pos_err = float(np.linalg.norm(np.asarray(ee) - np.asarray(target)))

    # Orientation (tilt) error via quaternion → z-axis
    quat: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = z_i.get('qpos', None)
    if qpos is not None and len(qpos) >= 4:
        quat = qpos[3:7] if len(qpos) >= 7 else qpos[:4]
    z_body: np.ndarray = _quat_to_z_axis(quat)
    tilt_err: float = float(np.arccos(
        np.clip(np.dot(z_body, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))

    # Energy excess beyond budget
    E: float = float(z_i.get('E_total', 0.0))
    energy_excess: float = max(0.0, E - goal.max_energy_inject)

    # End-effector velocity magnitude
    ee_vel: np.ndarray = z_i.get('ee_vel', np.zeros(6))
    vel_mag: float = float(np.linalg.norm(ee_vel[:3])) if ee_vel is not None else 0.0

    # Weighted sum of squared residuals
    eta: float = (w_pos * pos_err ** 2
                  + w_ori * tilt_err ** 2
                  + w_eng * energy_excess ** 2
                  + w_vel * vel_mag ** 2)

    return float(eta)
