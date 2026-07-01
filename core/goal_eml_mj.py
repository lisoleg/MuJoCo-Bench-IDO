"""
Goal-EML — IDO Goal Co-Set Invariants for MuJoCo Tasks
========================================================

Defines the GoalEML dataclass that encodes IDO task invariants
(position, orientation, energy budget, tolerance) and factory
functions for four dm_control benchmark tasks:

  - humanoid-stand: upright standing with ground contact
  - hopper-stand:   standing balance with ground contact
  - walker-run:     forward locomotion without falling
  - reacher-easy:   simple 2-DOF reaching

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

IDO_GOAL_EML_MJ_VERSION: str = "v1.0.0"


@dataclass
class GoalEML:
    """IDO Goal Co-Set Invariants for a MuJoCo physics task.

    Encodes the manifold constraints that the IDO agent must satisfy:
    which invariants to preserve, the target position, the κ-Snap
    delta_K tolerance, the energy injection budget, and position/
    orientation tolerances.

    Attributes:
        name: Task name string (e.g., 'humanoid_reach').
        invariants: List of invariant names the agent must preserve.
        target_pos: 3D target position for end-effector reach.
        delta_K: κ-Snap residual threshold (κ_thresh in agent).
        max_energy_inject: Maximum allowed energy injection (J).
        pos_tol: Position tolerance for goal achievement (m).
        ori_tol: Orientation tolerance for goal achievement (rad).
    """
    name: str
    invariants: List[str] = field(default_factory=list)
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_K: float = 0.05
    max_energy_inject: float = 500.0
    pos_tol: float = 0.02
    ori_tol: float = 0.15


def make_humanoid_stand_eml(physics,
                             delta_K: float = 0.05) -> GoalEML:
    """Factory for humanoid stand GoalEML.

    Creates a GoalEML for the humanoid-stand task where the torso must
    remain upright with feet on the ground.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for humanoid-stand task.
    """
    return GoalEML(
        name='humanoid_stand',
        invariants=['torso_upright', 'feet_on_ground', 'no_self_collide'],
        target_pos=np.array([0.0, 0.0, 1.4]),
        delta_K=delta_K,
        max_energy_inject=500.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_hopper_stand_eml(physics,
                           delta_K: float = 0.03) -> GoalEML:
    """Factory for hopper stand GoalEML.

    Creates a GoalEML for the hopper-stand task where the torso must
    remain upright with the foot on the ground.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for hopper-stand task.
    """
    return GoalEML(
        name='hopper_stand',
        invariants=['torso_upright', 'foot_on_ground', 'no_self_collide'],
        target_pos=np.array([0.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=200.0,
        pos_tol=0.05,
        ori_tol=0.20,
    )


def make_walker_run_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for walker run GoalEML.

    Creates a GoalEML for the walker-run task where the center-of-mass
    must advance forward without falling.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for walker-run task.
    """
    return GoalEML(
        name='walker_run',
        invariants=['com_x_advancing', 'not_fallen', 'no_self_collide'],
        target_pos=np.array([10.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=600.0,
        pos_tol=0.10,
        ori_tol=0.25,
    )


def make_reacher_easy_eml(physics,
                           delta_K: float = 0.02) -> GoalEML:
    """Factory for reacher easy GoalEML.

    Creates a GoalEML for the reacher-easy task where the end-effector
    must reach a small target offset.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for reacher-easy task.
    """
    return GoalEML(
        name='reacher_easy',
        invariants=['ee_at_target'],
        target_pos=np.array([0.1, 0.1, 0.0]),
        delta_K=delta_K,
        max_energy_inject=50.0,
        pos_tol=0.01,
        ori_tol=0.0,
    )
