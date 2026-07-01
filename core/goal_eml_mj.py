"""
Goal-EML — IDO Goal Co-Set Invariants for MuJoCo Tasks
========================================================

Defines the GoalEML dataclass that encodes IDO task invariants
(position, orientation, energy budget, tolerance) and factory
functions for dm_control benchmark tasks:

  - humanoid-stand / humanoid-walk / humanoid-run
  - walker-stand / walker-walk / walker-run
  - hopper-stand / hopper-hop
  - cheetah-run
  - cartpole-balance / cartpole-swingup / cartpole-balance_sparse / cartpole-swingup_sparse
  - reacher-easy / reacher-hard
  - fish-swim
  - manipulator-bring_ball
  - acrobot-swingup
  - pendulum-swingup
  - finger-spin / finger-turn_easy / finger-turn_hard
  - ball_in_cup-catch
  - swimmer-swim6 / swimmer-swim15

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

IDO_GOAL_EML_MJ_VERSION: str = "v1.1.0"


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
    must reach a small target offset (2D task in XY plane).

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for reacher-easy task.
    """
    return GoalEML(
        name='reacher_easy',
        invariants=['ee_at_target'],
        target_pos=np.array([0.0, 0.0]),  # 2D target (reacher operates in XY plane)
        delta_K=delta_K,
        max_energy_inject=50.0,
        pos_tol=0.01,
        ori_tol=0.0,
    )


def make_generic_eml(task_name: str,
                      physics,
                      delta_K: float = 0.05) -> GoalEML:
    """Generic GoalEML factory for any dm_control task.

    Derives sensible invariants and target position from the task name.
    Specialized tasks (e.g. humanoid-stand) should use their dedicated
    factories for finer control; this factory acts as a safe fallback.

    Args:
        task_name: Full task identifier such as 'humanoid-walk'.
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance with task-appropriate defaults.
    """
    t = task_name.lower()

    if 'reacher' in t:
        return make_reacher_easy_eml(physics, delta_K=delta_K)

    if 'cartpole' in t or 'pendulum' in t or 'acrobot' in t:
        return GoalEML(
            name=t,
            invariants=['pole_upright', 'not_fallen'],
            target_pos=np.array([0.0, 0.0, 1.0]),
            delta_K=delta_K,
            max_energy_inject=100.0,
            pos_tol=0.05,
            ori_tol=0.15,
        )

    if 'swimmer' in t or 'fish' in t:
        return GoalEML(
            name=t,
            invariants=['forward_velocity', 'efficient_motion'],
            target_pos=np.array([10.0, 0.0, 0.0]),
            delta_K=delta_K,
            max_energy_inject=300.0,
            pos_tol=0.20,
            ori_tol=0.30,
        )

    if 'finger' in t or 'ball_in_cup' in t or 'manipulator' in t:
        return GoalEML(
            name=t,
            invariants=['object_at_target', 'minimal_energy'],
            target_pos=np.array([0.0, 0.0, 0.5]),
            delta_K=delta_K,
            max_energy_inject=200.0,
            pos_tol=0.05,
            ori_tol=0.20,
        )

    if 'stand' in t:
        return GoalEML(
            name=t,
            invariants=['torso_upright', 'feet_on_ground', 'no_self_collide'],
            target_pos=np.array([0.0, 0.0, 1.3]),
            delta_K=delta_K,
            max_energy_inject=300.0,
            pos_tol=0.05,
            ori_tol=0.15,
        )

    if 'walk' in t or 'run' in t or 'hop' in t:
        return GoalEML(
            name=t,
            invariants=['com_x_advancing', 'not_fallen', 'no_self_collide'],
            target_pos=np.array([10.0, 0.0, 0.0]),
            delta_K=delta_K,
            max_energy_inject=600.0,
            pos_tol=0.10,
            ori_tol=0.25,
        )

    # Default fallback: try to stay upright and near origin.
    return GoalEML(
        name=t,
        invariants=['not_fallen', 'minimal_energy'],
        target_pos=np.array([0.0, 0.0, 1.0]),
        delta_K=delta_K,
        max_energy_inject=400.0,
        pos_tol=0.10,
        ori_tol=0.25,
    )


# ── v1.1.0: Extended task factory functions ──


def make_humanoid_walk_eml(physics,
                            delta_K: float = 0.05) -> GoalEML:
    """Factory for humanoid walk GoalEML.

    Creates a GoalEML for the humanoid-walk task where the humanoid
    must walk forward while maintaining upright posture.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for humanoid-walk task.
    """
    return GoalEML(
        name='humanoid_walk',
        invariants=['torso_upright', 'com_x_advancing', 'no_self_collide'],
        target_pos=np.array([5.0, 0.0, 1.4]),
        delta_K=delta_K,
        max_energy_inject=600.0,
        pos_tol=0.10,
        ori_tol=0.15,
    )


def make_humanoid_run_eml(physics,
                           delta_K: float = 0.08) -> GoalEML:
    """Factory for humanoid run GoalEML.

    Creates a GoalEML for the humanoid-run task where the humanoid
    must run forward at higher speed while maintaining upright posture.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for humanoid-run task.
    """
    return GoalEML(
        name='humanoid_run',
        invariants=['torso_upright', 'com_x_advancing', 'no_self_collide'],
        target_pos=np.array([10.0, 0.0, 1.4]),
        delta_K=delta_K,
        max_energy_inject=800.0,
        pos_tol=0.15,
        ori_tol=0.20,
    )


def make_walker_stand_eml(physics,
                           delta_K: float = 0.04) -> GoalEML:
    """Factory for walker stand GoalEML.

    Creates a GoalEML for the walker-stand task where the walker
    must remain upright and stable.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for walker-stand task.
    """
    return GoalEML(
        name='walker_stand',
        invariants=['torso_upright', 'foot_on_ground', 'no_self_collide'],
        target_pos=np.array([0.0, 0.0, 1.0]),
        delta_K=delta_K,
        max_energy_inject=300.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_walker_walk_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for walker walk GoalEML.

    Creates a GoalEML for the walker-walk task where the walker
    must walk forward without falling.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for walker-walk task.
    """
    return GoalEML(
        name='walker_walk',
        invariants=['com_x_advancing', 'not_fallen', 'no_self_collide'],
        target_pos=np.array([5.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=400.0,
        pos_tol=0.10,
        ori_tol=0.20,
    )


def make_cheetah_run_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for cheetah run GoalEML.

    Creates a GoalEML for the cheetah-run task where the cheetah
    must run forward as fast as possible.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cheetah-run task.
    """
    return GoalEML(
        name='cheetah_run',
        invariants=['com_x_advancing', 'not_fallen'],
        target_pos=np.array([10.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=500.0,
        pos_tol=0.10,
        ori_tol=0.0,
    )


def make_hopper_hop_eml(physics,
                         delta_K: float = 0.04) -> GoalEML:
    """Factory for hopper hop GoalEML.

    Creates a GoalEML for the hopper-hop task where the hopper
    must hop forward while maintaining balance.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for hopper-hop task.
    """
    return GoalEML(
        name='hopper_hop',
        invariants=['com_x_advancing', 'not_fallen', 'no_self_collide'],
        target_pos=np.array([3.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=250.0,
        pos_tol=0.10,
        ori_tol=0.25,
    )


def make_cartpole_balance_eml(physics,
                               delta_K: float = 0.02) -> GoalEML:
    """Factory for cartpole balance GoalEML.

    Creates a GoalEML for the cartpole-balance task where the pole
    must remain balanced upright on the cart.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cartpole-balance task.
    """
    return GoalEML(
        name='cartpole_balance',
        invariants=['pole_upright', 'cart_centered'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=100.0,
        pos_tol=0.05,
        ori_tol=0.10,
    )


def make_cartpole_swingup_eml(physics,
                               delta_K: float = 0.03) -> GoalEML:
    """Factory for cartpole swingup GoalEML.

    Creates a GoalEML for the cartpole-swingup task where the pole
    must be swung up and balanced from a downward starting position.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cartpole-swingup task.
    """
    return GoalEML(
        name='cartpole_swingup',
        invariants=['pole_upright', 'cart_centered'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=150.0,
        pos_tol=0.10,
        ori_tol=0.15,
    )


def make_cartpole_balance_sparse_eml(physics,
                                      delta_K: float = 0.02) -> GoalEML:
    """Factory for cartpole balance sparse GoalEML.

    Creates a GoalEML for the cartpole-balance_sparse task (sparse
    reward variant of cartpole-balance).

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cartpole-balance_sparse task.
    """
    return GoalEML(
        name='cartpole_balance_sparse',
        invariants=['pole_upright', 'cart_centered'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=100.0,
        pos_tol=0.05,
        ori_tol=0.10,
    )


def make_cartpole_swingup_sparse_eml(physics,
                                      delta_K: float = 0.03) -> GoalEML:
    """Factory for cartpole swingup sparse GoalEML.

    Creates a GoalEML for the cartpole-swingup_sparse task (sparse
    reward variant of cartpole-swingup).

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cartpole-swingup_sparse task.
    """
    return GoalEML(
        name='cartpole_swingup_sparse',
        invariants=['pole_upright', 'cart_centered'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=150.0,
        pos_tol=0.10,
        ori_tol=0.15,
    )


def make_reacher_hard_eml(physics,
                           delta_K: float = 0.03) -> GoalEML:
    """Factory for reacher hard GoalEML.

    Creates a GoalEML for the reacher-hard task where the end-effector
    must reach a larger target offset with smaller tolerance.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for reacher-hard task.
    """
    return GoalEML(
        name='reacher_hard',
        invariants=['ee_at_target'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=50.0,
        pos_tol=0.005,
        ori_tol=0.0,
    )


def make_fish_swim_eml(physics,
                        delta_K: float = 0.05) -> GoalEML:
    """Factory for fish swim GoalEML.

    Creates a GoalEML for the fish-swim task where the fish must
    swim toward a target while maintaining upright orientation.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for fish-swim task.
    """
    return GoalEML(
        name='fish_swim',
        invariants=['upright_swim', 'ee_at_target'],
        target_pos=np.array([0.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=200.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_manipulator_bring_ball_eml(physics,
                                     delta_K: float = 0.05) -> GoalEML:
    """Factory for manipulator bring ball GoalEML.

    Creates a GoalEML for the manipulator-bring_ball task where
    the manipulator must bring a ball to a target location.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for manipulator-bring_ball task.
    """
    return GoalEML(
        name='manipulator_bring_ball',
        invariants=['ball_at_target', 'no_self_collide'],
        target_pos=np.array([0.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=300.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_acrobot_swingup_eml(physics,
                              delta_K: float = 0.05) -> GoalEML:
    """Factory for acrobot swingup GoalEML.

    Creates a GoalEML for the acrobot-swingup task where the
    second link must be swung up to a vertical position.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for acrobot-swingup task.
    """
    return GoalEML(
        name='acrobot_swingup',
        invariants=['tip_upright'],
        target_pos=np.array([0.0]),
        delta_K=delta_K,
        max_energy_inject=100.0,
        pos_tol=0.10,
        ori_tol=0.20,
    )


def make_pendulum_swingup_eml(physics,
                               delta_K: float = 0.03) -> GoalEML:
    """Factory for pendulum swingup GoalEML.

    Creates a GoalEML for the pendulum-swingup task where the
    pendulum must be swung from hanging down to upright position.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for pendulum-swingup task.
    """
    return GoalEML(
        name='pendulum_swingup',
        invariants=['pole_upright'],
        target_pos=np.array([0.0]),
        delta_K=delta_K,
        max_energy_inject=50.0,
        pos_tol=0.05,
        ori_tol=0.10,
    )


def make_finger_spin_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for finger spin GoalEML.

    Creates a GoalEML for the finger-spin task where the finger
    must spin a object to a target rotation speed.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for finger-spin task.
    """
    return GoalEML(
        name='finger_spin',
        invariants=['object_spinning'],
        target_pos=np.array([0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=100.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_finger_turn_easy_eml(physics,
                               delta_K: float = 0.04) -> GoalEML:
    """Factory for finger turn easy GoalEML.

    Creates a GoalEML for the finger-turn_easy task where the finger
    must rotate a object to a target orientation.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for finger-turn_easy task.
    """
    return GoalEML(
        name='finger_turn_easy',
        invariants=['object_at_target_ori'],
        target_pos=np.array([0.0]),
        delta_K=delta_K,
        max_energy_inject=80.0,
        pos_tol=0.05,
        ori_tol=0.15,
    )


def make_finger_turn_hard_eml(physics,
                               delta_K: float = 0.03) -> GoalEML:
    """Factory for finger turn hard GoalEML.

    Creates a GoalEML for the finger-turn_hard task where the finger
    must rotate a object to a target orientation with smaller tolerance.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for finger-turn_hard task.
    """
    return GoalEML(
        name='finger_turn_hard',
        invariants=['object_at_target_ori'],
        target_pos=np.array([0.0]),
        delta_K=delta_K,
        max_energy_inject=80.0,
        pos_tol=0.02,
        ori_tol=0.10,
    )


def make_ball_in_cup_catch_eml(physics,
                                delta_K: float = 0.05) -> GoalEML:
    """Factory for ball in cup catch GoalEML.

    Creates a GoalEML for the ball_in_cup-catch task where the
    ball must be caught inside the cup.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for ball_in_cup-catch task.
    """
    return GoalEML(
        name='ball_in_cup_catch',
        invariants=['ball_in_cup'],
        target_pos=np.array([0.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=100.0,
        pos_tol=0.05,
        ori_tol=0.0,
    )


def make_swimmer_swim6_eml(physics,
                            delta_K: float = 0.05) -> GoalEML:
    """Factory for swimmer swim6 GoalEML.

    Creates a GoalEML for the swimmer-swim6 task where the 6-link
    swimmer must swim forward.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for swimmer-swim6 task.
    """
    return GoalEML(
        name='swimmer_swim6',
        invariants=['com_x_advancing', 'no_self_collide'],
        target_pos=np.array([5.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=200.0,
        pos_tol=0.10,
        ori_tol=0.0,
    )


def make_swimmer_swim15_eml(physics,
                             delta_K: float = 0.05) -> GoalEML:
    """Factory for swimmer swim15 GoalEML.

    Creates a GoalEML for the swimmer-swim15 task where the 15-link
    swimmer must swim forward.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for swimmer-swim15 task.
    """
    return GoalEML(
        name='swimmer_swim15',
        invariants=['com_x_advancing', 'no_self_collide'],
        target_pos=np.array([5.0, 0.0, 0.0]),
        delta_K=delta_K,
        max_energy_inject=300.0,
        pos_tol=0.10,
        ori_tol=0.0,
    )
