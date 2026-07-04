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

    v0.6.1 Upgrade: η-mode for locomotion tasks.
    For locomotion (walker/cheetah/hopper), the dm_control reward is
    velocity-based (speed ≥ threshold), not position-based (reach point).
    Using a fixed target_pos for η makes η unreachable — the agent's
    torso distance to [5,0,0] is ~5m even at start, producing η~25+.
    locomotion η-mode replaces position residual with velocity+height+upright
    residuals that align with dm_control's tolerance() reward formula:
        η = w_vel * vel_deficit^2 + w_height * height_deficit^2
            + w_upright * upright_deficit^2
    This makes η decrease as the agent walks/runs correctly, allowing
    it to enter near-goal PD stabilize mode.

    Attributes:
        name: Task name string (e.g., 'humanoid_reach').
        invariants: List of invariant names the agent must preserve.
        target_pos: 3D target position for end-effector reach.
        delta_K: κ-Snap residual threshold (κ_thresh in agent).
        max_energy_inject: Maximum allowed energy injection (J).
        pos_tol: Position tolerance for goal achievement (m).
        ori_tol: Orientation tolerance for goal achievement (rad).
        collide_thresh: Self-collision detection threshold (m).
            Locomotion tasks (humanoid, walker, cheetah, hopper) use
            higher values (0.05) because body parts are naturally close.
            Small tasks (reacher, fish) use lower values (0.01).
        eta_mode: η computation mode — 'point' (default, distance to
            target_pos) or 'locomotion' (velocity+height+upright deficit).
        target_speed: Target horizontal speed for locomotion η (m/s).
            walker-walk: 1.0, cheetah-run: 10.0.
        target_height: Target torso height for locomotion η (m).
            walker-walk: 1.2, cheetah-run: ~0.5.
        target_upright: Target torso upright score for locomotion η.
            walker-walk: 0.7, cheetah-run: ~0.3.
        eta_weights: Override weights for η computation.
            Default: w_pos=1.0, w_ori=0.3, w_eng=0.01, w_vel=0.05.
            Locomotion override: w_vel=1.0, w_height=0.5, w_upright=0.3,
                w_eng=0.01 (velocity-dominant, not position-dominant).
    """
    name: str
    invariants: List[str] = field(default_factory=list)
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_K: float = 0.05
    max_energy_inject: float = 500.0
    pos_tol: float = 0.02
    ori_tol: float = 0.15
    collide_thresh: float = 0.01
    # ── v0.6.1: Locomotion η-mode fields ──
    eta_mode: str = 'point'  # 'point' | 'locomotion'
    target_speed: float = 0.0  # m/s (0 = not applicable)
    target_height: float = 0.0  # m (0 = not applicable)
    target_upright: float = 0.0  # score (0 = not applicable)
    eta_weights: Optional[dict] = None  # override default η weights


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
        collide_thresh=0.01,  # humanoid: default (low = tolerant, with parent-child exclusion)
        eta_mode='locomotion',  # v0.9.0 FIX: humanoid-stand is locomotion (21-dof balance)
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
        collide_thresh=0.05,  # locomotion: legs naturally close
    )


def make_walker_run_eml(physics,
                          delta_K: float = 0.5) -> GoalEML:
    """Factory for walker run GoalEML.

    Creates a GoalEML for the walker-run task where the walker must run
    forward without falling.

    v0.6.1: Switched to locomotion η-mode (same as walker-walk but higher
    target speed). dm_control walker-run reward requires speed ≥ 5 m/s
    while staying upright and at sufficient height.

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold (locomotion: larger tolerance).

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
        collide_thresh=0.05,  # locomotion: legs naturally close
        eta_mode='locomotion',
        target_speed=5.0,
        target_height=1.2,
        target_upright=0.7,
        eta_weights=np.array([1.0, 0.5, 0.3, 0.01]),
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
            collide_thresh=0.05,  # locomotion: body segments naturally close
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
            collide_thresh=0.05,  # locomotion: legs naturally close
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
            collide_thresh=0.05,  # locomotion: legs naturally close
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
                            delta_K: float = 0.3) -> GoalEML:
    """Factory for humanoid walk GoalEML.

    Creates a GoalEML for the humanoid-walk task where the humanoid
    must walk forward while maintaining upright posture.

    v0.6.1: Switched to locomotion η-mode. dm_control humanoid-walk
    reward is velocity-based (speed ≥ 1.0 m/s + upright + height ≥ 1.4).

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
        collide_thresh=0.01,  # humanoid walk: default (parent-child exclusion handles proximity)
        eta_mode='locomotion',
        target_speed=1.0,
        target_height=1.4,
        target_upright=0.8,
        eta_weights=np.array([1.0, 0.5, 0.3, 0.01]),
    )


def make_humanoid_run_eml(physics,
                           delta_K: float = 0.5) -> GoalEML:
    """Factory for humanoid run GoalEML.

    Creates a GoalEML for the humanoid-run task where the humanoid
    must run forward at higher speed while maintaining upright posture.

    v0.6.1: Switched to locomotion η-mode. dm_control humanoid-run
    reward is velocity-based (speed ≥ 5 m/s + upright + height ≥ 1.4).

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
        collide_thresh=0.01,  # humanoid run: default (parent-child exclusion handles proximity)
        eta_mode='locomotion',
        target_speed=5.0,
        target_height=1.4,
        target_upright=0.7,
        eta_weights=np.array([1.0, 0.5, 0.3, 0.01]),
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
        collide_thresh=0.05,  # locomotion: legs naturally close
    )


def make_walker_walk_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for walker walk GoalEML.

    Creates a GoalEML for the walker-walk task where the walker
    must walk forward without falling.

    v0.6.1: Switched to locomotion η-mode. The dm_control walker-walk
    reward is velocity-based (speed ≥ 1.0 m/s + upright + height ≥ 1.2),
    not position-based. Using a fixed target_pos=[5,0,0] made η ~38
    (distance to point, never decreasing), trapping the agent in far-goal
    mode. Locomotion η-mode computes η from velocity/height/upright
    deficits, which decrease as the agent walks correctly.

    η formula (locomotion mode):
        η = w_vel * vel_deficit^2 + w_height * height_deficit^2
            + w_upright * upright_deficit^2 + w_eng * energy_excess^2
        vel_deficit = max(0, target_speed - current_speed)
        height_deficit = max(0, target_height - torso_z)
        upright_deficit = max(0, target_upright - torso_upright_score)

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for walker-walk task with locomotion η-mode.
    """
    return GoalEML(
        name='walker_walk',
        invariants=['com_x_advancing', 'not_fallen', 'no_self_collide'],
        target_pos=np.array([5.0, 0.0, 1.2]),  # kept for compatibility
        delta_K=0.3,  # v0.6.1: higher κ_thresh for locomotion (η scales differently)
        max_energy_inject=400.0,
        pos_tol=0.10,
        ori_tol=0.20,
        collide_thresh=0.05,  # locomotion: legs naturally close
        # ── v0.6.1: Locomotion η-mode ──
        eta_mode='locomotion',
        target_speed=1.0,   # dm_control: horizontal_velocity ≥ 1.0 m/s
        target_height=1.2,  # dm_control: torso_height ≥ 1.2 m
        target_upright=0.7, # dm_control: torso_upright ≥ 0.7 (xmat zz)
        eta_weights={'w_vel': 1.0, 'w_height': 0.5, 'w_upright': 0.3, 'w_eng': 0.01},
    )


def make_cheetah_run_eml(physics,
                          delta_K: float = 0.05) -> GoalEML:
    """Factory for cheetah run GoalEML.

    Creates a GoalEML for the cheetah-run task where the cheetah
    must run forward as fast as possible.

    v0.6.1: Switched to locomotion η-mode. The dm_control cheetah-run
    reward is purely speed-based (speed ≥ 10 m/s, linear ramp).
    Using target_pos=[10,0,0] made η ~100 (point distance), making
    η unreachable. Locomotion η-mode computes η from speed deficit,
    which decreases as the cheetah runs faster.

    η formula (locomotion mode):
        η = w_vel * vel_deficit^2 + w_height * height_deficit^2
            + w_upright * upright_deficit^2 + w_eng * energy_excess^2
        vel_deficit = max(0, target_speed - current_speed)
        For cheetah, upright is loosely required (not fallen = height > 0).

    Args:
        physics: dm_control Physics instance.
        delta_K: κ-Snap residual threshold.

    Returns:
        GoalEML instance for cheetah-run task with locomotion η-mode.
    """
    return GoalEML(
        name='cheetah_run',
        invariants=['com_x_advancing', 'not_fallen'],
        target_pos=np.array([10.0, 0.0, 0.0]),  # kept for compatibility
        delta_K=2.0,  # v0.6.1: much higher κ_thresh for fast-running task
        max_energy_inject=500.0,
        pos_tol=0.10,
        ori_tol=0.0,
        collide_thresh=0.05,  # locomotion: legs naturally close
        # ── v0.6.1: Locomotion η-mode ──
        eta_mode='locomotion',
        target_speed=10.0,   # dm_control: speed ≥ 10 m/s for max reward
        target_height=0.3,   # cheetah is low — just "not fallen"
        target_upright=0.3,  # cheetah is horizontal — upright loosely
        eta_weights={'w_vel': 1.0, 'w_height': 0.2, 'w_upright': 0.1, 'w_eng': 0.01},
    )


def make_hopper_hop_eml(physics,
                         delta_K: float = 0.3) -> GoalEML:
    """Factory for hopper hop GoalEML.

    Creates a GoalEML for the hopper-hop task where the hopper must hop
    forward while maintaining balance.

    v0.6.1: Switched to locomotion η-mode. dm_control hopper-hop reward
    requires forward speed ≥ 2 m/s while staying upright and hopping.

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
        collide_thresh=0.05,  # locomotion: legs naturally close
        eta_mode='locomotion',
        target_speed=2.0,
        target_height=0.8,
        target_upright=0.7,
        eta_weights=np.array([1.0, 0.5, 0.3, 0.01]),
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
        # v0.6.5: locomotion η-mode for fish (forward swimming)
        eta_mode='locomotion',
        target_speed=0.3,
        target_height=0.0,
        target_upright=1.0,
        eta_weights={'vel': 0.7, 'height': 0.0, 'upright': 0.3},
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
        collide_thresh=0.05,  # locomotion: body segments naturally close
        # v0.6.5: locomotion η-mode for swimmer (forward velocity goal)
        eta_mode='locomotion',
        target_speed=0.3,     # dm_control swimmer target ~0.3 m/s forward
        target_height=0.0,    # swimmer is 2D — height not meaningful
        target_upright=1.0,   # always upright in 2D swimmer
        eta_weights={'vel': 1.0, 'height': 0.0, 'upright': 0.0},  # only velocity matters
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
        collide_thresh=0.05,  # locomotion: body segments naturally close
        # v0.6.5: locomotion η-mode for swimmer (forward velocity goal)
        eta_mode='locomotion',
        target_speed=0.3,
        target_height=0.0,
        target_upright=1.0,
        eta_weights={'vel': 1.0, 'height': 0.0, 'upright': 0.0},
    )


# ═══════════════════════════════════════════════════════════════════
# v1.2.0: 焊接域 GoalEML 扩展 (Welding Domain Extension)
# ═══════════════════════════════════════════════════════════════════

# ── 4种焊接姿态的 EML 参数表 ──
WELDING_EML_PARAMS: dict = {
    "flat": {
        "invariants": ['seam_tracking', 'stickout_range', 'thermal_limit', 'porosity_control'],
        "delta_K": 0.03,
        "max_energy_inject": 2.0,   # kJ/mm — 平焊热输入上限较高
        "pos_tol": 0.5,             # mm
        "ori_tol": 0.10,            # rad
        "target_stickout": 15.0,    # mm
    },
    "horizontal": {
        "invariants": ['seam_tracking', 'stickout_range', 'thermal_limit', 'porosity_control'],
        "delta_K": 0.03,
        "max_energy_inject": 1.8,   # kJ/mm — 横焊热输入略低
        "pos_tol": 0.5,
        "ori_tol": 0.10,
        "target_stickout": 15.0,
    },
    "vertical": {
        "invariants": ['seam_tracking', 'stickout_range', 'thermal_limit', 'porosity_control'],
        "delta_K": 0.03,
        "max_energy_inject": 1.5,   # kJ/mm — 立焊需控制热输入防铁水流失
        "pos_tol": 0.4,
        "ori_tol": 0.08,
        "target_stickout": 12.0,
    },
    "overhead": {
        "invariants": ['seam_tracking', 'stickout_range', 'thermal_limit', 'porosity_control'],
        "delta_K": 0.03,
        "max_energy_inject": 1.2,   # kJ/mm — 仰焊热输入最低, 防铁水下淌
        "pos_tol": 0.3,
        "ori_tol": 0.08,
        "target_stickout": 10.0,
    },
}


def make_welding_eml(weld_type: str = "flat",
                     physics=None,
                     delta_K: float = 0.03) -> GoalEML:
    """焊接域 GoalEML 工厂函数.

    根据焊接姿态类型创建对应的 GoalEML 实例, 配置焊接专用不变量:
      - seam_tracking: 焊缝跟踪
      - stickout_range: 干伸长范围
      - thermal_limit: 热输入限制
      - porosity_control: 气孔控制

    不同焊接姿态的热输入上限不同:
      flat=2.0, horizontal=1.8, vertical=1.5, overhead=1.2 (kJ/mm)

    Args:
        weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead").
        physics: dm_control Physics 实例 (兼容接口, 焊接域不使用).
        delta_K: κ-Snap 残差阈值.

    Returns:
        GoalEML 实例, 配置焊接域参数:
          - eta_mode: 'welding'
          - invariants: 焊接专用不变量列表
          - max_energy_inject: 根据焊接种类查表
    """
    weld_type = weld_type.lower()
    params: dict = WELDING_EML_PARAMS.get(weld_type, WELDING_EML_PARAMS["flat"])

    return GoalEML(
        name=f'welding_{weld_type}',
        invariants=params["invariants"].copy(),
        target_pos=np.array([0.0, 0.0, 0.0]),  # 焊接域不使用固定目标点
        delta_K=params.get("delta_K", delta_K),
        max_energy_inject=params["max_energy_inject"],
        pos_tol=params["pos_tol"],
        ori_tol=params["ori_tol"],
        collide_thresh=0.01,  # 焊接域: 严格碰撞检测
        # 焊接域 η 模式
        eta_mode='welding',
        # 焊接域不使用 locomotion 参数, 但保留字段以兼容
        target_speed=0.0,
        target_height=0.0,
        target_upright=0.0,
        eta_weights={
            'w_seam': WELDING_EML_PARAMS.get(weld_type, WELDING_EML_PARAMS["flat"]).get("w_seam", 1.0),
            'w_tcp': 0.3,
            'w_stick': 0.5,
            'w_current': 0.2,
            'target_stickout': params["target_stickout"],
        },
    )
