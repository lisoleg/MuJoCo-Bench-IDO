"""
MuJoCo-Bench-IDO — Main Evaluation Harness
============================================

Runs IDO/TOMAS agent episodes on dm_control benchmark tasks and
collects performance metrics (steps-to-goal, final η, Noether
violations, elapsed time, average return).

Supported tasks: all dm_control.suite tasks registered in TASK_REGISTRY.

v0.2.0 Upgrade: SIP-Bench longitudinal evaluation mode
  - Three phases: T0 (Initial), T1 (Iterated), T2 (Retention)
  - Collects Hesitation-RMSE, Retry-VOC, Epiplexity metrics
  - Computes Retention Gain, Stability Index
  - CLI flag: --eval-mode sip

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import argparse
import copy
import json
import numpy as np
import os
import sys
import traceback
import time
from typing import Dict, List, Optional

from agent.mujoco_ido_agent import IDOMuJoCoAgent
from agent.psi_anchor import PsiAnchor
from core.goal_eml_mj import (GoalEML,
                               make_humanoid_stand_eml,
                               make_humanoid_walk_eml,
                               make_humanoid_run_eml,
                               make_hopper_stand_eml,
                               make_hopper_hop_eml,
                               make_walker_stand_eml,
                               make_walker_walk_eml,
                               make_walker_run_eml,
                               make_cheetah_run_eml,
                               make_cartpole_balance_eml,
                               make_cartpole_swingup_eml,
                               make_cartpole_balance_sparse_eml,
                               make_cartpole_swingup_sparse_eml,
                               make_reacher_easy_eml,
                               make_reacher_hard_eml,
                               make_fish_swim_eml,
                               make_manipulator_bring_ball_eml,
                               make_acrobot_swingup_eml,
                               make_pendulum_swingup_eml,
                               make_finger_spin_eml,
                               make_finger_turn_easy_eml,
                               make_finger_turn_hard_eml,
                               make_ball_in_cup_catch_eml,
                               make_swimmer_swim6_eml,
                               make_swimmer_swim15_eml)
from core.kappa_snap_mj import FlowMatchingEtaPredictor

IDO_RUN_MUJOCO_BENCH_VERSION: str = "v0.3.0"

TASK_REGISTRY: dict = {
    'humanoid-stand':            make_humanoid_stand_eml,
    'humanoid-walk':             make_humanoid_walk_eml,
    'humanoid-run':              make_humanoid_run_eml,
    'walker-stand':              make_walker_stand_eml,
    'walker-walk':               make_walker_walk_eml,
    'walker-run':                make_walker_run_eml,
    'hopper-stand':              make_hopper_stand_eml,
    'hopper-hop':                make_hopper_hop_eml,
    'cheetah-run':               make_cheetah_run_eml,
    'cartpole-balance':          make_cartpole_balance_eml,
    'cartpole-swingup':          make_cartpole_swingup_eml,
    'cartpole-balance_sparse':   make_cartpole_balance_sparse_eml,
    'cartpole-swingup_sparse':   make_cartpole_swingup_sparse_eml,
    'reacher-easy':              make_reacher_easy_eml,
    'reacher-hard':              make_reacher_hard_eml,
    'fish-swim':                 make_fish_swim_eml,
    'manipulator-bring_ball':    make_manipulator_bring_ball_eml,
    'acrobot-swingup':           make_acrobot_swingup_eml,
    'pendulum-swingup':          make_pendulum_swingup_eml,
    'finger-spin':               make_finger_spin_eml,
    'finger-turn_easy':          make_finger_turn_easy_eml,
    'finger-turn_hard':          make_finger_turn_hard_eml,
    'ball_in_cup-catch':         make_ball_in_cup_catch_eml,
    'swimmer-swim6':             make_swimmer_swim6_eml,
    'swimmer-swim15':            make_swimmer_swim15_eml,
}

# ── Per-Task Success Criteria (P0-2) ────────────────────────────────
# Each lambda takes (obs_dict, step_reward) and returns bool indicating
# whether the task goal has been achieved for that step. Thresholds are
# based on dm_control reward structure documentation and common eval
# conventions. obs_dict is timestep.observation (may be empty for some
# tasks); step_reward is the single-step dm_control reward (float).
TASK_SUCCESS_CRITERIA: dict = {
    # reacher: reward = -dist_to_target + small_control_cost; success ≈ close to target
    'reacher-easy': lambda obs, reward: reward > -0.01,
    'reacher-hard': lambda obs, reward: reward > -0.01,
    # humanoid: upright + standing/walking/running bonuses
    'humanoid-stand': lambda obs, reward: reward > 0.5,
    'humanoid-walk': lambda obs, reward: reward > 0.3,
    'humanoid-run': lambda obs, reward: reward > 0.3,
    # walker: upright + velocity reward
    'walker-walk': lambda obs, reward: reward > 0.5,
    'walker-run': lambda obs, reward: reward > 0.3,
    'walker-stand': lambda obs, reward: reward > 0.8,
    # hopper: standing + height reward
    'hopper-stand': lambda obs, reward: reward > 0.5,
    'hopper-hop': lambda obs, reward: reward > 0.3,
    # cheetah: forward velocity reward
    'cheetah-run': lambda obs, reward: reward > 0.5,
    # cartpole: balance reward (close to 1 = perfectly balanced)
    'cartpole-balance': lambda obs, reward: reward > 0.95,
    'cartpole-swingup': lambda obs, reward: reward > 0.8,
    'cartpole-balance_sparse': lambda obs, reward: reward > 0.0,  # sparse: 1 or 0
    'cartpole-swingup_sparse': lambda obs, reward: reward > 0.0,
    # finger: rotation reward
    'finger-spin': lambda obs, reward: reward > 0.3,
    'finger-turn_easy': lambda obs, reward: reward > 0.3,
    'finger-turn_hard': lambda obs, reward: reward > 0.2,
    # fish: swimming reward
    'fish-swim': lambda obs, reward: reward > 0.3,
    # manipulator: bring_ball reward
    'manipulator-bring_ball': lambda obs, reward: reward > 0.3,
    # acrobot: swingup reward (negative during swing, positive near goal)
    'acrobot-swingup': lambda obs, reward: reward > -0.5,
    # pendulum: balance reward (close to 1 = upright)
    'pendulum-swingup': lambda obs, reward: reward > 0.9,
    # ball_in_cup: catch reward (sparse, 1 on success)
    'ball_in_cup-catch': lambda obs, reward: reward > 0.9,
    # swimmer: forward swimming reward
    'swimmer-swim6': lambda obs, reward: reward > 0.3,
    'swimmer-swim15': lambda obs, reward: reward > 0.3,
}


def _import_env(task: str):
    """Import and load a dm_control environment by task name.

    Splits the task string on '-' to derive (domain, task_name) and
    calls dm_control.suite.load(). Falls back to error messages if
    dm_control is not installed or the task name is invalid.

    Args:
        task: Task identifier string (e.g., 'humanoid-stand').

    Returns:
        dm_control Environment instance.
    """
    try:
        import dm_control.suite as suite
        domain, task_name = task.split('-', 1)
        return suite.load(domain_name=domain, task_name=task_name)
    except ImportError:
        print("ERROR: dm_control not installed. pip install dm_control mujoco")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to load task '{task}': {e}")
        print(f"Choose from: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)


def run_single_episode(env, agent: IDOMuJoCoAgent,
                       max_steps: int = 1000,
                       task_name: Optional[str] = None) -> dict:
    """Run a single IDO agent episode and collect performance metrics.

    Executes the IDO decision loop for up to max_steps, checking oracle
    replay, Noether violations, and goal achievement at each step.

    v0.2.0: If agent has psi_anchor and flow_predictor, also collects
    Hesitation-RMSE, Retry-VOC, and Epiplexity metrics.

    P0-1 fix: episode_return is cumulative reward across all steps (not
    just the last single-step reward).

    P0-2 fix: success is determined per-task via TASK_SUCCESS_CRITERIA
    (no longer only humanoid-stand right_hand distance).

    P0-3 fix: NVR breakdown tracks energy/torque/collision violations
    separately.

    Args:
        env: dm_control Environment instance.
        agent: IDOMuJoCoAgent instance.
        max_steps: Maximum number of environment steps per episode.
        task_name: Task identifier for per-task success criteria lookup.

    Returns:
        Dict with keys: steps_to_goal, final_eta, noether_violations,
        nvr_breakdown, elapsed_s, episode_return, success,
        hesit_rmse, retry_voc, epiplexity_score.
    """
    timestep = env.reset()
    agent.prev_data = None
    agent.stall_count = 0
    agent._last_eta = None

    # Reset flow predictor if present
    if hasattr(agent, 'flow_predictor') and agent.flow_predictor is not None:
        agent.flow_predictor.clear()

    # Reset psi_anchor plateau counter if present
    if hasattr(agent, 'psi_anchor') and agent.psi_anchor is not None:
        agent.psi_anchor.eta_history = []
        agent.psi_anchor.plateau_steps = 0

    noether_violations: int = 0
    nvr_breakdown: dict = {"energy": 0, "torque": 0, "collision": 0}
    episode_return: float = 0.0
    success: bool = False
    steps: int = 0
    start_time: float = time.time()

    # P0-2: per-task success criteria lookup
    success_fn = TASK_SUCCESS_CRITERIA.get(task_name, None) if task_name else None

    for step_idx in range(max_steps):
        # Oracle replay takes precedence
        replay = agent.replay_oracle(step_idx)
        if replay is not None:
            action = replay
        else:
            action = agent.choose_action(timestep, physics=env.physics)

        try:
            timestep = env.step(action)
        except Exception:
            print(f"  [IDO] step {step_idx}: exception during env.step:")
            traceback.print_exc()
            break

        steps += 1

        # P0-1: accumulate episode return (dm_control timestep.reward is single-step)
        step_reward: float = float(timestep.reward or 0.0)
        episode_return += step_reward

        # P0-3: Noether check with breakdown tracking
        if agent.prev_data is not None:
            from core.noether_check_mj import noether_check_mj
            nvr_result: dict = noether_check_mj(agent.prev_data,
                                                  env.physics.data,
                                                  agent.goal)
            if not nvr_result["ok"]:
                noether_violations += 1
                nvr_breakdown["energy"] += nvr_result["energy"]
                nvr_breakdown["torque"] += nvr_result["torque"]
                nvr_breakdown["collision"] += nvr_result["collision"]

        # P0-2: per-task success criteria check
        if success_fn is not None and not success:
            obs_dict: dict = timestep.observation if hasattr(timestep, 'observation') else {}
            if success_fn(obs_dict, step_reward):
                success = True
                print(f"  [IDO] Success achieved at step {steps} "
                      f"(task={task_name}, step_reward={step_reward:.4f})")

        # Goal achievement check via end-effector distance (legacy for humanoid-stand)
        ee: Optional[np.ndarray] = None
        try:
            ee = env.physics.named.data.xpos['right_hand', :].copy()
        except (KeyError, IndexError):
            pass

        if ee is not None:
            dist: float = np.linalg.norm(ee - agent.goal.target_pos)
            if dist < agent.goal.pos_tol:
                print(f"  [IDO] Goal reached at step {steps} "
                      f"(dist={dist:.4f}m)")
                break

        if timestep.last():
            print(f"  [IDO] Episode ended (last timestep) at step {steps}")
            break

    elapsed: float = time.time() - start_time
    final_eta: float = agent._last_eta if agent._last_eta is not None else float('inf')

    # Collect v0.2.0 metrics from flow predictor and psi_anchor
    hesit_rmse: float = 0.0
    retry_voc: float = 0.0
    epiplexity_score: float = 0.0

    if hasattr(agent, 'flow_predictor') and agent.flow_predictor is not None:
        hesit_rmse = agent.flow_predictor.compute_hesitation_rmse()
        retry_voc = agent.flow_predictor.compute_retry_voc()

    if hasattr(agent, 'psi_anchor') and agent.psi_anchor is not None:
        epiplexity_score = agent.psi_anchor.epiplexity_score

    return {
        'steps_to_goal': steps,
        'final_eta': final_eta,
        'noether_violations': noether_violations,
        'nvr_breakdown': nvr_breakdown,
        'elapsed_s': elapsed,
        'episode_return': episode_return,
        'success': success,
        'hesit_rmse': hesit_rmse,
        'retry_voc': retry_voc,
        'epiplexity_score': epiplexity_score,
    }


def _aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate episode metrics into summary statistics.

    P0-1: avg_episode_return is mean of cumulative episode returns
    (not mean of single-step rewards).

    P0-2: success_rate is fraction of episodes where task success
    was achieved per TASK_SUCCESS_CRITERIA.

    P0-3: nvr_breakdown aggregates energy/torque/collision counts
    across all episodes.

    Args:
        results: List of per-episode metric dicts from run_single_episode.

    Returns:
        Summary dict with averaged metrics.
    """
    n: int = len(results)
    if n == 0:
        return {}

    summary: dict = {
        'avg_steps': float(np.mean([r['steps_to_goal'] for r in results])),
        'std_steps': float(np.std([r['steps_to_goal'] for r in results])),
        'avg_final_eta': float(np.mean([r['final_eta'] for r in results])),
        'total_noether_violations': sum(r['noether_violations'] for r in results),
        'avg_episode_return': float(np.mean([r.get('episode_return', 0.0) for r in results])),
        'success_rate': float(np.mean([int(r.get('success', False)) for r in results])),
        'nvr_breakdown': {
            'energy': sum(r.get('nvr_breakdown', {}).get('energy', 0) for r in results),
            'torque': sum(r.get('nvr_breakdown', {}).get('torque', 0) for r in results),
            'collision': sum(r.get('nvr_breakdown', {}).get('collision', 0) for r in results),
        },
        'avg_hesit_rmse': float(np.mean([r.get('hesit_rmse', 0.0) for r in results])),
        'avg_retry_voc': float(np.mean([r.get('retry_voc', 0.0) for r in results])),
        'avg_epiplexity': float(np.mean([r.get('epiplexity_score', 0.0) for r in results])),
    }
    return summary


def run_benchmark(task: str = 'humanoid-stand',
                  episodes: int = 5,
                  max_steps: int = 2000,
                  kappa_thresh: float = 0.05,
                  enable_critique: bool = True) -> dict:
    """Run a full IDO benchmark across multiple episodes.

    Prints per-episode results and a summary, saves JSON output to
    benchmarks/results/ directory.

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO agent.
        enable_critique: Whether stall-detection critique is active.

    Returns:
        Summary dict with averaged metrics across all episodes.
    """
    print(f"\n{'='*70}")
    print(f"  IDO/TOMAS MuJoCo-Bench — Task: {task}")
    print(f"  Episodes: {episodes}  |  Max steps: {max_steps}")
    print(f"  κ-Snap δ_K: {kappa_thresh}  |  Critique: {enable_critique}")
    print(f"{'='*70}")

    env = _import_env(task)
    goal_factory = TASK_REGISTRY.get(task)
    if goal_factory is None:
        print(f"ERROR: task '{task}' not in registry.")
        sys.exit(1)

    goal = goal_factory(env.physics, kappa_thresh)
    agent = IDOMuJoCoAgent(env, goal,
                            kappa_thresh=kappa_thresh,
                            enable_critique=enable_critique)

    results: List[dict] = []
    for ep in range(1, episodes + 1):
        print(f"\n── Episode {ep}/{episodes} ──")
        metrics = run_single_episode(env, agent, max_steps, task_name=task)
        results.append(metrics)
        print(f"  Result:  steps={metrics['steps_to_goal']},  "
              f"final_η={metrics['final_eta']:.6f},  "
              f"Noether_violations={metrics['noether_violations']},  "
              f"episode_return={metrics['episode_return']:.4f},  "
              f"success={metrics['success']},  "
              f"time={metrics['elapsed_s']:.1f}s")

    summary: dict = _aggregate_metrics(results)
    summary['task'] = task
    summary['episodes'] = episodes
    summary['kappa_thresh'] = kappa_thresh

    print(f"\n{'='*70}")
    print(f"  IDO BENCHMARK SUMMARY")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"  {k:30s} = {v}")
    print(f"{'='*70}\n")

    out_dir: str = "benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path: str = os.path.join(out_dir, f"ido_{task}_e{episodes}.json")
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'episodes': results}, f, indent=2)
    print(f"  Results saved to: {out_path}")

    return summary


# ── SIP-Bench: Longitudinal Evaluation ────────────────────────────────


def _run_sip_phase(env, agent: IDOMuJoCoAgent,
                   episodes: int, max_steps: int,
                   phase_name: str,
                   task_name: Optional[str] = None) -> Dict[str, object]:
    """Run one SIP-Bench phase (T0, T1, or T2).

    Args:
        env: dm_control Environment instance.
        agent: IDOMuJoCoAgent instance.
        episodes: Number of episodes in this phase.
        max_steps: Maximum steps per episode.
        phase_name: Phase identifier ('T0', 'T1', 'T2').
        task_name: Task name for per-task success criteria lookup.

    Returns:
        Dict with phase metrics and per-episode results.
    """
    print(f"\n{'─'*70}")
    print(f"  SIP-Bench Phase: {phase_name}")
    print(f"  Episodes: {episodes}  |  Max steps: {max_steps}")
    print(f"{'─'*70}")

    phase_results: List[dict] = []
    for ep in range(1, episodes + 1):
        print(f"  ── {phase_name} Episode {ep}/{episodes} ──")
        metrics = run_single_episode(env, agent, max_steps, task_name=task_name)
        phase_results.append(metrics)
        print(f"  steps={metrics['steps_to_goal']}, "
              f"η={metrics['final_eta']:.6f}, "
              f"NV={metrics['noether_violations']}, "
              f"return={metrics.get('episode_return', 0.0):.4f}, "
              f"success={metrics.get('success', False)}, "
              f"hesit_rmse={metrics.get('hesit_rmse', 0.0):.4f}, "
              f"retry_voc={metrics.get('retry_voc', 0.0):.4f}, "
              f"epiplexity={metrics.get('epiplexity_score', 0.0):.2f}")

    phase_summary: dict = _aggregate_metrics(phase_results)
    phase_summary['phase'] = phase_name
    phase_summary['episodes'] = episodes
    phase_summary['per_episode'] = phase_results

    return phase_summary


def run_sip_benchmark(task: str = 'humanoid-stand',
                      episodes: int = 5,
                      max_steps: int = 2000,
                      kappa_thresh: float = 0.05,
                      enable_critique: bool = True,
                      evolution_rounds: int = 3) -> dict:
    """Run SIP-Bench longitudinal evaluation.

    Three phases:
    - T0 (Initial): Baseline performance without any evolution.
      Agent runs with default settings, psi_anchor observes but does not
      actively adjust thresholds or evolve primitives.
    - T1 (Iterated): Performance after N rounds of ψ-Anchor evolution.
      psi_anchor actively adjusts delta_K, applies evolution policy to
      MotorPrimitives, and injects conservation anchors. Between each
      evolution round, the agent runs episodes and psi_anchor learns from
      η trajectories.
    - T2 (Retention): Performance after reset (testing if improvements persist).
      Agent is reset to fresh state but retains psi_anchor's adjusted
      delta_K and evolved macro IC-Values. This tests whether structural
      improvements survive reset.

    SIP-Bench Key Metrics:
    - Retention Gain = T0.avg_steps / T2.avg_steps (<1 means improvement persisted)
      NOTE: lower avg_steps is better, so if T2.avg_steps < T0.avg_steps,
      T0/T2 > 1 = improvement persisted. We use the convention that
      retention_gain > 1 means improvement persisted.
    - Stability Index = std(T2 metrics) / std(T0 metrics) (<1 means more stable)

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of episodes per phase.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO agent.
        enable_critique: Whether stall-detection critique is active.
        evolution_rounds: Number of ψ-Anchor evolution rounds in T1 phase.

    Returns:
        Dict with T0, T1, T2 phase results and SIP summary metrics.
    """
    print(f"\n{'='*70}")
    print(f"  SIP-Bench Longitudinal Evaluation — Task: {task}")
    print(f"  Episodes per phase: {episodes}  |  Evolution rounds: {evolution_rounds}")
    print(f"  κ-Snap δ_K: {kappa_thresh}  |  Max steps: {max_steps}")
    print(f"{'='*70}")

    env = _import_env(task)
    goal_factory = TASK_REGISTRY.get(task)
    if goal_factory is None:
        print(f"ERROR: task '{task}' not in registry.")
        sys.exit(1)

    original_goal = goal_factory(env.physics, kappa_thresh)

    # ── Phase T0: Initial (baseline, no evolution) ──
    # Create agent WITHOUT psi_anchor active evolution
    goal_t0: GoalEML = GoalEML(
        name=original_goal.name,
        invariants=list(original_goal.invariants),
        target_pos=original_goal.target_pos.copy(),
        delta_K=original_goal.delta_K,
        max_energy_inject=original_goal.max_energy_inject,
        pos_tol=original_goal.pos_tol,
        ori_tol=original_goal.ori_tol,
    )
    agent_t0 = IDOMuJoCoAgent(env, goal_t0,
                               kappa_thresh=kappa_thresh,
                               enable_critique=enable_critique)
    # Add psi_anchor for observation only (evolution disabled)
    agent_t0.psi_anchor = PsiAnchor(goal_t0)
    agent_t0.flow_predictor = FlowMatchingEtaPredictor()

    t0_result: dict = _run_sip_phase(env, agent_t0, episodes, max_steps, 'T0', task_name=task)

    # ── Phase T1: Iterated (with ψ-Anchor evolution) ──
    # Create agent with psi_anchor active evolution
    goal_t1: GoalEML = GoalEML(
        name=original_goal.name,
        invariants=list(original_goal.invariants),
        target_pos=original_goal.target_pos.copy(),
        delta_K=original_goal.delta_K,
        max_energy_inject=original_goal.max_energy_inject,
        pos_tol=original_goal.pos_tol,
        ori_tol=original_goal.ori_tol,
    )
    agent_t1 = IDOMuJoCoAgent(env, goal_t1,
                               kappa_thresh=kappa_thresh,
                               enable_critique=enable_critique)
    agent_t1.psi_anchor = PsiAnchor(goal_t1)
    agent_t1.flow_predictor = FlowMatchingEtaPredictor()

    # Evolution rounds: between each round, run episodes and evolve
    t1_phase_results: List[dict] = []
    for evo_round in range(1, evolution_rounds + 1):
        print(f"\n  ── Evolution Round {evo_round}/{evolution_rounds} ──")

        # Run episodes
        round_results: List[dict] = []
        for ep in range(1, episodes + 1):
            print(f"  ── T1 Evo-{evo_round} Episode {ep}/{episodes} ──")
            metrics = run_single_episode(env, agent_t1, max_steps, task_name=task)
            round_results.append(metrics)

        t1_phase_results.extend(round_results)

        # Apply ψ-Anchor evolution decisions
        trend: str = agent_t1.psi_anchor.analyze_eta_trend()
        evo_policy: str = agent_t1.psi_anchor.decide_evolution_policy()
        adjusted_dk: float = agent_t1.psi_anchor.adjust_delta_K(
            agent_t1.kappa_thresh)

        # Update agent thresholds based on ψ-Anchor
        agent_t1.kappa_thresh = adjusted_dk
        agent_t1.goal.delta_K = adjusted_dk

        # Apply evolution to MotorPrimitives macros
        if agent_t1.psi_anchor.should_trigger_evolution():
            agent_t1.macros = agent_t1.psi_anchor.apply_evolution_to_macros(
                agent_t1.macros, evo_policy)
            print(f"  [ψ-Anchor] Evolution triggered: policy={evo_policy}, "
                  f"trend={trend}, δ_K={adjusted_dk:.4f}")
        else:
            print(f"  [ψ-Anchor] Evolution NOT triggered: policy={evo_policy}, "
                  f"trend={trend}")

    t1_summary: dict = _aggregate_metrics(t1_phase_results)
    t1_summary['phase'] = 'T1'
    t1_summary['episodes'] = episodes * evolution_rounds
    t1_summary['evolution_rounds'] = evolution_rounds
    t1_summary['per_episode'] = t1_phase_results

    # ── Phase T2: Retention (reset agent, keep evolved params) ──
    # Create fresh agent but with psi_anchor's adjusted delta_K and evolved macros
    adjusted_dk_from_t1: float = agent_t1.psi_anchor.adjusted_delta_K
    evolved_macros_from_t1: list = list(agent_t1.macros)

    goal_t2: GoalEML = GoalEML(
        name=original_goal.name,
        invariants=list(original_goal.invariants),
        target_pos=original_goal.target_pos.copy(),
        delta_K=adjusted_dk_from_t1,  # Retained from T1
        max_energy_inject=original_goal.max_energy_inject,
        pos_tol=original_goal.pos_tol,
        ori_tol=original_goal.ori_tol,
    )
    agent_t2 = IDOMuJoCoAgent(env, goal_t2,
                               kappa_thresh=adjusted_dk_from_t1,
                               enable_critique=enable_critique)
    # Retain evolved macro IC-Values from T1
    agent_t2.macros = evolved_macros_from_t1
    agent_t2.psi_anchor = PsiAnchor(goal_t2)
    agent_t2.flow_predictor = FlowMatchingEtaPredictor()

    t2_result: dict = _run_sip_phase(env, agent_t2, episodes, max_steps, 'T2', task_name=task)

    # ── SIP-Bench Summary Metrics ──
    t0_avg_steps: float = t0_result.get('avg_steps', float('inf'))
    t2_avg_steps: float = t2_result.get('avg_steps', float('inf'))
    t0_std_steps: float = t0_result.get('std_steps', 0.0)
    t2_std_steps: float = t2_result.get('std_steps', 0.0)

    # Retention Gain: T0_avg / T2_avg (>1 means improvement persisted)
    if t2_avg_steps > 0:
        retention_gain: float = t0_avg_steps / t2_avg_steps
    else:
        retention_gain = float('inf')

    # Stability Index: std(T2) / std(T0) (<1 means more stable)
    if t0_std_steps > 0:
        stability_index: float = t2_std_steps / t0_std_steps
    else:
        stability_index = 0.0 if t2_std_steps == 0 else float('inf')

    sip_summary: dict = {
        'task': task,
        'eval_mode': 'sip',
        'episodes_per_phase': episodes,
        'evolution_rounds': evolution_rounds,
        'kappa_thresh': kappa_thresh,
        'T0': t0_result,
        'T1': t1_summary,
        'T2': t2_result,
        'retention_gain': retention_gain,
        'stability_index': stability_index,
    }

    # Print SIP-Bench summary
    print(f"\n{'='*70}")
    print(f"  SIP-Bench SUMMARY")
    print(f"{'='*70}")
    print(f"  T0 (Initial):   avg_steps={t0_avg_steps:.1f}, "
          f"avg_η={t0_result.get('avg_final_eta', 0.0):.6f}")
    print(f"  T1 (Iterated):  avg_steps={t1_summary.get('avg_steps', 0.0):.1f}, "
          f"avg_η={t1_summary.get('avg_final_eta', 0.0):.6f}")
    print(f"  T2 (Retention): avg_steps={t2_avg_steps:.1f}, "
          f"avg_η={t2_result.get('avg_final_eta', 0.0):.6f}")
    print(f"  Retention Gain = {retention_gain:.3f}  (>1 = improvement persisted)")
    print(f"  Stability Index = {stability_index:.3f}  (<1 = more stable)")
    print(f"{'='*70}\n")

    # Save SIP-Bench results
    out_dir: str = "benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path: str = os.path.join(out_dir, f"sip_{task}_e{episodes}_r{evolution_rounds}.json")
    with open(out_path, 'w') as f:
        json.dump(sip_summary, f, indent=2, default=str)
    print(f"  SIP-Bench results saved to: {out_path}")

    return sip_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IDO/TOMAS MuJoCo Benchmark Runner (v0.2.0)")
    parser.add_argument('--task', default='humanoid-stand',
                        help=f"Task name. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--kappa_thresh', type=float, default=0.05)
    parser.add_argument('--no_critique', action='store_true')
    parser.add_argument('--eval-mode', default='standard',
                        choices=['standard', 'sip'],
                        help="Evaluation mode: 'standard' (original) or "
                             "'sip' (SIP-Bench longitudinal)")
    parser.add_argument('--evolution_rounds', type=int, default=3,
                        help="Number of ψ-Anchor evolution rounds for SIP-Bench T1 phase")
    args = parser.parse_args()

    enable_critique: bool = not args.no_critique

    if args.eval_mode == 'sip':
        run_sip_benchmark(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            enable_critique=enable_critique,
            evolution_rounds=args.evolution_rounds,
        )
    else:
        run_benchmark(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            enable_critique=enable_critique,
        )
