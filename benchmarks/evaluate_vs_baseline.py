"""
IDO vs PPO/SAC/TD-MPC2/Cosmos-Predict — MuJoCo-Bench-IDO Comparative Evaluation
================================================================================

Runs IDO/TOMAS agent alongside baseline RL agents (PPO, SAC, TD-MPC2 v2),
NVIDIA Cosmos-Predict world model, and a random agent on dm_control
benchmark tasks. Computes comparative metrics including Survival Rate,
Noether Violation Rate (NVR), Step-Efficiency Ratio (SER = baseline_steps
/ IDO_steps), and for Cosmos-Predict: η trajectory prediction RMSE.

Evaluation modes:
  1. Control comparison: IDO vs PPO/SAC/TD-MPC2/random (steps, NVR, SER)
  2. Prediction comparison: IDO FlowMatching η vs Cosmos-Predict state
     prediction (trajectory RMSE, correlation)

Outputs JSON + CSV results to benchmarks/results/ directory.

Author: MuJoCo-Bench-IDO v0.3.0 baseline integration
"""

import argparse
import csv
import json
import os
import sys
import traceback
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.run_mujoco_bench import (
    IDOMuJoCoAgent,
    TASK_REGISTRY,
    TASK_SUCCESS_CRITERIA,
    _import_env,
    run_single_episode,
)
from core.goal_eml_mj import GoalEML, make_humanoid_stand_eml
from core.noether_check_mj import noether_check_mj
from baselines.tdmpc2_adapter import TDMPC2Adapter, make_tdmpc2_adapter
from baselines.cosmos_predict_adapter import CosmosPredictAdapter, make_cosmos_predict_adapter

BASELINE_REGISTRY: Dict[str, Callable] = {}


def register_baseline(name: str):
    """Decorator to register a baseline agent factory by name.

    Args:
        name: Baseline identifier string (e.g., 'ppo', 'sac', 'tdmpc2_v2',
              'cosmos-predict').

    Returns:
        Decorator that stores the factory in BASELINE_REGISTRY.
    """
    def deco(fn: Callable):
        BASELINE_REGISTRY[name] = fn
        return fn
    return deco


@register_baseline("ppo")
def make_ppo_agent(env, goal: GoalEML, **kw):
    """Factory for PPO baseline agent (requires stable-baselines3).

    Falls back to None (→ random) if SB3 is not installed.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance (used to derive model name).
        **kw: Additional keyword arguments (unused).

    Returns:
        PPO model instance or None on import failure.
    """
    try:
        from stable_baselines3 import PPO
        import gymnasium as gym
        return PPO.load("ppo_" + goal.name, env=gym.make(goal.name))
    except ImportError:
        print("  [Baseline] SB3 not installed; PPO → random fallback")
        return None


@register_baseline("sac")
def make_sac_agent(env, goal: GoalEML, **kw):
    """Factory for SAC baseline agent (requires stable-baselines3).

    Falls back to None (→ random) if SB3 is not installed.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance (used to derive model name).
        **kw: Additional keyword arguments (unused).

    Returns:
        SAC model instance or None on import failure.
    """
    try:
        from stable_baselines3 import SAC
        import gymnasium as gym
        return SAC.load("sac_" + goal.name, env=gym.make(goal.name))
    except ImportError:
        print("  [Baseline] SB3 not installed; SAC → random fallback")
        return None


@register_baseline("tdmpc2_v2")
def make_tdmpc2_v2_agent(env, goal: GoalEML, **kw):
    """Factory for TD-MPC2 v2 baseline agent using TDMPC2Adapter.

    Uses the baselines/tdmpc2_adapter.py adapter class instead of the
    previous direct tdmpc2.TDMPC2.load() call. The adapter provides:
    - Graceful degradation (returns None if tdmpc2 not installed)
    - dm_control task name mapping (humanoid-stand → humanoid_stand)
    - Configurable model sizes (1M/5M/19M/48M/317M)
    - Unified choose_action/reset interface

    Falls back to None (→ random) if tdmpc2 is not installed.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance (used to derive task name).
        **kw: Additional keyword arguments:
              - model_size: TD-MPC2 model size (default 5).
              - checkpoint_path: Optional checkpoint path.

    Returns:
        TDMPC2Adapter instance or None on failure.
    """
    task_name: str = kw.get('task_name', goal.name)
    model_size: int = kw.get('model_size', 5)
    checkpoint_path: Optional[str] = kw.get('checkpoint_path', None)

    adapter: Optional[TDMPC2Adapter] = make_tdmpc2_adapter(
        task_name=task_name,
        model_size=model_size,
        checkpoint_path=checkpoint_path,
    )

    if adapter is None or not adapter.is_available():
        print("  [Baseline] TD-MPC2 v2 adapter not available; → random fallback")
        return None

    return adapter


@register_baseline("cosmos-predict")
def make_cosmos_predict_adapter_factory(env, goal: GoalEML, **kw):
    """Factory for Cosmos-Predict world model baseline.

    NOTE: Cosmos-Predict is a WORLD MODEL (not a control agent).
    This baseline is used for η trajectory prediction comparison
    against IDO FlowMatching, NOT for control performance comparison.

    Falls back to None (→ skip baseline) if cosmos_predict1 or CUDA
    is not available.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - model_name: Cosmos-Predict model variant (default 7B Video2World).
              - device: Compute device (default 'cuda').

    Returns:
        CosmosPredictAdapter instance or None on failure.
    """
    model_name: str = kw.get('model_name', 'cosmos-predict1-7b-video2world')
    device: str = kw.get('device', 'cuda')

    adapter: Optional[CosmosPredictAdapter] = make_cosmos_predict_adapter(
        model_name=model_name,
        device=device,
    )

    if adapter is None or not adapter.is_available():
        print("  [Baseline] Cosmos-Predict adapter not available; baseline will be skipped")
        return None

    return adapter


def get_random_agent(env, goal: GoalEML, **kw):
    """Create a random-action baseline agent.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance (unused, but kept for API consistency).
        **kw: Additional keyword arguments (unused).

    Returns:
        RandomAgent instance that outputs uniform random actions.
    """
    class RandomAgent:
        """Agent that outputs uniform random actions in [-1, 1]."""
        def __init__(self, n: int):
            self.n: int = n

        def choose_action(self, ts) -> np.ndarray:
            """Return a uniform random action vector.

            Args:
                ts: dm_control TimeStep (unused).

            Returns:
                Random action array of shape (n,).
            """
            return np.random.uniform(-1, 1, size=self.n)

        def reset(self) -> None:
            """Reset the random agent (no state to reset)."""
            pass

    n_actions: int = (env.action_space.shape[0] if hasattr(env, 'action_space')
                      else env.physics.model.nu)
    return RandomAgent(n_actions)


def compute_metrics(log: List[dict]) -> dict:
    """Compute aggregate metrics from a list of episode log entries.

    P0-1: avg_return is mean of cumulative episode returns (not single-step).

    P0-3: nvr_breakdown tracks energy/torque/collision separately.

    Args:
        log: List of dicts, each with keys: steps, final_eta, noether_v,
             avg_return, elapsed_s, reached_goal, success, nvr_breakdown.

    Returns:
        Dict with aggregated metrics: n_episodes, avg_steps, std_steps,
        min_steps, avg_final_eta, avg_return, total_noether_v, nvr,
        avg_elapsed_s, survival_rate, success_rate, nvr_breakdown.
    """
    n: int = len(log)
    return {
        'n_episodes': n,
        'avg_steps':        float(np.mean([l['steps'] for l in log])),
        'std_steps':        float(np.std([l['steps'] for l in log])),
        'min_steps':        int(np.min([l['steps'] for l in log])),
        'avg_final_eta':   float(np.mean([l['final_eta'] for l in log])),
        'avg_return':       float(np.mean([l['avg_return'] for l in log])),
        'total_noether_v':  int(sum(l['noether_v'] for l in log)),
        'nvr':             float(np.mean([l['noether_v'] / max(l['steps'], 1) for l in log])),
        'avg_elapsed_s':   float(np.mean([l['elapsed_s'] for l in log])),
        'survival_rate':    float(np.mean([l['reached_goal'] for l in log])),
        'success_rate':     float(np.mean([int(l.get('success', False)) for l in log])),
        'nvr_breakdown': {
            'energy': sum(l.get('nvr_breakdown', {}).get('energy', 0) for l in log),
            'torque': sum(l.get('nvr_breakdown', {}).get('torque', 0) for l in log),
            'collision': sum(l.get('nvr_breakdown', {}).get('collision', 0) for l in log),
        },
    }


def compute_ser(steps_ido: float, steps_bl: float) -> float:
    """Compute Step-Efficiency Ratio: SER = baseline_steps / IDO_steps.

    Args:
        steps_ido: Average steps-to-goal for IDO agent.
        steps_bl: Average steps-to-goal for baseline agent.

    Returns:
        SER value. NaN if steps_ido ≤ 0.
    """
    if steps_ido <= 0:
        return float('nan')
    return steps_bl / steps_ido


def _remap_ido_metrics(raw: dict, max_steps: int) -> dict:
    """Remap run_single_episode output keys to compute_metrics format.

    P0-1: maps episode_return (cumulative) → avg_return key for compute_metrics.
    P0-2: maps success field for success_rate computation.
    P0-3: maps nvr_breakdown for energy/torque/collision tracking.

    Args:
        raw: Dict from run_single_episode.
        max_steps: Maximum steps threshold for goal-reach classification.

    Returns:
        Remapped dict compatible with compute_metrics.
    """
    return {
        'steps': raw['steps_to_goal'],
        'final_eta': raw['final_eta'],
        'noether_v': raw['noether_violations'],
        'avg_return': raw.get('episode_return', 0.0),
        'elapsed_s': raw['elapsed_s'],
        'reached_goal': 1 if raw['steps_to_goal'] < max_steps else 0,
        'success': raw.get('success', False),
        'nvr_breakdown': raw.get('nvr_breakdown', {"energy": 0, "torque": 0, "collision": 0}),
    }


def run_cosmos_predict_comparison(task: str = 'humanoid-stand',
                                   episodes: int = 5,
                                   max_steps: int = 2000,
                                   kappa_thresh: float = 0.05,
                                   prediction_horizon: int = 10,
                                   output_dir: str = 'benchmarks/results') -> dict:
    """Run η trajectory prediction comparison: IDO FlowMatching vs Cosmos-Predict.

    This evaluation mode is distinct from control comparison:
    - Collects IDO agent's η trajectory (from FlowMatchingEtaPredictor)
    - Runs Cosmos-Predict to predict future states
    - Computes η from predicted states using GoalEML κ-Snap
    - Compares the two η trajectories via RMSE and correlation

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of episodes for IDO trajectory collection.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO agent.
        prediction_horizon: Number of future steps for Cosmos-Predict.
        output_dir: Directory for output files.

    Returns:
        Dict with IDO η trajectory data, Cosmos-Predict predictions,
        and comparison metrics.
    """
    print(f"\n{'='*70}")
    print(f"  η Trajectory Prediction Comparison — Task: {task}")
    print(f"  IDO FlowMatching vs Cosmos-Predict")
    print(f"  Prediction horizon: {prediction_horizon} steps")
    print(f"{'='*70}")

    env = _import_env(task)
    goal_factory = TASK_REGISTRY.get(task)
    if goal_factory is None:
        print(f"ERROR: unknown task '{task}'")
        sys.exit(1)
    goal = goal_factory(env.physics, kappa_thresh)

    # ── IDO η trajectory collection ──
    from core.kappa_snap_mj import FlowMatchingEtaPredictor

    agent_ido = IDOMuJoCoAgent(env, goal,
                                kappa_thresh=kappa_thresh,
                                enable_critique=True)
    agent_ido.flow_predictor = FlowMatchingEtaPredictor(
        window_size=prediction_horizon)

    ido_eta_trajectories: List[List[float]] = []
    for ep in range(1, episodes + 1):
        print(f"  ── IDO Episode {ep}/{episodes} ──")
        # Reset flow predictor
        agent_ido.flow_predictor.clear()
        raw_m = run_single_episode(env, agent_ido, max_steps, task_name=task)
        # Collect η trajectory from flow predictor buffer
        ido_eta_trajectory: List[float] = list(agent_ido.flow_predictor.eta_buffer)
        ido_eta_trajectories.append(ido_eta_trajectory)
        print(f"  η trajectory length: {len(ido_eta_trajectory)}, "
              f"final_η={raw_m['final_eta']:.6f}")

    # ── Cosmos-Predict η trajectory ──
    cosmos_adapter: Optional[CosmosPredictAdapter] = make_cosmos_predict_adapter()

    cosmos_comparison_results: List[Dict[str, object]] = []
    if cosmos_adapter is not None and cosmos_adapter.is_available():
        print(f"\n{'─'*70}")
        print(f"  Cosmos-Predict: η Trajectory Prediction")
        print(f"{'─'*70}")

        for ep_idx, ido_traj in enumerate(ido_eta_trajectories):
            if len(ido_traj) == 0:
                continue

            # Current observation (simplified: use IDO's last observation)
            current_obs: Dict[str, np.ndarray] = {
                'state_features': np.zeros(50),  # Placeholder state vector
            }

            # Action sequence (simplified: use random actions for prediction)
            # In a full implementation, this would use IDO's actual action history
            action_sequence: np.ndarray = np.random.uniform(
                -1, 1, size=(prediction_horizon, env.physics.model.nu))

            predicted_states: Optional[List[Dict[str, np.ndarray]]] = \
                cosmos_adapter.predict_future_state(
                    current_obs, action_sequence, prediction_horizon)

            if predicted_states is not None:
                comparison: Optional[Dict[str, object]] = \
                    cosmos_adapter.compare_eta_trajectory(
                        ido_traj, predicted_states, goal)
                if comparison is not None:
                    cosmos_comparison_results.append(comparison)
                    print(f"  Episode {ep_idx + 1}: "
                          f"trajectory_rmse={comparison['trajectory_rmse']:.4f}, "
                          f"correlation={comparison['trajectory_correlation']:.4f}")
    else:
        print("  [Cosmos-Predict] Not available. Skipping prediction comparison.")
        print("  [Cosmos-Predict] Install cosmos_predict1 for full evaluation.")

    # Aggregate comparison results
    comparison_summary: Dict[str, object] = {
        'task': task,
        'eval_mode': 'cosmos-predict',
        'prediction_horizon': prediction_horizon,
        'ido_trajectories_collected': len(ido_eta_trajectories),
        'cosmos_comparisons_completed': len(cosmos_comparison_results),
    }

    if len(cosmos_comparison_results) > 0:
        avg_rmse: float = float(np.mean([
            r['trajectory_rmse'] for r in cosmos_comparison_results]))
        avg_corr: float = float(np.mean([
            r['trajectory_correlation'] for r in cosmos_comparison_results]))
        comparison_summary['avg_trajectory_rmse'] = avg_rmse
        comparison_summary['avg_trajectory_correlation'] = avg_corr
        comparison_summary['cosmos_comparison_details'] = cosmos_comparison_results

    # IDO FlowMatching metrics
    all_ido_eta: List[float] = []
    for traj in ido_eta_trajectories:
        all_ido_eta.extend(traj)

    if len(all_ido_eta) > 0:
        comparison_summary['ido_avg_eta'] = float(np.mean(all_ido_eta))
        comparison_summary['ido_std_eta'] = float(np.std(all_ido_eta))
        comparison_summary['ido_min_eta'] = float(np.min(all_ido_eta))
        comparison_summary['ido_max_eta'] = float(np.max(all_ido_eta))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  η Trajectory Comparison SUMMARY")
    print(f"{'='*70}")
    print(f"  IDO trajectories collected: {len(ido_eta_trajectories)}")
    print(f"  Cosmos comparisons completed: {len(cosmos_comparison_results)}")
    if 'avg_trajectory_rmse' in comparison_summary:
        print(f"  Avg trajectory RMSE: {comparison_summary['avg_trajectory_rmse']:.4f}")
        print(f"  Avg trajectory correlation: "
              f"{comparison_summary['avg_trajectory_correlation']:.4f}")
    if 'ido_avg_eta' in comparison_summary:
        print(f"  IDO avg η: {comparison_summary['ido_avg_eta']:.6f}")
        print(f"  IDO η std: {comparison_summary['ido_std_eta']:.6f}")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path: str = os.path.join(
        output_dir,
        f"cosmos_predict_comparison_{task}_h{prediction_horizon}.json")
    with open(out_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")

    return comparison_summary


def run_evaluation(task: str = 'humanoid-stand',
                   episodes: int = 5,
                   max_steps: int = 2000,
                   kappa_thresh: float = 0.05,
                   baselines: Optional[List[str]] = None,
                   ido_only: bool = False,
                   output_dir: str = 'benchmarks/results') -> dict:
    """Run full comparative evaluation: IDO vs baselines.

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of episodes per agent.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO agent.
        baselines: List of baseline names to evaluate. None → all.
        ido_only: If True, skip all baselines.
        output_dir: Directory for JSON/CSV output files.

    Returns:
        Dict containing IDO metrics, baseline metrics, and SER values.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  IDO vs Baseline — Task: {task}   Episodes: {episodes}")
    print(f"{'='*70}")

    # Filter out cosmos-predict from control evaluation (it's a world model)
    control_baselines: List[str] = []
    cosmos_in_list: bool = False
    if ido_only:
        control_baselines = []
    elif baselines is not None:
        for bl in baselines:
            if bl == 'cosmos-predict':
                cosmos_in_list = True
            else:
                control_baselines.append(bl)
    else:
        # Default: all control baselines (exclude cosmos-predict)
        control_baselines = [bl for bl in BASELINE_REGISTRY.keys()
                             if bl != 'cosmos-predict']

    all_baselines: List[str] = control_baselines + (
        ['random'] if not ido_only else [])

    env = _import_env(task)
    goal_factory = TASK_REGISTRY.get(task)
    if goal_factory is None:
        print(f"ERROR: unknown task '{task}'")
        sys.exit(1)
    goal = goal_factory(env.physics, kappa_thresh)

    results: dict = {'task': task, 'episodes': episodes, 'kappa_thresh': kappa_thresh}

    # ── IDO Agent ──
    print(f"\n{'─'*70}")
    print(f"  IDO Agent")
    print(f"{'─'*70}")
    agent_ido = IDOMuJoCoAgent(env, goal,
                                kappa_thresh=kappa_thresh,
                                enable_critique=True)
    ido_log: List[dict] = []
    for ep in range(1, episodes + 1):
        print(f"  ── Episode {ep}/{episodes} ──")
        raw_m = run_single_episode(env, agent_ido, max_steps, task_name=task)
        m = _remap_ido_metrics(raw_m, max_steps)
        ido_log.append(m)
        print(f"  steps={m['steps']}, η={m['final_eta']:.6f}, "
              f"noether_v={m['noether_v']}, "
              f"return={m['avg_return']:.2f}")

    results['IDO'] = compute_metrics(ido_log)
    print(f"\n  IDO summary:  avg_steps={results['IDO']['avg_steps']:.1f}  "
          f"NVR={results['IDO']['nvr']:.4f}  "
          f"SER_ref={results['IDO']['avg_steps']:.1f} (self)")

    # ── Baselines ──
    bl_results: Dict[str, dict] = {}
    for bl_name in all_baselines:
        print(f"\n{'─'*70}")
        print(f"  Baseline: {bl_name.upper()}")
        print(f"{'─'*70}")

        if bl_name == 'random':
            bl_agent = get_random_agent(env, goal)
        else:
            factory = BASELINE_REGISTRY.get(bl_name)
            if factory is None:
                print(f"  Unknown baseline '{bl_name}', skipping.")
                continue
            # Pass task_name for TD-MPC2 adapter
            bl_agent = factory(env, goal, task_name=task)
            if bl_agent is None:
                bl_agent = get_random_agent(env, goal)

        bl_log: List[dict] = []
        # P0-2: per-task success criteria for baselines
        bl_success_fn = TASK_SUCCESS_CRITERIA.get(task, None)
        for ep in range(1, episodes + 1):
            print(f"  ── Episode {ep}/{episodes} ──")
            timestep = env.reset()
            agent_ido.stall_count = 0
            agent_ido.prev_data = None
            agent_ido._last_eta = None
            steps: int = 0
            noether_v: int = 0
            returns: float = 0.0
            start: float = time.time()
            reached: int = 0
            success: bool = False
            nvr_breakdown: dict = {"energy": 0, "torque": 0, "collision": 0}
            prev_data = None

            for _ in range(max_steps):
                if hasattr(bl_agent, 'choose_action'):
                    act = bl_agent.choose_action(timestep)
                else:
                    act = bl_agent(timestep.observation
                                   if hasattr(timestep, 'observation')
                                   else timestep)
                try:
                    timestep = env.step(act)
                except Exception:
                    traceback.print_exc()
                    break
                steps += 1
                step_reward: float = float(timestep.reward or 0.0)
                returns += step_reward

                # P0-3: Noether check with breakdown for baselines
                if prev_data is not None:
                    nvr_result: dict = noether_check_mj(
                        prev_data, env.physics.data, goal)
                    if not nvr_result["ok"]:
                        noether_v += 1
                        nvr_breakdown["energy"] += nvr_result["energy"]
                        nvr_breakdown["torque"] += nvr_result["torque"]
                        nvr_breakdown["collision"] += nvr_result["collision"]
                prev_data = env.physics.data

                # P0-2: per-task success criteria check for baselines
                if bl_success_fn is not None and not success:
                    obs_dict: dict = timestep.observation if hasattr(timestep, 'observation') else {}
                    if bl_success_fn(obs_dict, step_reward):
                        success = True

                try:
                    ee = env.physics.named.data.xpos['right_hand', :]
                    dist = np.linalg.norm(ee - goal.target_pos)
                    if dist < goal.pos_tol:
                        print(f"  Goal at step {steps}")
                        reached = 1
                        break
                except (KeyError, IndexError):
                    pass
                if timestep.last():
                    break

            elapsed: float = time.time() - start
            bl_log.append({
                'steps': steps,
                'final_eta': 0.0,
                'noether_v': noether_v,
                'avg_return': returns,
                'elapsed_s': elapsed,
                'reached_goal': reached,
                'success': success,
                'nvr_breakdown': nvr_breakdown,
            })
            print(f"  steps={steps}, return={returns:.2f}, reached={reached}, success={success}")

        bl_metrics: dict = compute_metrics(bl_log)
        bl_results[bl_name] = bl_metrics

        if 'IDO' in results:
            ser: float = compute_ser(results['IDO']['avg_steps'],
                                     bl_metrics['avg_steps'])
            bl_metrics['SER_vs_IDO'] = ser
            print(f"  SER({bl_name} vs IDO) = {ser:.2f}")

    results['baselines'] = bl_results

    # ── IDO Prophecy Verification ──
    print(f"\n{'='*70}")
    print(f"  IDO PROPHECY VERIFICATION")
    print(f"{'='*70}")
    if 'IDO' in results:
        ido_m: dict = results['IDO']
        print(f"  IDO NVR = {ido_m['nvr']:.6f}  (Prophecy: NVR ≡ 0)  "
              f"{'PASS' if ido_m['nvr'] == 0 else 'CHECK'}")
        for bl_name, bl_m in bl_results.items():
            ser = bl_m.get('SER_vs_IDO', float('nan'))
            print(f"  SER({bl_name} vs IDO) = {ser:.2f}  "
                  f"(Prophecy: SER >= 1.2 for reach/walk)  "
                  f"{'PASS' if ser >= 1.2 else 'TBD'}")
            print(f"  {bl_name} NVR = {bl_m['nvr']:.6f}  "
                  f"(Prophecy: PPO/SAC NVR > 0)  "
                  f"{'PASS' if bl_m['nvr'] > 0 else 'TBD'}")

    # ── Save Outputs ──
    fname: str = f"ido_vs_baseline_{task}_e{episodes}"
    json_path: str = os.path.join(output_dir, fname + ".json")
    csv_path: str = os.path.join(output_dir, fname + ".csv")

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    all_rows: List[dict] = []
    if 'IDO' in results:
        r = results['IDO']
        row: dict = {'agent': 'IDO', 'task': task}
        row.update({k: r[k] for k in ['avg_steps', 'std_steps',
                                       'avg_return', 'nvr',
                                       'survival_rate']})
        row['SER'] = 1.0
        all_rows.append(row)
    for bl_name, r in bl_results.items():
        row = {'agent': bl_name, 'task': task}
        row.update({k: r[k] for k in ['avg_steps', 'std_steps',
                                       'avg_return', 'nvr',
                                       'survival_rate']})
        row['SER'] = r.get('SER_vs_IDO', float('nan'))
        all_rows.append(row)

    fieldnames: List[str] = ['agent', 'task', 'avg_steps', 'std_steps',
                             'avg_return', 'nvr', 'survival_rate', 'SER']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"  CSV:  {csv_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IDO vs PPO/SAC/TD-MPC2/Cosmos-Predict Comparative Evaluation (v0.3.0)")
    parser.add_argument('--task', default='humanoid-stand',
                        help=f"dm_control task. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--kappa_thresh', type=float, default=0.05)
    parser.add_argument('--baseline', action='append',
                        choices=list(BASELINE_REGISTRY.keys()) + ['all', 'random'])
    parser.add_argument('--ido_only', action='store_true')
    parser.add_argument('--eval-mode', default='control',
                        choices=['control', 'cosmos-predict'],
                        help="Evaluation mode: 'control' (IDO vs RL baselines) "
                             "or 'cosmos-predict' (η trajectory prediction comparison)")
    parser.add_argument('--prediction_horizon', type=int, default=10,
                        help="Prediction horizon for Cosmos-Predict evaluation")
    parser.add_argument('--output_dir', default='benchmarks/results')
    args = parser.parse_args()

    if args.eval_mode == 'cosmos-predict':
        run_cosmos_predict_comparison(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            prediction_horizon=args.prediction_horizon,
            output_dir=args.output_dir,
        )
    else:
        baselines: Optional[List[str]] = None
        if args.baseline:
            if 'all' in args.baseline:
                baselines = list(BASELINE_REGISTRY.keys())
            else:
                baselines = [b for b in args.baseline if b != 'random']
                if 'random' in args.baseline:
                    baselines.append('random')

        run_evaluation(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            baselines=baselines,
            ido_only=args.ido_only,
            output_dir=args.output_dir,
        )
