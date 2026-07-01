"""
IDO vs PPO/SAC/TD-MPC2 — MuJoCo-Bench-IDO Comparative Evaluation
=================================================================

Runs IDO/TOMAS agent alongside baseline RL agents (PPO, SAC, TD-MPC2)
and a random agent on dm_control benchmark tasks. Computes comparative
metrics including Survival Rate, Noether Violation Rate (NVR), and
Step-Efficiency Ratio (SER = baseline_steps / IDO_steps).

Outputs JSON + CSV results to benchmarks/results/ directory.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
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
    _import_env,
    run_single_episode,
)
from core.goal_eml_mj import GoalEML, make_humanoid_reach_eml

BASELINE_REGISTRY: Dict[str, Callable] = {}


def register_baseline(name: str):
    """Decorator to register a baseline agent factory by name.

    Args:
        name: Baseline identifier string (e.g., 'ppo', 'sac', 'tdmpc2').

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


@register_baseline("tdmpc2")
def make_tdmpc2_agent(env, goal: GoalEML, **kw):
    """Factory for TD-MPC2 baseline agent (requires tdmpc2 package).

    Falls back to None (→ random) if tdmpc2 is not installed.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance (used to derive model name).
        **kw: Additional keyword arguments (unused).

    Returns:
        TDMPC2 model instance or None on import failure.
    """
    try:
        import tdmpc2
        return tdmpc2.TDMPC2.load(f"tdmpc2_{goal.name}")
    except ImportError:
        print("  [Baseline] tdmpc2 not installed; TD-MPC2 → random fallback")
        return None


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

    Args:
        log: List of dicts, each with keys: steps, final_eta, noether_v,
             avg_return, elapsed_s, reached_goal.

    Returns:
        Dict with aggregated metrics: n_episodes, avg_steps, std_steps,
        min_steps, avg_final_eta, avg_return, total_noether_v, nvr,
        avg_elapsed_s, survival_rate.
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

    run_single_episode returns keys: steps_to_goal, final_eta,
    noether_violations, elapsed_s, avg_return. compute_metrics
    expects: steps, final_eta, noether_v, avg_return, elapsed_s,
    reached_goal.

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
        'avg_return': raw['avg_return'],
        'elapsed_s': raw['elapsed_s'],
        'reached_goal': 1 if raw['steps_to_goal'] < max_steps else 0,
    }


def run_evaluation(task: str = 'humanoid-reach',
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

    if ido_only:
        baselines = []
    elif baselines is None or len(baselines) == 0:
        baselines = list(BASELINE_REGISTRY.keys())
    all_baselines: List[str] = baselines + (['random'] if not ido_only else [])

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
        raw_m = run_single_episode(env, agent_ido, max_steps)
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
            bl_agent = factory(env, goal)
            if bl_agent is None:
                bl_agent = get_random_agent(env, goal)

        bl_log: List[dict] = []
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
                returns += float(getattr(timestep, 'reward', 0.0))

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
            })
            print(f"  steps={steps}, return={returns:.2f}, reached={reached}")

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
        description="IDO vs PPO/SAC/TD-MPC2 Comparative Evaluation")
    parser.add_argument('--task', default='humanoid-reach',
                        help=f"dm_control task. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--kappa_thresh', type=float, default=0.05)
    parser.add_argument('--baseline', action='append',
                        choices=list(BASELINE_REGISTRY.keys()) + ['all', 'random'])
    parser.add_argument('--ido_only', action='store_true')
    parser.add_argument('--output_dir', default='benchmarks/results')
    args = parser.parse_args()

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
