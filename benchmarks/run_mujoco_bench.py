"""
MuJoCo-Bench-IDO — Main Evaluation Harness
============================================

Runs IDO/TOMAS agent episodes on dm_control benchmark tasks and
collects performance metrics (steps-to-goal, final η, Noether
violations, elapsed time, average return).

Supported tasks: humanoid-reach, hopper-stand, walker-run, reacher-easy.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import argparse
import json
import numpy as np
import os
import sys
import traceback
import time

from agent.mujoco_ido_agent import IDOMuJoCoAgent
from core.goal_eml_mj import (make_humanoid_reach_eml,
                               make_hopper_stand_eml,
                               make_walker_run_eml,
                               make_reacher_easy_eml)

IDO_RUN_MUJOCO_BENCH_VERSION: str = "v1.0.0"

TASK_REGISTRY: dict = {
    'humanoid-reach':  make_humanoid_reach_eml,
    'hopper-stand':    make_hopper_stand_eml,
    'walker-run':      make_walker_run_eml,
    'reacher-easy':    make_reacher_easy_eml,
}


def _import_env(task: str):
    """Import and load a dm_control environment by task name.

    Splits the task string on '-' to derive (domain, task_name) and
    calls dm_control.suite.load(). Falls back to error messages if
    dm_control is not installed or the task name is invalid.

    Args:
        task: Task identifier string (e.g., 'humanoid-reach').

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
    except ValueError:
        print(f"ERROR: unknown task '{task}'. Choose from: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)


def run_single_episode(env, agent: IDOMuJoCoAgent,
                       max_steps: int = 1000) -> dict:
    """Run a single IDO agent episode and collect performance metrics.

    Executes the IDO decision loop for up to max_steps, checking oracle
    replay, Noether violations, and goal achievement at each step.

    Args:
        env: dm_control Environment instance.
        agent: IDOMuJoCoAgent instance.
        max_steps: Maximum number of environment steps per episode.

    Returns:
        Dict with keys: steps_to_goal, final_eta, noether_violations,
        elapsed_s, avg_return.
    """
    timestep = env.reset()
    agent.prev_data = None
    agent.stall_count = 0
    agent._last_eta = None

    noether_violations: int = 0
    steps: int = 0
    start_time: float = time.time()

    for step_idx in range(max_steps):
        # Oracle replay takes precedence
        replay = agent.replay_oracle(step_idx)
        if replay is not None:
            action = replay
        else:
            action = agent.choose_action(timestep)

        try:
            timestep = env.step(action)
        except Exception:
            print(f"  [IDO] step {step_idx}: exception during env.step:")
            traceback.print_exc()
            break

        steps += 1

        # Noether check between prev and current data
        if agent.prev_data is not None:
            from core.noether_check_mj import noether_check_mj
            ok, _ = noether_check_mj(agent.prev_data,
                                      env.physics.data,
                                      agent.goal)
            if not ok:
                noether_violations += 1

        # Goal achievement check via end-effector distance
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

    return {
        'steps_to_goal': steps,
        'final_eta': final_eta,
        'noether_violations': noether_violations,
        'elapsed_s': elapsed,
        'avg_return': getattr(timestep, 'reward', 0.0),
    }


def run_benchmark(task: str = 'humanoid-reach',
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
        metrics = run_single_episode(env, agent, max_steps)
        results.append(metrics)
        print(f"  Result:  steps={metrics['steps_to_goal']},  "
              f"final_η={metrics['final_eta']:.6f},  "
              f"Noether_violations={metrics['noether_violations']},  "
              f"time={metrics['elapsed_s']:.1f}s")

    n: int = len(results)
    summary: dict = {
        'task': task,
        'episodes': episodes,
        'kappa_thresh': kappa_thresh,
        'avg_steps': float(np.mean([r['steps_to_goal'] for r in results])),
        'std_steps': float(np.std([r['steps_to_goal'] for r in results])),
        'avg_final_eta': float(np.mean([r['final_eta'] for r in results])),
        'total_noether_violations': sum(r['noether_violations'] for r in results),
        'avg_return': float(np.mean([r['avg_return'] for r in results])),
        'total_time_s': float(np.sum([r['elapsed_s'] for r in results])),
    }

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IDO/TOMAS MuJoCo Benchmark Runner")
    parser.add_argument('--task', default='humanoid-reach',
                        help=f"Task name. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--kappa_thresh', type=float, default=0.05)
    parser.add_argument('--no_critique', action='store_true')
    args = parser.parse_args()

    run_benchmark(
        task=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        kappa_thresh=args.kappa_thresh,
        enable_critique=not args.no_critique,
    )
