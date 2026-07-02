"""
MuJoCo-Bench-IDO — Baseline Training Script
=============================================

Batch trains PPO and SAC baselines on dm_control benchmark tasks using
stable-baselines3 + shimmy DmControlCompatibilityV0 wrapper.

Supported tasks (8 core from PRD):
  humanoid-stand, walker-walk, cheetah-run, hopper-stand,
  reacher-easy, cartpole-balance, finger-turn_easy, fish-swim

After training, each model is:
  1. Saved as checkpoint to checkpoints/<task>/<algo>/model.zip
  2. Evaluated for N episodes (default 5)
  3. Episode returns + success rates logged

Results output: benchmarks/results/baseline_train_results.json

Usage:
  python benchmarks/train_baselines.py --steps 1000000 --eval-episodes 5
  python benchmarks/train_baselines.py --tasks humanoid-stand walker-walk --algo ppo
  python benchmarks/train_baselines.py --tasks all --algo ppo sac --steps 500000

Author: MuJoCo-Bench-IDO v0.4.5 P0-4 baseline training
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

# ── Core task list (8 tasks from PRD) ──
CORE_TASKS: List[str] = [
    'humanoid-stand',
    'walker-walk',
    'cheetah-run',
    'hopper-stand',
    'reacher-easy',
    'cartpole-balance',
    'finger-turn_easy',
    'fish-swim',
]

# ── Success criteria per task (from run_mujoco_bench.py TASK_SUCCESS_CRITERIA) ──
TASK_SUCCESS_THRESHOLDS: Dict[str, float] = {
    'humanoid-stand': 0.5,
    'walker-walk': 0.5,
    'cheetah-run': 0.5,
    'hopper-stand': 0.5,
    'reacher-easy': -0.01,  # reward > -0.01 means close to target
    'cartpole-balance': 0.95,
    'finger-turn_easy': 0.3,
    'fish-swim': 0.3,
}


def make_gym_env(task_name: str) -> Optional[object]:
    """Create a Gymnasium-compatible environment from a dm_control task.

    Uses shimmy DmControlCompatibilityV0 to wrap dm_control environments,
    then wraps with gymnasium FlattenObservation for SB3 MlpPolicy compatibility.

    Args:
        task_name: dm_control task identifier (e.g., 'humanoid-stand').

    Returns:
        Gymnasium Env instance, or None on failure.
    """
    try:
        import dm_control.suite as suite
        from shimmy.dm_control_compatibility import DmControlCompatibilityV0
        from gymnasium.wrappers import FlattenObservation

        domain, task = task_name.split('-', 1)
        dm_env = suite.load(domain_name=domain, task_name=task)
        gym_env = DmControlCompatibilityV0(dm_env)
        # DmControlCompatibilityV0 returns dict observations;
        # SB3 MlpPolicy requires flat arrays, so wrap with FlattenObservation.
        flat_env = FlattenObservation(gym_env)
        return flat_env
    except ImportError as e:
        print(f"  [train_baselines] FATAL: Cannot import dm_control/shimmy: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  [train_baselines] ERROR: Failed to create env for '{task_name}': {e}")
        return None


def train_single(task_name: str,
                 algo: str,
                 steps: int,
                 checkpoint_dir: str,
                 verbose: int = 1) -> Optional[Dict[str, object]]:
    """Train a single PPO or SAC model on one task.

    Creates a fresh model, trains for the specified number of steps,
    saves the checkpoint, and returns training metadata.

    Args:
        task_name: dm_control task name.
        algo: Algorithm name ('ppo' or 'sac').
        steps: Number of training steps.
        checkpoint_dir: Root checkpoint directory.
        verbose: SB3 verbose level.

    Returns:
        Dict with training metadata, or None on failure.
    """
    from stable_baselines3 import PPO, SAC

    gym_env = make_gym_env(task_name)
    if gym_env is None:
        return None

    algo_cls = PPO if algo == 'ppo' else SAC
    ckpt_path: str = os.path.join(checkpoint_dir, task_name, algo, "model.zip")

    # Skip if checkpoint already exists
    if os.path.isfile(ckpt_path):
        print(f"  [train_baselines] Checkpoint already exists: {ckpt_path}")
        print(f"  [train_baselines] Skipping training. Use --force to re-train.")
        return {
            'task': task_name,
            'algo': algo,
            'steps': steps,
            'checkpoint': ckpt_path,
            'skipped': True,
        }

    print(f"\n{'─'*60}")
    print(f"  Training {algo.upper()} on {task_name}")
    print(f"  Steps: {steps}  |  Checkpoint: {ckpt_path}")
    print(f"{'─'*60}")

    start_time: float = time.time()

    try:
        if algo == 'ppo':
            model = algo_cls(
                "MlpPolicy",
                gym_env,
                verbose=verbose,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
        else:  # sac
            model = algo_cls(
                "MlpPolicy",
                gym_env,
                verbose=verbose,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
            )

        model.learn(total_timesteps=steps)

        # Save checkpoint
        ckpt_dir: str = os.path.dirname(ckpt_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(ckpt_path)

        elapsed: float = time.time() - start_time
        print(f"  [train_baselines] Training complete: {steps} steps in {elapsed:.1f}s")
        print(f"  [train_baselines] Checkpoint saved: {ckpt_path}")

        return {
            'task': task_name,
            'algo': algo,
            'steps': steps,
            'checkpoint': ckpt_path,
            'elapsed_s': elapsed,
            'skipped': False,
        }
    except Exception as e:
        print(f"  [train_baselines] Training FAILED for {algo}/{task_name}: {e}")
        traceback.print_exc()
        return None


def evaluate_model(task_name: str,
                   algo: str,
                   checkpoint_dir: str,
                   n_episodes: int = 5) -> Optional[Dict[str, object]]:
    """Evaluate a trained PPO/SAC model on a task.

    Loads the checkpoint, runs n_episodes with deterministic policy,
    and computes episode_return + success_rate per episode.

    Args:
        task_name: dm_control task name.
        algo: Algorithm name ('ppo' or 'sac').
        checkpoint_dir: Root checkpoint directory.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with evaluation metrics, or None on failure.
    """
    from stable_baselines3 import PPO, SAC

    ckpt_path: str = os.path.join(checkpoint_dir, task_name, algo, "model.zip")
    if not os.path.isfile(ckpt_path):
        print(f"  [train_baselines] No checkpoint found: {ckpt_path}")
        return None

    gym_env = make_gym_env(task_name)
    if gym_env is None:
        return None

    algo_cls = PPO if algo == 'ppo' else SAC
    model = algo_cls.load(ckpt_path, env=gym_env)

    print(f"\n{'─'*60}")
    print(f"  Evaluating {algo.upper()} on {task_name}")
    print(f"  Episodes: {n_episodes}  |  Checkpoint: {ckpt_path}")
    print(f"{'─'*60}")

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    success_count: int = 0
    success_threshold: float = TASK_SUCCESS_THRESHOLDS.get(task_name, 0.0)

    for ep_idx in range(n_episodes):
        obs, info = gym_env.reset()
        total_reward: float = 0.0
        steps: int = 0
        done: bool = False
        episode_success: bool = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

            # Check success per step using task-specific threshold
            # For reacher-easy, success means reward > -0.01 (close to target)
            # For most tasks, success means avg_step_reward > threshold
            if not episode_success and total_reward / steps > success_threshold:
                episode_success = True

        # Also check final cumulative return against threshold
        # Some tasks (cartpole) have high per-step rewards near 1.0
        avg_step_reward: float = total_reward / max(steps, 1)
        if avg_step_reward > success_threshold:
            episode_success = True

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        if episode_success:
            success_count += 1

        print(f"  Episode {ep_idx + 1}: return={total_reward:.4f}, "
              f"steps={steps}, avg_reward={avg_step_reward:.4f}, "
              f"success={episode_success}")

    avg_return: float = float(np.mean(episode_returns))
    std_return: float = float(np.std(episode_returns))
    avg_steps: float = float(np.mean(episode_lengths))
    success_rate: float = success_count / n_episodes

    print(f"\n  Summary: avg_return={avg_return:.4f} (±{std_return:.4f}), "
          f"avg_steps={avg_steps:.1f}, success_rate={success_rate:.2f}")

    return {
        'task': task_name,
        'algo': algo,
        'n_episodes': n_episodes,
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_steps': avg_steps,
        'std_steps': float(np.std(episode_lengths)),
        'success_rate': success_rate,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }


def train_and_eval_all(tasks: List[str],
                       algos: List[str],
                       steps: int,
                       eval_episodes: int,
                       checkpoint_dir: str,
                       output_dir: str,
                       force: bool = False,
                       verbose: int = 1) -> Dict[str, object]:
    """Train and evaluate all task+algo combinations.

    For each (task, algo) pair:
      1. Train the model (skip if checkpoint exists unless --force)
      2. Save checkpoint to checkpoints/<task>/<algo>/model.zip
      3. Evaluate for eval_episodes episodes
      4. Collect results

    Args:
        tasks: List of task names to train on.
        algos: List of algorithm names ('ppo', 'sac').
        steps: Number of training steps per task+algo.
        eval_episodes: Number of evaluation episodes.
        checkpoint_dir: Root checkpoint directory.
        output_dir: Directory for JSON output.
        force: Whether to re-train even if checkpoint exists.
        verbose: SB3 verbose level.

    Returns:
        Dict with all training and evaluation results.
    """
    all_results: Dict[str, object] = {
        'version': 'v0.4.5',
        'timestamp': time.strftime('%Y-%m-%d_%H%M%S'),
        'steps': steps,
        'eval_episodes': eval_episodes,
        'tasks': tasks,
        'algos': algos,
        'train_results': [],
        'eval_results': [],
    }

    total_combos: int = len(tasks) * len(algos)
    completed: int = 0
    failed: int = 0

    print(f"\n{'='*70}")
    print(f"  MuJoCo-Bench-IDO Baseline Training")
    print(f"  Tasks: {len(tasks)}  |  Algorithms: {len(algos)}  |  Total: {total_combos}")
    print(f"  Steps: {steps}  |  Eval episodes: {eval_episodes}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*70}")

    for task_name in tasks:
        for algo in algos:
            completed += 1
            print(f"\n{'='*70}")
            print(f"  [{completed}/{total_combos}] {algo.upper()} on {task_name}")
            print(f"{'='*70}")

            # Check if checkpoint exists and not forcing re-train
            ckpt_path: str = os.path.join(
                checkpoint_dir, task_name, algo, "model.zip")

            if os.path.isfile(ckpt_path) and not force:
                print(f"  Checkpoint exists: {ckpt_path}")
                print(f"  Skipping training. Loading for evaluation.")
                train_result: Optional[Dict[str, object]] = {
                    'task': task_name,
                    'algo': algo,
                    'steps': steps,
                    'checkpoint': ckpt_path,
                    'skipped': True,
                }
            else:
                # Remove existing checkpoint if forcing re-train
                if force and os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"  Removed existing checkpoint: {ckpt_path}")

                train_result = train_single(
                    task_name, algo, steps, checkpoint_dir, verbose)

            if train_result is not None:
                all_results['train_results'].append(train_result)

                # Evaluate after training
                eval_result: Optional[Dict[str, object]] = evaluate_model(
                    task_name, algo, checkpoint_dir, eval_episodes)

                if eval_result is not None:
                    all_results['eval_results'].append(eval_result)
                else:
                    failed += 1
                    print(f"  Evaluation FAILED for {algo}/{task_name}")
            else:
                failed += 1
                print(f"  Training FAILED for {algo}/{task_name}")

    # Summary
    n_train: int = len(all_results['train_results'])
    n_eval: int = len(all_results['eval_results'])
    n_skipped: int = sum(
        1 for r in all_results['train_results'] if r.get('skipped', False))

    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total combos: {total_combos}")
    print(f"  Trained: {n_train - n_skipped}")
    print(f"  Skipped (existing checkpoint): {n_skipped}")
    print(f"  Evaluated: {n_eval}")
    print(f"  Failed: {failed}")

    # Print eval results table
    if n_eval > 0:
        print(f"\n  {'Task':<20s} {'Algo':<6s} {'Avg Return':>10s} "
              f"{'Success Rate':>12s} {'Avg Steps':>10s}")
        print(f"  {'─'*60}")
        for er in all_results['eval_results']:
            print(f"  {er['task']:<20s} {er['algo']:<6s} "
                  f"{er['avg_return']:>10.4f} "
                  f"{er['success_rate']:>12.2f} "
                  f"{er['avg_steps']:>10.1f}")

    print(f"{'='*70}\n")

    # Save JSON results
    os.makedirs(output_dir, exist_ok=True)
    out_path: str = os.path.join(output_dir, "baseline_train_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MuJoCo-Bench-IDO Baseline Training (PPO/SAC) — v0.4.5")
    parser.add_argument(
        '--tasks', nargs='+', default=CORE_TASKS,
        help=f"Task names to train. Use 'all' for all 8 core tasks. "
             f"Core tasks: {CORE_TASKS}")
    parser.add_argument(
        '--algo', nargs='+', default=['ppo', 'sac'],
        choices=['ppo', 'sac'],
        help="Algorithms to train (default: ppo sac)")
    parser.add_argument(
        '--steps', type=int, default=1_000_000,
        help="Training steps per task+algo (default: 1M)")
    parser.add_argument(
        '--eval-episodes', type=int, default=5,
        help="Evaluation episodes after training (default: 5)")
    parser.add_argument(
        '--output-dir', default='benchmarks/results',
        help="Directory for JSON output (default: benchmarks/results)")
    parser.add_argument(
        '--checkpoint-dir', default='checkpoints',
        help="Root checkpoint directory (default: checkpoints)")
    parser.add_argument(
        '--force', action='store_true',
        help="Force re-training even if checkpoint exists")
    parser.add_argument(
        '--verbose', type=int, default=1,
        help="SB3 verbose level (0=silent, 1=info, 2=debug)")
    parser.add_argument(
        '--eval-only', action='store_true',
        help="Only evaluate existing checkpoints (skip training)")
    args = parser.parse_args()

    # Handle 'all' tasks keyword
    if 'all' in args.tasks:
        tasks: List[str] = CORE_TASKS
    else:
        tasks = args.tasks

    # Validate task names against dm_control suite
    for t in tasks:
        domain, task = t.split('-', 1)
        try:
            import dm_control.suite as suite
            suite.load(domain_name=domain, task_name=task)
        except Exception as e:
            print(f"  WARNING: Task '{t}' may not be valid: {e}")

    if args.eval_only:
        # Only evaluate, no training
        eval_results: List[Dict[str, object]] = []
        for task_name in tasks:
            for algo in args.algo:
                result = evaluate_model(
                    task_name, algo, args.checkpoint_dir, args.eval_episodes)
                if result is not None:
                    eval_results.append(result)

        summary: Dict[str, object] = {
            'version': 'v0.4.5',
            'timestamp': time.strftime('%Y-%m-%d_%H%M%S'),
            'eval_only': True,
            'eval_episodes': args.eval_episodes,
            'tasks': tasks,
            'algos': args.algo,
            'eval_results': eval_results,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        out_path: str = os.path.join(args.output_dir, "baseline_eval_results.json")
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Eval results saved to: {out_path}")
    else:
        # Full train + eval
        train_and_eval_all(
            tasks=tasks,
            algos=args.algo,
            steps=args.steps,
            eval_episodes=args.eval_episodes,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            force=args.force,
            verbose=args.verbose,
        )
