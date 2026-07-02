"""
Hybrid Agent Comparison Benchmark — MuJoCo-Bench-IDO Phase 3
==============================================================

Comparison evaluation script supporting 4 agent types:
  1. IDO — Pure IDO/TOMAS agent (κ-Snap + PD controller + Noether)
  2. PPO — Pure Stable-Baselines3 PPO baseline (trained policy)
  3. SAC — Pure Stable-Baselines3 SAC baseline (trained policy)
  4. Hybrid — IDO+SB3 hybrid agent (SB3 body + IDO brain)

Evaluation metrics per agent:
  - avg_return: Mean cumulative episode return
  - NVR: Noether Violation Rate (violations / total steps)
  - η trajectory: Min/avg/std/final η values across episodes
  - success_rate: Per-task success criteria achievement rate
  - avg_steps: Mean steps per episode (efficiency)
  - mode_distribution: (Hybrid only) EXPLOIT/EXPLORE/SAFE mode counts

Distribution-shift test:
  Compares agent performance under standard evaluation vs perturbed
  initial conditions (random qpos offset). Measures robustness of
  each agent to distribution shift from training conditions.

Outputs JSON + CSV results to benchmarks/results/ directory.

Usage:
  python benchmarks/compare_hybrid.py --task humanoid-stand --episodes 10
  python benchmarks/compare_hybrid.py --task walker-walk --agents IDO PPO Hybrid
  python benchmarks/compare_hybrid.py --task cheetah-run --distribution-shift

Author: MuJoCo-Bench-IDO v0.6.0 Phase 3 hybrid benchmark
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──
from benchmarks.run_mujoco_bench import (
    IDOMuJoCoAgent,
    TASK_REGISTRY,
    TASK_SUCCESS_CRITERIA,
    _import_env,
    run_single_episode,
)
from core.goal_eml_mj import GoalEML, make_generic_eml
from core.noether_check_mj import noether_check_mj
from core.kappa_snap_mj import FlowMatchingEtaPredictor
from core.kappa_snap_logger import KappaSnapLogger
from core.cq import ConscienceQuotient
from agent.psi_anchor import PsiAnchor
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent, AgentMode
from baselines.sb3_adapter import (
    SB3PPOAdapter, SB3SACAdapter,
    make_sb3_ppo_adapter, make_sb3_sac_adapter,
)
from agent.task_pd_controllers import get_controller_for_task

COMPARE_HYBRID_VERSION: str = "v0.6.0"


# ──────────────────────────────────────────────────────────────
#  Agent factory registry
# ──────────────────────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, Callable] = {}


def register_agent(name: str):
    """Decorator to register an agent factory by name.

    Args:
        name: Agent identifier string (e.g., 'IDO', 'PPO', 'SAC', 'Hybrid').

    Returns:
        Decorator that stores the factory in AGENT_REGISTRY.
    """
    def deco(fn: Callable):
        AGENT_REGISTRY[name] = fn
        return fn
    return deco


@register_agent("IDO")
def make_ido_agent(env, goal: GoalEML, **kw) -> IDOMuJoCoAgent:
    """Factory for pure IDO agent.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - task_name: Task name string.
              - kappa_thresh: κ-Snap threshold.

    Returns:
        IDOMuJoCoAgent instance.
    """
    task_name: str = kw.get('task_name', goal.name)
    kappa_thresh: float = kw.get('kappa_thresh', 0.05)

    agent: IDOMuJoCoAgent = IDOMuJoCoAgent(
        env, goal,
        task_name=task_name,
        kappa_thresh=kappa_thresh,
        enable_critique=True,
    )
    return agent


@register_agent("PPO")
def make_ppo_agent(env, goal: GoalEML, **kw) -> SB3PPOAdapter:
    """Factory for pure PPO baseline agent.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - task_name: Task name string.
              - checkpoint_dir: Root checkpoint directory.
              - auto_train_steps: Auto-training steps.

    Returns:
        SB3PPOAdapter instance, or None on failure.
    """
    task_name: str = kw.get('task_name', goal.name)
    checkpoint_dir: str = kw.get('checkpoint_dir', 'checkpoints')
    auto_train_steps: int = kw.get('auto_train_steps', 100_000)

    adapter: Optional[SB3PPOAdapter] = make_sb3_ppo_adapter(
        task_name=task_name,
        checkpoint_dir=checkpoint_dir,
        auto_train_steps=auto_train_steps,
        verbose=0,
    )
    if adapter is None or not adapter.is_available():
        print("  [Agent] SB3PPOAdapter not available; skipping")
        return None
    return adapter


@register_agent("SAC")
def make_sac_agent(env, goal: GoalEML, **kw) -> SB3SACAdapter:
    """Factory for pure SAC baseline agent.

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - task_name: Task name string.
              - checkpoint_dir: Root checkpoint directory.
              - auto_train_steps: Auto-training steps.

    Returns:
        SB3SACAdapter instance, or None on failure.
    """
    task_name: str = kw.get('task_name', goal.name)
    checkpoint_dir: str = kw.get('checkpoint_dir', 'checkpoints')
    auto_train_steps: int = kw.get('auto_train_steps', 100_000)

    adapter: Optional[SB3SACAdapter] = make_sb3_sac_adapter(
        task_name=task_name,
        checkpoint_dir=checkpoint_dir,
        auto_train_steps=auto_train_steps,
        verbose=0,
    )
    if adapter is None or not adapter.is_available():
        print("  [Agent] SB3SACAdapter not available; skipping")
        return None
    return adapter


@register_agent("Hybrid-PPO")
def make_hybrid_ppo_agent(env, goal: GoalEML, **kw) -> HybridSB3IDOAgent:
    """Factory for IDO+PPO hybrid agent (SB3 PPO body + IDO brain).

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - task_name: Task name string.
              - kappa_thresh: κ-Snap threshold for hybrid agent.
              - checkpoint_dir: Root checkpoint directory.
              - auto_train_steps: Auto-training steps for SB3 adapter.

    Returns:
        HybridSB3IDOAgent instance with PPO motor layer, or None on failure.
    """
    task_name: str = kw.get('task_name', goal.name)
    kappa_thresh: float = kw.get('kappa_thresh', 0.15)
    checkpoint_dir: str = kw.get('checkpoint_dir', 'checkpoints')
    auto_train_steps: int = kw.get('auto_train_steps', 100_000)

    sb3_adapter: Optional[SB3PPOAdapter] = make_sb3_ppo_adapter(
        task_name=task_name,
        checkpoint_dir=checkpoint_dir,
        auto_train_steps=auto_train_steps,
        verbose=0,
    )
    if sb3_adapter is None or not sb3_adapter.is_available():
        print("  [Hybrid] SB3PPOAdapter not available; skipping Hybrid-PPO")
        return None

    task_controller = get_controller_for_task(task_name, env.physics)
    hybrid: HybridSB3IDOAgent = HybridSB3IDOAgent(
        sb3_adapter=sb3_adapter,
        goal_eml=goal,
        task_name=task_name,
        kappa_thresh=kappa_thresh,
        task_controller=task_controller,
    )
    return hybrid


@register_agent("Hybrid-SAC")
def make_hybrid_sac_agent(env, goal: GoalEML, **kw) -> HybridSB3IDOAgent:
    """Factory for IDO+SAC hybrid agent (SB3 SAC body + IDO brain).

    Args:
        env: dm_control Environment instance.
        goal: GoalEML instance.
        **kw: Additional keyword arguments:
              - task_name: Task name string.
              - kappa_thresh: κ-Snap threshold for hybrid agent.
              - checkpoint_dir: Root checkpoint directory.
              - auto_train_steps: Auto-training steps for SB3 adapter.

    Returns:
        HybridSB3IDOAgent instance with SAC motor layer, or None on failure.
    """
    task_name: str = kw.get('task_name', goal.name)
    kappa_thresh: float = kw.get('kappa_thresh', 0.15)
    checkpoint_dir: str = kw.get('checkpoint_dir', 'checkpoints')
    auto_train_steps: int = kw.get('auto_train_steps', 100_000)

    sb3_adapter: Optional[SB3SACAdapter] = make_sb3_sac_adapter(
        task_name=task_name,
        checkpoint_dir=checkpoint_dir,
        auto_train_steps=auto_train_steps,
        verbose=0,
    )
    if sb3_adapter is None or not sb3_adapter.is_available():
        print("  [Hybrid] SB3SACAdapter not available; skipping Hybrid-SAC")
        return None

    task_controller = get_controller_for_task(task_name, env.physics)
    hybrid: HybridSB3IDOAgent = HybridSB3IDOAgent(
        sb3_adapter=sb3_adapter,
        goal_eml=goal,
        task_name=task_name,
        kappa_thresh=kappa_thresh,
        task_controller=task_controller,
    )
    return hybrid


# ──────────────────────────────────────────────────────────────
#  Evaluation metrics
# ──────────────────────────────────────────────────────────────

def compute_agent_metrics(log: List[dict], agent_type: str = 'generic') -> dict:
    """Compute aggregate metrics from a list of episode log entries.

    Computes avg_return, NVR, η trajectory stats, success_rate, avg_steps,
    and (for Hybrid agents) mode distribution counts.

    Args:
        log: List of dicts, each with keys: steps, final_eta, noether_v,
             avg_return, elapsed_s, success, nvr_breakdown, mode_counts (optional).
        agent_type: Agent type string ('IDO', 'PPO', 'SAC', 'Hybrid-PPO', 'Hybrid-SAC').

    Returns:
        Dict with aggregated metrics.
    """
    n: int = len(log)
    if n == 0:
        return {
            'n_episodes': 0, 'avg_return': 0.0, 'avg_steps': 0.0,
            'nvr': 0.0, 'success_rate': 0.0, 'avg_final_eta': 0.0,
        }

    # Core metrics
    metrics: dict = {
        'n_episodes': n,
        'avg_steps': float(np.mean([l['steps'] for l in log])),
        'std_steps': float(np.std([l['steps'] for l in log])),
        'min_steps': int(np.min([l['steps'] for l in log])),
        'avg_return': float(np.mean([l['avg_return'] for l in log])),
        'std_return': float(np.std([l['avg_return'] for l in log])),
        'nvr': float(np.mean([l['noether_v'] / max(l['steps'], 1) for l in log])),
        'avg_final_eta': float(np.mean([l['final_eta'] for l in log])),
        'std_final_eta': float(np.std([l['final_eta'] for l in log])),
        'min_eta': float(np.min([l['final_eta'] for l in log])),
        'success_rate': float(np.mean([int(l.get('success', False)) for l in log])),
        'avg_elapsed_s': float(np.mean([l['elapsed_s'] for l in log])),
        'total_noether_v': int(sum(l['noether_v'] for l in log)),
        'nvr_breakdown': {
            'energy': sum(l.get('nvr_breakdown', {}).get('energy', 0) for l in log),
            'torque': sum(l.get('nvr_breakdown', {}).get('torque', 0) for l in log),
            'collision': sum(l.get('nvr_breakdown', {}).get('collision', 0) for l in log),
        },
    }

    # η trajectory stats (from eta_trajectory field)
    all_eta_values: List[float] = []
    for l in log:
        traj: List[float] = l.get('eta_trajectory', [])
        all_eta_values.extend(traj)
    if len(all_eta_values) > 0:
        metrics['eta_trajectory_avg'] = float(np.mean(all_eta_values))
        metrics['eta_trajectory_std'] = float(np.std(all_eta_values))
        metrics['eta_trajectory_min'] = float(np.min(all_eta_values))
        metrics['eta_trajectory_len'] = len(all_eta_values)
    else:
        metrics['eta_trajectory_avg'] = 0.0
        metrics['eta_trajectory_std'] = 0.0
        metrics['eta_trajectory_min'] = 0.0
        metrics['eta_trajectory_len'] = 0

    # v0.6.0: CQ (Conscience Quotient) metrics
    # Aggregate CQ from per-episode CQ reports
    all_cq_values: List[float] = []
    all_cq_noether: List[float] = []
    all_cq_pgate: List[float] = []
    all_cq_sentient: List[float] = []
    for l in log:
        cq_report: dict = l.get('cq_report', {})
        if cq_report:
            all_cq_values.append(cq_report.get('cq', 0.0))
            all_cq_noether.append(cq_report.get('cq_noether', 0.0))
            all_cq_pgate.append(cq_report.get('cq_pgate', 0.0))
            all_cq_sentient.append(cq_report.get('cq_sentient', 0.0))
    if len(all_cq_values) > 0:
        metrics['cq_avg'] = float(np.mean(all_cq_values))
        metrics['cq_std'] = float(np.std(all_cq_values))
        metrics['cq_min'] = float(np.min(all_cq_values))
        metrics['cq_noether_avg'] = float(np.mean(all_cq_noether))
        metrics['cq_pgate_avg'] = float(np.mean(all_cq_pgate))
        metrics['cq_sentient_avg'] = float(np.mean(all_cq_sentient))
    else:
        metrics['cq_avg'] = 0.0
        metrics['cq_std'] = 0.0
        metrics['cq_min'] = 0.0
        metrics['cq_noether_avg'] = 0.0
        metrics['cq_pgate_avg'] = 0.0
        metrics['cq_sentient_avg'] = 0.0

    # v0.6.0: κ-Snap Merkle chain output
    # Include Merkle chain from last episode if available
    last_merkle_chain: List[dict] = []
    for l in log:
        mc: List[dict] = l.get('merkle_chain', [])
        if mc:
            last_merkle_chain = mc
    metrics['merkle_chain'] = last_merkle_chain
    metrics['merkle_chain_verified'] = any(
        l.get('merkle_chain_verified', False) for l in log)

    # Mode distribution (Hybrid agents only)
    if agent_type.startswith('Hybrid'):
        mode_counts: Dict[str, int] = {'EXPLOIT': 0, 'EXPLORE': 0, 'SAFE': 0}
        for l in log:
            mc: Dict[str, int] = l.get('mode_counts', {})
            for mode_name in mode_counts:
                mode_counts[mode_name] += mc.get(mode_name, 0)
        total_mode_steps: int = sum(mode_counts.values())
        if total_mode_steps > 0:
            metrics['mode_distribution'] = {
                k: {'count': v, 'ratio': v / total_mode_steps}
                for k, v in mode_counts.items()
            }
        else:
            metrics['mode_distribution'] = {k: {'count': 0, 'ratio': 0.0} for k in mode_counts}

    return metrics


# ──────────────────────────────────────────────────────────────
#  Episode runner
# ──────────────────────────────────────────────────────────────

def run_agent_episode(env, agent, max_steps: int = 1000,
                      task_name: Optional[str] = None,
                      is_hybrid: bool = False,
                      perturbed_init: bool = False) -> dict:
    """Run a single agent episode and collect performance metrics.

    Supports IDO, PPO/SAC, and Hybrid agents with unified metric
    collection. For Hybrid agents, also tracks mode distribution
    (EXPLOIT/EXPLORE/SAFE counts).

    Args:
        env: dm_control Environment instance.
        agent: Agent instance with choose_action(timestep) or
               choose_action(timestep, physics) interface.
        max_steps: Maximum steps per episode.
        task_name: Task identifier for success criteria lookup.
        is_hybrid: Whether the agent is a HybridSB3IDOAgent instance.
        perturbed_init: Whether to apply distribution-shift perturbation
                       (random qpos offset at episode start).

    Returns:
        Dict with episode metrics: steps, final_eta, noether_v,
        avg_return, elapsed_s, success, nvr_breakdown, eta_trajectory,
        mode_counts (Hybrid only).
    """
    # Reset environment
    timestep = env.reset()

    # Distribution-shift perturbation: random qpos offset
    if perturbed_init:
        perturbation_scale: float = 0.1
        qpos_perturb: np.ndarray = np.random.uniform(
            -perturbation_scale, perturbation_scale,
            size=min(env.physics.model.nq, len(env.physics.data.qpos)),
        )
        env.physics.data.qpos[:len(qpos_perturb)] += qpos_perturb
        # Forward kinematics to update derived quantities
        try:
            import mujoco
            mujoco.mj_forward(env.physics.model.ptr, env.physics.data.ptr)
        except (ImportError, AttributeError):
            pass

    # Reset agent state
    if hasattr(agent, 'reset'):
        agent.reset()
    elif hasattr(agent, 'prev_data'):
        agent.prev_data = None
        agent._last_eta = None
    if hasattr(agent, 'stall_count'):
        agent.stall_count = 0
    if hasattr(agent, '_step_counter'):
        agent._step_counter = 0
    if hasattr(agent, 'flow_predictor') and agent.flow_predictor is not None:
        if hasattr(agent.flow_predictor, 'clear'):
            agent.flow_predictor.clear()
    if hasattr(agent, 'psi_anchor') and agent.psi_anchor is not None:
        agent.psi_anchor.eta_history = []
        agent.psi_anchor.plateau_steps = 0

    # Episode tracking
    steps: int = 0
    noether_violations: int = 0
    nvr_breakdown: dict = {"energy": 0, "torque": 0, "collision": 0}
    episode_return: float = 0.0
    success: bool = False
    eta_trajectory: List[float] = []
    mode_counts: Dict[str, int] = {'EXPLOIT': 0, 'EXPLORE': 0, 'SAFE': 0}
    prev_data = None
    goal: Optional[GoalEML] = getattr(agent, 'goal', None)

    # Per-task success criteria
    success_fn: Optional[Callable] = TASK_SUCCESS_CRITERIA.get(task_name, None) if task_name else None

    start_time: float = time.time()

    for step_idx in range(max_steps):
        # Get action from agent
        if is_hybrid:
            # Hybrid agent: choose_action(timestep, physics) interface
            action: np.ndarray = agent.choose_action(timestep, physics=env.physics)
            # Track mode
            current_mode: str = agent._mode.value  # 'EXPLOIT', 'EXPLORE', 'SAFE'
            mode_counts[current_mode] += 1
        elif hasattr(agent, 'choose_action'):
            # IDO agent: choose_action(timestep, physics)
            try:
                action = agent.choose_action(timestep, physics=env.physics)
            except TypeError:
                # SB3 adapter: choose_action(timestep) without physics
                action = agent.choose_action(timestep)
        else:
            action = np.random.uniform(-1, 1, size=env.physics.model.nu)

        # Step environment
        try:
            timestep = env.step(action)
        except Exception:
            traceback.print_exc()
            break

        steps += 1
        step_reward: float = float(timestep.reward or 0.0)
        episode_return += step_reward

        # Collect η trajectory (for IDO and Hybrid agents)
        if hasattr(agent, '_last_eta') and agent._last_eta is not None:
            eta_trajectory.append(agent._last_eta)

        # Noether check with breakdown tracking
        if prev_data is not None and goal is not None:
            nvr_result: dict = noether_check_mj(
                prev_data, env.physics.data, goal,
                collide_thresh=goal.collide_thresh,
            )
            if not nvr_result["ok"]:
                noether_violations += 1
                nvr_breakdown["energy"] += nvr_result["energy"]
                nvr_breakdown["torque"] += nvr_result["torque"]
                nvr_breakdown["collision"] += nvr_result["collision"]
        prev_data = env.physics.data

        # Per-task success criteria check
        if success_fn is not None and not success:
            obs_dict: dict = timestep.observation if hasattr(timestep, 'observation') else {}
            if success_fn(obs_dict, step_reward):
                success = True

        if timestep.last():
            break

    elapsed: float = time.time() - start_time

    # Final η (from agent's last state or compute manually)
    final_eta: float = 0.0
    if hasattr(agent, '_last_eta') and agent._last_eta is not None:
        final_eta = agent._last_eta
    elif hasattr(agent, '_compute_eta'):
        try:
            final_eta = agent._compute_eta(env.physics, timestep)
        except Exception:
            final_eta = 0.0

    result: dict = {
        'steps': steps,
        'final_eta': final_eta,
        'noether_v': noether_violations,
        'avg_return': episode_return,
        'elapsed_s': elapsed,
        'success': success,
        'nvr_breakdown': nvr_breakdown,
        'eta_trajectory': eta_trajectory,
    }

    if is_hybrid:
        result['mode_counts'] = mode_counts

    # v0.6.0: Collect CQ report and MerkleChain from agent
    cq_report: dict = {}
    merkle_chain: List[dict] = []
    merkle_verified: bool = False

    if hasattr(agent, 'get_cq_report'):
        cq_report = agent.get_cq_report()
    if hasattr(agent, 'get_merkle_chain'):
        merkle_chain = agent.get_merkle_chain()
    if hasattr(agent, 'verify_merkle_chain'):
        try:
            merkle_verified = agent.verify_merkle_chain()
        except Exception:
            merkle_verified = False

    result['cq_report'] = cq_report
    result['merkle_chain'] = merkle_chain
    result['merkle_chain_verified'] = merkle_verified

    return result


# ──────────────────────────────────────────────────────────────
#  Distribution-shift test
# ──────────────────────────────────────────────────────────────

def run_distribution_shift_test(task: str = 'humanoid-stand',
                                 episodes: int = 5,
                                 max_steps: int = 2000,
                                 kappa_thresh: float = 0.05,
                                 agents: Optional[List[str]] = None,
                                 perturbation_scale: float = 0.1,
                                 output_dir: str = 'benchmarks/results') -> dict:
    """Run distribution-shift robustness test.

    Compares agent performance under standard evaluation vs perturbed
    initial conditions (random qpos offset). Measures how much each
    agent's performance degrades under distribution shift.

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of episodes per condition per agent.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO/Hybrid agents.
        agents: List of agent names to evaluate. None → all registered.
        perturbation_scale: Scale of qpos perturbation at episode start.
        output_dir: Directory for output files.

    Returns:
        Dict with standard and perturbed results per agent, plus
        degradation ratios.
    """
    print(f"\n{'='*70}")
    print(f"  Distribution-Shift Robustness Test — Task: {task}")
    print(f"  Perturbation scale: {perturbation_scale}")
    print(f"  Episodes per condition: {episodes}")
    print(f"{'='*70}")

    # Run standard evaluation
    standard_results: dict = run_comparison(
        task=task, episodes=episodes, max_steps=max_steps,
        kappa_thresh=kappa_thresh, agents=agents,
        output_dir=None,  # Don't save intermediate results
    )

    # Run perturbed evaluation
    perturbed_results: dict = run_comparison(
        task=task, episodes=episodes, max_steps=max_steps,
        kappa_thresh=kappa_thresh, agents=agents,
        output_dir=None,
        perturbed_init=True,
    )

    # Compute degradation ratios
    degradation: Dict[str, dict] = {}
    for agent_name in standard_results:
        std_metrics: dict = standard_results[agent_name]
        pert_metrics: dict = perturbed_results.get(agent_name, {})

        if std_metrics.get('n_episodes', 0) > 0 and pert_metrics.get('n_episodes', 0) > 0:
            degradation[agent_name] = {
                'return_degradation': (
                    std_metrics.get('avg_return', 0) - pert_metrics.get('avg_return', 0)
                ) / max(abs(std_metrics.get('avg_return', 0)), 1e-6),
                'success_degradation': (
                    std_metrics.get('success_rate', 0) - pert_metrics.get('success_rate', 0)
                ),
                'nvr_increase': (
                    pert_metrics.get('nvr', 0) - std_metrics.get('nvr', 0)
                ),
                'eta_degradation': (
                    pert_metrics.get('avg_final_eta', 0) - std_metrics.get('avg_final_eta', 0)
                ),
            }

    shift_results: dict = {
        'task': task,
        'perturbation_scale': perturbation_scale,
        'episodes': episodes,
        'standard_results': standard_results,
        'perturbed_results': perturbed_results,
        'degradation': degradation,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    fname: str = f"distribution_shift_{task}_e{episodes}_p{perturbation_scale}"
    json_path: str = os.path.join(output_dir, fname + ".json")
    with open(json_path, 'w') as f:
        json.dump(shift_results, f, indent=2, default=str)
    print(f"  Results saved to: {json_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Distribution-Shift Summary")
    print(f"{'='*70}")
    for agent_name, deg in degradation.items():
        print(f"  {agent_name}:")
        print(f"    return_degradation: {deg['return_degradation']:.4f}")
        print(f"    success_degradation: {deg['success_degradation']:.4f}")
        print(f"    nvr_increase: {deg['nvr_increase']:.4f}")
        print(f"    eta_degradation: {deg['eta_degradation']:.4f}")
    print(f"{'='*70}\n")

    return shift_results


# ──────────────────────────────────────────────────────────────
#  Main comparison evaluation
# ──────────────────────────────────────────────────────────────

def run_comparison(task: str = 'humanoid-stand',
                   episodes: int = 5,
                   max_steps: int = 2000,
                   kappa_thresh: float = 0.05,
                   agents: Optional[List[str]] = None,
                   output_dir: Optional[str] = 'benchmarks/results',
                   perturbed_init: bool = False) -> Dict[str, dict]:
    """Run comparative evaluation across multiple agent types.

    Evaluates IDO, PPO, SAC, and Hybrid agents on the same task
    with unified metric collection.

    Args:
        task: Task name from TASK_REGISTRY.
        episodes: Number of episodes per agent.
        max_steps: Maximum steps per episode.
        kappa_thresh: κ-Snap threshold for IDO/Hybrid agents.
        agents: List of agent names to evaluate. None → all registered.
        output_dir: Directory for output files. None → don't save.
        perturbed_init: Whether to apply distribution-shift perturbation.

    Returns:
        Dict mapping agent name → aggregated metrics.
    """
    # Setup
    env = _import_env(task)
    goal_factory = TASK_REGISTRY.get(task)
    if goal_factory is None:
        print(f"ERROR: unknown task '{task}'")
        sys.exit(1)
    goal: GoalEML = goal_factory(env.physics, kappa_thresh)

    # Determine which agents to evaluate
    agent_names: List[str] = agents if agents is not None else list(AGENT_REGISTRY.keys())

    condition_label: str = "Perturbed" if perturbed_init else "Standard"
    print(f"\n{'='*70}")
    print(f"  Hybrid Comparison — Task: {task}   Episodes: {episodes}")
    print(f"  Condition: {condition_label}")
    print(f"  Agents: {agent_names}")
    print(f"{'='*70}")

    all_results: Dict[str, dict] = {}

    for agent_name in agent_names:
        factory: Optional[Callable] = AGENT_REGISTRY.get(agent_name)
        if factory is None:
            print(f"  Unknown agent '{agent_name}', skipping.")
            continue

        print(f"\n{'─'*70}")
        print(f"  Agent: {agent_name}")
        print(f"{'─'*70}")

        # Create agent
        agent_instance = factory(env, goal, task_name=task, kappa_thresh=kappa_thresh)
        if agent_instance is None:
            print(f"  Agent '{agent_name}' creation failed, skipping.")
            continue

        # Determine if this is a hybrid agent
        is_hybrid: bool = isinstance(agent_instance, HybridSB3IDOAgent)

        # Run episodes
        episode_log: List[dict] = []
        for ep in range(1, episodes + 1):
            print(f"  ── Episode {ep}/{episodes} ──")
            result: dict = run_agent_episode(
                env, agent_instance, max_steps=max_steps,
                task_name=task, is_hybrid=is_hybrid,
                perturbed_init=perturbed_init,
            )
            episode_log.append(result)
            print(f"  steps={result['steps']}, return={result['avg_return']:.2f}, "
                  f"η={result['final_eta']:.6f}, "
                  f"success={result.get('success', False)}, "
                  f"NVR_v={result['noether_v']}")

        # Compute aggregate metrics
        metrics: dict = compute_agent_metrics(episode_log, agent_type=agent_name)
        all_results[agent_name] = metrics

        # Print summary
        print(f"\n  {agent_name} summary:")
        print(f"    avg_return={metrics['avg_return']:.2f}")
        print(f"    avg_steps={metrics['avg_steps']:.1f}")
        print(f"    NVR={metrics['nvr']:.4f}")
        print(f"    success_rate={metrics['success_rate']:.4f}")
        print(f"    avg_final_eta={metrics['avg_final_eta']:.6f}")
        if 'eta_trajectory_avg' in metrics:
            print(f"    eta_trajectory_avg={metrics['eta_trajectory_avg']:.6f}")
        if 'cq_avg' in metrics and metrics['cq_avg'] > 0:
            print(f"    CQ={metrics['cq_avg']:.4f} "
                  f"(noether={metrics['cq_noether_avg']:.4f}, "
                  f"pgate={metrics['cq_pgate_avg']:.4f}, "
                  f"sentient={metrics['cq_sentient_avg']:.4f})")
        if 'merkle_chain_verified' in metrics:
            print(f"    merkle_chain_verified={metrics['merkle_chain_verified']}")

        # Print mode distribution for Hybrid agents
        if 'mode_distribution' in metrics:
            md: dict = metrics['mode_distribution']
            print(f"    mode_distribution:")
            for mode_name, mode_info in md.items():
                print(f"      {mode_name}: count={mode_info['count']}, "
                      f"ratio={mode_info['ratio']:.3f}")

    # ── Cross-agent comparison ──
    print(f"\n{'='*70}")
    print(f"  Cross-Agent Comparison Summary — {task} ({condition_label})")
    print(f"{'='*70}")
    print(f"  {'Agent':<15} {'Return':>10} {'Steps':>8} {'NVR':>8} "
          f"{'Success':>8} {'Avg η':>10}")
    print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for agent_name, m in all_results.items():
        print(f"  {agent_name:<15} {m['avg_return']:>10.2f} {m['avg_steps']:>8.1f} "
              f"{m['nvr']:>8.4f} {m['success_rate']:>8.4f} {m['avg_final_eta']:>10.6f}")
    print(f"{'='*70}\n")

    # ── Save Outputs ──
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        condition_suffix: str = "_perturbed" if perturbed_init else ""
        fname: str = f"hybrid_comparison_{task}_e{episodes}{condition_suffix}"
        json_path: str = os.path.join(output_dir, fname + ".json")
        csv_path: str = os.path.join(output_dir, fname + ".csv")

        # JSON output
        full_results: dict = {
            'task': task,
            'episodes': episodes,
            'max_steps': max_steps,
            'kappa_thresh': kappa_thresh,
            'condition': condition_label,
            'version': COMPARE_HYBRID_VERSION,
            'agents': all_results,
        }
        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        print(f"  JSON: {json_path}")

        # CSV output
        fieldnames: List[str] = [
            'agent', 'task', 'condition', 'avg_return', 'std_return',
            'avg_steps', 'std_steps', 'nvr', 'success_rate',
            'avg_final_eta', 'eta_trajectory_avg',
            'total_noether_v', 'cq_avg', 'cq_noether_avg',
            'cq_pgate_avg', 'cq_sentient_avg', 'merkle_chain_verified',
        ]
        all_rows: List[dict] = []
        for agent_name, m in all_results.items():
            row: dict = {
                'agent': agent_name,
                'task': task,
                'condition': condition_label,
                'avg_return': m.get('avg_return', 0.0),
                'std_return': m.get('std_return', 0.0),
                'avg_steps': m.get('avg_steps', 0.0),
                'std_steps': m.get('std_steps', 0.0),
                'nvr': m.get('nvr', 0.0),
                'success_rate': m.get('success_rate', 0.0),
                'avg_final_eta': m.get('avg_final_eta', 0.0),
                'eta_trajectory_avg': m.get('eta_trajectory_avg', 0.0),
                'total_noether_v': m.get('total_noether_v', 0),
                'cq_avg': m.get('cq_avg', 0.0),
                'cq_noether_avg': m.get('cq_noether_avg', 0.0),
                'cq_pgate_avg': m.get('cq_pgate_avg', 0.0),
                'cq_sentient_avg': m.get('cq_sentient_avg', 0.0),
                'merkle_chain_verified': m.get('merkle_chain_verified', False),
            }
            all_rows.append(row)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"  CSV:  {csv_path}")

    return all_results


# ──────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hybrid Agent Comparison Benchmark — MuJoCo-Bench-IDO Phase 3 "
                    f"(v{COMPARE_HYBRID_VERSION})")
    parser.add_argument('--task', default='humanoid-stand',
                        help=f"dm_control task. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument('--episodes', type=int, default=5,
                        help="Number of evaluation episodes per agent")
    parser.add_argument('--max_steps', type=int, default=2000,
                        help="Maximum steps per episode")
    parser.add_argument('--kappa_thresh', type=float, default=0.05,
                        help="κ-Snap threshold for IDO/Hybrid agents")
    parser.add_argument('--agents', nargs='+',
                        choices=list(AGENT_REGISTRY.keys()),
                        default=list(AGENT_REGISTRY.keys()),
                        help="Agent types to evaluate (default: all)")
    parser.add_argument('--distribution-shift', action='store_true',
                        help="Run distribution-shift robustness test "
                             "(standard + perturbed conditions)")
    parser.add_argument('--perturbation-scale', type=float, default=0.1,
                        help="Qpos perturbation scale for distribution-shift test")
    parser.add_argument('--output_dir', default='benchmarks/results',
                        help="Directory for JSON/CSV output files")
    args = parser.parse_args()

    if args.distribution_shift:
        run_distribution_shift_test(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            agents=args.agents,
            perturbation_scale=args.perturbation_scale,
            output_dir=args.output_dir,
        )
    else:
        run_comparison(
            task=args.task,
            episodes=args.episodes,
            max_steps=args.max_steps,
            kappa_thresh=args.kappa_thresh,
            agents=args.agents,
            output_dir=args.output_dir,
        )
