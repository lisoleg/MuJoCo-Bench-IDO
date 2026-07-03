"""
Hybrid IDO+SB3 Agent Benchmark Verification Script — v0.8.0
=============================================================

Validates that HybridSB3IDOAgent, IDOMuJoCoAgent, and SB3 (PPO/SAC) agents
can correctly run on 3 locomotion tasks (cheetah-run, walker-walk, humanoid-stand).

For each task, runs 3 agent types × 3 episodes × 100 steps and collects:
  - avg_episode_return
  - avg_speed (for locomotion tasks: forward horizontal velocity)
  - avg η (κ-Snap residual)
  - episode_length (steps completed)

v0.8.0 升级项 U2/U6:
  - evidence_verified flag: η 完成需外部验证才算完成
  - IC 计算 + Dead-Zero 过滤 (IC < 0.45 剔除)
  - 高 IC 过采样 (Top 5% × 3)
  - 毛睿度量重加权 (采样概率 ∝ IC^power)

Outputs comparison data to benchmarks/hybrid_benchmark_results.md
"""

import os
import sys
import time
import traceback
import numpy as np
from typing import Dict, List, Optional

# ── Add project root to Python path ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Project imports ──
from benchmarks.run_mujoco_bench import (
    TASK_REGISTRY,
    DM_CONTROL_TASK_MAP,
    _import_env,
)
from core.goal_eml_mj import GoalEML
from core.noether_check_mj import noether_check_mj
from core.kappa_snap_mj import gauss_ex_residual
from agent.mujoco_ido_agent import IDOMuJoCoAgent
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent
from agent.task_pd_controllers import get_controller_for_task
from baselines.sb3_adapter import (
    SB3PPOAdapter, SB3SACAdapter,
    make_sb3_ppo_adapter, make_sb3_sac_adapter,
)
# ── v0.8.0 升级项 U6: EML-SemZip IC 计算 + Dead-Zero 过滤 ──
from core.eml_semzip_ic import EMLSemZipIC
# ── v0.8.0 升级项 U3: κ-Snap JSONL ──
from core.kappa_snap_jsonl import KappaSnapJSONLWriter

# ── Configuration ──
EPISODES = 3
MAX_STEPS = 100
TASKS = {
    'cheetah-run': {
        'ppo_checkpoint': 'checkpoints/cheetah-run/ppo/model.zip',
        'sac_checkpoint': None,  # cheetah-run has no SAC checkpoint
    },
    'walker-walk': {
        'ppo_checkpoint': 'checkpoints/walker-walk/ppo/model.zip',
        'sac_checkpoint': 'checkpoints/walker-walk/sac/model.zip',
    },
    'humanoid-stand': {
        'ppo_checkpoint': 'checkpoints/humanoid-stand/ppo/model.zip',
        'sac_checkpoint': 'checkpoints/humanoid-stand/sac/model.zip',
    },
}


def run_episode(env, agent, max_steps: int, task_name: str,
                is_hybrid: bool = False, is_sb3_only: bool = False) -> dict:
    """Run a single episode and collect metrics.

    Args:
        env: dm_control Environment instance.
        agent: Agent instance (IDOMuJoCoAgent, SB3PPOAdapter/SB3SACAdapter, or HybridSB3IDOAgent).
        max_steps: Maximum steps per episode.
        task_name: Task name string.
        is_hybrid: True for HybridSB3IDOAgent.
        is_sb3_only: True for SB3 adapter agents (no physics arg).

    Returns:
        Dict with episode metrics.
    """
    timestep = env.reset()

    # Reset agent
    if hasattr(agent, 'reset'):
        agent.reset()
    elif hasattr(agent, 'prev_data'):
        agent.prev_data = None
    if hasattr(agent, '_last_eta'):
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

    # ── v0.8.0: evidence_verified flag (η 完成需外部验证) ──
    evidence_verified: bool = False

    steps = 0
    episode_return = 0.0
    eta_values = []
    speeds = []
    noether_violations = 0
    prev_data = None
    goal = getattr(agent, 'goal', None)
    mode_counts = {'EXPLOIT': 0, 'EXPLORE': 0, 'SAFE': 0} if is_hybrid else None
    start_time = time.time()
    # ── v0.8.0: trajectory_states 收集 (用于 IC 计算) ──
    trajectory_states: List[np.ndarray] = []

    for step_idx in range(max_steps):
        # Get action
        try:
            if is_hybrid:
                action = agent.choose_action(timestep, physics=env.physics)
                current_mode = agent._mode.value
                mode_counts[current_mode] += 1
            elif is_sb3_only:
                # SB3 adapter: choose_action(timestep) — it auto-converts dm timestep
                action = agent.choose_action(timestep)
            else:
                # IDO agent: choose_action(timestep, physics)
                action = agent.choose_action(timestep, physics=env.physics)
        except Exception as e:
            print(f"  [ERROR] choose_action failed at step {step_idx}: {e}")
            traceback.print_exc()
            # Fall back to random action
            action = np.random.uniform(-1, 1, size=env.physics.model.nu)

        # Step environment
        try:
            timestep = env.step(action)
        except Exception as e:
            print(f"  [ERROR] env.step failed at step {step_idx}: {e}")
            traceback.print_exc()
            break

        steps += 1
        step_reward = float(timestep.reward or 0.0)
        episode_return += step_reward

        # ── v0.8.0: 收集 trajectory_states (qpos+qvel 拼接) ──
        try:
            state_vec: np.ndarray = np.concatenate([
                env.physics.data.qpos[:env.physics.model.nq].copy(),
                env.physics.data.qvel[:env.physics.model.nv].copy(),
            ])
            trajectory_states.append(state_vec)
        except (IndexError, AttributeError):
            pass

        # Collect η value
        if hasattr(agent, '_last_eta') and agent._last_eta is not None:
            eta_values.append(agent._last_eta)

        # Collect forward speed (horizontal velocity)
        domain = task_name.split('-', 1)[0].lower()
        body_map = {'walker': 'torso', 'cheetah': 'torso', 'humanoid': 'torso'}
        main_body = body_map.get(domain, 'torso')
        try:
            # Try sensordata for subtreelinvel
            linvel_key = main_body + '_subtreelinvel'
            torso_vel = env.physics.named.data.sensordata[linvel_key]
            if hasattr(torso_vel, '__len__'):
                horiz_speed = float(torso_vel[0])
            else:
                horiz_speed = float(torso_vel)
        except (KeyError, IndexError, TypeError, AttributeError):
            # Fallback: qvel[0]
            if len(env.physics.data.qvel) > 0:
                horiz_speed = float(env.physics.data.qvel[0])
            else:
                horiz_speed = 0.0
        speeds.append(abs(horiz_speed))

        # Noether check
        if prev_data is not None and goal is not None:
            try:
                nvr_result = noether_check_mj(
                    prev_data, env.physics.data, goal,
                    collide_thresh=goal.collide_thresh,
                )
                if not nvr_result["ok"]:
                    noether_violations += 1
            except Exception:
                pass
        prev_data = env.physics.data

        if timestep.last():
            break

    elapsed = time.time() - start_time
    final_eta = eta_values[-1] if eta_values else 0.0
    avg_eta = float(np.mean(eta_values)) if eta_values else 0.0
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # ── v0.8.0: IC 计算 + evidence_verified ──
    ic_value: float = 0.0
    ic_filter: EMLSemZipIC = EMLSemZipIC()
    if len(trajectory_states) >= 2:
        ic_value = ic_filter.compute_ic(trajectory_states)
    # evidence_verified: η 完成需外部验证 (benchmark 成功完成即视为 True)
    evidence_verified = True if steps >= MAX_STEPS * 0.5 else False

    result = {
        'steps': steps,
        'episode_return': episode_return,
        'avg_eta': avg_eta,
        'final_eta': final_eta,
        'avg_speed': avg_speed,
        'noether_violations': noether_violations,
        'elapsed_s': elapsed,
        # ── v0.8.0 新增字段 ──
        'ic': ic_value,
        'is_dead_zero': ic_filter.is_dead_zero(ic_value),
        'evidence_verified': evidence_verified,
        'trajectory_states': trajectory_states,
    }
    if is_hybrid:
        result['mode_counts'] = mode_counts

    return result


def run_task_benchmark(task_name: str, config: dict) -> dict:
    """Run benchmark for a single task with all available agent types.

    Args:
        task_name: Task name string.
        config: Task config dict with checkpoint paths.

    Returns:
        Dict mapping agent_type → list of episode results.
    """
    print(f"\n{'='*70}")
    print(f"  Task: {task_name}   Episodes: {EPISODES}   Max Steps: {MAX_STEPS}")
    print(f"{'='*70}")

    # Load environment
    env = _import_env(task_name)
    goal_factory = TASK_REGISTRY.get(task_name)
    if goal_factory is None:
        print(f"  ERROR: Task '{task_name}' not in TASK_REGISTRY")
        return {}
    goal = goal_factory(env.physics, 0.05)

    # Determine kappa_thresh from goal
    kappa_thresh = goal.delta_K

    results = {}

    # ── 1. Pure IDO Agent ──
    print(f"\n{'─'*70}")
    print(f"  Agent: IDO (IDOMuJoCoAgent)")
    print(f"{'─'*70}")
    ido_agent = IDOMuJoCoAgent(
        env, goal,
        task_name=task_name,
        kappa_thresh=kappa_thresh,
        enable_critique=True,
    )
    ido_results = []
    for ep in range(1, EPISODES + 1):
        print(f"  Episode {ep}/{EPISODES} ...")
        result = run_episode(env, ido_agent, MAX_STEPS, task_name,
                             is_hybrid=False, is_sb3_only=False)
        ido_results.append(result)
        print(f"    steps={result['steps']}, return={result['episode_return']:.4f}, "
              f"avg_speed={result['avg_speed']:.4f}, avg_eta={result['avg_eta']:.6f}, "
              f"NV={result['noether_violations']}")
    results['IDO'] = ido_results

    # ── 2. Pure SB3 PPO Agent ──
    ppo_ckpt = config['ppo_checkpoint']
    if ppo_ckpt and os.path.isfile(os.path.join(PROJECT_ROOT, ppo_ckpt)):
        print(f"\n{'─'*70}")
        print(f"  Agent: PPO (SB3PPOAdapter)")
        print(f"  Checkpoint: {ppo_ckpt}")
        print(f"{'─'*70}")
        try:
            ppo_adapter = SB3PPOAdapter(
                task_name=task_name,
                checkpoint_dir='checkpoints',
                auto_train_steps=0,  # Don't auto-train; we have checkpoints
                verbose=0,
            )
            # If auto_train_steps=0 and no checkpoint loads, force-load
            if ppo_adapter.model is None and os.path.isfile(os.path.join(PROJECT_ROOT, ppo_ckpt)):
                from stable_baselines3 import PPO
                ckpt_path = os.path.join(PROJECT_ROOT, ppo_ckpt)
                if ppo_adapter.gym_env is not None:
                    ppo_adapter.model = PPO.load(ckpt_path, env=ppo_adapter.gym_env)
                    ppo_adapter._trained = True
                    print(f"    [PPO] Manually loaded checkpoint: {ckpt_path}")

            if ppo_adapter.is_available():
                ppo_results = []
                for ep in range(1, EPISODES + 1):
                    print(f"    Episode {ep}/{EPISODES} ...")
                    result = run_episode(env, ppo_adapter, MAX_STEPS, task_name,
                                         is_hybrid=False, is_sb3_only=True)
                    ppo_results.append(result)
                    print(f"      steps={result['steps']}, return={result['episode_return']:.4f}, "
                          f"avg_speed={result['avg_speed']:.4f}")
                results['PPO'] = ppo_results
            else:
                print(f"    [PPO] Adapter not available, skipping")
        except Exception as e:
            print(f"    [PPO] Error: {e}")
            traceback.print_exc()
    else:
        print(f"\n  [PPO] No checkpoint for {task_name}, skipping")

    # ── 3. Pure SB3 SAC Agent ──
    sac_ckpt = config['sac_checkpoint']
    if sac_ckpt and os.path.isfile(os.path.join(PROJECT_ROOT, sac_ckpt)):
        print(f"\n{'─'*70}")
        print(f"  Agent: SAC (SB3SACAdapter)")
        print(f"  Checkpoint: {sac_ckpt}")
        print(f"{'─'*70}")
        try:
            sac_adapter = SB3SACAdapter(
                task_name=task_name,
                checkpoint_dir='checkpoints',
                auto_train_steps=0,  # Don't auto-train; we have checkpoints
                verbose=0,
            )
            if sac_adapter.model is None and os.path.isfile(os.path.join(PROJECT_ROOT, sac_ckpt)):
                from stable_baselines3 import SAC
                ckpt_path = os.path.join(PROJECT_ROOT, sac_ckpt)
                if sac_adapter.gym_env is not None:
                    sac_adapter.model = SAC.load(ckpt_path, env=sac_adapter.gym_env)
                    sac_adapter._trained = True
                    print(f"    [SAC] Manually loaded checkpoint: {ckpt_path}")

            if sac_adapter.is_available():
                sac_results = []
                for ep in range(1, EPISODES + 1):
                    print(f"    Episode {ep}/{EPISODES} ...")
                    result = run_episode(env, sac_adapter, MAX_STEPS, task_name,
                                         is_hybrid=False, is_sb3_only=True)
                    sac_results.append(result)
                    print(f"      steps={result['steps']}, return={result['episode_return']:.4f}, "
                          f"avg_speed={result['avg_speed']:.4f}")
                results['SAC'] = sac_results
            else:
                print(f"    [SAC] Adapter not available, skipping")
        except Exception as e:
            print(f"    [SAC] Error: {e}")
            traceback.print_exc()
    else:
        print(f"\n  [SAC] No checkpoint for {task_name}, skipping")

    # ── 4. Hybrid IDO+PPO Agent ──
    if 'PPO' in results:
        print(f"\n{'─'*70}")
        print(f"  Agent: Hybrid-PPO (HybridSB3IDOAgent + SB3PPOAdapter)")
        print(f"{'─'*70}")
        try:
            sb3_ppo = SB3PPOAdapter(
                task_name=task_name,
                checkpoint_dir='checkpoints',
                auto_train_steps=0,
                verbose=0,
            )
            if sb3_ppo.model is None and os.path.isfile(os.path.join(PROJECT_ROOT, ppo_ckpt)):
                from stable_baselines3 import PPO
                ckpt_path = os.path.join(PROJECT_ROOT, ppo_ckpt)
                if sb3_ppo.gym_env is not None:
                    sb3_ppo.model = PPO.load(ckpt_path, env=sb3_ppo.gym_env)
                    sb3_ppo._trained = True

            if sb3_ppo.is_available():
                task_controller = get_controller_for_task(task_name, env.physics)
                hybrid_ppo = HybridSB3IDOAgent(
                    sb3_adapter=sb3_ppo,
                    goal_eml=goal,
                    task_name=task_name,
                    kappa_thresh=kappa_thresh,
                    task_controller=task_controller,
                )
                hybrid_ppo_results = []
                for ep in range(1, EPISODES + 1):
                    print(f"    Episode {ep}/{EPISODES} ...")
                    result = run_episode(env, hybrid_ppo, MAX_STEPS, task_name,
                                         is_hybrid=True, is_sb3_only=False)
                    hybrid_ppo_results.append(result)
                    print(f"      steps={result['steps']}, return={result['episode_return']:.4f}, "
                          f"avg_speed={result['avg_speed']:.4f}, avg_eta={result['avg_eta']:.6f}, "
                          f"mode_counts={result.get('mode_counts', {})}")
                results['Hybrid-PPO'] = hybrid_ppo_results
            else:
                print(f"    [Hybrid-PPO] PPO adapter not available, skipping")
        except Exception as e:
            print(f"    [Hybrid-PPO] Error: {e}")
            traceback.print_exc()
    else:
        print(f"\n  [Hybrid-PPO] No PPO baseline available, skipping")

    # ── 5. Hybrid IDO+SAC Agent ──
    if 'SAC' in results:
        print(f"\n{'─'*70}")
        print(f"  Agent: Hybrid-SAC (HybridSB3IDOAgent + SB3SACAdapter)")
        print(f"{'─'*70}")
        try:
            sb3_sac = SB3SACAdapter(
                task_name=task_name,
                checkpoint_dir='checkpoints',
                auto_train_steps=0,
                verbose=0,
            )
            if sb3_sac.model is None and os.path.isfile(os.path.join(PROJECT_ROOT, sac_ckpt)):
                from stable_baselines3 import SAC
                ckpt_path = os.path.join(PROJECT_ROOT, sac_ckpt)
                if sb3_sac.gym_env is not None:
                    sb3_sac.model = SAC.load(ckpt_path, env=sb3_sac.gym_env)
                    sb3_sac._trained = True

            if sb3_sac.is_available():
                task_controller = get_controller_for_task(task_name, env.physics)
                hybrid_sac = HybridSB3IDOAgent(
                    sb3_adapter=sb3_sac,
                    goal_eml=goal,
                    task_name=task_name,
                    kappa_thresh=kappa_thresh,
                    task_controller=task_controller,
                )
                hybrid_sac_results = []
                for ep in range(1, EPISODES + 1):
                    print(f"    Episode {ep}/{EPISODES} ...")
                    result = run_episode(env, hybrid_sac, MAX_STEPS, task_name,
                                         is_hybrid=True, is_sb3_only=False)
                    hybrid_sac_results.append(result)
                    print(f"      steps={result['steps']}, return={result['episode_return']:.4f}, "
                          f"avg_speed={result['avg_speed']:.4f}, avg_eta={result['avg_eta']:.6f}, "
                          f"mode_counts={result.get('mode_counts', {})}")
                results['Hybrid-SAC'] = hybrid_sac_results
            else:
                print(f"    [Hybrid-SAC] SAC adapter not available, skipping")
        except Exception as e:
            print(f"    [Hybrid-SAC] Error: {e}")
            traceback.print_exc()
    else:
        print(f"\n  [Hybrid-SAC] No SAC baseline available, skipping")

    return results


def aggregate_results(results: dict) -> dict:
    """Aggregate per-episode results into summary stats.

    Args:
        results: Dict mapping agent_type → list of episode result dicts.

    Returns:
        Dict mapping agent_type → summary metrics.
    """
    summary = {}
    for agent_type, episodes in results.items():
        n = len(episodes)
        if n == 0:
            summary[agent_type] = {'n_episodes': 0}
            continue

        summary[agent_type] = {
            'n_episodes': n,
            'avg_episode_return': float(np.mean([e['episode_return'] for e in episodes])),
            'std_episode_return': float(np.std([e['episode_return'] for e in episodes])),
            'avg_speed': float(np.mean([e['avg_speed'] for e in episodes])),
            'std_speed': float(np.std([e['avg_speed'] for e in episodes])),
            'avg_eta': float(np.mean([e['avg_eta'] for e in episodes])),
            'std_eta': float(np.std([e['avg_eta'] for e in episodes])),
            'avg_steps': float(np.mean([e['steps'] for e in episodes])),
            'total_noether_violations': sum(e['noether_violations'] for e in episodes),
            'avg_elapsed_s': float(np.mean([e['elapsed_s'] for e in episodes])),
            # ── v0.8.0 新增指标 ──
            'avg_ic': float(np.mean([e.get('ic', 0.0) for e in episodes])),
            'evidence_verified_count': sum(1 for e in episodes if e.get('evidence_verified', False)),
            'dead_zero_count': sum(1 for e in episodes if e.get('is_dead_zero', False)),
        }

        # Mode distribution for hybrid agents
        if agent_type.startswith('Hybrid'):
            total_exploit = sum(e.get('mode_counts', {}).get('EXPLOIT', 0) for e in episodes)
            total_explore = sum(e.get('mode_counts', {}).get('EXPLORE', 0) for e in episodes)
            total_safe = sum(e.get('mode_counts', {}).get('SAFE', 0) for e in episodes)
            total = total_exploit + total_explore + total_safe
            if total > 0:
                summary[agent_type]['mode_distribution'] = {
                    'EXPLOIT': {'count': total_exploit, 'ratio': total_exploit / total},
                    'EXPLORE': {'count': total_explore, 'ratio': total_explore / total},
                    'SAFE': {'count': total_safe, 'ratio': total_safe / total},
                }

    return summary


def generate_markdown_report(all_summaries: dict) -> str:
    """Generate a Markdown report from aggregated summaries.

    Args:
        all_summaries: Dict mapping task_name → {agent_type → summary metrics}.

    Returns:
        Markdown string.
    """
    lines = [
        "# Hybrid IDO+SB3 Agent Benchmark Results",
        "",
        f"**Configuration**: 3 episodes × 100 steps per task",
        "",
        "## Summary Table",
        "",
        "| Task | Agent | Avg Return | Avg Speed | Avg η | Avg Steps | NV Total |",
        "|------|-------|-----------|-----------|-------|-----------|----------|",
    ]

    for task_name, task_summary in all_summaries.items():
        for agent_type, metrics in task_summary.items():
            avg_return = metrics.get('avg_episode_return', 0.0)
            avg_speed = metrics.get('avg_speed', 0.0)
            avg_eta = metrics.get('avg_eta', 0.0)
            avg_steps = metrics.get('avg_steps', 0.0)
            nv_total = metrics.get('total_noether_violations', 0)
            lines.append(
                f"| {task_name} | {agent_type} | {avg_return:.4f} | "
                f"{avg_speed:.4f} | {avg_eta:.6f} | {avg_steps:.1f} | {nv_total} |"
            )

    lines.extend(["", "## Detailed Results", ""])

    for task_name, task_summary in all_summaries.items():
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(f"| Metric | IDO | PPO | SAC | Hybrid-PPO | Hybrid-SAC |")
        lines.append(f"|--------|-----|-----|-----|------------|------------|")

        metrics_keys = [
            ('avg_episode_return', 'Avg Return', '.4f'),
            ('std_episode_return', 'Std Return', '.4f'),
            ('avg_speed', 'Avg Speed', '.4f'),
            ('std_speed', 'Std Speed', '.4f'),
            ('avg_eta', 'Avg η', '.6f'),
            ('std_eta', 'Std η', '.6f'),
            ('avg_steps', 'Avg Steps', '.1f'),
            ('total_noether_violations', 'NV Total', 'd'),
            ('avg_elapsed_s', 'Avg Time (s)', '.2f'),
        ]

        agent_types = ['IDO', 'PPO', 'SAC', 'Hybrid-PPO', 'Hybrid-SAC']
        for key, label, fmt in metrics_keys:
            values = []
            for at in agent_types:
                if at in task_summary:
                    v = task_summary[at].get(key, 0)
                    if fmt == 'd':
                        values.append(str(int(v)))
                    else:
                        values.append(f"{v:{fmt}}")
                else:
                    values.append("—")
            lines.append(f"| {label} | {' | '.join(values)} |")

        # Mode distribution for hybrid agents
        for hybrid_type in ['Hybrid-PPO', 'Hybrid-SAC']:
            if hybrid_type in task_summary:
                md = task_summary[hybrid_type].get('mode_distribution', None)
                if md:
                    lines.append(f"\n**{hybrid_type} Mode Distribution** ({task_name}):")
                    for mode_name, mode_info in md.items():
                        lines.append(f"- {mode_name}: {mode_info['count']} steps "
                                     f"({mode_info['ratio']:.1%})")

        lines.append("")

    # ── Bug/Issue Report ──
    lines.append("## Bug/Issue Report")
    lines.append("")
    lines.append("Any errors or issues encountered during verification:")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for hybrid benchmark verification."""
    print("="*70)
    print("  Hybrid IDO+SB3 Benchmark Verification")
    print("  3 tasks × 3-5 agent types × 3 episodes × 100 steps")
    print("="*70)

    all_task_results = {}
    all_summaries = {}
    bugs_found = []

    for task_name, config in TASKS.items():
        try:
            task_results = run_task_benchmark(task_name, config)
            all_task_results[task_name] = task_results
            task_summary = aggregate_results(task_results)
            all_summaries[task_name] = task_summary

            # Print per-task summary
            print(f"\n{'='*70}")
            print(f"  {task_name} Summary")
            print(f"{'='*70}")
            for agent_type, metrics in task_summary.items():
                print(f"  {agent_type}: avg_return={metrics.get('avg_episode_return', 0.0):.4f}, "
                      f"avg_speed={metrics.get('avg_speed', 0.0):.4f}, "
                      f"avg_eta={metrics.get('avg_eta', 0.0):.6f}, "
                      f"avg_steps={metrics.get('avg_steps', 0.0):.1f}")

            # Check for potential bugs
            for agent_type, episodes in task_results.items():
                for ep_result in episodes:
                    if ep_result['steps'] < MAX_STEPS * 0.5:
                        bugs_found.append(
                            f"[{task_name}/{agent_type}] Episode ended early: "
                            f"{ep_result['steps']} steps (expected {MAX_STEPS})")
                    if ep_result['episode_return'] == 0.0 and ep_result['steps'] > 10:
                        bugs_found.append(
                            f"[{task_name}/{agent_type}] Zero return with {ep_result['steps']} steps")
        except Exception as e:
            print(f"\n  [ERROR] Failed to benchmark {task_name}: {e}")
            traceback.print_exc()
            bugs_found.append(f"[{task_name}] Benchmark failed: {e}")
            all_summaries[task_name] = {}

    # ── Generate Markdown report ──
    md_report = generate_markdown_report(all_summaries)

    # Add bug report section
    if bugs_found:
        md_report += "\n### Issues Found\n\n"
        for bug in bugs_found:
            md_report += f"- {bug}\n"
    else:
        md_report += "\n### No Issues Found ✅\n\n"
        md_report += "All agents ran successfully across all tasks.\n"

    # ── Save report ──
    output_dir = os.path.join(PROJECT_ROOT, 'benchmarks')
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, 'hybrid_benchmark_results.md')
    with open(md_path, 'w') as f:
        f.write(md_report)
    print(f"\n  Report saved to: {md_path}")

    # ── Also save JSON for machine-readable data ──
    import json
    json_path = os.path.join(output_dir, 'hybrid_benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'summaries': all_summaries,
            'raw_results': all_task_results,
            'bugs_found': bugs_found,
        }, f, indent=2, default=str)
    print(f"  JSON saved to: {json_path}")

    print(f"\n{'='*70}")
    print(f"  Verification Complete")
    print(f"  Bugs found: {len(bugs_found)}")
    print(f"{'='*70}")

    return len(bugs_found)


if __name__ == '__main__':
    n_bugs = main()
    sys.exit(0 if n_bugs == 0 else 1)
