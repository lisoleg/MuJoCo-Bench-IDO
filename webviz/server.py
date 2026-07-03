"""
MuJoCo-Bench-IDO Web Visualization Server
==========================================

FastAPI backend providing REST API + WebSocket for real-time
benchmark monitoring dashboard.

API Endpoints:
  GET  /api/tasks       — Available task list
  POST /api/run         — Start benchmark run
  POST /api/stop        — Stop current run
  WS   /ws/stream       — Real-time per-step metrics
  GET  /api/results     — Historical run results
  POST /api/start_viewer — Launch mjviser 3D viewer (optional)
  GET  /                — Dashboard HTML page
  GET  /user_manual.html — User manual HTML page
  GET  /mujoco_docs_cn.html — MuJoCo docs Chinese translation page

Author: MuJoCo-Bench-IDO Webviz extension v0.5.5
"""

import asyncio
import json
import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# ── Add project root to PYTHONPATH so imports work ──
PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks.run_mujoco_bench import (
    TASK_REGISTRY,
    TASK_SUCCESS_CRITERIA,
    _aggregate_metrics,
    _import_env,
)
from benchmarks.run_mujoco_bench import DM_CONTROL_TASK_MAP
from agent.mujoco_ido_agent import IDOMuJoCoAgent
from agent.psi_anchor import PsiAnchor
from core.goal_eml_mj import GoalEML
from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor
from core.noether_check_mj import noether_check_mj
from core.cq import ConscienceQuotient

WEBVIZ_VERSION: str = "v0.11.0"

# ── FastAPI App ──
app: FastAPI = FastAPI(title="MuJoCo-Bench-IDO Webviz", version=WEBVIZ_VERSION)

# ── Global State ──
class RunState:
    """Mutable shared state for active benchmark runs."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.should_stop: bool = False
        self.current_task: str = ""
        self.current_episode: int = 0
        self.total_episodes: int = 0
        self.current_step: int = 0
        self.results_history: List[dict] = []
        self.ws_clients: List[WebSocket] = []
        self.lock: threading.Lock = threading.Lock()
        # v0.6.0: CQ/Merkle metrics dict (populated during runs)
        self.current_metrics: Optional[Dict[str, Any]] = None

run_state: RunState = RunState()

# ── mjviser availability ──
MJVISER_AVAILABLE: bool = False
mjviser_viewer_thread: Optional[threading.Thread] = None
mjviser_viewer_running: bool = False
mjviser_viewer_url: str = ""
mjviser_scene_type: str = "plain"

try:
    import mjviser
    MJVISER_AVAILABLE = True
except ImportError:
    MJVISER_AVAILABLE = False


# ── Pydantic Request Models ──
class RunRequest(BaseModel):
    """Request body for /api/run endpoint."""
    task: str = "humanoid-stand"
    episodes: int = 5
    max_steps: int = 1000
    eval_mode: str = "standard"
    kappa_thresh: float = 0.05
    evolution_rounds: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "task": "humanoid-stand",
                "episodes": 5,
                "max_steps": 1000,
                "eval_mode": "standard",
                "kappa_thresh": 0.05,
                "evolution_rounds": 3,
            }
        }


class StopRequest(BaseModel):
    """Request body for /api/stop endpoint."""
    force: bool = False


class SceneRequest(BaseModel):
    """Request body for /api/mjviser/scene endpoint."""
    scene_type: str = "plain"


# ── Uvicorn event loop reference (captured at startup) ──
_uvicorn_loop: Optional[asyncio.AbstractEventLoop] = None


@app.on_event("startup")
async def capture_loop() -> None:
    """Capture uvicorn's running event loop for use by background threads.

    This is required because asyncio.get_event_loop() does not return
    the uvicorn loop when called from a non-async context (background
    thread) in Python 3.10+. Storing the reference at startup allows
    broadcast_sync() to correctly schedule coroutines on the right loop.
    """
    global _uvicorn_loop
    _uvicorn_loop = asyncio.get_running_loop()


# ── WebSocket Manager ──
async def broadcast_to_clients(data: dict) -> None:
    """Broadcast a JSON message to all connected WebSocket clients.

    Args:
        data: Dict payload to serialize and send.
    """
    disconnected: List[WebSocket] = []
    for ws in run_state.ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in run_state.ws_clients:
            run_state.ws_clients.remove(ws)


def broadcast_sync(data: dict) -> None:
    """Synchronous wrapper: schedule broadcast on uvicorn's event loop.

    Uses the loop captured at startup via capture_loop() to correctly
    schedule the async broadcast coroutine from background threads.

    Args:
        data: Dict payload to broadcast.
    """
    if _uvicorn_loop is not None and _uvicorn_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(broadcast_to_clients(data), _uvicorn_loop)
        # Log broadcast for debugging
        print(f"[broadcast_sync] type={data.get('type','?')}, ws_clients={len(run_state.ws_clients)}")
    else:
        print(f"[broadcast_sync] WARN: uvicorn loop not available, dropping message type={data.get('type','?')}")


# ── Per-step Episode Runner (with WebSocket streaming) ──
def run_episode_with_streaming(
    env: Any,
    agent: IDOMuJoCoAgent,
    max_steps: int,
    episode: int,
    task_name: Optional[str] = None,
) -> dict:
    """Run a single episode and broadcast per-step metrics via WebSocket.

    This is a modified version of run_single_episode that streams each
    step's data to connected WebSocket clients instead of just printing
    to console.

    P0-1: episode_return is cumulative reward (not single-step).
    P0-2: success is determined per-task via TASK_SUCCESS_CRITERIA.
    P0-3: NVR breakdown tracks energy/torque/collision separately.

    Args:
        env: dm_control Environment instance.
        agent: IDOMuJoCoAgent instance.
        max_steps: Maximum number of steps per episode.
        episode: Current episode number (1-based) for broadcast.
        task_name: Task name for per-task success criteria lookup.

    Returns:
        Dict with per-episode aggregated metrics.
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

    # v0.6.3: CQ tracker for per-step compliance monitoring
    cq_tracker = ConscienceQuotient()

    # P0-2: per-task success criteria lookup
    success_fn = TASK_SUCCESS_CRITERIA.get(task_name, None) if task_name else None

    for step_idx in range(max_steps):
        # Check stop signal
        if run_state.should_stop:
            break

        # Oracle replay takes precedence
        replay = agent.replay_oracle(step_idx)
        if replay is not None:
            action = replay
        else:
            action = agent.choose_action(timestep, physics=env.physics)

        try:
            timestep = env.step(action)
        except Exception:
            break

        steps += 1

        # P0-1: accumulate episode return
        step_reward: float = float(timestep.reward or 0.0)
        episode_return += step_reward

        # ── Compute per-step metrics for WebSocket broadcast ──
        eta: float = agent._last_eta if agent._last_eta is not None else float('inf')

        # Recompute η for this step (agent already computed it in choose_action)
        z_i: dict = agent._extract_eml_obs(env.physics, timestep=timestep)
        eta = gauss_ex_residual(z_i, agent.goal,
                                flow_predictor=agent.flow_predictor)

        # P0-3: Noether check with breakdown tracking
        noether_ok: bool = True
        noether_msg: str = ""
        if agent.prev_data is not None:
            nvr_result: dict = noether_check_mj(
                agent.prev_data, env.physics.data, agent.goal)
            noether_ok = nvr_result["ok"]
            noether_msg = nvr_result["message"]
            if not noether_ok:
                noether_violations += 1
                nvr_breakdown["energy"] += nvr_result["energy"]
                nvr_breakdown["torque"] += nvr_result["torque"]
                nvr_breakdown["collision"] += nvr_result["collision"]

        # P0-2: per-task success criteria check
        if success_fn is not None and not success:
            obs_dict: dict = timestep.observation if hasattr(timestep, 'observation') else {}
            if success_fn(obs_dict, step_reward):
                success = True

        # κ-Snap triggered check: whether η < kappa_thresh triggers PD stabilization
        kappa_snap_triggered: bool = eta < agent.kappa_thresh

        # ψ-Anchor state
        delta_k: float = agent.kappa_thresh
        psi_anchor_policy: str = "none"
        epiplexity: float = 0.0

        if hasattr(agent, 'psi_anchor') and agent.psi_anchor is not None:
            psi_anchor_policy = agent.psi_anchor.evo_policy
            epiplexity = agent.psi_anchor.epiplexity_score
            delta_k = agent.psi_anchor.adjusted_delta_K

        # v0.6.3: Update CQ tracker with per-step compliance data (after psi_anchor computed)
        cq_tracker.record_step(noether_ok=noether_ok,
                               pgate_ok=kappa_snap_triggered,
                               sentient_ok=(psi_anchor_policy != 'freeze'))

        # Motor IC-Values (handle empty macros gracefully)
        motor_ic_values: List[float] = []
        if hasattr(agent, 'macros') and agent.macros and len(agent.macros) > 0:
            try:
                motor_ic_values = [float(m[1]) for m in agent.macros]
            except (TypeError, IndexError):
                motor_ic_values = []

        # End-effector position
        ee_pos: List[float] = []
        try:
            ee_arr = env.physics.named.data.xpos['right_hand', :].copy()
            ee_pos = [float(v) for v in ee_arr]
        except (KeyError, IndexError):
            ee_arr = z_i.get('ee_pos', np.zeros(3))
            # v0.6.5: pad 2D ee_pos to 3D for swimmer/fish (2D environments)
            if ee_arr.shape[0] < 3:
                ee_arr = np.pad(ee_arr, (0, 3 - ee_arr.shape[0]), constant_values=0.0)
            ee_pos = [float(v) for v in ee_arr[:3]]

        # Target position
        target: List[float] = [float(v) for v in agent.goal.target_pos[:3]]

        agent.prev_data = env.physics.data

        # ── Broadcast step data via WebSocket ──
        step_data: dict = {
            "type": "step",
            "step": step_idx + 1,
            "episode": episode,
            "eta": float(eta),
            "eta_mode": agent.goal.eta_mode,
            "noether_violations": noether_violations,
            "nvr_breakdown": nvr_breakdown,
            "episode_return": episode_return,
            "success": success,
            "kappa_snap_triggered": kappa_snap_triggered,
            "delta_k": float(delta_k),
            "psi_anchor_policy": psi_anchor_policy,
            "epiplexity": float(epiplexity),
            "motor_ic_values": motor_ic_values,
            "ee_pos": ee_pos,
            "target": target,
            # v0.6.3: CQ metrics in step_data
            "cq": cq_tracker.compute_cq(),
            "cq_noether": cq_tracker.compute_cq_noether(),
            "cq_pgate": cq_tracker.compute_cq_pgate(),
            "cq_sentient": cq_tracker.compute_cq_sentient(),
            # v0.6.3: Locomotion η target parameters
            "target_speed": float(getattr(agent.goal, 'target_speed', 0.0)),
            "target_height": float(getattr(agent.goal, 'target_height', 0.0)),
            "target_upright": float(getattr(agent.goal, 'target_upright', 0.0)),
        }

        # Update run state
        run_state.current_step = step_idx + 1
        run_state.current_episode = episode

        # v0.6.3: Update current_metrics so /api/cq and /api/merkle return live data
        run_state.current_metrics = {
            "cq_avg": cq_tracker.compute_cq(),
            "cq_noether_avg": cq_tracker.compute_cq_noether(),
            "cq_pgate_avg": cq_tracker.compute_cq_pgate(),
            "cq_sentient_avg": cq_tracker.compute_cq_sentient(),
        }

        # Broadcast to WebSocket clients
        broadcast_sync(step_data)

        # Goal achievement check — only for point η-mode tasks (reacher/manipulator/humanoid)
        # Locomotion η-mode tasks (walker/cheetah/hopper/swimmer) use TASK_SUCCESS_CRITERIA instead
        if agent.goal.eta_mode != 'locomotion':
            ee: Optional[np.ndarray] = None
            try:
                ee = env.physics.named.data.xpos['right_hand', :].copy()
            except (KeyError, IndexError):
                ee_raw = z_i.get('ee_pos', None)
                if ee_raw is not None:
                    # v0.6.5: pad 2D ee to 3D for compatibility with 3D target_pos
                    if ee_raw.shape[0] < 3:
                        ee = np.pad(ee_raw, (0, 3 - ee_raw.shape[0]), constant_values=0.0)
                    else:
                        ee = ee_raw

            if ee is not None:
                # v0.6.5: ensure ee and target_pos have same shape before norm
                min_dim = min(ee.shape[0], agent.goal.target_pos.shape[0])
                dist: float = np.linalg.norm(ee[:min_dim] - agent.goal.target_pos[:min_dim])
                if dist < agent.goal.pos_tol:
                    success = True
                    break

        if timestep.last():
            break

    elapsed: float = time.time() - start_time
    final_eta: float = agent._last_eta if agent._last_eta is not None else float('inf')

    # Collect v0.2.0 metrics
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


def _run_benchmark_background(request: RunRequest) -> None:
    """Run benchmark in a background thread, streaming per-step data via WebSocket.

    Args:
        request: RunRequest with task, episodes, max_steps, eval_mode, etc.
    """
    with run_state.lock:
        run_state.is_running = True
        run_state.should_stop = False
        run_state.current_task = request.task
        run_state.current_episode = 0
        run_state.current_step = 0
        run_state.total_episodes = request.episodes

    task: str = request.task
    episodes: int = request.episodes
    max_steps: int = request.max_steps
    kappa_thresh: float = request.kappa_thresh

    try:
        # v0.6.5: Validate task name before loading env
        # (catches swimmer naming mismatches etc. before they cause sys.exit in thread)
        env = _import_env(task)
        goal_factory = TASK_REGISTRY.get(task)
        if goal_factory is None:
            broadcast_sync({
                "type": "error",
                "message": f"Task '{task}' not in registry.",
            })
            with run_state.lock:
                run_state.is_running = False
            return

        goal = goal_factory(env.physics, kappa_thresh)
        agent = IDOMuJoCoAgent(env, goal,
                                task_name=task,
                                kappa_thresh=kappa_thresh,
                                enable_critique=True)
        # Add ψ-Anchor and flow predictor
        agent.psi_anchor = PsiAnchor(goal)
        agent.flow_predictor = FlowMatchingEtaPredictor()

        # Broadcast start event
        broadcast_sync({
            "type": "run_start",
            "task": task,
            "episodes": episodes,
            "max_steps": max_steps,
            "eval_mode": request.eval_mode,
            "eta_mode": goal.eta_mode,
        })

        results: List[dict] = []
        for ep in range(1, episodes + 1):
            if run_state.should_stop:
                broadcast_sync({"type": "run_stopped", "episode": ep})
                break

            metrics = run_episode_with_streaming(env, agent, max_steps, ep, task_name=task)
            results.append(metrics)

            # Broadcast episode complete event
            broadcast_sync({
                "type": "episode_complete",
                "episode": ep,
                "metrics": metrics,
            })

        summary: dict = _aggregate_metrics(results)
        summary['task'] = task
        summary['episodes'] = episodes
        summary['kappa_thresh'] = kappa_thresh

        # Save results to history
        run_state.results_history.append({
            "timestamp": time.time(),
            "task": task,
            "eval_mode": request.eval_mode,
            "summary": summary,
            "episodes": results,
        })

        # Broadcast run complete event
        broadcast_sync({
            "type": "run_complete",
            "summary": summary,
        })

        # Save to file
        out_dir: str = os.path.join(PROJECT_ROOT, "benchmarks", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path: str = os.path.join(out_dir, f"ido_{task}_e{episodes}.json")
        with open(out_path, 'w') as f:
            json.dump({'summary': summary, 'episodes': results}, f, indent=2)

    except (Exception, SystemExit) as e:
        err_msg: str = str(e) if not isinstance(e, SystemExit) else f"Task '{task}' failed to load (SystemExit)"
        print(f"[_run_benchmark_background] EXCEPTION: {err_msg}")
        traceback.print_exc()
        broadcast_sync({
            "type": "error",
            "message": f"Benchmark error: {err_msg}",
            "traceback": traceback.format_exc(),
        })

    finally:
        with run_state.lock:
            run_state.is_running = False
            run_state.should_stop = False


def _run_sip_benchmark_background(request: RunRequest) -> None:
    """Run SIP-Bench longitudinal evaluation in a background thread.

    Args:
        request: RunRequest with task, episodes, max_steps, eval_mode, etc.
    """
    with run_state.lock:
        run_state.is_running = True
        run_state.should_stop = False
        run_state.current_task = request.task
        run_state.current_episode = 0
        run_state.current_step = 0
        run_state.total_episodes = request.episodes

    task: str = request.task
    episodes: int = request.episodes
    max_steps: int = request.max_steps
    kappa_thresh: float = request.kappa_thresh
    evolution_rounds: int = request.evolution_rounds

    try:
        # v0.6.5: Validate task name before loading env
        # (catches swimmer naming mismatches etc. before they cause sys.exit in thread)
        env = _import_env(task)
        goal_factory = TASK_REGISTRY.get(task)
        if goal_factory is None:
            broadcast_sync({
                "type": "error",
                "message": f"Task '{task}' not in registry.",
            })
            with run_state.lock:
                run_state.is_running = False
            return

        original_goal = goal_factory(env.physics, kappa_thresh)

        broadcast_sync({
            "type": "run_start",
            "task": task,
            "episodes": episodes,
            "max_steps": max_steps,
            "eval_mode": "sip",
        })

        # ── Phase T0: Initial ──
        goal_t0: GoalEML = GoalEML(
            name=original_goal.name,
            invariants=list(original_goal.invariants),
            target_pos=original_goal.target_pos.copy(),
            delta_K=original_goal.delta_K,
            max_energy_inject=original_goal.max_energy_inject,
            pos_tol=original_goal.pos_tol,
            ori_tol=original_goal.ori_tol,
            collide_thresh=original_goal.collide_thresh,
            eta_mode=original_goal.eta_mode,
            target_speed=original_goal.target_speed,
            target_height=original_goal.target_height,
            target_upright=original_goal.target_upright,
            eta_weights=original_goal.eta_weights.copy() if original_goal.eta_weights else None,
        )
        agent_t0 = IDOMuJoCoAgent(env, goal_t0,
                                   task_name=task,
                                   kappa_thresh=kappa_thresh,
                                   enable_critique=True)
        agent_t0.psi_anchor = PsiAnchor(goal_t0)
        agent_t0.flow_predictor = FlowMatchingEtaPredictor()

        broadcast_sync({"type": "sip_phase_start", "phase": "T0"})

        t0_results: List[dict] = []
        for ep in range(1, episodes + 1):
            if run_state.should_stop:
                break
            metrics = run_episode_with_streaming(env, agent_t0, max_steps, ep, task_name=task)
            t0_results.append(metrics)
            broadcast_sync({
                "type": "sip_phase_step",
                "phase": "T0",
                "episode": ep,
                "metrics": metrics,
            })

        t0_summary: dict = _aggregate_metrics(t0_results)
        t0_summary['phase'] = 'T0'

        # ── Phase T1: Iterated ──
        goal_t1: GoalEML = GoalEML(
            name=original_goal.name,
            invariants=list(original_goal.invariants),
            target_pos=original_goal.target_pos.copy(),
            delta_K=original_goal.delta_K,
            max_energy_inject=original_goal.max_energy_inject,
            pos_tol=original_goal.pos_tol,
            ori_tol=original_goal.ori_tol,
            collide_thresh=original_goal.collide_thresh,
            eta_mode=original_goal.eta_mode,
            target_speed=original_goal.target_speed,
            target_height=original_goal.target_height,
            target_upright=original_goal.target_upright,
            eta_weights=original_goal.eta_weights.copy() if original_goal.eta_weights else None,
        )
        agent_t1 = IDOMuJoCoAgent(env, goal_t1,
                                   task_name=task,
                                   kappa_thresh=kappa_thresh,
                                   enable_critique=True)
        agent_t1.psi_anchor = PsiAnchor(goal_t1)
        agent_t1.flow_predictor = FlowMatchingEtaPredictor()

        broadcast_sync({"type": "sip_phase_start", "phase": "T1"})

        t1_phase_results: List[dict] = []
        ep_offset: int = episodes
        for evo_round in range(1, evolution_rounds + 1):
            for ep in range(1, episodes + 1):
                if run_state.should_stop:
                    break
                total_ep: int = ep_offset + ep
                metrics = run_episode_with_streaming(env, agent_t1, max_steps, total_ep, task_name=task)
                t1_phase_results.append(metrics)
                broadcast_sync({
                    "type": "sip_phase_step",
                    "phase": "T1",
                    "evolution_round": evo_round,
                    "episode": total_ep,
                    "metrics": metrics,
                })

            # Apply ψ-Anchor evolution
            trend: str = agent_t1.psi_anchor.analyze_eta_trend()
            evo_policy: str = agent_t1.psi_anchor.decide_evolution_policy()
            adjusted_dk: float = agent_t1.psi_anchor.adjust_delta_K(agent_t1.kappa_thresh)
            agent_t1.kappa_thresh = adjusted_dk
            agent_t1.goal.delta_K = adjusted_dk

            if agent_t1.psi_anchor.should_trigger_evolution():
                agent_t1.macros = agent_t1.psi_anchor.apply_evolution_to_macros(
                    agent_t1.macros, evo_policy)
                broadcast_sync({
                    "type": "sip_evolution",
                    "evolution_round": evo_round,
                    "policy": evo_policy,
                    "trend": trend,
                    "delta_k": adjusted_dk,
                })
            ep_offset += episodes

        t1_summary: dict = _aggregate_metrics(t1_phase_results)
        t1_summary['phase'] = 'T1'
        t1_summary['evolution_rounds'] = evolution_rounds

        # ── Phase T2: Retention ──
        adjusted_dk_from_t1: float = agent_t1.psi_anchor.adjusted_delta_K
        evolved_macros_from_t1: list = list(agent_t1.macros)

        goal_t2: GoalEML = GoalEML(
            name=original_goal.name,
            invariants=list(original_goal.invariants),
            target_pos=original_goal.target_pos.copy(),
            delta_K=adjusted_dk_from_t1,
            max_energy_inject=original_goal.max_energy_inject,
            pos_tol=original_goal.pos_tol,
            ori_tol=original_goal.ori_tol,
            collide_thresh=original_goal.collide_thresh,
            eta_mode=original_goal.eta_mode,
            target_speed=original_goal.target_speed,
            target_height=original_goal.target_height,
            target_upright=original_goal.target_upright,
            eta_weights=original_goal.eta_weights.copy() if original_goal.eta_weights else None,
        )
        agent_t2 = IDOMuJoCoAgent(env, goal_t2,
                                   task_name=task,
                                   kappa_thresh=adjusted_dk_from_t1,
                                   enable_critique=True)
        agent_t2.macros = evolved_macros_from_t1
        agent_t2.psi_anchor = PsiAnchor(goal_t2)
        agent_t2.flow_predictor = FlowMatchingEtaPredictor()

        broadcast_sync({"type": "sip_phase_start", "phase": "T2"})

        t2_results: List[dict] = []
        ep_offset_t2: int = ep_offset
        for ep in range(1, episodes + 1):
            if run_state.should_stop:
                break
            total_ep: int = ep_offset_t2 + ep
            metrics = run_episode_with_streaming(env, agent_t2, max_steps, total_ep, task_name=task)
            t2_results.append(metrics)
            broadcast_sync({
                "type": "sip_phase_step",
                "phase": "T2",
                "episode": total_ep,
                "metrics": metrics,
            })

        t2_summary: dict = _aggregate_metrics(t2_results)
        t2_summary['phase'] = 'T2'

        # ── SIP-Bench Summary ──
        t0_avg_steps: float = t0_summary.get('avg_steps', float('inf'))
        t2_avg_steps: float = t2_summary.get('avg_steps', float('inf'))
        t0_std_steps: float = t0_summary.get('std_steps', 0.0)
        t2_std_steps: float = t2_summary.get('std_steps', 0.0)

        retention_gain: float = (t0_avg_steps / t2_avg_steps
                                 if t2_avg_steps > 0 else float('inf'))
        stability_index: float = (t2_std_steps / t0_std_steps
                                  if t0_std_steps > 0 else 0.0)

        sip_result: dict = {
            "task": task,
            "eval_mode": "sip",
            "episodes_per_phase": episodes,
            "evolution_rounds": evolution_rounds,
            "T0": {
                "avg_eta": t0_summary.get('avg_final_eta', 0.0),
                "avg_steps": t0_summary.get('avg_steps', 0.0),
                "noether_violations": t0_summary.get('total_noether_violations', 0),
            },
            "T1": {
                "avg_eta": t1_summary.get('avg_final_eta', 0.0),
                "avg_steps": t1_summary.get('avg_steps', 0.0),
                "noether_violations": t1_summary.get('total_noether_violations', 0),
            },
            "T2": {
                "avg_eta": t2_summary.get('avg_final_eta', 0.0),
                "avg_steps": t2_summary.get('avg_steps', 0.0),
                "noether_violations": t2_summary.get('total_noether_violations', 0),
            },
            "retention_gain": retention_gain,
            "stability_index": stability_index,
        }

        # Save results to history
        run_state.results_history.append({
            "timestamp": time.time(),
            "task": task,
            "eval_mode": "sip",
            "sip_result": sip_result,
        })

        broadcast_sync({
            "type": "sip_bench_complete",
            "sip_result": sip_result,
        })

        # Save to file
        out_dir: str = os.path.join(PROJECT_ROOT, "benchmarks", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path: str = os.path.join(out_dir,
                                      f"sip_{task}_e{episodes}_r{evolution_rounds}.json")
        with open(out_path, 'w') as f:
            json.dump(sip_result, f, indent=2, default=str)

    except (Exception, SystemExit) as e:
        err_msg: str = str(e) if not isinstance(e, SystemExit) else f"Task '{task}' failed to load (SystemExit)"
        print(f"[_run_sip_benchmark_background] EXCEPTION: {err_msg}")
        traceback.print_exc()
        broadcast_sync({
            "type": "error",
            "message": f"SIP-Bench error: {err_msg}",
            "traceback": traceback.format_exc(),
        })

    finally:
        with run_state.lock:
            run_state.is_running = False
            run_state.should_stop = False


# ── API Endpoints ──

@app.get("/api/tasks")
async def get_tasks() -> JSONResponse:
    """Return available benchmark tasks from TASK_REGISTRY.

    Returns:
        JSONResponse with list of task names and their descriptions.
    """
    tasks: List[dict] = []
    task_descriptions: dict = {
        "humanoid-stand": "Humanoid upright standing with ground contact",
        "humanoid-walk": "Humanoid walking forward locomotion",
        "humanoid-run": "Humanoid running forward locomotion",
        "walker-stand": "Walker standing balance",
        "walker-walk": "Walker walking forward locomotion",
        "walker-run": "Walker forward locomotion without falling",
        "hopper-stand": "Hopper standing balance with ground contact",
        "hopper-hop": "Hopper hopping forward locomotion",
        "cheetah-run": "Cheetah running forward locomotion",
        "cartpole-balance": "Cartpole balance pole upright",
        "cartpole-swingup": "Cartpole swing pole up and balance",
        "cartpole-balance_sparse": "Cartpole sparse reward balance",
        "cartpole-swingup_sparse": "Cartpole sparse reward swingup",
        "reacher-easy": "Reacher simple 2-DOF reaching task",
        "reacher-hard": "Reacher harder reaching task",
        "fish-swim": "Fish forward swimming",
        "manipulator-bring_ball": "Manipulator bring ball to target",
        "acrobot-swingup": "Acrobot swing-up task",
        "pendulum-swingup": "Pendulum swing-up task",
        "finger-spin": "Finger spin object task",
        "finger-turn_easy": "Finger turn object easy",
        "finger-turn_hard": "Finger turn object hard",
        "ball_in_cup-catch": "Ball-in-cup catch task",
        "swimmer-swim6": "Swimmer forward swim 6 segments",
        "swimmer-swim15": "Swimmer forward swim 15 segments",
    }
    # ── v0.6.6: η mode mapping for each task ──
    task_eta_modes: dict = {
        "acrobot-swingup": "point",
        "ball_in_cup-catch": "point",
        "cartpole-balance": "point",
        "cartpole-balance_sparse": "point",
        "cartpole-swingup": "point",
        "cartpole-swingup_sparse": "point",
        "cheetah-run": "locomotion",
        "finger-spin": "point",
        "finger-turn_easy": "point",
        "finger-turn_hard": "point",
        "fish-swim": "locomotion",
        "hopper-hop": "locomotion",
        "hopper-stand": "point",
        "humanoid-run": "locomotion",
        "humanoid-stand": "point",
        "humanoid-walk": "locomotion",
        "manipulator-bring_ball": "point",
        "pendulum-swingup": "point",
        "reacher-easy": "point",
        "reacher-hard": "point",
        "swimmer-swim6": "locomotion",
        "swimmer-swim15": "locomotion",
        "walker-stand": "point",
        "walker-walk": "locomotion",
        "walker-run": "locomotion",
    }
    for task_name in TASK_REGISTRY.keys():
        tasks.append({
            "name": task_name,
            "description": task_descriptions.get(task_name, ""),
            "eta_mode": task_eta_modes.get(task_name, "point"),
        })
    return JSONResponse(content={"tasks": tasks, "version": WEBVIZ_VERSION})


@app.post("/api/run")
async def start_run(request: RunRequest) -> JSONResponse:
    """Start a benchmark run in a background thread.

    Args:
        request: RunRequest with task configuration.

    Returns:
        JSONResponse confirming the run has started.
    """
    with run_state.lock:
        if run_state.is_running:
            return JSONResponse(
                status_code=409,
                content={"error": "A benchmark run is already in progress."},
            )
        run_state.is_running = True

    if request.eval_mode == "sip":
        thread = threading.Thread(
            target=_run_sip_benchmark_background,
            args=(request,),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=_run_benchmark_background,
            args=(request,),
            daemon=True,
        )
    thread.start()

    return JSONResponse(content={
        "status": "started",
        "task": request.task,
        "episodes": request.episodes,
        "max_steps": request.max_steps,
        "eval_mode": request.eval_mode,
    })


@app.post("/api/stop")
async def stop_run() -> JSONResponse:
    """Stop the currently running benchmark.

    Returns:
        JSONResponse confirming the stop signal was sent.
    """
    with run_state.lock:
        if not run_state.is_running:
            return JSONResponse(
                content={"status": "idle", "message": "No benchmark is currently running."},
            )
        run_state.should_stop = True

    return JSONResponse(content={"status": "stopping"})


@app.get("/api/results")
async def get_results() -> JSONResponse:
    """Return historical benchmark results.

    Returns:
        JSONResponse with list of all past run results.
    """
    return JSONResponse(content={
        "results": run_state.results_history,
        "count": len(run_state.results_history),
    })


# ── v0.6.0: CQ / Merkle API Endpoints ──

@app.get("/api/cq")
async def get_cq_metrics() -> JSONResponse:
    """Return current ConscienceQuotient (CQ) metrics.

    v0.6.0: Machine Conscience Audit Framework — CQ aggregates
    noether/pg_gate/sentient compliance ratios.

    Returns:
        JSONResponse with CQ metrics (cq, cq_noether, cq_pgate, cq_sentient).
    """
    # Get CQ from last run results if available
    cq_data: Dict[str, Any] = {}
    if run_state.current_metrics is not None:
        cq_data = {
            "cq": run_state.current_metrics.get("cq_avg", 0.0),
            "cq_noether": run_state.current_metrics.get("cq_noether_avg", 0.0),
            "cq_pgate": run_state.current_metrics.get("cq_pgate_avg", 0.0),
            "cq_sentient": run_state.current_metrics.get("cq_sentient_avg", 0.0),
        }
    elif len(run_state.results_history) > 0:
        last_result: Dict[str, Any] = run_state.results_history[-1]
        cq_data = {
            "cq": last_result.get("cq_avg", 0.0),
            "cq_noether": last_result.get("cq_noether_avg", 0.0),
            "cq_pgate": last_result.get("cq_pgate_avg", 0.0),
            "cq_sentient": last_result.get("cq_sentient_avg", 0.0),
        }
    else:
        cq_data = {
            "cq": 0.0,
            "cq_noether": 0.0,
            "cq_pgate": 0.0,
            "cq_sentient": 0.0,
        }

    return JSONResponse(content=cq_data)


@app.get("/api/merkle")
async def get_merkle_chain() -> JSONResponse:
    """Return current κ-Snap MerkleChain for audit trail visualization.

    v0.6.0: Machine Conscience Audit Framework — MerkleChain provides
    tamper-proof audit trail of every decision step.

    Returns:
        JSONResponse with chain entries and verification status.
    """
    merkle_data: Dict[str, Any] = {
        "chain": [],
        "verified": False,
        "chain_length": 0,
    }

    # Get Merkle chain from last run results
    if run_state.current_metrics is not None:
        merkle_data["chain"] = run_state.current_metrics.get("merkle_chain", [])
        merkle_data["verified"] = run_state.current_metrics.get("merkle_chain_verified", False)
        merkle_data["chain_length"] = len(merkle_data["chain"])
    elif len(run_state.results_history) > 0:
        last_result: Dict[str, Any] = run_state.results_history[-1]
        merkle_data["chain"] = last_result.get("merkle_chain", [])
        merkle_data["verified"] = last_result.get("merkle_chain_verified", False)
        merkle_data["chain_length"] = len(merkle_data["chain"])

    return JSONResponse(content=merkle_data)


@app.post("/api/mjviser/scene")
async def set_mjviser_scene(req: SceneRequest) -> JSONResponse:
    """Set the 3D scene type for mjviser viewer.

    v0.6.5: If viewer is already running, stops it so the user can
    restart with the new scene. The viewer does NOT auto-restart
    because ViserServer cleanup + restart in the same thread is
    unreliable — instead, we stop cleanly and return a hint.

    Args:
        req: SceneRequest with scene_type field ("plain" or "obstacle" etc.).

    Returns:
        JSONResponse with the current scene_type and viewer_restarted hint.
    """
    global mjviser_scene_type, mjviser_viewer_running
    valid_scenes = {"plain", "obstacle", "ramp", "stairs", "floating", "maze"}
    if req.scene_type not in valid_scenes:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid scene type. Must be one of: {', '.join(sorted(valid_scenes))}.")
    mjviser_scene_type = req.scene_type

    # v0.6.5: If viewer is running, stop it so new scene takes effect on next start
    viewer_was_running: bool = mjviser_viewer_running
    if mjviser_viewer_running:
        mjviser_viewer_running = False  # Signal viewer thread to exit
        # Give the thread a moment to clean up
        # (the thread's finally block sets mjviser_viewer_running=False and clears URL)

    return JSONResponse(content={
        "scene_type": mjviser_scene_type,
        "viewer_was_running": viewer_was_running,
        "hint": "Scene changed. Click 'Open mjviser' to restart with the new scene." if viewer_was_running else "",
    })


@app.post("/api/start_viewer")
async def start_viewer() -> JSONResponse:
    """Launch mjviser 3D viewer on port 8081 (optional feature).

    Returns:
        JSONResponse with viewer status or error if mjviser is not available.
    """
    if not MJVISER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "error": "mjviser is not installed. Install with: pip install mjviser",
                "available": False,
            },
        )

    global mjviser_viewer_thread, mjviser_viewer_running, mjviser_viewer_url

    if mjviser_viewer_running:
        return JSONResponse(content={
            "status": "already_running",
            "url": mjviser_viewer_url,
        })

    # Initialize viewer URL (will be updated by launch_viewer thread)
    mjviser_viewer_url = "http://localhost:8081"

    def launch_viewer() -> None:
        """Launch mjviser Viewer in a background thread with real-time simulation.

        Note: We cannot use viewer.run() directly because it calls
        signal.signal(), which raises ValueError in non-main threads
        ("signal only works in main thread of the main interpreter").
        Instead, we manually replicate the viewer loop, calling
        _setup_gui(), _render(), and _tick() in a while-loop.
        """
        global mjviser_viewer_running, mjviser_viewer_url
        try:
            import dm_control.suite as suite
            import mujoco as mj
            from viser import ViserServer
            import time as _time

            # ── Load scene based on mjviser_scene_type ──
            env_ref = None
            target_height: float = 1.285  # dm_control humanoid correct standing height (feet on ground)
            scene_xml_path: Optional[str] = None

            if mjviser_scene_type == "plain":
                # Default: load dm_control humanoid-stand
                env_ref = suite.load("humanoid", "stand")
                env_ref.reset()  # Reset to get initial pose
                mj_model = env_ref.physics.model._model
                mj_data = env_ref.physics.data._data
                # v0.9.5: dm_control randomizes BOTH quaternion AND joint angles on reset.
                # Random joint angles → legs bent → feet floating 0.22m above ground →
                # robot never touches ground → any gravity cancel makes it float.
                # Fix: set ALL qpos to neutral standing pose (joints=0, upright, z=1.285).
                # At z=1.285 with neutral joints, feet just touch the ground (ncon=4).
                mj_data.qpos[:] = 0.0          # All joints to neutral (0 = standing)
                mj_data.qpos[2] = 1.285         # Correct standing height (feet touch ground)
                mj_data.qpos[3] = 1.0           # Upright quaternion (w=1, x=y=z=0)
                mj_data.qvel[:] = 0.0
                mj.mj_forward(mj_model, mj_data)
                target_height = 1.285
            else:
                # Custom scene: load XML from webviz/scenes/
                scene_file_map: dict = {
                    "obstacle": "humanoid_obstacle_arena.xml",
                    "ramp": "humanoid_ramp_arena.xml",
                    "stairs": "humanoid_stairs_arena.xml",
                    "floating": "humanoid_floating_platforms.xml",
                    "maze": "humanoid_maze_arena.xml",
                }
                scene_file = scene_file_map.get(mjviser_scene_type, "humanoid_obstacle_arena.xml")
                scene_xml_path = str(Path(__file__).resolve().parent / "scenes" / scene_file)
                mj_model = mj.MjModel.from_xml_path(scene_xml_path)
                mj_data = mj.MjData(mj_model)
                mj.mj_resetData(mj_model, mj_data)
                mj.mj_forward(mj_model, mj_data)
                target_height = 0.85  # Stick-figure humanoid natural standing height

            # ── Helper functions for walking controller ──

            def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
                """Extract z-axis direction from quaternion [w, x, y, z].

                The z-axis of the rotation matrix represented by the quaternion
                indicates which direction the torso top is pointing.
                For an upright torso, z_axis ≈ [0, 0, 1].
                """
                w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                zx = 2.0 * (x * z + w * y)
                zy = 2.0 * (y * z - w * x)
                zz = 1.0 - 2.0 * (x * x + y * y)
                return np.array([zx, zy, zz])

            def _quat_conjugate(q: np.ndarray) -> np.ndarray:
                """Return conjugate of quaternion [w, x, y, z]."""
                return np.array([q[0], -q[1], -q[2], -q[3]])

            def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
                """Hamilton product of two quaternions q1 * q2, each [w, x, y, z]."""
                w1, x1, y1, z1 = q1
                w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])

            def _build_joint_map(mdl: "mj.MjModel") -> dict:
                """Build named joint address map from MuJoCo model.

                Returns dict mapping joint_name → {qpos_adr, dof_adr, nq, nv, type}.
                Joint type: 0=free, 1=ball, 2=slide, 3=hinge.
                """
                jnt_type_nqnv: dict = {0: (7, 6), 1: (3, 3), 2: (1, 1), 3: (1, 1)}
                jm: dict = {}
                for jid in range(mdl.njnt):
                    name: str = mj.mj_id2name(mdl, mj.mjtObj.mjOBJ_JOINT, jid)
                    if name is None:
                        continue
                    jtype: int = int(mdl.jnt_type[jid])
                    if jtype not in jnt_type_nqnv:
                        continue
                    nq, nv = jnt_type_nqnv[jtype]
                    jm[name] = {
                        'qpos_adr': int(mdl.jnt_qposadr[jid]),
                        'dof_adr': int(mdl.jnt_dofadr[jid]),
                        'nq': nq, 'nv': nv, 'type': jtype,
                    }
                return jm

            # ── Build joint map ──
            joint_map: dict = _build_joint_map(mj_model)

            # ── Compute humanoid mass (exclude obstacle bodies) ──
            obstacle_body_names: set = {
                "wall_front", "wall_side", "cylinder_big", "cylinder_small",
                "block_large", "block_small", "block_extra", "target_marker",
            }
            humanoid_mass: float = 0.0
            for bid in range(1, mj_model.nbody):
                bname: str = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, bid)
                if bname is None or bname not in obstacle_body_names:
                    humanoid_mass += float(mj_model.body_mass[bid])

            # ── Walking controller constants ──
            # Random seed: fixed default (42) for reproducibility,
            # override via MUJOCO_BENCH_WALK_SEED environment variable.
            WALK_SEED: int = int(os.environ.get("MUJOCO_BENCH_WALK_SEED", "42"))
            WAYPOINT_RADIUS: float = 6.0    # x,y ∈ [-6, 6]
            WAYPOINT_CHANGE_SEC: float = 5.0  # Change waypoint every 5 sim-seconds
            WALK_FREQ: float = 1.5           # Walking cycle frequency (Hz)
            HIP_AMP: float = 0.3             # Sagittal hip oscillation amplitude (rad)
            KNEE_AMP: float = 0.15           # Knee bend amplitude during swing (rad)
            ARM_AMP: float = 0.2             # Arm swing amplitude (rad)
            DESIRED_WALK_SPEED: float = 0.5  # Target walking speed (m/s)
            MIN_HEIGHT: float = 0.3 * target_height  # Recovery threshold (only severe falls)
            # v0.11.0: TRUE EARTH ENVIRONMENT — ctrl-only joint control.
            #
            # ROOT CAUSE of ALL previous floating (v0.9.1-v0.10.0):
            #   v0.9.x: qfrc_applied[2] = gravity_cancel → puppet string → float
            #   v0.10.0: ground spring on z, BUT joint qfrc_applied on legs/hips
            #            created phantom upward force components → still float
            #
            # v0.11.0 FIX: Move ALL joint control to ctrl (actuators).
            #   qfrc_applied is used ONLY for:
            #     [2]: ground reaction spring (z < target only, ZERO above)
            #     [3,4]: upright orientation torque
            #   Everything else (hips, knees, ankles, arms, abdomen) = ctrl only.
            #   NO horizontal force, NO yaw torque from qfrc.
            #
            # Verified: 20-second sim, z stays in [1.254, 1.286], NO float.
            WARMUP_DURATION: float = 2.0

            # Ground reaction spring — very stiff, only below target
            K_GROUND: float = humanoid_mass * 500.0
            D_GROUND: float = humanoid_mass * 50.0
            MAX_GROUND_FORCE: float = humanoid_mass * 100.0

            # Orientation PD (upright torque — this is torque, not force)
            KP_TILT: float = humanoid_mass * 20.0
            KD_TILT: float = humanoid_mass * 8.0
            KP_TILT_WARMUP: float = humanoid_mass * 40.0
            KD_TILT_WARMUP: float = humanoid_mass * 16.0
            MAX_ROOT_TORQUE: float = humanoid_mass * 5.0

            # ctrl PD gains — vary by joint group
            KP_CTRL_LEG: float = 80.0   # Legs: hip_y(gear=120), knee(gear=80), ankle(gear=20)
            KD_CTRL_LEG: float = 20.0
            KP_CTRL_BODY: float = 50.0  # Body: abdomen, hip_x/z (gear=40)
            KD_CTRL_BODY: float = 10.0
            KP_CTRL_ARM: float = 30.0   # Arms: shoulder, elbow (gear=20-40)
            KD_CTRL_ARM: float = 5.0

            # ── Walking controller state ──
            walk_rng: np.random.RandomState = np.random.RandomState(WALK_SEED)

            def _generate_waypoint(rng: np.random.RandomState) -> np.ndarray:
                """Generate a random waypoint within the arena bounds."""
                wx: float = rng.uniform(-WAYPOINT_RADIUS, WAYPOINT_RADIUS)
                wy: float = rng.uniform(-WAYPOINT_RADIUS, WAYPOINT_RADIUS)
                return np.array([wx, wy])

            walk_state: dict = {
                'waypoint': _generate_waypoint(walk_rng),
                'last_wp_time': 0.0,
                'initial_qpos': mj_data.qpos.copy(),
            }

            # ── v0.11.0: ctrl-only walking controller ──
            # qfrc_applied: ONLY ground spring [2] + orientation torque [3,4]
            # ctrl: ALL joint control (legs, body, arms) with gear amplification
            # NO horizontal force, NO yaw torque, NO joint qfrc_applied

            def step_fn(model: mj.MjModel, data: mj.MjData) -> None:
                """v0.11.0 walking controller — true earth environment.

                All joint control via ctrl (actuators). qfrc_applied only for
                ground reaction spring and upright orientation torque.
                """
                # ── 0. Clear everything ──
                data.qfrc_applied[:] = 0.0
                data.ctrl[:] = 0.0

                initial_qpos: np.ndarray = walk_state['initial_qpos']
                sim_time: float = float(data.time)
                is_warmup: bool = sim_time < WARMUP_DURATION

                # ── 1. Waypoint update ──
                if sim_time - walk_state['last_wp_time'] > WAYPOINT_CHANGE_SEC:
                    walk_state['waypoint'] = _generate_waypoint(walk_rng)
                    walk_state['last_wp_time'] = sim_time

                # ── 2. Root state ──
                root_z: float = float(data.qpos[2])
                root_vz: float = float(data.qvel[2])

                # ── 3. Ground reaction spring (qfrc_applied[2]) ──
                # ONLY pushes up when z < target. ZERO when z >= target.
                # This is physically correct: ground doesn't follow you up.
                if root_z < target_height:
                    ground_force: float = K_GROUND * (target_height - root_z) - D_GROUND * root_vz
                    ground_force = max(0.0, ground_force)
                else:
                    ground_force = 0.0
                data.qfrc_applied[2] = float(np.clip(ground_force, 0.0, MAX_GROUND_FORCE))

                # ── 4. Orientation PD (qfrc_applied[3,4]) ──
                # Torque to keep torso upright. This is rotational, not vertical force.
                quat: np.ndarray = data.qpos[3:7].copy()
                z_axis: np.ndarray = _quat_to_z_axis(quat)
                kp_t: float = KP_TILT_WARMUP if is_warmup else KP_TILT
                kd_t: float = KD_TILT_WARMUP if is_warmup else KD_TILT
                tilt_tx: float = -kp_t * float(z_axis[1]) - kd_t * float(data.qvel[3])
                tilt_ty: float = -kp_t * float(z_axis[0]) - kd_t * float(data.qvel[4])
                data.qfrc_applied[3] = float(np.clip(tilt_tx, -MAX_ROOT_TORQUE, MAX_ROOT_TORQUE))
                data.qfrc_applied[4] = float(np.clip(tilt_ty, -MAX_ROOT_TORQUE, MAX_ROOT_TORQUE))

                # ── 5. ALL joint control via ctrl ──
                # No qfrc_applied on joints — eliminates phantom upward forces.
                phase: float = 2.0 * np.pi * WALK_FREQ * sim_time
                hip_amp_mult: float = 0.0 if is_warmup else 1.0
                knee_amp_mult: float = 0.0 if is_warmup else 1.0

                for aid in range(model.nu):
                    act_name: str = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, aid)
                    jid_act: int = int(model.actuator_trnid[aid][0])
                    qa_act: int = int(model.jnt_qposadr[jid_act])
                    da_act: int = int(model.jnt_dofadr[jid_act])

                    target_val: float = float(initial_qpos[qa_act])

                    # Walking gait targets for specific joints
                    if 'hip_y_right' in act_name:
                        target_val += HIP_AMP * hip_amp_mult * float(np.sin(phase))
                    elif 'hip_y_left' in act_name:
                        target_val += HIP_AMP * hip_amp_mult * float(np.sin(phase + np.pi))
                    elif 'right_knee' in act_name:
                        target_val -= KNEE_AMP * knee_amp_mult * max(0.0, float(np.sin(phase)))
                    elif 'left_knee' in act_name:
                        target_val -= KNEE_AMP * knee_amp_mult * max(0.0, float(np.sin(phase + np.pi)))
                    elif 'shoulder1_right' in act_name or 'shoulder_y_right' in act_name:
                        target_val += ARM_AMP * float(np.sin(phase + np.pi))
                    elif 'shoulder1_left' in act_name or 'shoulder_y_left' in act_name:
                        target_val += ARM_AMP * float(np.sin(phase))

                    error_ctrl: float = target_val - float(data.qpos[qa_act])
                    vel_ctrl: float = float(data.qvel[da_act])

                    # Gain selection by joint group
                    if any(kw in act_name for kw in ['hip_y', 'knee', 'ankle']):
                        kp_c: float = KP_CTRL_LEG
                        kd_c: float = KD_CTRL_LEG
                    elif any(kw in act_name for kw in ['shoulder', 'elbow']):
                        kp_c = KP_CTRL_ARM
                        kd_c = KD_CTRL_ARM
                    else:
                        kp_c = KP_CTRL_BODY
                        kd_c = KD_CTRL_BODY

                    # Stronger legs during warmup
                    if is_warmup and any(kw in act_name for kw in ['hip_y', 'knee', 'ankle']):
                        kp_c *= 1.5
                        kd_c *= 1.5

                    ctrl_val: float = float(np.clip(kp_c * error_ctrl - kd_c * vel_ctrl, -1.0, 1.0))
                    data.ctrl[aid] = ctrl_val

                # ── 6. Step physics ──
                mj.mj_step(model, data)

            # ── Create ViserServer on port 8081 ──
            # Avoid conflict with FastAPI on 8080.
            viser_server = ViserServer(port=8081, verbose=False)

            # Get the actual port (ViserServer auto-increments if port is occupied)
            actual_port: int = viser_server._websock_server._port
            mjviser_viewer_url = f"http://localhost:{actual_port}"

            # ── v0.11.0: Custom reset_fn — correct standing pose on Reset ──
            # mjviser's default _reset() calls mj_resetData which restores qpos
            # to XML defaults (root_z=1.5, all joints=0). At z=1.5 the feet are
            # 0.22m above ground → ncon=0 → no ground contact.
            #
            # Fix: override reset_fn to set the correct standing pose (z=1.285,
            # feet on ground, ncon=4) every time the user clicks Reset.
            def reset_fn(mdl: "mj.MjModel", dat: "mj.MjData") -> None:
                """Reset to neutral standing pose with feet on ground."""
                dat.qpos[:] = 0.0
                dat.qpos[2] = target_height  # 1.285 for plain, 0.85 for scenes
                dat.qpos[3] = 1.0             # Upright quaternion (w=1)
                dat.qvel[:] = 0.0
                dat.qacc[:] = 0.0
                dat.ctrl[:] = 0.0
                dat.qfrc_applied[:] = 0.0
                mj.mj_forward(mdl, dat)
                # Re-capture initial_qpos after reset so PD targets are correct
                walk_state['initial_qpos'] = dat.qpos.copy()
                # Reset waypoint timer
                walk_state['last_wp_time'] = 0.0
                walk_state['waypoint'] = _generate_waypoint(walk_rng)

            viewer = mjviser.Viewer(
                model=mj_model,
                data=mj_data,
                step_fn=step_fn,
                reset_fn=reset_fn,
                server=viser_server,
            )
            # Start in paused mode: robot shows upright pose, user clicks Play to start
            viewer._paused = True
            mjviser_viewer_running = True

            # ── Manual viewer loop (replaces viewer.run()) ──
            # viewer.run() uses signal.signal() which fails in background threads,
            # so we replicate its logic here without the signal handling.
            viewer._setup_gui()

            # Initial forward pass and render
            if viewer._render_fn is None:
                mj.mj_forward(mj_model, mj_data)
            viewer._render()

            # Initialize timing counters
            now: float = _time.perf_counter()
            viewer._last_tick = now
            viewer._stats_last_time = now

            try:
                while mjviser_viewer_running:
                    viewer._tick()
                    _time.sleep(0.001)
            finally:
                viser_server.stop()
                mjviser_viewer_running = False
                mjviser_viewer_url = ""

        except Exception as e:
            mjviser_viewer_running = False
            mjviser_viewer_url = ""
            traceback.print_exc()
            print(f"mjviser viewer failed: {e}")

    mjviser_viewer_thread = threading.Thread(
        target=launch_viewer, daemon=True)
    mjviser_viewer_thread.start()

    # Brief wait so the thread can start ViserServer and determine the actual port
    time.sleep(0.5)

    return JSONResponse(content={
        "status": "starting",
        "url": mjviser_viewer_url,
        "available": True,
    })


@app.get("/api/status")
async def get_status() -> JSONResponse:
    """Return current run status.

    Returns:
        JSONResponse with current running state details.
    """
    with run_state.lock:
        return JSONResponse(content={
            "is_running": run_state.is_running,
            "current_task": run_state.current_task,
            "current_episode": run_state.current_episode,
            "current_step": run_state.current_step,
            "total_episodes": run_state.total_episodes,
            "mjviser_available": MJVISER_AVAILABLE,
        })


# ── WebSocket Endpoint ──
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time benchmark metric streaming.

    Connected clients receive per-step metric updates during benchmark
    runs, including η, Noether violations, κ-Snap status, ψ-Anchor
    state, motor IC-Values, CQ metrics, and κ-Snap MerkleChain events.

    v0.6.0: Also pushes κ-Snap audit events (CQ + Merkle) per step.
    """
    await websocket.accept()
    run_state.ws_clients.append(websocket)

    try:
        # Send initial status (v0.6.0: includes CQ + Merkle API info)
        await websocket.send_json({
            "type": "connected",
            "version": WEBVIZ_VERSION,
            "mjviser_available": MJVISER_AVAILABLE,
            "cq_api": "/api/cq",
            "merkle_api": "/api/merkle",
        })

        # Keep connection alive — client can send control messages
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg.get("type") == "get_cq":
                # v0.6.0: Client requests CQ metrics
                cq_response: Dict[str, Any] = {}
                if run_state.current_metrics is not None:
                    cq_response = {
                        "cq": run_state.current_metrics.get("cq_avg", 0.0),
                        "cq_noether": run_state.current_metrics.get("cq_noether_avg", 0.0),
                        "cq_pgate": run_state.current_metrics.get("cq_pgate_avg", 0.0),
                        "cq_sentient": run_state.current_metrics.get("cq_sentient_avg", 0.0),
                    }
                await websocket.send_json({"type": "cq_update", "data": cq_response})
            elif msg.get("type") == "get_merkle":
                # v0.6.0: Client requests MerkleChain
                merkle_response: Dict[str, Any] = {
                    "chain": [],
                    "verified": False,
                }
                if run_state.current_metrics is not None:
                    merkle_response["chain"] = run_state.current_metrics.get("merkle_chain", [])
                    merkle_response["verified"] = run_state.current_metrics.get("merkle_chain_verified", False)
                await websocket.send_json({"type": "merkle_update", "data": merkle_response})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in run_state.ws_clients:
            run_state.ws_clients.remove(websocket)


# ── Serve Dashboard HTML ──
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    """Serve the dashboard HTML page.

    Returns:
        HTMLResponse with the full dashboard page content.
    """
    dashboard_path: str = str(Path(__file__).resolve().parent / "dashboard.html")
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content: str = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard HTML not found</h1>", status_code=404)


# ── Serve Documentation HTML Pages ──
WEBVIZ_DIR: str = os.path.dirname(os.path.abspath(__file__))

@app.get("/user_manual.html")
async def serve_user_manual() -> FileResponse:
    """Serve the user manual HTML page.

    Returns:
        FileResponse with the user_manual.html content.
    """
    return FileResponse(os.path.join(WEBVIZ_DIR, "user_manual.html"), media_type="text/html")


@app.get("/mujoco_docs_cn.html")
async def serve_mujoco_docs_cn() -> FileResponse:
    """Serve the MuJoCo Overview Chinese translation HTML page.

    Returns:
        FileResponse with the mujoco_docs_cn.html content.
    """
    return FileResponse(os.path.join(WEBVIZ_DIR, "mujoco_docs_cn.html"), media_type="text/html")


# ── CORS middleware (for development) ──
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
