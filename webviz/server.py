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

Author: MuJoCo-Bench-IDO Webviz extension v0.3.0
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
    _aggregate_metrics,
    _import_env,
)
from agent.mujoco_ido_agent import IDOMuJoCoAgent
from agent.psi_anchor import PsiAnchor
from core.goal_eml_mj import GoalEML
from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor
from core.noether_check_mj import noether_check_mj

WEBVIZ_VERSION: str = "v0.4.2"

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
        asyncio.run_coroutine_threadsafe(broadcast_to_clients(data), _uvicorn_loop)


# ── Per-step Episode Runner (with WebSocket streaming) ──
def run_episode_with_streaming(
    env: Any,
    agent: IDOMuJoCoAgent,
    max_steps: int,
    episode: int,
) -> dict:
    """Run a single episode and broadcast per-step metrics via WebSocket.

    This is a modified version of run_single_episode that streams each
    step's data to connected WebSocket clients instead of just printing
    to console.

    Args:
        env: dm_control Environment instance.
        agent: IDOMuJoCoAgent instance.
        max_steps: Maximum number of steps per episode.
        episode: Current episode number (1-based) for broadcast.

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
    steps: int = 0
    start_time: float = time.time()

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

        # ── Compute per-step metrics for WebSocket broadcast ──
        eta: float = agent._last_eta if agent._last_eta is not None else float('inf')

        # Recompute η for this step (agent already computed it in choose_action)
        z_i: dict = agent._extract_eml_obs(env.physics, timestep=timestep)
        eta = gauss_ex_residual(z_i, agent.goal,
                                flow_predictor=agent.flow_predictor)

        # Noether check
        noether_ok: bool = True
        noether_msg: str = ""
        if agent.prev_data is not None:
            noether_ok, noether_msg = noether_check_mj(
                agent.prev_data, env.physics.data, agent.goal)
            if not noether_ok:
                noether_violations += 1

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

        # Motor IC-Values
        motor_ic_values: List[float] = [float(m[1]) for m in agent.macros]

        # End-effector position
        ee_pos: List[float] = []
        try:
            ee_arr = env.physics.named.data.xpos['right_hand', :].copy()
            ee_pos = [float(v) for v in ee_arr]
        except (KeyError, IndexError):
            ee_arr = z_i.get('ee_pos', np.zeros(3))
            ee_pos = [float(v) for v in ee_arr[:3]]

        # Target position
        target: List[float] = [float(v) for v in agent.goal.target_pos[:3]]

        agent.prev_data = env.physics.data

        # ── Broadcast step data via WebSocket ──
        step_data: dict = {
            "step": step_idx + 1,
            "episode": episode,
            "eta": float(eta),
            "noether_violations": noether_violations,
            "kappa_snap_triggered": kappa_snap_triggered,
            "delta_k": float(delta_k),
            "psi_anchor_policy": psi_anchor_policy,
            "epiplexity": float(epiplexity),
            "motor_ic_values": motor_ic_values,
            "ee_pos": ee_pos,
            "target": target,
        }

        # Update run state
        run_state.current_step = step_idx + 1
        run_state.current_episode = episode

        # Broadcast to WebSocket clients
        broadcast_sync(step_data)

        # Goal achievement check
        ee: Optional[np.ndarray] = None
        try:
            ee = env.physics.named.data.xpos['right_hand', :].copy()
        except (KeyError, IndexError):
            ee = z_i.get('ee_pos', None)

        if ee is not None:
            dist: float = np.linalg.norm(ee - agent.goal.target_pos)
            if dist < agent.goal.pos_tol:
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
        'elapsed_s': elapsed,
        'avg_return': getattr(timestep, 'reward', 0.0),
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
        })

        results: List[dict] = []
        for ep in range(1, episodes + 1):
            if run_state.should_stop:
                broadcast_sync({"type": "run_stopped", "episode": ep})
                break

            metrics = run_episode_with_streaming(env, agent, max_steps, ep)
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

    except Exception as e:
        broadcast_sync({
            "type": "error",
            "message": f"Benchmark error: {str(e)}",
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
        )
        agent_t0 = IDOMuJoCoAgent(env, goal_t0,
                                   kappa_thresh=kappa_thresh,
                                   enable_critique=True)
        agent_t0.psi_anchor = PsiAnchor(goal_t0)
        agent_t0.flow_predictor = FlowMatchingEtaPredictor()

        broadcast_sync({"type": "sip_phase_start", "phase": "T0"})

        t0_results: List[dict] = []
        for ep in range(1, episodes + 1):
            if run_state.should_stop:
                break
            metrics = run_episode_with_streaming(env, agent_t0, max_steps, ep)
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
        )
        agent_t1 = IDOMuJoCoAgent(env, goal_t1,
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
                metrics = run_episode_with_streaming(env, agent_t1, max_steps, total_ep)
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
        )
        agent_t2 = IDOMuJoCoAgent(env, goal_t2,
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
            metrics = run_episode_with_streaming(env, agent_t2, max_steps, total_ep)
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

    except Exception as e:
        broadcast_sync({
            "type": "error",
            "message": f"SIP-Bench error: {str(e)}",
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
        "hopper-stand": "Hopper standing balance with ground contact",
        "walker-run": "Walker forward locomotion without falling",
        "reacher-easy": "Reacher simple 2-DOF reaching task",
    }
    for task_name in TASK_REGISTRY.keys():
        tasks.append({
            "name": task_name,
            "description": task_descriptions.get(task_name, ""),
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
async def stop_run(request: StopRequest = None) -> JSONResponse:
    """Stop the currently running benchmark.

    Args:
        request: StopRequest (optional, defaults to graceful stop).

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


@app.post("/api/mjviser/scene")
async def set_mjviser_scene(req: SceneRequest) -> JSONResponse:
    """Set the 3D scene type for mjviser viewer.

    Args:
        req: SceneRequest with scene_type field ("plain" or "obstacle").

    Returns:
        JSONResponse with the current scene_type.
    """
    global mjviser_scene_type
    if req.scene_type not in {"plain", "obstacle"}:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid scene type. Must be 'plain' or 'obstacle'.")
    mjviser_scene_type = req.scene_type
    return JSONResponse(content={"scene_type": mjviser_scene_type})


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

            # Load scene based on mjviser_scene_type
            env_ref = None
            target_qpos = None  # PD controller target for plain scene
            actuator_to_qpos_dof = None  # Mapping for PD controller

            if mjviser_scene_type == "obstacle":
                scene_xml_path = str(Path(__file__).resolve().parent / "scenes" / "humanoid_obstacle_arena.xml")
                mj_model = mj.MjModel.from_xml_path(scene_xml_path)
                mj_data = mj.MjData(mj_model)
                mj.mj_resetData(mj_model, mj_data)
                mj.mj_forward(mj_model, mj_data)
            else:
                # Default: load dm_control humanoid-stand
                env_ref = suite.load("humanoid", "stand")
                env_ref.reset()  # Reset to get initial upright pose

                # ── Capture initial joint positions as PD target ──
                # The dm_control humanoid-stand reset() places the robot upright.
                # We use this as the target posture for the PD controller.
                target_qpos = env_ref.physics.data._data.qpos.copy()

                mj_model = env_ref.physics.model._model
                mj_data = env_ref.physics.data._data

                # ── Build actuator → (qpos_adr, dof_adr) mapping ──
                # Each dm_control humanoid actuator is a single-DOF hinge joint.
                # model.actuator_trnid[i, 0] gives the joint ID for actuator i.
                # model.jnt_qposadr[jnt_id] gives the qpos address for that joint.
                # model.jnt_dofadr[jnt_id] gives the DOF (qvel) address.
                actuator_to_qpos_dof = []
                num_actuators = mj_model.nu
                for i in range(num_actuators):
                    jnt_id = int(mj_model.actuator_trnid[i, 0])
                    qpos_adr = int(mj_model.jnt_qposadr[jnt_id])
                    dof_adr = int(mj_model.jnt_dofadr[jnt_id])
                    actuator_to_qpos_dof.append((qpos_adr, dof_adr))

            # Create ViserServer on port 8081 (avoid conflict with FastAPI on 8080)
            viser_server = ViserServer(port=8081, verbose=False)

            # Get the actual port (ViserServer auto-increments if port is occupied)
            actual_port: int = viser_server._websock_server._port
            mjviser_viewer_url = f"http://localhost:{actual_port}"

            # ── Standing Controller for plain humanoid scene ──
            # The dm_control humanoid has a FREE root joint (7 qpos, 6 DOF)
            # with no actuator. Without stabilization the robot falls under
            # gravity (≈400N downward).
            #
            # Strategy (hard-lock root + joint PD):
            # After each physics step, we hard-reset root position and
            # orientation to their initial values and zero root velocity.
            # This is equivalent to pinning the robot at a fixed point in
            # space — zero drift, zero flash, zero twitching.
            # Joint PD via qfrc_applied keeps the limbs at their upright pose.
            # NO periodic motion — pure pose lock for a calm, stable display.
            #
            # Verified: root stays at EXACTLY (0, 0, 1.5) for ≥20 simulated
            # seconds with zero horizontal drift and zero flash/twitch.
            KP_JOINT: float = 50.0
            KD_JOINT: float = 15.0

            def plain_step_fn(model: mj.MjModel, data: mj.MjData) -> None:
                """Standing controller — hard-lock root + joint PD (no motion).

                Two-layer approach:
                1. Joint PD via qfrc_applied: drives each hinge toward
                   target_qpos for upright pose. No periodic motion offsets.
                2. Hard-lock root after each step: reset root position,
                   orientation, and velocity to initial values. This pins
                   the robot at a fixed point — eliminates all drift, tilt,
                   and flash/twitch that PD-only root control cannot prevent.

                Args:
                    model: MuJoCo model.
                    data: MuJoCo data (modified in-place).
                """
                # ── Apply joint PD via qfrc_applied ──
                data.qfrc_applied[:] = 0.0
                for qpos_adr, dof_adr in actuator_to_qpos_dof:
                    error = target_qpos[qpos_adr] - data.qpos[qpos_adr]
                    vel = data.qvel[dof_adr]
                    torque = KP_JOINT * error - KD_JOINT * vel
                    torque = float(np.clip(torque, -200.0, 200.0))
                    data.qfrc_applied[dof_adr] = torque

                # ── Step physics ──
                mj.mj_step(model, data)

                # ── Hard-lock root position + orientation + velocity ──
                # After mj_step, reset root to exact initial pose. This
                # eliminates all drift (root_x, root_y) and tilt (root
                # rotation), while keeping joint angles governed by PD.
                data.qpos[0:3] = target_qpos[0:3]  # Lock root x, y, z
                data.qpos[3:7] = target_qpos[3:7]  # Lock root quaternion
                data.qvel[0:6] = 0.0                # Zero root velocity

            # ── obstacle scene: capture initial pose for hard-lock ──
            obstacle_target_qpos: np.ndarray = mj_data.qpos.copy()
            # Build actuator mapping for obstacle scene too
            obstacle_actuator_to_qpos_dof: list = []
            if mj_model.nu > 0:
                for i in range(mj_model.nu):
                    jnt_id = int(mj_model.actuator_trnid[i, 0])
                    qpos_adr = int(mj_model.jnt_qposadr[jnt_id])
                    dof_adr = int(mj_model.jnt_dofadr[jnt_id])
                    obstacle_actuator_to_qpos_dof.append((qpos_adr, dof_adr))

            def obstacle_step_fn(model: mj.MjModel, data: mj.MjData) -> None:
                """Standing controller for obstacle scene — hard-lock root + joint PD.

                Same strategy as plain scene: hard-lock root position and
                orientation after each step, use joint PD to hold upright
                pose. This keeps the humanoid standing amid obstacles instead
                of falling to the ground.

                Args:
                    model: MuJoCo model.
                    data: MuJoCo data (modified in-place).
                """
                # ── Apply joint PD via qfrc_applied ──
                data.qfrc_applied[:] = 0.0
                data.ctrl[:] = 0.0
                for qpos_adr, dof_adr in obstacle_actuator_to_qpos_dof:
                    error = obstacle_target_qpos[qpos_adr] - data.qpos[qpos_adr]
                    vel = data.qvel[dof_adr]
                    torque = KP_JOINT * error - KD_JOINT * vel
                    torque = float(np.clip(torque, -200.0, 200.0))
                    data.qfrc_applied[dof_adr] = torque

                # ── Step physics ──
                mj.mj_step(model, data)

                # ── Hard-lock root position + orientation + velocity ──
                data.qpos[0:3] = obstacle_target_qpos[0:3]
                data.qpos[3:7] = obstacle_target_qpos[3:7]
                data.qvel[0:6] = 0.0

            # Select step_fn based on scene type
            step_fn = plain_step_fn if mjviser_scene_type == "plain" else obstacle_step_fn

            viewer = mjviser.Viewer(
                model=mj_model,
                data=mj_data,
                step_fn=step_fn,
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
    state, and motor IC-Values.
    """
    await websocket.accept()
    run_state.ws_clients.append(websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "version": WEBVIZ_VERSION,
            "mjviser_available": MJVISER_AVAILABLE,
        })

        # Keep connection alive — client can send control messages
        while True:
            data = await websocket.receive_text()
            # Handle client messages (future: pause/resume commands)
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
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
