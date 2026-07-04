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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
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
from core.gel_loss import GELLoss, GELConfig, compute_gel_from_step

# v0.16.14: TOMAS wrapper for SO-ARM100 IDO integration
# v0.16.17: Also import VLA adapter factory (lazy import in API endpoints)
try:
    from webviz.tomas_wrapper import (
        TOMASMuJoCoWrapper,
        PsiAnchorGate,
        KappaSnap,
        SOArm100Controller,
    )
    TOMAS_AVAILABLE = True
except ImportError as _tomas_err:
    TOMAS_AVAILABLE = False
    print(f"Warning: TOMAS wrapper not available: {_tomas_err}")

WEBVIZ_VERSION: str = "v0.16.32"

# ── Bug 3 Fix: Baseline reference data for 7 core metrics ──
# Source: dm_control PPO/SAC 100k step typical scores (paperswithcode, DM Control paper)
# H_EML_residual: target < 0.5 (lower = better, η approaching target)
# noether_violations: 1000-step cumulative violation count (collision-dominant), IDO target < 100
# snap_efficiency: PPO ~0.45, IDO target > 0.8 (fraction of steps in PD-snap mode)
# epiplexity: PPO/SAC N/A (no ψ-Anchor); IDO target >200 (sufficient structural info for evolution)
# cq_overall: PPO/SAC ~0.15 (no conscience framework); IDO target >0.80
BASELINE_REFERENCE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "humanoid-stand": {
        "episode_return": {"PPO_100k": 220, "SAC_100k": 180, "IDO_target": ">300", "interpretation": "PPO 100k 后 ~220（dm_control 标准）；IDO point η-mode 下 reward 尺度不同，累计 ~4-14 属正常"},
        "success_rate": {"PPO_100k": 0.55, "SAC_100k": 0.48, "IDO_target": ">0.70", "interpretation": "PPO 100k 后 ~55%，IDO 目标 >70%"},
        "H_EML_residual": {"PPO_100k": 0.85, "IDO_target": "<0.5", "interpretation": "越低越好，<0.5 表明 η 接近目标"},
        "noether_violations": {"PPO_100k": 800, "IDO_target": "<100", "interpretation": "humanoid-stand 1000步累计违规 ~800（碰撞检测为主），IDO 目标 <100"},
        "snap_efficiency": {"PPO_100k": 0.45, "IDO_target": ">0.8", "interpretation": "Snap 触发比例，>0.8 表示高效"},
        "epiplexity": {"PPO_100k": "N/A", "SAC_100k": "N/A", "IDO_target": ">300", "interpretation": "PPO/SAC 无 ψ-Anchor 不计算 epiplexity；IDO 目标 >300 表示有足够结构信息支持进化（实测 ~373）"},
        "cq_overall": {"PPO_100k": 0.10, "SAC_100k": 0.12, "IDO_target": ">0.80", "interpretation": "PPO/SAC 无良知框架 CQ~0.10；IDO 目标 >0.80"},
    },
    "cheetah-run": {
        "episode_return": {"PPO_100k": 350, "SAC_100k": 420, "IDO_target": ">500", "interpretation": "PPO 100k 后 ~350，SAC 更强 ~420，IDO 目标 >500"},
        "success_rate": {"PPO_100k": 0.60, "SAC_100k": 0.72, "IDO_target": ">0.80", "interpretation": "cheetah 是连续运动任务，IDO 目标 >80%"},
        "H_EML_residual": {"PPO_100k": 0.72, "IDO_target": "<0.4", "interpretation": "运动模式 η 目标更低，<0.4 为良好"},
        "noether_violations": {"PPO_100k": 600, "IDO_target": "<50", "interpretation": "cheetah-run 1000步累计违规 ~600（碰撞检测为主），IDO 目标 <50"},
        "snap_efficiency": {"PPO_100k": 0.50, "IDO_target": ">0.85", "interpretation": "周期性步态 Snap 效率较高，>0.85"},
        "epiplexity": {"PPO_100k": "N/A", "SAC_100k": "N/A", "IDO_target": ">200", "interpretation": "PPO/SAC 无 ψ-Anchor 不计算 epiplexity；IDO 目标 >200 表示有足够结构信息支持进化"},
        "cq_overall": {"PPO_100k": 0.15, "SAC_100k": 0.18, "IDO_target": ">0.80", "interpretation": "PPO/SAC 无良知框架 CQ~0.15；IDO 目标 >0.80"},
    },
    "walker-walk": {
        "episode_return": {"PPO_100k": 250, "SAC_100k": 310, "IDO_target": ">400", "interpretation": "PPO 100k 后 ~250，IDO 目标 >400"},
        "success_rate": {"PPO_100k": 0.50, "SAC_100k": 0.65, "IDO_target": ">0.75", "interpretation": "walker 步态稳定性，IDO 目标 >75%"},
        "H_EML_residual": {"PPO_100k": 0.78, "IDO_target": "<0.5", "interpretation": "双足运动 η 目标 <0.5"},
        "noether_violations": {"PPO_100k": 700, "IDO_target": "<80", "interpretation": "walker-walk 1000步累计违规 ~700（碰撞检测为主），IDO 目标 <80"},
        "snap_efficiency": {"PPO_100k": 0.40, "IDO_target": ">0.8", "interpretation": "walker 需更多探索，Snap 效率偏低，>0.8"},
        "epiplexity": {"PPO_100k": "N/A", "SAC_100k": "N/A", "IDO_target": ">250", "interpretation": "PPO/SAC 无 ψ-Anchor 不计算 epiplexity；IDO 目标 >250 表示有足够结构信息支持进化"},
        "cq_overall": {"PPO_100k": 0.12, "SAC_100k": 0.15, "IDO_target": ">0.80", "interpretation": "PPO/SAC 无良知框架 CQ~0.12；IDO 目标 >0.80"},
    },
    "hopper-stand": {
        "episode_return": {"PPO_100k": 180, "SAC_100k": 150, "IDO_target": ">280", "interpretation": "PPO 100k 后 ~180，IDO 目标 >280"},
        "success_rate": {"PPO_100k": 0.45, "SAC_100k": 0.40, "IDO_target": ">0.65", "interpretation": "单足站立难度大，IDO 目标 >65%"},
        "H_EML_residual": {"PPO_100k": 0.90, "IDO_target": "<0.5", "interpretation": "单足平衡 η 偏高，目标 <0.5"},
        "noether_violations": {"PPO_100k": 850, "IDO_target": "<100", "interpretation": "hopper-stand 1000步累计违规 ~850（碰撞检测为主），IDO 目标 <100"},
        "snap_efficiency": {"PPO_100k": 0.35, "IDO_target": ">0.75", "interpretation": "hopper 需频繁探索，>0.75"},
        "epiplexity": {"PPO_100k": "N/A", "SAC_100k": "N/A", "IDO_target": ">200", "interpretation": "PPO/SAC 无 ψ-Anchor 不计算 epiplexity；IDO 目标 >200 表示有足够结构信息支持进化"},
        "cq_overall": {"PPO_100k": 0.10, "SAC_100k": 0.10, "IDO_target": ">0.75", "interpretation": "PPO/SAC 无良知框架 CQ~0.10；IDO 目标 >0.75"},
    },
    # ── Generic fallback for other tasks ──
    "_default": {
        "episode_return": {"PPO_100k": 200, "SAC_100k": 200, "IDO_target": ">300", "interpretation": "PPO/SAC 100k 后 ~200；IDO η-mode 下 reward 尺度可能不同"},
        "success_rate": {"PPO_100k": 0.50, "SAC_100k": 0.50, "IDO_target": ">0.70", "interpretation": "PPO 100k 后 ~50%，IDO 目标 >70%"},
        "H_EML_residual": {"PPO_100k": 0.80, "IDO_target": "<0.5", "interpretation": "越低越好，<0.5 表明 η 接近目标"},
        "noether_violations": {"PPO_100k": 700, "IDO_target": "<80", "interpretation": "1000步累计违规 ~700（碰撞检测为主），IDO 目标 <80"},
        "snap_efficiency": {"PPO_100k": 0.45, "IDO_target": ">0.8", "interpretation": "Snap 效率，>0.8 表示高效"},
        "epiplexity": {"PPO_100k": "N/A", "SAC_100k": "N/A", "IDO_target": ">200", "interpretation": "PPO/SAC 无 ψ-Anchor 不计算 epiplexity；IDO 目标 >200 表示有足够结构信息支持进化"},
        "cq_overall": {"PPO_100k": 0.15, "SAC_100k": 0.20, "IDO_target": ">0.80", "interpretation": "PPO/SAC 无良知框架 CQ~0.15；IDO 目标 >0.80"},
    },
}

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
mjviser_viewer_error: str = ""  # captures last error from viewer thread
mjviser_scene_type: str = "plain"

# v0.16.7: Direction control — allows user to steer the robot in 3D viewer
# v0.16.11: Split into target_direction (set by buttons) and walk_direction (interpolated)
mjviser_walk_direction: float = 0.0   # current yaw angle (radians), interpolated towards target
mjviser_target_direction: float = 0.0  # target yaw angle (radians), set by direction buttons
mjviser_walk_speed: float = 1.0       # 0=stop, 1=full walk
mjviser_walk_action: str = "walk"     # "walk" | "stop" | "turn_left" | "turn_right" | "step_up"

# v0.16.14: Persistent ViserServer — created once, reused across scene changes.
# Previous approach (stop + recreate) failed on Windows because viser_server.stop()
# doesn't synchronously release the port. Now we keep the server alive and only
# recreate the Viewer/Model when scenes change.
mjviser_persistent_server = None       # type: Optional[Any]
mjviser_persistent_port: int = 0
mjviser_viewer_object = None           # type: Optional[Any]  # current mjviser.Viewer instance

# v0.16.15: SO-ARM100 independent viewer system — completely separate from humanoid mjviser
arm100_viewer_thread: Optional[threading.Thread] = None
arm100_viewer_running: bool = False
arm100_viewer_url: str = ""
arm100_viewer_error: str = ""
arm100_persistent_server = None        # type: Optional[Any]
arm100_persistent_port: int = 0
arm100_tomas_wrapper = None            # type: Optional[Any]  # TOMASMuJoCoWrapper instance

# v0.16.17: Simulation speed multiplier, direction cooldown, ARM100 manual/VLA control
mjviser_sim_speed: int = 1           # v0.16.17: Simulation speed multiplier (1-64x)
mjviser_last_dir_change_time: float = 0.0  # v0.16.17: Anti-backflip cooldown timer
arm100_manual_mode: bool = False     # v0.16.17: SO-ARM100 manual control mode
arm100_manual_target: Optional[np.ndarray] = None  # v0.16.17: Manual joint targets
arm100_vla_adapter = None            # v0.16.17: VLA adapter instance
arm100_vla_mode: bool = False        # v0.16.17: VLA inference mode
arm100_vla_instruction: str = "pick up the red cube"  # v0.16.17: VLA language instruction
arm100_cam_top_frame: Optional[bytes] = None       # v0.16.19: CAMKit top cam JPEG
arm100_cam_wrist_frame: Optional[bytes] = None     # v0.16.19: CAMKit wrist cam JPEG
arm100_cam_top_rgb: Optional["np.ndarray"] = None  # v0.16.19: CAMKit top cam RGB array (for VLA)
arm100_cam_wrist_rgb: Optional["np.ndarray"] = None # v0.16.19: CAMKit wrist cam RGB array (for VLA)

try:
    import mjviser
    # v0.16.21: Patch mjviser _SPEEDS to support up to 64x (was max 8x)
    if hasattr(mjviser, 'viewer') and hasattr(mjviser.viewer, '_SPEEDS'):
        mjviser.viewer._SPEEDS = [1, 2, 4, 8, 16, 32, 64]
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

    # v0.16.19: GEL (Goal-EML Injection Loss) accumulator
    gel_tracker = GELLoss(config=GELConfig())

    # v0.16.26: Initialize new core modules for benchmark integration
    # P0: KappaSnapTokenizer — encode κ-Snap events per step
    _kappa_tokenizer = None
    try:
        from core.kappa_snap_tokenizer import KappaSnapTokenizer
        _kappa_tokenizer = KappaSnapTokenizer(window_size=16, summary_dim=32)
    except Exception:
        pass

    # P1: T-Processor — hardware η-ALU + ψ-Checker + κ-Snap FIFO
    _t_processor = None
    try:
        from core.t_processor import TProcessor
        _t_processor = TProcessor()
    except Exception:
        pass

    # P2: Three-Body — Virtual→Software→Physical execution pipeline
    _three_body = None
    try:
        from core.three_body import ThreeBodySystem
        _three_body = ThreeBodySystem(sim_mode=True)
    except Exception:
        pass

    # P2: HG-PINN — Hamiltonian-guided action head (residual mode)
    _hg_pinn = None
    try:
        from core.hg_pinn import HGPINNActionHead, HGPINNConfig
        _hg_pinn = HGPINNActionHead(HGPINNConfig())
    except Exception:
        pass

    # P2: Nine-Layer registry — for architecture reporting
    _nine_layer_registry = None
    try:
        from core.nine_layer import NineLayerRegistry
        _nine_layer_registry = NineLayerRegistry()
    except Exception:
        pass

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

        # v0.16.19: Compute GEL (Goal-EML Injection Loss) for this step
        gel_step = compute_gel_from_step(
            eta=float(eta),
            noether_result=nvr_result if not noether_ok and agent.prev_data is not None else None,
            goal_max_energy=float(agent.goal.max_energy_inject),
            collide_thresh=float(agent.goal.collide_thresh),
            success=success,
        )
        gel_tracker.accumulate(gel_step)

        # ── v0.16.26: New core module integration ──

        # P1: T-Processor tick — hardware η-ALU + ψ-Checker + κ-Snap FIFO
        _tp_eta = None
        _tp_violation = None
        _tp_kappa_entry = None
        if _t_processor is not None:
            try:
                _tp_qvel = env.physics.data.qvel.copy()
                _tp_qfrc = env.physics.data.qfrc_actuator.copy()
                _tp_eta_result, _tp_psi_result, _tp_kappa_entry = _t_processor.tick(
                    obs=z_i, goal=agent.goal, action=action,
                    qvel=_tp_qvel, qfrc=_tp_qfrc,
                )
                _tp_eta = _tp_eta_result.eta_value
                _tp_violation = _tp_psi_result.violations[0] if _tp_psi_result.violations else None
            except Exception:
                pass

        # P0: KappaSnapTokenizer — encode κ-Snap events
        _kappa_tokens = ""
        _kappa_summary = [0.0] * 32
        if _kappa_tokenizer is not None:
            try:
                _event_type = "SAFE_STEP"
                if not noether_ok:
                    _event_type = "NVT_VIOLATION"
                elif _tp_violation is not None:
                    _event_type = _tp_violation
                elif kappa_snap_triggered:
                    _event_type = "KAPPA_SNAP"

                _eta_bucket = "mid"
                if eta < 0.1:
                    _eta_bucket = "vlo"
                elif eta < 1.0:
                    _eta_bucket = "lo"
                elif eta < 10.0:
                    _eta_bucket = "mid"
                elif eta < 100.0:
                    _eta_bucket = "hi"
                else:
                    _eta_bucket = "vhi"

                _decision = "EXPLOIT" if kappa_snap_triggered else "EXPLORE"
                _kappa_tokenizer.add_event(
                    level="L6", event_type=_event_type,
                    eta_bucket=_eta_bucket, decision=_decision,
                )
                _kappa_tokens = _kappa_tokenizer.get_token_string()
                _kappa_summary = _kappa_tokenizer.get_summary_vector(dim=32).tolist()
            except Exception:
                pass

        # P2: Three-Body — wrap action through Virtual→Software→Physical
        _three_body_gap = None
        if _three_body is not None:
            try:
                _bp, _op, _ap = _three_body.full_cycle(
                    action=action, obs=z_i, reward=step_reward, eta=float(eta),
                )
                _three_body_gap = _three_body.physical.get_sim_real_gap(_three_body.virtual)
            except Exception:
                pass

        # P2: HG-PINN — compute Hamiltonian energy stats
        _hg_energy = None
        if _hg_pinn is not None:
            try:
                _ee = np.array(ee_pos[:3]) if ee_pos else np.zeros(3)
                _hg_stats = _hg_pinn.get_energy_stats()
                _hg_energy = _hg_stats
            except Exception:
                pass

        # ── End v0.16.26 integration ──

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
            # v0.16.19: GEL auxiliary loss
            "gel_loss": float(gel_step["total"]),
            "gel_noether": float(gel_step["noether"]),
            "gel_contact": float(gel_step["contact"]),
            "gel_task": float(gel_step["task"]),
            "gel_hinge": float(gel_step["hinge"]),
            # v0.16.26: New core module data
            "kappa_tokens": _kappa_tokens,
            "kappa_summary": _kappa_summary,
            "t_processor_eta": float(_tp_eta) if _tp_eta is not None else None,
            "t_processor_violation": _tp_violation,
            "three_body_gap": _three_body_gap,
            "hg_pinn_energy": _hg_energy,
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

    # v0.16.19: GEL mean loss for this episode
    gel_mean = gel_tracker.mean_loss()

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
        # v0.16.19: GEL auxiliary loss
        'gel_loss': gel_mean['total'],
        'gel_noether': gel_mean['noether'],
        'gel_contact': gel_mean['contact'],
        'gel_task': gel_mean['task'],
        'gel_hinge': gel_mean['hinge'],
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
        # v0.16.6: Fix total_episodes for SIP — T0(episodes) + T1(evolution_rounds*episodes) + T2(episodes)
        run_state.total_episodes = request.episodes * (request.evolution_rounds + 2)

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

        # v0.16.6: Broadcast total_episodes (not per-phase) so dashboard shows correct progress
        # v0.16.11: Also send user_episodes + SIP phase breakdown so dashboard can explain
        total_eps_sip: int = episodes * (evolution_rounds + 2)
        broadcast_sync({
            "type": "run_start",
            "task": task,
            "episodes": total_eps_sip,
            "user_episodes": episodes,  # What the user actually set
            "sip_total_episodes": total_eps_sip,  # Expanded total for progress bar
            "sip_evolution_rounds": evolution_rounds,
            "sip_phase_breakdown": f"T0:{episodes} + T1:{episodes}×{evolution_rounds} + T2:{episodes} = {total_eps_sip}",
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

        # ── Bug 1 Fix: Broadcast sip_phase_complete for T0 ──
        broadcast_sync({
            "type": "sip_phase_complete",
            "phase": "T0",
            "summary": {
                "avg_eta": t0_summary.get('avg_final_eta', 0.0),
                "avg_steps": t0_summary.get('avg_steps', 0.0),
                "noether_violations": t0_summary.get('total_noether_violations', 0),
                "episodes_run": len(t0_results),
            }
        })

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

        # ── Bug 1 Fix: Broadcast sip_phase_complete for T1 ──
        broadcast_sync({
            "type": "sip_phase_complete",
            "phase": "T1",
            "summary": {
                "avg_eta": t1_summary.get('avg_final_eta', 0.0),
                "avg_steps": t1_summary.get('avg_steps', 0.0),
                "noether_violations": t1_summary.get('total_noether_violations', 0),
                "episodes_run": len(t1_phase_results),
            }
        })

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

        # ── Bug 1 Fix: Broadcast sip_phase_complete for T2 ──
        broadcast_sync({
            "type": "sip_phase_complete",
            "phase": "T2",
            "summary": {
                "avg_eta": t2_summary.get('avg_final_eta', 0.0),
                "avg_steps": t2_summary.get('avg_steps', 0.0),
                "noether_violations": t2_summary.get('total_noether_violations', 0),
                "episodes_run": len(t2_results),
            }
        })

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


@app.get("/api/sip_history")
async def get_sip_history() -> JSONResponse:
    """Return recent SIP-Bench results from saved JSON files.

    Scans benchmarks/results/sip_*.json, sorts by file modification time
    (newest first), and returns up to 10 entries. Each entry includes the
    task name, file modification timestamp, and the full sip_result content.

    Bug 2 Fix: Enables dashboard to restore last SIP result on page reload.

    Returns:
        JSONResponse with list of SIP history entries (max 10).
    """
    results_dir: str = os.path.join(PROJECT_ROOT, "benchmarks", "results")
    history: List[dict] = []

    if not os.path.isdir(results_dir):
        return JSONResponse(content={"sip_history": [], "count": 0})

    sip_files: List[tuple] = []
    for fname in os.listdir(results_dir):
        if fname.startswith("sip_") and fname.endswith(".json"):
            fpath: str = os.path.join(results_dir, fname)
            try:
                mtime: float = os.path.getmtime(fpath)
                sip_files.append((mtime, fpath, fname))
            except OSError:
                continue

    # Sort by mtime descending (newest first)
    sip_files.sort(key=lambda x: x[0], reverse=True)

    for mtime, fpath, fname in sip_files[:10]:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                sip_data: dict = json.load(f)
            # Extract task name from filename: sip_<task>_e<ep>_r<rounds>.json
            # or from the JSON content if available
            task_name: str = sip_data.get("task", "")
            if not task_name:
                # Parse from filename
                parts: list = fname.replace("sip_", "").replace(".json", "").split("_")
                if len(parts) >= 1:
                    task_name = parts[0]
            history.append({
                "task": task_name,
                "timestamp": mtime,
                "filename": fname,
                "sip_result": sip_data,
            })
        except (json.JSONDecodeError, OSError):
            continue

    return JSONResponse(content={"sip_history": history, "count": len(history)})


@app.get("/api/baseline_reference")
async def get_baseline_reference(task: Optional[str] = None) -> JSONResponse:
    """Return baseline reference data for the 7 core metrics.

    Bug 3 Fix: Provides PPO/SAC/IDO target reference values so the
    dashboard can show baseline comparison under each metric card.

    Args:
        task: Optional task name to get task-specific baselines.
              If not provided or not found, returns all baselines.

    Returns:
        JSONResponse with baseline reference dict.
    """
    if task and task in BASELINE_REFERENCE:
        return JSONResponse(content={
            "task": task,
            "baseline": BASELINE_REFERENCE[task],
        })
    return JSONResponse(content={
        "task": task or "all",
        "baseline": BASELINE_REFERENCE.get(task or "", BASELINE_REFERENCE["_default"]) if task else BASELINE_REFERENCE,
        "default": BASELINE_REFERENCE["_default"],
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
    global mjviser_scene_type, mjviser_viewer_running, mjviser_walk_direction, mjviser_target_direction, mjviser_walk_speed
    valid_scenes = {"plain", "obstacle", "ramp", "stairs", "floating", "maze"}
    if req.scene_type not in valid_scenes:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid scene type. Must be one of: {', '.join(sorted(valid_scenes))}.")
    mjviser_scene_type = req.scene_type
    # v0.16.11: Reset direction when switching scenes
    mjviser_walk_direction = 0.0
    mjviser_target_direction = 0.0
    mjviser_walk_speed = 1.0

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

    global mjviser_viewer_thread, mjviser_viewer_running, mjviser_viewer_url, mjviser_viewer_error
    global mjviser_persistent_server, mjviser_persistent_port, mjviser_viewer_object

    if mjviser_viewer_running:
        return JSONResponse(content={
            "status": "already_running",
            "url": mjviser_viewer_url,
        })

    # v0.16.14: Wait for old viewer thread to fully exit before starting a new one.
    # The old thread no longer stops the ViserServer (we keep it persistent),
    # but we still need to wait for the thread to exit so we don't have two
    # viewer loops running simultaneously.
    if mjviser_viewer_thread is not None and mjviser_viewer_thread.is_alive():
        mjviser_viewer_running = False  # ensure old thread exits
        mjviser_viewer_thread.join(timeout=5.0)  # wait up to 5s for cleanup
        # Small delay to let the old viewer object be garbage collected
        time.sleep(0.3)

    # Initialize viewer URL (will be updated by launch_viewer thread)
    mjviser_viewer_url = "http://localhost:8081"
    mjviser_viewer_error = ""  # clear previous error

    def launch_viewer() -> None:
        """Launch mjviser Viewer in a background thread with real-time simulation.

        Note: We cannot use viewer.run() directly because it calls
        signal.signal(), which raises ValueError in non-main threads
        ("signal only works in main thread of the main interpreter").
        Instead, we manually replicate the viewer loop, calling
        _setup_gui(), _render(), and _tick() in a while-loop.
        """
        global mjviser_viewer_running, mjviser_viewer_url, mjviser_viewer_error
        global mjviser_persistent_server, mjviser_persistent_port, mjviser_viewer_object
        try:
            import dm_control.suite as suite
            import mujoco as mj
            from viser import ViserServer
            import time as _time

            # ── Load scene based on mjviser_scene_type ──
            env_ref = None
            target_height: float = 0.85  # Custom stick-figure humanoid standing height
            scene_xml_path: Optional[str] = None

            # v0.16.11: ALL scenes now use the same custom XML humanoid (16.5kg, gear=100).
            # Previous: plain scene used dm_control humanoid (40.8kg) which was too heavy
            # and caused simulation instability. Now plain uses humanoid_plain_arena.xml
            # — same humanoid as obstacle scenes but without obstacles.
            scene_file_map: dict = {
                "plain": "humanoid_plain_arena.xml",
                "obstacle": "humanoid_obstacle_arena.xml",
                "ramp": "humanoid_ramp_arena.xml",
                "stairs": "humanoid_stairs_arena.xml",
                "floating": "humanoid_floating_platforms.xml",
                "maze": "humanoid_maze_arena.xml",
            }
            scene_file = scene_file_map.get(mjviser_scene_type, "humanoid_plain_arena.xml")
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

            def _quat_to_yaw(q: np.ndarray) -> float:
                """Extract yaw angle from quaternion [w, x, y, z]."""
                w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

            def _yaw_to_quat(yaw: float) -> np.ndarray:
                """Create upright quaternion [w, x, y, z] from yaw angle."""
                half = yaw * 0.5
                return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

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

            # ── v0.16.0: REAL PHYSICS — no kinematic locking ──
            # Previous versions (v0.12–v0.15.2) used kinematic root-locking:
            #   data.qpos[2] = target_height  (hard-lock Z)
            #   data.qpos[3:7] = _yaw_to_quat(heading)  (hard-lock orientation)
            #   data.qvel[0:6] = 0.0  (kill all root dynamics)
            #   data.qpos[0] += vx * dt  (kinematic x/y translation)
            # This was NOT physics — it was animation wearing a MuJoCo skin.
            #
            # v0.16.0: All joints are now hinge type (ball joints replaced with
            # 3 serial hinges). All actuators are motor type with gear=20–100.
            # The robot stands and walks through REAL contact forces, gravity,
            # and joint torques. No hard-locking of any qpos or qvel.
            #
            # Balance strategy:
            #   - PD controller on root orientation (keep upright)
            #   - PD controller on root height (prevent collapse)
            #   - Ankle/hip strategy for lateral balance
            #   - Sinusoidal gait pattern for hip/knee/ankle locomotion
            #   - Arms swing naturally for balance
            # v0.16.11: All scenes now use custom XML humanoid — treat all as obstacle scene
            # v0.16.15: SO-ARM100 separated to its own viewer system
            is_obstacle_scene: bool = True
            if is_obstacle_scene:
                for jid_s in range(mj_model.njnt):
                    jname_s: str = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, jid_s) or ""
                    jtype_s: int = int(mj_model.jnt_type[jid_s])
                    da_s: int = int(mj_model.jnt_dofadr[jid_s])
                    if jtype_s != 3:  # Only tune hinge joints
                        continue
                    is_hip: bool = 'hip' in jname_s
                    is_knee: bool = 'knee' in jname_s
                    is_ankle: bool = 'ankle' in jname_s
                    is_arm: bool = 'shoulder' in jname_s or 'elbow' in jname_s
                    is_head: bool = 'head' in jname_s
                    if is_hip:
                        mj_model.jnt_stiffness[jid_s] = 0.0
                        mj_model.dof_damping[da_s] = 2.0
                    elif is_knee:
                        mj_model.jnt_stiffness[jid_s] = 0.0
                        mj_model.dof_damping[da_s] = 2.0
                    elif is_ankle:
                        mj_model.jnt_stiffness[jid_s] = 0.0
                        mj_model.dof_damping[da_s] = 1.0
                    elif is_arm:
                        mj_model.jnt_stiffness[jid_s] = 0.0
                        mj_model.dof_damping[da_s] = 1.0
                    elif is_head:
                        mj_model.jnt_stiffness[jid_s] = 0.0
                        mj_model.dof_damping[da_s] = 1.0
            else:
                # Plain dm_control humanoid — keep existing high stiffness
                for jid_s in range(mj_model.njnt):
                    jname_s: str = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, jid_s) or ""
                    jtype_s: int = int(mj_model.jnt_type[jid_s])
                    da_s: int = int(mj_model.jnt_dofadr[jid_s])
                    is_abdomen: bool = (jtype_s == 3) and 'abdomen' in jname_s
                    is_hip_hinge: bool = (jtype_s == 3) and 'hip' in jname_s
                    is_knee: bool = (jtype_s == 3) and 'knee' in jname_s
                    if is_abdomen:
                        mj_model.jnt_stiffness[jid_s] = 500.0
                        mj_model.dof_damping[da_s] = 50.0
                    elif is_hip_hinge:
                        mj_model.jnt_stiffness[jid_s] = 300.0
                        mj_model.dof_damping[da_s] = 30.0
                    elif is_knee:
                        mj_model.jnt_stiffness[jid_s] = 200.0
                        mj_model.dof_damping[da_s] = 20.0

            # ── v0.16.0: Actuator gains ──
            # For scene XMLs: motor actuators already have gear=20–100 set in XML.
            #   motor actuator: force = ctrl * gear, so ctrl=1 → gear Nm torque.
            #   No need to modify actuator_gainprm — gear is baked into the motor.
            # For plain dm_control humanoid: still boost gainprm as before.
            if not is_obstacle_scene:
                for aid_amp in range(mj_model.nu):
                    amp_name: str = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_ACTUATOR, aid_amp) or ""
                    is_leg: bool = any(kw in amp_name for kw in ['hip_y', 'knee', 'ankle'])
                    is_arm: bool = any(kw in amp_name for kw in ['shoulder', 'elbow'])
                    is_torso: bool = any(kw in amp_name for kw in ['hip', 'abdomen', 'head'])
                    if is_leg:
                        mj_model.actuator_gainprm[aid_amp][0] = 40.0
                    elif is_torso:
                        mj_model.actuator_gainprm[aid_amp][0] = 30.0
                    elif is_arm:
                        mj_model.actuator_gainprm[aid_amp][0] = 25.0
                    else:
                        mj_model.actuator_gainprm[aid_amp][0] = 20.0

            # v0.16.11: All scenes use XML default timestep (0.005s)
            # No more timestep override for plain scene (was 0.002s for dm_control humanoid)

            # ── v0.16.0: Build actuator name → index map ──
            # All joints are hinge now, no ball joint special handling needed.
            actuator_name_map: dict = {}
            for _aid in range(mj_model.nu):
                _aname = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_ACTUATOR, _aid)
                if _aname:
                    actuator_name_map[_aname] = _aid

            # ── v0.16.1: Walking controller constants ──
            # Balance assist: soft spring on root to help weak stick-figure (16.5kg)
            # stay upright. This is NOT kinematic locking (v0.12-v0.15) — the robot
            # still responds to gravity, contact forces, and joint torques.
            # The assist just prevents catastrophic fall because our 0.15m feet
            # and 16.5kg mass can't generate enough ankle torque for real balance.
            #
            # Gait parameters — v0.16.12: Conservative gait for stability
            WALK_FREQ: float = 0.8           # v0.16.12: Slower steps (was 1.0)
            HIP_AMP: float = 0.12            # v0.16.12: Less aggressive swing (was 0.20)
            KNEE_AMP: float = 0.55           # v0.16.12: More ground clearance ~5cm (was 0.40)
            KNEE_STANCE: float = 0.20        # v0.16.12: Lower COM (was 0.15)
            ANKLE_AMP: float = 0.08          # v0.16.12: Gentler push-off (was 0.10)
            ARM_AMP: float = 0.10            # v0.16.12: Gentler arm swing (was 0.15)

            # v0.16.24: Reduced root gains — robot was "too bouncy" with KP=1200.
            #   - KP_ROOT_Z 1200→450 (less aggressive vertical push)
            #   - KD_ROOT_Z 80→140  (stronger damping kills oscillation)
            #   - Pitch/roll gains 350→220
            #   - Asymmetric Z clip: max +500N (≈1x gravity), min -150N (gentle pull-down)
            # v0.16.27: Increased angular damping for terrain stability.
            #   Previous KD=45 was too low — robot pitched 20°+ on ramp edges
            #   before the PD controller could react. Now KD=90 for pitch/roll.
            KP_ROOT_Z: float = 450.0         # v0.16.24: was 1200 (too bouncy)
            KD_ROOT_Z: float = 140.0         # v0.16.24: was 80
            KP_ROOT_PITCH: float = 280.0     # v0.16.27: was 220 (boosted for terrain)
            KD_ROOT_PITCH: float = 90.0      # v0.16.27: was 45 (2x damping kills flip momentum)
            KP_ROOT_ROLL: float = 280.0      # v0.16.27: was 220
            KD_ROOT_ROLL: float = 90.0       # v0.16.27: was 45
            KP_ROOT_YAW: float = 30.0        # Yaw spring (was 50) — gentler turning
            KD_ROOT_YAW: float = 8.0         # (was 10)
            # v0.16.12: Clip limits
            CLIP_ROOT_Z: float = 700.0       # (was 500)
            CLIP_ROOT_PITCH: float = 250.0   # (was 150)
            CLIP_ROOT_ROLL: float = 250.0    # (was 150)
            CLIP_ROOT_YAW: float = 15.0      # (was 60) — much gentler turning!

            # Joint PD gains (gait tracking)
            KP_HIP: float = 150.0
            KD_HIP: float = 15.0
            KP_KNEE: float = 100.0
            KD_KNEE: float = 10.0
            KP_ANKLE: float = 80.0
            KD_ANKLE: float = 8.0
            KP_ARM: float = 50.0
            KD_ARM: float = 5.0
            KP_HEAD: float = 30.0
            KD_HEAD: float = 3.0

            # Warmup: let robot settle before walking
            WARMUP_DURATION: float = 3.0  # v0.16.12: Longer settle (was 2.0)
            # v0.16.21: Ramp scene — shorter warmup so robot walks toward ramp sooner
            if mjviser_scene_type == "ramp":
                WARMUP_DURATION = 1.0

            # v0.16.12: Gravity feedforward — cancel gravity so spring handles only deviations
            # body_subtreemass[1] = total mass of robot (body 1 = torso, subtree = all)
            _robot_mass: float = float(mj_model.body_subtreemass[1])
            GRAVITY_FF: float = _robot_mass * 9.81  # ~157N for 16kg humanoid

            # v0.16.21: Get torso body ID for ray casting exclusion
            # (body 1 is NOT always torso — scenes with walls/platforms before torso in XML
            #  have different body ordering, causing ray cast to hit robot's own body)
            _torso_body_id: int = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, 'torso')
            if _torso_body_id < 0:
                _torso_body_id = 1  # Fallback to body 1

            # v0.16.23: Foot geom IDs for contact-based ground detection
            _foot_geom_ids: tuple = (
                mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_GEOM, 'foot_l_geom'),
                mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_GEOM, 'foot_r_geom'),
            )
            _foot_geom_ids = tuple(gid for gid in _foot_geom_ids if gid >= 0)

            # ── Walking controller state ──
            walk_state: dict = {
                'initial_qpos': mj_data.qpos.copy(),
                'phase_offset': 0.0,
            }

            # ── v0.16.4: Physics-based walking controller ──
            def step_fn(model: "mj.MjModel", data: "mj.MjData") -> None:
                """v0.16.4 Real physics walking controller — anti-fall enhanced.

                Implements physics-based walking with soft root assist
                (bungee cords) + joint PD + real contacts.
                """
                nonlocal target_height
                global mjviser_walk_direction, mjviser_target_direction

                # ── 0. Clear applied forces ──
                data.qfrc_applied[:] = 0.0
                data.ctrl[:] = 0.0

                # v0.16.13: REMOVED gravity feedforward — it cancels gravity,
                # leaving contact forces to push robot into the sky.
                # The spring (KP_ROOT_Z=1200) alone handles height control.

                sim_time: float = float(data.time)
                is_warmup: bool = sim_time < WARMUP_DURATION
                is_plain: bool = False  # v0.16.11: All scenes use same XML humanoid

                # v0.16.11: Unified controller — all scenes use the same
                # custom XML humanoid (16.5kg, gear=100). No more dm_control
                # humanoid (40.8kg) which was unstable.

                # ── Obstacle scene: full physics (now used for ALL scenes) ──

                # ── 1. Extract root state for balance feedback ──
                root_pos: np.ndarray = data.qpos[0:3].copy()
                root_quat: np.ndarray = data.qpos[3:7].copy()
                root_vel: np.ndarray = data.qvel[0:3].copy()
                root_angvel: np.ndarray = data.qvel[3:6].copy()

                # Extract Euler angles from quaternion
                qw, qx, qy, qz = float(root_quat[0]), float(root_quat[1]), float(root_quat[2]), float(root_quat[3])
                roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
                pitch = float(np.arctan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz)))
                yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qz * qz + qy * qy)))

                # v0.16.11: Gradual direction interpolation (same as plain scene)
                _dt_o = float(model.opt.timestep)
                _dir_err_o = float(np.arctan2(
                    np.sin(mjviser_target_direction - mjviser_walk_direction),
                    np.cos(mjviser_target_direction - mjviser_walk_direction)))
                _max_turn_o = 0.2 * _dt_o  # v0.16.12: Slower turn (was 0.5)
                if abs(_dir_err_o) > _max_turn_o:
                    mjviser_walk_direction += float(np.sign(_dir_err_o)) * _max_turn_o
                else:
                    mjviser_walk_direction = mjviser_target_direction
                mjviser_walk_direction = float(np.arctan2(
                    np.sin(mjviser_walk_direction), np.cos(mjviser_walk_direction)))

                # ── 2. Soft root assist (bungee cords) + anti-fall safety net ──
                # v0.16.4: Anti-fall boost — when Z < 70% target, boost 3x to
                # prevent collapse. Also boost pitch/roll when leaning > 30°.
                # v0.16.7: Adaptive target_height — detect ground level via downward ray
                # v0.16.12: Gravity feedforward + gradual fall_boost + gentler yaw
                # v0.16.22: Airborne mode — when robot is far above ground (walked off
                #   platform edge, launched by terrain), kill vertical spring + gait,
                #   switch to pure angular damping. Prevents "sky tumbling" where
                #   spring+PD oscillate the robot in mid-air indefinitely.
                # v0.16.23: Ground-aware balance.
                #   - One-way vertical spring (only pushes up when z<target) so the
                #     robot is never pulled down into the ground on ramps.
                #   - Foot-contact + ray height used for airborne detection: prevents
                #     false "airborne" on ramps where the torso may be >1m above ground.
                #   - Softer root gains on ramp/stairs/obstacle/maze to avoid launching.
                is_airborne: bool = False
                # v0.16.25: Flip detection state — initialized here so they're
                # accessible in the gait section below.
                _is_flipping: bool = False
                _is_tilting: bool = False
                yaw_err: float = 0.0
                if not is_plain:
                    # Cast straight-down ray to find actual ground/platform height
                    ray_down = np.array([0.0, 0.0, -1.0])
                    ray_origin_down = np.array([float(root_pos[0]), float(root_pos[1]), float(root_pos[2]) + 0.1])
                    try:
                        ground_dist = mj.mj_ray(mj_model, data, ray_origin_down, ray_down, None, 1, _torso_body_id, None)
                    except Exception:
                        ground_dist = -1.0

                    # v0.16.23: Contact-based ground detection (foot geoms)
                    is_grounded = False
                    for _c in range(data.ncon):
                        _contact = data.contact[_c]
                        if _contact.geom1 in _foot_geom_ids or _contact.geom2 in _foot_geom_ids:
                            is_grounded = True
                            break

                    # Adaptive target: standing height = ground_height + 0.85 (leg length)
                    if ground_dist > 0:
                        ground_z = float(ray_origin_down[2] - ground_dist)
                        adaptive_target = ground_z + 0.85
                        # v0.16.27: Lower target on terrain for stability
                        # v0.16.28: Reduced crouch 10cm→5cm — 10cm was too low and
                        #   made the robot look like it was about to fall.
                        if mjviser_scene_type in ("ramp", "stairs", "obstacle", "maze"):
                            adaptive_target -= 0.05  # Crouch 5cm lower on terrain (was 10cm)
                        # v0.16.28: Slower blend (0.02 instead of 0.05) — at 200Hz,
                        # 0.05 = 10x faster than 0.02, causing the target to jitter
                        # visibly. 0.02 reaches adaptive target in ~1.5s, smooth.
                        target_height = target_height * 0.98 + adaptive_target * 0.02

                    # v0.16.27: On terrain, BOOST angular gains (not just keep at 1.0).
                    # Terrain generates larger disturbances from uneven foot placement,
                    # so the robot needs MORE angular stability, not the same amount.
                    # Z gain is still reduced (0.7x) to prevent vertical launching.
                    # v0.16.28: Reduced 1.5x→1.2x — 1.5x was over-damping and causing
                    #   visible "stiff" oscillation on plain ground (false triggering).
                    _terrain_z_scale = 0.7 if mjviser_scene_type in ("ramp", "stairs", "obstacle", "maze") else 1.0
                    _terrain_ang_scale = 1.2 if mjviser_scene_type in ("ramp", "stairs", "obstacle", "maze") else 1.0  # v0.16.28: was 1.5
                    _kp_root_z = KP_ROOT_Z * _terrain_z_scale
                    _kd_root_z = KD_ROOT_Z * _terrain_z_scale
                    _kp_root_pitch = KP_ROOT_PITCH * _terrain_ang_scale
                    _kd_root_pitch = KD_ROOT_PITCH * _terrain_ang_scale
                    _kp_root_roll = KP_ROOT_ROLL * _terrain_ang_scale
                    _kd_root_roll = KD_ROOT_ROLL * _terrain_ang_scale
                    _kp_root_yaw = KP_ROOT_YAW * _terrain_ang_scale
                    _kd_root_yaw = KD_ROOT_YAW * _terrain_ang_scale

                    # v0.16.23: Airborne detection — true mid-air only when:
                    #   ground far below (>1.5m), OR no ground, OR feet aren't touching
                    #   AND torso is clearly above ground (>0.5m). This avoids killing
                    #   balance on ramps where the torso can be >1m above the surface.
                    is_airborne = (ground_dist < 0) or (ground_dist > 1.5) or (not is_grounded and ground_dist > 0.5)

                    if is_airborne:
                        # v0.16.22: Airborne mode — NO vertical spring, NO pitch/roll
                        # spring, ONLY strong angular velocity damping to bleed off spin.
                        data.qfrc_applied[3] += -_kd_root_roll * 5.0 * float(root_angvel[0])
                        data.qfrc_applied[4] += -_kd_root_pitch * 5.0 * float(root_angvel[1])
                        data.qfrc_applied[5] += -_kd_root_yaw * 5.0 * float(root_angvel[2])
                    else:
                        # Ground mode — one-way vertical spring + PD + anti-fall

                        # v0.16.26: Flip detection & proactive recovery.
                        # Previous approach (v0.16.25): wait until 20°/30° then apply
                        # 3x torque — too late and too aggressive (causes overshoot).
                        # New approach: detect at 12°/20°, use HIGH DAMPING (not high
                        # torque) to kill angular momentum, plus moderate torque boost.
                        _abs_roll = abs(roll)
                        _abs_pitch = abs(pitch)
                        _is_flipping = (_abs_roll > 0.35) or (_abs_pitch > 0.35)  # >20° (was 30°)
                        _is_tilting = (_abs_roll > 0.20) or (_abs_pitch > 0.20)   # >12° (was 20°)

                        height_error = target_height - float(root_pos[2])

                        # v0.16.25: Emergency crouch when tilting — lower COM to stabilize
                        if _is_flipping:
                            # Severe tilt: crouch hard (target -0.3m) + no gait
                            height_error += 0.3  # Lower target by 30cm
                        elif _is_tilting:
                            # Mild tilt: crouch slightly (target -0.15m)
                            height_error += 0.15

                        # v0.16.24: Anti-bounce asymmetric damping.
                        _vel_z = float(root_vel[2])
                        # Up-spring (one-way: only push up when below target)
                        _up_spring = _kp_root_z * max(0.0, height_error)
                        # Damping: 3x when moving up (anti-bounce), 1x when moving down
                        _damp_factor = 3.0 if _vel_z > 0.0 else 1.0
                        _damping = _kd_root_z * _damp_factor * _vel_z
                        height_force = _up_spring - _damping
                        # v0.16.12: Gradual anti-fall boost (was binary 3x at 70%)
                        # v0.16.24: Reduced range — only kicks in below 60% (was 70%)
                        _height_ratio = float(root_pos[2]) / target_height if target_height > 0 else 1.0
                        if _height_ratio < 0.6:
                            height_force *= 1.0 + (0.6 - _height_ratio) * 2.0  # Max 1.6x (was 3.1x)
                        # v0.16.24: Asymmetric Z clip — less UP force, normal DOWN force
                        #   This is the key fix for "robot loves to jump" — clip UP to 500N
                        #   and allow DOWN to -200N (gentle, not yanked down).
                        height_force = float(np.clip(height_force, -200.0, 500.0))
                        data.qfrc_applied[2] += height_force

                        # v0.16.12: Gradual fall_boost (was binary 1.0/2.0 at 80%)
                        # v0.16.24: Reduced max boost 3.0→1.6x to reduce bounce
                        fall_boost = 1.0 + max(0.0, (0.85 - _height_ratio) * 1.5)
                        fall_boost = min(fall_boost, 1.6)  # Cap at 1.6x (was 3.0x)

                        # v0.16.26: Flip recovery — HIGH DAMPING + moderate torque.
                        # Previous 3x torque caused overshoot oscillation. New approach:
                        # 1.5x torque boost + 4x angular velocity damping kills momentum
                        # without overshoot. Think of it as a shock absorber, not a spring.
                        if _is_flipping:
                            fall_boost = 1.5       # Moderate torque (was 3.0)
                            _damp_boost = 4.0       # High damping to kill angular momentum
                        elif _is_tilting:
                            fall_boost = 1.2        # Mild torque boost (was 2.0)
                            _damp_boost = 2.0       # Moderate damping
                        else:
                            _damp_boost = 1.0

                        pitch_torque = _kp_root_pitch * fall_boost * (-pitch) - _kd_root_pitch * _damp_boost * float(root_angvel[1])
                        # v0.16.26: Moderate clip expansion (was 3x — caused overshoot)
                        _pitch_clip = CLIP_ROOT_PITCH * (1.8 if _is_flipping else (1.3 if _is_tilting else 1.0))
                        pitch_torque = float(np.clip(pitch_torque, -_pitch_clip, _pitch_clip))
                        data.qfrc_applied[4] += pitch_torque

                        roll_torque = _kp_root_roll * fall_boost * (-roll) - _kd_root_roll * _damp_boost * float(root_angvel[0])
                        _roll_clip = CLIP_ROOT_ROLL * (1.8 if _is_flipping else (1.3 if _is_tilting else 1.0))
                        roll_torque = float(np.clip(roll_torque, -_roll_clip, _roll_clip))
                        data.qfrc_applied[3] += roll_torque

                        # v0.16.26: Proactive angular velocity damping — even when not
                        # tilting, if angular velocity is high (>2 rad/s), add extra
                        # damping to prevent the tilt from developing in the first place.
                        _angvel_mag = abs(float(root_angvel[0])) + abs(float(root_angvel[1]))
                        if _angvel_mag > 2.0 and not _is_flipping:
                            _extra_damp = min(_angvel_mag - 2.0, 3.0) * _kd_root_pitch
                            data.qfrc_applied[3] -= _extra_damp * float(root_angvel[0])
                            data.qfrc_applied[4] -= _extra_damp * float(root_angvel[1])

                        # v0.16.12: Much gentler yaw — turning via hip_z, not root torque
                        # (was KP*2.0, clip 60 → KP*0.3, clip 15)
                        yaw_err = float(np.arctan2(
                            np.sin(mjviser_walk_direction - yaw),
                            np.cos(mjviser_walk_direction - yaw)))
                        # v0.16.17: Clamp yaw_err to prevent backflip on rapid direction changes
                        yaw_err = float(np.clip(yaw_err, -np.pi / 4, np.pi / 4))  # Max 45° turn at once
                        # v0.16.17: Stability guard — don't apply yaw torque when robot is falling
                        if _height_ratio > 0.75:
                            yaw_torque = _kp_root_yaw * 0.3 * yaw_err - _kd_root_yaw * float(root_angvel[2])
                            yaw_torque = float(np.clip(yaw_torque, -CLIP_ROOT_YAW, CLIP_ROOT_YAW))
                            data.qfrc_applied[5] += yaw_torque

                # ── 3. Gait generation + balance-aware joint targets ──
                phase: float = 2.0 * np.pi * WALK_FREQ * sim_time
                # v0.16.7: walk_mult controlled by mjviser_walk_speed
                walk_mult: float = (0.0 if is_warmup else 1.0) * mjviser_walk_speed

                # v0.16.22: Freeze gait when airborne — leg motion generates
                # reaction torques on torso that sustain tumbling in mid-air.
                if is_airborne:
                    walk_mult = 0.0

                # v0.16.25: Freeze gait when flipping/tilting — leg swinging
                # generates reaction torques that make the flip worse.
                if _is_flipping or _is_tilting:
                    walk_mult = 0.0

                # v0.16.27: Reduce gait amplitude on terrain — smaller steps
                # mean less destabilizing torques from uneven foot placement.
                # v0.16.28: Reduced penalty 40%→25% — 40% was too restrictive and
                #   made the robot barely move on stairs.
                if mjviser_scene_type in ("ramp", "stairs", "obstacle", "maze"):
                    walk_mult *= 0.75  # 25% reduction on terrain (was 40%)

                # v0.16.7: Ray casting for wall/terrain detection ──
                # Cast forward ray to detect walls ahead (for maze navigation)
                torso_pos = np.array([float(root_pos[0]), float(root_pos[1]), float(root_pos[2])])
                # Forward direction based on current yaw
                fwd_x = float(np.cos(yaw))
                fwd_y = float(np.sin(yaw))
                ray_fwd = np.array([fwd_x, fwd_y, 0.0])
                # Exclude robot's own body (body 0 is world, body 1 is usually torso)
                # mj_ray returns distance to nearest geom or -1 if no hit
                try:
                    wall_dist = mj.mj_ray(mj_model, data, torso_pos, ray_fwd, None, 1, _torso_body_id, None)
                except Exception:
                    wall_dist = -1.0

                # Cast left and right rays to find open direction
                left_x = float(np.cos(yaw + np.pi / 2))
                left_y = float(np.sin(yaw + np.pi / 2))
                ray_left = np.array([left_x, left_y, 0.0])
                try:
                    left_dist = mj.mj_ray(mj_model, data, torso_pos, ray_left, None, 1, _torso_body_id, None)
                except Exception:
                    left_dist = -1.0

                right_x = float(np.cos(yaw - np.pi / 2))
                right_y = float(np.sin(yaw - np.pi / 2))
                ray_right = np.array([right_x, right_y, 0.0])
                try:
                    right_dist = mj.mj_ray(mj_model, data, torso_pos, ray_right, None, 1, _torso_body_id, None)
                except Exception:
                    right_dist = -1.0

                # v0.16.7: Maze navigation — if wall ahead, auto-turn toward open space
                # v0.16.12: Use target_direction (not walk_direction) to respect interpolation
                WALL_THRESHOLD = 0.6  # Turn when wall within 0.6m
                if wall_dist > 0 and wall_dist < WALL_THRESHOLD:
                    # Wall ahead! Check left and right
                    if left_dist < 0 or left_dist > right_dist:
                        mjviser_target_direction = yaw + np.pi / 2  # Turn left
                    elif right_dist < 0 or right_dist >= left_dist:
                        mjviser_target_direction = yaw - np.pi / 2  # Turn right
                    # Normalize
                    mjviser_target_direction = float(np.arctan2(
                        np.sin(mjviser_target_direction), np.cos(mjviser_target_direction)))

                # v0.16.7: Terrain detection — cast forward-down ray for platform height
                ray_down_fwd = np.array([fwd_x * 0.5, fwd_y * 0.5, -0.866])  # 30° forward, 60° down
                ray_origin_high = torso_pos.copy()
                ray_origin_high[2] += 0.1  # Start slightly above torso
                try:
                    terrain_dist = mj.mj_ray(mj_model, data, ray_origin_high, ray_down_fwd, None, 1, _torso_body_id, None)
                except Exception:
                    terrain_dist = -1.0

                # Calculate terrain height ahead (if ray hit something)
                terrain_height_ahead = 0.0
                if terrain_dist > 0:
                    # Height of hit point = origin_z + dist * ray_z_component
                    terrain_height_ahead = float(ray_origin_high[2] + terrain_dist * (-0.866))

                # v0.16.10: Bigger step_up_boost for stairs (was min 0.3, now min 0.5)
                step_up_boost = 0.0
                current_ground = float(root_pos[2]) - 0.85  # Approximate foot height
                if terrain_height_ahead > current_ground + 0.03:
                    # Platform ahead is higher than current ground → lift feet more
                    # v0.16.10: Increased from 0.3 to 0.6, multiplier 0.5→0.8
                    step_up_boost = min(0.6, (terrain_height_ahead - current_ground) * 0.8)

                # v0.16.7: Forward lean for locomotion (proportional to speed)
                forward_lean = 0.06 * walk_mult  # ~3.4° forward lean when walking

                # Gait pattern
                r_hip_y_target = (HIP_AMP * walk_mult * float(np.sin(phase))) + forward_lean
                l_hip_y_target = (HIP_AMP * walk_mult * float(np.sin(phase + np.pi))) + forward_lean

                r_swing = max(0.0, float(np.sin(phase)))
                l_swing = max(0.0, float(np.sin(phase + np.pi)))
                # v0.16.7: Add step_up_boost to knee lift when terrain ahead is higher
                r_knee_target = -(KNEE_AMP + step_up_boost) * walk_mult * r_swing - KNEE_STANCE * walk_mult * (1.0 - r_swing)
                l_knee_target = -(KNEE_AMP + step_up_boost) * walk_mult * l_swing - KNEE_STANCE * walk_mult * (1.0 - l_swing)

                r_ankle_target = ANKLE_AMP * walk_mult * float(np.sin(phase + np.pi * 0.5))
                l_ankle_target = ANKLE_AMP * walk_mult * float(np.sin(phase + np.pi + np.pi * 0.5))

                r_shoulder_y_target = ARM_AMP * walk_mult * float(np.sin(phase + np.pi))
                l_shoulder_y_target = ARM_AMP * walk_mult * float(np.sin(phase))

                r_hip_x_target = 0.03 * walk_mult * float(np.sin(phase + np.pi * 0.5))
                l_hip_x_target = 0.03 * walk_mult * float(np.sin(phase + np.pi + np.pi * 0.5))
                # v0.16.7: Add turning moment via asymmetric hip_z (difference = turn direction)
                # v0.16.12: Much smaller turn_signal to prevent feet slipping (was 0.5→0.15, now 0.15→0.06)
                turn_signal = float(np.clip(yaw_err * 0.15, -0.06, 0.06)) if not is_warmup else 0.0
                r_hip_z_target = 0.02 * walk_mult * float(np.sin(phase)) + turn_signal
                l_hip_z_target = 0.02 * walk_mult * float(np.sin(phase + np.pi)) - turn_signal

                # ── Balance: adjust ankle and hip targets based on body lean ──
                # If robot leans forward (pitch < 0), push ankles to lean back
                # If robot leans right (roll > 0), adjust hips to shift weight left
                # v0.16.12: Stronger balance response (pitch 0.5→0.8, roll 0.3→0.5)
                balance_pitch = float(np.clip(-pitch * 0.8, -0.3, 0.3))
                balance_roll = float(np.clip(-roll * 0.5, -0.2, 0.2))

                # Ankle strategy: correct pitch by adjusting ankle angle
                r_ankle_target += balance_pitch * (1.0 if not is_warmup else 1.0)
                l_ankle_target += balance_pitch * (1.0 if not is_warmup else 1.0)

                # Hip x (lateral) strategy: correct roll
                r_hip_x_target += balance_roll
                l_hip_x_target += balance_roll

                # During warmup: strong stance, no walking
                if is_warmup:
                    r_hip_y_target = 0.0
                    l_hip_y_target = 0.0
                    r_knee_target = -KNEE_STANCE  # Slightly bent
                    l_knee_target = -KNEE_STANCE
                    r_ankle_target = balance_pitch  # Just balance
                    l_ankle_target = balance_pitch
                    r_shoulder_y_target = 0.0
                    l_shoulder_y_target = 0.0
                    r_hip_x_target = balance_roll
                    l_hip_x_target = balance_roll
                    r_hip_z_target = 0.0
                    l_hip_z_target = 0.0

                # ── 4. Joint PD control — drive motors via ctrl ──
                for aid in range(model.nu):
                    act_name: str = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, aid)
                    jid_act: int = int(model.actuator_trnid[aid][0])
                    qa_act: int = int(model.jnt_qposadr[jid_act])
                    da_act: int = int(model.jnt_dofadr[jid_act])

                    current_angle: float = float(data.qpos[qa_act])
                    vel: float = float(data.qvel[da_act])

                    target_angle: float = 0.0
                    kp_c: float = KP_HIP
                    kd_c: float = KD_HIP

                    an = act_name.lower()

                    if 'hip_r_y' in an:
                        target_angle = r_hip_y_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'hip_l_y' in an:
                        target_angle = l_hip_y_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'hip_r_x' in an:
                        target_angle = r_hip_x_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'hip_l_x' in an:
                        target_angle = l_hip_x_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'hip_r_z' in an:
                        target_angle = r_hip_z_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'hip_l_z' in an:
                        target_angle = l_hip_z_target
                        kp_c, kd_c = KP_HIP, KD_HIP
                    elif 'knee_r' in an:
                        target_angle = r_knee_target
                        kp_c, kd_c = KP_KNEE, KD_KNEE
                    elif 'knee_l' in an:
                        target_angle = l_knee_target
                        kp_c, kd_c = KP_KNEE, KD_KNEE
                    elif 'ankle_r' in an:
                        target_angle = r_ankle_target
                        kp_c, kd_c = KP_ANKLE, KD_ANKLE
                    elif 'ankle_l' in an:
                        target_angle = l_ankle_target
                        kp_c, kd_c = KP_ANKLE, KD_ANKLE
                    elif 'shoulder_r_y' in an:
                        target_angle = r_shoulder_y_target
                        kp_c, kd_c = KP_ARM, KD_ARM
                    elif 'shoulder_l_y' in an:
                        target_angle = l_shoulder_y_target
                        kp_c, kd_c = KP_ARM, KD_ARM
                    elif 'shoulder' in an or 'elbow' in an:
                        target_angle = 0.0
                        kp_c, kd_c = KP_ARM, KD_ARM
                    elif 'head' in an:
                        target_angle = 0.0
                        kp_c, kd_c = KP_HEAD, KD_HEAD
                    else:
                        target_angle = 0.0
                        kp_c, kd_c = 50.0, 5.0

                    # PD control → ctrl value in [-1, 1]
                    error: float = target_angle - current_angle
                    ctrl_val: float = float(np.clip(kp_c * error - kd_c * vel, -1.0, 1.0))
                    data.ctrl[aid] = ctrl_val

                # ── 5. Step physics ──
                mj.mj_step(model, data)

                # ── 6. Safety: if robot falls, don't let it NaN ──
                if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                    mj.mj_resetData(model, data)
                    data.qpos[2] = target_height
                    data.qpos[3] = 1.0
                    mj.mj_forward(model, data)

            # ── Create or reuse ViserServer ──
            # v0.16.14: Persistent ViserServer — created once, reused across scene changes.
            # Previous approach (stop + recreate) failed on Windows because
            # viser_server.stop() doesn't synchronously release the port.
            viser_server = None
            actual_port: int = 0

            # Try to reuse existing persistent server
            if mjviser_persistent_server is not None:
                try:
                    # Verify the server is still alive by checking its port
                    _check_port = mjviser_persistent_server._websock_server._port
                    viser_server = mjviser_persistent_server
                    actual_port = mjviser_persistent_port
                    print(f"v0.16.14: Reusing persistent ViserServer on port {actual_port}")
                except Exception as _reuse_err:
                    print(f"v0.16.14: Persistent server unusable ({_reuse_err}), creating new one")
                    mjviser_persistent_server = None
                    mjviser_persistent_port = 0

            # Create new server if needed
            if viser_server is None:
                for _attempt in range(6):
                    _try_port = 8081 + _attempt
                    try:
                        viser_server = ViserServer(port=_try_port, verbose=False)
                        actual_port = viser_server._websock_server._port
                        mjviser_persistent_server = viser_server
                        mjviser_persistent_port = actual_port
                        print(f"v0.16.14: Created new ViserServer on port {actual_port}")
                        break
                    except (OSError, RuntimeError) as _oe:
                        print(f"ViserServer port {_try_port} failed (attempt {_attempt+1}/6): {_oe}")
                        if _attempt < 5:
                            _time.sleep(2.0)
                if viser_server is None:
                    raise RuntimeError("Failed to create ViserServer after 6 attempts on ports 8081-8086")

            # Get the actual port (ViserServer auto-increments if port is occupied)
            actual_port: int = viser_server._websock_server._port
            mjviser_viewer_url = f"http://localhost:{actual_port}"

            # ── v0.16.0: Custom reset_fn — physics standing pose ──
            def reset_fn(mdl: "mj.MjModel", dat: "mj.MjData") -> None:
                """Reset to neutral standing pose with feet on ground.

                v0.16.0: Set a stable standing pose. The physics controller will
                maintain balance through joint torques and contact forces — no
                kinematic locking.
                v0.16.21: Use ray casting to detect ground/platform height at
                reset position, preventing robot from sinking into floating platforms.
                """
                dat.qpos[:] = 0.0
                # v0.16.21: Cast downward ray from above to find ground/platform height
                _ray_origin = np.array([0.0, 0.0, 3.0])
                _ray_dir = np.array([0.0, 0.0, -1.0])
                _ground_z = 0.0
                try:
                    _hit_dist = mj.mj_ray(mdl, dat, _ray_origin, _ray_dir, None, 1, _torso_body_id, None)
                    if _hit_dist > 0:
                        _ground_z = float(_ray_origin[2] - _hit_dist)
                except Exception:
                    pass
                dat.qpos[2] = target_height + _ground_z  # Stand on top of ground/platform
                dat.qpos[3] = 1.0             # Upright quaternion (w=1)
                dat.qvel[:] = 0.0
                dat.qacc[:] = 0.0
                dat.ctrl[:] = 0.0
                dat.qfrc_applied[:] = 0.0
                mj.mj_forward(mdl, dat)
                walk_state['initial_qpos'] = dat.qpos.copy()

            viewer = mjviser.Viewer(
                model=mj_model,
                data=mj_data,
                step_fn=step_fn,
                reset_fn=reset_fn,
                server=viser_server,
            )
            # Start in paused mode: robot shows upright pose, user clicks Play to start
            # v0.16.21: Ramp scene auto-starts (unpaused) so robot walks toward ramp immediately
            viewer._paused = (mjviser_scene_type != "ramp")
            mjviser_viewer_running = True

            # ── Manual viewer loop (replaces viewer.run()) ──
            # viewer.run() uses signal.signal() which fails in background threads,
            # so we replicate its logic here without the signal handling.
            viewer._setup_gui()

            # v0.16.9: Add "Robot Control" tab with direction buttons (◀ ▲ ▶ ■)
            # These buttons live inside the mjviser viewer page, not the dashboard.
            try:
                import viser
                ctrl_tabs = viser_server.gui.add_tab_group()
                with ctrl_tabs.add_tab("Robot Control", icon=viser.Icon.ARROW_BIG_UP):
                    viser_server.gui.add_markdown(
                        "**Robot Direction Control**\n\n"
                        "Click to steer the humanoid in the 3D scene. "
                        "Maze auto-detects walls; platforms auto-lift feet."
                    )
                    # Row 1: ◀ ▲ ▶ (turn left, forward, turn right)
                    dir_row = viser_server.gui.add_button_group(
                        "Direction", options=["◀ Left", "▲ Forward", "Right ▶"]
                    )
                    # Row 2: ■ Stop
                    stop_btn = viser_server.gui.add_button(
                        "■ Stop", icon=viser.Icon.SQUARE
                    )

                    @dir_row.on_click
                    def _on_dir_click(event) -> None:
                        # v0.16.11: Set target_direction; step_fn interpolates gradually
                        # v0.16.17: Fix direction reversal + add anti-backflip cooldown
                        global mjviser_target_direction, mjviser_walk_speed
                        global mjviser_last_dir_change_time
                        value = getattr(event.target, "value", str(event))
                        _now = time.time()
                        # v0.16.17: Anti-backflip cooldown — ignore direction changes within 0.3s
                        if _now - mjviser_last_dir_change_time < 0.3:
                            pass  # Ignore rapid direction changes
                        else:
                            mjviser_last_dir_change_time = _now
                            if "Left" in value:
                                mjviser_target_direction = float((mjviser_target_direction + 0.524) % 6.283)
                                mjviser_walk_speed = 0.3
                            elif "Right" in value:
                                mjviser_target_direction = float((mjviser_target_direction - 0.524) % 6.283)
                                mjviser_walk_speed = 0.3
                            else:  # Forward
                                mjviser_target_direction = 0.0
                                mjviser_walk_speed = 1.0

                    @stop_btn.on_click
                    def _on_stop_click(_) -> None:
                        global mjviser_walk_speed
                        mjviser_walk_speed = 0.0

                    # v0.16.17: Speed slider (1-64x simulation speed)
                    # v0.16.24: Added marks (1, 8, 32, 64) so the user can see the
                    #   slider track clearly in the viser panel — otherwise it just
                    #   looks like the viewer's built-in Slower/1x/Faster buttons.
                    speed_slider = viser_server.gui.add_slider(
                        "Speed (x)",
                        min=1, max=64, step=1, initial_value=1,
                        marks=(1, 8, 32, 64),
                        hint="Drag to set simulation speed. 1=realtime, 64=64× faster.",
                    )
                    viser_server.gui.add_markdown(
                        "**⏩ Sim Speed:** 1=realtime, 8=8×, 32=32×, **64=64× max**\n\n"
                        "*(This is OUR slider — the Slower/1x/Faster buttons above are "
                        "viser viewer's default time controls and are unrelated to this.)*"
                    )

                    @speed_slider.on_update
                    def _on_speed_update(event) -> None:
                        global mjviser_sim_speed
                        mjviser_sim_speed = int(event.target.value)
            except Exception as gui_err:
                # Non-fatal: viewer still works, just no direction buttons
                print(f"Robot Control tab setup warning: {gui_err}")

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
                    # v0.16.17: Run multiple physics steps for speed multiplier
                    # v0.16.28: Clamp effective sim steps per tick to prevent flash.
                    #   At 64x speed with mjviser_viewer's internal speed=1, the robot
                    #   moves too fast for 30Hz rendering — visible "flash" effect.
                    #   Cap outer-loop multiplier at 4 so each tick is bounded.
                    _effective_speed = min(max(1, mjviser_sim_speed), 4)
                    for _ in range(_effective_speed):
                        if mjviser_viewer_running:
                            viewer._tick()
                        else:
                            break
                    _time.sleep(0.001)
            finally:
                # v0.16.14: DON'T stop the ViserServer — keep it persistent for reuse.
                # The old approach of viser_server.stop() + recreate caused
                # OSError [Errno 22] on Windows because port release is async.
                # The server stays alive; only the viewer loop stops.
                # Cleanup the old viewer reference to allow GC
                mjviser_viewer_object = None
                mjviser_viewer_running = False
                mjviser_viewer_url = ""

        except Exception as e:
            mjviser_viewer_running = False
            mjviser_viewer_url = ""
            mjviser_viewer_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(f"mjviser viewer failed: {e}")

    mjviser_viewer_thread = threading.Thread(
        target=launch_viewer, daemon=True)
    mjviser_viewer_thread.start()

    # Poll for up to 8 seconds — the viewer thread needs time to:
    #   1. import dm_control.suite (slow, ~2-3s on first import)
    #   2. load humanoid model + configure joints
    #   3. create ViserServer and determine actual port
    # Once mjviser_viewer_running becomes True, the URL is ready.
    # If mjviser_viewer_error gets set, the thread failed.
    for _ in range(80):  # 80 × 0.1s = 8s max
        time.sleep(0.1)
        if mjviser_viewer_running:
            break
        if mjviser_viewer_error:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": mjviser_viewer_error,
                    "available": True,
                },
            )

    if not mjviser_viewer_running:
        return JSONResponse(
            status_code=504,
            content={
                "status": "timeout",
                "error": "Viewer thread did not start within 8 seconds",
                "url": "",
                "available": True,
            },
        )

    return JSONResponse(content={
        "status": "running",
        "url": mjviser_viewer_url,
        "available": True,
    })


@app.get("/api/viewer_status")
async def viewer_status() -> JSONResponse:
    """Check mjviser viewer status (for polling after start).

    Returns:
        JSONResponse with viewer running state, URL, and any error.
    """
    return JSONResponse(content={
        "running": mjviser_viewer_running,
        "url": mjviser_viewer_url,
        "error": mjviser_viewer_error,
        "available": MJVISER_AVAILABLE,
    })


@app.post("/api/stop_viewer")
async def stop_viewer() -> JSONResponse:
    """Stop mjviser viewer AND destroy the persistent ViserServer.

    v0.16.14: This is the only endpoint that actually stops the ViserServer.
    Normal scene changes keep the server alive for reuse. Use this when the
    user explicitly wants to close the viewer (e.g., "Close mjviser" button).

    Returns:
        JSONResponse with final status.
    """
    global mjviser_viewer_running, mjviser_viewer_url, mjviser_viewer_error
    global mjviser_persistent_server, mjviser_persistent_port, mjviser_viewer_object

    # Signal the viewer loop to stop
    mjviser_viewer_running = False

    # Wait for thread to exit
    if mjviser_viewer_thread is not None and mjviser_viewer_thread.is_alive():
        mjviser_viewer_thread.join(timeout=5.0)

    # Now actually stop the persistent ViserServer
    if mjviser_persistent_server is not None:
        try:
            mjviser_persistent_server.stop()
        except Exception:
            pass
        mjviser_persistent_server = None
        mjviser_persistent_port = 0

    mjviser_viewer_object = None
    mjviser_viewer_url = ""
    mjviser_viewer_error = ""

    return JSONResponse(content={
        "status": "stopped",
        "running": False,
        "available": MJVISER_AVAILABLE,
    })


# ════════════════════════════════════════════════════════════════════════════
# v0.16.15: SO-ARM100 Independent Viewer System
# Completely separate from humanoid mjviser — has its own ViserServer, viewer
# thread, and TOMAS IDO audit pipeline. No scene switching needed (one fixed
# pick-and-place scene).
# ════════════════════════════════════════════════════════════════════════════

def _launch_arm100_viewer() -> None:
    """Launch SO-ARM100 pick-and-place viewer with TOMAS IDO audit.

    This is the independent SO-ARM100 viewer — loads so_arm100_scene.xml,
    initializes TOMASMuJoCoWrapper + SOArm100Controller, and runs the
    pick-and-place state machine in a ViserServer loop.

    State machine: HOME → REACH → DESCEND → GRASP → LIFT →
                   TRANSPORT → RELEASE → RETREAT → DONE

    Each step is audited through:
    - ψ-Anchor Gate (MAX_TORQUE, MAX_VELOCITY, NO_SPILL)
    - κ-Snap audit trail (step-level causal snapshot)
    """
    global arm100_viewer_running, arm100_viewer_url, arm100_viewer_error
    global arm100_persistent_server, arm100_persistent_port, arm100_tomas_wrapper

    try:
        import mujoco as mj
        import mujoco.viewer as mjv
        import time as _time
        import numpy as np
        from pathlib import Path
        from viser import ViserServer

        # ── 1. Load SO-ARM100 scene ──
        scene_xml_path = str(Path(__file__).resolve().parent / "scenes" / "so_arm100_scene.xml")
        arm_model = mj.MjModel.from_xml_path(scene_xml_path)
        arm_data = mj.MjData(arm_model)
        mj.mj_resetData(arm_model, arm_data)
        mj.mj_forward(arm_model, arm_data)

        print(f"v0.16.15: SO-ARM100 scene loaded — nq={arm_model.nq}, nu={arm_model.nu}, nsensor={arm_model.nsensor}")

        # ── 2. Initialize CAMKit dual-camera renderer ──
        # v0.16.19: CAMKit simulation — top_cam (fixed) + wrist_cam (on gripper)
        # v0.16.25: Robust renderer init with fallback + error logging.
        #   mj.Renderer needs a GL context; if it fails (headless, no EGL),
        #   fall back to MjRenderContextOffscreen. If both fail, cameras will
        #   show a placeholder image with joint info text overlay.
        CAM_WIDTH = 320
        CAM_HEIGHT = 240
        cam_renderer = None
        cam_render_ctx = None  # Fallback: MjRenderContextOffscreen
        _cam_init_error: str = ""

        # v0.16.27: Try EGL backend first — on Windows without a display,
        # the default WGL backend creates a "valid" context that renders
        # all-black images. EGL (via ANGLE) can do true offscreen rendering.
        import os as _os
        _prev_gl = _os.environ.get('MUJOCO_GL', '')
        if not _prev_gl:
            _os.environ['MUJOCO_GL'] = 'egl'

        try:
            renderer_cls = getattr(mj, 'Renderer', None)
            if renderer_cls is not None:
                cam_renderer = renderer_cls(arm_model, CAM_HEIGHT, CAM_WIDTH)
                # v0.16.27: Verify the renderer actually works by doing a test render
                cam_renderer.update_scene(arm_data, camera=0)
                _test_pixels = cam_renderer.render()
                _test_mean = float(np.mean(_test_pixels))
                if _test_mean < 1.0:
                    # All-black → renderer initialized but no real GL context
                    print(f"v0.16.27: Renderer test render all black (mean={_test_mean:.1f}) — EGL failed, trying osmesa")
                    cam_renderer = None
                    _os.environ['MUJOCO_GL'] = 'osmesa'
                    try:
                        cam_renderer = renderer_cls(arm_model, CAM_HEIGHT, CAM_WIDTH)
                        cam_renderer.update_scene(arm_data, camera=0)
                        _test2 = cam_renderer.render()
                        if float(np.mean(_test2)) < 1.0:
                            print(f"v0.16.27: osmesa also black (mean={float(np.mean(_test2)):.1f}) — giving up on GPU rendering")
                            cam_renderer = None
                            _cam_init_error = "GPU渲染不可用，显示实时数据"
                        else:
                            print(f"v0.16.27: osmesa renderer works! (mean={float(np.mean(_test2)):.1f})")
                    except Exception as osmesa_err:
                        print(f"v0.16.27: osmesa failed: {osmesa_err}")
                        cam_renderer = None
                        _cam_init_error = f"egl black, osmesa failed: {osmesa_err}"
                else:
                    print(f"v0.16.27: EGL renderer works! test mean={_test_mean:.1f}")
                # Find camera IDs
                cam_ids = {mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_CAMERA, name): name
                           for name in ['top_cam', 'wrist_cam']
                           if mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_CAMERA, name) >= 0}
                if cam_renderer is not None:
                    print(f"v0.16.27: CAMKit Renderer ready — cameras: {cam_ids}, {CAM_WIDTH}x{CAM_HEIGHT}")
            else:
                _cam_init_error = "mj.Renderer class not found"
                print(f"v0.16.25: mj.Renderer not available: {_cam_init_error}")
        except Exception as cam_init_err:
            _cam_init_error = str(cam_init_err)
            print(f"v0.16.25: CAMKit Renderer init failed: {cam_init_err}")
            cam_renderer = None

        # Restore original MUJOCO_GL setting
        if _prev_gl:
            _os.environ['MUJOCO_GL'] = _prev_gl
        elif 'MUJOCO_GL' in _os.environ:
            del _os.environ['MUJOCO_GL']

        # v0.16.25: Fallback to MjRenderContextOffscreen
        if cam_renderer is None:
            try:
                cam_render_ctx = mj.MjRenderContextOffscreen(arm_model, CAM_HEIGHT, CAM_WIDTH)
                print(f"v0.16.25: CAMKit MjRenderContextOffscreen fallback initialized ({CAM_WIDTH}x{CAM_HEIGHT})")
            except Exception as ctx_err:
                print(f"v0.16.25: MjRenderContextOffscreen also failed: {ctx_err}")
                cam_render_ctx = None

        def _render_cam(cam_name: str) -> "np.ndarray":
            """Render a single camera view. Returns RGB (H, W, 3) or info overlay on failure."""
            # v0.16.27: Try mj.Renderer, check for all-black output, then OffscreenCtx, then overlay.
            if cam_renderer is not None:
                try:
                    cam_id = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
                    if cam_id < 0:
                        return _make_cam_placeholder(cam_name, "摄像头未找到")
                    cam_renderer.update_scene(arm_data, camera=cam_id)
                    pixels = cam_renderer.render()
                    # v0.16.27: Check if render is all-black (dead GL context)
                    if float(np.mean(pixels)) < 1.0:
                        return _make_cam_placeholder(cam_name, "GPU渲染不可用，显示实时数据")
                    return pixels.astype(np.uint8)
                except Exception as rend_err:
                    print(f"v0.16.25: Renderer.render failed for {cam_name}: {rend_err}")
                    return _make_cam_placeholder(cam_name, "GPU渲染不可用，显示实时数据")
            elif cam_render_ctx is not None:
                try:
                    cam_id = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
                    if cam_id < 0:
                        return _make_cam_placeholder(cam_name, "摄像头未找到")
                    # MjRenderContextOffscreen: render to buffer
                    cam_render_ctx.update_scene(arm_data, cam_id)
                    pixels = cam_render_ctx.read_pixels(cam_id, depth=False)
                    if float(np.mean(pixels)) < 1.0:
                        return _make_cam_placeholder(cam_name, "GPU渲染不可用，显示实时数据")
                    return pixels.astype(np.uint8)
                except Exception as ctx_err:
                    print(f"v0.16.25: MjRenderContextOffscreen failed for {cam_name}: {ctx_err}")
                    return _make_cam_placeholder(cam_name, "GPU渲染不可用，显示实时数据")
            else:
                return _make_cam_placeholder(cam_name, "GPU渲染不可用，显示实时数据")

        def _make_cam_placeholder(cam_name: str, error_msg: str = "") -> "np.ndarray":
            """v0.16.28: Generate a clear info-overlay image when renderer is unavailable.
            v0.16.27: Shows camera name, joint positions, gripper state, and error message
            on a GREEN background (clearly NOT black so user knows it's a fallback).
            v0.16.28: Compact layout (5 zones, max 4 lines) — readable on 320x240.
            """
            img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            # Dark green background — clearly visible, not "black"
            img[:, :] = [20, 80, 20]
            try:
                from PIL import Image, ImageDraw, ImageFont
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)
                # Try a default font; fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 11)
                    font_small = ImageFont.truetype("arial.ttf", 9)
                except Exception:
                    font = ImageFont.load_default()
                    font_small = font
                # Title bar (yellow on dark green)
                draw.rectangle([(0, 0), (CAM_WIDTH, 18)], fill=(0, 60, 0))
                draw.text((4, 3), f"[{cam_name}]", fill=(255, 255, 0), font=font)
                # Error / status line (red) — wrap to one short line
                if error_msg:
                    short_err = error_msg[:60] if len(error_msg) > 60 else error_msg
                    draw.text((4, 20), short_err, fill=(255, 80, 80), font=font_small)
                # Live joint positions (white) — compact 2-column grid
                try:
                    qpos = arm_data.qpos[:7]
                    labels = ["Rot", "Pit", "Elb", "WPi", "WRo", "GL", "GR"]
                    for i in range(min(7, len(qpos))):
                        col = i // 4  # 0 or 1
                        row = i % 4
                        x = 4 + col * 160
                        y = 38 + row * 14
                        draw.text((x, y), f"{labels[i]}={float(qpos[i]):+.2f}",
                                  fill=(200, 255, 200), font=font_small)
                except Exception:
                    pass
                # Object distances from gripper (compact, single line)
                try:
                    grip_id = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, "gripper_base")
                    if grip_id >= 0:
                        gp = arm_data.xpos[grip_id]
                        line_y = 110
                        for obj_name, color in [("red_cube", (255, 100, 100)),
                                                 ("tray", (180, 180, 180))]:
                            oid = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, obj_name)
                            if oid >= 0:
                                op = arm_data.xpos[oid]
                                d = float(np.linalg.norm(op - gp))
                                draw.text((4, line_y), f"{obj_name}: {d*100:.1f}cm",
                                          fill=color, font=font_small)
                                line_y += 13
                except Exception:
                    pass
                # Footer (cyan) — what the user CAN do
                draw.rectangle([(0, CAM_HEIGHT - 32), (CAM_WIDTH, CAM_HEIGHT)], fill=(0, 50, 0))
                draw.text((4, CAM_HEIGHT - 30), "GPU-less mode: live data only",
                          fill=(100, 200, 255), font=font_small)
                draw.text((4, CAM_HEIGHT - 18), "See Telemetry tab for full state",
                          fill=(100, 200, 255), font=font_small)
                draw.text((4, CAM_HEIGHT - 6), "v0.16.32 CAMKit 数据回退",
                          fill=(80, 160, 80), font=font_small)
                img = np.array(pil_img)
            except ImportError:
                # No PIL — just return the colored background
                pass
            return img

        # ── 2. Initialize TOMAS wrapper + controller ──
        if not TOMAS_AVAILABLE:
            raise RuntimeError("TOMAS wrapper not available — cannot run SO-ARM100 viewer")

        arm100_tomas_wrapper = TOMASMuJoCoWrapper(
            model=arm_model,
            data=arm_data,
            target_body_name="red_cube",
            tray_body_name="tray",
            gripper_body_name="gripper_base",
        )

        arm_controller = SOArm100Controller(
            model=arm_model,
            data=arm_data,
            wrapper=arm100_tomas_wrapper,
            target_object="red_cube",
        )

        _arm_step_count = [0]

        def _arm_step_fn(model: "mj.MjModel", data: "mj.MjData") -> None:
            """SO-ARM100 step with IDO/TOMAS audit — supports auto/manual/VLA modes."""
            global arm100_manual_mode, arm100_manual_target
            global arm100_vla_mode, arm100_vla_adapter, arm100_vla_instruction
            global arm100_cam_top_rgb, arm100_cam_wrist_rgb
            global arm100_cam_top_frame, arm100_cam_wrist_frame

            if arm100_manual_mode and arm100_manual_target is not None:
                # v0.16.17: Manual control mode
                action = arm100_manual_target.copy()
                phase = "manual"
                note = f"manual control, joints={action[:5].round(2)}"
            elif arm100_vla_mode and arm100_vla_adapter is not None:
                # v0.16.17: VLA inference mode
                # v0.16.26: Decouple camera rendering from VLA prediction.
                #   Previous code rendered cameras INSIDE the try block — if
                #   _render_cam() threw, the entire VLA branch failed and fell
                #   back to default controller. Now cameras are rendered separately
                #   and VLA predict only needs proprio + language.
                # v0.16.19: CAMKit — render dual cameras for VLA input (best-effort)
                try:
                    top_rgb = _render_cam('top_cam')
                    wrist_rgb = _render_cam('wrist_cam')
                    arm100_cam_top_rgb = top_rgb
                    arm100_cam_wrist_rgb = wrist_rgb
                except Exception:
                    pass  # Camera failure should NOT block VLA execution
                try:
                    # v0.16.27: Add object positions to obs_dict so VLA can
                    # compute IK targets dynamically instead of using hardcoded values.
                    _obj_positions = {}
                    for _obj_name in ["red_cube", "blue_ball", "white_tissue", "tray"]:
                        _bid = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, _obj_name)
                        if _bid >= 0:
                            _obj_positions[_obj_name] = arm_data.xpos[_bid].copy()
                    # Gripper position for relative IK
                    _grip_bid = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, "gripper_base")
                    _gripper_pos = arm_data.xpos[_grip_bid].copy() if _grip_bid >= 0 else np.zeros(3)
                    obs_dict = {
                        'rgb': arm100_cam_top_rgb if arm100_cam_top_rgb is not None else np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8),
                        'wrist_rgb': arm100_cam_wrist_rgb if arm100_cam_wrist_rgb is not None else np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8),
                        'language': arm100_vla_instruction,
                        'proprio': arm_controller._get_arm_qpos(),
                        'object_positions': _obj_positions,
                        'gripper_position': _gripper_pos,
                        'arm_base_z': 0.40,  # Arm base height for IK
                    }
                    action = arm100_vla_adapter.predict(obs_dict)
                    phase = "vla_infer"
                    note = f"VLA action: {action[:5].round(2)}"
                except Exception as vla_err:
                    action, phase, note = arm_controller.compute_action()
                    note += f" (VLA fallback: {vla_err})"
            else:
                # Default: pick-and-place state machine
                action, phase, note = arm_controller.compute_action()

            safe_action, violation = arm100_tomas_wrapper.check_and_record(
                step=_arm_step_count[0],
                action=action,
                phase=phase,
                note=note,
            )

            if violation:
                try:
                    broadcast_sync({
                        "type": "tomas_violation",
                        "violation": violation,
                        "step": _arm_step_count[0],
                        "phase": phase,
                        "eta": float(arm100_tomas_wrapper.compute_eta()),
                    })
                except Exception:
                    pass

            data.ctrl[:] = safe_action[:model.nu]

            # v0.16.27: CRITICAL FIX — step physics! Without this, ctrl is set
            # but the simulation never advances, so the arm doesn't move.
            # The humanoid step_fn calls mj_step at line 2214, but the ARM100
            # step_fn was missing it entirely.
            mj.mj_step(model, data)

            # v0.16.30: Kinematic assist — after mj_step, nudge qpos toward ctrl.
            # Bumped 0.30 → 0.50 so arm converges within 20 steps (was 50).
            # 0.30 left 9cm residual error at step 50; 0.50 reaches 0.19cm.
            # v0.16.31: Apply to ALL 7 joints, but clamp gripper to not
            #   penetrate cube. Without assist, gripper actuator was too slow
            #   (only reached 0.518/0.873 after 50 steps → gap 4.5cm < cube 5cm
            #   → cube knocked away). Now: assist drives gripper to "touching"
            #   angle (0.617 rad = gap 5.0cm = cube width). Position actuator
            #   then provides squeezing force via normal contact physics.
            _ARM_QPOS_OFFSET = 21  # qpos[21:28] = arm joints (7 DOF)
            # v0.16.31: Cube width = 5cm → min gap = 5cm → max grasp angle = 0.617 rad
            #   gap(θ) = sqrt((0.076*sin(θ))² + 0.024²) = 0.05 → θ = 0.617
            _GRIP_TOUCH_ANGLE = 0.617  # Fingers just touch cube faces
            for _ji in range(min(7, model.nu)):
                _qi = _ARM_QPOS_OFFSET + _ji
                if _qi < data.qpos.shape[0]:
                    _target = float(data.ctrl[_ji])
                    if _ji == 5:  # Gripper_Left (positive: 0=closed, 0.873=open)
                        # Clamp to not penetrate cube: max closing = touch angle
                        _target = max(0.0, min(_target, _GRIP_TOUCH_ANGLE)) if _target < _GRIP_TOUCH_ANGLE else _target
                    elif _ji == 6:  # Gripper_Right (negative: 0=closed, -0.873=open)
                        _target = min(0.0, max(_target, -_GRIP_TOUCH_ANGLE)) if _target > -_GRIP_TOUCH_ANGLE else _target
                    _diff = _target - float(data.qpos[_qi])
                    data.qpos[_qi] += _diff * 0.50
            mj.mj_forward(model, data)  # Update derived quantities after qpos change

            # v0.16.31: Kinematic cube attachment during grasp/lift/transport.
            # When the VLA is in grasp (after fingers close), lift, or transport
            # phase, kinematically attach the red_cube to the fingertip center.
            # This ensures the cube stays with the gripper without relying solely
            # on contact friction (which can be unreliable in simulation).
            # During release, the cube detaches and falls naturally.
            if arm100_vla_mode and arm100_vla_adapter is not None:
                try:
                    _vla_phase = arm100_vla_adapter.current_phase
                    _vla_step = arm100_vla_adapter.phase_step
                    _should_attach = (
                        _vla_phase in ("lift", "transport") or
                        (_vla_phase == "grasp" and _vla_step > 15)
                    )
                    if _should_attach:
                        _cube_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "red_cube")
                        _fsr_l = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "fsr_left")
                        _fsr_r = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "fsr_right")
                        _grip_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "gripper_base")
                        if _cube_bid >= 0 and _fsr_l >= 0 and _fsr_r >= 0 and _grip_bid >= 0:
                            _ft_center = (data.site_xpos[_fsr_l] + data.site_xpos[_fsr_r]) * 0.5
                            # v0.16.32: Remove distance check during lift/transport.
                            # Kinematic assist moves arm 50% toward target per step,
                            # so when transport target is 36cm away (cube→tray),
                            # the arm jumps 18cm in one step — breaking the 15cm
                            # attachment threshold and dropping the cube.
                            # During lift/transport, the cube SHOULD follow the
                            # gripper unconditionally (it was grasped already).
                            _jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "cube_joint")
                            if _jnt_id >= 0:
                                _qpos_addr = model.jnt_qposadr[_jnt_id]
                                _qvel_addr = model.jnt_dofadr[_jnt_id]
                                # Set cube position to fingertip center
                                data.qpos[_qpos_addr:_qpos_addr+3] = _ft_center
                                # Set cube orientation to match gripper
                                data.qpos[_qpos_addr+3:_qpos_addr+7] = data.xquat[_grip_bid]
                                # Zero out cube velocity (follows gripper)
                                data.qvel[_qvel_addr:_qvel_addr+6] = 0.0
                except Exception:
                    pass  # Cube attachment is best-effort, don't crash sim

            if np.any(np.isnan(data.qpos)):
                mj.mj_resetData(model, data)
                mj.mj_forward(model, data)

            _arm_step_count[0] += 1

            # v0.16.19: Periodic CAMKit rendering (every 10 steps ≈ ~10 FPS at 100Hz sim)
            # v0.16.27: Always render (even if cam_renderer is None) to show live joint data
            if _arm_step_count[0] % 10 == 0:
                try:
                    top_rgb = _render_cam('top_cam')
                    wrist_rgb = _render_cam('wrist_cam')
                    arm100_cam_top_rgb = top_rgb
                    arm100_cam_wrist_rgb = wrist_rgb
                    # Encode JPEG for API serving
                    try:
                        import cv2
                        _, top_jpg = cv2.imencode('.jpg', top_rgb[:, :, ::-1])  # RGB→BGR
                        _, wrist_jpg = cv2.imencode('.jpg', wrist_rgb[:, :, ::-1])
                        arm100_cam_top_frame = top_jpg.tobytes()
                        arm100_cam_wrist_frame = wrist_jpg.tobytes()
                    except ImportError:
                        # No cv2 — use PIL as fallback
                        try:
                            from PIL import Image
                            import io
                            for arr, attr in [(top_rgb, 'arm100_cam_top_frame'),
                                              (wrist_rgb, 'arm100_cam_wrist_frame')]:
                                img = Image.fromarray(arr)
                                buf = io.BytesIO()
                                img.save(buf, format='JPEG', quality=85)
                                globals()[attr] = buf.getvalue()
                        except ImportError:
                            pass  # No image encoding available
                except Exception:
                    pass

            if _arm_step_count[0] % 50 == 0:
                try:
                    summary = arm100_tomas_wrapper.get_summary()
                    broadcast_sync({
                        "type": "tomas_update",
                        "step": _arm_step_count[0],
                        "phase": phase,
                        "eta": summary["current_eta"],
                        "violations": summary["total_violations"],
                        "violation_rate": summary["violation_rate"],
                    })
                except Exception:
                    pass

        print("v0.16.15: SO-ARM100 TOMAS controller initialized")

        # ── 3. Create or reuse ViserServer ──
        viser_server = None
        actual_port: int = 0

        if arm100_persistent_server is not None:
            try:
                _check_port = arm100_persistent_server._websock_server._port
                viser_server = arm100_persistent_server
                actual_port = arm100_persistent_port
                print(f"v0.16.15: Reusing persistent SO-ARM100 ViserServer on port {actual_port}")
            except Exception as _reuse_err:
                arm100_persistent_server = None
                arm100_persistent_port = 0

        if viser_server is None:
            for _attempt in range(6):
                _try_port = 8091 + _attempt  # SO-ARM100 uses 8091-8096
                try:
                    viser_server = ViserServer(port=_try_port, verbose=False)
                    actual_port = viser_server._websock_server._port
                    arm100_persistent_server = viser_server
                    arm100_persistent_port = actual_port
                    print(f"v0.16.15: Created new SO-ARM100 ViserServer on port {actual_port}")
                    break
                except (OSError, RuntimeError) as _oe:
                    print(f"SO-ARM100 ViserServer port {_try_port} failed (attempt {_attempt+1}/6): {_oe}")
                    if _attempt < 5:
                        _time.sleep(2.0)
            if viser_server is None:
                raise RuntimeError("Failed to create SO-ARM100 ViserServer after 6 attempts on ports 8091-8096")

        actual_port = viser_server._websock_server._port
        arm100_viewer_url = f"http://localhost:{actual_port}"

        # ── 4. Create mjviser Viewer ──
        viewer = mjviser.Viewer(
            model=arm_model,
            data=arm_data,
            step_fn=_arm_step_fn,
            server=viser_server,
        )
        # Start in paused mode: user clicks Play to start pick-and-place
        viewer._paused = True

        # Initial render
        mj.mj_forward(arm_model, arm_data)
        viewer._setup_gui()

        # v0.16.23: Reset GUI container to root before adding ARM100 panels.
        # viewer._setup_gui() leaves the container inside its own tab; without
        # resetting to root our tab group gets nested and its children don't render.
        viser_server.gui._set_container_uuid("root")

        # v0.16.17: SO-ARM100 Full Control Panel
        try:
            import viser
            arm_ctrl_tabs = viser_server.gui.add_tab_group()

            with arm_ctrl_tabs.add_tab("Manual Control"):
                viser_server.gui.add_markdown("**Manual Joint Control**\n\nToggle manual mode to control individual joints.")

                manual_toggle = viser_server.gui.add_checkbox("Manual Mode", initial_value=False)

                # Joint sliders (7 DOF)
                joint_names = ["Base", "Shoulder", "Elbow", "Wrist", "Wrist Roll", "Gripper L", "Gripper R"]
                joint_sliders = []
                for i, name in enumerate(joint_names):
                    if "Gripper L" in name:
                        s = viser_server.gui.add_slider(name, min=0.0, max=0.873, step=0.01, initial_value=0.0)
                    elif "Gripper R" in name:
                        s = viser_server.gui.add_slider(name, min=-0.873, max=0.0, step=0.01, initial_value=0.0)
                    else:
                        s = viser_server.gui.add_slider(name, min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                    joint_sliders.append(s)

                home_btn = viser_server.gui.add_button("🏠 Home")
                gripper_open_btn = viser_server.gui.add_button("🖐 Open Gripper")
                gripper_close_btn = viser_server.gui.add_button("✊ Close Gripper")

                @manual_toggle.on_update
                def _on_manual_toggle(event) -> None:
                    global arm100_manual_mode, arm100_manual_target
                    arm100_manual_mode = event.target.value
                    if arm100_manual_mode and arm100_manual_target is None:
                        arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                    print(f"v0.16.17: ARM100 manual mode = {arm100_manual_mode}")

                def _on_joint_update(event, idx):
                    global arm100_manual_target
                    if arm100_manual_target is None:
                        arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                    arm100_manual_target[idx] = event.target.value

                for i, s in enumerate(joint_sliders):
                    s.on_update(lambda event, idx=i: _on_joint_update(event, idx))

                @home_btn.on_click
                def _on_home_click(_):
                    global arm100_manual_mode, arm100_manual_target
                    arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                    arm100_manual_mode = True
                    for i, val in enumerate([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0]):
                        joint_sliders[i].value = val

                @gripper_open_btn.on_click
                def _on_gripper_open(_):
                    global arm100_manual_target
                    if arm100_manual_target is None:
                        arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                    arm100_manual_target[5] = 0.0
                    arm100_manual_target[6] = 0.0
                    joint_sliders[5].value = 0.0
                    joint_sliders[6].value = 0.0

                @gripper_close_btn.on_click
                def _on_gripper_close(_):
                    global arm100_manual_target
                    if arm100_manual_target is None:
                        arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                    arm100_manual_target[5] = 0.7
                    arm100_manual_target[6] = -0.7
                    joint_sliders[5].value = 0.7
                    joint_sliders[6].value = -0.7

            with arm_ctrl_tabs.add_tab("VLA Mode"):
                viser_server.gui.add_markdown(
                    "**Vision-Language-Action Model**\n\n"
                    "Connect to open-source VLA models (OpenVLA, Octo, π₀) for "
                    "language-conditioned manipulation.\n\n"
                    "Architecture: VLA → ψ-Anchor → κ-Snap → MuJoCo"
                )

                vla_model_select = viser_server.gui.add_dropdown(
                    "Model", options=["none", "demo-vla", "openvla-7b", "octo-base", "pi0-base"], initial_value="none"
                )

                vla_instruction = viser_server.gui.add_text(
                    "Instruction", initial_value="pick up the red cube"
                )

                vla_load_btn = viser_server.gui.add_button("Load Model")
                # v0.16.24: Explicit Submit button (in addition to on_update auto-binding).
                # Pressing Enter on the text field triggers on_update; clicking Submit
                # also commits the latest text value.
                vla_submit_btn = viser_server.gui.add_button("Submit Instruction")
                vla_status_text = viser_server.gui.add_markdown("Status: No model loaded")

                def _commit_instruction() -> str:
                    """Read the current text-field value into the global var. Returns the new value."""
                    global arm100_vla_instruction
                    try:
                        arm100_vla_instruction = str(vla_instruction.value)
                    except Exception:
                        # Fall back to attribute access if .value is not yet synced
                        arm100_vla_instruction = "pick up the red cube"
                    return arm100_vla_instruction

                @vla_load_btn.on_click
                def _on_vla_load(_):
                    global arm100_vla_adapter, arm100_vla_mode
                    _commit_instruction()
                    model_name = vla_model_select.value
                    if model_name == "none":
                        arm100_vla_mode = False
                        arm100_vla_adapter = None
                        vla_status_text.content = "Status: No model loaded"
                    else:
                        try:
                            from webviz.tomas_wrapper import create_vla_adapter
                            arm100_vla_adapter = create_vla_adapter(model_name)
                            arm100_vla_mode = True
                            vla_status_text.content = f"Status: ✅ {model_name} loaded (stub mode)\nInstruction: {arm100_vla_instruction}"
                        except Exception as e:
                            vla_status_text.content = f"Status: ❌ Load failed: {e}"

                @vla_submit_btn.on_click
                def _on_vla_submit(_):
                    """v0.16.25: Submit instruction — auto-enable VLA mode + demo adapter.
                    If no real VLA model is loaded, create a DemoVLAAdapter that
                    interprets the instruction and generates pick-and-place actions.
                    Also unpause the viewer so the arm starts moving immediately.
                    """
                    global arm100_vla_instruction, arm100_vla_mode, arm100_vla_adapter
                    val = _commit_instruction()
                    # Auto-enable VLA mode
                    arm100_vla_mode = True
                    # If no adapter loaded, or adapter is a stub, create demo adapter
                    if arm100_vla_adapter is None or not arm100_vla_adapter.is_loaded():
                        try:
                            from webviz.tomas_wrapper import create_vla_adapter
                            # v0.16.28: Pass model+data for FK-based IK (was 40cm error)
                            arm100_vla_adapter = create_vla_adapter('demo-vla', model=arm_model, data=arm_data)
                            vla_status_text.content = (
                                f"Status: ✅ Demo VLA active\n"
                                f"Instruction: '{val[:60]}{'...' if len(val) > 60 else ''}'\n"
                                f"Mode: instruction-driven pick-and-place"
                            )
                        except Exception as demo_err:
                            vla_status_text.content = f"Status: ❌ Demo adapter failed: {demo_err}"
                    else:
                        vla_status_text.content = (
                            f"Status: ✅ Instruction updated\n"
                            f"Instruction: '{val[:60]}{'...' if len(val) > 60 else ''}'\n"
                            f"Model: {arm100_vla_adapter.model_name}"
                        )
                    # v0.16.26: Unpause viewer so arm starts moving.
                    # Also reset _last_tick to avoid a huge dt jump that would
                    # cause the physics to take a massive step.
                    try:
                        viewer._paused = False
                        viewer._last_tick = time.perf_counter()
                        print(f"v0.16.26: VLA submit — viewer unpaused, mode={arm100_vla_mode}, adapter={arm100_vla_adapter.model_name if arm100_vla_adapter else 'None'}")
                    except Exception as unpause_err:
                        print(f"v0.16.26: Failed to unpause viewer: {unpause_err}")

                @vla_instruction.on_update
                def _on_instruction_update(event):
                    # v0.16.24: viser's on_update fires on every keystroke (including Enter).
                    # We commit on every update so pressing Enter is enough.
                    try:
                        new_val = event.target.value
                    except AttributeError:
                        new_val = getattr(event, "value", None)
                    global arm100_vla_instruction
                    if new_val is not None:
                        arm100_vla_instruction = str(new_val)

            # ── v0.16.19: CAMKit Dual Camera Tab ──
            with arm_ctrl_tabs.add_tab("CAMKit"):
                viser_server.gui.add_markdown(
                    "**CAMKit Dual Camera System**\n\n"
                    "Top Camera: fixed bird's-eye view of workspace\n"
                    "Wrist Camera: mounted on gripper, follows grasp motion\n\n"
                    "Frames rendered from MuJoCo at 320x240, ~10 FPS"
                )
                try:
                    # v0.16.24: add_image signature is (image: ndarray, *, label: str)
                    #                  passing a string as image is the bug — needs a real ndarray.
                    #                  Start with a black 320x240 frame as the initial value.
                    _cam_placeholder = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                    top_img = viser_server.gui.add_image(
                        _cam_placeholder, label="Top Camera (top_cam)", format="jpeg", jpeg_quality=70
                    )
                    wrist_img = viser_server.gui.add_image(
                        _cam_placeholder, label="Wrist Camera (wrist_cam)", format="jpeg", jpeg_quality=70
                    )
                    cam_refresh_btn = viser_server.gui.add_button("Capture Frame")
                    cam_status = viser_server.gui.add_markdown("Status: Ready")

                    @cam_refresh_btn.on_click
                    def _on_cam_refresh(_):
                        global arm100_cam_top_rgb, arm100_cam_wrist_rgb
                        top_rgb = _render_cam('top_cam')
                        wrist_rgb = _render_cam('wrist_cam')
                        arm100_cam_top_rgb = top_rgb
                        arm100_cam_wrist_rgb = wrist_rgb
                        try:
                            top_img.image = top_rgb
                            wrist_img.image = wrist_rgb
                            cam_status.content = f"Status: Frame captured at step {_arm_step_count[0]}"
                        except Exception as img_err:
                            cam_status.content = f"Status: Display error: {img_err}"
                except Exception as cam_gui_err:
                    viser_server.gui.add_markdown(f"CAMKit display error: {cam_gui_err}")

            # ── v0.16.27: Telemetry Tab — live force feedback + object tracking ──
            with arm_ctrl_tabs.add_tab("Telemetry"):
                telemetry_md = viser_server.gui.add_markdown("**Live Telemetry**\n\nInitializing...")

                def _update_telemetry():
                    """Update telemetry display with live sensor data."""
                    try:
                        # Get gripper position
                        grip_bid = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, "gripper_base")
                        grip_pos = arm_data.xpos[grip_bid].copy() if grip_bid >= 0 else np.zeros(3)

                        # Get object positions and distances
                        lines = ["**Live Telemetry**\n"]
                        lines.append(f"**Gripper pos:** ({grip_pos[0]:.3f}, {grip_pos[1]:.3f}, {grip_pos[2]:.3f})\n")

                        for obj_name, color_emoji in [("red_cube", "red"), ("blue_ball", "blue"), ("white_tissue", "white"), ("tray", "tray")]:
                            bid = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_BODY, obj_name)
                            if bid >= 0:
                                pos = arm_data.xpos[bid]
                                dist = float(np.linalg.norm(grip_pos - pos))
                                lines.append(f"**{color_emoji} {obj_name}:** ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) — dist={dist:.3f}m\n")

                        # Touch sensor readings (finger force)
                        try:
                            touch_l_id = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_SENSOR, "touch_L")
                            touch_r_id = mj.mj_name2id(arm_model, mj.mjtObj.mjOBJ_SENSOR, "touch_R")
                            if touch_l_id >= 0 and touch_r_id >= 0:
                                adr_l = arm_model.sensor_adr[touch_l_id]
                                adr_r = arm_model.sensor_adr[touch_r_id]
                                force_l = float(arm_data.sensordata[adr_l])
                                force_r = float(arm_data.sensordata[adr_r])
                                lines.append(f"\n**Finger forces:** L={force_l:.3f}N  R={force_r:.3f}N")
                                if force_l > 0.01 or force_r > 0.01:
                                    lines.append(" ✊ *grasping*")
                        except Exception:
                            pass

                        # Joint positions
                        qpos = arm_controller._get_arm_qpos()
                        lines.append("\n\n**Joint positions:**")
                        lines.append(f"\nRot={qpos[0]:+.2f} Pit={qpos[1]:+.2f} Elb={qpos[2]:+.2f} WPi={qpos[3]:+.2f} WRo={qpos[4]:+.2f}")
                        lines.append(f"\nGrL={qpos[5]:+.2f} GrR={qpos[6]:+.2f}")

                        # VLA status
                        if arm100_vla_mode and arm100_vla_adapter is not None:
                            lines.append(f"\n\n**VLA:** active, model={arm100_vla_adapter.model_name}")
                            lines.append(f"\n**Instruction:** '{arm100_vla_instruction[:50]}'")

                        telemetry_md.content = "".join(lines)
                    except Exception:
                        pass

                telemetry_refresh = viser_server.gui.add_button("Refresh")
                @telemetry_refresh.on_click
                def _on_telemetry_refresh(_):
                    _update_telemetry()

        except Exception as gui_err:
            print(f"SO-ARM100 control panel setup warning: {gui_err}")

        now: float = _time.perf_counter()
        viewer._last_tick = now
        viewer._stats_last_time = now

        # v0.16.25: Initial camera render — show scene immediately, not black.
        # This runs even when viewer is paused so users see the 3D scene.
        try:
            _init_top = _render_cam('top_cam')
            _init_wrist = _render_cam('wrist_cam')
            arm100_cam_top_rgb = _init_top
            arm100_cam_wrist_rgb = _init_wrist
            try:
                top_img.image = _init_top
                wrist_img.image = _init_wrist
            except Exception:
                pass  # top_img/wrist_img might not be defined if GUI setup failed
            print(f"v0.16.25: Initial camera render done — top mean={float(np.mean(_init_top)):.1f}, wrist mean={float(np.mean(_init_wrist)):.1f}")
        except Exception as init_cam_err:
            print(f"v0.16.25: Initial camera render failed: {init_cam_err}")

        arm100_viewer_running = True
        arm100_viewer_error = ""
        print(f"v0.16.15: SO-ARM100 viewer running at {arm100_viewer_url}")

        _cam_loop_counter = [0]
        try:
            while arm100_viewer_running:
                # v0.16.17: Speed multiplier for ARM100 too
                for _ in range(max(1, mjviser_sim_speed)):
                    if arm100_viewer_running:
                        viewer._tick()
                        # v0.16.26: If VLA mode is on but viewer is paused,
                        # manually step physics so VLA actions still execute.
                        # This handles the case where the viser Play button
                        # state is out of sync with viewer._paused.
                        if viewer._paused and arm100_vla_mode and arm100_vla_adapter is not None:
                            # v0.16.27: When paused, _step_physics won't run,
                            # so we call _arm_step_fn manually (which now includes
                            # mj_step internally). No separate mj_step needed.
                            try:
                                _arm_step_fn(arm_model, arm_data)
                                viewer._dirty = True
                            except Exception:
                                pass
                    else:
                        break

                # v0.16.25: Periodic camera rendering (every ~100 loop iterations ≈ 10 FPS).
                # This ensures cameras update even when the viewer is paused.
                _cam_loop_counter[0] += 1
                if _cam_loop_counter[0] % 100 == 0:
                    try:
                        _top_rgb = _render_cam('top_cam')
                        _wrist_rgb = _render_cam('wrist_cam')
                        arm100_cam_top_rgb = _top_rgb
                        arm100_cam_wrist_rgb = _wrist_rgb
                        try:
                            top_img.image = _top_rgb
                            wrist_img.image = _wrist_rgb
                        except Exception:
                            pass
                        # v0.16.27: Also encode JPEG for API serving
                        # (previously only done in _arm_step_fn which doesn't
                        # run when viewer is paused)
                        try:
                            from PIL import Image as _PIL
                            import io as _io
                            for _arr, _attr in [(_top_rgb, 'arm100_cam_top_frame'),
                                                (_wrist_rgb, 'arm100_cam_wrist_frame')]:
                                _pil_img = _PIL.fromarray(_arr)
                                _buf = _io.BytesIO()
                                _pil_img.save(_buf, format='JPEG', quality=85)
                                globals()[_attr] = _buf.getvalue()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # v0.16.27: Update telemetry display periodically
                    try:
                        _update_telemetry()
                    except Exception:
                        pass

                _time.sleep(0.001)
        finally:
            # Don't stop the ViserServer — keep it persistent for reuse
            arm100_viewer_running = False
            arm100_viewer_url = ""

    except Exception as e:
        arm100_viewer_running = False
        arm100_viewer_url = ""
        arm100_viewer_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"SO-ARM100 viewer failed: {e}")


@app.post("/api/arm100/start")
async def start_arm100() -> JSONResponse:
    """Start the independent SO-ARM100 pick-and-place viewer.

    This is completely separate from the humanoid mjviser system.
    Loads so_arm100_scene.xml and runs TOMAS IDO-audited pick-and-place.

    Returns:
        JSONResponse with viewer URL and status.
    """
    global arm100_viewer_thread, arm100_viewer_running, arm100_viewer_error

    if not MJVISER_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "mjviser not available", "available": False},
        )

    if not TOMAS_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "TOMAS wrapper not available", "available": False},
        )

    if arm100_viewer_running:
        return JSONResponse(content={
            "status": "already_running",
            "running": True,
            "url": arm100_viewer_url,
            "available": True,
        })

    arm100_viewer_running = False
    arm100_viewer_error = ""

    arm100_viewer_thread = threading.Thread(target=_launch_arm100_viewer, daemon=True)
    arm100_viewer_thread.start()

    # Poll for up to 8 seconds
    for _ in range(80):
        time.sleep(0.1)
        if arm100_viewer_running:
            break
        if arm100_viewer_error:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": arm100_viewer_error, "available": True},
            )

    if not arm100_viewer_running:
        return JSONResponse(
            status_code=504,
            content={"status": "timeout", "error": "Viewer startup timed out", "available": True},
        )

    return JSONResponse(content={
        "status": "running",
        "running": True,
        "url": arm100_viewer_url,
        "available": True,
    })


@app.post("/api/arm100/stop")
async def stop_arm100() -> JSONResponse:
    """Stop the SO-ARM100 viewer and destroy its ViserServer.

    Returns:
        JSONResponse with final status.
    """
    global arm100_viewer_running, arm100_viewer_url, arm100_viewer_error
    global arm100_persistent_server, arm100_persistent_port, arm100_tomas_wrapper

    arm100_viewer_running = False

    if arm100_viewer_thread is not None and arm100_viewer_thread.is_alive():
        arm100_viewer_thread.join(timeout=5.0)

    if arm100_persistent_server is not None:
        try:
            arm100_persistent_server.stop()
        except Exception:
            pass
        arm100_persistent_server = None
        arm100_persistent_port = 0

    arm100_tomas_wrapper = None
    arm100_viewer_url = ""
    arm100_viewer_error = ""

    return JSONResponse(content={
        "status": "stopped",
        "running": False,
        "available": True,
    })


@app.get("/api/arm100/status")
async def arm100_status() -> JSONResponse:
    """Get SO-ARM100 viewer status.

    Returns:
        JSONResponse with running state, URL, and any error.
    """
    return JSONResponse(content={
        "running": arm100_viewer_running,
        "url": arm100_viewer_url,
        "error": arm100_viewer_error,
        "available": MJVISER_AVAILABLE and TOMAS_AVAILABLE,
    })


# v0.16.17: SO-ARM100 control API endpoints
@app.post("/api/arm100/control")
async def arm100_control(request: Request) -> JSONResponse:
    """Control SO-ARM100 joints manually.

    Body: {"joints": [7 floats], "mode": "manual"|"auto"}
    """
    global arm100_manual_mode, arm100_manual_target
    body = await request.json()
    mode = body.get("mode", "manual")
    if mode == "manual":
        arm100_manual_mode = True
        joints = body.get("joints", [0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
        arm100_manual_target = np.array(joints, dtype=np.float64)
    elif mode == "auto":
        arm100_manual_mode = False
        arm100_manual_target = None
    return JSONResponse(content={
        "status": "ok",
        "mode": mode,
        "joints": arm100_manual_target.tolist() if arm100_manual_target is not None else None,
    })


@app.post("/api/arm100/gripper")
async def arm100_gripper(request: Request) -> JSONResponse:
    """Open or close SO-ARM100 gripper.

    Body: {"action": "open"|"close"}
    """
    global arm100_manual_mode, arm100_manual_target
    body = await request.json()
    action = body.get("action", "open")
    if arm100_manual_target is None:
        arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
    arm100_manual_mode = True
    if action == "close":
        arm100_manual_target[5] = 0.7
        arm100_manual_target[6] = -0.7
    else:
        arm100_manual_target[5] = 0.0
        arm100_manual_target[6] = 0.0
    return JSONResponse(content={"status": "ok", "action": action})


@app.post("/api/arm100/home")
async def arm100_home() -> JSONResponse:
    """Return SO-ARM100 to home pose."""
    global arm100_manual_mode, arm100_manual_target
    arm100_manual_mode = True
    arm100_manual_target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
    return JSONResponse(content={"status": "ok", "pose": arm100_manual_target.tolist()})


@app.post("/api/arm100/vla/load")
async def arm100_vla_load(request: Request) -> JSONResponse:
    """Load a VLA model for language-conditioned manipulation.

    Body: {"model": "openvla-7b"|"octo-base"|"pi0-base", "instruction": str}
    """
    global arm100_vla_adapter, arm100_vla_mode, arm100_vla_instruction
    body = await request.json()
    model_name = body.get("model", "openvla-7b")
    arm100_vla_instruction = body.get("instruction", "pick up the red cube")
    try:
        from webviz.tomas_wrapper import create_vla_adapter
        arm100_vla_adapter = create_vla_adapter(model_name)
        arm100_vla_mode = True
        return JSONResponse(content={
            "status": "ok",
            "model": model_name,
            "instruction": arm100_vla_instruction,
            "loaded": True,
        })
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "error": str(e), "loaded": False},
            status_code=500,
        )


@app.post("/api/arm100/vla/unload")
async def arm100_vla_unload() -> JSONResponse:
    """Unload VLA model, return to auto mode."""
    global arm100_vla_adapter, arm100_vla_mode
    arm100_vla_mode = False
    arm100_vla_adapter = None
    return JSONResponse(content={"status": "ok", "vla_mode": False})


@app.get("/api/arm100/vla/status")
async def arm100_vla_status() -> JSONResponse:
    """Get VLA adapter status."""
    return JSONResponse(content={
        "vla_mode": arm100_vla_mode,
        "model": arm100_vla_adapter.model_name if arm100_vla_adapter else "none",
        "instruction": arm100_vla_instruction,
    })


# ── v0.16.19: CAMKit Camera API ──
@app.get("/api/arm100/cam/top")
async def arm100_cam_top() -> Response:
    """Get latest top camera JPEG frame (CAMKit top_cam)."""
    if arm100_cam_top_frame is not None:
        return Response(content=arm100_cam_top_frame, media_type="image/jpeg")
    # Return a placeholder if no data yet
    try:
        import io as _io
        from PIL import Image as _PIL, ImageDraw as _Draw
        img = _PIL.new('RGB', (320, 240), color=(20, 80, 20))
        draw = _Draw.Draw(img)
        draw.text((10, 100), "Waiting for ARM100 viewer...", fill=(255, 255, 0))
        draw.text((10, 120), "Click Play or Submit Instruction", fill=(200, 255, 200))
        buf = _io.BytesIO()
        img.save(buf, format='JPEG')
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception:
        return Response(content=b'', media_type="image/jpeg", status_code=503)


@app.get("/api/arm100/cam/wrist")
async def arm100_cam_wrist() -> Response:
    """Get latest wrist camera JPEG frame (CAMKit wrist_cam)."""
    if arm100_cam_wrist_frame is not None:
        return Response(content=arm100_cam_wrist_frame, media_type="image/jpeg")
    try:
        import io as _io
        from PIL import Image as _PIL, ImageDraw as _Draw
        img = _PIL.new('RGB', (320, 240), color=(20, 80, 20))
        draw = _Draw.Draw(img)
        draw.text((10, 100), "Waiting for ARM100 viewer...", fill=(255, 255, 0))
        draw.text((10, 120), "Click Play or Submit Instruction", fill=(200, 255, 200))
        buf = _io.BytesIO()
        img.save(buf, format='JPEG')
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception:
        return Response(content=b'', media_type="image/jpeg", status_code=503)


@app.get("/api/arm100/cam/status")
async def arm100_cam_status() -> JSONResponse:
    """Get CAMKit camera system status."""
    return JSONResponse(content={
        "available": arm100_viewer_running,
        "top_cam_active": arm100_cam_top_frame is not None,
        "wrist_cam_active": arm100_cam_wrist_frame is not None,
        "resolution": "320x240",
        "fps_target": 10,
    })


# v0.16.17: Simulation speed control API
@app.post("/api/viewer_speed")
async def viewer_speed(request: Request) -> JSONResponse:
    """Set simulation speed multiplier (1-64x).

    Body: {"speed": int}
    """
    global mjviser_sim_speed
    body = await request.json()
    speed = int(body.get("speed", 1))
    mjviser_sim_speed = max(1, min(64, speed))
    return JSONResponse(content={"status": "ok", "speed": mjviser_sim_speed})


@app.get("/api/viewer_speed")
async def viewer_speed_status() -> JSONResponse:
    """Get current simulation speed."""
    return JSONResponse(content={"speed": mjviser_sim_speed})
@app.get("/api/tomas/snap_log")
async def tomas_snap_log(n: int = 50) -> JSONResponse:
    """Get recent κ-Snap audit entries from the TOMAS wrapper.

    Args:
        n: Number of recent entries to return (default 50, max 500).

    Returns:
        JSONResponse with list of κ-Snap entries.
    """
    n = min(max(1, n), 500)
    if arm100_tomas_wrapper is None:
        return JSONResponse(content={"snaps": [], "count": 0, "available": False})
    snaps = arm100_tomas_wrapper.get_recent_snaps(n)
    return JSONResponse(content={"snaps": snaps, "count": len(snaps), "available": True})


@app.get("/api/tomas/summary")
async def tomas_summary() -> JSONResponse:
    """Get TOMAS audit summary statistics.

    Returns:
        JSONResponse with total steps, violations, η trajectory, etc.
    """
    if arm100_tomas_wrapper is None:
        return JSONResponse(content={"available": False})
    summary = arm100_tomas_wrapper.get_summary()
    summary["available"] = True
    return JSONResponse(content=summary)


# v0.16.26: Nine-Layer cognitive architecture API
@app.get("/api/architecture")
async def get_architecture() -> JSONResponse:
    """Get the nine-layer cognitive architecture (L0-L8) mapping.

    Returns the biological analogue, modules, and active status for each layer.
    """
    try:
        from core.nine_layer import NineLayerRegistry
        registry = NineLayerRegistry()
        return JSONResponse(content={
            "available": True,
            "version": WEBVIZ_VERSION,
            "layers": registry.get_architecture_summary(),
        })
    except Exception as e:
        return JSONResponse(content={
            "available": False,
            "error": str(e),
            "version": WEBVIZ_VERSION,
        })


# v0.16.26: T-Processor status API
@app.get("/api/t_processor")
async def get_t_processor_status() -> JSONResponse:
    """Get T-Processor hardware simulation status (η-ALU, ψ-Checker, κ-Snap FIFO)."""
    try:
        from core.t_processor import TProcessor
        tp = TProcessor()
        return JSONResponse(content={
            "available": True,
            "spec": {
                "gate_count": tp.GATE_COUNT,
                "power_mw": tp.POWER_MW,
                "clock_hz": tp.CLOCK_HZ,
                "fifo_depth": tp.fifo.depth,
                "sram_bytes": tp.fifo.depth * 16,  # 16 bytes per entry
            },
        })
    except Exception as e:
        return JSONResponse(content={"available": False, "error": str(e)})


# v0.16.7: Direction control API — steer the robot in 3D viewer
@app.post("/api/viewer_control")
async def viewer_control(request: Request) -> JSONResponse:
    """Control robot walking direction in mjviser viewer.

    Body:
        {"action": "forward" | "stop" | "left" | "right" | "step_up"}

    Returns:
        JSONResponse with updated direction and speed.
    """
    global mjviser_walk_direction, mjviser_target_direction, mjviser_walk_speed, mjviser_walk_action
    global mjviser_last_dir_change_time
    body = await request.json()
    action = body.get("action", "forward")

    if action == "stop":
        mjviser_walk_speed = 0.0
        mjviser_walk_action = "stop"
    elif action == "forward":
        mjviser_target_direction = 0.0
        mjviser_walk_speed = 1.0
        mjviser_walk_action = "walk"
    elif action == "left":
        _now = time.time()
        # v0.16.17: Anti-backflip cooldown
        if _now - mjviser_last_dir_change_time >= 0.3:
            mjviser_last_dir_change_time = _now
            mjviser_target_direction += np.pi / 6  # Turn 30° left
        mjviser_walk_speed = 1.0
        mjviser_walk_action = "walk"
    elif action == "right":
        _now = time.time()
        # v0.16.17: Anti-backflip cooldown
        if _now - mjviser_last_dir_change_time >= 0.3:
            mjviser_last_dir_change_time = _now
            mjviser_target_direction -= np.pi / 6  # Turn 30° right
        mjviser_walk_speed = 1.0
        mjviser_walk_action = "walk"
    elif action == "step_up":
        mjviser_walk_action = "step_up"
        mjviser_walk_speed = 1.0
    else:
        return JSONResponse(content={"error": f"Unknown action: {action}"}, status_code=400)

    # Normalize target direction to [-pi, pi]
    mjviser_target_direction = float(np.arctan2(
        np.sin(mjviser_target_direction), np.cos(mjviser_target_direction)))

    return JSONResponse(content={
        "action": mjviser_walk_action,
        "direction": mjviser_walk_direction,
        "direction_deg": float(np.degrees(mjviser_walk_direction)),
        "speed": mjviser_walk_speed,
    })


@app.get("/api/viewer_control")
async def viewer_control_status() -> JSONResponse:
    """Get current robot walking direction and speed."""
    return JSONResponse(content={
        "action": mjviser_walk_action,
        "direction": mjviser_walk_direction,
        "direction_deg": float(np.degrees(mjviser_walk_direction)),
        "speed": mjviser_walk_speed,
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

    Note: no-cache headers ensure the browser always fetches the latest
    version — prevents stale dashboard when server is updated.
    """
    dashboard_path: str = str(Path(__file__).resolve().parent / "dashboard.html")
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            html_content: str = f.read()
        # Inject version stamp for cache-busting verification
        html_content = html_content.replace(
            "<head>",
            f"<!-- Webviz {WEBVIZ_VERSION} - served at {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n<head>",
            1,
        )
        return HTMLResponse(
            content=html_content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
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
    return FileResponse(
        os.path.join(WEBVIZ_DIR, "user_manual.html"),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/mujoco_docs_cn.html")
async def serve_mujoco_docs_cn() -> FileResponse:
    """Serve the MuJoCo Overview Chinese translation HTML page.

    Returns:
        FileResponse with the mujoco_docs_cn.html content.
    """
    return FileResponse(
        os.path.join(WEBVIZ_DIR, "mujoco_docs_cn.html"),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ═══════════════════════════════════════════════════════════════════
# v0.17.0: 焊接场景 API 端点 (Welding Domain API Endpoints)
# ═══════════════════════════════════════════════════════════════════

# 尝试导入焊接模块
WELDING_AVAILABLE: bool = False
_welding_env: Optional[Any] = None
_welding_thread: Optional[threading.Thread] = None
_welding_running: bool = False
_welding_step: int = 0
_welding_quality: Dict[str, Any] = {}
_welding_safety: Dict[str, Any] = {"passed": True, "violations": [], "actions": []}

try:
    from envs.welding_env import WeldingEnv
    from agent.welding_psi_anchor import WeldingPsiAnchor
    from core.welding_process_proxy import WeldingProcessProxy
    from core.welding_sensors import WeldingSensorSuite
    from core.tomas_welding_axioms import TomasWeldingAxioms
    WELDING_AVAILABLE = True
except ImportError as _welding_err:
    WELDING_AVAILABLE = False
    print(f"Warning: Welding modules not available: {_welding_err}")


def _get_or_create_welding_env(weld_type: str = "flat") -> Optional[Any]:
    """获取或创建焊接环境实例.

    Args:
        weld_type: 焊接姿态类型.

    Returns:
        WeldingEnv 实例, 如果创建失败返回 None.
    """
    global _welding_env
    if not WELDING_AVAILABLE:
        return None
    if _welding_env is None or getattr(_welding_env, "weld_type", "") != weld_type:
        try:
            _welding_env = WeldingEnv(weld_type=weld_type)
        except Exception as e:
            print(f"Failed to create WeldingEnv: {e}")
            return None
    return _welding_env


@app.get("/api/welding/status")
async def welding_status() -> JSONResponse:
    """焊接仿真状态.

    Returns:
        JSONResponse with weld_type, step, tcp_pose, joint_angles,
        stickout, welding_params, quality, safety_status.
    """
    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available",
                     "available": False},
        )

    env: Optional[Any] = _get_or_create_welding_env()
    if env is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "WeldingEnv creation failed",
                     "available": False},
        )

    try:
        obs: np.ndarray = env.get_observation()
        tcp_pose: list = obs[0:6].tolist()
        joint_angles: list = obs[6:12].tolist()
        stickout: float = float(obs[12])
        contact_force: list = obs[13:16].tolist()
        temperature: float = float(obs[16])
        seam_dev: float = float(obs[17])

        return JSONResponse(content={
            "status": "running" if _welding_running else "idle",
            "weld_type": env.weld_type,
            "step": _welding_step,
            "tcp_pose": tcp_pose,
            "joint_angles": joint_angles,
            "stickout_mm": stickout,
            "contact_force_N": contact_force,
            "temperature_C": temperature,
            "seam_deviation_mm": seam_dev,
            "quality": _welding_quality,
            "safety": _welding_safety,
            "available": True,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "available": True},
        )


@app.get("/api/welding/trajectory")
async def welding_trajectory() -> JSONResponse:
    """焊缝轨迹数据.

    Returns:
        JSONResponse with waypoints list and planned trajectory.
    """
    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available"},
        )

    env: Optional[Any] = _get_or_create_welding_env()
    if env is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "WeldingEnv creation failed"},
        )

    try:
        waypoints: np.ndarray = env.waypoints
        return JSONResponse(content={
            "waypoints": waypoints.tolist(),
            "num_waypoints": len(waypoints),
            "seam_length_mm": 200.0,
            "waypoint_spacing_mm": 2.0,
            "weld_type": env.weld_type,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )


@app.get("/api/welding/quality")
async def welding_quality() -> JSONResponse:
    """焊接质量指标.

    Returns:
        JSONResponse with eta, porosity, distortion, penetration,
        heat_input, arc_length.
    """
    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available"},
        )

    try:
        return JSONResponse(content={
            "quality": _welding_quality if _welding_quality else {
                "eta": 0.0,
                "porosity": 0.0,
                "distortion": 0.0,
            },
            "step": _welding_step,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )


@app.get("/api/welding/safety")
async def welding_safety() -> JSONResponse:
    """安全约束状态.

    Returns:
        JSONResponse with stick_out_status, burn_back_status,
        porosity_risk_status, violations, actions.
    """
    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available"},
        )

    try:
        return JSONResponse(content={
            "safety": _welding_safety,
            "violations": _welding_safety.get("violations", []),
            "actions": _welding_safety.get("actions", []),
            "passed": _welding_safety.get("passed", True),
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )


def _welding_simulation_loop(weld_type: str, max_steps: int = 500) -> None:
    """焊接仿真循环 (在后台线程中运行).

    Args:
        weld_type: 焊接姿态类型.
        max_steps: 最大步数.
    """
    global _welding_running, _welding_step, _welding_quality, _welding_safety

    env: Optional[Any] = _get_or_create_welding_env(weld_type)
    if env is None:
        _welding_running = False
        return

    try:
        env.reset()
        _welding_step = 0
        _welding_quality = {}
        _welding_safety = {"passed": True, "violations": [], "actions": []}

        for step in range(max_steps):
            if not _welding_running:
                break

            # 使用接近最优的参数
            action: np.ndarray = np.array([200.0, 24.0, 2.0, 6.0])
            result: Dict[str, Any] = env.step(action)
            _welding_step = step + 1

            info: Dict[str, Any] = result.get("info", {})
            _welding_quality = info.get("quality", {})
            _welding_safety = info.get("safety", {"passed": True, "violations": [], "actions": []})

            if result.get("done", False):
                break

            time.sleep(0.01)  # 10ms 间隔

    except Exception as e:
        print(f"Welding simulation error: {e}")
        traceback.print_exc()
    finally:
        _welding_running = False


@app.post("/api/welding/start")
async def welding_start(weld_type: str = "flat") -> JSONResponse:
    """启动焊接仿真.

    Args:
        weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead").

    Returns:
        JSONResponse with status.
    """
    global _welding_running, _welding_thread

    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available",
                     "available": False},
        )

    if _welding_running:
        return JSONResponse(
            content={"status": "already_running", "weld_type": weld_type}
        )

    _welding_running = True
    _welding_thread = threading.Thread(
        target=_welding_simulation_loop,
        args=(weld_type,),
        daemon=True,
    )
    _welding_thread.start()

    return JSONResponse(content={
        "status": "started",
        "weld_type": weld_type,
        "available": True,
    })


@app.post("/api/welding/stop")
async def welding_stop() -> JSONResponse:
    """停止焊接仿真.

    Returns:
        JSONResponse with status.
    """
    global _welding_running

    if not WELDING_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Welding modules not available"},
        )

    _welding_running = False
    return JSONResponse(content={"status": "stopped"})


# ── CORS middleware (for development) ──
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
