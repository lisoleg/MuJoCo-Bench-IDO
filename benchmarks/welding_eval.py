"""
WeldingEval -- Real Baseline Evaluation Engine
==============================================

Runs 4 agents (Random, Constant, SAC, Expert) on the same WeldingEnv,
collects 9 quality/performance metrics per agent, and outputs a
structured comparison (JSON + Markdown table).

Usage:
  CLI:    python benchmarks/welding_eval.py
  Import: from benchmarks.welding_eval import WeldingEvaluator, run_evaluation

Agents:
  1. RandomAgent   -- uniform random actions in physical range
  2. ConstantAgent -- fixed near-optimal parameters [200, 24, 2, 6]
  3. SACAgent      -- loads SB3 SAC checkpoint, maps [-1,1] to physical
  4. ExpertAgent   -- near-optimal params with small random perturbation

Metrics (9):
  eta_residual, porosity_risk, penetration_depth, angular_distortion,
  weld_progress, heat_input, current_fluctuation, safety_violations,
  episode_return

Author: MuJoCo-Bench-IDO Welding Module v0.5.0
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Add project root to path
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.welding_env import WeldingEnv
from core.welding_process_proxy import WeldingProcessProxy, WeldingQuality

# Try importing SB3 for SAC agent
try:
    from stable_baselines3 import SAC
    _HAS_SB3: bool = True
except ImportError:
    _HAS_SB3 = False

# ── Constants ──
ACTION_LOW: np.ndarray = np.array([50.0, 14.0, 0.0, 2.0])
ACTION_HIGH: np.ndarray = np.array([350.0, 32.0, 5.0, 15.0])

DEFAULT_MAX_STEPS: int = 3500  # 3500步 × 0.012mm/步 = 42mm, 可跑完 20 waypoints × 2mm = 40mm
DEFAULT_WELD_TYPE: str = "flat"
DEFAULT_SAC_CHECKPOINT: str = os.path.join(
    _PROJECT_ROOT, "checkpoints", "sac_weld", "sac_weld_flat.zip"
)

# ── 焊缝类型 → 最优参数映射 (AWS D1.1 经验值) ──
WELD_TYPE_OPTIMAL: Dict[str, np.ndarray] = {
    "flat":        np.array([200.0, 24.0, 2.0, 6.0]),   # 平焊: 标准参数
    "horizontal":  np.array([180.0, 22.0, 3.0, 5.0]),   # 横焊: 降电流防铁水下淌
    "vertical":    np.array([160.0, 20.0, 4.0, 4.0]),   # 立焊: 更低电流+大摆动
    "overhead":    np.array([170.0, 21.0, 2.0, 7.0]),   # 仰焊: 高速防滴落
}

# ── 焊缝类型 → SAC checkpoint 路径 ──
WELD_TYPE_CHECKPOINT: Dict[str, str] = {
    wt: os.path.join(_PROJECT_ROOT, "checkpoints", "sac_weld", f"sac_weld_{wt}.zip")
    for wt in WELD_TYPE_OPTIMAL
}

ALL_WELD_TYPES: List[str] = list(WELD_TYPE_OPTIMAL.keys())

# 14 metrics definition: name -> (lower_better, unit, display_name)
# v0.18: expanded from 9 to 14 with industry-leading metrics
METRIC_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "eta_residual": {"lower_better": True, "unit": "", "display": "eta residual"},
    "porosity_risk": {"lower_better": True, "unit": "", "display": "porosity risk"},
    "penetration_depth": {"lower_better": False, "unit": "mm", "display": "penetration"},
    "angular_distortion": {"lower_better": True, "unit": "deg", "display": "distortion"},
    "weld_progress": {"lower_better": False, "unit": "", "display": "weld progress"},
    "heat_input": {"lower_better": True, "unit": "kJ/mm", "display": "heat input"},
    "current_fluctuation": {"lower_better": True, "unit": "A", "display": "current std"},
    "safety_violations": {"lower_better": True, "unit": "", "display": "safety violations"},
    "episode_return": {"lower_better": False, "unit": "", "display": "episode return"},
    # v0.18: Industry-leading metrics
    "bead_width": {"lower_better": False, "unit": "mm", "display": "bead width"},
    "bead_height": {"lower_better": False, "unit": "mm", "display": "bead height"},
    "spatter_rate": {"lower_better": True, "unit": "", "display": "spatter rate"},
    "deposition_rate": {"lower_better": False, "unit": "kg/h", "display": "deposition rate"},
    "arc_stability": {"lower_better": False, "unit": "", "display": "arc stability"},
}

METRIC_NAMES: List[str] = list(METRIC_DEFINITIONS.keys())


# ═══════════════════════════════════════════════════════════════
# Agents
# ═══════════════════════════════════════════════════════════════

class RandomAgent:
    """Agent that generates uniform random actions in the physical range.

    Each step samples [current, voltage, weave, speed] uniformly from
    [50,350] x [14,32] x [0,5] x [2,15].

    Attributes:
        name: Agent identifier.
        rng: numpy random generator.
    """

    def __init__(self, seed: int = 123) -> None:
        """Initialize the random agent.

        Args:
            seed: Random seed for reproducibility.
        """
        self.name: str = "Random"
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Generate a random action.

        Args:
            obs: Current observation (ignored).

        Returns:
            4-dim action array [current, voltage, weave, speed].
        """
        action: np.ndarray = self.rng.uniform(ACTION_LOW, ACTION_HIGH)
        return action.astype(np.float64)


class ConstantAgent:
    """Agent that always outputs fixed near-optimal parameters.

    Uses weld-type-specific optimal parameters from WELD_TYPE_OPTIMAL.
    For flat welding: [200A, 24V, 2mm weave, 6mm/s speed].

    Attributes:
        name: Agent identifier.
        action: Fixed action array.
    """

    def __init__(self, weld_type: str = DEFAULT_WELD_TYPE) -> None:
        """Initialize the constant agent with weld-type-specific params.

        Args:
            weld_type: Welding posture type for parameter selection.
        """
        self.name: str = "Constant"
        self.action: np.ndarray = WELD_TYPE_OPTIMAL.get(
            weld_type, WELD_TYPE_OPTIMAL["flat"]
        ).copy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return the fixed action.

        Args:
            obs: Current observation (ignored).

        Returns:
            Fixed 4-dim action array.
        """
        return self.action.copy()


class SACAgent:
    """Agent that loads a Stable-Baselines3 SAC checkpoint.

    Loads the SAC model from checkpoint_path. If SB3 is unavailable or
    the checkpoint cannot be loaded, falls back to ConstantAgent behavior.

    The SB3 SAC model outputs actions in [-1, 1] normalized space.
    These are mapped to physical ranges using:
        physical = low + (action + 1.0) * 0.5 * (high - low)

    Attributes:
        name: Agent identifier.
        model: Loaded SB3 SAC model, or None if unavailable.
        _fallback: ConstantAgent used when model is None.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_SAC_CHECKPOINT,
        weld_type: str = DEFAULT_WELD_TYPE,
    ) -> None:
        """Initialize the SAC agent.

        Attempts to load the SB3 SAC checkpoint. Falls back to
        constant parameters if loading fails.

        Args:
            checkpoint_path: Path to the SAC checkpoint .zip file.
            weld_type: Welding posture type for fallback parameters.
        """
        self.name: str = "SAC"
        self.model: Optional[Any] = None

        if _HAS_SB3 and os.path.exists(checkpoint_path):
            try:
                self.model = SAC.load(checkpoint_path)
                print(f"[SACAgent] Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load SAC checkpoint: {e}")
                self.model = None
        elif not _HAS_SB3:
            print("Warning: stable_baselines3 not available, SAC agent uses constant fallback")
        elif not os.path.exists(checkpoint_path):
            print(f"Warning: SAC checkpoint not found at {checkpoint_path}, using constant fallback")

        # Fallback to weld-type-specific constant parameters
        self._fallback: ConstantAgent = ConstantAgent(weld_type=weld_type)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Generate an action using the SAC model or fallback.

        Args:
            obs: Current observation vector (18-dim).

        Returns:
            4-dim physical action array [current, voltage, weave, speed].
        """
        if self.model is not None:
            try:
                # SB3 SAC outputs [-1, 1] normalized action
                obs_array: np.ndarray = np.asarray(obs, dtype=np.float32).flatten()
                action, _ = self.model.predict(obs_array, deterministic=True)
                action = np.asarray(action, dtype=np.float64).flatten()
                # Map [-1, 1] to physical range
                physical: np.ndarray = ACTION_LOW + (action + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW)
                return physical
            except Exception as e:
                # If prediction fails at runtime, use fallback for this step
                print(f"Warning: SAC predict failed ({e}), using fallback")
                return self._fallback.act(obs)
        else:
            return self._fallback.act(obs)


class ExpertAgent:
    """Agent that uses near-optimal parameters with small perturbation.

    Outputs weld-type-specific optimal params +/- Gaussian noise.
    Represents a skilled human welder who knows the right parameters
    for each welding position.

    Attributes:
        name: Agent identifier.
        rng: numpy random generator.
    """

    def __init__(self, seed: int = 456, weld_type: str = DEFAULT_WELD_TYPE) -> None:
        """Initialize the expert agent with weld-type-specific params.

        Args:
            seed: Random seed for reproducibility.
            weld_type: Welding posture type for parameter selection.
        """
        self.name: str = "Expert"
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._base: np.ndarray = WELD_TYPE_OPTIMAL.get(
            weld_type, WELD_TYPE_OPTIMAL["flat"]
        ).copy()
        self._noise_std: np.ndarray = np.array([10.0, 1.0, 0.5, 0.5], dtype=np.float64)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Generate a near-optimal action with small perturbation.

        Args:
            obs: Current observation (ignored).

        Returns:
            4-dim action array near optimal parameters.
        """
        noise: np.ndarray = self.rng.normal(0.0, self._noise_std)
        action: np.ndarray = self._base + noise
        # Clip to valid range
        return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float64)


# ═══════════════════════════════════════════════════════════════
# Evaluation Engine
# ═══════════════════════════════════════════════════════════════

class WeldingEvaluator:
    """Real baseline evaluation engine for welding robots.

    Runs multiple agents on the same WeldingEnv, collects 9 metrics
    per agent, and returns a structured comparison result.

    Attributes:
        weld_type: Type of weld (flat, horizontal, vertical, overhead).
        max_steps: Maximum steps per episode.
        agents: List of agent instances to evaluate.
    """

    def __init__(
        self,
        weld_type: str = DEFAULT_WELD_TYPE,
        max_steps: int = DEFAULT_MAX_STEPS,
        sac_checkpoint: str = "",
    ) -> None:
        """Initialize the welding evaluator.

        Args:
            weld_type: Welding posture type.
            max_steps: Maximum simulation steps per agent.
            sac_checkpoint: Path to SAC checkpoint file. If empty, auto-resolve by weld_type.
        """
        self.weld_type: str = weld_type
        self.max_steps: int = max_steps

        # Auto-resolve SAC checkpoint by weld_type if not specified
        if sac_checkpoint:
            self.sac_checkpoint: str = sac_checkpoint
        else:
            self.sac_checkpoint = WELD_TYPE_CHECKPOINT.get(
                weld_type, DEFAULT_SAC_CHECKPOINT
            )

        # Build agent list with weld-type-specific parameters
        self.agents: List[Any] = [
            RandomAgent(seed=123),
            ConstantAgent(weld_type=weld_type),
            SACAgent(checkpoint_path=self.sac_checkpoint, weld_type=weld_type),
            ExpertAgent(seed=456, weld_type=weld_type),
        ]

    def _run_agent(self, agent: Any) -> Dict[str, float]:
        """Run a single agent on WeldingEnv and collect 9 metrics.

        Creates a fresh WeldingEnv, resets it, runs the agent for up
        to max_steps steps, and collects per-step metrics. Returns
        averaged/aggregated metrics for the episode.

        Args:
            agent: Agent instance with act(obs) -> action method.

        Returns:
            Dictionary of 9 metrics for this agent's episode.
        """
        # Create fresh environment for each agent
        env: WeldingEnv = WeldingEnv(weld_type=self.weld_type)
        obs: np.ndarray = env.reset()

        # Per-step collectors
        eta_values: List[float] = []
        porosity_values: List[float] = []
        penetration_values: List[float] = []
        distortion_values: List[float] = []
        progress_values: List[float] = []
        heat_input_values: List[float] = []
        current_values: List[float] = []
        safety_violation_count: int = 0
        episode_return: float = 0.0
        steps_taken: int = 0
        # v0.18: New industry metrics collectors
        bead_width_values: List[float] = []
        bead_height_values: List[float] = []
        spatter_rate_values: List[float] = []
        deposition_rate_values: List[float] = []
        arc_stability_values: List[float] = []

        for step in range(self.max_steps):
            action: np.ndarray = agent.act(obs)
            result: Dict[str, Any] = env.step(action)

            obs = result["observation"]
            reward: float = float(result["reward"])
            done: bool = bool(result["done"])
            info: Dict[str, Any] = result["info"]

            steps_taken += 1
            episode_return += reward

            # Extract quality metrics from info["quality"]
            quality: Dict[str, float] = info.get("quality", {})
            eta_values.append(float(quality.get("eta", 0.0)))
            porosity_values.append(float(quality.get("porosity", 0.0)))
            penetration_values.append(float(quality.get("penetration", 0.0)))
            distortion_values.append(float(quality.get("distortion", 0.0)))

            # v0.18: New industry metrics
            bead_width_values.append(float(quality.get("bead_width", info.get("bead_width", 0.0))))
            bead_height_values.append(float(quality.get("bead_height", info.get("bead_height", 0.0))))
            spatter_rate_values.append(float(quality.get("spatter_rate", info.get("spatter_rate", 0.0))))
            deposition_rate_values.append(float(quality.get("deposition_rate", info.get("deposition_rate", 0.0))))
            arc_stability_values.append(float(quality.get("arc_stability", info.get("arc_stability", 0.0))))

            # Other info metrics
            progress_values.append(float(info.get("weld_progress", 0.0)))
            # info["heat_input"] is in J/mm, convert to kJ/mm
            heat_j_mm: float = float(info.get("heat_input", 0.0))
            heat_input_values.append(heat_j_mm / 1000.0)
            current_values.append(float(info.get("current", 200.0)))

            # Safety violations — only count critical (passed=False),
            # not warnings (passed=True with violations in list)
            safety: Dict[str, Any] = info.get("safety", {})
            if not safety.get("passed", True):
                safety_violation_count += 1

            if done:
                break

        # Aggregate metrics
        def _safe_mean(values: List[float]) -> float:
            """Compute mean, returning 0.0 for empty lists.

            Args:
                values: List of float values.

            Returns:
                Mean value or 0.0.
            """
            if not values:
                return 0.0
            return float(np.mean(values))

        def _safe_std(values: List[float]) -> float:
            """Compute std, returning 0.0 for empty or single-element lists.

            Args:
                values: List of float values.

            Returns:
                Std value or 0.0.
            """
            if len(values) < 2:
                return 0.0
            return float(np.std(values))

        metrics: Dict[str, float] = {
            "eta_residual": _safe_mean(eta_values),
            "porosity_risk": _safe_mean(porosity_values),
            "penetration_depth": _safe_mean(penetration_values),
            "angular_distortion": _safe_mean(distortion_values),
            "weld_progress": _safe_mean(progress_values),
            "heat_input": _safe_mean(heat_input_values),
            "current_fluctuation": _safe_std(current_values),
            "safety_violations": float(safety_violation_count),
            "episode_return": float(episode_return),
            # v0.18: Industry-leading metrics
            "bead_width": _safe_mean(bead_width_values),
            "bead_height": _safe_mean(bead_height_values),
            "spatter_rate": _safe_mean(spatter_rate_values),
            "deposition_rate": _safe_mean(deposition_rate_values),
            "arc_stability": _safe_mean(arc_stability_values),
        }

        # Print progress line
        print(
            f"  [{agent.name}] Steps={steps_taken}, "
            f"Return={episode_return:.2f}, "
            f"Eta={metrics['eta_residual']:.4f}"
        )

        return metrics

    def evaluate(self) -> Dict[str, Any]:
        """Run all agents and return the full comparison result.

        Returns:
            Dictionary with structure:
            {
                "agents": {name: {metric: value, ...}, ...},
                "metrics": [metric_name, ...],
                "metric_info": {metric: {lower_better, unit, display}, ...},
                "best_agent": "Expert",
                "timestamp": "2026-07-04T12:00:00"
            }
        """
        print(f"\n{'='*60}")
        print(f"Welding Baseline Evaluation")
        print(f"  Weld type: {self.weld_type}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Agents: {[a.name for a in self.agents]}")
        print(f"  SB3 available: {_HAS_SB3}")
        print(f"{'='*60}\n")

        agent_results: Dict[str, Dict[str, float]] = {}

        for agent in self.agents:
            agent_results[agent.name] = self._run_agent(agent)

        # Determine best agent: highest episode_return
        best_agent: str = "Expert"
        best_return: float = float("-inf")
        for name, metrics in agent_results.items():
            ret: float = metrics.get("episode_return", float("-inf"))
            if ret > best_return:
                best_return = ret
                best_agent = name

        # Build metric_info
        metric_info: Dict[str, Dict[str, Any]] = {}
        for mname, mdef in METRIC_DEFINITIONS.items():
            metric_info[mname] = {
                "lower_better": mdef["lower_better"],
                "unit": mdef["unit"],
                "display": mdef["display"],
            }

        result: Dict[str, Any] = {
            "agents": agent_results,
            "metrics": METRIC_NAMES,
            "metric_info": metric_info,
            "best_agent": best_agent,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def generate_markdown_table(self, result: Dict[str, Any]) -> str:
        """Generate a Markdown comparison table from evaluation results.

        Args:
            result: Evaluation result dictionary from evaluate().

        Returns:
            Markdown-formatted table string.
        """
        agents_data: Dict[str, Dict[str, float]] = result["agents"]
        agent_names: List[str] = list(agents_data.keys())
        metrics_list: List[str] = result["metrics"]
        metric_info: Dict[str, Dict[str, Any]] = result["metric_info"]

        # Build header
        header: str = "| Metric |" + "|".join(f" {a} " for a in agent_names) + "|"
        separator: str = "|--------|" + "|".join(["------"] * len(agent_names)) + "|"

        lines: List[str] = [header, separator]

        for mname in metrics_list:
            minfo: Dict[str, Any] = metric_info.get(mname, {})
            unit: str = minfo.get("unit", "")
            lower_better: bool = minfo.get("lower_better", True)
            direction: str = " (lower better)" if lower_better else " (higher better)"

            # Find best value
            values: Dict[str, float] = {
                a: agents_data[a].get(mname, 0.0) for a in agent_names
            }
            if lower_better:
                best_val: float = min(values.values()) if values else 0.0
            else:
                best_val = max(values.values()) if values else 0.0

            # Format cell values, highlight best with **
            cells: List[str] = []
            for a in agent_names:
                val: float = values[a]
                cell_str: str = f"{val:.4f}"
                if abs(val - best_val) < 1e-9:
                    cell_str = f"**{cell_str}**"
                cells.append(f" {cell_str} ")

            unit_str: str = f" ({unit})" if unit else ""
            row: str = f"| {mname}{unit_str}{direction} |" + "|".join(cells) + "|"
            lines.append(row)

        # Add best agent summary
        lines.append("")
        lines.append(f"**Best Agent:** {result.get('best_agent', 'N/A')}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    weld_type: str = DEFAULT_WELD_TYPE,
    max_steps: int = DEFAULT_MAX_STEPS,
    sac_checkpoint: str = "",
) -> Dict[str, Any]:
    """Run the full welding baseline evaluation for a single weld type.

    Convenience function that creates a WeldingEvaluator, runs it,
    and returns the result dictionary.

    Args:
        weld_type: Welding posture type (flat, horizontal, vertical, overhead).
        max_steps: Maximum simulation steps per agent.
        sac_checkpoint: Path to SAC checkpoint file. If empty, auto-resolve.

    Returns:
        Evaluation result dictionary with agents, metrics, metric_info,
        best_agent, and timestamp.
    """
    evaluator: WeldingEvaluator = WeldingEvaluator(
        weld_type=weld_type,
        max_steps=max_steps,
        sac_checkpoint=sac_checkpoint,
    )
    return evaluator.evaluate()


def run_multi_type_evaluation(
    weld_types: Optional[List[str]] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Run evaluation across multiple weld types and generate cross-type comparison.

    Args:
        weld_types: List of weld types to evaluate. If None, evaluates all 4 types.
        max_steps: Maximum simulation steps per agent.

    Returns:
        Dictionary with per-type results and cross-type summary:
        {
            "per_type": {weld_type: evaluation_result, ...},
            "cross_type_summary": {agent_name: {weld_type: episode_return, ...}, ...},
            "best_agent_overall": "Expert",
            "timestamp": "..."
        }
    """
    if weld_types is None:
        weld_types = ALL_WELD_TYPES

    print(f"\n{'#'*60}")
    print(f"# Multi-Type Welding Evaluation: {weld_types}")
    print(f"{'#'*60}")

    per_type: Dict[str, Dict[str, Any]] = {}
    cross_type: Dict[str, Dict[str, float]] = {}

    for wt in weld_types:
        print(f"\n{'='*60}")
        print(f"Evaluating weld type: {wt}")
        print(f"  Optimal params: {WELD_TYPE_OPTIMAL[wt]}")
        print(f"{'='*60}")

        result = run_evaluation(weld_type=wt, max_steps=max_steps)

        # Add weld-type-specific info
        result["optimal_params"] = WELD_TYPE_OPTIMAL[wt].tolist()
        per_type[wt] = result

        # Collect cross-type data
        for agent_name, metrics in result.get("agents", {}).items():
            if agent_name not in cross_type:
                cross_type[agent_name] = {}
            cross_type[agent_name][wt] = metrics.get("episode_return", 0.0)

    # Determine best agent overall (highest mean return across types)
    best_agent: str = "Expert"
    best_mean_return: float = float("-inf")
    for agent_name, type_returns in cross_type.items():
        mean_ret = float(np.mean(list(type_returns.values())))
        if mean_ret > best_mean_return:
            best_mean_return = mean_ret
            best_agent = agent_name

    return {
        "per_type": per_type,
        "cross_type_summary": cross_type,
        "best_agent_overall": best_agent,
        "weld_types_evaluated": weld_types,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    """CLI entry point: python benchmarks/welding_eval.py.

    Runs the evaluation, prints JSON and Markdown table to stdout.

    Supports:
      --weld-type flat|horizontal|vertical|overhead  (default: flat)
      --all-types                                    (evaluate all 4 types)
      --max-steps N                                  (default: 3500)

    Returns:
        Exit code (0 = success).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Welding baseline evaluation engine"
    )
    parser.add_argument(
        "--weld-type", type=str, default=DEFAULT_WELD_TYPE,
        choices=ALL_WELD_TYPES,
        help=f"Weld type to evaluate (default: {DEFAULT_WELD_TYPE})",
    )
    parser.add_argument(
        "--all-types", action="store_true",
        help="Evaluate all 4 weld types (flat, horizontal, vertical, overhead)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per agent (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output JSON file path (default: stdout only)",
    )
    args = parser.parse_args()

    if args.all_types:
        # ── Multi-type evaluation ──
        result = run_multi_type_evaluation(max_steps=args.max_steps)

        print("\n" + "=" * 60)
        print("Multi-Type Evaluation — Cross-Type Summary")
        print("=" * 60)

        # Print cross-type table
        weld_types = result["weld_types_evaluated"]
        agents = list(result["cross_type_summary"].keys())

        header = "| Agent |" + "|".join(f" {wt} " for wt in weld_types) + "| Mean |"
        sep = "|-------|" + "|".join(["------"] * len(weld_types)) + "|------|"
        print(header)
        print(sep)
        for agent in agents:
            cells = []
            vals = []
            for wt in weld_types:
                v = result["cross_type_summary"][agent].get(wt, 0.0)
                vals.append(v)
                cells.append(f" {v:.2f} ")
            mean_v = float(np.mean(vals))
            cells.append(f" {mean_v:.2f} ")
            print(f"| {agent} |" + "|".join(cells) + "|")

        print(f"\n**Best Agent Overall:** {result['best_agent_overall']}")

        # Print per-type details
        for wt in weld_types:
            type_result = result["per_type"][wt]
            evaluator = WeldingEvaluator(weld_type=wt)
            md_table = evaluator.generate_markdown_table(type_result)
            print(f"\n--- {wt.upper()} ---")
            print(f"Optimal: {WELD_TYPE_OPTIMAL[wt]}")
            print(md_table)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")

    else:
        # ── Single-type evaluation ──
        result = run_evaluation(
            weld_type=args.weld_type,
            max_steps=args.max_steps,
        )

        print(f"\nOptimal params for {args.weld_type}: {WELD_TYPE_OPTIMAL[args.weld_type]}")

        # Print JSON output
        print("\n" + "=" * 60)
        print("JSON Output")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Print Markdown table
        evaluator = WeldingEvaluator(weld_type=args.weld_type)
        md_table = evaluator.generate_markdown_table(result)

        print("\n" + "=" * 60)
        print("Markdown Comparison Table")
        print("=" * 60)
        print(md_table)
        print("=" * 60)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
