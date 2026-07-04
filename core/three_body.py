"""
Three-Body Architecture — 虚拟身体 / 软件身体 / 物理身体
=========================================================

v0.16.25 P2: Three-Body Architecture

Implements the "三身体" (Three-Body) architecture from the IDO/TOMAS
nine-layer cognitive framework:

  1. 虚拟身体 (Virtual Body): Behavior Print — digital twin in simulation
     - Runs in MuJoCo simulation
     - Generates "behavior prints" (行为打印) — observed motion patterns
     - Safe to experiment, no physical consequences

  2. 软件身体 (Software Body): Operation Print — middleware/abstraction layer
     - Translates behavior prints to operation prints (操作打印)
     - Handles action space mapping, coordinate transforms
     - Enforces action isomorphism (动作同构) between virtual and physical

  3. 物理身体 (Physical Body): Action Print — real hardware (SO-ARM100)
     - Executes actions on real hardware
     - Generates "action prints" (行动打印) — sensor feedback
     - Provides ground truth for sim-to-real gap measurement

The three bodies share the same cognitive architecture (L0-L8) but operate
at different fidelity levels. The κ-Snap audit trail records prints from
all three bodies, enabling cross-body verification.

Author: MuJoCo-Bench-IDO v0.16.25 — P2 Feature
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class BehaviorPrint:
    """行为打印 — Virtual Body output.

    Captures the agent's behavior in simulation (virtual body).
    This is the "digital shadow" of the action.

    Attributes:
        step: Simulation step.
        action: Action vector applied in simulation.
        obs: Observation after action.
        reward: Reward received.
        eta: η (GaussEx residual) after action.
        timestamp: Sim time.
    """
    step: int = 0
    action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    obs: np.ndarray = field(default_factory=lambda: np.zeros(10))
    reward: float = 0.0
    eta: float = 0.0
    timestamp: float = 0.0


@dataclass
class OperationPrint:
    """操作打印 — Software Body output.

    Translates behavior prints to operation-ready commands.
    Handles action space mapping, coordinate transforms, and
    action isomorphism verification.

    Attributes:
        step: Simulation step.
        mapped_action: Action after coordinate/scale mapping.
        isomorphism_ok: Whether action isomorphism check passed.
        isomorphism_error: Error metric (0 = perfect isomorphism).
        transform_matrix: Transform applied (if any).
        timestamp: Sim time.
    """
    step: int = 0
    mapped_action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    isomorphism_ok: bool = True
    isomorphism_error: float = 0.0
    transform_matrix: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class ActionPrint:
    """行动打印 — Physical Body output.

    Records the actual action executed on physical hardware.
    Includes sensor feedback for sim-to-real comparison.

    Attributes:
        step: Execution step.
        executed_action: Action actually executed (may differ from mapped).
        sensor_readings: Sensor feedback (joint angles, torques, forces).
        success: Whether execution succeeded.
        latency_ms: Execution latency in milliseconds.
        timestamp: Real time.
    """
    step: int = 0
    executed_action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    latency_ms: float = 0.0
    timestamp: float = 0.0


class VirtualBody:
    """虚拟身体 — Digital twin in MuJoCo simulation.

    Runs the agent's cognitive architecture in simulation and generates
    behavior prints. This is the safe experimentation environment where
    the agent can explore without physical consequences.

    Attributes:
        name: Body identifier.
        _print_buffer: Sliding window of recent behavior prints.
    """

    def __init__(self, name: str = "virtual_humanoid", buffer_size: int = 1000) -> None:
        self.name: str = name
        self._print_buffer: deque = deque(maxlen=buffer_size)
        self._step: int = 0

    def execute(self, action: np.ndarray, obs: np.ndarray, reward: float, eta: float) -> BehaviorPrint:
        """Execute action in simulation and record behavior print.

        Args:
            action: Action vector applied.
            obs: Observation after action.
            reward: Reward received.
            eta: η residual after action.

        Returns:
            BehaviorPrint recording this step.
        """
        print = BehaviorPrint(
            step=self._step,
            action=action.copy(),
            obs=obs.copy(),
            reward=reward,
            eta=eta,
            timestamp=self._step * 0.01,  # 100Hz sim
        )
        self._print_buffer.append(print)
        self._step += 1
        return print

    def get_recent_prints(self, n: int = 10) -> List[BehaviorPrint]:
        """Get the last N behavior prints."""
        return list(self._print_buffer)[-n:]

    def get_action_history(self) -> np.ndarray:
        """Get all action history as a numpy array."""
        if not self._print_buffer:
            return np.zeros((0, 7))
        return np.array([p.action for p in self._print_buffer])


class SoftwareBody:
    """软件身体 — Middleware/abstraction layer.

    Translates behavior prints from the virtual body to operation prints
    suitable for the physical body. Handles:
      - Action space mapping (sim → real joint limits)
      - Coordinate transforms (sim frame → real frame)
      - Action isomorphism verification (动作同构)

    Action Isomorphism (动作同构):
      The action spaces of virtual and physical bodies must be structurally
      isomorphic — there exists a bijective mapping φ: A_virtual → A_physical
      that preserves the action semantics. If the isomorphism is violated
      (e.g., sim joint range exceeds real hardware), the operation is flagged.

    Attributes:
        name: Body identifier.
        action_scale: Per-DOF scale factors (sim → real).
        action_offset: Per-DOF offset (sim → real).
        joint_limits_real: Real hardware joint limits (min, max) per DOF.
    """

    def __init__(
        self,
        name: str = "software_bridge",
        action_scale: Optional[np.ndarray] = None,
        action_offset: Optional[np.ndarray] = None,
        joint_limits_real: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        self.name: str = name
        # Default: identity mapping (sim = real)
        self.action_scale: np.ndarray = action_scale if action_scale is not None else np.ones(7)
        self.action_offset: np.ndarray = action_offset if action_offset is not None else np.zeros(7)
        self.joint_limits_real: Tuple[np.ndarray, np.ndarray] = joint_limits_real if joint_limits_real is not None else (
            np.array([-3.14, -3.14, -3.14, -3.14, -3.14, 0.0, -0.873]),
            np.array([3.14, 3.14, 3.14, 3.14, 3.14, 0.873, 0.0]),
        )
        self._print_buffer: deque = deque(maxlen=1000)
        self._isomorphism_errors: List[float] = []

    def translate(self, behavior_print: BehaviorPrint) -> OperationPrint:
        """Translate a behavior print to an operation print.

        Applies scale/offset mapping and checks action isomorphism.

        Args:
            behavior_print: Behavior print from virtual body.

        Returns:
            OperationPrint with mapped action and isomorphism check.
        """
        # Apply scale + offset
        mapped = behavior_print.action * self.action_scale + self.action_offset

        # Check isomorphism: mapped action must be within real joint limits
        min_lim, max_lim = self.joint_limits_real
        in_limits = np.all(mapped >= min_lim) and np.all(mapped <= max_lim)
        # Isomorphism error: how far outside limits (0 = within limits)
        violations = np.maximum(0, min_lim - mapped) + np.maximum(0, mapped - max_lim)
        iso_error = float(np.max(violations)) if len(violations) > 0 else 0.0

        # Build transform matrix (diagonal scale + offset)
        transform = np.diag(self.action_scale)

        op_print = OperationPrint(
            step=behavior_print.step,
            mapped_action=mapped,
            isomorphism_ok=in_limits,
            isomorphism_error=iso_error,
            transform_matrix=transform,
            timestamp=behavior_print.timestamp,
        )
        self._print_buffer.append(op_print)
        self._isomorphism_errors.append(iso_error)
        return op_print

    def get_isomorphism_stats(self) -> Dict[str, float]:
        """Get action isomorphism statistics."""
        errors = np.array(self._isomorphism_errors) if self._isomorphism_errors else np.zeros(1)
        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "violation_rate": float(np.mean(errors > 0.01)),
            "total_translations": len(self._isomorphism_errors),
        }


class PhysicalBody:
    """物理身体 — Real hardware (SO-ARM100) interface.

    Executes operation prints on real hardware and records action prints
    with sensor feedback. In simulation mode, this is a stub that
    simulates hardware execution with realistic latency and noise.

    Attributes:
        name: Body identifier.
        sim_mode: Whether running in simulation mode (no real hardware).
    """

    def __init__(self, name: str = "so_arm100", sim_mode: bool = True) -> None:
        self.name: str = name
        self.sim_mode: bool = sim_mode
        self._print_buffer: deque = deque(maxlen=1000)
        self._step: int = 0
        self._noise_std: float = 0.01  # 10mrad joint noise

    def execute(self, operation_print: OperationPrint) -> ActionPrint:
        """Execute an operation print on physical hardware.

        In sim mode, adds realistic noise and latency.

        Args:
            operation_print: Operation print from software body.

        Returns:
            ActionPrint with execution results.
        """
        if self.sim_mode:
            # Simulate execution with noise
            executed = operation_print.mapped_action + np.random.normal(0, self._noise_std, size=operation_print.mapped_action.shape)
            # Simulate sensor readings
            sensor = {f"joint_{i}_pos": float(executed[i]) for i in range(min(7, len(executed)))}
            sensor["gripper_force"] = float(abs(executed[5]) * 2.0) if len(executed) > 5 else 0.0
            latency = np.random.uniform(5.0, 15.0)  # 5-15ms latency
            success = True
        else:
            # Real hardware execution (placeholder)
            executed = operation_print.mapped_action
            sensor = {}
            latency = 0.0
            success = True  # Would check hardware response

        action_print = ActionPrint(
            step=self._step,
            executed_action=executed,
            sensor_readings=sensor,
            success=success,
            latency_ms=latency,
            timestamp=time.time(),
        )
        self._print_buffer.append(action_print)
        self._step += 1
        return action_print

    def get_sim_real_gap(self, virtual_body: VirtualBody) -> Dict[str, float]:
        """Compute sim-to-real gap between virtual and physical bodies.

        Args:
            virtual_body: Virtual body to compare against.

        Returns:
            Dict with gap metrics (action MSE, latency, success rate).
        """
        v_actions = virtual_body.get_action_history()
        p_prints = list(self._print_buffer)

        if len(v_actions) == 0 or len(p_prints) == 0:
            return {"action_mse": 0.0, "n_compared": 0}

        n = min(len(v_actions), len(p_prints))
        v_recent = v_actions[-n:]
        p_recent = np.array([p.executed_action for p in p_prints[-n:]])

        # Align shapes
        min_dim = min(v_recent.shape[1] if v_recent.ndim > 1 else 1,
                      p_recent.shape[1] if p_recent.ndim > 1 else 1)
        v_flat = v_recent[:, :min_dim] if v_recent.ndim > 1 else v_recent.reshape(-1, 1)
        p_flat = p_recent[:, :min_dim] if p_recent.ndim > 1 else p_recent.reshape(-1, 1)

        mse = float(np.mean((v_flat - p_flat) ** 2))
        return {
            "action_mse": mse,
            "n_compared": n,
            "mean_latency_ms": float(np.mean([p.latency_ms for p in p_prints[-n:]])),
            "success_rate": float(np.mean([p.success for p in p_prints[-n:]])),
        }


class ThreeBodySystem:
    """三身体系统 — Coordinates Virtual, Software, and Physical bodies.

    The three-body system manages the flow of prints across the three
    bodies and provides cross-body verification through the κ-Snap audit
    trail.

    Flow:
        Virtual Body (behavior print)
            → Software Body (operation print + isomorphism check)
                → Physical Body (action print + sensor feedback)
                    → κ-Snap audit (cross-body verification)

    Usage:
        system = ThreeBodySystem()
        bprint = system.virtual.execute(action, obs, reward, eta)
        oprint = system.software.translate(bprint)
        aprint = system.physical.execute(oprint)
        gap = system.get_sim_real_gap()
    """

    def __init__(self, sim_mode: bool = True) -> None:
        self.virtual: VirtualBody = VirtualBody()
        self.software: SoftwareBody = SoftwareBody()
        self.physical: PhysicalBody = PhysicalBody(sim_mode=sim_mode)

    def full_cycle(
        self,
        action: np.ndarray,
        obs: np.ndarray,
        reward: float,
        eta: float,
    ) -> Tuple[BehaviorPrint, OperationPrint, ActionPrint]:
        """Execute a full three-body cycle.

        Args:
            action: Action from cognitive architecture.
            obs: Current observation.
            reward: Current reward.
            eta: Current η residual.

        Returns:
            Tuple of (BehaviorPrint, OperationPrint, ActionPrint).
        """
        bprint = self.virtual.execute(action, obs, reward, eta)
        oprint = self.software.translate(bprint)
        aprint = self.physical.execute(oprint)
        return bprint, oprint, aprint

    def get_cross_body_stats(self) -> Dict[str, Any]:
        """Get cross-body verification statistics."""
        return {
            "virtual_steps": self.virtual._step,
            "software_steps": len(self.software._print_buffer),
            "physical_steps": self.physical._step,
            "isomorphism": self.software.get_isomorphism_stats(),
            "sim_real_gap": self.physical.get_sim_real_gap(self.virtual),
        }
