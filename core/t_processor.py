"""
T-Processor — IDO-Aware Hardware Coprocessor Simulation
========================================================

v0.16.25 P1: T-Processor Python Simulation

Simulates the T-Processor from the IDO/TOMAS nine-layer architecture.
The T-Processor is a dedicated hardware coprocessor that handles:

  1. η-ALU (Arithmetic Logic Unit): Computes GaussEx residual (η) in hardware
     — fast, deterministic, no floating-point surprises.

  2. Ψ-Check (Physical Constraint Checker): Evaluates ψ-Anchor constraints
     (MAX_TORQUE, MAX_VELOCITY, ZMP, ENERGY_DRIFT) in real-time, before
     the action reaches the actuator.

  3. κ-Snap FIFO (Audit Trail Buffer): Hardware FIFO buffer that stores
     the last N κ-Snap events. When the FIFO fills, oldest events are
     flushed to the Merkle chain (S-Layer) for permanent storage.

Hardware spec (target):
  - 65k gates @ 28nm CMOS
  - 3.3mW power consumption
  - 100MHz clock → 10ns per η computation
  - 256-entry κ-Snap FIFO (4KB SRAM)
  - 32-bit fixed-point arithmetic (Q16.16)

This Python simulation mirrors the hardware behavior for software testing
and integration with the MuJoCo-Bench-IDO framework.

L0 in the nine-layer architecture: T-Processor = "心脏" (Heart)
  - Constant rhythm: 100Hz tick
  - η-ALU: computes residual every tick
  - Ψ-Check: gates every action
  - κ-FIFO: records every step

Author: MuJoCo-Bench-IDO v0.16.25 — P1 Feature
"""

import time
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


# ── Hardware Constants ──

T_PROCESSOR_VERSION: str = "v0.16.25"
CLOCK_HZ: float = 100e6          # 100 MHz
TICK_HZ: float = 100.0           # 100 Hz control loop
TICK_PERIOD_S: float = 1.0 / TICK_HZ  # 10ms per tick
FIFO_DEPTH: int = 256            # 256-entry κ-Snap FIFO
Q16_16_SCALE: float = 65536.0    # Fixed-point Q16.16 scale factor
GATE_COUNT: int = 65000          # 65k gates
POWER_MW: float = 3.3            # 3.3 mW


@dataclass
class EtaALUResult:
    """η-ALU computation result.

    Attributes:
        eta: Computed η (GaussEx residual) value.
        eta_fixed: Fixed-point representation (Q16.16).
        computation_ns: Computation time in nanoseconds.
        overflow: Whether fixed-point overflow occurred.
    """
    eta: float = 0.0
    eta_fixed: int = 0
    computation_ns: float = 10.0  # 1 clock cycle at 100MHz
    overflow: bool = False


@dataclass
class PsiCheckResult:
    """Ψ-Check evaluation result.

    Attributes:
        passed: Whether all ψ-Anchor constraints passed.
        violations: List of violation names.
        check_ns: Check time in nanoseconds.
        clamped_action: Action after ψ-Anchor clamping (if any).
    """
    passed: bool = True
    violations: List[str] = field(default_factory=list)
    check_ns: float = 50.0  # 5 clock cycles
    clamped_action: Optional[np.ndarray] = None


@dataclass
class KappaSnapFIFOEntry:
    """κ-Snap FIFO buffer entry.

    Attributes:
        step: Simulation step number.
        timestamp: Simulation time (seconds).
        eta: η value at this step.
        snap_id: Merkle chain snap ID.
        psi_passed: Whether Ψ-Check passed.
        violation: Violation name (if any).
        action_hash: SHA-256 hash of action (first 8 chars).
    """
    step: int = 0
    timestamp: float = 0.0
    eta: float = 0.0
    snap_id: str = ""
    psi_passed: bool = True
    violation: str = ""
    action_hash: str = ""


class EtaALU:
    """η-ALU: Hardware GaussEx residual computation unit.

    Simulates the fixed-point arithmetic unit that computes the η (GaussEx
    residual) value. In hardware, this uses Q16.16 fixed-point arithmetic
    to avoid floating-point nondeterminism.

    The GaussEx residual is:
        η = ||obs - goal||² / (2 * δ_K²)

    In hardware, this is computed as:
        1. diff = obs - goal (vector subtraction, fixed-point)
        2. norm_sq = dot(diff, diff) (multiply-accumulate)
        3. η = norm_sq / (2 * δ_K²) (division, fixed-point)
    """

    def __init__(self, delta_k: float = 0.05) -> None:
        """Initialize η-ALU with tolerance δ_K.

        Args:
            delta_k: GaussEx tolerance parameter (smaller = stricter).
        """
        self.delta_k: float = delta_k
        self.delta_k_fixed: int = int(delta_k * Q16_16_SCALE)
        self._computation_count: int = 0
        self._total_time_ns: float = 0.0

    def compute(self, obs: np.ndarray, goal: np.ndarray) -> EtaALUResult:
        """Compute η (GaussEx residual) from observation and goal.

        Simulates the hardware computation pipeline:
        1. Convert to fixed-point (Q16.16)
        2. Vector subtraction
        3. Dot product (MAC unit)
        4. Division by 2*δ_K²
        5. Convert back to float

        Args:
            obs: Observation vector.
            goal: Goal vector (same shape as obs).

        Returns:
            EtaALUResult with computed η.
        """
        start_ns = 10.0  # 1 clock cycle at 100MHz

        # Convert to fixed-point
        obs_fp = (obs * Q16_16_SCALE).astype(np.int64)
        goal_fp = (goal * Q16_16_SCALE).astype(np.int64)

        # Vector subtraction (fixed-point)
        diff_fp = obs_fp - goal_fp

        # Dot product (MAC unit — multiply-accumulate)
        norm_sq_fp = int(np.dot(diff_fp, diff_fp))

        # Division by 2 * δ_K² (fixed-point)
        denom_fp = 2 * self.delta_k_fixed * self.delta_k_fixed
        if denom_fp > 0:
            # Q16.16 division: result_fp = (norm_sq_fp << 16) / denom_fp
            eta_fp = (norm_sq_fp << 16) // denom_fp
        else:
            eta_fp = 0

        # Check overflow (32-bit signed range)
        overflow = abs(eta_fp) > (2**31 - 1)

        # Convert back to float
        eta = float(eta_fp) / Q16_16_SCALE

        self._computation_count += 1
        self._total_time_ns += start_ns

        return EtaALUResult(
            eta=eta,
            eta_fixed=eta_fp,
            computation_ns=start_ns,
            overflow=overflow,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get ALU statistics."""
        return {
            "computation_count": self._computation_count,
            "total_time_ns": self._total_time_ns,
            "avg_time_ns": self._total_time_ns / max(self._computation_count, 1),
            "delta_k": self.delta_k,
            "clock_hz": CLOCK_HZ,
        }


class PsiChecker:
    """Ψ-Check: Real-time physical constraint evaluation unit.

    Evaluates ψ-Anchor constraints in hardware before the action reaches
    the actuator. This is the hardware implementation of the C-Layer
    (κ-Gate) in the IDO/TOMAS architecture.

    Constraints checked:
        - MAX_TORQUE: Per-actuator torque limit
        - MAX_VELOCITY: Per-joint velocity limit
        - MAX_GRIP_FORCE: Gripper force limit
        - ZMP: Zero Moment Point stability (locomotion)
        - ENERGY_DRIFT: Cumulative energy budget

    In hardware, each constraint is checked in parallel (single cycle),
    and any violation triggers a hardware interrupt that clamps the action.
    """

    def __init__(
        self,
        max_torque: float = 2.5,
        max_velocity: float = 3.0,
        max_grip_force: float = 2.0,
        zmp_margin: float = 0.05,
        max_energy_drift: float = 10.0,
    ) -> None:
        """Initialize Ψ-Checker with constraint thresholds.

        All thresholds are stored in fixed-point for hardware-accurate comparison.
        """
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        self.max_grip_force = max_grip_force
        self.zmp_margin = zmp_margin
        self.max_energy_drift = max_energy_drift

        # Fixed-point thresholds
        self._max_torque_fp = int(max_torque * Q16_16_SCALE)
        self._max_velocity_fp = int(max_velocity * Q16_16_SCALE)
        self._max_grip_force_fp = int(max_grip_force * Q16_16_SCALE)

        self._check_count: int = 0
        self._violation_count: int = 0
        self._cumulative_energy: float = 0.0

    def check(
        self,
        action: np.ndarray,
        joint_velocities: np.ndarray,
        joint_forces: Optional[np.ndarray] = None,
        gripper_indices: Optional[List[int]] = None,
    ) -> PsiCheckResult:
        """Evaluate all ψ-Anchor constraints in parallel.

        Simulates the parallel hardware evaluation. In real hardware, all
        checks complete in 5 clock cycles (50ns at 100MHz).

        Args:
            action: Action vector (joint position targets or torques).
            joint_velocities: Current joint velocities.
            joint_forces: Current joint forces (qfrc_actuator), optional.
            gripper_indices: Indices of gripper actuators in action vector.

        Returns:
            PsiCheckResult with pass/fail and clamped action.
        """
        check_ns = 50.0  # 5 clock cycles
        violations: List[str] = []
        clamped = action.copy()
        gripper_indices = gripper_indices or []

        # Check 1: MAX_TORQUE (parallel with other checks)
        if joint_forces is not None and len(joint_forces) > 0:
            for i, f in enumerate(joint_forces):
                if abs(float(f)) > self.max_torque:
                    violations.append(f"MAX_TORQUE[joint_{i}]")
                    # Clamp: scale down to max_torque
                    if abs(float(f)) > 0:
                        clamped[i] = clamped[i] * (self.max_torque / abs(float(f)))

        # Check 2: MAX_VELOCITY
        if len(joint_velocities) > 0:
            max_vel = float(np.max(np.abs(joint_velocities)))
            if max_vel > self.max_velocity:
                violations.append(f"MAX_VELOCITY({max_vel:.3f} > {self.max_velocity})")

        # Check 3: MAX_GRIP_FORCE
        if gripper_indices:
            for idx in gripper_indices:
                if idx < len(clamped) and abs(float(clamped[idx])) > self.max_grip_force:
                    violations.append(f"MAX_GRIP_FORCE[joint_{idx}]")
                    clamped[idx] = np.clip(clamped[idx], -self.max_grip_force, self.max_grip_force)

        # Check 4: ZMP (if COM data available — checked externally, here we just flag)
        # ZMP requires COM position/accel which isn't available in the action vector.
        # This is checked by the PsiAnchorGate.check_zmp() method in tomas_wrapper.py.

        # Check 5: ENERGY_DRIFT (tracked cumulatively)
        # Energy is tracked externally; here we just check the cumulative value.
        if self._cumulative_energy > self.max_energy_drift:
            violations.append(f"ENERGY_DRIFT({self._cumulative_energy:.2f}J > {self.max_energy_drift}J)")

        self._check_count += 1
        passed = len(violations) == 0
        if not passed:
            self._violation_count += 1

        return PsiCheckResult(
            passed=passed,
            violations=violations,
            check_ns=check_ns,
            clamped_action=clamped if not passed else None,
        )

    def update_energy(self, energy_drift: float) -> None:
        """Update cumulative energy drift."""
        self._cumulative_energy += energy_drift

    def reset_energy(self) -> None:
        """Reset energy tracker (call at episode start)."""
        self._cumulative_energy = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get checker statistics."""
        return {
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "violation_rate": self._violation_count / max(self._check_count, 1),
            "cumulative_energy_drift": self._cumulative_energy,
            "max_energy_drift": self.max_energy_drift,
        }


class KappaSnapFIFO:
    """κ-Snap FIFO: Hardware audit trail buffer.

    A 256-entry FIFO buffer that stores κ-Snap events. When the FIFO
    fills, the oldest entries are flushed to the Merkle chain (S-Layer)
    for permanent tamper-proof storage.

    In hardware, this is a 4KB SRAM block (256 entries × 16 bytes each).
    Each entry stores:
        - step (4 bytes)
        - eta_fp (4 bytes, Q16.16)
        - snap_id_hash (4 bytes, first 32 bits of SHA-256)
        - flags (4 bytes: psi_passed, violation_code)

    The FIFO supports:
        - push(): Add new entry (O(1))
        - flush(): Flush all entries to Merkle chain
        - peek(n): Peek at last N entries without removing
    """

    def __init__(self, depth: int = FIFO_DEPTH) -> None:
        """Initialize FIFO with specified depth.

        Args:
            depth: FIFO depth in entries. Default 256.
        """
        self.depth: int = depth
        self._buffer: deque = deque(maxlen=depth)
        self._flush_count: int = 0
        self._total_entries: int = 0

    def push(self, entry: KappaSnapFIFOEntry) -> None:
        """Push a new entry into the FIFO.

        If the FIFO is full, the oldest entry is automatically discarded
        (deque behavior with maxlen).

        Args:
            entry: KappaSnapFIFOEntry to push.
        """
        self._buffer.append(entry)
        self._total_entries += 1

    def flush(self) -> List[KappaSnapFIFOEntry]:
        """Flush all entries from the FIFO.

        Returns all entries and clears the buffer. In hardware, this
        triggers a DMA transfer to the S-Layer Merkle chain storage.

        Returns:
            List of all KappaSnapFIFOEntry objects that were in the FIFO.
        """
        entries = list(self._buffer)
        self._buffer.clear()
        self._flush_count += 1
        return entries

    def peek(self, n: int = 10) -> List[KappaSnapFIFOEntry]:
        """Peek at the last N entries without removing them.

        Args:
            n: Number of recent entries to peek.

        Returns:
            List of up to N most recent KappaSnapFIFOEntry objects.
        """
        return list(self._buffer)[-n:]

    def is_full(self) -> bool:
        """Check if the FIFO is full."""
        return len(self._buffer) >= self.depth

    def get_stats(self) -> Dict[str, Any]:
        """Get FIFO statistics."""
        return {
            "depth": self.depth,
            "current_entries": len(self._buffer),
            "total_entries_pushed": self._total_entries,
            "flush_count": self._flush_count,
            "utilization": len(self._buffer) / self.depth,
        }


class TProcessor:
    """T-Processor: IDO-aware hardware coprocessor simulation.

    Combines η-ALU, Ψ-Checker, and κ-Snap FIFO into a single coprocessor
    that operates at the L0 (Heart) level of the nine-layer architecture.

    The T-Processor runs at 100Hz (matching the MuJoCo simulation timestep)
    and provides:
        1. η computation (η-ALU) — every tick
        2. Action safety check (Ψ-Check) — every tick, before action
        3. Audit trail recording (κ-FIFO) — every tick, after action

    Usage:
        tproc = TProcessor(delta_k=0.05)

        # Per-tick processing:
        eta_result = tproc.compute_eta(obs, goal)
        psi_result = tproc.check_action(action, qvel, qfrc)
        tproc.record_snap(step, sim_time, eta_result.eta, psi_result)

        # Periodic flush:
        if tproc.fifo.is_full():
            entries = tproc.flush_fifo()
            # Store entries in Merkle chain...

    Attributes:
        eta_alu: EtaALU instance for η computation.
        psi_checker: PsiChecker instance for constraint checking.
        fifo: KappaSnapFIFO instance for audit trail buffering.
    """

    VERSION: str = T_PROCESSOR_VERSION

    # Hardware spec
    GATE_COUNT: int = GATE_COUNT
    POWER_MW: float = POWER_MW
    CLOCK_HZ: float = CLOCK_HZ
    TICK_HZ: float = TICK_HZ

    def __init__(
        self,
        delta_k: float = 0.05,
        max_torque: float = 2.5,
        max_velocity: float = 3.0,
        max_grip_force: float = 2.0,
        fifo_depth: int = FIFO_DEPTH,
    ) -> None:
        """Initialize T-Processor with all sub-units.

        Args:
            delta_k: GaussEx tolerance for η-ALU.
            max_torque: Maximum joint torque for Ψ-Check.
            max_velocity: Maximum joint velocity for Ψ-Check.
            max_grip_force: Maximum gripper force for Ψ-Check.
            fifo_depth: κ-Snap FIFO depth.
        """
        self.eta_alu: EtaALU = EtaALU(delta_k=delta_k)
        self.psi_checker: PsiChecker = PsiChecker(
            max_torque=max_torque,
            max_velocity=max_velocity,
            max_grip_force=max_grip_force,
        )
        self.fifo: KappaSnapFIFO = KappaSnapFIFO(depth=fifo_depth)

        self._tick_count: int = 0
        self._start_time: float = time.perf_counter()

    def tick(
        self,
        obs: np.ndarray,
        goal: np.ndarray,
        action: np.ndarray,
        joint_velocities: np.ndarray,
        joint_forces: Optional[np.ndarray] = None,
        gripper_indices: Optional[List[int]] = None,
    ) -> Tuple[EtaALUResult, PsiCheckResult, KappaSnapFIFOEntry]:
        """Process one T-Processor tick (100Hz).

        Executes the full T-Processor pipeline:
        1. η-ALU computes residual from obs and goal
        2. Ψ-Check evaluates action against constraints
        3. κ-FIFO records the audit entry

        Args:
            obs: Current observation vector.
            goal: Goal vector.
            action: Proposed action vector.
            joint_velocities: Current joint velocities.
            joint_forces: Current joint forces (optional).
            gripper_indices: Indices of gripper actuators (optional).

        Returns:
            Tuple of (EtaALUResult, PsiCheckResult, KappaSnapFIFOEntry).
        """
        # 1. Compute η
        eta_result = self.eta_alu.compute(obs, goal)

        # 2. Check action
        psi_result = self.psi_checker.check(
            action, joint_velocities, joint_forces, gripper_indices
        )

        # 3. Record audit entry
        action_hash = hashlib.sha256(action.tobytes()).hexdigest()[:8]
        snap_id = hashlib.sha256(
            f"{self._tick_count}_{eta_result.eta:.6f}_{action_hash}".encode()
        ).hexdigest()[:16]

        entry = KappaSnapFIFOEntry(
            step=self._tick_count,
            timestamp=self._tick_count * TICK_PERIOD_S,
            eta=eta_result.eta,
            snap_id=snap_id,
            psi_passed=psi_result.passed,
            violation=";".join(psi_result.violations) if psi_result.violations else "",
            action_hash=action_hash,
        )
        self.fifo.push(entry)

        self._tick_count += 1
        return eta_result, psi_result, entry

    def flush_fifo(self) -> List[KappaSnapFIFOEntry]:
        """Flush the κ-Snap FIFO to permanent storage."""
        return self.fifo.flush()

    def get_hardware_spec(self) -> Dict[str, Any]:
        """Get hardware specification."""
        return {
            "version": self.VERSION,
            "gate_count": self.GATE_COUNT,
            "power_mw": self.POWER_MW,
            "clock_hz": self.CLOCK_HZ,
            "tick_hz": self.TICK_HZ,
            "tick_period_s": TICK_PERIOD_S,
            "fifo_depth": self.fifo.depth,
            "arithmetic": "Q16.16 fixed-point",
            "process_node": "28nm CMOS",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive T-Processor statistics."""
        uptime = time.perf_counter() - self._start_time
        return {
            "tick_count": self._tick_count,
            "uptime_s": uptime,
            "effective_hz": self._tick_count / max(uptime, 0.001),
            "eta_alu": self.eta_alu.get_stats(),
            "psi_checker": self.psi_checker.get_stats(),
            "fifo": self.fifo.get_stats(),
            "hardware_spec": self.get_hardware_spec(),
        }

    def reset(self) -> None:
        """Reset T-Processor state (call at episode start)."""
        self._tick_count = 0
        self._start_time = time.perf_counter()
        self.psi_checker.reset_energy()
        self.fifo.flush()
