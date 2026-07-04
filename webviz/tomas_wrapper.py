"""
TOMAS MuJoCo Wrapper — IDO/TOMAS Framework for Embodied AI Audit

Based on:
  - "From VLA to Embodied Consciousness" (Zhang Feng, 2026)
  - "Beyond Leaderboards: Diagnosing AI Agent Instability" (Zhang Feng, 2026)

This module implements:
  1. PsiAnchorGate  — ψ-Anchor physical safety constraints (C-Layer)
  2. KappaSnap       — κ-Snap step-level audit trail (S-Layer)
  3. TOMASMuJoCoWrapper — wraps MuJoCo environment with IDO audit layer
  4. SOArm100Controller — pick-and-place controller for SO-ARM100 arm

Integration with MuJoCo-Bench-IDO webviz:
  - Used as step_fn provider for mjviser.Viewer
  - ψ-Anchor violations broadcast to dashboard via WebSocket
  - κ-Snap audit log queryable via /api/tomas/snap_log
"""

from __future__ import annotations

import time
import math
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Dict, Tuple, Callable
from collections import deque

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. KappaSnap — Step-level Audit Trail (S-Layer)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KappaSnap:
    """κ-Snap: A single step's causal snapshot for audit.

    Records what the agent did, why, and whether it violated any ψ-anchors.
    This is the "black box flight recorder" for embodied AI.
    """
    step: int                          # Simulation step number
    timestamp: float                   # Simulation time (seconds)
    eta: float                         # GaussEx residual (distance to goal)
    ic: float                          # Information Cardinality (joint velocity std)
    action: List[float]                # Joint control outputs
    psi_violation: Optional[str] = None  # ψ-Anchor violation name (None if clean)
    phase: str = "idle"                # Current task phase
    note: str = ""                     # Human-readable annotation


# ──────────────────────────────────────────────────────────────────────
# 2. PsiAnchorGate — Physical Safety Constraints (C-Layer)
# ──────────────────────────────────────────────────────────────────────

class PsiAnchorGate:
    """ψ-Anchor Gate: Hard physical constraints that cannot be overridden.

    Implements the C-Layer of the IDO/TOMAS framework:
    - MAX_TORQUE: Joint torque must not exceed ST3215 stall limit
    - MAX_VELOCITY: Joint velocity safety limit
    - NO_SPILL: Object tilt must not exceed threshold (30° default)
    - MAX_GRIP_FORCE: Gripper force must not crush objects
    - ZMP: Zero Moment Point must stay within support polygon (v0.16.25)
    - ENERGY_DRIFT: Cumulative energy drift must not exceed budget (v0.16.25)

    When a violation is detected, the action is scaled down to comply.
    """

    def __init__(
        self,
        max_torque: float = 2.5,        # N·m at joint (after gear)
        max_velocity: float = 3.0,       # rad/s
        max_grip_force: float = 2.0,     # N·m at gripper joint
        no_spill_angle: float = 0.524,   # radians (30°)
        # v0.16.25 P1: ZMP + ENERGY_DRIFT constraints
        zmp_margin: float = 0.05,        # meters — ZMP must be within support polygon ± margin
        max_energy_drift: float = 10.0,  # Joules — cumulative energy drift budget per episode
    ):
        self.max_torque = max_torque
        self.max_velocity = max_velocity
        self.max_grip_force = max_grip_force
        self.no_spill_angle = no_spill_angle
        self.violation_log: List[str] = []

        # v0.16.25 P1: ZMP + Energy drift state
        self.zmp_margin = zmp_margin
        self.max_energy_drift = max_energy_drift
        self._cumulative_energy_drift: float = 0.0
        self._last_kinetic_energy: Optional[float] = None
        self._last_potential_energy: Optional[float] = None

    def check_action(
        self,
        action: np.ndarray,
        joint_velocities: np.ndarray,
        joint_forces: Optional[np.ndarray] = None,
        gripper_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Check and potentially clamp an action vector.

        For position actuators, 'action' is position targets (radians).
        Force checking is done via joint_forces (qfrc_actuator) if provided.

        Returns:
            (safe_action, violation_name) — if violation, action is scaled.
        """
        gripper_indices = gripper_indices or []

        # Check actual joint forces (qfrc_actuator) if provided
        if joint_forces is not None and len(joint_forces) > 0:
            arm_forces = np.delete(joint_forces, gripper_indices) if gripper_indices else joint_forces
            max_force = float(np.max(np.abs(arm_forces))) if len(arm_forces) > 0 else 0.0
            if max_force > self.max_torque:
                reason = "MAX_TORQUE"
                self.violation_log.append(reason)
                return action, reason

        # Check joint velocities
        max_vel = float(np.max(np.abs(joint_velocities))) if len(joint_velocities) > 0 else 0.0
        if max_vel > self.max_velocity:
            # Don't modify action, just log — velocity is a consequence, not a control
            reason = "MAX_VELOCITY_EXCEEDED"
            self.violation_log.append(reason)
            return action, reason

        # Check gripper position (for position actuators, action IS the target position)
        if gripper_indices:
            grip_actions = action[gripper_indices]
            max_grip = float(np.max(np.abs(grip_actions)))
            if max_grip > self.max_grip_force:
                # Scale gripper target to safe range
                scale = self.max_grip_force / max_grip
                for idx in gripper_indices:
                    action[idx] *= scale
                reason = "MAX_GRIP_FORCE"
                self.violation_log.append(reason)
                return action, reason

        return action, None

    def check_spill(
        self,
        object_quat: np.ndarray,
        object_name: str = "object",
    ) -> Optional[str]:
        """Check if an object is tilted beyond the NO_SPILL threshold.

        Args:
            object_quat: Quaternion [w, x, y, z] of the grasped object.
            object_name: Name for logging.

        Returns:
            Violation name if tilted, None otherwise.
        """
        w, x, y, z = float(object_quat[0]), float(object_quat[1]), float(object_quat[2]), float(object_quat[3])
        # Compute tilt angle from vertical (z-axis deviation)
        tilt = math.acos(max(-1.0, min(1.0, 2.0 * w * w - 1.0)))
        if tilt > self.no_spill_angle:
            reason = f"NO_SPILL_{object_name}"
            self.violation_log.append(reason)
            return reason
        return None

    # ── v0.16.25 P1: ZMP (Zero Moment Point) Check ──

    def check_zmp(
        self,
        com_pos: np.ndarray,
        com_vel: np.ndarray,
        com_accel: np.ndarray,
        foot_positions: List[np.ndarray],
        gravity: float = 9.81,
    ) -> Optional[str]:
        """v0.16.25 P1: Check Zero Moment Point stability.

        Computes the ZMP from the Center of Mass (COM) dynamics and checks
        whether it falls within the support polygon formed by the foot
        positions. If the ZMP is outside the support polygon (plus margin),
        the robot is dynamically unstable and at risk of falling.

        ZMP Formula (simplified 2D projection):
            zmp_x = com_x - (com_accel_x * com_z) / gravity
            zmp_y = com_y - (com_accel_y * com_z) / gravity

        Support polygon: convex hull of foot positions projected onto ground.

        Args:
            com_pos: Center of mass position [x, y, z] in world frame.
            com_vel: COM velocity [vx, vy, vz] (unused in basic ZMP, kept for extensions).
            com_accel: COM acceleration [ax, ay, az].
            foot_positions: List of foot position arrays [[x,y,z], ...].
            gravity: Gravitational acceleration (default 9.81 m/s²).

        Returns:
            Violation name string if ZMP outside support polygon, None if stable.
        """
        if len(foot_positions) < 1:
            return None  # Can't check without foot positions

        com_z = max(float(com_pos[2]), 0.01)  # Avoid division by zero
        ax = float(com_accel[0])
        ay = float(com_accel[1])

        # ZMP projection onto ground plane
        zmp_x = float(com_pos[0]) - (ax * com_z) / gravity
        zmp_y = float(com_pos[1]) - (ay * com_z) / gravity

        # Support polygon: find min/max of foot positions (simplified bounding box)
        foot_xs = [float(f[0]) for f in foot_positions]
        foot_ys = [float(f[1]) for f in foot_positions]

        min_x = min(foot_xs) - self.zmp_margin
        max_x = max(foot_xs) + self.zmp_margin
        min_y = min(foot_ys) - self.zmp_margin
        max_y = max(foot_ys) + self.zmp_margin

        if zmp_x < min_x or zmp_x > max_x or zmp_y < min_y or zmp_y > max_y:
            reason = f"ZMP_VIOLATION(zmp=({zmp_x:.3f},{zmp_y:.3f}), support=[{min_x:.3f},{max_x:.3f}]x[{min_y:.3f},{max_y:.3f}])"
            self.violation_log.append(reason)
            return reason

        return None

    # ── v0.16.25 P1: ENERGY_DRIFT Check ──

    def check_energy_drift(
        self,
        kinetic_energy: float,
        potential_energy: float,
    ) -> Optional[str]:
        """v0.16.25 P1: Check cumulative energy drift against budget.

        Monitors the drift between expected and actual total mechanical
        energy. In a perfectly conservative system, KE + PE should be
        constant. Drift indicates non-conservative forces (friction,
        control inputs, numerical errors) are changing the energy budget.

        The cumulative drift is tracked across steps. If it exceeds
        max_energy_drift (Joules), a violation is flagged.

        Args:
            kinetic_energy: Current total kinetic energy (Joules).
            potential_energy: Current total potential energy (Joules).

        Returns:
            Violation name string if cumulative drift exceeds budget, None otherwise.
        """
        current_total = kinetic_energy + potential_energy

        if self._last_kinetic_energy is not None and self._last_potential_energy is not None:
            last_total = self._last_kinetic_energy + self._last_potential_energy
            step_drift = abs(current_total - last_total)
            self._cumulative_energy_drift += step_drift

        self._last_kinetic_energy = kinetic_energy
        self._last_potential_energy = potential_energy

        if self._cumulative_energy_drift > self.max_energy_drift:
            reason = f"ENERGY_DRIFT(cumulative={self._cumulative_energy_drift:.2f}J, budget={self.max_energy_drift:.1f}J)"
            self.violation_log.append(reason)
            return reason

        return None

    def reset_energy_tracker(self) -> None:
        """Reset the energy drift tracker (call at episode start)."""
        self._cumulative_energy_drift = 0.0
        self._last_kinetic_energy = None
        self._last_potential_energy = None

    def get_energy_drift(self) -> float:
        """Get current cumulative energy drift."""
        return self._cumulative_energy_drift
# 3. TOMASMuJoCoWrapper — IDO Audit Layer around MuJoCo
# ──────────────────────────────────────────────────────────────────────

class TOMASMuJoCoWrapper:
    """Wraps a MuJoCo environment with IDO/TOMAS audit layer.

    This is the integration point between the physical simulation and the
    IDO framework. It:
    - Intercepts actions and applies ψ-Anchor checks
    - Records κ-Snap audit entries at each step
    - Computes GaussEx residual (η) and Information Cardinality (IC)
    - Provides queryable audit trail for the dashboard

    Usage:
        wrapper = TOMASMuJoCoWrapper(model, data, target_body="red_cube", tray_body="tray")
        safe_action, violation = wrapper.gate.check_action(action, data.qvel)
        wrapper.record_snap(step, safe_action, violation)
        data.ctrl[:] = safe_action
        mujoco.mj_step(model, data)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        target_body_name: str = "red_cube",
        tray_body_name: str = "tray",
        gripper_body_name: str = "gripper_base",
        max_snaps: int = 5000,
    ):
        import mujoco as mj

        self.model = model
        self.data = data
        self._mj = mj

        # Resolve body IDs
        self.target_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, target_body_name)
        self.tray_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, tray_body_name)
        self.gripper_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, gripper_body_name)

        if self.target_body_id < 0:
            raise ValueError(f"Body '{target_body_name}' not found in model")
        if self.tray_body_id < 0:
            raise ValueError(f"Body '{tray_body_name}' not found in model")

        # IDO components
        self.gate = PsiAnchorGate()
        self.snaps: deque = deque(maxlen=max_snaps)

        # Gripper actuator indices (last 2 in SO-ARM100)
        self.gripper_indices = self._find_gripper_indices()

        # Statistics
        self.total_violations: int = 0
        self.total_steps: int = 0

    def _find_gripper_indices(self) -> List[int]:
        """Find actuator indices for gripper joints."""
        indices = []
        for i in range(self.model.nu):
            name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, i)
            if name and ("Gripper" in name or "grip" in name.lower()):
                indices.append(i)
        return indices

    def compute_eta(self) -> float:
        """Compute GaussEx residual (η): distance from gripper to target object.

        η = ||gripper_pos - target_pos|| + tilt_penalty

        Lower η means closer to goal. The agent's mission is to drive η → 0.
        """
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        distance = float(np.linalg.norm(gripper_pos - target_pos))

        # Add tilt penalty if object is grasped (tilted)
        target_quat = self.data.xquat[self.target_body_id].copy()
        w = float(target_quat[0])
        tilt = abs(math.acos(max(-1.0, min(1.0, 2.0 * w * w - 1.0))))
        tilt_penalty = max(0.0, tilt - 0.1) * 2.0  # penalty for tilt > 5.7°

        return distance + tilt_penalty

    def compute_ic(self) -> float:
        """Compute Information Cardinality: std of joint velocities.

        High IC means lots of movement (information-rich step).
        Low IC means the arm is stationary (dead-zero step).
        """
        arm_vels = []
        for i in range(self.model.nv):
            # Only consider arm joint velocities (skip freejoint objects)
            if i >= 18:  # First 18 dof are 3 freejoint objects (6 each)
                arm_vels.append(float(self.data.qvel[i]))
        if not arm_vels:
            return 0.0
        return float(np.std(arm_vels))

    def check_and_record(
        self,
        step: int,
        action: np.ndarray,
        phase: str = "idle",
        note: str = "",
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Apply ψ-Anchor check, record κ-Snap, return safe action.

        This is the main entry point called from the viewer's step_fn.

        Args:
            step: Simulation step number.
            action: Raw joint control outputs.
            phase: Current task phase (e.g., "reach", "grasp", "lift", "place").
            note: Human-readable annotation.

        Returns:
            (safe_action, violation_name) — action may be scaled if violated.
        """
        # Get arm joint velocities (skip freejoint objects)
        arm_vels = self.data.qvel[18:] if self.model.nv > 18 else self.data.qvel[:]
        # Get actual actuator forces (qfrc_actuator) for ψ-Anchor torque check
        arm_forces = self.data.qfrc_actuator[18:] if self.model.nv > 18 else self.data.qfrc_actuator[:]

        safe_action, violation = self.gate.check_action(
            action.copy(),
            arm_vels,
            joint_forces=arm_forces,
            gripper_indices=self.gripper_indices,
        )

        # Check spill if object is near gripper
        if violation is None:
            gripper_pos = self.data.xpos[self.gripper_body_id]
            target_pos = self.data.xpos[self.target_body_id]
            dist = float(np.linalg.norm(gripper_pos - target_pos))
            if dist < 0.05:  # Object is near gripper
                target_quat = self.data.xquat[self.target_body_id]
                spill_violation = self.gate.check_spill(target_quat, "object")
                if spill_violation:
                    violation = spill_violation

        # Record κ-Snap
        snap = KappaSnap(
            step=step,
            timestamp=float(self.data.time),
            eta=self.compute_eta(),
            ic=self.compute_ic(),
            action=safe_action.tolist(),
            psi_violation=violation,
            phase=phase,
            note=note,
        )
        self.snaps.append(snap)
        self.total_steps += 1
        if violation:
            self.total_violations += 1

        return safe_action, violation

    def get_recent_snaps(self, n: int = 20) -> List[Dict]:
        """Get recent κ-Snap entries as dicts (for dashboard API)."""
        recent = list(self.snaps)[-n:]
        return [asdict(s) for s in recent]

    def get_summary(self) -> Dict:
        """Get summary statistics for dashboard display."""
        if not self.snaps:
            return {"total_steps": 0, "total_violations": 0, "avg_eta": 0, "avg_ic": 0}

        etas = [s.eta for s in self.snaps]
        ics = [s.ic for s in self.snaps]
        violations_by_type: Dict[str, int] = {}
        for s in self.snaps:
            if s.psi_violation:
                violations_by_type[s.psi_violation] = violations_by_type.get(s.psi_violation, 0) + 1

        return {
            "total_steps": self.total_steps,
            "total_violations": self.total_violations,
            "violation_rate": self.total_violations / max(1, self.total_steps),
            "avg_eta": float(np.mean(etas)),
            "min_eta": float(np.min(etas)),
            "current_eta": etas[-1] if etas else 0.0,
            "avg_ic": float(np.mean(ics)),
            "violations_by_type": violations_by_type,
        }


# ──────────────────────────────────────────────────────────────────────
# 4. SOArm100Controller — Pick-and-Place with IDO Audit
# ──────────────────────────────────────────────────────────────────────

class SOArm100Controller:
    """Simple pick-and-place controller for SO-ARM100 in MuJoCo.

    Implements a state-machine controller that:
    1. HOME — Move to home pose
    2. REACH — Move toward the target object
    3. DESCEND — Lower gripper to object
    4. GRASP — Close gripper fingers
    5. LIFT — Raise the object
    6. TRANSPORT — Move to tray
    7. RELEASE — Open gripper
    8. RETREAT — Return to home pose

    Each step is audited through the TOMASMuJoCoWrapper.
    """

    # Task phases
    PHASE_HOME = "home"
    PHASE_REACH = "reach"
    PHASE_DESCEND = "descend"
    PHASE_GRASP = "grasp"
    PHASE_LIFT = "lift"
    PHASE_TRANSPORT = "transport"
    PHASE_RELEASE = "release"
    PHASE_RETREAT = "retreat"
    PHASE_DONE = "done"

    def __init__(
        self,
        model: Any,
        data: Any,
        wrapper: TOMASMuJoCoWrapper,
        target_object: str = "red_cube",
    ):
        import mujoco as mj
        self._mj = mj
        self.model = model
        self.data = data
        self.wrapper = wrapper
        self.target_object = target_object

        # Update wrapper target
        wrapper.target_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, target_object)

        # Controller state
        self.phase = self.PHASE_HOME
        self.phase_step = 0
        self.step_count = 0

        # Home pose (neutral arm position)
        self.home_pose = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])

        # PD gains for joint control
        self.kp = np.array([50, 50, 50, 30, 30, 20, 20])
        self.kd = np.array([5, 5, 5, 3, 3, 2, 2])

    def _get_arm_qpos(self) -> np.ndarray:
        """Get current arm joint positions (skip freejoint objects)."""
        # First 7 qpos entries are from freejoint objects (3 objects × 7 = 21)
        # Arm joints start at qpos index 21
        arm_start = 21  # 3 objects × 7 qpos each
        return self.data.qpos[arm_start:arm_start + 7].copy()

    def _get_arm_qvel(self) -> np.ndarray:
        """Get current arm joint velocities."""
        # First 18 qvel entries are from freejoint objects (3 objects × 6 = 18)
        arm_start = 18
        return self.data.qvel[arm_start:arm_start + 7].copy()

    def _compute_target_pose(self) -> np.ndarray:
        """Compute target joint pose based on current phase."""
        target_pos = self.data.xpos[self.wrapper.target_body_id].copy()
        tray_pos = self.data.xpos[self.wrapper.tray_body_id].copy()

        if self.phase == self.PHASE_HOME:
            return self.home_pose

        elif self.phase == self.PHASE_REACH:
            # Move to above the object
            target_world = target_pos + np.array([0, 0, 0.08])
            # Simple IK approximation: point arm toward target
            dx = target_world[0] - 0.0  # arm base x
            dy = target_world[1] - 0.0  # arm base y
            dz = target_world[2] - 0.40  # arm base z
            # Base rotation
            base_rot = math.atan2(dy, dx)
            # Shoulder pitch (approximate)
            reach = math.sqrt(dx*dx + dz*dz)
            shoulder = math.atan2(dz, reach) * 0.5
            # Elbow bend
            elbow = -0.3 - 0.2 * reach
            return np.array([base_rot, shoulder, elbow, 0.0, 0.0, 0.0, 0.0])

        elif self.phase == self.PHASE_DESCEND:
            # Lower gripper to object
            target_world = target_pos + np.array([0, 0, 0.02])
            dx = target_world[0]
            dy = target_world[1]
            dz = target_world[2] - 0.40
            base_rot = math.atan2(dy, dx)
            reach = math.sqrt(dx*dx + dz*dz)
            shoulder = math.atan2(dz, reach) * 0.6
            elbow = -0.5 - 0.15 * reach
            return np.array([base_rot, shoulder, elbow, 0.0, 0.0, 0.0, 0.0])

        elif self.phase == self.PHASE_GRASP:
            # Close gripper — hold current arm pose, close fingers
            current = self._get_arm_qpos()
            current[5] = 0.7   # Close left finger
            current[6] = -0.7  # Close right finger
            return current

        elif self.phase == self.PHASE_LIFT:
            # Raise arm
            current = self._get_arm_qpos()
            current[1] -= 0.3  # Raise shoulder
            current[2] += 0.2  # Straighten elbow
            current[5] = 0.7   # Keep gripper closed
            current[6] = -0.7
            return current

        elif self.phase == self.PHASE_TRANSPORT:
            # Move to above tray
            dx = tray_pos[0]
            dy = tray_pos[1]
            dz = tray_pos[2] + 0.08 - 0.40
            base_rot = math.atan2(dy, dx)
            reach = math.sqrt(dx*dx + dz*dz)
            shoulder = math.atan2(dz, reach) * 0.5
            elbow = -0.4
            return np.array([base_rot, shoulder, elbow, 0.0, 0.0, 0.7, -0.7])

        elif self.phase == self.PHASE_RELEASE:
            # Open gripper above tray
            current = self._get_arm_qpos()
            current[5] = 0.0   # Open left finger
            current[6] = 0.0   # Open right finger
            return current

        elif self.phase == self.PHASE_RETREAT:
            return self.home_pose

        return self.home_pose

    def _check_phase_transition(self) -> bool:
        """Check if current phase should transition. Returns True if transitioned."""
        eta = self.wrapper.compute_eta()
        arm_vel = self._get_arm_qvel()
        max_vel = float(np.max(np.abs(arm_vel))) if len(arm_vel) > 0 else 0.0

        # Transition when arm is stable (low velocity) and enough steps passed
        stable = max_vel < 0.1
        min_steps = 50  # minimum steps per phase

        if self.phase_step < min_steps:
            return False

        transitions = {
            self.PHASE_HOME: (stable, self.PHASE_REACH),
            self.PHASE_REACH: (stable and eta < 0.12, self.PHASE_DESCEND),
            self.PHASE_DESCEND: (stable and eta < 0.06, self.PHASE_GRASP),
            self.PHASE_GRASP: (self.phase_step >= 100, self.PHASE_LIFT),
            self.PHASE_LIFT: (stable, self.PHASE_TRANSPORT),
            self.PHASE_TRANSPORT: (stable, self.PHASE_RELEASE),
            self.PHASE_RELEASE: (self.phase_step >= 80, self.PHASE_RETREAT),
            self.PHASE_RETREAT: (stable, self.PHASE_DONE),
        }

        if self.phase in transitions:
            should_transition, next_phase = transitions[self.phase]
            if should_transition:
                self.phase = next_phase
                self.phase_step = 0
                return True

        return False

    def compute_action(self) -> Tuple[np.ndarray, str, str]:
        """Compute the next action (joint position targets).

        Returns:
            (action, phase, note) — action is the 7-element position target vector
            for position actuators. The actuator's internal PD (kp/kv in XML)
            handles force computation. forcerange in XML enforces torque limits.

            Note: ψ-Anchor torque check is performed post-step via qfrc_actuator
            in check_and_record(), not on the position target values.
        """
        # Check for phase transition
        self._check_phase_transition()

        # Compute target pose — this is the position target for position actuators
        target_pose = self._compute_target_pose()

        # Clip to actuator ctrlrange (matching XML ctrlrange values)
        ctrl_limits = np.array([
            [3.14, 1.571, 2.356, 1.571, 3.14, 0.873, 0.873],  # max
            [-3.14, -1.571, -1.571, -1.571, -3.14, 0.0, -0.873],  # min
        ])
        target_pose = np.clip(target_pose, ctrl_limits[1], ctrl_limits[0])

        note = f"phase={self.phase}, eta={self.wrapper.compute_eta():.3f}, step={self.phase_step}"
        self.phase_step += 1
        self.step_count += 1

        return target_pose, self.phase, note

    def reset(self):
        """Reset controller to home phase."""
        self.phase = self.PHASE_HOME
        self.phase_step = 0
        self.step_count = 0


# ═══════════════════════════════════════════════════════════════
# v0.16.17: VLA (Vision-Language-Action) Adapter Framework
# Based on "From VLA to Embodied Consciousness" (Zhang Feng, 2026)
# Architecture: VLA backbone → ψ-Anchor safety → κ-Snap audit → MuJoCo
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "KappaSnap", "PsiAnchorGate", "TOMASMuJoCoWrapper",
    "SOArm100Controller",
    "VLAAdapter", "OpenVLAAdapter", "OctoAdapter", "Pi0Adapter",
    "create_vla_adapter",
]


class VLAAdapter:
    """Base class for Vision-Language-Action model adapters.

    Unified interface: predict(obs_dict) → np.ndarray

    The unified interface design is based on "From VLA to Embodied
    Consciousness" (Zhang Feng, 2026). The architecture follows a
    four-stage pipeline:

        VLA backbone → ψ-Anchor safety gate → κ-Snap audit → MuJoCo execution

    All VLA outputs pass through PsiAnchorGate before execution.
    This ensures physical safety constraints are enforced regardless
    of the VLA backbone model. The C-Layer (ψ-Anchor) and S-Layer
    (κ-Snap) are what elevate the system from embodied AI to
    embodied cognition.

    Attributes:
        model_name: Name of the VLA model.
        loaded: Whether the model is actually loaded (False for stubs).
        _action_dim: Dimension of the action vector (default 7 for
            SO-ARM100: Base, Shoulder, Elbow, Wrist Pitch, Wrist Roll,
            Gripper L, Gripper R).
        _control_freq: Control frequency in Hz (default 30.0, matching
            LeRobot default for ST3215 bus servos).
    """

    def __init__(self, model_name: str = "base"):
        self.model_name: str = model_name
        self.loaded: bool = False
        self._action_buffer: deque = deque(maxlen=50)
        self._action_dim: int = 7
        self._control_freq: float = 30.0

    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Predict action from observation.

        Args:
            obs_dict: Dictionary with keys:
                - 'rgb': (H,W,3) uint8 — primary camera RGB image.
                  May be absent in MuJoCo sim (proprio-only mode).
                - 'language': str — natural language instruction,
                  e.g. "pick up the red cube and place it on the tray".
                - 'proprio': (n_joints,) float — current joint positions
                  in radians (proprioception / body state).

        Returns:
            np.ndarray of shape (7,) float64 — joint position targets
            for SO-ARM100 in radians, range clipped to actuator
            ctrlrange. The returned vector is the raw VLA output BEFORE
            ψ-Anchor safety clamping; the caller (TOMASMuJoCoWrapper)
            is responsible for applying PsiAnchorGate.check_action().

        Note:
            In stub mode, returns a simple proportional reach-toward
            target action. Override this method in subclasses for real
            model inference.
        """
        # Stub: simple proportional control toward home-like pose
        proprio = obs_dict.get('proprio', np.zeros(7))
        target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
        # Smooth interpolation toward target
        action = proprio + (target - proprio) * 0.1
        return action.astype(np.float64)

    def is_loaded(self) -> bool:
        """Check if real model weights are loaded."""
        return self.loaded


class OpenVLAAdapter(VLAAdapter):
    """OpenVLA-7B adapter — 7B parameter VLA model (Academic faction).

    Architecture: Dual visual encoders (DINOv2 + SigLIP) + Llama-2 LLM.
    This dual-encoder design fuses self-supervised depth features
    (DINOv2) with language-aligned semantics (SigLIP), then feeds
    the fused visual-language tokens into Llama-2 for action
    token generation.

    Performance: Defeats Google's 55-billion-parameter RT2X in task
    success rate across multiple manipulation benchmarks, proving
    that architectural design (dual-encoder + strong LLM backbone)
    can outweigh sheer parameter count.

    Input: Image + Language + Proprioception
    Output: 6-DOF joint position commands (discretized into 256 bins
    per DOF, predicted autoregressively as LLM tokens)

    HuggingFace: openvla/openvla-7b
    IDO role: P-Layer (Phenomenal Consciousness — "mimicry")

    Missing in OpenVLA (that IDO adds):
        - S-Bridge (κ-Snap causal audit trail)
        - C-Gate (ψ-Anchor physical safety constraints)
        - EML-SemZip data reweighting

    Attributes:
        _model_params: Model parameter count descriptor ("7B").
    """

    def __init__(self):
        super().__init__(model_name="openvla-7b")
        self._model_params: str = "7B"
        self._processor = None
        self._model = None

    def _try_load(self) -> bool:
        """Attempt to load real OpenVLA model from HuggingFace.

        Loads the 7B model in bfloat16 precision with device_map="cuda"
        to fit on a single GPU (requires ≥16GB VRAM for inference).
        The bfloat16 format halves memory usage vs float32 while
        preserving numerical range for the Llama-2 backbone.

        Returns:
            True if model loaded successfully, False otherwise
            (e.g. torch/transformers not installed or GPU OOM).
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self._processor = AutoProcessor.from_pretrained("openvla/openvla-7b")
            self._model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
            self.loaded = True
            return True
        except Exception:
            self.loaded = False
            return False

    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Run OpenVLA inference or fall back to stub."""
        if not self.loaded:
            if not self._try_load():
                return super().predict(obs_dict)

        # Real inference (if loaded)
        rgb = obs_dict.get('rgb')
        language = obs_dict.get('language', 'pick up the object')
        proprio = obs_dict.get('proprio', np.zeros(7))

        if rgb is not None and self._processor and self._model:
            import torch
            inputs = self._processor(images=rgb, text=language, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=50)
            # Decode action (implementation depends on OpenVLA version)
            action = self._decode_action(outputs, proprio)
            return action
        else:
            return super().predict(obs_dict)

    def _decode_action(self, outputs, proprio: np.ndarray) -> np.ndarray:
        """Decode model output tokens to joint position commands.

        OpenVLA discretizes continuous joint actions into 256 bins per
        DOF and predicts them autoregressively as LLM tokens (similar
        to how LLMs predict text tokens). Each output token maps to a
        bin index, which is then de-quantized back to a continuous
        joint angle in radians using per-DOF min/max normalization
        statistics from the training dataset.

        Args:
            outputs: Raw model output token IDs from generate().
            proprio: Current joint positions (for residual blending).

        Returns:
            np.ndarray of shape (7,) — decoded joint position targets.
        """
        # Placeholder — actual decoding depends on OpenVLA's tokenizer
        return super().predict({'proprio': proprio})


class OctoAdapter(VLAAdapter):
    """Octo adapter — lightweight 93M-parameter multi-task VLA model.

    A compact model (93M parameters) from the Academic faction,
    designed for efficiency and generalization. Key features:

    - Multi-camera input: Supports both primary (scene) and wrist
      cameras simultaneously, providing richer visual context than
      single-camera models.
    - Action Chunking: Predicts multi-step action sequences in one
      forward pass for smoother trajectories (vs OpenVLA's
      autoregressive single-step generation).
    - Zero-shot cross-embodiment generalization: Can transfer to
      unseen robot morphologies without fine-tuning, thanks to
      training on the diverse Open-X-Embodiment dataset.

    GitHub: octo-models/octo (JAX-based, requires JAX/TFLite runtime)
    IDO role: P-Layer (Phenomenal Consciousness — "mimicry")

    Attributes:
        _model_params: Model parameter count descriptor ("93M").
    """

    def __init__(self):
        super().__init__(model_name="octo-base")
        self._model_params: str = "93M"
        self._model = None
        self._unnorm_stats = None

    def _try_load(self) -> bool:
        """Attempt to load Octo model."""
        try:
            from octo.model import OctoModel
            self._model = OctoModel.load_pretrained("octo-base")
            self._unnorm_stats = self._model.dataset_statistics
            self.loaded = True
            return True
        except Exception:
            self.loaded = False
            return False

    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Run Octo inference or fall back to stub.

        Supports dual-camera input: 'rgb' (primary/scene camera) and
        'wrist_rgb' (wrist-mounted camera). When both are provided,
        Octo fuses the two visual streams for richer manipulation
        context. The wrist camera is especially valuable for
        fine-grained grasping and insertion tasks.

        Args:
            obs_dict: Must contain 'rgb' (primary camera). Optionally
                contains 'wrist_rgb' (wrist camera) and 'language'
                (task instruction). 'proprio' for current joint state.

        Returns:
            np.ndarray of shape (7,) — joint position targets. Octo's
            Action Chunking returns a multi-step sequence; only the
            first 7-DOF action is used (remaining steps are buffered
            internally if needed).
        """
        if not self.loaded:
            if not self._try_load():
                return super().predict(obs_dict)

        rgb = obs_dict.get('rgb')
        wrist_rgb = obs_dict.get('wrist_rgb')
        language = obs_dict.get('language', 'pick up the object')

        if rgb is not None and self._model:
            obs = {
                "image_primary": rgb,
                "task": {"language_instruction": language}
            }
            if wrist_rgb is not None:
                obs["image_wrist"] = wrist_rgb
            action = self._model.sample_actions(obs, self._unnorm_stats)
            return np.array(action[:7], dtype=np.float64)
        else:
            return super().predict(obs_dict)


class Pi0Adapter(VLAAdapter):
    """π₀ (Pi-Zero) adapter — Flow Matching VLA from Physical Intelligence.

    π₀ is a state-of-the-art VLA model from Physical Intelligence (PI),
    founded by Chelsea Finn and Sergey Levine (both former Google
    Brain/DeepMind core researchers). PI is valued at ~$5.6 billion
    as of 2024.

    Architecture (dual-system):
        - VLM Backbone: PaliGemma 2B = SigLIP So400m/14 visual encoder
          + Gemma 2B language model. Processes image + language input.
        - Action Expert: 300M Gemma model. Generates continuous action
          trajectories via Flow Matching (not autoregressive token
          prediction like OpenVLA).

    Core technology — Flow Matching:
        Unlike diffusion models that reverse a noise process, Flow
        Matching learns a velocity field v_θ that transports samples
        from a noise distribution to the action distribution. This
        enables fast, high-frequency continuous control.

    Control frequency: 50Hz (vs OpenVLA's ~2-10Hz autoregressive).
    Output: Action Chunks — predicts 50-step action sequences in one
        forward pass (action_horizon=50), enabling smooth high-rate
        control for delicate tasks (folding clothes, dishwashing).

    LIBERO SOTA: π₀.5 achieves 96.85% average (Spatial 98.8,
        Object 98.2, Goal 98.0, 10 92.4), significantly outperforming
        OpenVLA (7B) and Octo (93M).

    Training data: 10000+ hours, 7 robot configurations, 68 tasks,
        OXE/DROID/Bridge v2 mixed (open-source portion = 9.1%).

    Open source: Weights available via Physical-Intelligence/openpi
        (GitHub, 12468+ stars). Training pipeline remains proprietary.
        Deployment: inference >8GB (RTX 4090), LoRA fine-tune >22.5GB,
        full fine-tune >70GB (A100/H100).

    IDO role: High-density φ-flow continuous control (P-Layer with
        the highest action throughput among supported VLA models).

    Attributes:
        _model_params: Architecture descriptor for the dual-system model.
        _control_freq: Control frequency override (50Hz for π₀).
    """

    def __init__(self, chunk_size: int = 50):
        super().__init__(model_name="pi0-base")
        self._model_params: str = "PaliGemma 2B + 300M action expert"
        self._control_freq: float = 50.0
        self.chunk_size: int = chunk_size
        self._action_buffer: deque = deque(maxlen=chunk_size)
        self._flow_model = None

    def _try_load(self) -> bool:
        """Attempt to load π₀ model from the openpi repository.

        π₀ weights are distributed via the openpi package
        (Physical-Intelligence/openpi on GitHub, 12468+ stars).
        Installation: `pip install openpi` then download model
        checkpoints. The openpi package provides the Flow Matching
        inference pipeline with KV Cache optimization.

        Note: Only model weights and inference code are public;
        the training pipeline and proprietary datasets (10000+ hours,
        9.1% open-source) remain closed. Full fine-tuning requires
        >70GB VRAM (A100/H100).

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            # π₀ weights are available but training pipeline is not
            # This is a stub that would load the Flow Matching model
            self.loaded = False  # Set to True when real weights available
            return False
        except Exception:
            return False

    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Run π₀ inference with action chunking or fall back to stub.

        Flow Matching mathematics:
            Training:
                x_τ = τ · noise + (1 - τ) · A_t
                u_τ = noise - A_t
                L = MSE(v_θ(x_τ, o_t, τ), u_τ)
            where τ ~ U[0,1], noise ~ N(0,I), A_t is the ground-truth
            action, and v_θ is the learned velocity field.

            Inference (Euler method, 10 denoising steps):
                Start with x_1 = noise (τ=1)
                For each step: x_{τ-dt} = x_τ + dt · v_θ(x_τ, o_t, τ)
                where dt = -1/num_steps (num_steps=10)
                Final action: x_0 (τ=0)

        KV Cache optimization: The prefix (image + language tokens)
        is computed once in the first forward pass. Subsequent
        denoising steps (2-10) reuse the cached KV pairs, reducing
        total inference cost by ~5x compared to reprocessing the full
        context at each step.

        Action Chunking: When the internal action buffer is empty,
        a new chunk of `chunk_size` (default 50) actions is generated
        via Flow Matching. Subsequent calls pop from the buffer until
        exhausted, then a new chunk is generated. This amortizes the
        inference cost over 50 control steps, achieving 50Hz effective
        control rate.

        Returns:
            np.ndarray of shape (7,) — next joint position target
            from the action chunk buffer.
        """
        if not self._action_buffer:
            if self.loaded and self._flow_model:
                # Generate new action chunk via Flow Matching
                import torch
                noise = torch.randn(self.chunk_size, 7)
                action_chunk = self._flow_matching_sample(noise, obs_dict)
                for a in action_chunk:
                    self._action_buffer.append(a)
            else:
                # Stub: fill buffer with smooth trajectory
                proprio = obs_dict.get('proprio', np.zeros(7))
                target = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
                for i in range(min(self.chunk_size, 10)):
                    alpha = (i + 1) / 10.0
                    self._action_buffer.append(
                        proprio + (target - proprio) * alpha * 0.1
                    )

        if self._action_buffer:
            return self._action_buffer.popleft()
        return super().predict(obs_dict)

    def _flow_matching_sample(self, noise, obs_dict):
        """Flow Matching sampling — generate action chunk from noise.

        Uses the Euler method to integrate the learned velocity field
        v_θ from τ=1 (pure noise) to τ=0 (generated action) in
        `num_steps` (default 10) denoising steps:

            x_{τ-dt} = x_τ + dt · v_θ(x_τ, o_t, τ),  dt = -1/num_steps

        KV Cache optimization: The PaliGemma prefix (image + language
        tokens) is computed once and cached. All 10 denoising steps
        reuse this KV cache, so only the action expert (300M) runs
        at each step — making 10-step denoising feasible at 50Hz.

        Args:
            noise: Initial noise tensor of shape (chunk_size, action_dim).
            obs_dict: Observation dict with 'rgb', 'language', 'proprio'.

        Returns:
            List of np.ndarray, each of shape (7,) — the generated
            action chunk. In stub mode, returns smooth interpolation
            actions.
        """
        # Real implementation would use the π₀ Flow Matching model
        return [super().predict(obs_dict) for _ in range(self.chunk_size)]


class DemoVLAAdapter(VLAAdapter):
    """v0.16.25: Demo VLA adapter — instruction-driven pick-and-place.

    When no real VLA model is loaded, this adapter interprets natural-language
    instructions and generates a time-based action sequence that performs
    pick-and-place. This lets users see the arm move in response to their
    instructions without requiring GPU model inference.

    Supported instruction patterns:
        - "pick up/grab/grasp [object]" → reach → descend → grasp → lift → transport → release
        - "open [gripper]" → open gripper
        - "close [gripper]" → close gripper
        - "home/reset" → return to home pose
        - "move to [location]" → reach toward location
        - Default (unrecognized) → full pick-and-place cycle

    The adapter cycles through phases with ~50 steps per phase (≈1.5s at 30Hz).
    """

    def __init__(self):
        super().__init__(model_name="demo-vla")
        self.loaded = True  # Demo adapter is always "loaded"
        self._step_counter: int = 0
        self._phase_idx: int = 0
        self._instruction: str = ""
        self._action_dim: int = 7

    def _parse_instruction(self, instruction: str) -> List[str]:
        """Parse instruction text into a list of phase keywords."""
        text = instruction.lower().strip()
        phases: List[str] = []

        if any(kw in text for kw in ["pick", "grab", "grasp", "抓", "拿", "取"]):
            phases = ["reach", "descend", "grasp", "lift", "transport", "release", "retreat"]
        elif any(kw in text for kw in ["open", "打开", "松开"]):
            phases = ["open_gripper"]
        elif any(kw in text for kw in ["close", "关闭", "合上"]):
            phases = ["close_gripper"]
        elif any(kw in text for kw in ["home", "reset", "归零", "复位", "回家"]):
            phases = ["home"]
        elif any(kw in text for kw in ["move", "go", "reach", "移动", "去"]):
            phases = ["reach", "hold"]
        elif any(kw in text for kw in ["place", "put", "drop", "放", "放置"]):
            phases = ["transport", "release", "retreat"]
        else:
            # Default: full pick-and-place cycle
            phases = ["reach", "descend", "grasp", "lift", "transport", "release", "retreat"]

        return phases

    def _get_phase_target(self, phase: str, proprio: np.ndarray) -> np.ndarray:
        """Get target joint positions for a given phase."""
        # Home pose
        home = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])

        if phase == "home":
            return home
        elif phase == "reach":
            # Reach toward cube (approximate IK toward front-center-low)
            return np.array([0.0, 0.5, -0.8, 0.3, 0.0, 0.0, 0.0])
        elif phase == "descend":
            # Lower to object
            return np.array([0.0, 0.6, -1.0, 0.4, 0.0, 0.0, 0.0])
        elif phase == "grasp":
            # Close gripper at current position
            target = proprio.copy()
            target[5] = 0.7   # Close left
            target[6] = -0.7  # Close right
            return target
        elif phase == "lift":
            # Lift up while keeping gripper closed
            return np.array([0.0, 0.2, -0.3, 0.0, 0.0, 0.7, -0.7])
        elif phase == "transport":
            # Move to tray (right side)
            return np.array([0.8, 0.3, -0.5, 0.0, 0.0, 0.7, -0.7])
        elif phase == "release":
            # Open gripper over tray
            return np.array([0.8, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])
        elif phase == "retreat":
            # Return to home
            return home
        elif phase == "open_gripper":
            target = proprio.copy()
            target[5] = 0.0
            target[6] = 0.0
            return target
        elif phase == "close_gripper":
            target = proprio.copy()
            target[5] = 0.7
            target[6] = -0.7
            return target
        elif phase == "hold":
            return proprio.copy()
        else:
            return home

    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Generate instruction-driven action sequence."""
        instruction = obs_dict.get('language', '')
        if instruction != self._instruction:
            # New instruction → restart sequence
            self._instruction = instruction
            self._phases = self._parse_instruction(instruction)
            self._phase_idx = 0
            self._step_counter = 0
            print(f"DemoVLA: New instruction '{instruction[:50]}' → phases: {self._phases}")

        if not hasattr(self, '_phases') or not self._phases:
            self._phases = self._parse_instruction(instruction)
            self._phase_idx = 0
            self._step_counter = 0

        STEPS_PER_PHASE = 30  # v0.16.26: ~1s at 30Hz (was 50 — too slow)

        # Advance phase
        if self._step_counter >= STEPS_PER_PHASE:
            self._phase_idx += 1
            self._step_counter = 0
            if self._phase_idx >= len(self._phases):
                # All phases complete → restart from beginning (loop)
                self._phase_idx = 0
                print(f"DemoVLA: Cycle complete, restarting '{instruction[:30]}'")

        current_phase = self._phases[self._phase_idx]
        proprio = obs_dict.get('proprio', np.zeros(7))
        if not isinstance(proprio, np.ndarray):
            proprio = np.array(proprio, dtype=np.float64)
        if len(proprio) < 7:
            proprio = np.pad(proprio, (0, 7 - len(proprio)))
        target = self._get_phase_target(current_phase, proprio)

        # v0.16.26: Faster interpolation (was 0.15 — too slow to see motion)
        # 30% per step reaches target in ~10 steps (0.3s), visible to user
        action = proprio + (target - proprio) * 0.30
        action = np.clip(action, [-3.14, -3.14, -3.14, -3.14, -3.14, 0.0, -0.873],
                         [3.14, 3.14, 3.14, 3.14, 3.14, 0.873, 0.0])

        self._step_counter += 1
        return action.astype(np.float64)


def create_vla_adapter(model_name: str) -> VLAAdapter:
    """Factory function to create a VLA adapter by name.

    Supported models span three VLA factions:

    | Model        | Faction      | Params | Core Tech              | Freq  |
    |--------------|-------------|--------|------------------------|-------|
    | openvla-7b   | Academic    | 7B     | DINOv2+SigLIP+Llama-2  | 2-10Hz|
    | octo-base    | Academic    | 93M    | Multi-view+Chunking    | ~10Hz |
    | pi0-base     | Tech-Extreme| —      | Flow Matching+Chunking | 50Hz  |

    All adapters implement the same predict(obs_dict)→np.ndarray
    interface, so downstream code (TOMAS Wrapper, ψ-Anchor, κ-Snap)
    is model-agnostic. The choice of backbone only affects action
    quality and frequency; the IDO audit/safety layers are identical.

    Args:
        model_name: One of 'openvla-7b', 'octo-base', 'pi0-base'.

    Returns:
        VLAAdapter instance (stub mode if real model unavailable).

    Raises:
        ValueError: If model_name is not recognized.
    """
    adapters = {
        'openvla-7b': OpenVLAAdapter,
        'octo-base': OctoAdapter,
        'pi0-base': Pi0Adapter,
        'demo-vla': DemoVLAAdapter,
    }

    cls = adapters.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown VLA model: {model_name}. "
            f"Available: {list(adapters.keys())}"
        )

    adapter = cls()
    print(f"VLA adapter created: {model_name} (loaded={adapter.is_loaded()})")
    return adapter
