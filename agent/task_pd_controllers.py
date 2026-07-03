"""
Task-PD Controllers — Per-task PD Controllers for dm_control Benchmarks
========================================================================

Replaces MotorPrimitives' 5 fixed macros with goal-directed PD controllers
that extract real targets from dm_control observation spaces and compute
per-task control actions with individually tuned KP/KD gains.

Core root cause fix: the old MotorPrimitives only covered ctrl[:2] for
humanoid (21 actuators), used squat for all tasks, and fell back to
uniform noise.  This made the IDO agent ≈ Random Agent (episode_return ≈ 0,
η not descending, NVR 981/ep).

Architecture:
  - TaskPDController (ABC): base class with compute_action / compute_safe_action
  - 14+ task-specific PD controllers with detailed implementations:
      HumanoidStandPD, HumanoidWalkPD, HumanoidRunPD, ReacherTargetPD,
      WalkerStandPD, WalkerWalkPD, WalkerRunPD, CheetahRunPD,
      CartpoleBalancePD, CartpoleSwingupPD, HopperStandPD, HopperHopPD,
      FishSwimPD, SwimmerSwimPD, FingerSpinPD, FingerTurnEasyPD,
      FingerTurnHardPD, BallInCupCatchPD, AcrobotSwingupPD,
      PendulumSwingupPD, ManipulatorBringBallPD
  - GenericPDController: fallback for remaining tasks
  - TASK_CONTROLLER_MAP: maps all 25 task names → controller class

Author: tomas-arc3-solver project · MuJoCo-Bench-IDO v0.5.0 Phase 1
"""
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


# ── Abstract Base ────────────────────────────────────────────────────


class TaskPDController(ABC):
    """Base class for per-task PD controllers.

    Each controller extracts the task's real target from the dm_control
    timestep observation dict, computes PD-governed control actions, and
    provides a safe-action fallback for Noether violations.

    Attributes:
        phy: dm_control Physics instance (for model dimensions).
        kp: Proportional gain (per-task default, can be overridden).
        kd: Derivative gain (per-task default, can be overridden).
        nu: Number of actuators (physics.model.nu).
    """

    def __init__(self, physics, kp: float = 30.0, kd: float = 3.0) -> None:
        """Initialize the PD controller with physics model and gains.

        Args:
            physics: dm_control Physics instance.
            kp: Proportional gain.
            kd: Derivative gain.
        """
        self.phy = physics
        self.kp: float = kp
        self.kd: float = kd
        self.nu: int = physics.model.nu
        self._step_counter: int = 0

    @abstractmethod
    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute goal-directed control action for current timestep.

        Args:
            timestep: dm_control TimeStep (with .observation dict).
            physics: dm_control Physics instance.

        Returns:
            Control array of shape (nu,) clipped to [-1, 1].
        """
        pass

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Compute a conservative safe action for Noether violation fallback.

        Default implementation: zero ctrl (full brake / stabilize).
        Subclasses can override with task-specific safe actions (e.g.,
        small damping torques to slow down without injecting energy).

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe control array of shape (nu,) clipped to [-1, 1].
        """
        return np.zeros(self.nu)

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Clip control array to [-1, 1] with correct dimension.

        Args:
            ctrl: Raw control array (may be shorter or longer than nu).

        Returns:
            Clipped array of shape (nu,).
        """
        result: np.ndarray = np.zeros(self.nu)
        n: int = min(len(ctrl), self.nu)
        result[:n] = np.clip(ctrl[:n], -1.0, 1.0)
        return result

    def _pd_scalar(self, error: float, error_dot: float,
                   kp: float, kd: float) -> float:
        """Compute scalar PD output: kp * error + kd * error_dot.

        Args:
            error: Proportional error.
            error_dot: Derivative error (rate of change).
            kp: Proportional gain.
            kd: Derivative gain.

        Returns:
            Scalar PD output value.
        """
        return kp * error + kd * error_dot

    def _increment_step(self) -> int:
        """Increment and return internal step counter for gait timing.

        Returns:
            Current step counter value.
        """
        self._step_counter += 1
        return self._step_counter


# ── Core Controllers (8+ detailed implementations) ───────────────────


class HumanoidStandPD(TaskPDController):
    """PD controller for humanoid-stand: maintain standing height + upright.

    dm_control reward formula (critical alignment):
        standing = tolerance(head_height, bounds=(1.4, inf), margin=0.35)
        upright = tolerance(torso_upright, bounds=(0.9, inf), sigmoid='linear',
                           margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        small_control = tolerance(ctrl, margin=1, value_at_margin=0,
                                  sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5
        dont_move = tolerance(horizontal_velocity, margin=2).mean()
        reward = small_control * stand_reward * dont_move

    KEY: reward PENALIZES large control signals (small_control factor).
    - ctrl close to 0 → small_control ≈ 1.0 (good!)
    - ctrl > 1.0 → small_control ≈ 0 (bad!)
    So humanoid-stand should use VERY SMALL control (0.01-0.05 range).

    Strategy (v0.5.2 — dm_control reward alignment):
    1. Use VERY small control actions (0.01-0.05 range) to maximize
       small_control factor.
    2. Joint-level PD toward standing pose with ultra-low gains.
    3. Velocity damping to minimize horizontal movement (dont_move).
    4. Clip all actions to [-0.08, 0.08] (5-10x reduction from v0.5.0).

    PD gains (v0.5.2 — dm_control reward aligned):
        - Height: kp=0.8, kd=0.15 (ultra-low for small control)
        - Upright: kp=0.5, kd=0.1 (ultra-low for small control)
        - Joint damping: kp=0.05, kd=0.02 (near-zero for small control)

    Output: ctrl[0:21], all clipped to [-0.08, 0.08].
    """

    def __init__(self, physics) -> None:
        """Initialize HumanoidStandPD with ultra-low dm_control-aligned gains.

        v0.5.2: All gains reduced 5-10x from v0.5.0 to maximize small_control
        factor in dm_control reward. Clips reduced from [-0.3, 0.3] to
        [-0.08, 0.08].

        v0.5.5: REVERTED from two-phase recovery→stabilize (caused regression:
        avg_return 8.47→3.87). Two-phase with ctrl_clip=0.3 in Phase 1 makes
        upright WORSE (0.069 vs 0.223) and small_control WORSE (0.985 vs 0.995)
        without improving stand_reward (0.014 stays ≈ same). Single-phase
        ctrl_clip=0.08 remains the best approach.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=2.0, kd=0.4)
        self.target_height: float = 1.4
        self.head_target_height: float = 1.4
        self.height_kp: float = 2.0
        self.height_kd: float = 0.4
        self.upright_kp: float = 1.5
        self.upright_kd: float = 0.3
        self.joint_kp: float = 0.3
        self.joint_kd: float = 0.08
        self.ctrl_clip: float = 0.08  # max absolute control value

        # Standing pose targets for joint-level PD
        # dm_control humanoid actuator layout (21 actuators):
        #   0: abdomen_y (lateral bending)
        #   1: abdomen_z (vertical core twist)
        #   2: abdomen_x (forward/backward lean)
        #   3-5: right_hip_x/z/y, right_knee
        #   6-8: left_hip_x/z/y, left_knee
        #   9-10: right_ankle_x/y
        #   11-12: left_ankle_x/y
        #   13-15: right_shoulder_x/y/z
        #   16-18: left_shoulder_x/y/z
        #   19: right_elbow
        #   20: left_elbow
        self.standing_targets: np.ndarray = np.array([
            0.0,    # abdomen_y
            0.0,    # abdomen_z
            0.0,    # abdomen_x
            0.0,    # right_hip_x
            0.0,    # right_hip_z
            0.0,    # right_hip_y
            0.05,   # right_knee (slight bend)
            0.0,    # left_hip_x
            0.0,    # left_hip_z
            0.0,    # left_hip_y
            0.05,   # left_knee (slight bend)
            0.0,    # right_ankle_x
            0.0,    # right_ankle_y
            0.0,    # left_ankle_x
            0.0,    # left_ankle_y
            0.0,    # right_shoulder_x
            0.0,    # right_shoulder_y
            0.0,    # right_shoulder_z
            0.0,    # left_shoulder_x
            0.0,    # left_shoulder_y
            0.0,    # left_shoulder_z
            0.0,    # right_elbow
            0.0,    # left_elbow
        ][:21])

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute standing PD control with ultra-small actions for dm_control reward.

        v0.5.2 Strategy (dm_control reward alignment):
        1. Joint-level PD toward standing pose with ultra-low gains.
           action = kp*(standing_target - joint_angle) - kd*joint_vel
        2. Height contribution: if head height < 1.4m, push legs up gently.
        3. Upright contribution: if torso tilted, push core gently.
        4. Horizontal velocity damping for dont_move factor.
        5. ALL actions clipped to [-0.08, 0.08] for small_control factor.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:21] clipped to [-0.08, 0.08].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # ── 1. Get current state ──
        # Head height from dm_control: phys.named.data.xpos['head', 'z']
        head_z: float = 0.0
        try:
            head_pos = phys.named.data.xpos['head', :]
            head_z = float(head_pos[2]) if hasattr(head_pos, '__len__') else float(head_pos)
        except (KeyError, IndexError, TypeError):
            # Fallback: root z + approximate
            head_z = float(phys.data.qpos[2]) if len(phys.data.qpos) > 2 else 0.0

        # Torso upright from dm_control: xmat['torso','zz']
        # The zz element of the rotation matrix: 1.0 = perfectly upright
        torso_upright: float = 1.0
        try:
            torso_mat = phys.named.data.xmat['torso', :]
            torso_upright = float(torso_mat[8]) if hasattr(torso_mat, '__len__') else float(torso_mat)
        except (KeyError, IndexError, TypeError):
            # Fallback from quaternion
            if len(phys.data.qpos) >= 7:
                qw: float = float(phys.data.qpos[3])
                qx: float = float(phys.data.qpos[4])
                torso_upright = 1.0 - 2.0 * qx * qx

        # ── 2. Height PD (ultra-low gains) ──
        height_error: float = self.head_target_height - head_z
        height_torque: float = self.height_kp * height_error

        # ── 3. Upright PD (ultra-low gains) ──
        upright_error: float = max(0.0, 0.9 - torso_upright)
        upright_torque: float = self.upright_kp * upright_error

        # ── 4. Horizontal velocity damping (for dont_move factor) ──
        # dm_control: horizontal_velocity = torso_subtreelinvel x-component
        horiz_vel: float = 0.0
        try:
            torso_vel = phys.named.data.sensordata['torso_subtreelinvel']
            horiz_vel = float(torso_vel[0]) if hasattr(torso_vel, '__len__') else float(torso_vel)
        except (KeyError, IndexError, TypeError):
            # Fallback: qvel root x velocity
            horiz_vel = float(phys.data.qvel[0]) if len(phys.data.qvel) > 0 else 0.0

        # ── 5. Joint-level PD toward standing pose ──
        # dm_control humanoid: 7 root qpos entries (position+quaternion),
        # then 21 joint angles starting at qpos[7]
        for i in range(min(self.nu, 21)):
            # Get joint angle from qpos (shifted by 7 root DOFs)
            qpos_idx: int = i + 7
            joint_angle: float = 0.0
            if qpos_idx < len(phys.data.qpos):
                joint_angle = float(phys.data.qpos[qpos_idx])

            # Get joint velocity from qvel (shifted by 6 root velocity DOFs)
            qvel_idx: int = i + 6
            joint_vel: float = 0.0
            if qvel_idx < len(phys.data.qvel):
                joint_vel = float(phys.data.qvel[qvel_idx])

            # PD toward standing pose: kp*(target - angle) - kd*vel
            target_angle: float = float(self.standing_targets[min(i, len(self.standing_targets) - 1)])
            joint_pd: float = (self.joint_kp * (target_angle - joint_angle)
                               - self.joint_kd * joint_vel)

            # Add height contribution for leg joints (3-12)
            height_contribution: float = 0.0
            if 3 <= i <= 12:  # leg joints
                height_contribution = height_torque * 0.05

            # Add upright contribution for core joints (0-2) and leg joints
            upright_contribution: float = 0.0
            if 0 <= i <= 2:  # core (abdomen)
                upright_contribution = upright_torque * 0.06
            elif 3 <= i <= 12:  # legs
                upright_contribution = upright_torque * 0.03

            # Add horizontal velocity damping (for dont_move)
            vel_damping: float = 0.0
            if 0 <= i <= 2:  # core — resist horizontal movement
                vel_damping = -0.03 * horiz_vel
            elif 3 <= i <= 12:  # legs — resist horizontal movement
                vel_damping = -0.01 * horiz_vel

            ctrl[i] = np.clip(
                joint_pd + height_contribution + upright_contribution + vel_damping,
                -self.ctrl_clip, self.ctrl_clip)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: ultra-small damping for Noether violation fallback.

        v0.5.2: All actions in [-0.05, 0.05] to maintain small_control
        factor even during Noether recovery.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Ultra-small damping ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        safe_clip: float = 0.05
        for i in range(self.nu):
            qvel_idx: int = i + 6
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.03 * vel, -safe_clip, safe_clip)
        return self._clip_ctrl(ctrl)


class HumanoidWalkPD(TaskPDController):
    """PD controller for humanoid-walk: walk forward + maintain upright.

    Targets:
        - Forward COM velocity ≈ target_speed (~1.0 m/s for walk)
        - Torso upright
        - Height ≈ 1.4 m

    PD gains (v0.5.0 tuned):
        - Forward velocity: kp=20, kd=2
        - Upright: kp=20, kd=3 (lowered from 30)
        - Height: kp=30, kd=5 (lowered from 50)
        - Gait oscillation: sinusoidal phase for legs

    Output: ctrl[0:21].
    """

    def __init__(self, physics) -> None:
        """Initialize HumanoidWalkPD with tuned gains.

        v0.5.0: height_kp lowered 50→30, upright_kp lowered 30→20.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=20.0, kd=2.0)
        self.target_speed: float = 1.0
        self.target_height: float = 1.4
        self.vel_kp: float = 20.0
        self.vel_kd: float = 2.0
        self.height_kp: float = 30.0
        self.height_kd: float = 5.0
        self.upright_kp: float = 20.0
        self.upright_kd: float = 3.0
        self.gait_freq: float = 2.0  # Hz for walking gait

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute walking PD control with gait oscillation.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:21] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── Forward Velocity PD ──
        # COM forward velocity: qvel[0] (root x velocity in dm_control humanoid)
        com_vx: float = float(phys.data.qvel[0])
        vel_error: float = self.target_speed - com_vx
        vel_error_dot: float = 0.0  # approximate as zero (or use velocity change)
        forward_torque: float = self._pd_scalar(
            vel_error, vel_error_dot, self.vel_kp, self.vel_kd)

        # ── Height PD ──
        root_z: float = float(phys.data.qpos[2])
        height_error: float = self.target_height - root_z
        root_vz: float = float(phys.data.qvel[2])
        height_error_dot: float = -root_vz
        height_torque: float = self._pd_scalar(
            height_error, height_error_dot, self.height_kp, self.height_kd)

        # ── Upright PD ──
        if len(phys.data.qpos) >= 7:
            qw: float = float(phys.data.qpos[3])
            qx: float = float(phys.data.qpos[4])
            upright_error: float = -qx
            upright_vel: float = float(phys.data.qvel[3])
            upright_torque: float = self._pd_scalar(
                upright_error, -upright_vel,
                self.upright_kp, self.upright_kd)

        # ── Gait Oscillation ──
        # Sinusoidal phase for leg stepping
        phase: float = math.sin(self.gait_freq * step * 0.01)

        if self.nu >= 21:
            # Core stabilization
            ctrl[0] = np.clip(height_torque * 0.02 + upright_torque * 0.02, -1.0, 1.0)
            ctrl[1] = np.clip(height_torque * 0.01 + upright_torque * 0.01, -1.0, 1.0)

            # Right leg: forward push + gait phase
            for i in range(2, 6):
                ctrl[i] = np.clip(
                    forward_torque * 0.03 + phase * 0.2, -1.0, 1.0)

            # Left leg: forward push + opposite gait phase
            for i in range(6, 10):
                ctrl[i] = np.clip(
                    forward_torque * 0.03 - phase * 0.2, -1.0, 1.0)

            # Upper body: damp oscillation
            for i in range(10, 21):
                joint_vel: float = float(phys.data.qvel[min(
                    i + 3, len(phys.data.qvel) - 1)])
                ctrl[i] = np.clip(-joint_vel * 0.1, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping to stop forward motion + hold upright.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(self.nu):
            joint_vel: float = float(physics.data.qvel[min(
                i + 3, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(-0.3 * joint_vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class HumanoidRunPD(TaskPDController):
    """PD controller for humanoid-run: faster forward speed + upright.

    Same structure as HumanoidWalkPD but with higher target_speed
    and stronger gains.

    PD gains (v0.5.0 tuned):
        - Forward velocity: kp=30, kd=3 (stronger push)
        - Upright: kp=25, kd=3 (lowered from 35)
        - Height: kp=35, kd=4 (lowered from 55)
        - Gait freq: 3.0 Hz (faster cadence)

    Output: ctrl[0:21].
    """

    def __init__(self, physics) -> None:
        """Initialize HumanoidRunPD with tuned gains.

        v0.5.0: height_kp lowered 55→35, upright_kp lowered 35→25.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=30.0, kd=3.0)
        self.target_speed: float = 2.5
        self.target_height: float = 1.4
        self.vel_kp: float = 30.0
        self.vel_kd: float = 3.0
        self.height_kp: float = 35.0
        self.height_kd: float = 4.0
        self.upright_kp: float = 25.0
        self.upright_kd: float = 3.0
        self.gait_freq: float = 3.0

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute running PD control with faster gait.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:21] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        com_vx: float = float(phys.data.qvel[0])
        vel_error: float = self.target_speed - com_vx
        forward_torque: float = self._pd_scalar(
            vel_error, 0.0, self.vel_kp, self.vel_kd)

        root_z: float = float(phys.data.qpos[2])
        height_error: float = self.target_height - root_z
        root_vz: float = float(phys.data.qvel[2])
        height_torque: float = self._pd_scalar(
            height_error, -root_vz, self.height_kp, self.height_kd)

        upright_torque: float = 0.0
        if len(phys.data.qpos) >= 7:
            qx: float = float(phys.data.qpos[4])
            upright_vel: float = float(phys.data.qvel[3])
            upright_torque = self._pd_scalar(
                -qx, -upright_vel, self.upright_kp, self.upright_kd)

        phase: float = math.sin(self.gait_freq * step * 0.01)

        if self.nu >= 21:
            ctrl[0] = np.clip(height_torque * 0.02 + upright_torque * 0.02, -1.0, 1.0)
            ctrl[1] = np.clip(height_torque * 0.01 + upright_torque * 0.01, -1.0, 1.0)
            for i in range(2, 6):
                ctrl[i] = np.clip(
                    forward_torque * 0.04 + phase * 0.3, -1.0, 1.0)
            for i in range(6, 10):
                ctrl[i] = np.clip(
                    forward_torque * 0.04 - phase * 0.3, -1.0, 1.0)
            for i in range(10, 21):
                joint_vel: float = float(phys.data.qvel[min(
                    i + 3, len(phys.data.qvel) - 1)])
                ctrl[i] = np.clip(-joint_vel * 0.12, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping to decelerate + stabilize.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(self.nu):
            joint_vel: float = float(physics.data.qvel[min(
                i + 3, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(-0.4 * joint_vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class ReacherTargetPD(TaskPDController):
    """PD controller for reacher-easy/hard: reach end-effector to target.

    Targets:
        - End-effector reaches target position (from timestep observation)

    Observations used:
        - timestep.observation['position'] (2 joint angles)
        - timestep.observation['to_target'] (vector from EE to target)

    PD gains:
        - kp=5.0, kd=0.5 (gentle, reacher is sensitive)

    Output: ctrl[0:2], reacher has 2 actuators.
    """

    def __init__(self, physics) -> None:
        """Initialize ReacherTargetPD with tuned gains.

        v0.5.0: kp increased from 5.0 to 8.0 for stronger convergence.
        kd increased from 0.5 to 1.0 for better damping.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=8.0, kd=1.0)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute reaching PD control using polar coordinate angle mapping.

        v0.5.0: Replaced naive direct mapping (to_target[i] → ctrl[i]) with
        a proper kinematics-aware approach using polar coordinates:

        1. Convert to_target (2D Cartesian) to polar: distance + target angle.
        2. Current end-effector angle ≈ position[0] + position[1]
           (shoulder + elbow accumulated angles).
        3. Compute angle_error = target_angle - current_angle.
        4. PD drives both joints to reduce angle_error.
        5. Scale output by distance to slow down near target (avoid oscillation).

        This fixes episode_return=0 root cause: the old code mapped a 2D
        Cartesian direction vector directly to joint torques without any
        kinematic transformation, producing meaningless actions.

        Args:
            timestep: dm_control TimeStep (must have .observation dict).
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:2] clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # ── Extract observation components ──
        to_target: np.ndarray = obs.get('to_target', np.zeros(2))
        if to_target is None:
            to_target = np.zeros(2)

        # Current joint angles and velocities
        position: np.ndarray = obs.get('position', np.zeros(2))
        if position is None:
            position = np.zeros(2)
        velocity: np.ndarray = obs.get('velocity', np.zeros(2))
        if velocity is None:
            velocity = np.zeros(2)

        # ── Polar coordinate target angle ──
        dist: float = float(np.linalg.norm(to_target))
        target_angle: float = float(np.arctan2(to_target[1], to_target[0]))

        # Current end-effector angle ≈ shoulder + elbow (accumulated)
        current_angle: float = (float(position[0] + position[1])
                                if len(position) >= 2 else 0.0)

        # Angle error: how much we need to rotate to face the target
        angle_error: float = target_angle - current_angle
        # Normalize to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # ── PD on angle error for both joints ──
        # Shoulder gets full angle error; elbow gets half (finer adjustment)
        for i in range(min(self.nu, 2)):
            # Scale by dist so we slow down when close (avoid oscillation)
            scale: float = min(dist * 5.0, 1.0)  # ramp up with distance, cap at 1.0
            err: float = angle_error * scale * (1.0 if i == 0 else 0.5)
            err_dot: float = -float(velocity[i])
            ctrl[i] = self._pd_scalar(err, err_dot, self.kp, self.kd)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle damping on both joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Small damping ctrl array of shape (2,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 2)):
            vel: float = float(physics.data.qvel[i])
            ctrl[i] = np.clip(-0.5 * vel, -0.3, 0.3)
        return self._clip_ctrl(ctrl)


class WalkerStandPD(TaskPDController):
    """PD controller for walker-stand: remain upright and stable.

    Targets:
        - Root height z ≈ 1.0 m (walker standing height)
        - Torso upright
        - Joint stabilization

    PD gains (v0.5.0 tuned):
        - Height: kp=35, kd=3.5 (lowered from 40)
        - Upright: kp=20, kd=2.0 (lowered from 25)
        - Joint stabilize: kp=2, kd=1 (lowered from 8)

    Output: ctrl[0:6], walker has 6 actuators.
    """

    def __init__(self, physics) -> None:
        """Initialize WalkerStandPD with tuned gains.

        v0.5.0: height_kp lowered 40→35, upright_kp lowered 25→20,
        joint_kp lowered 8→2 for collision avoidance.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=35.0, kd=3.5)
        self.target_height: float = 1.0
        self.height_kp: float = 35.0
        self.height_kd: float = 3.5
        self.upright_kp: float = 20.0
        self.upright_kd: float = 2.0
        self.joint_kp: float = 2.0
        self.joint_kd: float = 1.0

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute standing stabilization PD control for walker.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:6] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # ── Height PD (v0.6.3: use xpos for walker, not raw qpos) ──
        # Walker model: nq=9, root qpos = [rootz, rootx, rooty] (3 entries, NOT 7)
        # Raw qpos[2] = rooty (pitch angle, NOT height!)
        # Use xpos['torso','z'] for actual height
        root_z: float = float(phys.named.data.xpos['torso', :][2])
        height_error: float = self.target_height - root_z
        root_vz: float = float(phys.data.qvel[0])  # rootz velocity (NOT qvel[2])
        height_torque: float = self._pd_scalar(
            height_error, -root_vz, self.height_kp, self.height_kd)

        # ── Upright PD (v0.6.3: use xmat for walker, not raw qpos) ──
        # Walker has NO quaternion in qpos. Use xmat['torso','zz'] for upright.
        torso_upright: float = float(phys.named.data.xmat['torso', :][8])
        upright_error: float = 1.0 - torso_upright  # want upright >= 1.0
        root_pitch_vel: float = float(phys.data.qvel[2])  # rooty angular velocity
        upright_torque: float = self._pd_scalar(
            upright_error, -root_pitch_vel, self.upright_kp, self.upright_kd)

        # ── Joint stabilization (v0.6.3: correct qvel indices for walker) ──
        # Walker: nv=9, root qvel = [rootz_vel, rootx_vel, rooty_vel] (3 entries)
        # Joint qvel starts at qvel[3], NOT qvel[5]
        for i in range(min(self.nu, 6)):
            qvel_idx: int = i + 3  # v0.6.3: walker has 3 root velocity DOFs
            if qvel_idx < len(phys.data.qvel):
                joint_vel: float = float(phys.data.qvel[qvel_idx])
                joint_ctrl: float = -self.joint_kp * joint_vel * 0.1
            else:
                joint_ctrl: float = 0.0
            # Mix height + upright + joint stabilize
            ctrl[i] = np.clip(
                height_torque * 0.02 + upright_torque * 0.01 + joint_ctrl,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping on walker joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (6,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 6)):
            qvel_idx: int = i + 3  # v0.6.3: walker has 3 root velocity DOFs
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.3 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class WalkerWalkPD(TaskPDController):
    """PD controller for walker-walk: 3-phase recovery → stabilize → walking gait.

    dm_control walker-walk reward formula (critical alignment):
        standing = tolerance(torso_height(), bounds=(1.2, inf), margin=0.6)
        upright = (1 + torso_upright()) / 2  # torso_upright = xmat['torso','zz']
        stand_reward = (3*standing + upright) / 4
        move_reward = tolerance(horizontal_velocity(), bounds=(1.0, inf),
                                margin=0.5, value_at_margin=0.5, sigmoid='linear')
        reward = stand_reward * (5*move_reward + 1) / 6

    Maximum reward per step = 1.0 (when standing=1, upright=1, move_reward=1).
    Needs: torso_height ≥ 1.2m, torso_upright ≥ 1.0, horizontal_velocity ≥ 1.0 m/s.

    Walker actuator layout (CONFIRMED — TORQUE actuators, NOT position):
        ctrl[0]: right_hip torque (gain=1, ctrlrange [-1,1], jnt_range [-0.35, 1.75])
        ctrl[1]: right_knee torque (gain=1, ctrlrange [-1,1], jnt_range [-2.62, 0])
        ctrl[2]: right_ankle torque (gain=1, ctrlrange [-1,1], jnt_range [-0.79, 0.79])
        ctrl[3]: left_hip torque (same as right_hip)
        ctrl[4]: left_knee torque (same as right_knee)
        ctrl[5]: left_ankle torque (same as right_ankle)
        ALL ctrl values ARE torques in [-1, 1] range.

    Walker model structure (CONFIRMED from diagnostic):
        nq=9, nv=9, nu=6
        Root: rootz(slide), rootx(slide), rooty(hinge) → qpos[0:3], qvel[0:3]
        Joints: right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle → qpos[3:9]
        NO actuator for root_y — torso orientation controlled INDIRECTLY via leg joints
        torso_upright = xmat['torso','zz'] — can be NEGATIVE (upside down) at start

    Strategy (v0.6.3 — torque-actuator + one-sided height PD):
    Phase 1 (Recovery): height < 1.0 or upright < 0.5
        - Positive torque on hip/ankle to push walker up and forward
        - One-sided height PD: only pushes UP (never pushes down)
        - Upright PD: pushes toward upright orientation
    Phase 2 (Stabilize): upright for 30 steps
        - Joint PD toward standing pose + height PD + upright PD
        - One-sided height PD maintains height ≥ 1.2
    Phase 3 (Walking gait): stabilized for 30+ steps
        - Full-sine gait + height PD + upright PD
        - If height < 0.8 or upright < 0.4 → reset to Phase 1

    Output: ctrl[0:6], ALL clipped to [-1, 1] (torque range).
    """

    def __init__(self, physics) -> None:
        """Initialize WalkerWalkPD with torque-actuator-aware gains.

        v0.6.3 CRITICAL FIX: Walker actuators are TORQUE (gain=1, bias=0),
        ctrl range [-1, 1]. Previous versions incorrectly used per-joint angle
        ranges as clip bounds. Now all ctrl clipped to [-1, 1].

        Also: height PD is one-sided (only push UP when below target).
        When walker is above 1.2m, height PD doesn't push it down.

        Standing pose uses moderate targets instead of raw reset angles,
        since reset angles are randomized and not ideal for standing.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=5.0, kd=1.5)
        self.target_speed: float = 1.0
        self.target_height: float = 1.2
        # ── Phase 1: Recovery — fixed torque push + height/upright PD ──
        # Don't use angle targets — they destabilize because initial state varies randomly.
        # Instead: push walker up with positive hip/ankle torque, let knee support weight.
        # These "default recovery torques" are based on ctrl→joint diagnostic:
        #   ctrl=0.5 on all joints → upright=0.98, so positive hip/ankle = upright.
        self.recovery_hip_torque: float = 0.5   # push hip forward → lift torso
        self.recovery_knee_torque: float = -0.3  # bend knee → support weight
        self.recovery_ankle_torque: float = 0.2  # push ankle forward → balance
        self.recovery_vel_kd: float = 1.0        # velocity damping during recovery
        # ── Phase 2: Stabilize gains (WalkerStandPD-style) ──
        self.stabilize_kp: float = 2.0   # joint damping gain (NOT target-tracking)
        self.stabilize_kd: float = 0.0    # unused for damping-only mode
        # ── Height PD (one-sided: only push UP, but always damp velocity) ──
        self.height_kp: float = 35.0     # v0.6.3: aligned with WalkerStandPD kp=35
        self.height_kd: float = 3.5      # v0.6.3: aligned with WalkerStandPD kd=3.5
        # ── Upright PD ──
        self.upright_kp: float = 20.0    # v0.6.3: aligned with WalkerStandPD kp=20
        self.upright_kd: float = 2.0      # v0.6.3: aligned with WalkerStandPD kd=2.0
        # ── Standing pose targets (for Recovery only; Stabilize uses damping) ──
        self.standing_targets: np.ndarray = np.array([0.4, -0.5, 0.0, 0.4, -0.5, 0.0])
        # ── Phase 3: Walking gait parameters ──
        self.gait_freq: float = 10.0
        self.gait_amplitude: float = 0.45
        self.knee_amplitude: float = 0.40
        self.ankle_amplitude: float = 0.15
        # ── Velocity feedback ──
        self.vel_kp: float = 2.5
        self.forward_bias_min: float = 0.0
        self.forward_bias_max: float = 2.0
        # ── Forward lean bias ──
        self.forward_lean_bias: float = 0.21
        # ── Phase thresholds ──
        self.recovery_height_thresh: float = 1.0
        self.recovery_upright_thresh: float = 0.5  # lowered: upright can start very negative
        self.gait_fallback_height_thresh: float = 0.8
        self.gait_fallback_upright_thresh: float = 0.4
        # ── Stabilize step counter ──
        self._stabilize_steps: int = 0
        self._stabilize_target: int = 30

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute 3-phase walker-walk control with one-sided height+upright PD.

        v0.6.3 CRITICAL FIXES:
        1. ctrl = TORQUE in [-1,1] (confirmed from model diagnostic).
           Per-joint actuator range clipping was WRONG — all ctrl [-1,1].
        2. Height PD is ONE-SIDED: only pushes UP when below target.
           When walker is above 1.2m, height PD = 0 (no downward push).
        3. upright can be NEGATIVE (walker starts nearly upside down).
           upright_thresh lowered to 0.5 to accommodate negative starts.
        4. Standing targets use moderate values, not raw reset angles.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:6] clipped to [-1, 1] (torque range).
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── 1. Get walker state ──
        # Walker model: nq=9, nv=9, nu=6
        # Root joints: rootz(slide)=qpos[0], rootx(slide)=qpos[1], rooty(hinge)=qpos[2]
        # Joint qpos: qpos[3:9] = [right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle]
        # Note: qpos[0]=rootz=0 after reset (2D walker convention), use xpos for actual height
        torso_z: float = float(phys.named.data.xpos['torso', :][2])  # actual torso height from xpos
        torso_upright: float = float(phys.named.data.xmat['torso', :][8])  # zz rotation matrix component
        root_vz: float = float(phys.data.qvel[0])  # rootz velocity (vertical)
        root_pitch_vel: float = float(phys.data.qvel[2])  # rooty velocity (pitch angular)
        horiz_vel: float = float(phys.data.qvel[1])  # rootx velocity (horizontal)

        # ── 2. One-sided height PD (only push UP when below target) ──
        height_error: float = max(0.0, self.target_height - torso_z)
        height_torque: float = self.height_kp * height_error - self.height_kd * root_vz

        # ── 3. Upright PD (push toward upright ≥ 1.0) ──
        # torso_upright = xmat['torso','zz'] ranges from -1 (upside down) to 1 (upright)
        # dm_control reward uses (1 + upright) / 2, so upright=1 → reward=1, upright=-1 → reward=0
        upright_error: float = 1.0 - torso_upright  # want upright ≥ 1.0
        upright_torque: float = self.upright_kp * upright_error - self.upright_kd * root_pitch_vel

        # ── Phase determination ──
        need_recovery: bool = (torso_z < self.recovery_height_thresh
                                or torso_upright < self.recovery_upright_thresh)
        fell_during_gait: bool = (torso_z < self.gait_fallback_height_thresh
                                  or torso_upright < self.gait_fallback_upright_thresh)

        if need_recovery:
            # ── Phase 1: Recovery — fixed torque + height/upright PD ──
            # v0.6.3: Use fixed base torques instead of PD toward angle targets.
            # The initial state varies randomly, so angle targets don't work for all cases.
            # Strategy: positive hip torque pushes walker up, negative knee for support.
            self._stabilize_steps = 0
            for i in range(min(self.nu, 6)):
                qvel_idx: int = i + 3
                joint_vel: float = float(phys.data.qvel[qvel_idx]) if qvel_idx < len(phys.data.qvel) else 0.0
                # Fixed base torques (0=right_hip, 1=right_knee, 2=right_ankle, 3=left_hip, 4=left_knee, 5=left_ankle)
                if i in (0, 3):  # hip — push forward to lift torso
                    base_torque: float = self.recovery_hip_torque
                elif i in (1, 4):  # knee — bend for support
                    base_torque = self.recovery_knee_torque
                elif i in (2, 5):  # ankle — push forward for balance
                    base_torque = self.recovery_ankle_torque
                else:
                    base_torque = 0.0
                # Velocity damping
                vel_damp: float = -self.recovery_vel_kd * joint_vel * 0.1
                # Height/upright PD (WalkerStandPD-style)
                height_contrib: float = height_torque * 0.02
                upright_contrib: float = upright_torque * 0.01
                ctrl[i] = np.clip(base_torque + vel_damp + height_contrib + upright_contrib, -1.0, 1.0)

        elif self._stabilize_steps < self._stabilize_target:
            # ── Phase 2: Stabilize — WalkerStandPD-style damping + height/upright ──
            # v0.6.3 KEY: Use damping-only (no angle targets) like WalkerStandPD.
            # Previous version used joint PD toward standing targets which destabilized
            # the walker by pushing joints away from their natural standing pose.
            # WalkerStandPD uses: ctrl[i] = height_torque*0.02 + upright_torque*0.01 + damping
            self._stabilize_steps += 1
            for i in range(min(self.nu, 6)):
                qvel_idx: int = i + 3
                joint_vel: float = float(phys.data.qvel[qvel_idx]) if qvel_idx < len(phys.data.qvel) else 0.0
                # Joint damping (like WalkerStandPD: -kp * vel * 0.1)
                joint_ctrl: float = -self.stabilize_kp * joint_vel * 0.1
                # Height/upright PD contributions (WalkerStandPD-style scaling)
                height_contrib: float = height_torque * 0.02  # all joints get height PD
                upright_contrib: float = upright_torque * 0.01  # all joints get upright PD
                ctrl[i] = np.clip(joint_ctrl + height_contrib + upright_contrib, -1.0, 1.0)

        else:
            # ── Phase 3: Walking gait + height/upright PD ──
            if fell_during_gait:
                self._stabilize_steps = 0
                for i in range(min(self.nu, 6)):
                    qvel_idx = i + 3
                    joint_vel: float = float(phys.data.qvel[qvel_idx]) if qvel_idx < len(phys.data.qvel) else 0.0
                    if i in (0, 3):
                        base_torque = self.recovery_hip_torque
                    elif i in (1, 4):
                        base_torque = self.recovery_knee_torque
                    elif i in (2, 5):
                        base_torque = self.recovery_ankle_torque
                    else:
                        base_torque = 0.0
                    vel_damp: float = -self.recovery_vel_kd * joint_vel * 0.1
                    height_contrib: float = height_torque * 0.02
                    upright_contrib: float = upright_torque * 0.01
                    ctrl[i] = np.clip(base_torque + vel_damp + height_contrib + upright_contrib, -1.0, 1.0)
                return self._clip_ctrl(ctrl)

            # ── Gait: full-sine drive + height/upright PD ──
            t: float = step * 0.002
            phase_r: float = math.sin(self.gait_freq * t)
            phase_l: float = math.sin(self.gait_freq * t + math.pi)
            ankle_r: float = max(-phase_r, 0.0)
            ankle_l: float = max(-phase_l, 0.0)
            vel_error: float = self.target_speed - horiz_vel
            forward_bias: float = np.clip(vel_error * self.vel_kp, self.forward_bias_min, self.forward_bias_max)

            # Right leg (ctrl 0-2)
            ctrl[0] = np.clip(
                forward_bias + self.forward_lean_bias
                + phase_r * self.gait_amplitude
                + height_torque * 0.03
                + upright_torque * 0.02,
                -1.0, 1.0)
            ctrl[1] = np.clip(
                -phase_r * self.knee_amplitude + forward_bias * 0.2,
                -1.0, 1.0)
            ctrl[2] = np.clip(
                ankle_r * self.ankle_amplitude
                + height_torque * 0.01
                + upright_torque * 0.01,
                -1.0, 1.0)

            # Left leg (ctrl 3-5)
            ctrl[3] = np.clip(
                forward_bias + self.forward_lean_bias
                + phase_l * self.gait_amplitude
                + height_torque * 0.03
                + upright_torque * 0.02,
                -1.0, 1.0)
            ctrl[4] = np.clip(
                -phase_l * self.knee_amplitude + forward_bias * 0.2,
                -1.0, 1.0)
            ctrl[5] = np.clip(
                ankle_l * self.ankle_amplitude
                + height_torque * 0.01
                + upright_torque * 0.01,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle damping on walker joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (6,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 6)):
            qvel_idx: int = i + 3
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.2 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class WalkerRunPD(TaskPDController):
    """PD controller for walker-run: 3-phase recovery → stabilize → running gait.

    Same 3-phase structure as WalkerWalkPD but with:
    - Higher target speed: 8 m/s (vs 1.0 m/s for walk)
    - Higher gait amplitude: 0.7 (vs 0.55 for walk)
    - Higher gait frequency: 16 rad/s (vs 12 for walk)
    - Stronger velocity feedback: vel_kp=0.8 (vs 0.5 for walk)
    - Forward bias clip: (-0.3, 1.0) (vs (-0.3, 0.8) for walk)

    dm_control walker-run reward: same formula as walk but
    horizontal_velocity target is higher (running faster).

    Strategy (v0.5.4 — 3-phase control with stabilize buffer):
    Phase 1 (Recovery): torso_z < 1.0 or upright < 0.7 → aggressive PD
    Phase 2 (Stabilize): just upright, hold for 30 steps → gentle PD
    Phase 3 (Running gait): stabilized 30+ steps → fast gait
        If falls (z < 0.8 or upright < 0.6) → reset to Phase 1

    PD gains (v0.5.4 — 3-phase enhanced):
        Recovery: kp=2.5, kd=1.0 (stronger for faster recovery)
        Stabilize: kp=0.5, kd=0.2
        Running: gait_freq=16, gait_amp=0.7, vel_kp=0.8

    Output: ctrl[0:6], clipped per phase.
    """

    def __init__(self, physics) -> None:
        """Initialize WalkerRunPD with 3-phase high-speed gait gains.

        v0.5.4: Complete rewrite — 3-phase control with stabilize buffer,
        same as WalkerWalkPD but with higher target_speed, gait amplitude,
        and velocity feedback for running.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=2.5, kd=1.0)
        self.target_speed: float = 8.0
        self.target_height: float = 1.2
        # ── Phase 1: Recovery gains (aggressive) ──
        self.recovery_kp: float = 2.5
        self.recovery_kd: float = 1.0
        self.standing_targets: np.ndarray = np.array([0.0, -0.5, 0.0, 0.0, -0.5, 0.0])
        # ── Phase 2: Stabilize gains (gentle) ──
        self.stabilize_kp: float = 0.5
        self.stabilize_kd: float = 0.2
        # ── Phase 3: Running gait parameters ──
        self.gait_freq: float = 16.0  # rad/s (was 12)
        self.gait_amplitude: float = 0.75  # (was 0.7) — larger stride for running
        self.knee_amplitude: float = 0.5  # (was 0.45)
        self.ankle_amplitude: float = 0.2   # (was 0.15)
        # ── Velocity feedback (stronger for running) ──
        self.vel_kp: float = 1.0  # (was 0.8)
        self.forward_bias_min: float = -0.2  # (was -0.3)
        self.forward_bias_max: float = 1.2   # (was 1.0)
        # ── Forward lean bias ──
        self.forward_lean_bias: float = 0.07  # (~4° forward lean for running)
        # ── Upright maintenance during running ──
        self.upright_kp: float = 0.6
        self.upright_kd: float = 0.25
        # ── Phase thresholds ──
        self.recovery_height_thresh: float = 1.0
        self.recovery_upright_thresh: float = 0.7
        self.gait_fallback_height_thresh: float = 0.8
        self.gait_fallback_upright_thresh: float = 0.6
        # ── Stabilize step counter ──
        self._stabilize_steps: int = 0
        self._stabilize_target: int = 5

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute 3-phase walker-run control: recovery → stabilize → fast gait.

        v0.5.4 Strategy:
        Phase 1 (Recovery): fallen → aggressive PD
        Phase 2 (Stabilize): just upright → gentle PD for 30 steps
        Phase 3 (Running gait): stabilized → fast gait targeting 8 m/s
            If falls → reset to Phase 1

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:6] clipped per phase.
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── 1. Get walker state ──
        torso_z: float = 0.0
        try:
            torso_pos = phys.named.data.xpos['torso', :]
            torso_z = float(torso_pos[2]) if hasattr(torso_pos, '__len__') else float(torso_pos)
        except (KeyError, IndexError, TypeError):
            torso_z = float(phys.data.qpos[2]) if len(phys.data.qpos) > 2 else 0.0

        torso_upright: float = 1.0
        try:
            torso_mat = phys.named.data.xmat['torso', :]
            torso_upright = float(torso_mat[8]) if hasattr(torso_mat, '__len__') else float(torso_mat)
        except (KeyError, IndexError, TypeError):
            if len(phys.data.qpos) >= 4:
                qw = float(phys.data.qpos[3])
                qx = float(phys.data.qpos[4])
                torso_upright = 1.0 - 2.0 * qx * qx

        horiz_vel: float = 0.0
        try:
            torso_vel = phys.named.data.sensordata['torso_subtreelinvel']
            horiz_vel = float(torso_vel[0]) if hasattr(torso_vel, '__len__') else float(torso_vel)
        except (KeyError, IndexError, TypeError):
            horiz_vel = float(phys.data.qvel[1]) if len(phys.data.qvel) > 1 else 0.0

        # ── Phase determination ──
        need_recovery: bool = (torso_z < self.recovery_height_thresh
                                or torso_upright < self.recovery_upright_thresh)
        fell_during_gait: bool = (torso_z < self.gait_fallback_height_thresh
                                  or torso_upright < self.gait_fallback_upright_thresh)

        if need_recovery:
            # ── Phase 1: Recovery — aggressive joint-level PD ──
            # Walker actuator ranges: hip [-0.35, 1.75], knee [-2.62, 0], ankle [-0.79, 0.79]
            self._stabilize_steps = 0
            actuator_ranges = [(-0.35, 1.75), (-2.62, 0.0), (-0.79, 0.79),
                               (-0.35, 1.75), (-2.62, 0.0), (-0.79, 0.79)]
            for i in range(min(self.nu, 6)):
                qpos_idx: int = i + 3
                joint_angle: float = 0.0
                if qpos_idx < len(phys.data.qpos):
                    joint_angle = float(phys.data.qpos[qpos_idx])
                qvel_idx: int = i + 3
                joint_vel: float = 0.0
                if qvel_idx < len(phys.data.qvel):
                    joint_vel = float(phys.data.qvel[qvel_idx])
                target_angle: float = float(self.standing_targets[min(i, len(self.standing_targets) - 1)])
                lo, hi = actuator_ranges[min(i, len(actuator_ranges) - 1)]
                ctrl[i] = np.clip(
                    self.recovery_kp * (target_angle - joint_angle)
                    - self.recovery_kd * joint_vel,
                    lo, hi)

        elif self._stabilize_steps < self._stabilize_target:
            # ── Phase 2: Stabilize — gentle PD to hold standing pose ──
            self._stabilize_steps += 1
            for i in range(min(self.nu, 6)):
                qpos_idx = i + 3
                joint_angle: float = 0.0
                if qpos_idx < len(phys.data.qpos):
                    joint_angle = float(phys.data.qpos[qpos_idx])
                qvel_idx = i + 3
                joint_vel: float = 0.0
                if qvel_idx < len(phys.data.qvel):
                    joint_vel = float(phys.data.qvel[qvel_idx])
                target_angle: float = float(self.standing_targets[min(i, len(self.standing_targets) - 1)])
                ctrl[i] = np.clip(
                    self.stabilize_kp * (target_angle - joint_angle)
                    - self.stabilize_kd * joint_vel,
                    -0.8, 0.8)

        else:
            # ── Phase 3: Running gait (enhanced) with CoG shift ──
            if fell_during_gait:
                self._stabilize_steps = 0
                for i in range(min(self.nu, 6)):
                    qpos_idx = i + 3
                    joint_angle: float = 0.0
                    if qpos_idx < len(phys.data.qpos):
                        joint_angle = float(phys.data.qpos[qpos_idx])
                    qvel_idx = i + 3
                    joint_vel: float = 0.0
                    if qvel_idx < len(phys.data.qvel):
                        joint_vel = float(phys.data.qvel[qvel_idx])
                    target_angle: float = float(self.standing_targets[min(i, len(self.standing_targets) - 1)])
                    lo, hi = actuator_ranges[min(i, len(actuator_ranges) - 1)]
                    ctrl[i] = np.clip(
                        self.recovery_kp * (target_angle - joint_angle)
                        - self.recovery_kd * joint_vel,
                        lo, hi)
                return self._clip_ctrl(ctrl)

            t: float = step * 0.02

            phase_r: float = math.sin(self.gait_freq * t)
            phase_l: float = math.sin(self.gait_freq * t + math.pi)

            right_push: float = max(phase_r, 0.0)
            left_push: float = max(-phase_l, 0.0)

            # ── CoG shift: standing leg hip shifts forward to move CoG ──
            cog_shift_r: float = 0.04 if right_push > 0.3 else 0.0  # larger for running
            cog_shift_l: float = 0.04 if left_push > 0.3 else 0.0

            vel_error: float = self.target_speed - horiz_vel
            forward_bias: float = np.clip(
                vel_error * self.vel_kp,
                self.forward_bias_min, self.forward_bias_max)

            upright_error: float = max(0.0, 1.0 - torso_upright)
            upright_correction: float = self.upright_kp * upright_error

            # ── Right leg (ctrl 0-2) with forward lean + CoG shift ──
            ctrl[0] = np.clip(
                forward_bias + right_push * self.gait_amplitude
                + self.forward_lean_bias
                + upright_correction * 0.1
                + cog_shift_r,
                -1.0, 1.0)
            ctrl[1] = np.clip(
                -right_push * self.knee_amplitude + forward_bias * 0.4,
                -1.0, 1.0)
            ctrl[2] = np.clip(
                right_push * self.ankle_amplitude,
                -1.0, 1.0)

            # ── Left leg (ctrl 3-5) with forward lean + CoG shift ──
            ctrl[3] = np.clip(
                forward_bias + left_push * self.gait_amplitude
                + self.forward_lean_bias
                + upright_correction * 0.1
                + cog_shift_l,
                -1.0, 1.0)
            ctrl[4] = np.clip(
                -left_push * self.knee_amplitude + forward_bias * 0.4,
                -1.0, 1.0)
            ctrl[5] = np.clip(
                left_push * self.ankle_amplitude,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle damping on walker joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (6,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 6)):
            qvel_idx: int = i + 3
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.3 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class CheetahRunPD(TaskPDController):
    """PD controller for cheetah-run: gallop gait for high speed (≥ 10 m/s).

    dm_control cheetah-run reward formula (critical alignment):
        reward = tolerance(speed(), bounds=(10, inf), margin=10,
                          value_at_margin=0, sigmoid='linear')
    Max reward per step = 1.0 when speed ≥ 10 m/s.
    Linear ramp from 0 at speed=0 to 1.0 at speed=10.
    Total episode: ~400 steps → max ~400.
    Even speed=3 gives reward=0.3 → significant partial reward.

    Cheetah actuator layout (confirmed from dm_control):
        ctrl[0]: bthigh (back thigh, positive = forward swing)
        ctrl[1]: bshin (back shin, negative = extend backward)
        ctrl[2]: bfoot (back foot)
        ctrl[3]: fthigh (front thigh, positive = forward)
        ctrl[4]: fshin (front shin, negative = extend)
        ctrl[5]: ffoot (front foot)

    Strategy (v0.5.5 — gallop gait with concentrated push):
    1. Gallop gait: back legs push together, then front legs push together.
       Unlike bounding (π/2 offset), gallop uses π*0.35 offset —
       closer to real cheetah running pattern.
       back_phase = sin(freq*t), front_phase = sin(freq*t + π*0.35)
    2. Concentrated push profile: push^1.5 instead of push^1.0
       → sharper thrust at peak of stance phase, less wasted energy.
    3. Velocity feedback: vel_kp=0.5, target=10 m/s
       forward_bias = clip(vel_error * 0.5, -0.3, 1.0)
    4. Acceleration boost: 1.5x amplitude for first 50 steps to build speed.
    5. Torso stabilization: mild PD on torso pitch.

    PD gains (v0.5.5 — gallop gait):
        Gait: gait_freq=18.0 rad/s
        Back thigh amplitude: 0.9 (was 0.7)
        Front thigh amplitude: 0.8 (was 0.6)
        Back shin amplitude: 0.6 (was 0.5)
        Front shin amplitude: 0.5 (was 0.4)
        Velocity feedback: vel_kp=0.5 (was 0.3)
        Forward bias clip: (-0.3, 1.0) (was (-0.3, 0.8))
        Acceleration boost: 1.5x for first 50 steps

    Output: ctrl[0:6], clipped to [-1, 1].
    """

    def __init__(self, physics) -> None:
        """Initialize CheetahRunPD with gallop gait parameters.

        v0.5.5: Complete rewrite — gallop gait (π*0.35 offset instead
        of π/2 bounding), concentrated push profile (phase^1.5),
        acceleration boost for first 50 steps, enhanced amplitudes.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=0.3, kd=0.1)
        self.target_speed: float = 10.0
        # Gallop gait parameters (v0.6.1 enhanced)
        self.gait_freq: float = 22.0  # v0.6.1: increased from 18 for faster cycle
        self.gallop_offset: float = math.pi * 0.35  # gallop phase offset (not π/2)
        self.back_thigh_amp: float = 1.1   # v0.6.1: increased from 0.9
        self.back_shin_amp: float = 0.7    # v0.6.1: increased from 0.6
        self.back_foot_amp: float = 0.45   # v0.6.1: increased from 0.35
        self.front_thigh_amp: float = 1.0  # v0.6.1: increased from 0.8
        self.front_shin_amp: float = 0.6   # v0.6.1: increased from 0.5
        self.front_foot_amp: float = 0.30  # v0.6.1: increased from 0.25
        # Velocity feedback (v0.6.1 enhanced)
        self.vel_kp: float = 0.8   # v0.6.1: increased from 0.5 for stronger drive
        self.forward_bias_min: float = -0.2  # v0.6.1: less negative → more forward
        self.forward_bias_max: float = 1.5  # v0.6.1: increased from 1.0
        # Torso pitch stabilization (v0.6.1 enhanced)
        self.pitch_kp: float = 0.4  # v0.6.1: increased from 0.3
        self.pitch_kd: float = 0.15 # v0.6.1: increased from 0.1
        # Acceleration boost: 1.5x amplitude for first 80 steps
        self.boost_factor: float = 1.5
        self.boost_steps: int = 80  # v0.6.1: increased from 50

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute gallop gait control for cheetah with concentrated push.

        v0.5.5 Strategy:
        1. Gallop gait: back legs push first, then front legs with
           π*0.35 offset (not π/2 bounding).
        2. Concentrated push: phase^1.5 → sharper thrust at peak.
        3. Velocity feedback adjusts gait amplitude targeting 10 m/s.
        4. Acceleration boost: 1.5x for first 50 steps.
        5. Torso pitch stabilization prevents excessive pitching.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:6] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── 1. Get cheetah state ──
        speed: float = 0.0
        try:
            torso_vel = phys.named.data.sensordata['torso_subtreelinvel']
            speed = float(torso_vel[0]) if hasattr(torso_vel, '__len__') else float(torso_vel)
        except (KeyError, IndexError, TypeError):
            speed = float(phys.data.qvel[0]) if len(phys.data.qvel) > 0 else 0.0

        torso_pitch: float = 0.0
        try:
            torso_pitch = float(phys.data.qpos[2]) if len(phys.data.qpos) > 2 else 0.0
        except (IndexError, AttributeError):
            torso_pitch = 0.0

        torso_pitch_vel: float = 0.0
        try:
            torso_pitch_vel = float(phys.data.qvel[2]) if len(phys.data.qvel) > 2 else 0.0
        except (IndexError, AttributeError):
            torso_pitch_vel = 0.0

        # ── 2. Velocity feedback ──
        vel_error: float = self.target_speed - speed
        forward_bias: float = np.clip(
            vel_error * self.vel_kp,
            self.forward_bias_min, self.forward_bias_max)

        # ── 3. Torso pitch stabilization ──
        pitch_correction: float = (
            -self.pitch_kp * torso_pitch
            - self.pitch_kd * torso_pitch_vel)

        # ── 4. Acceleration boost ──
        boost: float = self.boost_factor if step < self.boost_steps else 1.0

        # ── 5. Gallop gait ──
        t: float = step * 0.02  # time in seconds

        # Gallop: back legs push first, then front legs with π*0.35 offset
        back_phase: float = math.sin(self.gait_freq * t)
        front_phase: float = math.sin(self.gait_freq * t + self.gallop_offset)

        # Concentrated push profile: phase^1.5 → sharper thrust at peak
        back_push: float = max(back_phase, 0.0) ** 1.5
        front_push: float = max(front_phase, 0.0) ** 1.5

        # Swing phases (retract legs during non-push phases)
        back_swing: float = min(back_phase, 0.0)  # negative = swing
        front_swing: float = min(front_phase, 0.0)  # negative = swing

        if self.nu >= 6:
            # ── Back leg (ctrl 0-2) ──
            # ctrl[0]: bthigh — forward push during stance, retract during swing
            ctrl[0] = np.clip(
                forward_bias + back_push * self.back_thigh_amp * boost
                + back_swing * self.back_thigh_amp * 0.3  # gentle retract
                + pitch_correction * 0.2,
                -1.0, 1.0)
            # ctrl[1]: bshin — extend backward during stance (negative)
            ctrl[1] = np.clip(
                -back_push * self.back_shin_amp * boost
                - back_swing * self.back_shin_amp * 0.3,
                -1.0, 1.0)
            # ctrl[2]: bfoot — push down during stance
            ctrl[2] = np.clip(
                back_push * self.back_foot_amp * boost,
                -1.0, 1.0)

            # ── Front leg (ctrl 3-5) ──
            # ctrl[3]: fthigh — forward push during stance, retract during swing
            ctrl[3] = np.clip(
                forward_bias + front_push * self.front_thigh_amp * boost
                + front_swing * self.front_thigh_amp * 0.3  # gentle retract
                + pitch_correction * 0.2,
                -1.0, 1.0)
            # ctrl[4]: fshin — extend backward during stance
            ctrl[4] = np.clip(
                -front_push * self.front_shin_amp * boost
                - front_swing * self.front_shin_amp * 0.3,
                -1.0, 1.0)
            # ctrl[5]: ffoot — push down during stance
            ctrl[5] = np.clip(
                front_push * self.front_foot_amp * boost,
                -1.0, 1.0)
        else:
            # Fallback for fewer actuators
            for i in range(self.nu):
                push: float = back_push if i < 3 else front_push
                ctrl[i] = np.clip(
                    forward_bias + push * 0.5 * boost,
                    -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle damping on cheetah joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (6,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 6)):
            qvel_idx: int = i + 3  # cheetah root has ~3 velocity DOFs
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.2 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class HopperStandPD(TaskPDController):
    """PD controller for hopper-stand: maintain height + upright.

    Targets:
        - Torso height ≈ target_height (standing)
        - Upright torso
        - Joint stabilization

    PD gains (v0.5.0 tuned):
        - Height: kp=30, kd=3 (lowered from 40)
        - Upright: kp=15, kd=1.5 (lowered from 20)
        - Joint stabilize: kp=2, kd=0.5 (lowered from 8)

    Output: ctrl[0:4], hopper has 4 actuators.
    """

    def __init__(self, physics) -> None:
        """Initialize HopperStandPD with tuned gains.

        v0.5.0: height_kp lowered 40→30, upright_kp lowered 20→15,
        joint_kp lowered 8→2 for collision avoidance.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=30.0, kd=3.0)
        self.target_height: float = 1.2
        self.height_kp: float = 30.0
        self.height_kd: float = 3.0
        self.upright_kp: float = 15.0
        self.upright_kd: float = 1.5
        self.joint_kp: float = 2.0
        self.joint_kd: float = 0.5

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute standing stabilization PD control for hopper.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:4] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # ── Height PD ──
        root_z: float = float(phys.data.qpos[2])
        height_error: float = self.target_height - root_z
        root_vz: float = float(phys.data.qvel[2])
        height_torque: float = self._pd_scalar(
            height_error, -root_vz, self.height_kp, self.height_kd)

        # ── Upright PD ──
        upright_torque: float = 0.0
        if len(phys.data.qpos) >= 4:
            qx: float = float(phys.data.qpos[3])
            if len(phys.data.qvel) >= 3:
                upright_vel: float = float(phys.data.qvel[2])
                upright_torque = self._pd_scalar(
                    -qx, -upright_vel, self.upright_kp, self.upright_kd)

        # ── Joint Stabilization ──
        # Hopper joints: hip, knee, ankle (roughly indices 1-3 after root)
        for i in range(min(self.nu, 4)):
            qvel_idx: int = i + 1  # hopper root has ~1 velocity DOF
            if qvel_idx < len(phys.data.qvel):
                joint_vel: float = float(phys.data.qvel[qvel_idx])
                joint_ctrl: float = -self.joint_kp * joint_vel * 0.1
            else:
                joint_ctrl: float = 0.0

            if i == 0:
                # Hip: height + upright + stabilize
                ctrl[i] = np.clip(
                    height_torque * 0.03 + upright_torque * 0.02 + joint_ctrl,
                    -1.0, 1.0)
            else:
                # Other joints: height + stabilize
                ctrl[i] = np.clip(
                    height_torque * 0.02 + joint_ctrl, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping on hopper joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (4,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(min(self.nu, 4)):
            qvel_idx: int = i + 1
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.4 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class HopperHopPD(TaskPDController):
    """Recovery→Stand→Hop controller for hopper-hop with correct actuator signs.

    Actuator sign conventions (empirically verified via diagnostic tests):
    - ctrl[0] (waist, gear=30): POSITIVE → rooty DECREASES (torso leans backward)
      → For upright stabilization: ctrl[0] ∝ +rooty (NOT -rooty!)
    - ctrl[1] (hip, gear=40): POSITIVE → hip angle INCREASES (thigh swings forward)
      → Reaction torque also pushes rooty negative (torso backward)
    - ctrl[2] (knee, gear=30): POSITIVE → knee angle INCREASES (knee BENDS more)
      → Leg SHORTENS → height DECREASES
      → For push-up: ctrl[2] NEGATIVE (straighten knee to extend leg)
    - ctrl[3] (ankle, gear=10): assumed POSITIVE → ankle extends (pushes foot down)

    dm_control hopper-hop reward:
    - standing = tolerance(height, bounds=(0.6, 2)) where height = torso_z - foot_z
    - hopping = tolerance(speed, bounds=(2, inf), margin=1, sigmoid='linear')
    - reward = standing * hopping

    Three operating modes based on |rooty|:
    1. RECOVERY (|rooty| > 1.0): Maximum waist torque to flip upright, retract leg
    2. STANDING (|rooty| < 1.0, height < 0.6): Extend leg (negative ctrl[2]) to push up
    3. HOPPING (|rooty| < 0.5, height >= 0.6): Rhythmic hop + forward lean

    v0.8.0: Complete rewrite fixing the fundamental waist sign bug.
    Old code used ctrl[0] ∝ -rooty (wrong!), causing rooty to diverge.
    New code uses ctrl[0] ∝ +rooty (correct, pushes rooty toward 0).

    Output: ctrl[0:4] clipped to [-1, 1].
    """

    def __init__(self, physics) -> None:
        """Initialize HopperHopPD with verified gains and sign conventions.

        Args:
            physics: dm_control Physics instance (hopper.Physics).
        """
        super().__init__(physics, kp=40.0, kd=4.0)

        # ── Target parameters ──
        self.target_speed: float = 2.0       # dm_control _HOP_SPEED threshold
        self.target_height: float = 1.0      # torso_z - foot_z in [0.6, 2]
        self.target_knee_stand: float = 0.5  # knee angle for standing (slight bend)
        self.target_hip_stand: float = -0.1  # hip angle for standing (slightly behind vertical)

        # ── Waist (upright) PD gains ──
        # CRITICAL: Both proportional AND damping signs are inverted vs standard PD
        # because positive ctrl[0] makes rooty DECREASE.
        # Correct formula: ctrl[0] = Kp * rooty + Kd * rooty_vel
        #   - rooty > 0 → ctrl > 0 → rooty decreases (correct)
        #   - rooty < 0 → ctrl < 0 → rooty increases (correct)
        #   - rooty_vel < 0 (falling backward) → Kd*vel < 0 → ctrl more negative → rooty increases (correct damping)
        #   - rooty_vel > 0 (falling forward) → Kd*vel > 0 → ctrl more positive → rooty decreases (correct damping)
        self.upright_kp: float = 1.2      # proportional gain on rooty
        self.upright_kd: float = 0.15     # damping on rooty angular velocity
        self.upright_kp_recovery: float = 0.7  # recovery (already saturates at ctrl=1)
        self.upright_kd_recovery: float = 0.15

        # ── Hip PD gains ──
        self.hip_kp: float = 0.3
        self.hip_kd: float = 0.08

        # ── Knee PD gains ──
        # ctrl[2] = Kp * (knee_target - knee_pos) + Kd * (-knee_vel)
        # Negative ctrl[2] when knee > target → knee straightens → push up
        self.knee_kp: float = 0.5
        self.knee_kd: float = 0.1

        # ── Ankle PD gains ──
        self.ankle_kp: float = 0.2
        self.ankle_kd: float = 0.05

        # ── Hopping parameters ──
        self.hop_freq: float = 2.5       # Hz, hopping cycle frequency
        self.hop_dt: float = 0.02        # control timestep

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute recovery→stand→hop control for hopper.

        Three modes based on torso lean angle (rooty):
        1. RECOVERY: |rooty| > 1.0 → flip upright via waist, retract leg
        2. STANDING: |rooty| < 1.0, height < 0.6 → extend leg to push up
        3. HOPPING: |rooty| < 0.5, height >= 0.6 → rhythmic hop + forward

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:4] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── Read state ──
        rooty: float = float(phys.data.qpos[2])    # torso lean angle (rad)
        rooty_vel: float = float(phys.data.qvel[2]) # torso angular velocity
        height: float = float(phys.height())         # torso_z - foot_z
        speed: float = float(phys.speed())           # forward subtree velocity
        root_vz: float = float(phys.data.qvel[1])    # vertical velocity

        # Joint positions
        hip_pos: float = float(phys.data.qpos[4])
        knee_pos: float = float(phys.data.qpos[5])
        ankle_pos: float = float(phys.data.qpos[6])

        # Joint velocities
        hip_vel: float = float(phys.data.qvel[4])
        knee_vel: float = float(phys.data.qvel[5])
        ankle_vel: float = float(phys.data.qvel[6])

        abs_rooty: float = abs(rooty)

        # ── MODE 1: RECOVERY (|rooty| > 1.0) ──
        # Priority: flip upright using waist only (hip interferes with leg).
        # Waist: ctrl[0] = +rooty * K + rooty_vel * Kd (inverted actuator sign)
        if abs_rooty > 1.0:
            ctrl[0] = np.clip(
                rooty * self.upright_kp_recovery
                + rooty_vel * self.upright_kd_recovery,
                -1.0, 1.0)

            # Gentle hip PD toward 0 (don't fight the flip)
            hip_error: float = -hip_pos
            ctrl[1] = np.clip(hip_error * 0.15, -0.5, 0.5)

            # Retract leg: bend knee to avoid pushing wrong direction
            knee_retract_error: float = 1.5 - knee_pos  # target 1.5 rad (bent)
            ctrl[2] = np.clip(
                knee_retract_error * 0.3,
                -0.5, 0.8)

            # Ankle neutral
            ctrl[3] = 0.0

            return self._clip_ctrl(ctrl)

        # ── MODE 2 & 3: Upright enough to try standing/hopping ──
        # Upright correction (always on when |rooty| < 1.0)
        # ctrl[0] = Kp * rooty + Kd * rooty_vel (both signs inverted vs standard PD)
        ctrl[0] = np.clip(
            rooty * self.upright_kp + rooty_vel * self.upright_kd,
            -1.0, 1.0)

        if height < 0.5:
            # ── MODE 2: STANDING (push up to standing height) ──
            # Knee: extend (straighten) to push torso up
            # ctrl[2] = Kp * (target - knee_pos) + Kd * (-knee_vel)
            # When knee_pos > target: error < 0 → ctrl[2] < 0 → knee decreases → extends
            knee_error: float = self.target_knee_stand - knee_pos
            ctrl[2] = np.clip(
                knee_error * self.knee_kp - knee_vel * self.knee_kd,
                -1.0, 1.0)

            # Hip: PD toward standing position (slightly behind vertical)
            hip_error = self.target_hip_stand - hip_pos
            ctrl[1] = np.clip(
                hip_error * self.hip_kp - hip_vel * self.hip_kd,
                -1.0, 1.0)

            # Ankle: PD toward flat foot
            ctrl[3] = np.clip(
                -ankle_pos * self.ankle_kp - ankle_vel * self.ankle_kd,
                -1.0, 1.0)

            # Add small forward lean bias to build speed (positive rooty = forward)
            # Subtract from ctrl[0] to increase rooty (forward lean)
            ctrl[0] = np.clip(ctrl[0] - 0.03, -1.0, 1.0)

            # Emergency push: if very low and falling, extend harder
            if height < 0.3 and root_vz < -0.3:
                ctrl[2] = np.clip(ctrl[2] - 0.5, -1.0, 1.0)

            return self._clip_ctrl(ctrl)

        # ── MODE 3: HOPPING (upright and height OK) ──
        # Add forward lean + rhythmic hop

        # Hop phase (sinusoidal modulation)
        hop_t: float = step * self.hop_dt * self.hop_freq
        hop_mod: float = np.sin(2.0 * np.pi * hop_t)
        push_phase: float = max(0.0, hop_mod)   # push during first half of cycle
        retract_phase: float = max(0.0, -hop_mod)  # retract during second half

        # ── Waist: upright + slight forward lean for speed ──
        speed_error: float = self.target_speed - speed
        lean_bias: float = 0.05  # slight forward lean (positive rooty = forward)
        # Note: positive ctrl[0] makes rooty decrease, so to add forward lean
        # (increase rooty), we subtract from ctrl[0]
        # Both Kp and Kd use inverted signs (see __init__ comment)
        ctrl[0] = np.clip(
            rooty * self.upright_kp + rooty_vel * self.upright_kd - lean_bias,
            -1.0, 1.0)

        # ── Hip: PD + forward propulsion ──
        hip_target: float = self.target_hip_stand + speed_error * 0.03
        hip_error = hip_target - hip_pos
        ctrl[1] = np.clip(
            hip_error * self.hip_kp - hip_vel * self.hip_kd,
            -1.0, 1.0)

        # ── Knee: hop rhythm ──
        # Push phase: extend knee (lower target → negative ctrl → straighten)
        # Retract phase: bend knee (higher target → positive ctrl → bend)
        knee_target: float = self.target_knee_stand - push_phase * 0.3 + retract_phase * 0.4
        knee_error = knee_target - knee_pos
        ctrl[2] = np.clip(
            knee_error * self.knee_kp - knee_vel * self.knee_kd,
            -1.0, 1.0)

        # ── Ankle: push during stance ──
        ctrl[3] = np.clip(
            -ankle_pos * self.ankle_kp - ankle_vel * self.ankle_kd
            + push_phase * 0.2,
            -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping on hopper joints to slow down.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (4,) clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        # Joint velocity damping: waist, hip, knee, ankle
        # qvel indices: 3=waist, 4=hip, 5=knee, 6=ankle
        for i in range(min(self.nu, 4)):
            qvel_idx: int = i + 3
            if qvel_idx < len(physics.data.qvel):
                vel: float = float(physics.data.qvel[qvel_idx])
                ctrl[i] = np.clip(-0.5 * vel, -0.5, 0.5)
        # Also damp root angular velocity (rooty) via waist
        if len(physics.data.qvel) > 2:
            rooty_vel: float = float(physics.data.qvel[2])
            # ctrl[0] = +rooty_vel * K for damping (inverted actuator sign)
            ctrl[0] = np.clip(rooty_vel * 0.3, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class CartpoleBalancePD(TaskPDController):
    """PD controller for cartpole-balance/swingup: keep pole upright.

    Targets:
        - Pole angle ≈ 0 (upright)
        - Cart position ≈ 0 (centered)

    Observations used:
        - Pole angle and angular velocity (from qpos/qvel)
        - Cart position and velocity

    PD gains (v0.5.0 tuned):
        - Pole angle: kp=60, kd=12 (increased from 50 for stronger upright)
        - Cart position: kp=5, kd=2 (gentle centering)

    Output: ctrl[0:1], cartpole has 1 actuator (cart force).
    """

    def __init__(self, physics) -> None:
        """Initialize CartpoleBalancePD with tuned gains.

        v0.5.0: pole_kp increased 50→60 for better pole retention.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=60.0, kd=12.0)
        self.pole_kp: float = 60.0
        self.pole_kd: float = 12.0
        self.cart_kp: float = 5.0
        self.cart_kd: float = 2.0

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute balancing PD control for cartpole.

        Strategy:
        Classic cartpole PD: move cart to counter pole tilt.
        - Pole angle error → cart force direction
        - Pole angular velocity → anticipatory damping
        - Cart position → gentle centering correction

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:1] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # ── Pole Angle PD ──
        # Cartpole: qpos[0] = cart x, qpos[1] = pole angle
        # qvel[0] = cart velocity, qvel[1] = pole angular velocity
        pole_angle: float = float(phys.data.qpos[1])
        pole_vel: float = float(phys.data.qvel[1])
        pole_error: float = -pole_angle  # upright = angle ≈ 0
        pole_error_dot: float = -pole_vel

        # ── Cart Position PD ──
        cart_pos: float = float(phys.data.qpos[0])
        cart_vel: float = float(phys.data.qvel[0])
        cart_error: float = -cart_pos  # center = pos ≈ 0
        cart_error_dot: float = -cart_vel

        # Combined PD: pole dominates (balance), cart secondary (centering)
        pole_torque: float = self._pd_scalar(
            pole_error, pole_error_dot, self.pole_kp, self.pole_kd)
        cart_torque: float = self._pd_scalar(
            cart_error, cart_error_dot, self.cart_kp, self.cart_kd)

        # Single actuator: cart force
        total_force: float = pole_torque + cart_torque
        # Scale to [-1, 1] (normalize large PD outputs)
        ctrl[0] = np.clip(total_force / 100.0, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle brake (small opposing force to cart velocity).

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (1,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        cart_vel: float = float(physics.data.qvel[0])
        ctrl[0] = np.clip(-0.5 * cart_vel, -0.3, 0.3)
        return self._clip_ctrl(ctrl)


class CartpoleSwingupPD(TaskPDController):
    """PD controller for cartpole-swingup: swing pole from down to up.

    Strategy:
    When pole is near bottom (angle ≈ π), use energy pumping to swing
    up. Once near upright (angle ≈ 0), switch to balance PD.

    PD gains:
        - Balance phase: kp=50, kd=10 (same as balance)
        - Swingup phase: kp=10, kd=3 (energy pumping)

    Output: ctrl[0:1].
    """

    def __init__(self, physics) -> None:
        """Initialize CartpoleSwingupPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=50.0, kd=10.0)
        self.balance_kp: float = 50.0
        self.balance_kd: float = 10.0
        self.swingup_kp: float = 10.0
        self.swingup_kd: float = 3.0
        self.cart_kp: float = 5.0
        self.cart_kd: float = 2.0
        self.swingup_thresh: float = 0.5  # radians from upright to switch

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute swingup/balance hybrid PD control.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:1] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        pole_angle: float = float(phys.data.qpos[1])
        pole_vel: float = float(phys.data.qvel[1])
        cart_pos: float = float(phys.data.qpos[0])
        cart_vel: float = float(phys.data.qvel[0])

        # Determine phase: near upright or far from it
        # Normalize angle to [-π, π]
        angle_normalized: float = pole_angle % (2 * math.pi)
        if angle_normalized > math.pi:
            angle_normalized -= 2 * math.pi

        if abs(angle_normalized) < self.swingup_thresh:
            # ── Balance Phase ──
            pole_error: float = -pole_angle
            pole_error_dot: float = -pole_vel
            pole_torque: float = self._pd_scalar(
                pole_error, pole_error_dot,
                self.balance_kp, self.balance_kd)
            cart_torque: float = self._pd_scalar(
                -cart_pos, -cart_vel, self.cart_kp, self.cart_kd)
            total_force: float = pole_torque + cart_torque
            ctrl[0] = np.clip(total_force / 100.0, -1.0, 1.0)
        else:
            # ── Swingup Phase ──
            # Energy pumping: add energy to swing pole upward
            # Strategy: push cart in direction that adds angular velocity
            # toward upright. Sign of force depends on pole angular velocity.
            energy_torque: float = self._pd_scalar(
                -pole_angle, -pole_vel,
                self.swingup_kp, self.swingup_kd)
            # Add small centering to prevent cart runaway
            centering: float = self._pd_scalar(
                -cart_pos, -cart_vel, self.cart_kp * 0.5, self.cart_kd * 0.5)
            ctrl[0] = np.clip(
                (energy_torque + centering) / 50.0, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: gentle brake.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (1,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        cart_vel: float = float(physics.data.qvel[0])
        ctrl[0] = np.clip(-0.3 * cart_vel, -0.2, 0.2)
        return self._clip_ctrl(ctrl)


class FishSwimPD(TaskPDController):
    """PD controller for fish-swim: swim toward target + upright.

    Targets:
        - Swim toward target direction (from observation)
        - Maintain upright orientation
        - Joint oscillation for swimming propulsion

    Observations used:
        - timestep.observation for target direction
        - root orientation for upright
        - Joint angles/velocities

    PD gains (v0.5.0 tuned):
        - Target direction: kp=25, kd=2.5 (increased from 20 for better steering)
        - Upright: kp=20, kd=2.5 (increased from 15)
        - Joint oscillation freq: 3 Hz

    Output: ctrl covering fish actuators (5 actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize FishSwimPD with tuned gains.

        v0.5.0: target_kp increased 8→25, upright_kp increased 15→20.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=25.0, kd=2.5)
        self.target_kp: float = 25.0
        self.target_kd: float = 2.5
        self.upright_kp: float = 20.0
        self.upright_kd: float = 2.5
        self.gait_freq: float = 3.0

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute swimming PD control for fish.

        Strategy:
        1. Extract target direction from observation.
        2. PD toward target direction → tail/body oscillation.
        3. Upright correction → prevent rolling.
        4. Sinusoidal joint oscillation → swimming propulsion.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # Extract target direction from observation
        # Fish task provides 'target' or direction-like observations
        target_dir: np.ndarray = np.zeros(3)
        if 'target' in obs:
            target_obs = obs['target']
            if target_obs is not None:
                target_dir = np.asarray(target_obs).flatten()
                if len(target_dir) >= 3:
                    target_dir = target_dir[:3]

        # ── Upright PD ──
        # Fish root orientation: qpos includes orientation quaternion
        upright_torque: float = 0.0
        if len(phys.data.qpos) >= 4:
            qw: float = float(phys.data.qpos[3])
            qx: float = float(phys.data.qpos[4])
            upright_error: float = -qx
            if len(phys.data.qvel) >= 3:
                upright_vel: float = float(phys.data.qvel[3])
                upright_torque = self._pd_scalar(
                    upright_error, -upright_vel,
                    self.upright_kp, self.upright_kd)

        # ── Target Direction PD ──
        # Convert target direction to joint control
        # Fish actuators control tail/fin oscillation
        target_force: float = 0.0
        if len(target_dir) >= 2:
            # Simplified: target x-component drives forward, y turns
            target_force = self._pd_scalar(
                float(target_dir[0]), 0.0, self.target_kp, self.target_kd)

        # ── Swimming Oscillation ──
        phase: float = math.sin(self.gait_freq * step * 0.01)

        # Fish typically has 5 actuators:
        # tail1, tail2, fin1, fin2, body_control
        for i in range(min(self.nu, 5)):
            if i < 2:
                # Tail joints: oscillation + target direction
                ctrl[i] = np.clip(
                    phase * 0.3 * (1 if i == 0 else -1)
                    + target_force * 0.01, -1.0, 1.0)
            elif i < 4:
                # Fin joints: oscillation + upright correction
                ctrl[i] = np.clip(
                    phase * 0.15 + upright_torque * 0.01, -1.0, 1.0)
            else:
                # Body stabilization: upright + damping
                ctrl[i] = np.clip(upright_torque * 0.02, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping on fish joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(self.nu):
            vel: float = float(physics.data.qvel[min(
                i + 1, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(-0.3 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class SwimmerSwimPD(TaskPDController):
    """PD controller for swimmer-swim6/swim15: forward swimming locomotion.

    Targets:
        - Forward COM velocity ≈ target_speed
        - Joint oscillation for serpentine swimming pattern

    Observations used:
        - COM velocity (from qvel)
        - Joint angles/velocities

    PD gains (v0.5.0 tuned):
        - Forward velocity: kp=20, kd=2 (increased from 15/10 for more thrust)
        - Joint oscillation: traveling sinusoidal wave along body

    Output: ctrl covering all swimmer actuators (5 for swim6, 14 for swim15).
    """

    def __init__(self, physics) -> None:
        """Initialize SwimmerSwimPD with tuned gains.

        v0.5.0: vel_kp increased 10→20 for stronger forward propulsion.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=20.0, kd=2.0)
        self.target_speed: float = 0.5
        self.vel_kp: float = 20.0
        self.vel_kd: float = 2.0
        self.gait_freq: float = 2.0
        self.phase_offset: float = 0.5  # traveling wave offset per joint

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute swimming PD control with traveling wave pattern.

        Strategy:
        1. Forward velocity PD → overall swimming force.
        2. Traveling sinusoidal wave → each joint oscillates with
           progressive phase shift (serpentine locomotion).

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # ── Forward Velocity PD ──
        com_vx: float = float(phys.data.qvel[0])
        vel_error: float = self.target_speed - com_vx
        forward_torque: float = self._pd_scalar(
            vel_error, 0.0, self.vel_kp, self.vel_kd)

        # ── Traveling Wave ──
        # Each joint gets sinusoidal oscillation with phase offset
        # proportional to its position along the body
        t: float = step * 0.01
        for i in range(self.nu):
            joint_phase: float = self.gait_freq * t - i * self.phase_offset
            oscillation: float = math.sin(joint_phase)
            # Mix forward velocity PD with oscillation
            ctrl[i] = np.clip(
                forward_torque * 0.01 + oscillation * 0.4, -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: damping on swimmer joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(self.nu):
            vel: float = float(physics.data.qvel[min(
                i + 2, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(-0.3 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


class FingerSpinPD(TaskPDController):
    """PD controller for finger-spin: spin object to target rotation.

    Targets:
        - Object rotation speed toward target (from observation)

    PD gains:
        - kp=10, kd=1

    Output: ctrl covering finger actuators (2 actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize FingerSpinPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=10.0, kd=1.0)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute spinning PD control for finger.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # Extract target rotation or object velocity from observation
        target_vel: np.ndarray = np.zeros(2)
        if 'target' in obs:
            target_vel = np.asarray(obs.get('target', np.zeros(2))).flatten()

        # Simple oscillatory spin action
        phase: float = math.sin(3.0 * step * 0.01)
        for i in range(min(self.nu, 2)):
            ctrl[i] = np.clip(
                float(target_vel[min(i, len(target_vel) - 1)]) * 0.5
                + phase * 0.3, -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class FingerTurnEasyPD(TaskPDController):
    """PD controller for finger-turn_easy: rotate object to target orientation.

    Targets:
        - Object orientation toward target (from observation)

    PD gains:
        - kp=8, kd=1

    Output: ctrl covering finger actuators (2 actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize FingerTurnEasyPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=8.0, kd=1.0)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute turning PD control for finger.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # Extract target orientation error from observation
        target: np.ndarray = np.zeros(2)
        if 'target' in obs:
            target = np.asarray(obs.get('target', np.zeros(2))).flatten()

        for i in range(min(self.nu, 2)):
            error: float = float(target[min(i, len(target) - 1)])
            vel: float = float(physics.data.qvel[min(
                i + 1, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(
                self._pd_scalar(error, -vel, self.kp, self.kd) * 0.05,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class FingerTurnHardPD(TaskPDController):
    """PD controller for finger-turn_hard: harder variant of turn.

    Same as FingerTurnEasyPD but with tighter gains.

    PD gains:
        - kp=12, kd=1.5

    Output: ctrl covering finger actuators (2 actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize FingerTurnHardPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=12.0, kd=1.5)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute turning PD control (hard variant) for finger.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        target: np.ndarray = np.zeros(2)
        if 'target' in obs:
            target = np.asarray(obs.get('target', np.zeros(2))).flatten()

        for i in range(min(self.nu, 2)):
            error: float = float(target[min(i, len(target) - 1)])
            vel: float = float(physics.data.qvel[min(
                i + 1, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(
                self._pd_scalar(error, -vel, self.kp, self.kd) * 0.04,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class BallInCupCatchPD(TaskPDController):
    """PD controller for ball_in_cup-catch: catch ball in cup.

    Targets:
        - Cup position matches ball trajectory (from observation)

    PD gains:
        - kp=15, kd=2

    Output: ctrl covering ball_in_cup actuators (2 actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize BallInCupCatchPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=15.0, kd=2.0)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute catching PD control for ball_in_cup.

        Strategy:
        Oscillate cup to catch the ball. The cup must be positioned
        below the ball when it descends.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # Extract ball position relative to cup
        ball_pos: np.ndarray = np.zeros(2)
        if 'ball' in obs:
            ball_pos = np.asarray(obs.get('ball', np.zeros(2))).flatten()
        elif 'position' in obs:
            ball_pos = np.asarray(obs.get('position', np.zeros(2))).flatten()

        # Oscillation to swing cup and catch ball
        phase: float = math.sin(2.0 * step * 0.01)

        for i in range(min(self.nu, 2)):
            error: float = float(ball_pos[min(i, len(ball_pos) - 1)])
            vel: float = float(physics.data.qvel[min(
                i + 1, len(physics.data.qvel) - 1)])
            ctrl[i] = np.clip(
                self._pd_scalar(-error, -vel, self.kp, self.kd) * 0.03
                + phase * 0.3, -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class AcrobotSwingupPD(TaskPDController):
    """PD controller for acrobot-swingup: swing second link upright.

    Targets:
        - Second link angle ≈ 0 (upright)

    PD gains:
        - kp=10, kd=3 (energy pumping then balance)

    Output: ctrl[0:1], acrobot has 1 actuator (first joint torque).
    """

    def __init__(self, physics) -> None:
        """Initialize AcrobotSwingupPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=10.0, kd=3.0)
        self.swingup_kp: float = 10.0
        self.swingup_kd: float = 3.0
        self.balance_kp: float = 30.0
        self.balance_kd: float = 6.0
        self.swingup_thresh: float = 0.5

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute swingup PD control for acrobot.

        Strategy:
        Similar to cartpole swingup: energy pumping when far from upright,
        PD balance when near upright.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:1] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # Acrobot: qpos[0] = first link angle, qpos[1] = second link angle
        # qvel[0] = first angular vel, qvel[1] = second angular vel
        joint1_angle: float = float(phys.data.qpos[0])
        joint2_angle: float = float(phys.data.qpos[1])
        joint1_vel: float = float(phys.data.qvel[0])
        joint2_vel: float = float(phys.data.qvel[1])

        # Normalize second link angle
        angle_norm: float = joint2_angle % (2 * math.pi)
        if angle_norm > math.pi:
            angle_norm -= 2 * math.pi

        if abs(angle_norm) < self.swingup_thresh:
            # ── Balance Phase ──
            error: float = -joint2_angle
            error_dot: float = -joint2_vel
            torque: float = self._pd_scalar(
                error, error_dot, self.balance_kp, self.balance_kd)
            ctrl[0] = np.clip(torque / 50.0, -1.0, 1.0)
        else:
            # ── Swingup Phase ──
            # Energy pumping: oscillate first joint to add energy
            torque: float = self._pd_scalar(
                -joint1_angle, -joint1_vel,
                self.swingup_kp, self.swingup_kd)
            ctrl[0] = np.clip(torque / 20.0, -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class PendulumSwingupPD(TaskPDController):
    """PD controller for pendulum-swingup: swing pendulum upright.

    Targets:
        - Pendulum angle ≈ 0 (upright from hanging)

    PD gains:
        - Balance: kp=30, kd=6
        - Swingup: kp=8, kd=2

    Output: ctrl[0:1], pendulum has 1 actuator.
    """

    def __init__(self, physics) -> None:
        """Initialize PendulumSwingupPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=30.0, kd=6.0)
        self.balance_kp: float = 30.0
        self.balance_kd: float = 6.0
        self.swingup_kp: float = 8.0
        self.swingup_kd: float = 2.0
        self.swingup_thresh: float = 0.5

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute swingup/balance PD control for pendulum.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl[0:1] clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)

        # Pendulum: qpos[0] = angle, qvel[0] = angular velocity
        angle: float = float(phys.data.qpos[0])
        vel: float = float(phys.data.qvel[0])

        angle_norm: float = angle % (2 * math.pi)
        if angle_norm > math.pi:
            angle_norm -= 2 * math.pi

        if abs(angle_norm) < self.swingup_thresh:
            # Balance phase
            ctrl[0] = np.clip(
                self._pd_scalar(-angle, -vel,
                                self.balance_kp, self.balance_kd) / 50.0,
                -1.0, 1.0)
        else:
            # Swingup: energy pumping with sign-based oscillation
            # Push in direction that adds energy toward upright
            sign: float = 1.0 if vel > 0 else -1.0
            ctrl[0] = np.clip(sign * 0.8, -1.0, 1.0)

        return self._clip_ctrl(ctrl)


class ManipulatorBringBallPD(TaskPDController):
    """PD controller for manipulator-bring_ball: bring ball to target.

    Targets:
        - Ball position at target location (from observation)

    PD gains:
        - kp=10, kd=1

    Output: ctrl covering manipulator actuators.
    """

    def __init__(self, physics) -> None:
        """Initialize ManipulatorBringBallPD with tuned gains.

        Args:
            physics: dm_control Physics instance.
        """
        super().__init__(physics, kp=10.0, kd=1.0)

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute bring-ball PD control for manipulator.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        ctrl: np.ndarray = np.zeros(self.nu)

        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # Extract target and current ball/EE position
        target: np.ndarray = np.zeros(3)
        if 'target' in obs:
            target = np.asarray(obs.get('target', np.zeros(3))).flatten()
        elif 'to_target' in obs:
            target = np.asarray(obs.get('to_target', np.zeros(3))).flatten()

        # PD toward target for each actuator
        for i in range(self.nu):
            error: float = float(target[min(i, len(target) - 1)])
            vel_idx: int = min(i + 1, len(physics.data.qvel) - 1)
            vel: float = float(physics.data.qvel[vel_idx])
            ctrl[i] = np.clip(
                self._pd_scalar(error, -vel, self.kp, self.kd) * 0.05,
                -1.0, 1.0)

        return self._clip_ctrl(ctrl)


# ── Generic Fallback ─────────────────────────────────────────────────


class GenericPDController(TaskPDController):
    """Generic fallback PD controller for tasks without specialized implementations.

    Uses simple proportional control toward a zero-pose target with
    velocity damping. Works reasonably for locomotion tasks where
    maintaining stability is more important than precise goal achievement.

    Attributes:
        target_pose: Target joint angles (default: zeros = neutral pose).
        gait_freq: Oscillation frequency for locomotion-like tasks.
    """

    def __init__(self, physics, kp: float = 10.0, kd: float = 1.0,
                 target_pose: Optional[np.ndarray] = None,
                 gait_freq: float = 2.0) -> None:
        """Initialize GenericPDController.

        Args:
            physics: dm_control Physics instance.
            kp: Proportional gain.
            kd: Derivative gain.
            target_pose: Target joint angles (zeros if None).
            gait_freq: Oscillation frequency for gait pattern.
        """
        super().__init__(physics, kp=kp, kd=kd)
        if target_pose is not None:
            self.target_pose: np.ndarray = target_pose.copy()
        else:
            self.target_pose: np.ndarray = np.zeros(self.nu)
        self.gait_freq: float = gait_freq

    def compute_action(self, timestep, physics) -> np.ndarray:
        """Compute generic PD control: pose stabilization + gait oscillation.

        Strategy:
        1. PD toward target pose (zero pose by default).
        2. Add sinusoidal oscillation for locomotion tasks.
        3. Velocity damping for all joints.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            ctrl clipped to [-1, 1].
        """
        phys = physics
        ctrl: np.ndarray = np.zeros(self.nu)
        step: int = self._increment_step()

        # Determine if task involves locomotion from observation
        obs: dict = {}
        if hasattr(timestep, 'observation') and timestep.observation is not None:
            obs = timestep.observation

        # ── Pose PD + Velocity Damping ──
        for i in range(self.nu):
            # Joint angle: shift by root DOF count
            qpos_idx: int = i + 3  # generic: root has ~3-7 qpos entries
            if qpos_idx < len(phys.data.qpos):
                joint_angle: float = float(phys.data.qpos[qpos_idx])
            else:
                joint_angle: float = 0.0

            target_angle: float = float(self.target_pose[min(i, len(self.target_pose) - 1)])
            error: float = target_angle - joint_angle

            # Velocity damping
            qvel_idx: int = i + 2  # generic: root has ~2-6 qvel entries
            if qvel_idx < len(phys.data.qvel):
                vel: float = float(phys.data.qvel[qvel_idx])
            else:
                vel: float = 0.0

            ctrl[i] = np.clip(
                self._pd_scalar(error, -vel, self.kp, self.kd) * 0.1,
                -1.0, 1.0)

        # ── Gait Oscillation (if locomotion task) ──
        # Check if observation contains velocity-related keys
        is_locomotion: bool = any(
            key in obs for key in ['velocity', 'veloc', 'forward_velocity'])
        if is_locomotion or self.nu > 2:
            phase: float = math.sin(self.gait_freq * step * 0.01)
            for i in range(min(self.nu, 6)):
                ctrl[i] = np.clip(
                    ctrl[i] + phase * 0.15 * (1 if i % 2 == 0 else -1),
                    -1.0, 1.0)

        return self._clip_ctrl(ctrl)

    def compute_safe_action(self, timestep, physics) -> np.ndarray:
        """Safe action: velocity damping on all actuators.

        Args:
            timestep: dm_control TimeStep.
            physics: dm_control Physics instance.

        Returns:
            Safe ctrl array of shape (nu,).
        """
        ctrl: np.ndarray = np.zeros(self.nu)
        for i in range(self.nu):
            qvel_idx: int = min(i + 2, len(physics.data.qvel) - 1)
            vel: float = float(physics.data.qvel[qvel_idx])
            ctrl[i] = np.clip(-0.3 * vel, -0.5, 0.5)
        return self._clip_ctrl(ctrl)


# ── Task Controller Registry ─────────────────────────────────────────


TASK_CONTROLLER_MAP: Dict[str, type] = {
    'humanoid-stand':            HumanoidStandPD,
    'humanoid-walk':             HumanoidWalkPD,
    'humanoid-run':              HumanoidRunPD,
    'reacher-easy':              ReacherTargetPD,
    'reacher-hard':              ReacherTargetPD,
    'walker-stand':              WalkerStandPD,
    'walker-walk':               WalkerWalkPD,
    'walker-run':                WalkerRunPD,
    'hopper-stand':              HopperStandPD,
    'hopper-hop':                HopperHopPD,
    'cheetah-run':               CheetahRunPD,
    'cartpole-balance':          CartpoleBalancePD,
    'cartpole-swingup':          CartpoleSwingupPD,
    'cartpole-balance_sparse':   CartpoleBalancePD,
    'cartpole-swingup_sparse':   CartpoleSwingupPD,
    'fish-swim':                 FishSwimPD,
    'finger-spin':               FingerSpinPD,
    'finger-turn_easy':          FingerTurnEasyPD,
    'finger-turn_hard':          FingerTurnHardPD,
    'ball_in_cup-catch':         BallInCupCatchPD,
    'swimmer-swim6':             SwimmerSwimPD,
    'swimmer-swim15':            SwimmerSwimPD,
    'acrobot-swingup':           AcrobotSwingupPD,
    'pendulum-swingup':          PendulumSwingupPD,
    'manipulator-bring_ball':    ManipulatorBringBallPD,
}


def get_controller_for_task(task_name: str, physics) -> TaskPDController:
    """Factory function: create per-task PD controller by task name.

    Falls back to GenericPDController if task_name not in TASK_CONTROLLER_MAP.

    Args:
        task_name: Task identifier string (e.g., 'humanoid-stand').
        physics: dm_control Physics instance.

    Returns:
        TaskPDController instance for the specified task.
    """
    controller_cls: type = TASK_CONTROLLER_MAP.get(task_name, GenericPDController)
    return controller_cls(physics)
