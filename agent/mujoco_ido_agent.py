"""
IDO/TOMAS MuJoCo Agent  —  Continuous-control IDO Harness (L2 shell)
==============================================================
Inherits the tomas-arc3-solver v7.2 IDO architecture:
  Inflow  → EML_obs (qpos/qvel/ee_pos/energy)  ← dm_control timestep
  κ-Snap  → gauss_ex_residual(z_i, Goal-EML)  →  η
  Noether  → ΔIC ≈ 0  or ↓ only (energy/momentum/self-collision)
  NARLA   → MotorPrimitives (IC-Value gated)
  Oracle   → ExpertDemonstrationReplay (lightweight EML edge rewrite)
  Critique → stall-detect → relax δ_K / shrink amplitude

v0.2.0 Upgrade:
  ψ-Anchor → meta-management layer (dynamic δ_K, evolution policy,
             Noether anchoring, epiplexity)
  FlowMatchingEtaPredictor → forward-looking η trend prediction

v0.2.1 Upgrade (B4 η stagnation fix):
  Step counter for familiarity decay in κ-Snap
  Epiplexity passthrough from ψ-Anchor to gauss_ex_residual

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import traceback

from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor
from core.noether_check_mj import noether_check_mj, NoetherViolation
from core.goal_eml_mj import GoalEML
from agent.psi_anchor import PsiAnchor


class MotorPrimitives:
    """NARLA macro library: motor-level primitives, each with an IC-Value score.

    IDO Light Update: promotes/demotes primitives by ΔIC without touching M_θ.

    Attributes:
        phy: The dm_control physics instance.
        n_joints: Number of joints in the model.
        zero_ctrl: Zero-vector control of dimension nu (number of actuators).
    """

    def __init__(self, physics) -> None:
        """Initialize MotorPrimitives with the dm_control physics engine.

        Args:
            physics: dm_control Physics instance providing model & data.
        """
        self.phy = physics
        self.n_joints: int = physics.model.njnt
        self.zero_ctrl: np.ndarray = np.zeros(physics.model.nu)

    def get_library(self) -> List[Tuple[Callable, float]]:
        """Return ordered list of (primitive_fn, base_ic_value) tuples.

        Returns:
            List of motor primitives paired with their initial IC-Value scores.
        """
        return [
            (self.step_forward,   0.70),
            (self.step_left,      0.65),
            (self.step_right,     0.65),
            (self.squat,          0.50),
            (self.torque_explore, 0.40),
        ]

    def step_forward(self, phy) -> None:
        """Apply forward stepping control.

        Adds +0.3 to the first two joint controls, clipped to [-1, 1].
        Note: Does NOT advance physics; env.step() handles stepping.

        Args:
            phy: dm_control Physics instance.
        """
        ctrl = phy.data.ctrl.copy()
        if self.n_joints >= 2:
            ctrl[:2] = np.clip(ctrl[:2] + 0.3, -1.0, 1.0)
        phy.data.ctrl[:] = ctrl

    def step_left(self, phy) -> None:
        """Apply leftward stepping control.

        Subtracts 0.3 from the first joint control, clipped to [-1, 1].

        Args:
            phy: dm_control Physics instance.
        """
        ctrl = phy.data.ctrl.copy()
        if self.n_joints >= 2:
            ctrl[0] = np.clip(ctrl[0] - 0.3, -1.0, 1.0)
        phy.data.ctrl[:] = ctrl

    def step_right(self, phy) -> None:
        """Apply rightward stepping control.

        Adds 0.3 to the first joint control, clipped to [-1, 1].

        Args:
            phy: dm_control Physics instance.
        """
        ctrl = phy.data.ctrl.copy()
        if self.n_joints >= 2:
            ctrl[0] = np.clip(ctrl[0] + 0.3, -1.0, 1.0)
        phy.data.ctrl[:] = ctrl

    def squat(self, phy) -> None:
        """Apply squatting control.

        Subtracts 0.2 from the second joint control, clipped to [-1, 1].

        Args:
            phy: dm_control Physics instance.
        """
        ctrl = phy.data.ctrl.copy()
        if self.n_joints >= 2:
            ctrl[1] = np.clip(ctrl[1] - 0.2, -1.0, 1.0)
        phy.data.ctrl[:] = ctrl

    def torque_explore(self, phy) -> None:
        """Apply random torque exploration.

        Adds uniform noise in [-0.1, 0.1] to all joint controls.

        Args:
            phy: dm_control Physics instance.
        """
        ctrl = phy.data.ctrl.copy()
        noise = np.random.uniform(-0.1, 0.1, size=self.n_joints)
        ctrl[:self.n_joints] = np.clip(ctrl[:self.n_joints] + noise, -1.0, 1.0)
        phy.data.ctrl[:] = ctrl

    def pd_stabilize(self, phy, target_pos: np.ndarray,
                     ee_pos: np.ndarray) -> np.ndarray:
        """Compute PD-controller delta for end-effector stabilization.

        Uses proportional gain Kp=30, derivative gain Kd=3 to compute
        a control delta steering the end-effector toward target_pos.

        Args:
            phy: dm_control Physics instance.
            target_pos: Desired end-effector position (3-vector).
            ee_pos: Current end-effector position (3-vector).

        Returns:
            Control delta array clipped to [-0.5, 0.5] on the first two
            actuators, zero elsewhere.
        """
        Kp: float = 30.0
        Kd: float = 3.0
        err = target_pos - ee_pos
        delta = Kp * err
        ctrl_delta = np.zeros_like(phy.data.ctrl)
        if self.n_joints >= 2:
            ctrl_delta[:2] = np.clip(delta[:2], -0.5, 0.5)
        return ctrl_delta


class IDOMuJoCoAgent:
    """IDO/TOMAS Self-Referential Manifold Agent for Continuous Control.

    Orchestrates κ-Snap residual, Noether conservation gate, NARLA motor
    primitives, PD stabilization, stall-detection critique, and oracle
    replay within a single choose_action loop.

    v0.2.0: Optionally integrates ψ-Anchor (meta-management) and
    FlowMatchingEtaPredictor (η trend prediction). These are optional
    and can be set externally (e.g., by SIP-Bench). When present,
    the decision loop is enhanced with:
    - ψ-Anchor dynamic δ_K adjustment (replaces simple stall critique)
    - ψ-Anchor evolution policy for MotorPrimitives
    - ψ-Anchor Noether conservation anchoring
    - FlowMatchingEtaPredictor for forward-looking η computation

    Attributes:
        env: dm_control environment instance.
        goal: GoalEML defining the task's invariants and tolerances.
        kappa_thresh: Threshold for η below which PD stabilization activates.
        max_stall: Maximum consecutive stall steps before critique kicks in.
        enable_critique: Whether to apply stall-detection critique.
        stall_count: Current consecutive stall counter.
        prev_data: Previous MuJoCo data for Noether comparison.
        _last_eta: Most recent κ-Snap residual value.
        mp: MotorPrimitives instance.
        macros: List of (primitive_fn, base_ic_value) from MotorPrimitives.
        oracle_buffer: Buffer for oracle demonstration actions.
        psi_anchor: Optional ψ-Anchor meta-management layer instance.
        flow_predictor: Optional FlowMatchingEtaPredictor instance.
    """

    def __init__(self,
                 env,
                 goal_eml: GoalEML,
                 kappa_thresh: float = 0.05,
                 max_stall: int = 80,
                 enable_critique: bool = True,
                 psi_anchor: Optional[PsiAnchor] = None,
                 flow_predictor: Optional[FlowMatchingEtaPredictor] = None) -> None:
        """Initialize the IDO MuJoCo Agent.

        Args:
            env: dm_control environment.
            goal_eml: GoalEML defining task invariants and tolerances.
            kappa_thresh: η threshold for PD stabilization activation.
            max_stall: Max consecutive stall steps before critique relax.
            enable_critique: Whether stall-detection critique is active.
            psi_anchor: Optional PsiAnchor instance for meta-management.
                        If None, can be set later via attribute assignment.
            flow_predictor: Optional FlowMatchingEtaPredictor for η trend.
                           If None, can be set later via attribute assignment.
        """
        self.env = env
        self.goal = goal_eml
        self.kappa_thresh: float = kappa_thresh
        self.max_stall: int = max_stall
        self.enable_critique: bool = enable_critique

        self.stall_count: int = 0
        self.prev_data = None
        self._last_eta: Optional[float] = None
        self._step_counter: int = 0
        self.mp = MotorPrimitives(env.physics)

        self.macros: List[Tuple[Callable, float]] = self.mp.get_library()
        self.oracle_buffer: List[np.ndarray] = []

        # v0.2.0: ψ-Anchor and FlowMatchingEtaPredictor (optional, set externally)
        self.psi_anchor: Optional[PsiAnchor] = psi_anchor
        self.flow_predictor: Optional[FlowMatchingEtaPredictor] = flow_predictor

    def _extract_eml_obs(self, physics, timestep=None) -> dict:
        """Extract EML observation dict from dm_control physics state.

        Reads qpos, qvel, end-effector position/velocity, and potential/
        kinetic/total energy from the physics state.

        Handles task-specific observation differences:
        - humanoid: ee_pos from xpos['right_hand']
        - reacher: ee_pos from timestep.observation['position'] + 'to_target'
        - hopper/walker: ee_pos from qpos[:3] (center-of-mass proxy)

        Args:
            physics: dm_control Physics instance (from env.physics).
            timestep: dm_control TimeStep (optional, used for observation-based extraction).

        Returns:
            Dict with keys: ee_pos, qpos, qvel, E_pot, E_kin, E_total, ee_vel.
        """
        phys = physics
        obs: dict = {}

        # End-effector position — fallback chain:
        # 1. Try xpos['right_hand'] (humanoid)
        # 2. Try timestep.observation for reacher-style tasks
        # 3. Fallback to qpos[:3] (hopper/walker)
        try:
            obs['ee_pos'] = phys.named.data.xpos['right_hand', :].copy()
        except (KeyError, IndexError, AttributeError):
            if timestep is not None and hasattr(timestep, 'observation'):
                # For reacher: to_target gives vector to target
                to_target = timestep.observation.get('to_target', None)
                if to_target is not None:
                    # reacher: position + to_target → ee_pos relative to target
                    pos = timestep.observation.get('position', np.zeros(2))
                    obs['ee_pos'] = np.array(pos)  # 2D ee_pos
                else:
                    obs['ee_pos'] = phys.data.qpos[:3].copy()
            else:
                obs['ee_pos'] = phys.data.qpos[:min(3, len(phys.data.qpos))].copy()

        # Generalized positions and velocities (clipped to model dimensions)
        nq: int = min(phys.model.nq, len(phys.data.qpos))
        obs['qpos'] = phys.data.qpos[:nq].copy()
        obs['qvel'] = phys.data.qvel[:min(phys.model.nv,
                                          len(phys.data.qvel))].copy()

        # Energy components
        obs['E_pot'] = float(phys.data.energy[0])
        obs['E_kin'] = float(phys.data.energy[1])
        obs['E_total'] = obs['E_pot'] + obs['E_kin']

        # End-effector velocity (fallback to zero 6-vector)
        try:
            obs['ee_vel'] = phys.named.data.cvel['right_hand', :].copy()
        except (KeyError, IndexError):
            obs['ee_vel'] = np.zeros(6)

        return obs

    def _compute_kappa_snap(self, z_i: dict, step_index: int = 0) -> float:
        """Compute κ-Snap GaussEx residual η for current observation.

        v0.2.0: If flow_predictor is set, passes it to gauss_ex_residual
        for forward-looking η computation with trend blending.

        v0.2.1: Passes step_index and epiplexity from ψ-Anchor to
        gauss_ex_residual for familiarity decay computation.

        Args:
            z_i: EML observation dict from _extract_eml_obs.
            step_index: Current decision-step index for familiarity decay.

        Returns:
            Scalar residual η measuring deviation from goal manifold.
        """
        epiplexity: float = self.psi_anchor.epiplexity_score if self.psi_anchor else 0.0
        return gauss_ex_residual(z_i, self.goal,
                                 flow_predictor=self.flow_predictor,
                                 step_index=step_index,
                                 epiplexity=epiplexity)

    def _run_noether_check(self) -> Tuple[bool, str]:
        """Run Noether conservation gate between previous and current data.

        Returns:
            Tuple of (ok: bool, message: str). ok=True means no violations.
        """
        if self.prev_data is None:
            return True, ""
        return noether_check_mj(self.prev_data,
                                self.env.physics.data,
                                self.goal)

    def choose_action(self, timestep, physics=None) -> np.ndarray:
        """Select control action via IDO decision loop.

        Decision path:
        1. Extract EML observation → compute η via κ-Snap (with flow predictor).
        2. Run Noether check → if violation, execute squat fallback.
           v0.2.0: ψ-Anchor injects conservation anchor from Noether result.
        3. v0.2.0: ψ-Anchor updates η history and adjusts δ_K dynamically.
        4. If η < κ_thresh → PD stabilize toward goal.
        5. Else → select highest-scoring NARLA motor primitive.
           v0.2.0: ψ-Anchor decides evolution policy for macros.
        6. Critique: if stall detected, relax κ_thresh / increase max_stall.

        Args:
            timestep: dm_control TimeStep (observation, reward, etc).
            physics: dm_control Physics instance (from env.physics).
                If None, attempts timestep.physics (legacy compatibility).

        Returns:
            Control array of shape (nu,) clipped to [-1, 1].
        """
        phys = physics if physics is not None else getattr(timestep, 'physics', None)
        if phys is None:
            raise ValueError("physics must be provided (either via physics arg or timestep.physics)")
        z_i = self._extract_eml_obs(phys, timestep=timestep)

        # ── v0.2.1: Increment step counter for familiarity decay ──
        self._step_counter += 1

        # ── v0.2.0: κ-Snap with flow predictor (forward-looking η) ──
        # v0.2.1: Also passes step_index and epiplexity for familiarity decay
        eta: float = self._compute_kappa_snap(z_i, step_index=self._step_counter)

        # ── Noether Gate ──
        noether_ok, noether_msg = self._run_noether_check()

        # v0.2.0: ψ-Anchor injects conservation anchor
        if self.psi_anchor is not None:
            self.psi_anchor.inject_conservation_anchor(noether_ok, noether_msg)

        if not noether_ok:
            self.mp.squat(phys)
            self.stall_count += 1
            self.prev_data = phys.data
            self._last_eta = eta
            return phys.data.ctrl.copy()

        self.prev_data = phys.data

        # ── v0.2.0: ψ-Anchor η history update and dynamic δ_K ──
        if self.psi_anchor is not None:
            self.psi_anchor.update_eta_history(eta)
            adjusted_dk: float = self.psi_anchor.adjust_delta_K(self.kappa_thresh)
            self.kappa_thresh = adjusted_dk

            # ψ-Anchor evolution policy — decides when to evolve macros
            evo_policy: str = self.psi_anchor.decide_evolution_policy()
            if self.psi_anchor.should_trigger_evolution():
                self.macros = self.psi_anchor.apply_evolution_to_macros(
                    self.macros, evo_policy)

        # ── Near-Goal: PD Stabilize ──
        if eta < self.kappa_thresh:
            delta = self.mp.pd_stabilize(phys, self.goal.target_pos,
                                         z_i['ee_pos'])
            ctrl = phys.data.ctrl.copy()
            ctrl = np.clip(ctrl + delta, -1.0, 1.0)
            phys.data.ctrl[:] = ctrl
            self.stall_count = 0
            self._last_eta = eta
            return ctrl

        # ── Far-Goal: NARLA Motor Primitive Selection ──
        ee_pos: np.ndarray = z_i['ee_pos']
        target: np.ndarray = self.goal.target_pos
        best_macro: Optional[Callable] = None
        best_score: float = -np.inf

        for macro_fn, base_score in self.macros:
            # Align dimensions: pad shorter array to match
            max_d: int = max(len(ee_pos), len(target))
            ee_pad: np.ndarray = np.zeros(max_d)
            ee_pad[:len(ee_pos)] = ee_pos
            tgt_pad: np.ndarray = np.zeros(max_d)
            tgt_pad[:len(target)] = target
            desired = tgt_pad - ee_pad
            score: float = base_score - np.linalg.norm(desired)
            if score > best_score:
                best_score = score
                best_macro = macro_fn

        if best_macro is not None:
            best_macro(phys)

        # ── Critique: Stall Detection ──
        if self.enable_critique and self._last_eta is not None:
            if abs(eta - self._last_eta) < 1e-6:
                self.stall_count += 1
            else:
                self.stall_count = 0
        self._last_eta = eta

        if self.enable_critique and self.stall_count >= self.max_stall:
            self.kappa_thresh *= 1.5
            self.max_stall = int(self.max_stall * 1.2)
            self.stall_count = 0

        return phys.data.ctrl.copy()

    def store_oracle_step(self, action: np.ndarray) -> None:
        """Append an oracle demonstration action to the replay buffer.

        Args:
            action: Control array from an expert demonstration step.
        """
        self.oracle_buffer.append(action.copy())

    def replay_oracle(self, step_idx: int) -> Optional[np.ndarray]:
        """Retrieve an oracle action from the replay buffer by step index.

        Args:
            step_idx: Zero-based step index within the oracle buffer.

        Returns:
            Action array if step_idx is within buffer bounds, else None.
        """
        if step_idx < len(self.oracle_buffer):
            return self.oracle_buffer[step_idx]
        return None
