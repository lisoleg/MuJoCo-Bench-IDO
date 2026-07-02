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

v0.5.0 Upgrade (Phase 1 Motor layer refactor):
  Replaces MotorPrimitives macro selection with per-task PD controllers.
  Each dm_control task gets a TaskPDController that extracts real targets
  from observation and computes goal-directed PD actions with per-task
  KP/KD gains.  MotorPrimitives retained as fallback for pd_stabilize.

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import traceback

from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor
from core.noether_check_mj import noether_check_mj, NoetherViolation
from core.goal_eml_mj import GoalEML
from core.pg_gate import PGGate
from core.kappa_snap_logger import KappaSnapLogger
from agent.psi_anchor import PsiAnchor
from agent.task_pd_controllers import (
    TaskPDController, GenericPDController, get_controller_for_task,
)


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

    Orchestrates κ-Snap residual, Noether conservation gate, per-task PD
    controllers, PD stabilization, stall-detection critique, and oracle
    replay within a single choose_action loop.

    v0.2.0: Optionally integrates ψ-Anchor (meta-management) and
    FlowMatchingEtaPredictor (η trend prediction). These are optional
    and can be set externally (e.g., by SIP-Bench). When present,
    the decision loop is enhanced with:
    - ψ-Anchor dynamic δ_K adjustment (replaces simple stall critique)
    - ψ-Anchor evolution policy for MotorPrimitives
    - ψ-Anchor Noether conservation anchoring
    - FlowMatchingEtaPredictor for forward-looking η computation

    v0.5.0 (Phase 1 Motor layer refactor):
    Replaces MotorPrimitives macro selection with per-task PD controllers.
    Each task gets a dedicated TaskPDController that:
    - Extracts real targets from dm_control observation
    - Computes goal-directed PD actions with per-task KP/KD gains
    - Provides safe-action fallback for Noether violations
    MotorPrimitives is retained for pd_stabilize near-goal refinement.

    Attributes:
        env: dm_control environment instance.
        goal: GoalEML defining the task's invariants and tolerances.
        task_name: Task identifier string (e.g., 'humanoid-stand').
        kappa_thresh: Threshold for η below which PD stabilization activates.
        max_stall: Maximum consecutive stall steps before critique kicks in.
        enable_critique: Whether to apply stall-detection critique.
        stall_count: Current consecutive stall counter.
        prev_data: Previous MuJoCo data for Noether comparison.
        _last_eta: Most recent κ-Snap residual value.
        mp: MotorPrimitives instance (retained for pd_stabilize fallback).
        task_controller: Per-task TaskPDController instance.
        macros: List of (primitive_fn, base_ic_value) from MotorPrimitives.
            (Retained for ψ-Anchor evolution compatibility, but NOT used
            in choose_action decision loop.)
        oracle_buffer: Buffer for oracle demonstration actions.
        psi_anchor: Optional ψ-Anchor meta-management layer instance.
        flow_predictor: Optional FlowMatchingEtaPredictor instance.
    """

    def __init__(self,
                 env,
                 goal_eml: GoalEML,
                 task_name: str = 'humanoid-stand',
                 kappa_thresh: float = 0.05,
                 max_stall: int = 80,
                 enable_critique: bool = True,
                 psi_anchor: Optional[PsiAnchor] = None,
                 flow_predictor: Optional[FlowMatchingEtaPredictor] = None) -> None:
        """Initialize the IDO MuJoCo Agent.

        Args:
            env: dm_control environment.
            goal_eml: GoalEML defining task invariants and tolerances.
            task_name: Task identifier for per-task PD controller selection.
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
        self.task_name: str = task_name
        self.kappa_thresh: float = kappa_thresh
        self.max_stall: int = max_stall
        self.enable_critique: bool = enable_critique

        self.stall_count: int = 0
        self.prev_data = None
        self._last_eta: Optional[float] = None
        self._step_counter: int = 0

        # ── v0.5.2: Creative-Probe (SAI article 章锋2026) ──
        # When η stagnates for probe_stall_threshold consecutive steps,
        # perturb the PD controller's gait parameters (phase offset,
        # gain multiplier) using octonion non-associativity analogy:
        # different bracketing (timing order) of macro sequences →
        # different outcomes. κ-Snap gates whether to accept the
        # perturbation (η must decrease after applying it).
        self._probe_active: bool = False
        self._probe_stall_threshold: int = 30  # steps of η stagnation
        self._probe_eta_history: List[float] = []  # last η values for stagnation detection
        self._probe_perturbation: Optional[np.ndarray] = None
        self._probe_phase_offset: float = 0.0
        self._probe_gain_multiplier: float = 1.0

        # ── v0.5.0: Per-task PD controller (replaces macro selection) ──
        self.task_controller: TaskPDController = get_controller_for_task(
            task_name, env.physics)

        # MotorPrimitives retained for pd_stabilize near-goal fallback
        self.mp = MotorPrimitives(env.physics)

        # macros retained for ψ-Anchor evolution compatibility (NOT used in
        # choose_action decision loop anymore)
        self.macros: List[Tuple[Callable, float]] = self.mp.get_library()
        self.oracle_buffer: List[np.ndarray] = []

        # v0.2.0: ψ-Anchor and FlowMatchingEtaPredictor
        # v0.5.0 fix: Always create PsiAnchor + FlowPredictor by default,
        # so that epiplexity > 0 and η familiarity decay is active even in
        # standard benchmark mode (not just SIP-Bench). Previously, psi_anchor
        # was None in standard runs, causing decay_epi_ratio = 0 → η never decayed.
        self.psi_anchor: PsiAnchor = psi_anchor if psi_anchor is not None else PsiAnchor(goal_eml)
        self.flow_predictor: FlowMatchingEtaPredictor = flow_predictor if flow_predictor is not None else FlowMatchingEtaPredictor()

        # v0.6.0: PG-Gate + KappaSnapLogger for Machine Conscience Audit
        self._pg_gate: PGGate = PGGate()
        self._logger: KappaSnapLogger = KappaSnapLogger()

    # ── v0.5.2: Main body name map for locomotion ee_pos extraction ──
    # For walker/cheetah/hopper tasks without 'to_target' observation,
    # use the torso xpos as ee_pos so that η = ||torso_pos - target_pos||²
    # decreases as the body advances toward the goal.
    _MAIN_BODY_MAP: dict = {
        'walker': 'torso',
        'cheetah': 'torso',
        'humanoid': 'torso',
        'hopper': 'torso',
        'swimmer': 'head',
        'fish': 'tail',
    }

    def _extract_eml_obs(self, physics, timestep=None) -> dict:
        """Extract EML observation dict from dm_control physics state.

        Reads qpos, qvel, end-effector position/velocity, and potential/
        kinetic/total energy from the physics state.

        ee_pos extraction priority (v0.5.2 — PHL-inspired Cartesian fix):
        1. xpos['right_hand'] (humanoid tasks — EE position)
        2. timestep.observation['to_target'] (reacher/fish/manipulator —
           Cartesian distance to target, η = ||to_target||² when target=[0,0])
        3. _MAIN_BODY_MAP[domain] xpos (locomotion: walker/cheetah/hopper —
           actual Cartesian world position of main body, NOT qpos[:3])
        4. xpos['torso'] / xpos['head'] / xpos['pelvis'] (fallback torso)
        5. qpos[:3] (last resort, only for tasks without xpos)

        PHL insight (章锋2026): "物理定律本身就是符号化的规则" —
        ee_pos must be a physically meaningful Cartesian coordinate,
        not a raw qpos vector that mixes meters and radians.

        Args:
            physics: dm_control Physics instance (from env.physics).
            timestep: dm_control TimeStep (optional, used for observation-based extraction).

        Returns:
            Dict with keys: ee_pos, qpos, qvel, E_pot, E_kin, E_total, ee_vel.
        """
        phys = physics
        obs: dict = {}

        # End-effector position — v0.5.2 Cartesian fix (PHL-inspired):
        # ee_pos must be a physically meaningful 3D Cartesian position,
        # so that η = ||ee_pos - target_pos||² is a true distance metric.

        # 1. Humanoid tasks: use right_hand xpos (EE position)
        ee_pos_extracted: bool = False
        try:
            obs['ee_pos'] = phys.named.data.xpos['right_hand', :].copy()
            ee_pos_extracted = True
        except (KeyError, IndexError, AttributeError):
            pass

        if not ee_pos_extracted:
            # 2. Tasks with to_target observation (reacher/fish/manipulator):
            # to_target is Cartesian distance to goal → η = ||to_target||²
            if timestep is not None and hasattr(timestep, 'observation') and timestep.observation is not None:
                to_target = timestep.observation.get('to_target', None)
                if to_target is not None:
                    obs['ee_pos'] = np.array(to_target).flatten()
                    ee_pos_extracted = True

            if not ee_pos_extracted:
                # 3. Locomotion tasks: use _MAIN_BODY_MAP for domain-specific
                # body name. For walker/cheetah/hopper WITHOUT to_target,
                # use the torso xpos as ee_pos so η decreases as body advances.
                # Determine domain from task_name if available
                domain: str = ''
                if hasattr(self, 'task_name') and self.task_name:
                    domain = self.task_name.split('-', 1)[0].lower()
                main_body: str = self._MAIN_BODY_MAP.get(domain, 'torso')

                try:
                    obs['ee_pos'] = phys.named.data.xpos[main_body, :].copy()
                    ee_pos_extracted = True
                except (KeyError, IndexError):
                    # 4. Fallback: try common body names
                    for body_name in ['torso', 'head', 'pelvis']:
                        try:
                            obs['ee_pos'] = phys.named.data.xpos[body_name, :3].copy()
                            ee_pos_extracted = True
                            break
                        except (KeyError, IndexError):
                            continue

                if not ee_pos_extracted:
                    # 5. Fallback: first non-world body
                    try:
                        for bi in range(1, phys.model.nbody):
                            body_xpos = phys.data.xpos[bi, :3].copy()
                            obs['ee_pos'] = body_xpos
                            ee_pos_extracted = True
                            break
                    except (IndexError, AttributeError):
                        pass

                if not ee_pos_extracted:
                    # 6. Last resort: qpos[:3] (may be physically wrong)
                    obs['ee_pos'] = phys.data.qpos[:min(3, len(phys.data.qpos))].copy()

            # If no timestep provided, still try domain-specific body
            if timestep is None and not ee_pos_extracted:
                domain: str = ''
                if hasattr(self, 'task_name') and self.task_name:
                    domain = self.task_name.split('-', 1)[0].lower()
                main_body: str = self._MAIN_BODY_MAP.get(domain, 'torso')
                try:
                    obs['ee_pos'] = phys.named.data.xpos[main_body, :].copy()
                    ee_pos_extracted = True
                except (KeyError, IndexError):
                    for body_name in ['torso', 'head', 'pelvis']:
                        try:
                            obs['ee_pos'] = phys.named.data.xpos[body_name, :3].copy()
                            ee_pos_extracted = True
                            break
                        except (KeyError, IndexError):
                            continue
                    if not ee_pos_extracted:
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

        v0.5.0 fix: noether_check_mj() now returns a dict with keys
        {ok, total, energy, torque, collision, message} instead of a
        (bool, str) tuple. This method adapts the dict to the tuple
        interface expected by choose_action().

        Returns:
            Tuple of (ok: bool, message: str). ok=True means no violations.
        """
        if self.prev_data is None:
            return True, ""
        result: dict = noether_check_mj(self.prev_data,
                                          self.env.physics.data,
                                          self.goal,
                                          collide_thresh=self.goal.collide_thresh)
        return result["ok"], result.get("message", "")

    def choose_action(self, timestep, physics=None) -> np.ndarray:
        """Select control action via IDO decision loop with per-task PD.

        Decision path (v0.5.0 — Motor layer refactor):
        1. Extract EML observation → compute η via κ-Snap (with flow predictor).
        2. Run Noether check → if violation, use task_controller.compute_safe_action()
           (NOT squat anymore — squat was too task-blind).
           v0.2.0: ψ-Anchor injects conservation anchor from Noether result.
        3. v0.2.0: ψ-Anchor updates η history and adjusts δ_K dynamically.
        4. If η < κ_thresh → PD stabilize toward goal (MotorPrimitives.pd_stabilize
           + task_controller for near-goal refinement).
        5. Else → use task_controller.compute_action() for goal-directed action
           (replaces old macro selection from MotorPrimitives).
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
            # v0.5.0: Use task_controller safe action instead of squat
            # squat was too task-blind — only touched ctrl[1]
            ctrl = self.task_controller.compute_safe_action(timestep, phys)
            phys.data.ctrl[:] = ctrl
            self.stall_count += 1
            self.prev_data = phys.data
            self._last_eta = eta
            return ctrl

        self.prev_data = phys.data

        # ── v0.2.0: ψ-Anchor η history update and dynamic δ_K ──
        if self.psi_anchor is not None:
            self.psi_anchor.update_eta_history(eta)
            adjusted_dk: float = self.psi_anchor.adjust_delta_K(self.kappa_thresh)
            self.kappa_thresh = adjusted_dk

            # ψ-Anchor evolution policy — decides when to evolve macros
            # (macros still maintained for ψ-Anchor compatibility even though
            #  they are no longer used in the decision loop)
            evo_policy: str = self.psi_anchor.decide_evolution_policy()
            if self.psi_anchor.should_trigger_evolution():
                self.macros = self.psi_anchor.apply_evolution_to_macros(
                    self.macros, evo_policy)

        # ── Near-Goal: PD Stabilize + Task Controller ──
        if eta < self.kappa_thresh:
            # Blend MotorPrimitives pd_stabilize with task controller
            # for near-goal refinement
            delta = self.mp.pd_stabilize(phys, self.goal.target_pos,
                                         z_i['ee_pos'])
            task_ctrl = self.task_controller.compute_action(timestep, phys)
            # Blend: PD stabilize delta on top of task controller base
            ctrl = np.clip(task_ctrl + delta, -1.0, 1.0)
            phys.data.ctrl[:] = ctrl
            self.stall_count = 0
            self._last_eta = eta
            return ctrl

        # ── Far-Goal: Task PD Controller (replaces macro selection) ──
        # v0.5.0: No longer select from MotorPrimitives macros.
        # Instead, use per-task PD controller for goal-directed action.
        ctrl = self.task_controller.compute_action(timestep, phys)

        # ── v0.5.2: Creative-Probe (SAI 章锋2026) ──
        # When η stagnates (no decrease for probe_stall_threshold steps),
        # Creative-Probe perturbs the gait parameters using octonion
        # non-associativity analogy: (ab)c ≠ a(bc) → different macro
        # sequence bracketings (timings) produce different outcomes.
        # Perturbation: random phase offset, gain multiplier, or action
        # noise. κ-Snap gates acceptance: only keep perturbation if η
        # decreases after applying it for N probe steps.
        self._probe_eta_history.append(eta)
        if len(self._probe_eta_history) > self._probe_stall_threshold:
            self._probe_eta_history = self._probe_eta_history[-self._probe_stall_threshold:]

        # Detect η stagnation: no decrease in last probe_stall_threshold steps
        if len(self._probe_eta_history) >= self._probe_stall_threshold:
            eta_min_recent = min(self._probe_eta_history)
            eta_max_recent = max(self._probe_eta_history)
            # Stagnation: η range is small relative to η magnitude
            stagnation_ratio = (eta_max_recent - eta_min_recent) / max(abs(eta), 1e-6)
            if stagnation_ratio < 0.05 and eta > self.kappa_thresh * 2:
                # η is stagnant AND far from goal → trigger Creative-Probe
                if not self._probe_active:
                    self._probe_active = True
                    # Generate random perturbation (octonion non-associativity
                    # analogy: different bracketings → different timings)
                    self._probe_phase_offset = np.random.uniform(-0.5, 0.5)
                    self._probe_gain_multiplier = np.random.uniform(0.8, 1.3)
                    self._probe_perturbation = np.random.uniform(
                        -0.1, 0.1, size=self.nu if hasattr(self, 'nu') else len(ctrl))

        if self._probe_active:
            # Apply Creative-Probe perturbation to control action
            perturbed_ctrl = ctrl * self._probe_gain_multiplier
            perturbed_ctrl += self._probe_perturbation[:len(ctrl)]
            # Also shift gait phase offset if task_controller has gait_freq
            if hasattr(self.task_controller, 'gait_freq'):
                # Inject phase offset via step counter adjustment
                # (ab)c ≠ a(bc): shifting timing = different bracketing
                self.task_controller._step_offset = (
                    getattr(self.task_controller, '_step_offset', 0)
                    + int(self._probe_phase_offset * 100))
            ctrl = np.clip(perturbed_ctrl, -1.0, 1.0)

            # Check if Creative-Probe reduced η (κ-Snap gate)
            if self._last_eta is not None and eta < self._last_eta * 0.95:
                # η decreased by ≥5% → probe successful, keep perturbation
                # but gradually reduce perturbation magnitude
                self._probe_gain_multiplier = (
                    1.0 + (self._probe_gain_multiplier - 1.0) * 0.5)
                self._probe_perturbation *= 0.5
                # After sufficient η decrease, deactivate probe
                if abs(self._probe_gain_multiplier - 1.0) < 0.05:
                    self._probe_active = False
                    self._probe_perturbation = None
                    self._probe_phase_offset = 0.0
                    self._probe_gain_multiplier = 1.0
                    if hasattr(self.task_controller, '_step_offset'):
                        self.task_controller._step_offset = 0
            elif self._last_eta is not None and eta > self._last_eta * 1.05:
                # η increased → probe failing, reduce perturbation
                self._probe_gain_multiplier = (
                    1.0 + (self._probe_gain_multiplier - 1.0) * 0.3)
                self._probe_perturbation *= 0.3

        phys.data.ctrl[:] = ctrl

        # ── v0.6.0: PG-Gate hard anchor clamp ──
        # PG-Gate runs after action modulation, before ctrl assignment
        pgate_result: np.ndarray = self._pg_gate.gate(ctrl, phys, self._logger)
        phys.data.ctrl[:] = pgate_result

        # ── v0.6.0: ψ-Anchor sentient finger limit check ──
        sentient_result: dict = self.psi_anchor.check_sentient_finger_limit(pgate_result, phys)
        if not sentient_result["ok"]:
            phys.data.ctrl[:] = sentient_result["clamped_action"]
            self._logger.log(
                "FINGER_TORQUE_CLAMPED", "L2", eta, "sentient_clamp",
                details={
                    "joint_name": str(sentient_result.get("violated_indices", [])),
                    "original_torque": str(sentient_result.get("original_torques", {})),
                    "clamped_torque": str(sentient_result.get("clamped_torques", {})),
                },
            )

        # ── v0.6.0: MerkleChain audit recording ──
        final_action = phys.data.ctrl[:].copy()
        self._logger.log("ACTION_ACCEPT", "L0", eta, primary_mode if hasattr(self, '_last_primary_mode') else "IDO")

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

        return ctrl

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
