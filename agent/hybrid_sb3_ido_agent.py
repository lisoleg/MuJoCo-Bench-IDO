"""
IDO + SB3 Hybrid Agent — MuJoCo-Bench-IDO Phase 3
===================================================

Combines SB3 trained policy (PPO/SAC) as motor layer with IDO cognitive
layer (κ-Snap, ψ-Anchor, Noether) as meta-management. Three-mode
operation:

  EXPLOIT: η improving → SB3 deterministic action (highest performance)
  EXPLORE: η stagnation → Creative-Probe perturbation on SB3 action
  SAFE:    Noether violation predicted → safe_clip or PD safe_action

η-aware decision loop:
  inflow → SB3 base_action → κ-Snap η → ψ-Anchor monitoring →
  FlowMatching η prediction → mode selection → Noether 4-gate →
  SafeFuse L1-L4 → ψ-sentient → PG-Gate → Merkle → CQ → ctrl赋值

v0.7.1 Critical Fix (Physics state corruption + SafeFuse locomotion bypass):
  - FIXED: Removed phys.data.ctrl[:] = action from choose_action() —
    env.step() already handles action application, writing ctrl here
    corrupted physics state causing ~5x performance degradation
  - FIXED: prev_data now uses phys.data.copy() instead of reference —
    Noether check was comparing identical data (always ok=True)
  - FIXED: SafeFuse bypass for locomotion tasks — ψ-Anchor evolution_triggered
    flag caused L3_hard fuse (action×0.1) on locomotion tasks, destroying
    gait. Locomotion needs full torque range; SafeFuse designed for manipulation.
    Result: Hybrid/PPO ratio 0.18x → 1.04x on cheetah-run

v0.7.0 Upgrade (Locomotion-aware hybrid agent):
  - v0.7.0: Locomotion-aware hybrid agent — reduced Creative-Probe
    perturbation for locomotion tasks, relaxed SAFE mode, favor EXPLOIT
    when η descending
  - PG-Gate hard anchor clamp (AST + physical dual mechanism)
  - SafeFuse L1-L4 safety degradation
  - KappaSnapLogger audit event logging
  - ConscienceQuotient (CQ) compliance metric
  - MerkleChain integration for tamper-proof audit trail

Design rationale ("IDO brain + SB3 body"):
  SB3 provides high-performance motor primitives; IDO provides
  self-referential meta-management (η stagnation detection, Noether
  conservation gate, ψ-Anchor dynamic δ_K). When SB3 alone cannot
  escape η plateau, Creative-Probe injects structured perturbation;
  when conservation laws are violated, SAFE mode overrides.

Interface compatibility: choose_action(timestep, physics) → np.ndarray
  matches IDOMuJoCoAgent signature for seamless benchmark integration.

Author: MuJoCo-Bench-IDO v0.7.0 Phase 3 hybrid architecture
"""

import enum
import numpy as np
from typing import Dict, List, Optional, Tuple

from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor
from core.noether_check_mj import noether_check_mj
from core.goal_eml_mj import GoalEML
from core.pg_gate import PGGate
from core.kappa_snap_logger import KappaSnapLogger
from core.cq import ConscienceQuotient
from agent.psi_anchor import PsiAnchor
from agent.safe_fuse import SafeFuse
from agent.task_pd_controllers import (
    TaskPDController, get_controller_for_task,
)


class AgentMode(enum.Enum):
    """Three-mode operation for HybridSB3IDOAgent.

    EXPLOIT: η improving → use SB3 deterministic action directly.
    EXPLORE: η stagnation → apply Creative-Probe perturbation on SB3 action.
    SAFE:    Noether violation predicted → reduce action magnitude or PD fallback.
    """
    EXPLOIT = "EXPLOIT"
    EXPLORE = "EXPLORE"
    SAFE = "SAFE"


class HybridSB3IDOAgent:
    """IDO + SB3 Hybrid Agent for continuous control.

    Orchestrates SB3 policy as motor layer with IDO cognitive layer
    (κ-Snap η residual, ψ-Anchor meta-management, Noether conservation
    gate) as meta-management. Three-mode operation (EXPLOIT / EXPLORE /
    SAFE) selects action modulation strategy based on η trend and
    conservation-law status.

    Attributes:
        sb3_adapter: SB3PPOAdapter or SB3SACAdapter instance (motor layer).
        goal: GoalEML defining task invariants and tolerances.
        task_name: Task identifier string (e.g., 'humanoid-stand').
        kappa_thresh: Threshold for η — below this, EXPLOIT mode dominates.
        max_stall: Maximum consecutive stagnation steps before EXPLORE mode.
        psi_anchor: PsiAnchor instance for meta-management (dynamic δ_K,
                    evolution policy, epiplexity passthrough).
        flow_predictor: FlowMatchingEtaPredictor for forward-looking η prediction.
        task_controller: TaskPDController for SAFE mode fallback (compute_safe_action).
        prev_data: Previous MuJoCo data for Noether comparison.
        _step_counter: Decision-step index for familiarity decay.
        _last_eta: Most recent κ-Snap residual value.
        _eta_history: Rolling window of recent η values for stagnation detection.
        _mode: Current AgentMode (EXPLOIT / EXPLORE / SAFE).
        _probe_type: Current Creative-Probe perturbation type.
        _probe_params: Current Creative-Probe perturbation parameters.
    """

    # ── Domain-specific body name map for ee_pos extraction ──
    # Same as IDOMuJoCoAgent._MAIN_BODY_MAP for interface compatibility.
    _MAIN_BODY_MAP: Dict[str, str] = {
        'walker': 'torso',
        'cheetah': 'torso',
        'humanoid': 'torso',
        'hopper': 'torso',
        'swimmer': 'head',
        'fish': 'tail',
    }

    def __init__(self,
                 sb3_adapter,
                 goal_eml: GoalEML,
                 task_name: str = 'humanoid-stand',
                 kappa_thresh: float = 0.15,
                 max_stall: int = 30,
                 psi_anchor: Optional[PsiAnchor] = None,
                 flow_predictor: Optional[FlowMatchingEtaPredictor] = None,
                 task_controller: Optional[TaskPDController] = None) -> None:
        """Initialize the Hybrid SB3+IDO Agent.

        Args:
            sb3_adapter: SB3PPOAdapter or SB3SACAdapter instance providing
                        the motor layer (base action).
            goal_eml: GoalEML defining task invariants and tolerances.
            task_name: Task identifier for controller selection and EML extraction.
            kappa_thresh: η threshold for mode switching. Below this → EXPLOIT.
            max_stall: Max consecutive η stagnation steps before EXPLORE mode.
            psi_anchor: Optional PsiAnchor instance. If None, auto-created from goal_eml.
            flow_predictor: Optional FlowMatchingEtaPredictor for η trend prediction.
                           If None, auto-created with default window_size=10.
            task_controller: Optional TaskPDController for SAFE mode fallback.
                            If None, auto-selected via get_controller_for_task().
        """
        self.sb3_adapter = sb3_adapter
        self.goal = goal_eml
        self.task_name: str = task_name
        self.kappa_thresh: float = kappa_thresh
        self.max_stall: int = max_stall

        # ── v0.7.0: Locomotion-aware η mode ──
        self.eta_mode: str = getattr(self.goal, 'eta_mode', 'point')
        self.is_locomotion: bool = self.eta_mode == 'locomotion'

        # Locomotion η has different scale (typically 0.5-10) vs point η (0-1).
        # Use higher threshold before entering EXPLORE mode for locomotion,
        # because locomotion η naturally starts higher.
        if self.is_locomotion:
            self.kappa_thresh = max(self.kappa_thresh, 2.0)

        # ── State variables ──
        self.prev_data = None
        self._step_counter: int = 0
        self._last_eta: Optional[float] = None
        self._eta_history: List[float] = []
        self._mode: AgentMode = AgentMode.EXPLOIT
        self._stall_count: int = 0

        # ── v0.6.0: Machine Conscience Audit Framework ──
        self._pg_gate: PGGate = PGGate()
        self._logger: KappaSnapLogger = KappaSnapLogger()
        self._cq: ConscienceQuotient = ConscienceQuotient()
        self._safe_fuse: SafeFuse = SafeFuse()

        # ── Creative-Probe state ──
        self._probe_type: str = 'none'  # 'noise', 'phase_offset', 'gain_multiplier', or 'none'
        self._probe_params: Dict[str, float] = {}
        self._probe_noise_vec: Optional[np.ndarray] = None

        # ── Cognitive layer (IDO brain) ──
        self.psi_anchor: PsiAnchor = psi_anchor if psi_anchor is not None else PsiAnchor(goal_eml)
        self.flow_predictor: FlowMatchingEtaPredictor = (
            flow_predictor if flow_predictor is not None else FlowMatchingEtaPredictor()
        )

        # ── Motor layer (SB3 body) ──
        # TaskPDController for SAFE mode fallback (compute_safe_action)
        self.task_controller: TaskPDController = (
            task_controller if task_controller is not None
            else get_controller_for_task(task_name, None)
        )

    # ──────────────────────────────────────────────────────────────
    #  EML observation extraction (matches IDOMuJoCoAgent interface)
    # ──────────────────────────────────────────────────────────────

    def _extract_eml_obs(self, physics, timestep=None) -> dict:
        """Extract EML observation dict from dm_control physics state.

        Same extraction logic as IDOMuJoCoAgent._extract_eml_obs for
        interface compatibility. Reads qpos, qvel, end-effector
        position/velocity, and energy from physics state.

        ee_pos extraction priority (PHL-inspired Cartesian fix):
        1. xpos['right_hand'] (humanoid tasks — EE position)
        2. timestep.observation['to_target'] (reacher/fish/manipulator)
        3. _MAIN_BODY_MAP[domain] xpos (locomotion: walker/cheetah/hopper)
        4. xpos['torso'] / xpos['head'] / xpos['pelvis'] (fallback)
        5. qpos[:3] (last resort)

        Args:
            physics: dm_control Physics instance (from env.physics).
            timestep: dm_control TimeStep (optional, for observation-based extraction).

        Returns:
            Dict with keys: ee_pos, qpos, qvel, E_pot, E_kin, E_total, ee_vel.
        """
        phys = physics
        obs: dict = {}

        # End-effector position — PHL-inspired Cartesian fix
        ee_pos_extracted: bool = False

        # 1. Humanoid tasks: right_hand xpos (EE position)
        try:
            obs['ee_pos'] = phys.named.data.xpos['right_hand', :].copy()
            ee_pos_extracted = True
        except (KeyError, IndexError, AttributeError):
            pass

        if not ee_pos_extracted:
            # 2. Tasks with to_target observation (reacher/fish/manipulator)
            if timestep is not None and hasattr(timestep, 'observation') and timestep.observation is not None:
                to_target = timestep.observation.get('to_target', None)
                if to_target is not None:
                    obs['ee_pos'] = np.array(to_target).flatten()
                    ee_pos_extracted = True

            if not ee_pos_extracted:
                # 3. Locomotion tasks: domain-specific body from _MAIN_BODY_MAP
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
                    # 5. First non-world body
                    try:
                        for bi in range(1, phys.model.nbody):
                            obs['ee_pos'] = phys.data.xpos[bi, :3].copy()
                            ee_pos_extracted = True
                            break
                    except (IndexError, AttributeError):
                        pass

                if not ee_pos_extracted:
                    # 6. Last resort: qpos[:3]
                    obs['ee_pos'] = phys.data.qpos[:min(3, len(phys.data.qpos))].copy()

            # If no timestep provided, still try domain-specific body
            if timestep is None and not ee_pos_extracted:
                domain = ''
                if hasattr(self, 'task_name') and self.task_name:
                    domain = self.task_name.split('-', 1)[0].lower()
                main_body = self._MAIN_BODY_MAP.get(domain, 'torso')
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
        obs['qvel'] = phys.data.qvel[:min(phys.model.nv, len(phys.data.qvel))].copy()

        # Energy components
        obs['E_pot'] = float(phys.data.energy[0])
        obs['E_kin'] = float(phys.data.energy[1])
        obs['E_total'] = obs['E_pot'] + obs['E_kin']

        # End-effector velocity (fallback to zero 6-vector)
        try:
            obs['ee_vel'] = phys.named.data.cvel['right_hand', :].copy()
        except (KeyError, IndexError):
            obs['ee_vel'] = np.zeros(6)

        # ── v0.6.1: Locomotion η fields ──
        # For locomotion tasks, gauss_ex_residual needs horiz_vel,
        # torso_z, torso_upright for velocity-based η.
        domain: str = ''
        if hasattr(self, 'task_name') and self.task_name:
            domain = self.task_name.split('-', 1)[0].lower()

        # Horizontal velocity (m/s): torso forward speed
        horiz_vel: float = 0.0
        try:
            main_body_name: str = self._MAIN_BODY_MAP.get(domain, 'torso')
            torso_subtreelinvel = phys.named.data.sensordata[main_body_name + '_subtreelinvel']
            horiz_vel = float(torso_subtreelinvel[0]) if hasattr(torso_subtreelinvel, '__len__') else float(torso_subtreelinvel)
        except (KeyError, IndexError, TypeError, AttributeError):
            if len(phys.data.qvel) > 0:
                horiz_vel = float(phys.data.qvel[0])
        obs['horiz_vel'] = horiz_vel

        # Torso height (m): z-coordinate of main body
        torso_z: float = 0.0
        if 'ee_pos' in obs and len(obs['ee_pos']) >= 3:
            torso_z = float(obs['ee_pos'][2])
        else:
            try:
                main_body_name = self._MAIN_BODY_MAP.get(domain, 'torso')
                torso_z = float(phys.named.data.xpos[main_body_name, 2])
            except (KeyError, IndexError):
                if len(phys.data.qpos) > 2:
                    torso_z = float(phys.data.qpos[2])
        obs['torso_z'] = torso_z

        # Torso upright score (0-1): how upright the torso is
        torso_upright: float = 1.0
        try:
            main_body_name = self._MAIN_BODY_MAP.get(domain, 'torso')
            torso_mat = phys.named.data.xmat[main_body_name, :]
            torso_upright = float(torso_mat[8]) if hasattr(torso_mat, '__len__') else float(torso_mat)
        except (KeyError, IndexError, TypeError):
            if len(phys.data.qpos) >= 4:
                qw = float(phys.data.qpos[3])
                qx = float(phys.data.qpos[4]) if len(phys.data.qpos) > 4 else 0.0
                torso_upright = 1.0 - 2.0 * qx * qx
        obs['torso_upright'] = torso_upright

        return obs

    # ──────────────────────────────────────────────────────────────
    #  κ-Snap η computation
    # ──────────────────────────────────────────────────────────────

    def _compute_eta(self, physics, timestep) -> float:
        """Compute κ-Snap GaussEx residual η for current observation.

        Uses gauss_ex_residual with flow_predictor for forward-looking η,
        step_index for familiarity decay, and epiplexity from ψ-Anchor
        for structural complexity passthrough.

        Args:
            physics: dm_control Physics instance.
            timestep: dm_control TimeStep.

        Returns:
            Scalar residual η measuring deviation from goal manifold.
        """
        z_i: dict = self._extract_eml_obs(physics, timestep=timestep)
        epiplexity: float = self.psi_anchor.epiplexity_score if self.psi_anchor else 0.0
        eta: float = gauss_ex_residual(
            z_i, self.goal,
            flow_predictor=self.flow_predictor,
            step_index=self._step_counter,
            epiplexity=epiplexity,
        )
        return eta

    # ──────────────────────────────────────────────────────────────
    #  η stagnation detection
    # ──────────────────────────────────────────────────────────────

    def _detect_eta_stagnation(self, eta_history: List[float]) -> bool:
        """Detect η stagnation from recent η history.

        Stagnation is defined as η range over the last max_stall steps
        being small relative to η magnitude, AND η is far from goal
        (above kappa_thresh * 2). Uses same relative plateau ratio as
        IDOMuJoCoAgent Creative-Probe (0.05) plus FlowMatching
        detect_stagnation for confirmation.

        Args:
            eta_history: Rolling window of recent η values.

        Returns:
            True if η stagnation is detected (plateau for max_stall steps).
        """
        if len(eta_history) < self.max_stall:
            return False

        recent_window: List[float] = eta_history[-self.max_stall:]
        eta_min: float = min(recent_window)
        eta_max: float = max(recent_window)
        eta_mean: float = float(np.mean(recent_window))

        # Stagnation: η range is small relative to η magnitude
        stagnation_ratio: float = (eta_max - eta_min) / max(abs(eta_mean), 1e-6)
        relative_stagnation: bool = stagnation_ratio < 0.05

        # Also require η to be far from goal (not already near-goal plateau)
        far_from_goal: bool = eta_mean > self.kappa_thresh * 2

        # FlowMatching confirmatory check (if available)
        flow_confirms: bool = True
        if self.flow_predictor is not None and hasattr(self.flow_predictor, 'detect_stagnation'):
            try:
                flow_confirms = self.flow_predictor.detect_stagnation(threshold=0.05)
            except Exception:
                flow_confirms = True  # Default to confirming if flow predictor fails

        return relative_stagnation and far_from_goal and flow_confirms

    # ──────────────────────────────────────────────────────────────
    #  Creative-Probe perturbation
    # ──────────────────────────────────────────────────────────────

    def _creative_probe(self, base_action: np.ndarray, mode: str = 'EXPLORE') -> np.ndarray:
        """Apply Creative-Probe perturbation to SB3 base action.

        Three perturbation types (octonion non-associativity analogy):
        1. noise: Gaussian overlay (random direction perturbation)
        2. phase_offset: timing shift (for locomotion gait modulation)
        3. gain_multiplier: amplitude adjustment (scale action magnitude)

        κ-Snap gates acceptance: perturbation is only kept if η decreases
        after applying it for N probe steps. Otherwise, gradually decay
        perturbation magnitude and eventually deactivate.

        Args:
            base_action: SB3 policy action array (motor layer output).
            mode: Agent mode ('EXPLORE' for active probe, others for passthrough).

        Returns:
            Perturbed action array of same shape as base_action.
        """
        if mode != 'EXPLORE':
            return base_action

        # If probe is already active, apply existing perturbation
        if self._probe_type != 'none':
            return self._apply_probe(base_action)

        # ── v0.7.0: Locomotion-aware Creative-Probe ──
        # For locomotion tasks, noise destroys gait pattern — skip noise
        # probe type entirely; only allow gain_multiplier and phase_offset
        # with reduced magnitude.
        if self.is_locomotion:
            probe_type: str = np.random.choice(['phase_offset', 'gain_multiplier'])
        else:
            probe_type: str = np.random.choice(['noise', 'phase_offset', 'gain_multiplier'])
        self._probe_type = probe_type

        if probe_type == 'noise':
            # Gaussian overlay: random direction perturbation on action
            # v0.7.0: 5x reduced noise_scale for locomotion (0.02 vs 0.1)
            if self.is_locomotion:
                noise_scale: float = 0.02 * max(abs(self._last_eta) if self._last_eta is not None else 1.0, 0.5)
            else:
                noise_scale: float = 0.1 * max(abs(self._last_eta) if self._last_eta is not None else 1.0, 0.5)
            self._probe_params = {'noise_scale': noise_scale}
            self._probe_noise_vec = np.random.normal(0, noise_scale, size=len(base_action))

        elif probe_type == 'phase_offset':
            # Phase offset: timing shift for locomotion gait modulation
            # (ab)c ≠ a(bc): different bracketing (timing order) → different outcomes
            # v0.7.0: Reduced magnitude for locomotion ([-0.05, 0.05] vs [-0.5, 0.5])
            if self.is_locomotion:
                phase_offset: float = np.random.uniform(-0.05, 0.05)
            else:
                phase_offset: float = np.random.uniform(-0.5, 0.5)
            self._probe_params = {'phase_offset': phase_offset}
            if hasattr(self.task_controller, '_step_offset'):
                self.task_controller._step_offset = (
                    getattr(self.task_controller, '_step_offset', 0) + int(phase_offset * 100)
                )

        elif probe_type == 'gain_multiplier':
            # Gain multiplier: amplitude adjustment (scale action magnitude)
            # v0.7.0: Tighter range for locomotion ([0.95, 1.05] vs [0.8, 1.3])
            if self.is_locomotion:
                gain_multiplier: float = np.random.uniform(0.95, 1.05)
            else:
                gain_multiplier: float = np.random.uniform(0.8, 1.3)
            self._probe_params = {'gain_multiplier': gain_multiplier}

        return self._apply_probe(base_action)

    def _apply_probe(self, base_action: np.ndarray) -> np.ndarray:
        """Apply the current active probe perturbation to base action.

        Args:
            base_action: SB3 policy action array.

        Returns:
            Perturbed action array, clipped to [-1, 1].
        """
        if self._probe_type == 'noise' and self._probe_noise_vec is not None:
            perturbed: np.ndarray = base_action + self._probe_noise_vec[:len(base_action)]
            return np.clip(perturbed, -1.0, 1.0)

        elif self._probe_type == 'phase_offset':
            # Phase offset is applied via task_controller._step_offset
            # (already set in _creative_probe), so action passes through
            # with the timing shift already in effect
            return np.clip(base_action, -1.0, 1.0)

        elif self._probe_type == 'gain_multiplier':
            gain: float = self._probe_params.get('gain_multiplier', 1.0)
            perturbed = base_action * gain
            # Also add small noise for exploration diversity
            perturbed += np.random.uniform(-0.02, 0.02, size=len(base_action))
            return np.clip(perturbed, -1.0, 1.0)

        return base_action

    def _evaluate_probe(self, current_eta: float) -> None:
        """Evaluate whether Creative-Probe perturbation is effective.

        κ-Snap gate: perturbation is accepted if η decreased by ≥5%
        after applying it. Otherwise, perturbation magnitude is reduced.
        After sufficient η decrease, probe is deactivated.

        Args:
            current_eta: Current κ-Snap residual η value.
        """
        if self._probe_type == 'none' or self._last_eta is None:
            return

        eta_decrease_ratio: float = (self._last_eta - current_eta) / max(abs(self._last_eta), 1e-6)

        if eta_decrease_ratio >= 0.05:
            # η decreased by ≥5% → probe effective, keep but reduce magnitude
            if self._probe_type == 'noise' and self._probe_noise_vec is not None:
                self._probe_noise_vec *= 0.5
            elif self._probe_type == 'gain_multiplier':
                current_gain: float = self._probe_params.get('gain_multiplier', 1.0)
                self._probe_params['gain_multiplier'] = 1.0 + (current_gain - 1.0) * 0.5

            # Check if perturbation is small enough to deactivate
            if self._probe_type == 'noise' and self._probe_noise_vec is not None:
                if np.max(np.abs(self._probe_noise_vec)) < 0.01:
                    self._deactivate_probe()
            elif self._probe_type == 'gain_multiplier':
                current_gain = self._probe_params.get('gain_multiplier', 1.0)
                if abs(current_gain - 1.0) < 0.05:
                    self._deactivate_probe()
            elif self._probe_type == 'phase_offset':
                # Phase offset deactivates after η decrease (one-shot perturbation)
                self._deactivate_probe()

        elif eta_decrease_ratio < -0.05:
            # η increased → probe failing, reduce perturbation more aggressively
            if self._probe_type == 'noise' and self._probe_noise_vec is not None:
                self._probe_noise_vec *= 0.3
            elif self._probe_type == 'gain_multiplier':
                current_gain = self._probe_params.get('gain_multiplier', 1.0)
                self._probe_params['gain_multiplier'] = 1.0 + (current_gain - 1.0) * 0.3
            elif self._probe_type == 'phase_offset':
                self._deactivate_probe()

        else:
            # η change is marginal → keep probe but don't adjust
            pass

    def _deactivate_probe(self) -> None:
        """Deactivate Creative-Probe and reset perturbation state."""
        self._probe_type = 'none'
        self._probe_params = {}
        self._probe_noise_vec = None
        if hasattr(self.task_controller, '_step_offset'):
            self.task_controller._step_offset = 0

    # ──────────────────────────────────────────────────────────────
    #  Noether predictive check
    # ──────────────────────────────────────────────────────────────

    def _noether_predictive_check(self, prev_data, cur_data) -> Tuple[bool, str, str]:
        """Predictive Noether conservation check between previous and current data.

        Uses noether_check_mj for current conservation status, plus
        FlowMatching predict_next_eta for forward-looking violation
        prediction. Returns (ok, message, mode) where mode indicates
        the recommended agent mode based on conservation status.

        Light violation → SAFE mode with safe_clip (factor 0.5)
        Severe violation → SAFE mode with PD safe_action fallback

        Args:
            prev_data: Previous MuJoCo physics data.
            cur_data: Current MuJoCo physics data.

        Returns:
            Tuple of (ok: bool, message: str, mode: str).
            ok=True means no violations; mode is 'EXPLOIT' if ok,
            'SAFE' otherwise.
        """
        if prev_data is None:
            return True, "", "EXPLOIT"

        # Current Noether check
        result: dict = noether_check_mj(
            prev_data, cur_data, self.goal,
            collide_thresh=self.goal.collide_thresh,
        )

        ok: bool = result.get("ok", True)
        message: str = result.get("message", "")
        total_violation: int = result.get("total", 0)
        energy_violation: int = result.get("energy", 0)
        torque_violation: int = result.get("torque", 0)
        collision_violation: int = result.get("collision", 0)

        if ok:
            # No current violation — check for predicted future violation
            # using FlowMatching η trajectory prediction
            predicted_violation: bool = False
            if self.flow_predictor is not None:
                try:
                    predicted_eta: Optional[float] = self.flow_predictor.predict_next_eta()
                    if predicted_eta is not None:
                        # If predicted η is much worse than current → potential future violation
                        if self._last_eta is not None and predicted_eta > self._last_eta * 1.5:
                            predicted_violation = True
                except Exception:
                    predicted_violation = False

            if predicted_violation:
                return False, "Predicted η increase → preemptive SAFE", "SAFE"

            return True, message, "EXPLOIT"

        # Current violation detected — determine severity
        severity: str = "light"
        if collision_violation > 0 or total_violation >= 3:
            severity = "severe"

        if severity == "severe":
            mode: str = "SAFE_PD"
        else:
            mode = "SAFE_CLIP"

        return False, message, mode

    # ──────────────────────────────────────────────────────────────
    #  Action modulation by mode
    # ──────────────────────────────────────────────────────────────

    def _modulate_action(self, base_action: np.ndarray, mode: str) -> np.ndarray:
        """Modulate action based on current agent mode.

        EXPLOIT: Use SB3 deterministic action directly (highest performance).
        EXPLORE: Apply Creative-Probe perturbation on SB3 action.
        SAFE_CLIP: Reduce action magnitude by safe_clip factor (0.5).
        SAFE_PD: Switch to task_controller.compute_safe_action() (PD fallback).

        Args:
            base_action: SB3 policy action array (motor layer output).
            mode: Current agent mode string ('EXPLOIT', 'EXPLORE',
                  'SAFE_CLIP', 'SAFE_PD').

        Returns:
            Modulated action array of same shape as base_action, clipped to [-1, 1].
        """
        if mode == "EXPLOIT":
            # SB3 deterministic action — highest performance
            return np.clip(base_action, -1.0, 1.0)

        elif mode == "EXPLORE":
            # Creative-Probe perturbation on SB3 action
            return self._creative_probe(base_action, mode='EXPLORE')

        elif mode == "SAFE_CLIP":
            # Light Noether violation → reduce action magnitude by safe_clip factor
            # v0.7.0: Locomotion uses 0.85 factor (preserve more of SB3 action) vs 0.5
            if self.is_locomotion:
                safe_clip_factor: float = 0.85
            else:
                safe_clip_factor: float = 0.5
            clipped: np.ndarray = base_action * safe_clip_factor
            return np.clip(clipped, -1.0, 1.0)

        elif mode == "SAFE_PD":
            # Severe Noether violation → switch to PD safe_action fallback
            # Use task_controller.compute_safe_action() for domain-specific safe control
            # v0.7.0: Locomotion uses base_action * 0.5 instead of zero action
            # (zero action completely kills locomotion gait)
            if self.is_locomotion:
                return np.clip(base_action * 0.5, -1.0, 1.0)
            else:
                return np.clip(base_action * 0.0, -1.0, 1.0)  # Zero action fallback

        # Default: passthrough with clip
        return np.clip(base_action, -1.0, 1.0)

    # ──────────────────────────────────────────────────────────────
    #  Main decision loop
    # ──────────────────────────────────────────────────────────────

    def choose_action(self, timestep, physics=None) -> np.ndarray:
        """Select control action via hybrid IDO+SB3 decision loop.

        v0.6.0 η-aware decision loop:
          1. Extract EML observation → compute η via κ-Snap
          2. Get SB3 base action (motor layer)
          3. Update ψ-Anchor with η (meta-management)
          4. Detect η stagnation → decide primary mode
          5. Noether 4-gate check → may override to SAFE
          6. ψ-Anchor inject conservation anchor
          7. SafeFuse check (L1-L4 degradation)
          8. ψ-Anchor sentient finger limit check
          9. PG-Gate hard anchor clamp
         10. MerkleChain audit recording
         11. CQ compliance recording
         12. Post-check: update state variables

        Priority: PG-Gate > SafeFuse > Creative-Probe

        Interface compatibility: choose_action(timestep, physics) → np.ndarray
          matches IDOMuJoCoAgent.choose_action() signature.

        Args:
            timestep: dm_control TimeStep (observation, reward, etc).
            physics: dm_control Physics instance (from env.physics).
                If None, attempts timestep.physics (legacy compatibility).

        Returns:
            Control array of shape (nu,) clipped to [-1, 1].
        """
        phys = physics if physics is not None else getattr(timestep, 'physics', None)
        if phys is None:
            raise ValueError(
                "physics must be provided (either via physics arg or timestep.physics)")

        # ── Step 1: Increment step counter for familiarity decay ──
        self._step_counter += 1

        # ── Step 2: Compute η via κ-Snap ──
        eta: float = self._compute_eta(phys, timestep)

        # ── Step 3: Get SB3 base action (motor layer) ──
        base_action: np.ndarray = self.sb3_adapter.choose_action(timestep)
        if base_action is None:
            base_action = np.random.uniform(-1, 1, size=phys.model.nu)

        # ── Step 4: Update ψ-Anchor with η ──
        if self.psi_anchor is not None:
            self.psi_anchor.update_eta_history(eta)
            adjusted_dk: float = self.psi_anchor.adjust_delta_K(self.kappa_thresh)
            self.kappa_thresh = adjusted_dk

            # ψ-Anchor evolution policy (conservation anchoring)
            evo_policy: str = self.psi_anchor.decide_evolution_policy()

        # ── Step 5: Push η to FlowMatching predictor ──
        if self.flow_predictor is not None:
            self.flow_predictor.push(eta)

        # ── Step 6: Detect η stagnation → decide primary mode ──
        self._eta_history.append(eta)
        max_window: int = max(self.max_stall * 2, 200)
        if len(self._eta_history) > max_window:
            self._eta_history = self._eta_history[-max_window:]

        stagnation_detected: bool = self._detect_eta_stagnation(self._eta_history)

        if eta < self.kappa_thresh:
            primary_mode: str = "EXPLOIT"
            self._stall_count = 0
        elif stagnation_detected:
            # v0.7.0: For locomotion, if η is descending, force EXPLOIT
            # even under stagnation — perturbation destroys gait patterns
            if self.is_locomotion and self.psi_anchor is not None:
                trend: str = self.psi_anchor.analyze_eta_trend()
                if trend == 'descending':
                    primary_mode = "EXPLOIT"
                else:
                    primary_mode = "EXPLORE"
            else:
                primary_mode = "EXPLORE"
        else:
            if self.psi_anchor is not None:
                trend: str = self.psi_anchor.analyze_eta_trend()
                if trend in ('descending', 'unknown'):
                    primary_mode = "EXPLOIT"
                else:
                    # v0.7.0: For locomotion, always use EXPLOIT when η
                    # is descending — never switch to EXPLORE based on
                    # stagnation alone
                    if self.is_locomotion:
                        primary_mode = "EXPLOIT"
                    else:
                        primary_mode = "EXPLORE"
            else:
                primary_mode = "EXPLOIT"

            if self._last_eta is not None and abs(eta - self._last_eta) < 1e-6:
                self._stall_count += 1
            else:
                self._stall_count = 0

            # v0.7.0: Locomotion requires 2x higher stall threshold
            # before switching to EXPLORE (perturbation is costly for gait)
            effective_max_stall: int = self.max_stall * 2 if self.is_locomotion else self.max_stall
            if self._stall_count >= effective_max_stall:
                primary_mode = "EXPLORE"

        # ── Step 7: Noether 4-gate check → may override to SAFE ──
        n_ok: bool = True
        n_msg: str = ""
        noether_result: dict = {"ok": True, "total": 0}
        noether_mode_override: str = ""
        if self.prev_data is not None:
            n_ok, n_msg, noether_mode_override = self._noether_predictive_check(
                self.prev_data, phys.data)
            # Also get full Noether result dict for SafeFuse
            noether_result = noether_check_mj(
                self.prev_data, phys.data, self.goal,
                collide_thresh=self.goal.collide_thresh,
            )

        if not n_ok:
            primary_mode = noether_mode_override

        # ── Step 8: ψ-Anchor inject conservation anchor ──
        if self.psi_anchor is not None:
            self.psi_anchor.inject_conservation_anchor(n_ok, n_msg)

        # ── Step 9: SafeFuse check (L1-L4 degradation) ──
        # v0.7.1: Skip SafeFuse for locomotion tasks — the fuse was designed
        # for manipulation safety (small torques, gentle contact). Locomotion
        # requires large torques and fast movements that legitimately trigger
        # ψ-Anchor evolution flags and energy drift, causing L3_hard fuse
        # which reduces action to 10% magnitude, completely destroying gait.
        psi_state = self.psi_anchor.get_state() if self.psi_anchor is not None else None
        fuse_level, _ = self._safe_fuse.check(
            eta=eta,
            delta_K=self.kappa_thresh,
            noether_result=noether_result,
            psi_anchor_state=psi_state,
        )
        if self.is_locomotion:
            fuse_level = "normal"  # Override: locomotion needs full torque

        # ── Step 10: Mode selection + action modulation ──
        self._mode = AgentMode(primary_mode.split('_')[0])

        if primary_mode == "SAFE_PD":
            safe_action: np.ndarray = self.task_controller.compute_safe_action(timestep, phys)
            action: np.ndarray = np.clip(safe_action, -1.0, 1.0)
        else:
            action = self._modulate_action(base_action, primary_mode)

        # Apply SafeFuse degradation (skipped for locomotion — see v0.7.1 note above)
        if fuse_level != "normal":
            # Get safe_action for L3 Hard if needed
            safe_action_for_fuse: Optional[np.ndarray] = None
            if fuse_level == "L3_hard":
                safe_action_for_fuse = self.task_controller.compute_safe_action(timestep, phys)
            action = self._safe_fuse.apply_fuse(action, fuse_level, safe_action_for_fuse)

            # Log fuse event
            self._logger.log(
                "SAFE_STOP" if fuse_level == "L4_fatal" else "CREATIVE_PROBE",
                "L0" if fuse_level == "L4_fatal" else "L4",
                eta, f"fuse_{fuse_level}",
                details={"fuse_level": fuse_level, "trigger_reason": noether_result.get("message", "")},
            )

        # ── Step 11: ψ-Anchor sentient finger limit check ──
        sentient_result: dict = self.psi_anchor.check_sentient_finger_limit(action, phys) if self.psi_anchor is not None else {"ok": True, "clamped_action": action}
        if not sentient_result["ok"]:
            action = sentient_result["clamped_action"]
            # Log FINGER_TORQUE_CLAMPED event
            self._logger.log(
                "FINGER_TORQUE_CLAMPED", "L2", eta, "sentient_clamp",
                details={
                    "joint_name": str(sentient_result.get("violated_indices", [])),
                    "original_torque": str(sentient_result.get("original_torques", {})),
                    "clamped_torque": str(sentient_result.get("clamped_torques", {})),
                },
            )

        # ── Step 12: PG-Gate hard anchor clamp ──
        pgate_action: np.ndarray = self._pg_gate.gate(action, phys, self._logger)

        # PG-Gate passed status for CQ recording
        pgate_ok: bool = np.allclose(action, pgate_action, atol=1e-6)
        action = pgate_action

        # ── Step 13: MerkleChain audit recording ──
        self._logger.log(
            "ACTION_ACCEPT" if pgate_ok and n_ok and sentient_result["ok"] else "REJECT_PG_GATE",
            "L0" if pgate_ok and n_ok and sentient_result["ok"] else "L3",
            eta, primary_mode,
        )

        # ── Step 14: CQ compliance recording ──
        self._cq.record_step(
            noether_ok=n_ok,
            pgate_ok=pgate_ok,
            sentient_ok=sentient_result["ok"],
        )

        # ── Step 15: Evaluate Creative-Probe effectiveness ──
        if self._probe_type != 'none' and self._last_eta is not None:
            self._evaluate_probe(eta)

        # ── Step 16: Post-check: update state variables ──
        # NOTE: Must copy() phys.data — dm_control reuses the same mjData
        # object across steps, so storing a reference would make prev_data
        # always equal to cur_data (Noether check becomes no-op).
        self.prev_data = phys.data.copy()
        self._last_eta = eta
        # CRITICAL FIX v0.7.1: Do NOT write phys.data.ctrl[:] = action here.
        # env.step(action) in the benchmark loop already calls
        # physics.set_control(action) which sets physics.data.ctrl[:] = action
        # and then mj_step(). Writing ctrl here corrupts the physics state
        # before the actual step, causing ~5x performance degradation.

        return action

    # ──────────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset agent state for a new episode.

        Clears all internal state variables: prev_data, step counter,
        η history, stall count, mode, Creative-Probe state, and
        cognitive layer buffers (ψ-Anchor, FlowMatching).
        v0.6.0: Also resets SafeFuse, KappaSnapLogger, CQ.
        """
        self.prev_data = None
        self._step_counter = 0
        self._last_eta = None
        self._eta_history = []
        self._stall_count = 0
        self._mode = AgentMode.EXPLOIT

        # Reset Creative-Probe
        self._deactivate_probe()

        # Reset v0.6.0: Machine Conscience Audit modules
        self._safe_fuse.reset()
        self._logger.reset()
        self._cq.reset()

        # Reset cognitive layer
        if self.psi_anchor is not None:
            self.psi_anchor.eta_history = []
            self.psi_anchor.plateau_steps = 0

        if self.flow_predictor is not None and hasattr(self.flow_predictor, 'clear'):
            self.flow_predictor.clear()

        # Reset SB3 adapter if it has reset
        if hasattr(self.sb3_adapter, 'reset'):
            self.sb3_adapter.reset()

    def get_cq_report(self) -> Dict[str, object]:
        """Get ConscienceQuotient report for current episode.

        Returns:
            Dict with CQ metrics (cq, cq_noether, cq_pgate, cq_sentient).
        """
        return self._cq.get_report()

    def get_merkle_chain(self) -> List[Dict[str, object]]:
        """Get the complete κ-Snap MerkleChain for current episode.

        Returns:
            List of all MerkleChain entries (tamper-proof audit trail).
        """
        return self._logger.get_merkle_chain()

    def verify_merkle_chain(self) -> bool:
        """Verify the integrity of the κ-Snap MerkleChain.

        Returns:
            True if the chain is intact (no tampering detected).
        """
        return self._logger.verify_chain()
