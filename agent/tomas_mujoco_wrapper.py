"""
TOMAS MuJoCo Wrapper — IDO Three-Layer Integration
====================================================

Integrates the three IDO layers into a single MuJoCo environment wrapper:

  P-Layer (Phenomenon): VLA Policy / Motor Primitives → raw action
  C-Layer (Constraint): ψ-Anchor + PG-Gate → hard physical constraint
  S-Layer (Self-Audit): κ-Snap Logger → causal snapshot audit trail

The wrapper intercepts every env.step() to:
  1. Query the VLA policy for a raw action (P-Layer)
  2. Pass through PG-Gate for sentient-body torque clamp (C-Layer)
  3. Pass through ψ-Anchor for conservation/meta-management check (C-Layer)
  4. Log the complete decision to κ-Snap MerkleChain (S-Layer)
  5. If ψ-Anchor rejects → degrade action to safe fallback
  6. Execute action in MuJoCo environment
  7. Compute η (GaussEx residual) and feed back to ψ-Anchor

This module is the **integration glue** between the existing MuJoCo-Bench-IDO
framework components (PsiAnchor, PGGate, KappaSnapLogger, GoalEML) and the
TOMAS deployment architecture described in the IDO whitepaper.

Author: MuJoCo-Bench-IDO v0.17.0 — TOMAS Integration Layer
"""

import time
import numpy as np
from typing import Any, Dict, Optional, Tuple, Callable

from agent.psi_anchor import PsiAnchor
from core.pg_gate import PGGate
from core.kappa_snap_logger import KappaSnapLogger
from core.goal_eml_mj import GoalEML

IDO_TOMAS_WRAPPER_VERSION: str = "v0.17.0"


class TOMASMuJoCoWrapper:
    """TOMAS MuJoCo environment wrapper with three-layer IDO integration.

    Wraps any MuJoCo environment (dm_control or gymnasium) and intercepts
    every step to apply the P→C→S layer pipeline.

    Attributes:
        env: The base MuJoCo environment (must have step() and _get_obs()).
        policy: Callable or object with predict() for VLA action generation.
        psi_anchor: PsiAnchor instance for meta-management.
        pg_gate: PGGate instance for hard physical clamping.
        snap_logger: KappaSnapLogger for audit trail.
        goal: Current goal vector (for η computation).
        step_count: Current step counter.
        max_steps: Maximum steps before truncation.
    """

    VERSION: str = IDO_TOMAS_WRAPPER_VERSION

    def __init__(self,
                 base_env: Any,
                 vla_policy: Any = None,
                 goal_eml: Optional[GoalEML] = None,
                 max_steps: int = 1000,
                 tau_safe: float = 0.05,
                 device: str = "cpu") -> None:
        """Initialize TOMAS wrapper.

        Args:
            base_env: MuJoCo environment with step() and _get_obs().
            vla_policy: VLA policy object with predict(obs) → action,
                       or None to use external actions.
            goal_eml: GoalEML instance defining task invariants.
                     If None, a default one is created.
            max_steps: Maximum steps before truncation.
            tau_safe: PG-Gate safe torque threshold (N·m).
            device: Device string for VLA inference ("cpu" or "cuda").
        """
        self.env = base_env
        self.policy = vla_policy
        self.device = device

        # C-Layer: ψ-Anchor + PG-Gate
        if goal_eml is None:
            goal_eml = GoalEML(
                name='default',
                target_pos=np.zeros(3),
                invariants=['energy', 'momentum'],
                delta_K=0.05,
                max_energy_inject=10.0,
            )
        self.psi_anchor: PsiAnchor = PsiAnchor(goal_eml=goal_eml)
        self.pg_gate: PGGate = PGGate(tau_safe=tau_safe)

        # S-Layer: κ-Snap audit
        self.snap_logger: KappaSnapLogger = KappaSnapLogger()

        # State
        self.goal: np.ndarray = goal_eml.target_pos if hasattr(goal_eml, 'target_pos') else np.zeros(3)
        self.step_count: int = 0
        self.max_steps: int = max_steps
        self._last_eta: float = 0.0
        self._last_action: Optional[np.ndarray] = None
        self._last_raw_action: Optional[np.ndarray] = None
        self._safety_violations: list = []

    def _vla_infer(self, obs: np.ndarray) -> np.ndarray:
        """Query VLA policy for action generation (P-Layer).

        Args:
            obs: Current observation from environment.

        Returns:
            Raw action vector from VLA policy.
        """
        if self.policy is None:
            # No policy — return zero action (external mode)
            return np.zeros(self._get_action_dim())

        if hasattr(self.policy, 'predict'):
            return self.policy.predict(obs)
        elif callable(self.policy):
            return self.policy(obs)
        else:
            return np.zeros(self._get_action_dim())

    def _get_action_dim(self) -> int:
        """Get action dimension from environment."""
        if hasattr(self.env, 'action_spec'):
            spec = self.env.action_spec()
            return int(np.prod(spec.shape))
        elif hasattr(self.env, 'action_space'):
            return int(np.prod(self.env.action_space.shape))
        return 7  # default for SO-ARM100

    def _get_physics(self) -> Optional[Any]:
        """Extract MuJoCo physics data from environment."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        elif hasattr(self.env, 'data') and hasattr(self.env, 'model'):
            return type('Physics', (), {
                'model': self.env.model,
                'data': self.env.data,
            })()
        return None

    def _compute_eta(self, obs: np.ndarray) -> float:
        """Compute GaussEx residual η (physical distance to goal).

        v0.17.2 FIX: Previously used obs[:3] (joint angles) vs goal=[0,0,0],
        which gave meaningless eta = ||joint_angles|| ≈ 1.5. Now uses
        obs[14:17] (gripper-to-target distance vector) when available,
        giving the true physical eta = ||gripper_pos - target_pos||.

        For environments where obs[14:17] is the distance vector:
            eta = ||obs[14:17]||
        For other environments, falls back to obs[:goal_dim] vs goal.

        Args:
            obs: Current observation.

        Returns:
            η value (float).
        """
        # v0.17.2: Use distance vector from obs[14:17] if available
        # (HeadlessMuJoCoEnv provides gripper-to-target distance at [14:17])
        if len(obs) >= 17:
            dist_vec = obs[14:17]
            return float(np.linalg.norm(dist_vec))

        # Fallback: use first goal_dim elements vs goal
        goal_dim = min(len(self.goal), len(obs))
        diff = obs[:goal_dim] - self.goal[:goal_dim]
        return float(np.linalg.norm(diff))

    def _degrade_action(self, action: np.ndarray, reason: str) -> np.ndarray:
        """Degrade action to safe fallback when ψ-Anchor rejects.

        Strategy:
        - MAX_TORQUE: scale down to 50% of original
        - PITCH_ROLL: zero out and freeze
        - ZMP_VIOLATION: reduce to 10% (creep mode)
        - default: scale to 30%

        Args:
            action: Original action to degrade.
            reason: Violation reason string from ψ-Anchor.

        Returns:
            Degraded (safe) action.
        """
        reason_upper = reason.upper()

        if 'TORQUE' in reason_upper:
            return action * 0.5
        elif 'PITCH' in reason_upper or 'ROLL' in reason_upper:
            return np.zeros_like(action)
        elif 'ZMP' in reason_upper:
            return action * 0.1
        else:
            return action * 0.3

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one TOMAS step: P→C→S pipeline.

        Pipeline:
        1. P-Layer: Get raw action (from VLA or external)
        2. C-Layer: PG-Gate clamp (sentient body protection)
        3. C-Layer: ψ-Anchor check (physical constraints)
        4. S-Layer: κ-Snap log (audit trail)
        5. Execute in MuJoCo
        6. Compute η and feed back

        Args:
            action: External action (if None, VLA policy is used).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
            info contains: eta, snap_id, psi_state, safety_violations.
        """
        self.step_count += 1

        # ── P-Layer: Phenomenon (VLA → raw action) ──
        obs = self._get_obs()
        raw_action = action if action is not None else self._vla_infer(obs)
        self._last_raw_action = raw_action.copy()

        physics = self._get_physics()

        # ── C-Layer: PG-Gate hard clamp ──
        clamped_action = self.pg_gate.gate(
            raw_action, physics=physics, kappa_snap_logger=None
        )

        # ── C-Layer: ψ-Anchor sentient finger check ──
        psi_result = self.psi_anchor.check_sentient_finger_limit(
            clamped_action, physics=physics
        )

        if not psi_result["ok"]:
            # ψ-Anchor rejected — degrade action
            clamped_action = psi_result["clamped_action"]
            violation_reason = "MAX_TORQUE"
            self._safety_violations.append({
                'step': self.step_count,
                'reason': violation_reason,
                'original': raw_action.tolist(),
                'clamped': clamped_action.tolist(),
            })

            # κ-Snap: log ψ-Anchor rejection
            self.snap_logger.log(
                event_type="PSI_ANCHOR_REJECT",
                level="L2",
                eta=self._last_eta,
                decision="DEGRADE",
                details={
                    "violation": violation_reason,
                    "violated_indices": psi_result.get("violated_indices", []),
                }
            )
        else:
            violation_reason = None

        # ── S-Layer: κ-Snap audit ──
        self.snap_logger.log(
            event_type="ACTION_ACCEPT",
            level="L0",
            eta=self._last_eta,
            decision="EXECUTE",
            details={
                "step": self.step_count,
                "action_norm": float(np.linalg.norm(clamped_action)),
                "raw_norm": float(np.linalg.norm(raw_action)),
                "psi_ok": psi_result["ok"],
                "psi_violations": self._safety_violations[-1] if self._safety_violations else [],
                "pg_gate_clamped": bool(np.any(clamped_action != raw_action)),
            }
        )

        # ── Execute in MuJoCo ──
        result = self.env.step(clamped_action)
        if len(result) == 5:
            mujoco_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            mujoco_obs, reward, done, info = result
            terminated = done
            truncated = self.step_count >= self.max_steps
        else:
            raise ValueError(f"Unexpected env.step() return length: {len(result)}")

        # ── Post-step: compute η and update ψ-Anchor ──
        # v0.17.2: Use env's physical eta if available (more accurate than
        # recomputing from obs, since env has direct access to body positions)
        if "eta" in info and info["eta"] is not None:
            self._last_eta = float(info["eta"])
        else:
            self._last_eta = self._compute_eta(mujoco_obs)
        self.psi_anchor.update_eta_history(self._last_eta)

        # Update adjusted δ_K based on η trend
        self.psi_anchor.adjust_delta_K(self.psi_anchor.adjusted_delta_K)

        self._last_action = clamped_action.copy()

        # Enrich info with TOMAS metadata
        info['eta'] = self._last_eta
        info['step'] = self.step_count
        info['psi_state'] = self.psi_anchor.get_state()
        info['safety_violations'] = len(self._safety_violations)
        info['psi_violations'] = self._safety_violations[-1] if self._safety_violations else []
        info['raw_action'] = raw_action.tolist()
        info['snap_chain_verified'] = self.snap_logger.verify_chain()
        info['tomas_version'] = self.VERSION

        truncated = truncated or self.step_count >= self.max_steps

        return mujoco_obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation from environment."""
        if hasattr(self.env, '_get_obs'):
            return self.env._get_obs()
        elif hasattr(self.env, 'obs'):
            return self.env.obs
        else:
            return np.zeros(10)

    def reset(self) -> Any:
        """Reset environment and TOMAS state."""
        self.step_count = 0
        self._last_eta = 0.0
        self._last_action = None
        self._last_raw_action = None
        self._safety_violations = []
        self.snap_logger.reset()

        if hasattr(self.env, 'reset'):
            result = self.env.reset()
            if isinstance(result, tuple):
                return result
            return result
        return None

    def get_audit_trail(self) -> list:
        """Get complete κ-Snap audit trail (log buffer with full event details).

        Returns:
            List of event dicts with snap_id linkage and details.
            Each entry includes: snap_id, prev_snap_id, eta, decision,
            event_type, level, details (step, action_norm, psi_violations, etc.).
        """
        return self.snap_logger.get_log_buffer()

    def get_safety_report(self) -> Dict[str, Any]:
        """Get safety violation summary report.

        Returns:
            Dict with total_violations, violation_breakdown, chain_integrity.
        """
        violations = self._safety_violations
        breakdown: Dict[str, int] = {}
        for v in violations:
            reason = v.get('reason', 'unknown')
            breakdown[reason] = breakdown.get(reason, 0) + 1

        return {
            'total_violations': len(violations),
            'violation_breakdown': breakdown,
            'chain_integrity': self.snap_logger.verify_chain(),
            'steps_executed': self.step_count,
            'violation_rate': len(violations) / max(self.step_count, 1),
        }

    def get_eta_history(self) -> list:
        """Get η history buffer from ψ-Anchor.

        Returns:
            List of η values recorded during episode.
        """
        return list(self.psi_anchor.eta_history)
