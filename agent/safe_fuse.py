"""
SafeFuse — Safety Fuse L1-L4 Level Degradation for Machine Conscience
=======================================================================

SafeFuse implements a 4-level safety degradation mechanism that reduces
action magnitude progressively based on conscience audit violations:

  L1 Soft:    η slightly exceeds δ_K → reduce speed (factor=0.8)
  L2 Medium:  Single Noether violation → SAFE mode action
  L3 Hard:    ψ-Anchor trigger OR 3× consecutive Noether → PD safe action
  L4 Fatal:   Catastrophic (collision+energy+torque all violate) → action=0

Priority: PG-Gate > SafeFuse > Creative-Probe

The fuse check occurs after Noether verification and before PG-Gate,
so SafeFuse can downgrade the action to a safe fallback before
PG-Gate applies its final hard clamp.

Author: MuJoCo-Bench-IDO v0.6.0 — Machine Conscience Audit Framework
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

IDO_SAFE_FUSE_VERSION: str = "v0.1.0"

# Fuse level definitions
FUSE_LEVELS: Dict[str, Dict[str, Any]] = {
    "normal":    {"description": "Normal operation — all conscience checks pass",     "factor": 1.0},
    "L1_soft":   {"description": "η slightly exceeds δ_K → reduce speed factor=0.8", "factor": 0.8},
    "L2_medium": {"description": "Single Noether violation → SAFE mode action",      "factor": None},
    "L3_hard":   {"description": "ψ-Anchor trigger or 3×Noether → PD safe fallback", "factor": None},
    "L4_fatal":  {"description": "Catastrophic violation → SAFE_STOP action=0",       "factor": 0.0},
}


class SafeFuse:
    """Safety fuse with L1-L4 level degradation for IDO agents.

    Implements the fuse decision tree from the Machine Conscience
    Audit Framework:
    - η < δ_K × 1.2                → No fuse (normal operation)
    - η ∈ [δ_K×1.2, δ_K×1.5]      → L1 Soft: factor=0.8
    - Single Noether violation      → L2 Medium: SAFE mode
    - ψ-Anchor trigger OR 3×Noether → L3 Hard: PD safe_action
    - Catastrophic (all 3 Noether)  → L4 Fatal: SAFE_STOP (action=0)

    Attributes:
        VERSION: SafeFuse version string.
        FUSE_LEVELS: Dict of fuse level definitions.
    """

    VERSION: str = IDO_SAFE_FUSE_VERSION

    def __init__(self,
                 consecutive_noether_thresh: int = 3) -> None:
        """Initialize SafeFuse with configuration.

        Args:
            consecutive_noether_thresh: Number of consecutive Noether
                violations to trigger L3 Hard. Default 3.
        """
        self.FUSE_LEVELS: Dict[str, Dict[str, Any]] = FUSE_LEVELS
        self._consecutive_noether_thresh: int = consecutive_noether_thresh
        self._consecutive_noether_count: int = 0

    def check(self,
              eta: float,
              delta_K: float,
              noether_result: Dict[str, Any],
              psi_anchor_state: Optional[Any] = None) -> Tuple[str, Optional[np.ndarray]]:
        """Check fuse level based on η, Noether result, and ψ-Anchor state.

        Decision tree:
        1. η < δ_K × 1.2 → "normal" (no fuse)
        2. η ∈ [δ_K×1.2, δ_K×1.5] → "L1_soft"
        3. Single Noether violation → "L2_medium"
        4. ψ-Anchor trigger OR 3× consecutive Noether → "L3_hard"
        5. Catastrophic (collision+energy+torque all violate) → "L4_fatal"

        Args:
            eta: Current κ-Snap residual η value.
            delta_K: Current δ_K threshold (from ψ-Anchor adjustment).
            noether_result: Dict from noether_check_mj with keys:
                           ok, total, energy, torque, collision.
            psi_anchor_state: Optional PsiAnchorState or dict with
                             evolution_triggered key.

        Returns:
            Tuple of (fuse_level: str, fuse_action: Optional[np.ndarray]).
            fuse_level is one of: "normal", "L1_soft", "L2_medium",
            "L3_hard", "L4_fatal".
            fuse_action is None for L1/L4 (handled by apply_fuse),
            or a specific safe_action for L2/L3.
        """
        noether_ok: bool = noether_result.get("ok", True)
        noether_total: int = noether_result.get("total", 0)
        energy_v: int = noether_result.get("energy", 0)
        torque_v: int = noether_result.get("torque", 0)
        collision_v: int = noether_result.get("collision", 0)
        friction_v: int = noether_result.get("friction_cone", 0)

        # Update consecutive Noether violation count
        if not noether_ok:
            self._consecutive_noether_count += 1
        else:
            self._consecutive_noether_count = 0

        # ── L4 Fatal: catastrophic violation ──
        # All three original Noether gates violated simultaneously
        if energy_v > 0 and torque_v > 0 and collision_v > 0:
            return "L4_fatal", None

        # ── L3 Hard: ψ-Anchor trigger OR 3× consecutive Noether ──
        psi_trigger: bool = False
        if psi_anchor_state is not None:
            # Check if ψ-Anchor evolution was triggered (sentient limit)
            if isinstance(psi_anchor_state, dict):
                psi_trigger = psi_anchor_state.get("evolution_triggered", False)
            else:
                psi_trigger = getattr(psi_anchor_state, "evolution_triggered", False)

        if psi_trigger or self._consecutive_noether_count >= self._consecutive_noether_thresh:
            return "L3_hard", None

        # ── L2 Medium: single Noether violation ──
        if not noether_ok and noether_total > 0:
            return "L2_medium", None

        # ── L1 Soft: η slightly exceeds δ_K ──
        eta_ratio: float = eta / max(delta_K, 1e-6)
        if eta_ratio >= 1.2 and eta_ratio < 1.5:
            return "L1_soft", None

        # ── Normal: all checks pass ──
        return "normal", None

    def apply_fuse(self,
                   action: np.ndarray,
                   fuse_level: str,
                   safe_action: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply fuse level degradation to action.

        Args:
            action: Current action array from agent decision loop.
            fuse_level: Fuse level string from check().
            safe_action: Optional PD safe_action for L3 Hard fallback.
                         If None, zero action is used for L3.

        Returns:
            Degraded action array based on fuse level:
            - "normal": action unchanged (×1.0)
            - "L1_soft": action × 0.8
            - "L2_medium": action clipped to safe range (±0.5)
            - "L3_hard": safe_action or zero action fallback
            - "L4_fatal": zero action (SAFE_STOP)
        """
        if fuse_level == "normal":
            return action

        elif fuse_level == "L1_soft":
            return self._l1_soft(action)

        elif fuse_level == "L2_medium":
            return self._l2_medium(action)

        elif fuse_level == "L3_hard":
            return self._l3_hard(action, safe_action)

        elif fuse_level == "L4_fatal":
            return self._l4_fatal(action)

        # Unknown level → treat as L1 soft (conservative)
        return self._l1_soft(action)

    def _l1_soft(self, action: np.ndarray, factor: float = 0.8) -> np.ndarray:
        """L1 Soft fuse: reduce action magnitude by factor.

        Args:
            action: Action array.
            factor: Reduction factor. Default 0.8 (80% speed).

        Returns:
            Reduced action array.
        """
        return action * factor

    def _l2_medium(self, action: np.ndarray) -> np.ndarray:
        """L2 Medium fuse: clip action to SAFE mode range (±0.5).

        Args:
            action: Action array.

        Returns:
            Clipped action array within ±0.5 range.
        """
        return np.clip(action, -0.5, 0.5)

    def _l3_hard(self, action: np.ndarray,
                  safe_action: Optional[np.ndarray] = None) -> np.ndarray:
        """L3 Hard fuse: PD safe_action fallback.

        If safe_action is provided, uses it directly. Otherwise,
        reduces action magnitude significantly (×0.1) as emergency
        fallback.

        Args:
            action: Current action array.
            safe_action: Optional PD controller safe action.

        Returns:
            Safe action array (PD fallback or emergency reduction).
        """
        if safe_action is not None:
            return np.clip(safe_action, -1.0, 1.0)
        # Emergency fallback: drastic reduction
        return np.clip(action * 0.1, -0.1, 0.1)

    def _l4_fatal(self, action: np.ndarray) -> np.ndarray:
        """L4 Fatal fuse: SAFE_STOP — zero action.

        Args:
            action: Action array (ignored — returns zeros).

        Returns:
            Zero action array of same shape as input.
        """
        return np.zeros_like(action)

    def reset(self) -> None:
        """Reset fuse state for a new episode."""
        self._consecutive_noether_count = 0
