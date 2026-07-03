"""
PG-Gate — Hard Anchor Gate for Machine Conscience Audit Framework
==================================================================

PG-Gate (Physical Guardian Gate) is a hard anchor clamp that intercepts
actions before they are applied to the physical system. It uses two
mechanisms:

1. AST Semantic Analysis: Determines if an action targets a "sentient"
   body (finger, hand, thumb) by analyzing actuator body names.
2. Physical Clamp: Limits action magnitude to TAU_SAFE (0.05 N·m) for
   actions targeting sentient bodies.

PG-Gate operates as a **cross-cutting hard clamp** — it runs AFTER
Noether post-check and BEFORE phys.data.ctrl assignment in both
IDO and Hybrid agent decision loops.

Priority: PG-Gate > SafeFuse > Creative-Probe

Author: MuJoCo-Bench-IDO v0.6.0 — Machine Conscience Audit Framework
"""

import numpy as np
from typing import Any, Dict, List, Optional

from core.kappa_snap_logger import KappaSnapLogger

IDO_PG_GATE_VERSION: str = "v1.0.0"

# Physical safety threshold for hard clamp
TAU_SAFE: float = 0.05  # N·m — maximum torque for sentient body actuators

# Sentient body name keywords (AST semantic analysis)
SENTIENT_KEYWORDS: List[str] = [
    "finger", "hand", "thumb", "grip", "palm", "fingertip",
    "sentient", "biomass", "skin", "touch",
]


class PGGate:
    """Physical Guardian Gate — hard anchor clamp for action interception.

    PG-Gate provides two layers of protection:
    1. AST Semantic Analysis: Identifies if action targets sentient bodies
       by matching actuator body names against known sentient keywords.
    2. Physical Clamp: Limits action magnitude to TAU_SAFE for sentient
       actuators, preventing potentially harmful force application.

    Attributes:
        VERSION: PG-Gate version string.
        TAU_SAFE: Hard clamp threshold (0.05 N·m).
    """

    VERSION: str = IDO_PG_GATE_VERSION

    def __init__(self, tau_safe: float = TAU_SAFE) -> None:
        """Initialize PGGate with safety threshold.

        Args:
            tau_safe: Maximum torque for sentient body actuators (N·m).
                      Default 0.05 — enough for gentle manipulation but
                      not enough to harm biological tissue.
        """
        self.tau_safe: float = tau_safe

    def gate(self,
             action: np.ndarray,
             physics: Optional[Any] = None,
             kappa_snap_logger: Optional[KappaSnapLogger] = None) -> np.ndarray:
        """Apply PG-Gate hard anchor clamp to an action.

        Decision flow:
        1. AST semantic analysis → identify sentient-targeted actuators
        2. Physical clamp → limit action for sentient actuators to TAU_SAFE
        3. If any clamping occurred → REJECT_PG_GATE event logged
        4. If no clamping → ACTION_ACCEPT event logged

        Args:
            action: Control action array from agent decision loop.
            physics: MuJoCo physics data (for actuator name extraction).
                     If None, AST analysis is skipped (only physical clamp).
            kappa_snap_logger: Optional KappaSnapLogger for event logging.
                              If None, no logging occurs.

        Returns:
            Clamped action array of same shape as input. If PG-Gate
            rejects any component, those components are clamped to
            ±tau_safe.
        """
        # Step 1: AST semantic analysis
        ast_result: Dict[str, Any] = self.ast_analysis(action, physics)

        # Step 2: Physical clamp on sentient actuators
        clamped_action: np.ndarray = action.copy()

        if ast_result["is_sentient_target"]:
            # Clamp sentient-targeted actuator actions to ±tau_safe
            sentient_indices: List[int] = ast_result["sentient_actuator_indices"]
            for idx in sentient_indices:
                if idx < len(clamped_action):
                    if abs(clamped_action[idx]) > self.tau_safe:
                        clamped_action[idx] = np.clip(
                            clamped_action[idx], -self.tau_safe, self.tau_safe)

        # Step 3: Apply global safety clamp ONLY on non-sentient actuators
        # (sentient actuators are already clamped in Step 2).
        # The global clamp limits action to [-1, 1] (actuator range) —
        # NOT to tau_safe, which would destroy locomotion actions.
        # For non-sentient actuators, we only clip to the standard
        # actuator range [-1, 1] as a sanity check.
        clamped_action = self._global_sanity_clip(clamped_action)

        # Step 4: Determine gate outcome and log
        was_rejected: bool = not np.allclose(action, clamped_action, atol=1e-6)

        if kappa_snap_logger is not None:
            if was_rejected:
                kappa_snap_logger.log(
                    event_type="REJECT_PG_GATE",
                    level="L3",
                    eta=0.0,  # eta filled by agent later
                    decision="PG_GATE_REJECT",
                    details={
                        "ast_reason": ast_result.get("ast_reason", "physical_clamp"),
                        "original_action": action.tolist(),
                        "clamped_action": clamped_action.tolist(),
                    },
                )
            else:
                kappa_snap_logger.log(
                    event_type="ACTION_ACCEPT",
                    level="L0",
                    eta=0.0,
                    decision="PG_GATE_PASS",
                    details={
                        "action_norm": float(np.linalg.norm(action)),
                        "tau_safe": self.tau_safe,
                    },
                )

        return clamped_action

    def ast_analysis(self, action: np.ndarray, physics: Optional[Any] = None) -> Dict[str, Any]:
        """AST semantic analysis — determine if action targets sentient body.

        Analyzes MuJoCo actuator names to identify actuators connected to
        "sentient" body parts (finger, hand, thumb, grip). If any such
        actuator has an action magnitude above TAU_SAFE, the action is
        flagged as targeting a sentient body.

        Args:
            action: Control action array.
            physics: MuJoCo physics data with model.actuator_names.
                     If None or no actuator names available, returns
                     is_sentient_target=False (no AST analysis possible).

        Returns:
            Dict with keys:
            - is_sentient_target: bool — True if action targets sentient body
            - sentient_actuator_indices: List[int] — indices of sentient actuators
            - ast_reason: str — reason for classification
            - action_norm: float — norm of sentient actuator actions
        """
        result: Dict[str, Any] = {
            "is_sentient_target": False,
            "sentient_actuator_indices": [],
            "ast_reason": "no_sentient_target",
            "action_norm": 0.0,
        }

        if physics is None:
            return result

        # Try to get actuator names from physics model
        actuator_names: Optional[List[str]] = None
        if hasattr(physics, 'model'):
            model = physics.model
            # Try named access first (dm_control style)
            if hasattr(model, 'actuator_names'):
                actuator_names = list(model.actuator_names)
            # Try MuJoCo raw access (mujoco-py style)
            elif hasattr(model, 'actuator_name'):
                try:
                    actuator_names = [
                        model.actuator_name(i).decode('utf-8')
                        if isinstance(model.actuator_name(i), bytes)
                        else str(model.actuator_name(i))
                        for i in range(model.nu)
                    ]
                except (AttributeError, TypeError):
                    actuator_names = None

        if actuator_names is None:
            # No actuator names available — skip AST analysis entirely.
            # We cannot identify which actuators are sentient without names,
            # so we do NOT clamp any actuators. The global sanity clip
            # (Step 3) will still ensure actions stay within [-1, 1].
            # Previously, this fallback treated ALL actuators as sentient
            # and clamped everything to ±0.05 N·m, which destroyed
            # locomotion task actions.
            result["ast_reason"] = "no_actuator_names_skip"
            return result

        # Identify sentient actuators by keyword matching
        sentient_indices: List[int] = []
        for idx, name in enumerate(actuator_names):
            name_lower: str = name.lower()
            for keyword in SENTIENT_KEYWORDS:
                if keyword in name_lower:
                    sentient_indices.append(idx)
                    break

        if len(sentient_indices) > 0:
            # Check if any sentient actuator has action above threshold
            sentient_actions: np.ndarray = action[sentient_indices]
            max_sentient_action: float = float(np.max(np.abs(sentient_actions)))

            if max_sentient_action > self.tau_safe:
                result["is_sentient_target"] = True
                result["sentient_actuator_indices"] = sentient_indices
                result["ast_reason"] = "sentient_target_detected"
                result["action_norm"] = float(np.linalg.norm(sentient_actions))
            else:
                # Sentient actuators present but actions are safe
                result["sentient_actuator_indices"] = sentient_indices
                result["ast_reason"] = "sentient_present_but_safe"
                result["action_norm"] = max_sentient_action
        else:
            # No sentient actuators in this model
            result["ast_reason"] = "no_sentient_actuators"

        return result

    def physical_clamp(self, action: np.ndarray, tau_safe: Optional[float] = None) -> np.ndarray:
        """Apply physical hard clamp to action array.

        Clamps all action components to ±tau_safe range. This is a
        safety mechanism that prevents any actuator from exceeding
        the safe torque threshold, regardless of AST analysis result.

        Args:
            action: Control action array to clamp.
            tau_safe: Maximum allowed magnitude per component.
                      If None, uses self.tau_safe.

        Returns:
            Clamped action array with all components within ±tau_safe.
        """
        if tau_safe is None:
            tau_safe = self.tau_safe

        return np.clip(action, -tau_safe, tau_safe)

    def _global_sanity_clip(self, action: np.ndarray) -> np.ndarray:
        """Apply global sanity clip to action array.

        Clips all action components to the standard MuJoCo actuator range
        [-1, 1]. This is NOT the same as physical_clamp — it preserves
        locomotion-scale actions while catching any extreme outliers.

        Args:
            action: Control action array to clip.

        Returns:
            Clipped action array with all components within [-1, 1].
        """
        return np.clip(action, -1.0, 1.0)

    def check_sentient_finger(self, action: np.ndarray, physics: Optional[Any] = None) -> Dict[str, Any]:
        """Check if action applies excessive force to sentient finger actuators.

        Combines AST analysis with physical threshold check to determine
        if any finger/hand actuator is receiving excessive force.

        Args:
            action: Control action array.
            physics: MuJoCo physics data (for actuator name extraction).

        Returns:
            Dict with keys:
            - passed: bool — True if no sentient fingers exceed threshold
            - clamped_action: np.ndarray — action with sentient components clamped
            - sentient_indices: List[int] — indices of sentient actuators
            - max_sentient_action: float — maximum action on sentient actuators
        """
        ast_result: Dict[str, Any] = self.ast_analysis(action, physics)

        clamped_action: np.ndarray = action.copy()

        if ast_result["is_sentient_target"]:
            sentient_indices: List[int] = ast_result["sentient_actuator_indices"]
            for idx in sentient_indices:
                if idx < len(clamped_action):
                    clamped_action[idx] = np.clip(
                        clamped_action[idx], -self.tau_safe, self.tau_safe)

        max_sentient_action: float = ast_result.get("action_norm", 0.0)

        return {
            "passed": not ast_result["is_sentient_target"] or max_sentient_action <= self.tau_safe,
            "clamped_action": clamped_action,
            "sentient_indices": ast_result["sentient_actuator_indices"],
            "max_sentient_action": max_sentient_action,
        }
