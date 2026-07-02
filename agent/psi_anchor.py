"""
ψ-Anchor — IDO Meta-Management Layer
======================================

The ψ-Anchor is the IDO meta-management layer responsible for:
  1. Dynamic threshold adjustment: adapts δ_K based on η trend analysis
  2. MotorPrimitives evolution strategy: decides 'light' (promote/demote)
     vs 'freeze' (parameter solidification)
  3. Physical constraint injection: anchors Noether conservation quantities
     as ψ-Anchor constraints (Noether's Razor principle)
  4. Epiplexity scoring: quantifies structural information density of the task
  5. Self-evolution timing: determines when evolution should be triggered
     (Survey When dimension)

Inspired by:
  - Noether's Razor: conservation invariants as model selection principle
  - Self-evolving agents Survey: When/What/How dimensions
  - IDO/TOMAS architecture: κ-Snap residual + Noether gate synergy

Author: tomas-arc3-solver project · MuJoCo-Bench-IDO v0.2.0 extension
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.goal_eml_mj import GoalEML

IDO_PSI_ANCHOR_VERSION: str = "v0.2.1"

# Default configuration constants
ETA_HISTORY_MAX_LEN: int = 100
ETA_TREND_WINDOW: int = 10
PLATEAU_THRESHOLD: float = 0.001
EPIPLEXITY_EVOLUTION_THRESH: float = 2.0
PLATEAU_EVOLUTION_MIN_STEPS: int = 5


@dataclass
class PsiAnchorState:
    """Snapshot of ψ-Anchor internal state for serialization/logging.

    Attributes:
        eta_trend: Current η trend classification string.
        evo_policy: Current evolution policy ('light' or 'freeze').
        adjusted_delta_K: Currently adjusted δ_K value.
        epiplexity_score: Current epiplexity score.
        conservation_score: Current Noether conservation score.
        evolution_triggered: Whether evolution was triggered this step.
        plateau_steps: Number of consecutive steps in plateau trend.
    """
    eta_trend: str = 'unknown'
    evo_policy: str = 'light'
    adjusted_delta_K: float = 0.05
    epiplexity_score: float = 0.0
    conservation_score: float = 1.0
    evolution_triggered: bool = False
    plateau_steps: int = 0


class PsiAnchor:
    """IDO ψ-Anchor meta-management layer.

    Manages dynamic threshold adjustment, MotorPrimitives evolution strategy,
    Noether constraint anchoring, epiplexity scoring, and self-evolution timing.

    The ψ-Anchor sits above the IDO decision loop and observes η trajectories
    to make meta-level decisions about when and how the agent should adapt:
    - If η is descending (converging), tighten δ_K for precision
    - If η is plateaued (stalled), relax δ_K to break out or freeze primitives
    - If η is ascending (diverging), preserve current threshold and try new strategies

    Attributes:
        goal_eml: GoalEML instance defining task invariants and tolerances.
        eta_history: Buffer of recent η values for trend analysis.
        evo_policy: Current evolution policy ('light' or 'freeze').
        conservation_anchors: List of Noether invariant names from GoalEML.
        epiplexity_score: Computed structural information density score.
        plateau_steps: Consecutive steps where η trend was 'plateau'.
        adjusted_delta_K: The ψ-Anchor-adjusted δ_K threshold.
    """

    def __init__(self,
                 goal_eml: GoalEML,
                 eta_history_max_len: int = ETA_HISTORY_MAX_LEN,
                 eta_trend_window: int = ETA_TREND_WINDOW,
                 plateau_threshold: float = PLATEAU_THRESHOLD,
                 epiplexity_thresh: float = EPIPLEXITY_EVOLUTION_THRESH,
                 plateau_evolution_min: int = PLATEAU_EVOLUTION_MIN_STEPS,
                 relative_plateau_ratio: float = 0.01,
                 abs_plateau_floor: float = 0.001) -> None:
        """Initialize ψ-Anchor with GoalEML and configuration parameters.

        Args:
            goal_eml: GoalEML instance defining the task's invariants.
            eta_history_max_len: Maximum length of η history buffer.
            eta_trend_window: Number of recent η values used for trend analysis.
            plateau_threshold: Legacy absolute threshold (kept for backward compat).
                              Effective threshold now uses relative+floor formula.
            epiplexity_thresh: Epiplexity threshold for evolution triggering.
            plateau_evolution_min: Minimum plateau steps before evolution trigger.
            relative_plateau_ratio: Relative plateau threshold ratio (1% default).
                                    Effective threshold = max(abs_floor, ratio * η_mean).
            abs_plateau_floor: Absolute plateau threshold floor for small η values.
                               Prevents overly tight thresholds when η is near zero.
        """
        self.goal_eml: GoalEML = goal_eml
        self.eta_history: List[float] = []
        self.eta_history_max_len: int = eta_history_max_len
        self.eta_trend_window: int = eta_trend_window
        self.plateau_threshold: float = plateau_threshold
        self.epiplexity_thresh: float = epiplexity_thresh
        self.plateau_evolution_min: int = plateau_evolution_min
        self.relative_plateau_ratio: float = relative_plateau_ratio
        self.abs_plateau_floor: float = abs_plateau_floor

        # Evolution policy: 'light' (promote/demote primitives) or 'freeze' (固化参数)
        self.evo_policy: str = 'light'

        # Conservation anchors derived from GoalEML invariants (Noether's Razor)
        self.conservation_anchors: List[str] = list(goal_eml.invariants)

        # Epiplexity score — computed from GoalEML structural properties
        self.epiplexity_score: float = self.compute_epiplexity(goal_eml)

        # Plateau tracking
        self.plateau_steps: int = 0

        # Adjusted δ_K — starts from GoalEML's delta_K
        self.adjusted_delta_K: float = goal_eml.delta_K

        # Conservation state tracking
        self._last_noether_ok: bool = True
        self._last_noether_msg: str = ""
        self._conservation_score: float = 1.0

    def update_eta_history(self, eta: float) -> None:
        """Append η to history buffer and analyze trend.

        Maintains a bounded buffer of η values. When the buffer exceeds
        eta_history_max_len, the oldest values are discarded.

        Args:
            eta: Current κ-Snap residual value from this decision step.
        """
        self.eta_history.append(eta)
        if len(self.eta_history) > self.eta_history_max_len:
            self.eta_history = self.eta_history[-self.eta_history_max_len:]

    def analyze_eta_trend(self) -> str:
        """Analyze η trend over the recent window.

        Uses the last eta_trend_window η values to classify the trend:
        - 'descending': η is consistently decreasing (convergence)
        - 'plateau': η changes are below effective plateau threshold (stalled)
        - 'ascending': η is consistently increasing (diverging)
        - 'unknown': insufficient data (< 3 values in window)

        v0.2.1: Plateau threshold is now relative (scales with η magnitude):
            effective_threshold = max(abs_plateau_floor, relative_plateau_ratio * η_mean)
        This prevents η≈1.0 tasks from being incorrectly classified as plateau
        when a 1% relative change (abs_delta≈0.01) exceeds the old fixed 0.001.

        Returns:
            Trend classification string: 'descending', 'plateau', 'ascending', 'unknown'.
        """
        window: List[float] = self.eta_history[-self.eta_trend_window:]
        if len(window) < 3:
            return 'unknown'

        # Compute consecutive deltas
        deltas: List[float] = []
        for i in range(1, len(window)):
            deltas.append(window[i] - window[i - 1])

        mean_delta: float = float(np.mean(deltas))
        abs_mean_delta: float = abs(mean_delta)

        # v0.2.1: Relative plateau threshold — scales with η magnitude
        eta_mean: float = float(np.mean(window))
        effective_threshold: float = max(self.abs_plateau_floor,
                                         self.relative_plateau_ratio * eta_mean)

        if abs_mean_delta < effective_threshold:
            trend: str = 'plateau'
        elif mean_delta < 0:
            trend = 'descending'
        else:
            trend = 'ascending'

        # Update plateau counter
        if trend == 'plateau':
            self.plateau_steps += 1
        else:
            self.plateau_steps = 0

        return trend

    def adjust_delta_K(self, current_delta_K: float) -> float:
        """Dynamically adjust δ_K based on η trend.

        Adjustment strategy:
        - descending: tighten δ_K (×0.8) → more precision needed as agent converges
        - plateau: relax δ_K (×1.2) → break out of stall by widening acceptance
        - ascending: freeze δ_K → preserve current threshold (no change)
        - unknown: freeze δ_K → insufficient data to adjust

        Args:
            current_delta_K: Current δ_K threshold value.

        Returns:
            Adjusted δ_K value. Frozen if trend is ascending/unknown.
        """
        trend: str = self.analyze_eta_trend()

        if trend == 'descending':
            # Tighten: more precision as η decreases (convergence)
            adjusted: float = current_delta_K * 0.8
        elif trend == 'plateau':
            # Relax: widen acceptance to break out of stall
            adjusted = current_delta_K * 1.2
        elif trend == 'ascending':
            # Freeze: preserve current threshold
            adjusted = current_delta_K
        else:
            # Unknown: insufficient data, freeze
            adjusted = current_delta_K

        # Clamp adjusted δ_K to reasonable bounds [1e-4, 10.0]
        adjusted = max(1e-4, min(adjusted, 10.0))
        self.adjusted_delta_K = adjusted

        return adjusted

    def decide_evolution_policy(self) -> str:
        """Decide MotorPrimitives evolution policy based on η trend and epiplexity.

        Policy rules:
        - plateau + low epiplexity → 'freeze' (参数固化，避免无意义变异)
        - plateau + high epiplexity → 'light' (有结构信息但停滞，需要探索)
        - descending + high epiplexity → 'light' (轻量进化，加速收敛)
        - descending + low epiplexity → 'freeze' (简单任务，固化避免干扰)
        - ascending → 'light' (尝试新策略以 reverse divergence)
        - unknown → 'light' (default safe exploration policy)

        Returns:
            Evolution policy string: 'light' or 'freeze'.
        """
        trend: str = self.analyze_eta_trend()
        epi: float = self.epiplexity_score

        if trend == 'plateau':
            if epi < self.epiplexity_thresh:
                # Low structure density in plateau → freeze to avoid noise
                policy: str = 'freeze'
            else:
                # High structure density in plateau → light evolution to explore
                policy = 'light'
        elif trend == 'descending':
            if epi >= self.epiplexity_thresh:
                # High epiplexity while converging → light evolution accelerates
                policy = 'light'
            else:
                # Low epiplexity while converging → freeze to maintain progress
                policy = 'freeze'
        elif trend == 'ascending':
            # Diverging → always try new strategies
            policy = 'light'
        else:
            # Unknown → default to light (safe exploration)
            policy = 'light'

        self.evo_policy = policy
        return policy

    def inject_conservation_anchor(self, noether_ok: bool,
                                    noether_msg: str) -> Dict[str, object]:
        """Inject Noether conservation constraint as anchoring signal.

        Translates the Noether gate result into a ψ-Anchor anchoring signal
        that constrains evolution decisions. If Noether violations occurred,
        the conservation score decreases and evolution should be more cautious.

        Args:
            noether_ok: Whether all Noether gates passed (True = no violations).
            noether_msg: Human-readable violation message (empty if ok=True).

        Returns:
            Anchor dict with keys:
            - 'ok': bool — conservation gates passed
            - 'violations': List[str] — extracted violation codes from message
            - 'conservation_score': float — 1.0 if ok, else degraded score
        """
        self._last_noether_ok = noether_ok
        self._last_noether_msg = noether_msg

        # Parse violation codes from message
        violations: List[str] = []
        if not noether_ok and noether_msg:
            # Extract codes like 'Noether-E', 'Noether-F', 'Noether-C'
            for token in noether_msg.split():
                if token.startswith('Noether-'):
                    violations.append(token.rstrip(':'))

        # Compute conservation score: 1.0 if ok, degrades with violations
        if noether_ok:
            conservation_score: float = 1.0
        else:
            # Each violation reduces score by 0.3 (min 0.1)
            conservation_score = max(0.1, 1.0 - 0.3 * len(violations))

        self._conservation_score = conservation_score

        return {
            'ok': noether_ok,
            'violations': violations,
            'conservation_score': conservation_score,
        }

    def compute_epiplexity(self, goal_eml: GoalEML) -> float:
        """Compute epiplexity score — structural information density of the task.

        Epiplexity quantifies how much structured information the task contains,
        which determines how much room there is for meaningful evolution.

        Formula:
            epiplexity = len(invariants) * (1/delta_K) * log(max_energy_inject)

        Higher epiplexity means:
        - More invariants to satisfy (richer structure)
        - Tighter tolerance (more precision required)
        - Higher energy budget (more physical complexity)
        → More room for evolution and improvement

        Lower epiplexity means:
        - Fewer invariants (simpler task)
        - Looser tolerance
        - Lower energy budget
        → Less room for evolution, freeze policy preferred

        Args:
            goal_eml: GoalEML instance to compute epiplexity from.

        Returns:
            Epiplexity score (float). Typical range: 0.5 - 15.0.
        """
        n_invariants: int = len(goal_eml.invariants)
        delta_K: float = max(goal_eml.delta_K, 1e-6)  # avoid division by zero
        max_energy: float = max(goal_eml.max_energy_inject, 1.0)  # avoid log(0)

        # epiplexity = n_invariants / delta_K * log(max_energy)
        epiplexity: float = n_invariants * (1.0 / delta_K) * math.log(max_energy)

        self.epiplexity_score = epiplexity
        return epiplexity

    def should_trigger_evolution(self) -> bool:
        """Determine if evolution should be triggered (When dimension).

        Evolution is triggered when:
        1. η has been plateaued for ≥ plateau_evolution_min consecutive steps
        2. AND epiplexity > epiplexity_thresh (enough structure to evolve)
        3. AND conservation_score > 0.5 (Noether constraints are mostly satisfied)

        This implements the "When" dimension from the self-evolving agents Survey:
        evolution should only occur when the agent is stalled AND there is
        sufficient structural information to make evolution meaningful AND
        the agent is not currently violating fundamental conservation laws.

        Returns:
            True if evolution conditions are met, False otherwise.
        """
        trend: str = self.analyze_eta_trend()

        # Condition 1: plateau for sufficient steps
        plateau_cond: bool = (trend == 'plateau'
                              and self.plateau_steps >= self.plateau_evolution_min)

        # Condition 2: sufficient epiplexity for meaningful evolution
        epi_cond: bool = self.epiplexity_score > self.epiplexity_thresh

        # Condition 3: conservation constraints are mostly satisfied
        conserv_cond: bool = self._conservation_score > 0.5

        should_evolve: bool = plateau_cond and epi_cond and conserv_cond
        return should_evolve

    def get_state(self) -> PsiAnchorState:
        """Get current ψ-Anchor internal state as a snapshot.

        Returns:
            PsiAnchorState dataclass with current internal state values.
        """
        return PsiAnchorState(
            eta_trend=self.analyze_eta_trend(),
            evo_policy=self.evo_policy,
            adjusted_delta_K=self.adjusted_delta_K,
            epiplexity_score=self.epiplexity_score,
            conservation_score=self._conservation_score,
            evolution_triggered=self.should_trigger_evolution(),
            plateau_steps=self.plateau_steps,
        )

    def apply_evolution_to_macros(self,
                                   macros: List[Tuple[object, float]],
                                   evo_policy: str) -> List[Tuple[object, float]]:
        """Apply evolution policy to MotorPrimitives macro list.

        'light' policy: promote top-scoring primitive (+0.05),
                        demote worst-scoring primitive (-0.05)
        'freeze' policy: no changes to IC-Value scores

        Args:
            macros: List of (primitive_fn, ic_value) tuples from MotorPrimitives.
            evo_policy: Evolution policy string ('light' or 'freeze').

        Returns:
            Updated macro list with adjusted IC-Value scores.
        """
        if evo_policy == 'freeze' or len(macros) == 0:
            # No evolution — return unchanged macros
            return list(macros)

        # Light evolution: promote best, demote worst
        updated_macros: List[Tuple[object, float]] = list(macros)

        # Find best and worst by IC-Value score
        scores: List[float] = [m[1] for m in updated_macros]
        best_idx: int = int(np.argmax(scores))
        worst_idx: int = int(np.argmin(scores))

        # Promote best: increase IC-Value by +0.05
        best_fn, best_score = updated_macros[best_idx]
        updated_macros[best_idx] = (best_fn, min(best_score + 0.05, 1.0))

        # Demote worst: decrease IC-Value by -0.05
        worst_fn, worst_score = updated_macros[worst_idx]
        updated_macros[worst_idx] = (worst_fn, max(worst_score - 0.05, 0.1))

        return updated_macros
