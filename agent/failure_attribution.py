"""
TOMAS Failure Attribution — LLM-Powered Root Cause Analysis
=============================================================

Provides structured failure attribution prompts for the TOMAS S-Layer
self-diagnosis. When a κ-Snap event records a failure (PSI_ANCHOR_REJECT,
NOETHER_VIOLATION, ETA_DIVERGENCE), this module constructs a structured
prompt that can be sent to an LLM for root cause analysis.

The prompt follows the TOMAS failure attribution template:
  1. Context: What was the agent trying to do?
  2. Evidence: κ-Snap audit trail (snap_id, η, psi_violation)
  3. Hypothesis: What went wrong? (WHY_THIS_ACTION)
  4. Correction: How to fix it? (LEARN_SKILL)
  5. Confidence: How sure are we?

The module includes a pattern-based fallback parser that works offline
(no LLM required) by matching common failure signatures.

Author: MuJoCo-Bench-IDO v0.17.0 — TOMAS S-Layer Self-Diagnosis
"""

import json
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

IDO_FAILURE_ATTRIBUTION_VERSION: str = "v0.17.0"


@dataclass
class FailureAttributionResult:
    """Structured result of failure attribution analysis.

    Attributes:
        why: Root cause hypothesis string.
        how_fix: Proposed correction strategy.
        confidence: Confidence score [0.0, 1.0].
        adjustment_type: Type of adjustment needed
                        ('scale_down', 'freeze', 'explore', 'replan', 'none').
        adjustment_params: Parameters for the adjustment.
        failure_type: Classified failure type string.
    """
    why: str = ""
    how_fix: str = ""
    confidence: float = 0.0
    adjustment_type: str = "none"
    adjustment_params: Dict[str, Any] = field(default_factory=dict)
    failure_type: str = "unknown"


class TOMASFailureAttributor:
    """TOMAS failure attribution engine with LLM prompt construction.

    Builds structured prompts for LLM-based root cause analysis of agent
    failures. Includes offline pattern-matching fallback for when no LLM
    is available.

    The six TOMAS flow-pathology types:
    1. local_optimum_trap: η plateaus at non-zero value
    2. eta_false_convergence: η → 0 but task not solved
    3. premature_release: Agent stops before completing task
    4. eta_escape: η suddenly increases after convergence
    5. psi_anchor_overkill: ψ-Anchor too restrictive, blocks valid actions
    6. validation_gap: No κ-Snap audit, untracked behavior

    Attributes:
        system_prompt: System-level prompt for LLM context.
        max_snap_events: Maximum κ-Snap events to include in prompt.
    """

    VERSION: str = IDO_FAILURE_ATTRIBUTION_VERSION

    # Six flow-pathology types from TOMAS whitepaper
    PATHOLOGY_TYPES: List[str] = [
        "local_optimum_trap",
        "eta_false_convergence",
        "premature_release",
        "eta_escape",
        "psi_anchor_overkill",
        "validation_gap",
    ]

    SYSTEM_PROMPT: str = """You are TOMAS, an IDO (Information Dynamics Ontology) failure attribution engine.
Your role is to analyze agent failures using the κ-Snap audit trail and identify root causes.

## IDO Framework Context
- η (GaussEx residual): L2 distance between current state and goal. η→0 means convergence.
- IC (Information Cardinality): Shannon entropy of state changes. IC<0.45 = dead-zone.
- ψ-Anchor: Physical hard constraints (torque, velocity, pitch, ZMP).
- κ-Snap: Causal snapshot with MerkleChain linkage for tamper-proof audit.

## Six Flow-Pathology Types
1. local_optimum_trap: η plateaus at non-zero value, agent stuck in local optimum.
2. eta_false_convergence: η→0 but task not actually solved (wrong metric).
3. premature_release: Agent releases/opens gripper before task complete.
4. eta_escape: η suddenly increases after period of convergence.
5. psi_anchor_overkill: ψ-Anchor too restrictive, blocking valid actions.
6. validation_gap: Missing κ-Snap audit, behavior untracked.

## Output Format
Respond with a JSON object:
{
  "why": "<root cause hypothesis>",
  "how_fix": "<proposed correction strategy>",
  "confidence": <0.0-1.0>,
  "adjustment": {
    "type": "<scale_down|freeze|explore|replan|none>",
    "params": {<key-value pairs>}
  },
  "failure_type": "<one of 6 pathology types>"
}
"""

    def __init__(self, max_snap_events: int = 20) -> None:
        """Initialize failure attributor.

        Args:
            max_snap_events: Maximum κ-Snap events to include in prompt.
        """
        self.max_snap_events: int = max_snap_events

    def build_prompt(self,
                     snap_trail: List[Dict[str, Any]],
                     eta_history: List[float],
                     task_description: str = "",
                     psi_violations: Optional[List[Dict]] = None) -> str:
        """Build a structured failure attribution prompt.

        Args:
            snap_trail: κ-Snap audit trail (list of event dicts).
            eta_history: Recent η values.
            task_description: Human-readable task description.
            psi_violations: List of ψ-Anchor violation records.

        Returns:
            Formatted prompt string for LLM input.
        """
        # Extract last N snap events
        recent_snaps = snap_trail[-self.max_snap_events:]

        # Classify η trend
        eta_trend = self._classify_eta_trend(eta_history)

        # Detect pathology type (offline heuristic)
        pathology = self._detect_pathology(eta_history, recent_snaps, psi_violations or [])

        prompt = f"""## Failure Attribution Request

### Task Context
{task_description or "Unknown task"}

### η Trajectory Analysis
- Current η: {eta_history[-1] if eta_history else 'N/A'}
- η trend: {eta_trend}
- η history (last 20): {eta_history[-20:] if eta_history else []}
- Min η: {min(eta_history) if eta_history else 'N/A'}
- Max η: {max(eta_history) if eta_history else 'N/A'}

### κ-Snap Audit Trail (last {len(recent_snaps)} events)
{json.dumps(recent_snaps, indent=2, default=str)}

### ψ-Anchor Violations
{json.dumps(psi_violations or [], indent=2, default=str)}

### Preliminary Pathology Detection
Detected: {pathology}

### Questions
1. WHY_THIS_ACTION: Why did the agent choose actions that led to failure?
2. AUDIT_SNAP: Is the κ-Snap trail complete and consistent?
3. LEARN_SKILL: What skill/adjustment should the agent learn from this failure?

Please provide your attribution analysis in the JSON format specified in the system prompt.
"""
        return prompt

    def parse_response(self, llm_response: str) -> FailureAttributionResult:
        """Parse LLM response into structured result.

        Attempts JSON extraction first, then falls back to pattern matching.

        Args:
            llm_response: Raw LLM response string.

        Returns:
            FailureAttributionResult with parsed fields.
        """
        # Try JSON extraction
        json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
        if not json_match:
            # Try nested JSON
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                adjustment = data.get('adjustment', {})
                return FailureAttributionResult(
                    why=data.get('why', ''),
                    how_fix=data.get('how_fix', ''),
                    confidence=float(data.get('confidence', 0.0)),
                    adjustment_type=adjustment.get('type', 'none') if isinstance(adjustment, dict) else 'none',
                    adjustment_params=adjustment.get('params', {}) if isinstance(adjustment, dict) else {},
                    failure_type=data.get('failure_type', 'unknown'),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback: pattern matching on text
        return self._offline_attribution(llm_response)

    def offline_attribuate(self,
                           eta_history: List[float],
                           snap_trail: List[Dict[str, Any]],
                           psi_violations: Optional[List[Dict]] = None) -> FailureAttributionResult:
        """Offline failure attribution without LLM (pattern-based).

        Uses heuristic rules to classify failure type and suggest corrections.
        This is the fallback when no LLM is available.

        Args:
            eta_history: Recent η values.
            snap_trail: κ-Snap audit trail.
            psi_violations: ψ-Anchor violation records.

        Returns:
            FailureAttributionResult with heuristic attribution.
        """
        psi_violations = psi_violations or []
        pathology = self._detect_pathology(eta_history, snap_trail, psi_violations)

        results: Dict[str, FailureAttributionResult] = {
            "local_optimum_trap": FailureAttributionResult(
                why="η has plateaued at a non-zero value, indicating the agent "
                    "is stuck in a local optimum and cannot reduce η further.",
                how_fix="Relax δ_K threshold by 20% and increase exploration "
                        "(Creative-Probe magnitude) to escape the basin.",
                confidence=0.7,
                adjustment_type="explore",
                adjustment_params={"delta_K_scale": 1.2, "probe_magnitude": 1.5},
                failure_type="local_optimum_trap",
            ),
            "eta_false_convergence": FailureAttributionResult(
                why="η appears to converge to 0 but the task is not actually "
                    "solved. The distance metric may not capture task completion.",
                how_fix="Add task-specific success criteria beyond η. Check if "
                        "goal vector is correctly mapped to task state.",
                confidence=0.6,
                adjustment_type="replan",
                adjustment_params={"check_success_criteria": True},
                failure_type="eta_false_convergence",
            ),
            "premature_release": FailureAttributionResult(
                why="Agent terminated action sequence before task completion. "
                    "κ-Snap shows TERMINATE event before η reached threshold.",
                how_fix="Increase max_steps and add η-threshold gate before "
                        "allowing termination.",
                confidence=0.65,
                adjustment_type="freeze",
                adjustment_params={"min_eta_for_terminate": 0.01, "max_steps_boost": 1.5},
                failure_type="premature_release",
            ),
            "eta_escape": FailureAttributionResult(
                why="η was converging but suddenly increased, suggesting "
                    "external perturbation or policy instability.",
                how_fix="Tighten δ_K and reduce action magnitude. Enable "
                        "SafeFuse conservative mode.",
                confidence=0.75,
                adjustment_type="scale_down",
                adjustment_params={"delta_K_scale": 0.8, "action_scale": 0.5},
                failure_type="eta_escape",
            ),
            "psi_anchor_overkill": TOMASFailureAttributor._psi_overkill_result(),
            "validation_gap": FailureAttributionResult(
                why="κ-Snap audit trail is incomplete or contains gaps. "
                    "Behavior cannot be verified.",
                how_fix="Enable JSONL step-level audit logging and verify "
                        "MerkleChain integrity.",
                confidence=0.8,
                adjustment_type="none",
                adjustment_params={"enable_jsonl": True, "verify_chain": True},
                failure_type="validation_gap",
            ),
        }

        return results.get(pathology, FailureAttributionResult(
            why="Unknown failure pattern. Manual inspection required.",
            how_fix="Review κ-Snap trail and η history manually.",
            confidence=0.3,
            adjustment_type="none",
            failure_type="unknown",
        ))

    def _classify_eta_trend(self, eta_history: List[float]) -> str:
        """Classify η trend from history.

        Args:
            eta_history: List of η values.

        Returns:
            Trend string: 'descending', 'plateau', 'ascending', 'escape', 'unknown'.
        """
        if len(eta_history) < 3:
            return 'unknown'

        window = eta_history[-10:]
        deltas = [window[i] - window[i-1] for i in range(1, len(window))]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        abs_mean = abs(mean_delta)
        eta_mean = sum(window) / len(window) if window else 0.0
        threshold = max(0.001, 0.01 * eta_mean)

        # Check for sudden spike (escape)
        if len(window) >= 3:
            last_delta = window[-1] - window[-2]
            if last_delta > threshold * 5 and window[-1] > window[-3]:
                return 'escape'

        if abs_mean < threshold:
            return 'plateau'
        elif mean_delta < 0:
            return 'descending'
        else:
            return 'ascending'

    def _detect_pathology(self,
                          eta_history: List[float],
                          snap_trail: List[Dict[str, Any]],
                          psi_violations: List[Dict]) -> str:
        """Detect pathology type from available evidence.

        Args:
            eta_history: η values.
            snap_trail: κ-Snap events.
            psi_violations: ψ-Anchor violations.

        Returns:
            Pathology type string (one of PATHOLOGY_TYPES).
        """
        # Check validation gap first
        if len(snap_trail) == 0:
            return "validation_gap"

        # Check ψ-Anchor overkill (many violations with low η)
        if len(psi_violations) > 5:
            avg_eta = sum(eta_history) / len(eta_history) if eta_history else 1.0
            if avg_eta < 0.1:
                return "psi_anchor_overkill"

        trend = self._classify_eta_trend(eta_history)

        if trend == 'escape':
            return "eta_escape"
        elif trend == 'plateau':
            current_eta = eta_history[-1] if eta_history else 1.0
            if current_eta > 0.05:
                return "local_optimum_trap"
            else:
                return "eta_false_convergence"

        # Check for premature release (TERMINATE event before η threshold)
        for snap in snap_trail:
            if isinstance(snap, dict):
                decision = str(snap.get('decision', ''))
                if 'TERMINATE' in decision or 'STOP' in decision:
                    current_eta = float(snap.get('eta', 1.0))
                    if current_eta > 0.05:
                        return "premature_release"

        return "unknown"

    @staticmethod
    def _psi_overkill_result() -> FailureAttributionResult:
        """Create ψ-Anchor overkill result."""
        return FailureAttributionResult(
            why="ψ-Anchor is too restrictive, blocking valid actions. "
                "High violation count with low η suggests over-constraint.",
            how_fix="Relax ψ-Anchor thresholds (tau_safe +20%, velocity +10%) "
                    "and switch to graded SafeFuse instead of hard reject.",
            confidence=0.7,
            adjustment_type="scale_down",
            adjustment_params={"tau_safe_scale": 1.2, "use_graded_fuse": True},
            failure_type="psi_anchor_overkill",
        )

    def _offline_attribution(self, text: str) -> FailureAttributionResult:
        """Offline pattern matching fallback for LLM response parsing.

        Args:
            text: Unstructured text response.

        Returns:
            FailureAttributionResult with best-effort attribution.
        """
        text_lower = text.lower()

        for pathology in self.PATHOLOGY_TYPES:
            if pathology in text_lower:
                return self.offline_attribuate([], [], [])

        # Keyword-based fallback
        if 'torque' in text_lower or 'clamp' in text_lower:
            return FailureAttributionResult(
                why=text[:200],
                how_fix="Adjust torque limits",
                confidence=0.4,
                adjustment_type="scale_down",
                failure_type="psi_anchor_overkill",
            )

        return FailureAttributionResult(
            why=text[:200],
            how_fix="Manual review required",
            confidence=0.2,
            adjustment_type="none",
            failure_type="unknown",
        )
