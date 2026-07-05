"""
TOMAS Deploy — TOMAS Agent Deployment Orchestrator
===================================================

v0.17.0: TOMAS Agent deployment orchestration with MetaQuery self-attribution.

This module implements the TOMAS Agent deployment pipeline that integrates:
  - P-Layer: VLA (Vision-Language-Action) policy inference
  - C-Layer: psi-Anchor hard constraints + PG-Gate physical clamping
  - S-Layer: kappa-Snap causal snapshot auditing + MetaQuery self-attribution

The deploy() method is the main entry point for running a TOMAS agent
in a MuJoCo environment. It orchestrates the full P->C->S pipeline
for each step, collects audit trails, and provides self-diagnostic
capabilities via MetaQuery.

MetaQuery System:
  Three fundamental questions the agent can ask itself:
    1. WHY_THIS_ACTION() — Why did I choose this action?
    2. AUDIT_SNAP() — What does the causal trail say?
    3. LEARN_SKILL() — Can I abstract this into a reusable skill?

Author: MuJoCo-Bench-IDO v0.17.0
"""

import numpy as np
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import TOMAS integration components
from agent.psi_anchor import PsiAnchor, PsiAnchorState
from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper
from agent.failure_attribution import TOMASFailureAttributor, FailureAttributionResult

logger = logging.getLogger(__name__)


class DeployStatus(Enum):
    """Deployment execution status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


class MetaQueryType(Enum):
    """MetaQuery self-attribution question types."""
    WHY_THIS_ACTION = "why_this_action"
    AUDIT_SNAP = "audit_snap"
    LEARN_SKILL = "learn_skill"


@dataclass
class MetaQueryResult:
    """Result of a MetaQuery self-attribution call."""
    query_type: MetaQueryType
    answer: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SkillRecord:
    """A learned skill abstracted from successful episodes."""
    name: str
    description: str
    trigger_condition: str
    action_template: np.ndarray
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class DeployReport:
    """Comprehensive deployment report."""
    status: DeployStatus
    total_steps: int
    total_episodes: int
    avg_eta: float
    final_eta: float
    psi_violations: int
    kappa_snap_count: int
    safety_report: Dict[str, Any]
    eta_history: List[float] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    meta_queries: List[MetaQueryResult] = field(default_factory=list)
    learned_skills: List[SkillRecord] = field(default_factory=list)
    failure_attributions: List[FailureAttributionResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "avg_eta": round(self.avg_eta, 6),
            "final_eta": round(self.final_eta, 6),
            "psi_violations": self.psi_violations,
            "kappa_snap_count": self.kappa_snap_count,
            "safety_report": self.safety_report,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "meta_queries_count": len(self.meta_queries),
            "learned_skills_count": len(self.learned_skills),
            "failure_attributions_count": len(self.failure_attributions),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class TOMASAgent:
    """TOMAS Agent — full P->C->S deployment orchestrator.

    This class wraps a TOMASMuJoCoWrapper and provides higher-level
    deployment orchestration including:
      - Multi-episode rollout management
      - MetaQuery self-attribution
      - Skill learning and replay
      - Failure attribution and adaptive adjustment
      - Comprehensive deployment reporting

    Usage:
        agent = TOMASAgent(
            env=my_mujoco_env,
            vla_policy=my_vla,
            goal_eml=my_goal_eml,
        )
        report = agent.deploy(num_episodes=10, max_steps_per_episode=200)
        print(report.to_json())

    Attributes:
        wrapper: The underlying TOMASMuJoCoWrapper instance.
        attributor: Failure attribution engine.
        skills: Learned skill library.
        deploy_history: History of deployment reports.
    """

    VERSION: str = "v0.17.0"

    def __init__(
        self,
        env: Any,
        vla_policy: Optional[Callable] = None,
        goal_eml: Optional[Any] = None,
        max_steps: int = 200,
        tau_safe: float = 0.05,
        enable_failure_attribution: bool = True,
        enable_skill_learning: bool = True,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize TOMAS Agent.

        Args:
            env: MuJoCo environment (must have step() and reset()).
            vla_policy: Callable that takes obs and returns action. If None,
                        uses a zero-action placeholder (for testing).
            goal_eml: GoalEML instance for eta computation. If None,
                      wrapper will use a default L2 distance.
            max_steps: Maximum steps per episode.
            tau_safe: psi-Anchor safety torque limit (N.m).
            enable_failure_attribution: Whether to run failure attribution
                                        on episodes with high final eta.
            enable_skill_learning: Whether to learn skills from successful
                                   episodes.
            llm_client: Optional LLM client for failure attribution.
                        If None, uses offline heuristic mode.
        """
        self.wrapper = TOMASMuJoCoWrapper(
            base_env=env,
            vla_policy=vla_policy,
            goal_eml=goal_eml,
            max_steps=max_steps,
            tau_safe=tau_safe,
        )

        self.enable_failure_attribution = enable_failure_attribution
        self.enable_skill_learning = enable_skill_learning

        self.attributor = TOMASFailureAttributor() if enable_failure_attribution else None
        self.skills: List[SkillRecord] = []
        self.deploy_history: List[DeployReport] = []
        self._current_episode_actions: List[np.ndarray] = []
        self._current_episode_etas: List[float] = []

    def deploy(
        self,
        num_episodes: int = 1,
        max_steps_per_episode: Optional[int] = None,
        reset_env: bool = True,
        verbose: bool = False,
    ) -> DeployReport:
        """Run the TOMAS agent for multiple episodes.

        This is the main deployment method. For each episode:
        1. Reset environment (optional)
        2. Run P->C->S pipeline for each step
        3. Collect audit trail and eta history
        4. On episode end: evaluate success, optionally run MetaQuery

        Args:
            num_episodes: Number of episodes to run.
            max_steps_per_episode: Override max_steps for this deploy.
            reset_env: Whether to call env.reset() between episodes.
            verbose: Whether to log per-step info.

        Returns:
            DeployReport with comprehensive results.
        """
        start_time = time.time()
        all_eta_history: List[float] = []
        all_audit_trail: List[Dict[str, Any]] = []
        all_meta_queries: List[MetaQueryResult] = []
        all_failure_attributions: List[FailureAttributionResult] = []
        total_psi_violations = 0
        total_kappa_snaps = 0
        episodes_completed = 0
        final_status = DeployStatus.SUCCESS

        steps_this_deploy = max_steps_per_episode or self.wrapper.max_steps

        for ep in range(num_episodes):
            if verbose:
                logger.info(f"Episode {ep + 1}/{num_episodes} starting...")

            # Reset
            if reset_env:
                self.wrapper.reset()

            self._current_episode_actions = []
            self._current_episode_etas = []

            # Run episode
            ep_done = False
            for step in range(steps_this_deploy):
                try:
                    result = self.wrapper.step(action=None)
                except Exception as e:
                    logger.error(f"Step {step} failed: {e}")
                    final_status = DeployStatus.FAILED
                    ep_done = True
                    break

                # Handle both 4-tuple and 5-tuple returns
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                eta = info.get("eta", 0.0)
                psi_violations = info.get("psi_violations", [])

                self._current_episode_actions.append(
                    np.asarray(info.get("raw_action", np.zeros(self.wrapper._get_action_dim())))
                )
                self._current_episode_etas.append(eta)
                all_eta_history.append(eta)

                if psi_violations:
                    total_psi_violations += len(psi_violations)

                if verbose and step % 50 == 0:
                    logger.info(
                        f"  Step {step}: eta={eta:.4f}, "
                        f"psi_violations={len(psi_violations)}"
                    )

                if done:
                    ep_done = True
                    break

            episodes_completed += 1

            # Post-episode analysis
            final_eta = self._current_episode_etas[-1] if self._current_episode_etas else 1.0
            avg_eta = float(np.mean(self._current_episode_etas)) if self._current_episode_etas else 1.0

            # Skill learning from successful episodes
            if self.enable_skill_learning and final_eta < 0.1:
                skill = self._learn_skill_from_episode(
                    self._current_episode_actions,
                    self._current_episode_etas,
                )
                if skill is not None:
                    self.skills.append(skill)
                    if verbose:
                        logger.info(f"  Learned skill: {skill.name}")

            # Failure attribution for high-eta episodes
            if self.enable_failure_attribution and final_eta > 0.3:
                attribution = self._run_failure_attribution()
                if attribution is not None:
                    all_failure_attributions.append(attribution)
                    if verbose:
                        logger.info(
                            f"  Failure attributed: {attribution.failure_type} "
                            f"(confidence={attribution.confidence:.2f})"
                        )

            # MetaQuery: AUDIT_SNAP
            audit_query = self.meta_query(MetaQueryType.AUDIT_SNAP)
            if audit_query:
                all_meta_queries.append(audit_query)

        # Collect audit trail
        audit_trail = self.wrapper.get_audit_trail()
        all_audit_trail = audit_trail

        # Safety report
        safety_report = self.wrapper.get_safety_report()

        # Kappa-Snap count
        snap_logger = getattr(self.wrapper, "snap_logger", None)
        if snap_logger is not None:
            total_kappa_snaps = len(snap_logger.get_log_buffer())

        # Determine final status
        if episodes_completed < num_episodes:
            final_status = DeployStatus.PARTIAL
        if episodes_completed == 0:
            final_status = DeployStatus.FAILED

        elapsed = time.time() - start_time

        report = DeployReport(
            status=final_status,
            total_steps=len(all_eta_history),
            total_episodes=episodes_completed,
            avg_eta=float(np.mean(all_eta_history)) if all_eta_history else 1.0,
            final_eta=all_eta_history[-1] if all_eta_history else 1.0,
            psi_violations=total_psi_violations,
            kappa_snap_count=total_kappa_snaps,
            safety_report=safety_report,
            eta_history=all_eta_history,
            audit_trail=all_audit_trail,
            meta_queries=all_meta_queries,
            learned_skills=list(self.skills),
            failure_attributions=all_failure_attributions,
            elapsed_seconds=elapsed,
        )

        self.deploy_history.append(report)
        return report

    def meta_query(
        self,
        query_type: MetaQueryType,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[MetaQueryResult]:
        """Execute a MetaQuery self-attribution.

        This is the S-Layer's self-reflection mechanism. The agent asks
        itself one of three fundamental questions:

        WHY_THIS_ACTION: Analyzes why the last action was chosen,
            including VLA inference, psi-Anchor adjustments, and
            PG-Gate clamping.

        AUDIT_SNAP: Reviews the kappa-Snap causal trail to identify
            patterns, anomalies, or pathology signatures.

        LEARN_SKILL: Attempts to abstract the recent action sequence
            into a reusable SkillRecord.

        Args:
            query_type: Type of MetaQuery to execute.
            context: Optional additional context.

        Returns:
            MetaQueryResult with the answer and evidence, or None if
            insufficient data.
        """
        context = context or {}
        evidence: Dict[str, Any] = {}
        answer = ""
        confidence = 0.0

        if query_type == MetaQueryType.WHY_THIS_ACTION:
            answer, evidence, confidence = self._query_why_this_action(context)

        elif query_type == MetaQueryType.AUDIT_SNAP:
            answer, evidence, confidence = self._query_audit_snap(context)

        elif query_type == MetaQueryType.LEARN_SKILL:
            answer, evidence, confidence = self._query_learn_skill(context)

        if not answer:
            return None

        result = MetaQueryResult(
            query_type=query_type,
            answer=answer,
            evidence=evidence,
            confidence=confidence,
        )
        return result

    def _query_why_this_action(self, context: Dict) -> Tuple[str, Dict, float]:
        """WHY_THIS_ACTION: Explain the last action decision."""
        audit_trail = self.wrapper.get_audit_trail()
        if not audit_trail:
            return "No actions have been taken yet.", {}, 0.0

        last = audit_trail[-1]
        details = last.get("details", {})
        evidence = {
            "step": details.get("step", last.get("step", -1)),
            "eta": last.get("eta", 0.0),
            "psi_violations": details.get("psi_violations", []),
            "pg_gate_clamped": details.get("pg_gate_clamped", False),
            "action_norm": details.get("action_norm", 0.0),
        }

        parts = []
        parts.append(f"At step {evidence['step']}, the action was chosen by the VLA policy.")
        parts.append(f"The GaussEx residual (eta) was {evidence['eta']:.4f}.")

        if evidence["psi_violations"]:
            parts.append(
                f"psi-Anchor detected {len(evidence['psi_violations'])} violation(s): "
                f"{evidence['psi_violations']}. Action was degraded accordingly."
            )
        else:
            parts.append("psi-Anchor reported no violations.")

        if evidence["pg_gate_clamped"]:
            parts.append("PG-Gate clamped the action to satisfy physical constraints.")

        parts.append(f"Final action norm: {evidence['action_norm']:.4f}.")

        confidence = 0.9 if evidence["step"] >= 0 else 0.0
        return " ".join(parts), evidence, confidence

    def _query_audit_snap(self, context: Dict) -> Tuple[str, Dict, float]:
        """AUDIT_SNAP: Review the kappa-Snap causal trail."""
        audit_trail = self.wrapper.get_audit_trail()
        eta_history = self.wrapper.get_eta_history()

        if not audit_trail:
            return "No audit trail available.", {}, 0.0

        evidence = {
            "total_steps": len(audit_trail),
            "avg_eta": float(np.mean(eta_history)) if eta_history else 1.0,
            "final_eta": eta_history[-1] if eta_history else 1.0,
            "eta_trend": self._classify_eta_trend(eta_history),
            "total_violations": sum(
                len(step.get("details", {}).get("psi_violations", step.get("psi_violations", [])))
                for step in audit_trail
            ),
        }

        parts = []
        parts.append(
            f"Audit trail contains {evidence['total_steps']} steps."
        )
        parts.append(
            f"Average eta: {evidence['avg_eta']:.4f}, "
            f"final eta: {evidence['final_eta']:.4f}."
        )
        parts.append(f"Eta trend: {evidence['eta_trend']}.")
        parts.append(
            f"Total psi-Anchor violations: {evidence['total_violations']}."
        )

        # Detect pathology
        pathology = self._detect_pathology(eta_history, audit_trail)
        if pathology:
            parts.append(f"Detected possible pathology: {pathology}.")
            evidence["pathology"] = pathology

        confidence = min(0.95, 0.5 + len(audit_trail) / 200.0)
        return " ".join(parts), evidence, confidence

    def _query_learn_skill(self, context: Dict) -> Tuple[str, Dict, float]:
        """LEARN_SKILL: Attempt to abstract recent actions into a skill."""
        if not self._current_episode_actions:
            return "No actions to learn from.", {}, 0.0

        eta_history = self._current_episode_etas
        final_eta = eta_history[-1] if eta_history else 1.0

        if final_eta > 0.15:
            return (
                f"Episode final eta ({final_eta:.4f}) too high for skill extraction. "
                "Skill learning requires eta < 0.15.",
                {"final_eta": final_eta},
                0.3,
            )

        skill = self._learn_skill_from_episode(
            self._current_episode_actions,
            eta_history,
        )
        if skill is None:
            return "Could not extract a meaningful skill pattern.", {}, 0.2

        evidence = {
            "skill_name": skill.name,
            "description": skill.description,
            "success_rate": skill.success_rate,
        }
        return (
            f"Learned skill '{skill.name}': {skill.description}",
            evidence,
            0.7,
        )

    def _learn_skill_from_episode(
        self,
        actions: List[np.ndarray],
        etas: List[float],
    ) -> Optional[SkillRecord]:
        """Extract a skill from a successful episode's action sequence.

        Uses EML-SemZip compression:
        1. Dead-zone pruning: Remove steps with IC < threshold
        2. Normalize remaining actions to a template
        3. Store as SkillRecord

        Args:
            actions: List of action vectors.
            etas: List of eta values.

        Returns:
            SkillRecord or None if no meaningful pattern found.
        """
        if len(actions) < 5:
            return None

        # Compute Information Cardinality (IC) per step
        ics = []
        for i in range(1, len(actions)):
            delta = actions[i] - actions[i - 1]
            position_entropy = float(np.std(delta))
            velocity_var = float(np.var(delta))
            ic = position_entropy + velocity_var
            ics.append(ic)

        # Dead-zone pruning: IC < 0.45 threshold
        ic_threshold = 0.45
        high_ic_indices = [i for i, ic in enumerate(ics) if ic > ic_threshold]

        if len(high_ic_indices) < 3:
            return None

        # Extract high-IC actions as template
        template_actions = [actions[i] for i in high_ic_indices]
        template = np.mean(template_actions, axis=0)

        # Create skill record
        skill_name = f"skill_{len(self.skills)}_{int(time.time())}"
        final_eta = etas[-1] if etas else 1.0

        skill = SkillRecord(
            name=skill_name,
            description=f"Auto-learned skill from episode with final_eta={final_eta:.4f}, "
                        f"{len(high_ic_indices)} high-IC steps",
            trigger_condition=f"eta > {np.mean(etas):.4f} and similar observation",
            action_template=template,
            success_count=1,
            metadata={
                "episode_length": len(actions),
                "high_ic_steps": len(high_ic_indices),
                "avg_ic": float(np.mean(ics)),
                "final_eta": final_eta,
            },
        )
        return skill

    def _run_failure_attribution(self) -> Optional[FailureAttributionResult]:
        """Run failure attribution on the current episode."""
        if self.attributor is None:
            return None

        audit_trail = self.wrapper.get_audit_trail()
        eta_history = self.wrapper.get_eta_history()

        if not audit_trail or not eta_history:
            return None

        # Extract psi violations from audit trail
        psi_violations = []
        for step in audit_trail:
            violations = step.get("details", {}).get("psi_violations", step.get("psi_violations", []))
            if violations:
                psi_violations.append({
                    "step": step.get("details", {}).get("step", step.get("step", 0)),
                    "violations": violations,
                })

        result = self.attributor.offline_attribuate(
            eta_history=eta_history,
            snap_trail=audit_trail,
            psi_violations=psi_violations if psi_violations else None,
        )
        return result

    @staticmethod
    def _classify_eta_trend(eta_history: List[float]) -> str:
        """Classify the eta trend.

        Returns one of: descending, plateau, ascending, escape
        """
        if len(eta_history) < 5:
            return "insufficient_data"

        recent = eta_history[-min(20, len(eta_history)):]
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2 :]

        avg_first = float(np.mean(first_half))
        avg_second = float(np.mean(second_half))

        if avg_second < avg_first * 0.8:
            return "descending"
        elif avg_second > avg_first * 1.2:
            # Check for escape (sudden spike)
            if len(recent) >= 3:
                last_delta = abs(recent[-1] - recent[-2])
                avg_delta = float(np.mean(np.abs(np.diff(recent))))
                if last_delta > avg_delta * 3:
                    return "escape"
            return "ascending"
        else:
            return "plateau"

    @staticmethod
    def _detect_pathology(
        eta_history: List[float],
        audit_trail: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Detect TOMAS flow pathology from eta history and audit trail.

        Six pathology types:
        1. local_optimum_trap: eta plateaus at non-zero value
        2. eta_false_convergence: eta drops fast then stalls
        3. premature_release: psi-Anchor violations spike after eta drops
        4. eta_escape: sudden eta increase after convergence
        5. psi_anchor_overkill: excessive violations with low eta
        6. validation_gap: audit trail gaps
        """
        if len(eta_history) < 10:
            return None

        trend = TOMASAgent._classify_eta_trend(eta_history)

        # local_optimum_trap: plateau at non-zero
        if trend == "plateau":
            avg_recent = float(np.mean(eta_history[-10:]))
            if avg_recent > 0.1:
                return "local_optimum_trap"

        # eta_false_convergence: fast drop then stall
        if len(eta_history) >= 20:
            first_5 = float(np.mean(eta_history[:5]))
            next_5 = float(np.mean(eta_history[5:10]))
            last_10 = float(np.mean(eta_history[-10:]))
            if first_5 - next_5 > 0.2 and abs(next_5 - last_10) < 0.05:
                return "eta_false_convergence"

        # eta_escape: sudden spike
        if trend == "escape":
            return "eta_escape"

        # psi_anchor_overkill: many violations with low eta
        total_violations = sum(
            len(step.get("details", {}).get("psi_violations", step.get("psi_violations", [])))
            for step in audit_trail
        )
        avg_eta = float(np.mean(eta_history))
        if total_violations > 20 and avg_eta < 0.1:
            return "psi_anchor_overkill"

        # premature_release: violations increase after eta drops
        if len(audit_trail) >= 20:
            early_violations = sum(
                len(step.get("details", {}).get("psi_violations", step.get("psi_violations", [])))
                for step in audit_trail[: len(audit_trail) // 2]
            )
            late_violations = sum(
                len(step.get("details", {}).get("psi_violations", step.get("psi_violations", [])))
                for step in audit_trail[len(audit_trail) // 2 :]
            )
            if late_violations > early_violations * 2 and trend == "descending":
                return "premature_release"

        # validation_gap: audit trail shorter than eta history
        if len(audit_trail) < len(eta_history) * 0.8:
            return "validation_gap"

        return None

    def get_skill_library(self) -> List[Dict[str, Any]]:
        """Get the current skill library as a list of dictionaries."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "trigger_condition": s.trigger_condition,
                "success_rate": s.success_rate,
                "success_count": s.success_count,
                "failure_count": s.failure_count,
                "metadata": s.metadata,
            }
            for s in self.skills
        ]

    def get_deploy_summary(self) -> Dict[str, Any]:
        """Get a summary of all deployments."""
        if not self.deploy_history:
            return {"total_deploys": 0}

        latest = self.deploy_history[-1]
        return {
            "total_deploys": len(self.deploy_history),
            "latest_status": latest.status.value,
            "latest_avg_eta": round(latest.avg_eta, 6),
            "latest_final_eta": round(latest.final_eta, 6),
            "latest_psi_violations": latest.psi_violations,
            "latest_kappa_snap_count": latest.kappa_snap_count,
            "total_skills_learned": len(self.skills),
            "total_failure_attributions": sum(
                len(r.failure_attributions) for r in self.deploy_history
            ),
        }
