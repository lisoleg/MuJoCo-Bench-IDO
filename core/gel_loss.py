"""GEL (Goal-EML Injection Loss) — auxiliary training loss from VG-Pair theory.

Implements the GEL loss defined in 章锋(2026) [5]:
    L_GEL = λ1 * ||η_Noether||^2 + λ2 * ||η_contact||^2
          + λ3 * ||η_task||^2   + λ4 * hinge(task_success_pred)

GEL converts the IDO Verifier-gate signals (Noether-Check, κ-Snap) into
differentiable loss terms, bridging "gate-based verification" and
"gradient-based optimization".  When the IDO agent uses a learnable
component (e.g., FlowMatchingEtaPredictor, future neural motor layer),
GEL provides the auxiliary loss signal that aligns the latent dynamics
with the Goal-EML coset constraints.

κ-Phase 1: Loss computation from per-step η values
κ-Phase 2: Accumulation over episode for mean GEL
κ-Phase 3: Integration with agent training loop (future)

References:
    [5] 章锋. 从显式物理到隐式流贯：VG-Pair, C-IPP, GEL与双引擎AGI. 2026.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

__all__ = ["GELLoss", "GELConfig", "compute_gel_from_step"]


# ── κ-Phase 1: Configuration ──

@dataclass
class GELConfig:
    """Configuration for GEL loss weights.

    Attributes:
        lambda_noether: Weight for Noether energy drift term (λ1).
        lambda_contact: Weight for contact/collision violation term (λ2).
        lambda_task: Weight for task residual (η) term (λ3).
        lambda_hinge: Weight for task-success hinge loss term (λ4).
        hinge_margin: Margin for hinge loss (success prediction threshold).
        normalize: If True, normalize each term to [0, 1] before weighting.
    """

    lambda_noether: float = 1.0
    lambda_contact: float = 0.5
    lambda_task: float = 2.0
    lambda_hinge: float = 0.3
    hinge_margin: float = 0.5
    normalize: bool = True


# ── κ-Phase 2: Per-step GEL computation ──

@dataclass
class GELLoss:
    """Goal-EML Injection Loss accumulator and computer.

    Computes the GEL auxiliary loss from per-step IDO signals:
    - η_Noether: energy drift beyond max_energy_inject budget
    - η_contact: collision penetration depth (collide_thresh - min_dist)
    - η_task: κ-Snap residual (gauss_ex_residual η value)
    - hinge: task success prediction hinge loss

    Usage:
        gel = GELLoss(config=GELConfig())
        # Per step:
        step_loss = gel.compute_step(eta_task=0.15, energy_drift=2.5,
                                     energy_budget=500.0, min_geom_dist=0.03,
                                     collide_thresh=0.01, success=False)
        gel.accumulate(step_loss)
        # End of episode:
        mean_gel = gel.mean_loss()
        gel.reset()

    Attributes:
        config: GELConfig with loss weights.
        step_count: Number of accumulated steps.
        total_loss: Cumulative GEL loss across steps.
        component_totals: Per-component cumulative values.
    """

    config: GELConfig = field(default_factory=GELConfig)
    step_count: int = 0
    total_loss: float = 0.0
    component_totals: Dict[str, float] = field(
        default_factory=lambda: {
            "noether": 0.0,
            "contact": 0.0,
            "task": 0.0,
            "hinge": 0.0,
        }
    )

    def compute_step(
        self,
        eta_task: float,
        energy_drift: float = 0.0,
        energy_budget: float = 500.0,
        min_geom_dist: float = 1.0,
        collide_thresh: float = 0.01,
        success: bool = False,
    ) -> Dict[str, float]:
        """Compute per-step GEL loss components.

        Args:
            eta_task: κ-Snap residual (η from gauss_ex_residual).
            energy_drift: Energy change ΔE since previous step (J).
            energy_budget: Maximum allowed energy injection (goal.max_energy_inject).
            min_geom_dist: Minimum geom-geom distance in current state.
            collide_thresh: Self-collision threshold from GoalEML.
            success: Whether task goal was achieved this step.

        Returns:
            Dict with keys: 'noether', 'contact', 'task', 'hinge', 'total'.
        """
        cfg = self.config

        # ── η_Noether: energy drift beyond budget (clamped to ≥0) ──
        eta_noether: float = max(0.0, energy_drift - energy_budget)

        # ── η_contact: collision penetration depth (clamped to ≥0) ──
        eta_contact: float = max(0.0, collide_thresh - min_geom_dist)

        # ── η_task: κ-Snap residual (already ≥0) ──
        eta_task_val: float = float(eta_task)

        # ── hinge(task_success_pred): max(0, margin - success_indicator) ──
        success_indicator: float = 1.0 if success else 0.0
        hinge_loss: float = max(0.0, cfg.hinge_margin - success_indicator)

        # ── Normalization (optional) ──
        if cfg.normalize:
            # Normalize each component to roughly [0, 1] using soft caps
            eta_noether_norm = float(np.tanh(eta_noether / 100.0))
            eta_contact_norm = float(np.tanh(eta_contact / 0.1))
            eta_task_norm = float(np.tanh(eta_task_val / 5.0))
            hinge_norm = hinge_loss  # Already in [0, hinge_margin]
        else:
            eta_noether_norm = eta_noether
            eta_contact_norm = eta_contact
            eta_task_norm = eta_task_val
            hinge_norm = hinge_loss

        # ── Weighted sum ──
        loss_noether: float = cfg.lambda_noether * eta_noether_norm ** 2
        loss_contact: float = cfg.lambda_contact * eta_contact_norm ** 2
        loss_task: float = cfg.lambda_task * eta_task_norm ** 2
        loss_hinge: float = cfg.lambda_hinge * hinge_norm ** 2

        total: float = loss_noether + loss_contact + loss_task + loss_hinge

        return {
            "noether": loss_noether,
            "contact": loss_contact,
            "task": loss_task,
            "hinge": loss_hinge,
            "total": total,
            # Raw (unweighted, unnormalized) values for logging
            "raw_noether": eta_noether,
            "raw_contact": eta_contact,
            "raw_task": eta_task_val,
            "raw_hinge": hinge_loss,
        }

    def accumulate(self, step_result: Dict[str, float]) -> None:
        """Accumulate a step's GEL result into episode totals.

        Args:
            step_result: Output from compute_step().
        """
        self.step_count += 1
        self.total_loss += step_result.get("total", 0.0)
        for key in ("noether", "contact", "task", "hinge"):
            self.component_totals[key] += step_result.get(key, 0.0)

    def mean_loss(self) -> Dict[str, float]:
        """Compute mean GEL loss over accumulated steps.

        Returns:
            Dict with mean total and per-component losses.
        """
        n: int = max(1, self.step_count)
        return {
            "total": self.total_loss / n,
            "noether": self.component_totals["noether"] / n,
            "contact": self.component_totals["contact"] / n,
            "task": self.component_totals["task"] / n,
            "hinge": self.component_totals["hinge"] / n,
            "steps": self.step_count,
        }

    def reset(self) -> None:
        """Reset accumulator for a new episode."""
        self.step_count = 0
        self.total_loss = 0.0
        self.component_totals = {
            "noether": 0.0,
            "contact": 0.0,
            "task": 0.0,
            "hinge": 0.0,
        }


# ── κ-Phase 3: Convenience function for server integration ──

def compute_gel_from_step(
    eta: float,
    noether_result: Optional[Dict[str, Any]] = None,
    goal_max_energy: float = 500.0,
    collide_thresh: float = 0.01,
    success: bool = False,
    config: Optional[GELConfig] = None,
) -> Dict[str, float]:
    """Compute GEL loss from per-step IDO signals.

    Convenience wrapper that extracts energy drift and collision info
    from a Noether check result dict.

    Args:
        eta: κ-Snap residual value (gauss_ex_residual output).
        noether_result: Output dict from noether_check_mj(). If None,
            assumes no Noether violations.
        goal_max_energy: max_energy_inject from GoalEML.
        collide_thresh: Self-collision threshold from GoalEML.
        success: Whether the task goal was achieved this step.
        config: Optional GELConfig. Defaults to GELConfig().

    Returns:
        Dict with GEL loss components (same as GELLoss.compute_step).
    """
    cfg = config or GELConfig()
    gel = GELLoss(config=cfg)

    energy_drift: float = 0.0
    min_dist: float = 1.0  # Default: no collision

    if noether_result is not None:
        # Extract energy drift from message or direct computation
        # noether_result has 'energy': 0/1 (gate pass/fail)
        # We approximate η_Noether from the gate status
        if noether_result.get("energy", 0) > 0:
            # Energy gate failed — use budget as drift estimate
            energy_drift = goal_max_energy * 1.1
        if noether_result.get("collision", 0) > 0:
            # Collision gate failed — penetration exists
            min_dist = 0.0  # Maximum penetration

    return gel.compute_step(
        eta_task=eta,
        energy_drift=energy_drift,
        energy_budget=goal_max_energy,
        min_geom_dist=min_dist,
        collide_thresh=collide_thresh,
        success=success,
    )


# ── Self-test ──

def _self_test() -> None:
    """Run self-test for GEL loss computation."""
    print("=== GEL Loss Self-Test ===")

    gel = GELLoss(config=GELConfig())

    # Test 1: No violations, no task residual
    result = gel.compute_step(
        eta_task=0.0, energy_drift=0.0, energy_budget=500.0,
        min_geom_dist=1.0, collide_thresh=0.01, success=True,
    )
    assert result["total"] < 0.01, f"Expected near-zero loss, got {result['total']}"
    print(f"  Test 1 (all clear): total={result['total']:.6f} [OK]")

    # Test 2: Task residual only
    result = gel.compute_step(
        eta_task=2.5, energy_drift=0.0, energy_budget=500.0,
        min_geom_dist=1.0, collide_thresh=0.01, success=False,
    )
    assert result["task"] > 0.0, "Expected non-zero task loss"
    assert result["hinge"] > 0.0, "Expected non-zero hinge loss (not success)"
    print(f"  Test 2 (task residual): task={result['task']:.6f}, hinge={result['hinge']:.6f} [OK]")

    # Test 3: Energy violation
    result = gel.compute_step(
        eta_task=0.0, energy_drift=600.0, energy_budget=500.0,
        min_geom_dist=1.0, collide_thresh=0.01, success=False,
    )
    assert result["noether"] > 0.0, "Expected non-zero Noether loss"
    print(f"  Test 3 (energy violation): noether={result['noether']:.6f} [OK]")

    # Test 4: Collision
    result = gel.compute_step(
        eta_task=0.0, energy_drift=0.0, energy_budget=500.0,
        min_geom_dist=0.005, collide_thresh=0.01, success=False,
    )
    assert result["contact"] > 0.0, "Expected non-zero contact loss"
    print(f"  Test 4 (collision): contact={result['contact']:.6f} [OK]")

    # Test 5: Accumulation and mean
    gel.reset()
    for _ in range(10):
        gel.accumulate(gel.compute_step(
            eta_task=1.0, energy_drift=0.0, energy_budget=500.0,
            min_geom_dist=1.0, collide_thresh=0.01, success=False,
        ))
    mean = gel.mean_loss()
    assert mean["steps"] == 10, f"Expected 10 steps, got {mean['steps']}"
    assert mean["total"] > 0.0, "Expected positive mean loss"
    print(f"  Test 5 (accumulation): mean_total={mean['total']:.6f}, steps={mean['steps']} [OK]")

    # Test 6: compute_gel_from_step convenience function
    noether_result = {"ok": False, "total": 2, "energy": 1, "collision": 1}
    result = compute_gel_from_step(
        eta=1.5, noether_result=noether_result,
        goal_max_energy=500.0, collide_thresh=0.01, success=False,
    )
    assert result["noether"] > 0.0, "Expected Noether loss from violation"
    assert result["contact"] > 0.0, "Expected contact loss from collision"
    print(f"  Test 6 (convenience fn): total={result['total']:.6f} [OK]")

    print("=== All GEL tests passed ===")


if __name__ == "__main__":
    _self_test()
