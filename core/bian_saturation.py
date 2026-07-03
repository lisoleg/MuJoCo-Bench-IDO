"""
Bian's 5/6 Folding Saturation Threshold — Information Capacity Saturation
==========================================================================

Implements Bian's (卞氏) 5/6 Folding Saturation Theorem for 2-dimensional
developable manifolds embedded in 3-ball with n self-similar non-self-intersecting
folds.

Theorem (Bian's 5/6):
    C(n) = C_max × n/(n+1)
    Marginal gain R_n = 1/(n(n+1))
    Saturation Ratio SR(n) = n/(n+1)

    At n=5: SR = 5/6 ≈ 83.33%, R_5 = 1/30 ≈ 0.033
    R_5 vs R_4: 33% decrease in marginal gain

    This is the empirical saturation ratio under typical biological/silicon
    3D-packaging constraints (RC wire delay dominates when folds exceed
    critical thickness).

IDO/TOMAS Interpretation:
    - Folding = EML hypergraph embedding depth (layers of abstraction)
    - C(n) = effective information capacity at fold depth n
    - R_n = marginal information gain per additional fold
    - At n=5, EML-SemZip triggers aggressive Dead-Zero pruning
      (new superedges don't reduce d_sem → redundant → prune)
    - Corresponds to Mao Rui metric: at saturation, adding edges
      has zero semantic distance gain → prune-able

    This connects Bian's fractal geometry to IDO information dynamics:
    the 5/6 threshold is the pruning trigger point where
    EML-SemZip switches from conservative to aggressive mode.

Reference:
    Bian, W. (2026). Fractional Geometry of Folding Saturation: The 5/6 Limit.
    Zhang, F. (2026). 三熵统一: 具身认知的三熵统一.
    复合体理学 WeChat: mp.weixin.qq.com/s/B0X2XFKRAW70DAM6YWPrYA

Author: MuJoCo-Bench-IDO v0.6.4 — Bian Saturation Module
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

IDO_BIAN_SATURATION_VERSION: str = "v0.1.0"


@dataclass
class BianConfig:
    """Configuration for Bian's saturation computation.

    Attributes:
        c_max: Maximum information capacity (normalized to 1.0).
        cost_alpha: Cost scaling exponent (linear=1.0, superlinear>1.0).
                Biological neural: ~1.5 (RC delay superlinear)
                Silicon 3D-IC: ~1.2 (moderate superlinear)
        critical_n: Critical fold count for saturation trigger (default=5).
        saturation_ratio_threshold: SR threshold for aggressive pruning.
    """
    c_max: float = 1.0
    cost_alpha: float = 1.5  # biological neural superlinear
    critical_n: int = 5
    saturation_ratio_threshold: float = 5.0 / 6.0


class BianSaturation:
    """Bian's 5/6 Folding Saturation Threshold computation.

    Computes information capacity C(n), marginal gain R_n,
    and saturation ratio SR for folding depth n.

    Key insight: at n=5 (5/6 saturation), EML-SemZip should
    trigger aggressive Dead-Zero pruning because additional
    folding/abstraction layers provide diminishing returns.

    Theorem:
        C(n) = C_max × n/(n+1)
        R_n = 1/(n(n+1))
        SR(n) = n/(n+1) = C(n)/C_max
    """

    VERSION: str = IDO_BIAN_SATURATION_VERSION

    def __init__(self, config: Optional[BianConfig] = None) -> None:
        """Initialize Bian saturation with configuration."""
        self.config = config or BianConfig()

    def compute_capacity(self, n: int) -> float:
        """Compute information capacity C(n) = C_max × n/(n+1).

        Args:
            n: Number of self-similar folds.

        Returns:
            Information capacity at fold depth n.
        """
        if n <= 0:
            return 0.0
        return self.config.c_max * n / (n + 1)

    def compute_marginal_gain(self, n: int) -> float:
        """Compute marginal information gain R_n = 1/(n(n+1)).

        Args:
            n: Fold depth (R_n is the gain of going from n-1 to n).

        Returns:
            Marginal gain at fold n.
        """
        if n <= 0:
            return float('inf')
        return 1.0 / (n * (n + 1))

    def compute_saturation_ratio(self, n: int) -> float:
        """Compute saturation ratio SR(n) = n/(n+1).

        SR(5) = 5/6 ≈ 83.33% — the Bian's critical threshold.

        Args:
            n: Number of folds.

        Returns:
            Saturation ratio ∈ [0, 1).
        """
        if n <= 0:
            return 0.0
        return n / (n + 1)

    def compute_cost(self, n: int) -> float:
        """Compute folding cost C_cost(n) = α × n^(cost_alpha).

        Biological: α=1.5 (RC delay dominates, superlinear)
        Silicon: α=1.2 (moderate superlinear)
        Idealized: α=1.0 (linear, rare)

        Args:
            n: Number of folds.

        Returns:
            Folding cost at depth n.
        """
        return self.config.cost_alpha * (n ** self.config.cost_alpha)

    def compute_net_benefit_ratio(self, n: int) -> float:
        """Compute net benefit ratio = (gain - cost) / gain.

        At Bian's critical n=5, net benefit starts declining
        because cost (superlinear) exceeds marginal gain (hyperbolic).

        Args:
            n: Number of folds.

        Returns:
            Net benefit ratio.
        """
        capacity = self.compute_capacity(n)
        cost = self.compute_cost(n)
        if capacity <= 0:
            return 0.0
        return max(0.0, (capacity - cost) / capacity)

    def is_saturated(self, n: int) -> bool:
        """Check if fold depth n exceeds Bian's 5/6 saturation threshold.

        Args:
            n: Number of folds.

        Returns:
            True if SR(n) ≥ 5/6 threshold (system is saturated).
        """
        return self.compute_saturation_ratio(n) >= self.config.saturation_ratio_threshold

    def critical_n_for_sr(self, target_sr: float) -> int:
        """Find minimum n that achieves target saturation ratio.

        SR(n) = n/(n+1) ≥ target_sr
        → n ≥ target_sr / (1 - target_sr)

        Args:
            target_sr: Target saturation ratio (e.g., 5/6).

        Returns:
            Minimum fold depth n achieving target SR.
        """
        if target_sr >= 1.0 or target_sr <= 0:
            return 0
        n_min = target_sr / (1.0 - target_sr)
        return math.ceil(n_min)

    def should_trigger_aggressive_pruning(self, n: int) -> bool:
        """Whether EML-SemZip should switch to aggressive Dead-Zero pruning.

        At Bian's saturation point (n ≥ 5), additional folds/abstractions
        provide diminishing returns. EML-SemZip should prune aggressively.

        Args:
            n: Current abstraction/folding depth.

        Returns:
            True if aggressive pruning should be triggered.
        """
        return self.is_saturated(n)

    def compute_table(self, max_n: int = 20) -> List[Dict]:
        """Compute full saturation table from n=1 to max_n.

        Args:
            max_n: Maximum fold depth to compute.

        Returns:
            List of dicts with n, C(n), R_n, SR, is_saturated.
        """
        results: List[Dict] = []
        for n in range(1, max_n + 1):
            results.append({
                "n": n,
                "C_n": self.compute_capacity(n),
                "R_n": self.compute_marginal_gain(n),
                "SR": self.compute_saturation_ratio(n),
                "cost": self.compute_cost(n),
                "net_benefit_ratio": self.compute_net_benefit_ratio(n),
                "is_saturated": self.is_saturated(n),
            })
        return results

    def get_report(self) -> Dict:
        """Generate Bian saturation summary report.

        Returns:
            Dict with critical thresholds and current config.
        """
        critical_n = self.config.critical_n
        return {
            "version": self.VERSION,
            "critical_n": critical_n,
            "SR_at_critical": self.compute_saturation_ratio(critical_n),
            "R_at_critical": self.compute_marginal_gain(critical_n),
            "R_decrease_vs_prev": self.compute_marginal_gain(critical_n) / self.compute_marginal_gain(critical_n - 1),
            "c_max": self.config.c_max,
            "cost_alpha": self.config.cost_alpha,
            "config": self.config,
        }
