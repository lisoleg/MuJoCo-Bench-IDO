"""
Mao Rui Generalized Metric Space — Semantic Distance for EML Hypergraph
========================================================================

Implements the Mao Rui (毛睿) generalized metric space as the mathematical
foundation for TOMAS EML hypergraph semantic distance computation.

The Mao Rui metric extends classical metrics with three key relaxations
that match the informational dynamics of embodied cognition:

  1. Quasi-metric:  d(x,y) ≠ d(y,x)  — asymmetric perception vs action costs
     (G_ego阴敛 reading LATENT > G_ego阳发 writing MANIFEST)
  2. Relaxed Triangle Inequality:  d(x,z) ≤ d(x,y) + d(y,z) + φ
     φ = topological tension (EML unsynchronized: φ>0, synchronized: φ→0)
  3. Pseudo-metric:  d(x,y)=0 does NOT imply x=y
     (EML-indiscernible nodes: same predicate + same ℐ but different κ-read)

EML Semantic Distance (毛睿加权):
    d_sem(e) = 1/(ℐ(e) + ε) × w_base × f_dir

Where:
    ℐ(e)  — Information Existence Degree (TOMAS Axiom 1)
    w_base — Observation basis weight (sensor vs DMN)
    f_dir  — Direction factor (1=阳发 MANIFEST, >1=阴敛 LATENT ψ-read)
    ε     — Anti-zero constant

Properties:
    - High ℐ → small d_sem (near, core semantic)
    - Low ℐ → large d_sem (far, prune-able)
    - f_dir embodies asymmetry (阴敛 cost > 阳发 cost)
    - d_sem=0 with ℐ_equal → EML-indiscernible (isomorphic merger candidate)

Reference:
    Mao, R. (2020). Generalized Metric Spaces and Their Applications.
    Tsinghua University Press.
    Zhang, F. (2026). EML-SemZip + 毛睿度量 + 卞氏5/6.
    复合体理学 WeChat: mp.weixin.qq.com/s/2Jtk_WAqU0joCG39cTqLzg

Author: MuJoCo-Bench-IDO v0.6.4 — Mao Rui Metric Module
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

IDO_MAO_RUI_METRIC_VERSION: str = "v0.1.0"


@dataclass
class HyperEdge:
    """EML Hypergraph edge with Mao Rui semantic distance.

    Represents a hyperedge in the TOMAS EML hypergraph, carrying
    information existence degree ℐ, basis weight, and direction factor
    for Mao Rui metric computation.

    Attributes:
        eid: Edge identifier.
        predicate: Semantic predicate (e.g., 'grasp', 'walk', 'stand').
        I_value: Information Existence Degree ℐ(e) ∈ [0, 1].
        base_weight: Observation basis weight (physical sensor=1.0, psychological=0.7).
        dir_factor: Direction factor (1.0=阳发 MANIFEST, >1.0=阴敛 LATENT ψ-read).
        d_sem: Mao Rui semantic distance (computed in stage 3).
    """
    eid: str
    predicate: str
    I_value: float
    base_weight: float = 1.0
    dir_factor: float = 1.0  # 1=阳发, >1=阴敛(G_ego read ψ)
    d_sem: float = 0.0  # populated in stage 3


@dataclass
class MaoRuiConfig:
    """Configuration for Mao Rui metric computation.

    Attributes:
        epsilon: Anti-zero constant for ℐ denominator.
        theta_dead: Dead-zero pruning threshold (ℐ < theta_dead → noise).
        phi_default: Default topological tension for relaxed triangle inequality.
        keep_ratio: κ-Snap selection ratio (top-K percentage to keep).
    """
    epsilon: float = 1e-9
    theta_dead: float = 0.45
    phi_default: float = 0.1
    keep_ratio: float = 0.15


class MaoRuiMetric:
    """Mao Rui Generalized Metric computation for TOMAS EML hypergraph.

    Provides three metric relaxation properties:
    1. Quasi-metric: asymmetry via dir_factor
    2. Relaxed triangle inequality: φ tolerance
    3. Pseudo-metric: EML-indiscernible identification

    Main computation:
        d_sem(e) = 1/(ℐ(e) + ε) × w_base × f_dir
    """

    VERSION: str = IDO_MAO_RUI_METRIC_VERSION

    def __init__(self, config: Optional[MaoRuiConfig] = None) -> None:
        """Initialize Mao Rui metric with configuration."""
        self.config = config or MaoRuiConfig()

    def compute_d_sem(self, I_value: float,
                      base_weight: float = 1.0,
                      dir_factor: float = 1.0) -> float:
        """Compute Mao Rui semantic distance d_sem for a hyperedge.

        d_sem(e) = 1/(ℐ(e) + ε) × w_base × f_dir

        High ℐ → small d_sem (near, core semantic)
        Low ℐ → large d_sem (far, prune-able)
        f_dir embodies asymmetry (阴敛 cost > 阳发 cost)

        Args:
            I_value: Information Existence Degree ℐ(e) ∈ [0, 1].
            base_weight: Observation basis weight.
            dir_factor: Direction factor (1=阳发, >1=阴敛).

        Returns:
            Mao Rui semantic distance d_sem.
        """
        return (1.0 / (I_value + self.config.epsilon)) * base_weight * dir_factor

    def compute_d_sem_batch(self, edges: List[HyperEdge]) -> List[float]:
        """Compute d_sem for a batch of hyperedges (EML-SemZip Stage 3).

        Args:
            edges: List of HyperEdge instances with I_value populated.

        Returns:
            List of d_sem values, also populates edge.d_sem in-place.
        """
        results: List[float] = []
        for e in edges:
            e.d_sem = self.compute_d_sem(e.I_value, e.base_weight, e.dir_factor)
            results.append(e.d_sem)
        return results

    def is_indiscernible(self, e1: HyperEdge, e2: HyperEdge,
                         tol: float = 0.01) -> bool:
        """Check if two hyperedges are EML-indiscernible (pseudo-metric d=0).

        Two edges are EML-indiscernible if:
        - Same predicate
        - Same ℐ (within tolerance)
        - Same node type

        Args:
            e1, e2: HyperEdge instances to compare.
            tol: Tolerance for ℐ comparison.

        Returns:
            True if edges are EML-indiscernible (isomorphic merger candidate).
        """
        return (e1.predicate == e2.predicate
                and abs(e1.I_value - e2.I_value) < tol
                and e1.base_weight == e2.base_weight)

    def verify_triangle_inequality(self, d_xy: float, d_yz: float,
                                   d_xz: float,
                                   phi: Optional[float] = None) -> bool:
        """Verify relaxed triangle inequality: d(x,z) ≤ d(x,y) + d(y,z) + φ.

        Args:
            d_xy: d(x,y) distance.
            d_yz: d(y,z) distance.
            d_xz: d(x,z) distance.
            phi: Topological tension (default uses config.phi_default).

        Returns:
            True if relaxed triangle inequality holds.
        """
        if phi is None:
            phi = self.config.phi_default
        return d_xz <= d_xy + d_yz + phi

    def eml_semzip_stage3(self, edges: List[HyperEdge]) -> List[HyperEdge]:
        """EML-SemZip Stage 3: Mao Rui Metric Weighting.

        Populates d_sem for all edges using the Mao Rui formula.

        Args:
            edges: List of HyperEdge instances (after S1+S2 stages).

        Returns:
            Same list with d_sem populated.
        """
        self.compute_d_sem_batch(edges)
        return edges

    def eml_semzip_stage4(self, edges: List[HyperEdge]) -> List[HyperEdge]:
        """EML-SemZip Stage 4: κ-Snap Selection.

        Select Top-K by high ℐ (= low d_sem) + closure extension.

        Args:
            edges: List of weighted HyperEdge instances.

        Returns:
            κ-Snap semantic kernel (selected edges).
        """
        # Sort by ℐ descending (= d_sem ascending)
        sorted_edges = sorted(edges, key=lambda e: e.I_value, reverse=True)
        k = max(int(len(sorted_edges) * self.config.keep_ratio), 1)
        return list(sorted_edges[:k])

    def dead_zero_prune(self, edges: List[HyperEdge]) -> List[HyperEdge]:
        """EML-SemZip Stage 1: Dead-Zero Pruning (φ-filter).

        Remove edges with ℐ < theta_dead (topological noise).

        Args:
            edges: Full list of hyperedges.

        Returns:
            Pruned list with ℐ ≥ theta_dead.
        """
        return [e for e in edges if e.I_value >= self.config.theta_dead]

    def isomorphism_merge(self, edges: List[HyperEdge]) -> List[HyperEdge]:
        """EML-SemZip Stage 2: Isomorphism Merger (pseudo-metric).

        Merge EML-indiscernible edges (same predicate + same ℐ).

        Args:
            edges: Pruned hyperedge list.

        Returns:
            Merged list with duplicates removed.
        """
        merged: List[HyperEdge] = []
        for e in edges:
            is_dup = False
            for m in merged:
                if self.is_indiscernible(e, m):
                    is_dup = True
                    break
            if not is_dup:
                merged.append(e)
        return merged

    def full_compress(self, edges: List[HyperEdge]) -> Tuple[List[HyperEdge], dict]:
        """Execute full EML-SemZip 5-stage compression pipeline.

        S1: Dead-Zero Prune → S2: Isomorphism Merge →
        S3: Mao Rui Weighting → S4: κ-Snap Selection

        Args:
            edges: Full hyperedge list.

        Returns:
            (kernel_edges, metrics_dict) — selected kernel and compression stats.
        """
        n_original = len(edges)

        # S1: Dead-Zero Prune
        pruned = self.dead_zero_prune(edges)
        n_pruned = len(pruned)

        # S2: Isomorphism Merge
        merged = self.isomorphism_merge(pruned)
        n_merged = len(merged)

        # S3: Mao Rui Weighting
        weighted = self.eml_semzip_stage3(merged)

        # S4: κ-Snap Selection
        kernel = self.eml_semzip_stage4(weighted)
        n_kernel = len(kernel)

        # Metrics
        scr = n_original / max(n_kernel, 1)  # Semantic Compression Ratio
        metrics = {
            "n_original": n_original,
            "n_pruned": n_pruned,
            "n_merged": n_merged,
            "n_kernel": n_kernel,
            "scr": scr,
            "dead_zero_removed": n_original - n_pruned,
            "isomorphism_merged": n_pruned - n_merged,
            "k_snap_selected": n_kernel,
        }

        return kernel, metrics
