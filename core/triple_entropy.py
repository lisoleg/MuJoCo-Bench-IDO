"""
Triple-Entropy Unification — IDO/TOMAS Cognitive-Metabolic-Communication Loss
===============================================================================

Implements the IDO/TOMAS Triple-Entropy Unification framework that integrates
three entropy streams into a unified loss function for embodied intelligence:

  1. GaussEx Residual η (Cognitive Layer / P-Layer)
     — Goal co-set distance, the core of consciousness
     — Already implemented in core/kappa_snap_mj.py::gauss_ex_residual

  2. Thermodynamic Entropy S (Metabolic Layer)
     — Physical implementation energy cost (biological/silicon constraint)
     — Discrete: ΔS = Q/T, continuous: S = ∫dQ/T
     — Controlled by Bian's 5/6 folding saturation threshold

  3. Shannon Entropy H (Communication Layer)
     — Statistical uncertainty of information transmission
     — H(X) = -Σ p_i log₂(p_i)
     — Reduced by EML-SemZip semantic compression ratio SCR

Total Loss:
    𝓛_Total = α'·η + β'·S + γ'·H - log(SCR)

Dynamic weights α', β', γ' are adjusted by ψ-Anchor (PG-Gate):
    α' = α·(1 + ψ_scale·max(0, τ/τ_max - 1))     (violation → increase)
    β' = β·(1 + ψ_scale·max(0, S/S_max - 1))      (violation → increase)
    γ' = γ·(1 - ψ_scale·max(0, 1 - SCR/SCR_min))   (violation → decrease)

This module provides the unified loss computation and ψ-Anchor dynamic
weight adjustment for the TOMAS Triple-Entropy Trainer.

Reference:
    Zhang, F. (2026). 具身认知的三熵统一: IDO/TOMAS框架下基于毛睿度量与卞氏饱和阈值的意识动力学.
    复合体理学 WeChat: mp.weixin.qq.com/s/B0X2XFKRAW70DAM6YWPrYA

Author: MuJoCo-Bench-IDO v0.6.4 — Triple Entropy Module
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

IDO_TRIPLE_ENTROPY_VERSION: str = "v0.1.0"


@dataclass
class EntropyConfig:
    """Configuration for triple-entropy loss computation.

    Attributes:
        alpha: Cognitive weight (η term).
        beta: Metabolic weight (S term).
        gamma: Communication weight (H term).
        psi_scale: ψ-Anchor violation scaling factor.
        tau_max: Maximum torque threshold for α violation.
        S_max: Maximum thermodynamic entropy for β violation.
        SCR_min: Minimum semantic compression ratio for γ violation.
    """
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.1
    psi_scale: float = 2.0
    tau_max: float = 0.05  # N·m (finger joint, same as PsiAnchor)
    S_max: float = 1.0     # normalized
    SCR_min: float = 10.0  # minimum SCR for safe communication


class ShannonEntropy:
    """Shannon Entropy H — Communication Layer.

    H(X) = -Σ p_i · log₂(p_i)
    Measures statistical uncertainty of information transmission.
    EML-SemZip SCR reduces effective H.
    """

    def compute(self, logits: np.ndarray) -> float:
        """Compute Shannon entropy from logits (softmax → probs → H).

        Args:
            logits: Raw logit values (any shape, will be flattened).

        Returns:
            Shannon entropy H in bits.
        """
        flat = logits.flatten()
        # Softmax for probability
        exp_vals = np.exp(flat - np.max(flat))  # stability
        probs = exp_vals / (np.sum(exp_vals) + 1e-9)
        # Shannon entropy
        H = -np.sum(probs * np.log2(probs + 1e-9))
        return float(H)

    def compression_ratio(self, H_original: float,
                          H_compressed: float) -> float:
        """Compute compression ratio from original and compressed entropy.

        SCR = H_original / max(H_compressed, epsilon)

        Args:
            H_original: Shannon entropy of original representation.
            H_compressed: Shannon entropy of compressed representation.

        Returns:
            Semantic compression ratio SCR.
        """
        return H_original / max(H_compressed, 1e-9)


class ThermodynamicEntropy:
    """Thermodynamic Entropy S — Metabolic Layer.

    S = ∫dQ/T or discrete ΔS = Q/T
    Measures physical implementation energy cost.
    Connected to Bian's 5/6 folding saturation (n=5 → S dominates).
    """

    def compute(self, energy_joules: float,
                temperature: float = 300.0) -> float:
        """Compute thermodynamic entropy from energy and temperature.

        S = Q / T (discrete approximation)

        Args:
            energy_joules: Energy in Joules (e.g., motor energy).
            temperature: Temperature in Kelvin (default 300K for silicon).

        Returns:
            Thermodynamic entropy S.
        """
        if temperature <= 0:
            return float('inf')
        return energy_joules / temperature

    def compute_from_kinetic(self, qvel: np.ndarray,
                             mass_matrix: Optional[np.ndarray] = None) -> float:
        """Compute kinetic energy and thermodynamic entropy from velocity.

        E_kinetic = 0.5 × Σ m_i × v_i²
        S = E_kinetic / T

        Args:
            qvel: Joint velocities (nv,).
            mass_matrix: Mass/inertia matrix (nv, nv). If None, assume unit mass.

        Returns:
            Thermodynamic entropy S.
        """
        if mass_matrix is not None:
            KE = 0.5 * float(qvel @ mass_matrix @ qvel)
        else:
            KE = 0.5 * float(np.sum(qvel ** 2))
        return self.compute(KE)

    def folding_saturation_check(self, n: int,
                                 threshold_sr: float = 5.0 / 6.0) -> Dict:
        """Check Bian's 5/6 saturation for metabolic cost estimation.

        When n ≥ 5, metabolic cost (S) dominates marginal information gain.
        This triggers EML-SemZip aggressive pruning.

        Args:
            n: Abstraction/folding depth.
            threshold_sr: Saturation ratio threshold (default 5/6).

        Returns:
            Dict with SR, R_n, and whether saturated.
        """
        sr = n / (n + 1)
        r_n = 1.0 / (n * (n + 1)) if n > 0 else float('inf')
        return {
            "n": n,
            "SR": sr,
            "R_n": r_n,
            "is_saturated": sr >= threshold_sr,
        }


class PsiAnchorGate:
    """ψ-Anchor Dynamic Weight Adjustment — PG-Gate Soft Constraint.

    Real-time monitoring of violation behavior (torque, SCR, entropy),
    dynamically adjusting α, β, γ weights:

    α' = α·(1 + ψ_scale·max(0, τ/τ_max - 1))   violation → increase (force safety)
    β' = β·(1 + ψ_scale·max(0, S/S_max - 1))    violation → increase (punish waste)
    γ' = γ·(1 - ψ_scale·max(0, 1 - SCR/SCR_min)) violation → decrease (simplify to survive)
    """

    def __init__(self, config: Optional[EntropyConfig] = None) -> None:
        """Initialize ψ-Anchor gate with configuration."""
        self.config = config or EntropyConfig()

    def adjust_weights(self, tau: float, S: float,
                       SCR: float) -> Dict:
        """Dynamically adjust α, β, γ weights based on violations.

        Args:
            tau: Current torque magnitude.
            S: Current thermodynamic entropy.
            SCR: Current semantic compression ratio.

        Returns:
            Dict with adjusted α', β', γ' and violation flags.
        """
        c = self.config

        # Violation checks
        torque_violation = tau > c.tau_max
        entropy_violation = S > c.S_max
        scr_violation = SCR < c.SCR_min

        # Dynamic adjustments
        alpha_prime = c.alpha * (1 + c.psi_scale * max(0.0, tau / c.tau_max - 1.0))
        beta_prime = c.beta * (1 + c.psi_scale * max(0.0, S / c.S_max - 1.0))
        gamma_prime = c.gamma * (1 - c.psi_scale * max(0.0, 1.0 - SCR / c.SCR_min))

        # Ensure gamma doesn't go negative (clamp to 0)
        gamma_prime = max(0.0, gamma_prime)

        return {
            "alpha_prime": float(alpha_prime),
            "beta_prime": float(beta_prime),
            "gamma_prime": float(gamma_prime),
            "torque_violation": torque_violation,
            "entropy_violation": entropy_violation,
            "scr_violation": scr_violation,
            "any_violation": torque_violation or entropy_violation or scr_violation,
        }


class TripleEntropyLoss:
    """Triple-Entropy Unified Loss Function — 𝓛_Total.

    𝓛_Total = α'·η + β'·S + γ'·H - log(SCR)

    Combines cognitive (η), metabolic (S), and communication (H) entropy
    with ψ-Anchor dynamic weight adjustment.
    """

    VERSION: str = IDO_TRIPLE_ENTROPY_VERSION

    def __init__(self, config: Optional[EntropyConfig] = None) -> None:
        """Initialize triple-entropy loss with configuration."""
        self.config = config or EntropyConfig()
        self.shannon = ShannonEntropy()
        self.thermo = ThermodynamicEntropy()
        self.psi_gate = PsiAnchorGate(self.config)

    def compute(self,
                eta: float,
                energy_joules: float = 0.0,
                logits: Optional[np.ndarray] = None,
                tau: float = 0.0,
                SCR: float = 100.0,
                temperature: float = 300.0) -> Dict:
        """Compute full triple-entropy loss.

        Args:
            eta: GaussEx residual (cognitive layer).
            energy_joules: Physical energy (for thermodynamic entropy).
            logits: Action/control logits (for Shannon entropy).
            tau: Torque magnitude (for ψ-Anchor violation check).
            SCR: Semantic compression ratio.
            temperature: Temperature in Kelvin.

        Returns:
            Complete loss dict with all components and adjusted weights.
        """
        # S: Thermodynamic entropy
        S = self.thermo.compute(energy_joules, temperature)

        # H: Shannon entropy
        H = self.shannon.compute(logits) if logits is not None else 0.0

        # ψ-Anchor dynamic weights
        weights = self.psi_gate.adjust_weights(tau, S, SCR)

        # Total loss
        L_total = (
            weights["alpha_prime"] * eta
            + weights["beta_prime"] * S
            + weights["gamma_prime"] * H
            - np.log(max(SCR, 1.0))
        )

        return {
            "L_total": float(L_total),
            "eta": float(eta),
            "S": float(S),
            "H": float(H),
            "SCR": float(SCR),
            "alpha_prime": weights["alpha_prime"],
            "beta_prime": weights["beta_prime"],
            "gamma_prime": weights["gamma_prime"],
            "torque_violation": weights["torque_violation"],
            "entropy_violation": weights["entropy_violation"],
            "scr_violation": weights["scr_violation"],
            "any_violation": weights["any_violation"],
        }

    def compute_for_mujoco_step(self,
                                eta: float,
                                qvel: np.ndarray,
                                tau_ctrl: np.ndarray,
                                SCR: float = 100.0,
                                mass_matrix: Optional[np.ndarray] = None) -> Dict:
        """Compute triple-entropy loss for a MuJoCo simulation step.

        Convenience method that extracts energy and torque from MuJoCo data.

        Args:
            eta: GaussEx residual for this step.
            qvel: Joint velocities (for kinetic energy → thermodynamic entropy).
            tau_ctrl: Control torques (for ψ-Anchor violation check).
            SCR: Semantic compression ratio.
            mass_matrix: MuJoCo mass matrix (optional).

        Returns:
            Complete loss dict.
        """
        # S from kinetic energy
        S = self.thermo.compute_from_kinetic(qvel, mass_matrix)

        # τ max for violation
        tau_max_val = float(np.max(np.abs(tau_ctrl)))

        # H from control logits
        H = self.shannon.compute(tau_ctrl)

        # ψ-Anchor weights
        weights = self.psi_gate.adjust_weights(tau_max_val, S, SCR)

        # Total loss
        L_total = (
            weights["alpha_prime"] * eta
            + weights["beta_prime"] * S
            + weights["gamma_prime"] * H
            - np.log(max(SCR, 1.0))
        )

        return {
            "L_total": float(L_total),
            "eta": float(eta),
            "S": float(S),
            "H": float(H),
            "SCR": float(SCR),
            "alpha_prime": weights["alpha_prime"],
            "beta_prime": weights["beta_prime"],
            "gamma_prime": weights["gamma_prime"],
            "tau_max": tau_max_val,
            "torque_violation": weights["torque_violation"],
            "entropy_violation": weights["entropy_violation"],
            "scr_violation": weights["scr_violation"],
            "any_violation": weights["any_violation"],
        }
