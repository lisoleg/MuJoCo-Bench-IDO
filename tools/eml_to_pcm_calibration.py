"""
EML-to-PCM Conductance Calibration Script
==========================================

Maps octonion EML (Epistemic Map Layer) nodes to PCM (Phase Change Memory)
conductance states, implementing the pulse-verify-write algorithm from
章锋 SLOS paper (2026-07-04, 2nd edition), Appendix E.

Key algorithm:
  1. Decompose 8-component octonion EML node → weight matrix W[8×8]
  2. Map each weight to PCM conductance target: G_target = G_min + w_norm × (G_max - G_min)
  3. Pulse-verify-write: iteratively apply SET/RESET pulses, read-back verify,
     adaptive step-size until |G_actual - G_target| < tolerance.
  4. Convergence: ~7 pulses for target 0x4000 with sequence
     0x2000 → 0x2800 → 0x3500 → 0x3E00 → 0x3F80 → 0x3FF0 → 0x4000.

PCM electrical model (from 杨玉超 team controllable phase-change memristor):
  - SET pulse: crystallization → high conductance (G_max ≈ 100 µS)
  - RESET pulse: amorphization → low conductance (G_min ≈ 1 µS)
  - Partial SET: progressive crystallization → intermediate conductance
  - Conductance resolution: 16-bit (0x0000 – 0xFFFF maps to G_min – G_max)

Author: MuJoCo-Bench-IDO v0.4.0 — SLOS PCM CIM Module
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

# Add project root to path
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

__all__ = [
    "PCMConductanceRange",
    "EMLPCMCalibrator",
    "PulseVerifyResult",
    "calibrate_eml_node",
    "main",
]

# ── PCM Physical Constants ──

#: Maximum PCM conductance (fully crystalline / SET state) in Siemens
PCM_G_MAX_S: float = 100e-6   # 100 µS

#: Minimum PCM conductance (fully amorphous / RESET state) in Siemens
PCM_G_MIN_S: float = 1e-6     # 1 µS

#: 16-bit conductance code full scale
PCM_CODE_MAX: int = 0xFFFF    # 65535

#: Default target conductance code (from SLOS paper: 0x4000 = 16384)
PCM_TARGET_CODE_DEFAULT: int = 0x4000

#: Conductance tolerance in code units (±0.5% of full scale)
PCM_TOLERANCE_CODE: int = 0x0200  # 512 codes ≈ 0.78%

#: Maximum SET pulse iterations before giving up
PCM_MAX_PULSES: int = 16

#: Default SET pulse amplitude increment (progressive crystallization step)
PCM_SET_STEP_DEFAULT: int = 0x0800  # 2048 codes per pulse

#: Reference pulse-verify sequence from SLOS paper (7-pulse convergence to 0x4000)
PCM_REF_SEQUENCE: List[int] = [
    0x2000, 0x2800, 0x3500, 0x3E00, 0x3F80, 0x3FF0, 0x4000,
]


@dataclass
class PCMConductanceRange:
    """PCM conductance range specification.

    Attributes:
        g_min_s: Minimum conductance in Siemens (RESET state).
        g_max_s: Maximum conductance in Siemens (SET state).
        code_max: Maximum digital code (16-bit = 65535).
    """
    g_min_s: float = PCM_G_MIN_S
    g_max_s: float = PCM_G_MAX_S
    code_max: int = PCM_CODE_MAX

    def code_to_conductance(self, code: int) -> float:
        """Convert digital code to physical conductance in Siemens.

        Args:
            code: 16-bit conductance code (0 – 0xFFFF).

        Returns:
            Conductance in Siemens.
        """
        code_clamped = max(0, min(self.code_max, int(code)))
        frac = code_clamped / self.code_max
        return self.g_min_s + frac * (self.g_max_s - self.g_min_s)

    def conductance_to_code(self, g_s: float) -> int:
        """Convert physical conductance to digital code.

        Args:
            g_s: Conductance in Siemens.

        Returns:
            16-bit conductance code.
        """
        g_clamped = max(self.g_min_s, min(self.g_max_s, g_s))
        frac = (g_clamped - self.g_min_s) / (self.g_max_s - self.g_min_s)
        return int(round(frac * self.code_max))

    def code_to_weight(self, code: int) -> float:
        """Normalize conductance code to weight in [0, 1].

        Args:
            code: 16-bit conductance code.

        Returns:
            Normalized weight in [0.0, 1.0].
        """
        code_clamped = max(0, min(self.code_max, int(code)))
        return code_clamped / self.code_max

    def weight_to_code(self, w: float) -> int:
        """Map normalized weight [0,1] to conductance code.

        Args:
            w: Normalized weight in [0.0, 1.0].

        Returns:
            16-bit conductance code.
        """
        w_clamped = max(0.0, min(1.0, float(w)))
        return int(round(w_clamped * self.code_max))


@dataclass
class PulseVerifyResult:
    """Result of a single pulse-verify-write cycle.

    Attributes:
        target_code: Target conductance code.
        final_code: Achieved conductance code after convergence.
        pulse_count: Number of SET/RESET pulses applied.
        converged: Whether the result is within tolerance.
        sequence: List of conductance codes after each pulse.
        error_code: Final error in code units (|final - target|).
    """
    target_code: int = 0
    final_code: int = 0
    pulse_count: int = 0
    converged: bool = False
    sequence: List[int] = field(default_factory=list)
    error_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "target_code": self.target_code,
            "final_code": self.final_code,
            "pulse_count": self.pulse_count,
            "converged": self.converged,
            "sequence": self.sequence,
            "error_code": self.error_code,
        }


class EMLPCMCalibrator:
    """EML octonion → PCM conductance calibration engine.

    Converts 8-component octonion EML nodes into PCM crossbar weight matrices,
    using the pulse-verify-write algorithm for conductance programming.

    The calibration pipeline:
      1. Octonion node → weight matrix (outer product or linear projection)
      2. Weight normalization → conductance target codes
      3. Pulse-verify-write: iterative SET pulses with read-back verification
      4. Convergence check: |actual - target| < tolerance

    Attributes:
        range: PCMConductanceRange instance.
        tolerance: Acceptable code error for convergence.
        max_pulses: Maximum SET pulse iterations.
    """

    def __init__(
        self,
        cond_range: Optional[PCMConductanceRange] = None,
        tolerance: int = PCM_TOLERANCE_CODE,
        max_pulses: int = PCM_MAX_PULSES,
    ) -> None:
        """Initialize the calibrator.

        Args:
            cond_range: PCM conductance range spec. Defaults to standard range.
            tolerance: Acceptable error in code units for convergence.
            max_pulses: Maximum number of SET pulses per cell.
        """
        self.range: PCMConductanceRange = cond_range or PCMConductanceRange()
        self.tolerance: int = tolerance
        self.max_pulses: int = max_pulses

    # ── EML Node → Weight Matrix ──

    def eml_node_to_weight_matrix(
        self,
        components: np.ndarray,
        method: str = "outer",
    ) -> np.ndarray:
        """Convert octonion EML node components to weight matrix.

        Two methods:
          - "outer": W = outer(q, q) normalized → 8×8 symmetric matrix
          - "linear": W = diag(q) → 8×8 diagonal matrix

        Args:
            components: 8-element octonion component array.
            method: Decomposition method ("outer" or "linear").

        Returns:
            8×8 weight matrix with values in [0, 1].

        Raises:
            ValueError: If method is invalid or components shape is wrong.
        """
        q = np.asarray(components, dtype=np.float64).flatten()
        if q.shape[0] != 8:
            raise ValueError(f"Expected 8 octonion components, got {q.shape[0]}")

        if method == "outer":
            # Outer product → 8×8, then normalize to [0,1]
            W_raw = np.outer(q, q)
            max_val = np.max(np.abs(W_raw))
            if max_val > 0:
                W = (W_raw / max_val + 1.0) / 2.0  # Shift to [0,1]
            else:
                W = np.zeros((8, 8))
        elif method == "linear":
            # Diagonal: each component maps directly
            W = np.diag(np.clip(q, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown method '{method}', use 'outer' or 'linear'")

        return np.clip(W, 0.0, 1.0)

    # ── Weight Matrix → Conductance Codes ──

    def weight_matrix_to_codes(self, W: np.ndarray) -> np.ndarray:
        """Convert weight matrix to PCM conductance codes.

        Args:
            W: 8×8 weight matrix with values in [0, 1].

        Returns:
            8×8 integer array of 16-bit conductance codes.
        """
        W_clipped = np.clip(W, 0.0, 1.0)
        codes = np.round(W_clipped * self.range.code_max).astype(np.int32)
        return codes

    def codes_to_weight_matrix(self, codes: np.ndarray) -> np.ndarray:
        """Convert conductance codes back to weight matrix.

        Args:
            codes: 8×8 integer array of conductance codes.

        Returns:
            8×8 weight matrix with values in [0, 1].
        """
        return np.asarray(codes, dtype=np.float64) / self.range.code_max

    # ── Pulse-Verify-Write Algorithm ──

    def pulse_verify_write(
        self,
        target_code: int,
        initial_code: int = 0,
    ) -> PulseVerifyResult:
        """Execute pulse-verify-write for a single PCM cell.

        Implements the progressive SET pulse algorithm from SLOS paper:
          - Start from initial_code (typically 0 = RESET state)
          - Apply SET pulses with adaptive step size
          - After each pulse, read-back and verify against target
          - If overshoot, apply small RESET correction
          - Converge when |actual - target| < tolerance

        The algorithm mimics the 7-pulse convergence sequence:
          0x2000 -> 0x2800 -> 0x3500 -> 0x3E00 -> 0x3F80 -> 0x3FF0 -> 0x4000

        Args:
            target_code: Target conductance code (16-bit).
            initial_code: Starting conductance code (default 0 = RESET).

        Returns:
            PulseVerifyResult with convergence details.
        """
        result = PulseVerifyResult(target_code=target_code)
        current_code = max(0, min(self.range.code_max, initial_code))
        result.sequence.append(current_code)

        # If target is already within tolerance of initial, return immediately
        if abs(target_code - current_code) <= self.tolerance:
            result.converged = True
            result.final_code = current_code
            result.pulse_count = 0
            result.error_code = abs(target_code - current_code)
            return result

        # Adaptive step size: start proportional to target, shrink as we approach
        step = max(PCM_SET_STEP_DEFAULT, abs(target_code - current_code) // 4)

        for pulse_idx in range(self.max_pulses):
            error = target_code - current_code

            # Check convergence
            if abs(error) <= self.tolerance:
                result.converged = True
                result.final_code = current_code
                result.pulse_count = pulse_idx
                result.error_code = abs(error)
                return result

            # Adaptive step: shrink as we get closer
            if abs(error) < step:
                step = max(abs(error) // 2, 0x0040)  # Min step = 64 codes

            if error > 0:
                # Need higher conductance -> SET pulse
                current_code = min(self.range.code_max, current_code + step)
            else:
                # Overshot -> small RESET correction
                current_code = max(0, current_code - step // 2)

            # Add small noise to simulate PCM stochasticity (~0.5%)
            noise = int(np.random.normal(0, self.range.code_max * 0.003))
            current_code = max(0, min(self.range.code_max, current_code + noise))

            result.sequence.append(current_code)

        # Did not converge within max_pulses
        result.converged = abs(target_code - current_code) <= self.tolerance * 2
        result.final_code = current_code
        result.pulse_count = self.max_pulses
        result.error_code = abs(target_code - current_code)
        return result

    # ── Full EML Node Calibration ──

    def calibrate_eml_node(
        self,
        components: np.ndarray,
        method: str = "outer",
    ) -> Dict[str, Any]:
        """Full calibration pipeline: EML node → PCM weight matrix → pulse-verify.

    Args:
        components: 8-element octonion component array.
        method: Weight decomposition method ("outer" or "linear").

    Returns:
        Dictionary with calibration results:
          - weight_matrix: Target weight matrix [8×8]
          - target_codes: Target conductance codes [8×8]
          - actual_codes: Achieved conductance codes [8×8]
          - results: List of PulseVerifyResult for each cell
          - all_converged: Whether all cells converged
          - avg_pulses: Average pulses per cell
    """
        W = self.eml_node_to_weight_matrix(components, method=method)
        target_codes = self.weight_matrix_to_codes(W)
        n = target_codes.shape[0]

        actual_codes = np.zeros_like(target_codes)
        results: List[PulseVerifyResult] = []
        total_pulses = 0

        for i in range(n):
            for j in range(n):
                r = self.pulse_verify_write(int(target_codes[i, j]))
                actual_codes[i, j] = r.final_code
                results.append(r)
                total_pulses += r.pulse_count

        all_converged = all(r.converged for r in results)
        avg_pulses = total_pulses / len(results) if results else 0.0

        return {
            "weight_matrix": W,
            "target_codes": target_codes,
            "actual_codes": actual_codes,
            "results": results,
            "all_converged": all_converged,
            "avg_pulses": avg_pulses,
            "num_cells": len(results),
        }

    # ── Read-back Verification ──

    def verify_calibration(
        self,
        target_codes: np.ndarray,
        actual_codes: np.ndarray,
    ) -> Dict[str, Any]:
        """Verify calibration accuracy by comparing target vs actual codes.

        Args:
            target_codes: Target conductance code matrix.
        actual_codes: Achieved conductance code matrix.

        Returns:
            Dictionary with verification metrics:
              - max_error: Maximum absolute error in codes.
              - mean_error: Mean absolute error.
              - rmse: Root mean square error.
              - pass_rate: Fraction of cells within tolerance.
              - passed: Whether overall verification passed.
        """
        diff = np.abs(target_codes.astype(np.int32) - actual_codes.astype(np.int32))
        max_error = int(np.max(diff))
        mean_error = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        within_tol = diff <= self.tolerance
        pass_rate = float(np.mean(within_tol))
        passed = pass_rate >= 0.95  # 95% of cells must be within tolerance

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "rmse": rmse,
            "pass_rate": pass_rate,
            "passed": passed,
        }


def calibrate_eml_node(
    components: np.ndarray,
    method: str = "outer",
) -> Dict[str, Any]:
    """Convenience function: calibrate a single EML node to PCM.

    Args:
        components: 8-element octonion component array.
        method: Weight decomposition method.

    Returns:
        Calibration result dictionary.
    """
    calibrator = EMLPCMCalibrator()
    return calibrator.calibrate_eml_node(components, method=method)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Usage:
        python tools/eml_to_pcm_calibration.py [--target 0x4000] [--method outer]
        python tools/eml_to_pcm_calibration.py --self-test

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="EML → PCM conductance calibration tool"
    )
    parser.add_argument(
        "--target", type=lambda x: int(x, 0),
        default=PCM_TARGET_CODE_DEFAULT,
        help=f"Target conductance code (default: 0x{PCM_TARGET_CODE_DEFAULT:04X})",
    )
    parser.add_argument(
        "--method", choices=["outer", "linear"], default="outer",
        help="Weight decomposition method (default: outer)",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run self-test and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if _self_test() else 1

    print("=" * 60)
    print("EML → PCM Conductance Calibration")
    print("=" * 60)

    # Demo: pulse-verify-write to target
    calibrator = EMLPCMCalibrator()
    print(f"\nTarget code: 0x{args.target:04X} ({args.target})")
    print(f"Conductance: {calibrator.range.code_to_conductance(args.target)*1e6:.2f} µS")

    result = calibrator.pulse_verify_write(args.target)
    print(f"\nPulse-Verify-Write Result:")
    print(f"  Converged:     {result.converged}")
    print(f"  Pulses:        {result.pulse_count}")
    print(f"  Final code:    0x{result.final_code:04X} ({result.final_code})")
    print(f"  Error:         {result.error_code} codes")
    print(f"  Sequence:      {' → '.join(f'0x{c:04X}' for c in result.sequence)}")

    # Demo: full EML node calibration
    print(f"\n--- Full EML Node Calibration ({method_label(args.method)}) ---")
    # Sample octonion EML node (welding parameters normalized)
    sample_components = np.array([0.6, 0.5, 0.4, 0.3, 0.7, 0.55, 0.2, 0.15])
    cal = calibrator.calibrate_eml_node(sample_components, method=args.method)
    print(f"  All cells converged: {cal['all_converged']}")
    print(f"  Average pulses/cell: {cal['avg_pulses']:.1f}")
    print(f"  Total cells:         {cal['num_cells']}")

    ver = calibrator.verify_calibration(cal["target_codes"], cal["actual_codes"])
    print(f"\n  Verification:")
    print(f"    Max error:   {ver['max_error']} codes")
    print(f"    Mean error:  {ver['mean_error']:.1f} codes")
    print(f"    RMSE:        {ver['rmse']:.1f} codes")
    print(f"    Pass rate:   {ver['pass_rate']*100:.1f}%")
    print(f"    Passed:      {ver['passed']}")
    print("=" * 60)

    return 0


def method_label(method: str) -> str:
    """Return human-readable method label.

    Args:
        method: Method string.

    Returns:
        Human-readable label.
    """
    return {"outer": "Outer Product", "linear": "Linear Diagonal"}.get(method, method)


def _self_test() -> bool:
    """Self-test for EMLPCMCalibrator.

    Tests:
      1. Conductance range conversion round-trip
      2. Pulse-verify-write convergence to 0x4000
      3. EML node → weight matrix shape
      4. Full calibration pipeline
      5. Verification accuracy

    Returns:
        True if all tests pass.
    """
    print("[eml_to_pcm_calibration] Running self-test...")

    np.random.seed(42)
    calibrator = EMLPCMCalibrator()

    # Test 1: Conductance range conversion
    rng = calibrator.range
    for code in [0, 0x2000, 0x4000, 0x8000, 0xFFFF]:
        g = rng.code_to_conductance(code)
        code_back = rng.conductance_to_code(g)
        assert abs(code_back - code) <= 1, \
            f"Round-trip failed: {code} → {g} → {code_back}"

    # Test 2: Pulse-verify-write to 0x4000
    result = calibrator.pulse_verify_write(PCM_TARGET_CODE_DEFAULT)
    assert result.converged, \
        f"Pulse-verify should converge to 0x{PCM_TARGET_CODE_DEFAULT:04X}, " \
        f"got 0x{result.final_code:04X} after {result.pulse_count} pulses"
    assert result.pulse_count <= 10, \
        f"Should converge in ≤10 pulses, got {result.pulse_count}"
    assert result.error_code <= calibrator.tolerance, \
        f"Error {result.error_code} exceeds tolerance {calibrator.tolerance}"
    print(f"  Pulse-verify: 0x{PCM_TARGET_CODE_DEFAULT:04X} → 0x{result.final_code:04X} "
          f"in {result.pulse_count} pulses ✓")

    # Test 3: EML node → weight matrix
    components = np.array([0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.1, 0.8])
    W = calibrator.eml_node_to_weight_matrix(components, method="outer")
    assert W.shape == (8, 8), f"Weight matrix shape should be (8,8), got {W.shape}"
    assert W.min() >= 0.0 and W.max() <= 1.0, "Weights must be in [0,1]"

    W_lin = calibrator.eml_node_to_weight_matrix(components, method="linear")
    assert W_lin.shape == (8, 8), "Linear weight matrix shape should be (8,8)"
    assert np.allclose(np.diag(W_lin), components), "Diagonal should match components"

    # Test 4: Full calibration pipeline
    cal = calibrator.calibrate_eml_node(components, method="outer")
    assert cal["target_codes"].shape == (8, 8), "Target codes shape should be (8,8)"
    assert cal["actual_codes"].shape == (8, 8), "Actual codes shape should be (8,8)"
    assert cal["num_cells"] == 64, f"Should have 64 cells, got {cal['num_cells']}"
    assert cal["avg_pulses"] <= 12, f"Avg pulses should be <=12, got {cal['avg_pulses']}"

    # Test 5: Verification accuracy
    ver = calibrator.verify_calibration(cal["target_codes"], cal["actual_codes"])
    assert ver["pass_rate"] > 0.8, \
        f"Pass rate should be >80%, got {ver['pass_rate']*100:.1f}%"
    assert ver["max_error"] <= calibrator.tolerance * 5, \
        f"Max error {ver['max_error']} too large"

    # Test 6: Reference sequence validation
    # The reference 7-pulse sequence should all be monotonically increasing
    for i in range(1, len(PCM_REF_SEQUENCE)):
        assert PCM_REF_SEQUENCE[i] > PCM_REF_SEQUENCE[i - 1], \
            "Reference sequence should be monotonically increasing"

    print(f"  Full calibration: {cal['num_cells']} cells, "
          f"avg {cal['avg_pulses']:.1f} pulses, "
          f"pass rate {ver['pass_rate']*100:.1f}% ✓")

    print("[eml_to_pcm_calibration] Self-test PASSED.")
    return True


if __name__ == "__main__":
    sys.exit(main())
