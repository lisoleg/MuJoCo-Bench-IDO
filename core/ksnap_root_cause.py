"""
κ-Snap Root Cause Code Generator
================================

Generates causal root-cause codes from η residual excursions and multimodal
sensor signals, implementing the κ-Snap Root Cause framework from
章锋 SLOS paper (2026-07-04, 2nd edition), Section 5.

When η residual exceeds a threshold, the system:
  1. Captures ±100ms of full multimodal data (current, voltage, arc, gas, temp)
  2. Runs causal inference to identify the root cause
  3. Generates a structured RootCauseCode with cause, action, confidence
  4. Feeds back to TOMAS → adaptive library → distill new EML node → update T-Processor

Example output:
  RootCause: Gas_Contamination; Action: Increase_Flow_20%; Confidence: 0.94

Root cause types (from SLOS welding domain):
  - Gas_Contamination: Shielding gas flow too low or contaminated
  - Wire_Stick: Wire stubbing / burn-back / sticking to workpiece
  - Arc_Instability: Arc wandering, unstable arc length
  - Low_Penetration: Insufficient heat input, shallow fusion
  - Excess_Spatter: Excessive spatter generation
  - Contact_Tube_Wear: Contact tip wear causing erratic wire feed
  - Voltage_Drop: Power supply voltage sag
  - Plate_Contamination: Surface oil/rust/moisture

Author: MuJoCo-Bench-IDO v0.4.0 — SLOS κ-Snap Root Cause Module
"""

from __future__ import annotations

import time
import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

import numpy as np

# Add project root to path
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

__all__ = [
    "RootCauseType",
    "RootCauseCode",
    "KSnapRootCauseGenerator",
    "MultimodalSnapshot",
    "generate_root_cause",
    "main",
]


class RootCauseType(Enum):
    """Enumeration of welding root cause types.

    Each type corresponds to a specific physical failure mode
    detectable from multimodal sensor signals.
    """

    GAS_CONTAMINATION = "Gas_Contamination"
    WIRE_STICK = "Wire_Stick"
    ARC_INSTABILITY = "Arc_Instability"
    LOW_PENETRATION = "Low_Penetration"
    EXCESS_SPATTER = "Excess_Spatter"
    CONTACT_TUBE_WEAR = "Contact_Tube_Wear"
    VOLTAGE_DROP = "Voltage_Drop"
    PLATE_CONTAMINATION = "Plate_Contamination"
    UNKNOWN = "Unknown"


@dataclass
class MultimodalSnapshot:
    """Snapshot of multimodal sensor data at the moment of η excursion.

    Captures ±100ms of sensor data when η exceeds threshold, providing
    the input for causal root-cause inference.

    Attributes:
        timestamp: Time of the excursion event (seconds).
        current_a: Welding current array (A).
        voltage_v: Welding voltage array (V).
        arc_length_mm: Arc length array (mm).
        gas_flow_lpm: Shielding gas flow array (L/min).
        temperature_c: Temperature array (°C).
        wire_feed_mpm: Wire feed speed array (m/min).
        eta_residual: η residual at the excursion point.
        nominal_current: Nominal/target current (A).
        nominal_voltage: Nominal/target voltage (V).
        nominal_gas_flow: Nominal gas flow (L/min).
    """
    timestamp: float = 0.0
    current_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    voltage_v: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    arc_length_mm: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    gas_flow_lpm: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    temperature_c: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    wire_feed_mpm: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    eta_residual: float = 0.0
    nominal_current: float = 200.0
    nominal_voltage: float = 24.0
    nominal_gas_flow: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (arrays → lists for JSON serialization).

        Returns:
            Dictionary representation.
        """
        return {
            "timestamp": self.timestamp,
            "current_a": self.current_a.tolist(),
            "voltage_v": self.voltage_v.tolist(),
            "arc_length_mm": self.arc_length_mm.tolist(),
            "gas_flow_lpm": self.gas_flow_lpm.tolist(),
            "temperature_c": self.temperature_c.tolist(),
            "wire_feed_mpm": self.wire_feed_mpm.tolist(),
            "eta_residual": self.eta_residual,
            "nominal_current": self.nominal_current,
            "nominal_voltage": self.nominal_voltage,
            "nominal_gas_flow": self.nominal_gas_flow,
        }


@dataclass
class RootCauseCode:
    """Structured root-cause code generated from κ-Snap excursion.

    Follows the SLOS paper format:
      RootCause: <cause>; Action: <action>; Confidence: <confidence>

    Attributes:
        cause: Root cause type string (e.g., "Gas_Contamination").
        action: Recommended corrective action string.
        confidence: Confidence score [0, 1].
        timestamp: Event timestamp.
        eta_residual: η residual that triggered the analysis.
        evidence: Dictionary of evidence signals and their values.
        root_cause_type: Enum value for programmatic use.
    """
    cause: str = "Unknown"
    action: str = "Investigate"
    confidence: float = 0.0
    timestamp: float = 0.0
    eta_residual: float = 0.0
    evidence: Dict[str, float] = field(default_factory=dict)
    root_cause_type: RootCauseType = RootCauseType.UNKNOWN

    def format_string(self) -> str:
        """Format as the SLOS paper output string.

        Returns:
            Formatted string: "RootCause: X; Action: Y; Confidence: Z"
        """
        return (
            f"RootCause: {self.cause}; "
            f"Action: {self.action}; "
            f"Confidence: {self.confidence:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "cause": self.cause,
            "action": self.action,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "eta_residual": self.eta_residual,
            "evidence": self.evidence,
            "root_cause_type": self.root_cause_type.value,
            "formatted": self.format_string(),
        }


class KSnapRootCauseGenerator:
    """κ-Snap root cause code generator.

    Analyzes multimodal sensor snapshots to identify the root cause of
    η residual excursions, using rule-based causal inference with
    weighted evidence scoring.

    The inference engine evaluates each candidate root cause against
    the sensor data, computing an evidence score for each. The cause
    with the highest score (above a minimum threshold) is selected.

    Attributes:
        eta_threshold: η residual threshold that triggers root cause analysis.
        min_confidence: Minimum confidence to accept a root cause.
    """

    # ── Default actions for each root cause type ──
    DEFAULT_ACTIONS: Dict[RootCauseType, str] = {
        RootCauseType.GAS_CONTAMINATION: "Increase_Flow_20%",
        RootCauseType.WIRE_STICK: "Reduce_Wire_Feed_15%",
        RootCauseType.ARC_INSTABILITY: "Adjust_Voltage_+2V",
        RootCauseType.LOW_PENETRATION: "Increase_Current_10%",
        RootCauseType.EXCESS_SPATTER: "Reduce_Current_5%_Increase_Gas_10%",
        RootCauseType.CONTACT_TUBE_WEAR: "Replace_Contact_Tip",
        RootCauseType.VOLTAGE_DROP: "Check_Power_Supply_Cables",
        RootCauseType.PLATE_CONTAMINATION: "Clean_Surface_Area",
        RootCauseType.UNKNOWN: "Manual_Investigation_Required",
    }

    def __init__(
        self,
        eta_threshold: float = 0.5,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize the root cause generator.

        Args:
            eta_threshold: η residual threshold for triggering analysis.
            min_confidence: Minimum confidence to accept a root cause.
        """
        self.eta_threshold: float = eta_threshold
        self.min_confidence: float = min_confidence

    def should_trigger(self, eta_residual: float) -> bool:
        """Check if η residual exceeds the threshold for root cause analysis.

        Args:
            eta_residual: Current η residual value.

        Returns:
            True if analysis should be triggered.
        """
        return eta_residual > self.eta_threshold

    def analyze(self, snapshot: MultimodalSnapshot) -> RootCauseCode:
        """Analyze a multimodal snapshot and generate a root cause code.

        Runs the causal inference engine against all candidate root causes,
        scoring each based on evidence from the sensor data.

        Args:
            snapshot: Multimodal sensor snapshot at the excursion point.

        Returns:
            RootCauseCode with the identified cause, action, and confidence.
        """
        # Compute evidence scores for each root cause
        scores: Dict[RootCauseType, Tuple[float, Dict[str, float]]] = {}

        scores[RootCauseType.GAS_CONTAMINATION] = self._score_gas_contamination(snapshot)
        scores[RootCauseType.WIRE_STICK] = self._score_wire_stick(snapshot)
        scores[RootCauseType.ARC_INSTABILITY] = self._score_arc_instability(snapshot)
        scores[RootCauseType.LOW_PENETRATION] = self._score_low_penetration(snapshot)
        scores[RootCauseType.EXCESS_SPATTER] = self._score_excess_spatter(snapshot)
        scores[RootCauseType.VOLTAGE_DROP] = self._score_voltage_drop(snapshot)
        scores[RootCauseType.CONTACT_TUBE_WEAR] = self._score_contact_tube_wear(snapshot)
        scores[RootCauseType.PLATE_CONTAMINATION] = self._score_plate_contamination(snapshot)

        # Select the root cause with the highest score
        best_type = RootCauseType.UNKNOWN
        best_score = 0.0
        best_evidence: Dict[str, float] = {}

        for rct, (score, evidence) in scores.items():
            if score > best_score:
                best_score = score
                best_type = rct
                best_evidence = evidence

        # Build the root cause code
        if best_score < self.min_confidence:
            best_type = RootCauseType.UNKNOWN
            best_score = 0.0

        # Scale confidence by η residual severity (higher η → higher confidence)
        severity_factor = min(1.0, snapshot.eta_residual / (self.eta_threshold * 3))
        confidence = min(1.0, best_score * (0.7 + 0.3 * severity_factor))

        return RootCauseCode(
            cause=best_type.value,
            action=self.DEFAULT_ACTIONS.get(best_type, "Investigate"),
            confidence=confidence,
            timestamp=snapshot.timestamp,
            eta_residual=snapshot.eta_residual,
            evidence=best_evidence,
            root_cause_type=best_type,
        )

    def generate_feedback(self, code: RootCauseCode) -> Dict[str, Any]:
        """Generate process feedback for the TOMAS adaptive library.

        The feedback is used to:
          1. Update the TOMAS adaptive library with the new root cause
          2. Distill the root cause into a new EML node
          3. Update the T-Processor with the new EML calibration

        Args:
            code: The root cause code to generate feedback for.

        Returns:
            Dictionary with feedback for the adaptive library:
              - eml_node_hint: Suggested EML node adjustment
              - process_param_delta: Suggested parameter changes
              - confidence: Confidence of the feedback
        """
        param_deltas: Dict[str, float] = {}

        if code.root_cause_type == RootCauseType.GAS_CONTAMINATION:
            param_deltas["gas_flow"] = +0.20  # +20%
        elif code.root_cause_type == RootCauseType.WIRE_STICK:
            param_deltas["wire_feed"] = -0.15  # -15%
        elif code.root_cause_type == RootCauseType.ARC_INSTABILITY:
            param_deltas["voltage"] = +0.08   # +8% (~+2V)
        elif code.root_cause_type == RootCauseType.LOW_PENETRATION:
            param_deltas["current"] = +0.10   # +10%
        elif code.root_cause_type == RootCauseType.EXCESS_SPATTER:
            param_deltas["current"] = -0.05
            param_deltas["gas_flow"] = +0.10
        elif code.root_cause_type == RootCauseType.VOLTAGE_DROP:
            param_deltas["voltage"] = +0.05
        elif code.root_cause_type == RootCauseType.PLATE_CONTAMINATION:
            param_deltas["preheat"] = +0.15
        else:
            param_deltas = {}

        return {
            "root_cause": code.cause,
            "action": code.action,
            "confidence": code.confidence,
            "process_param_delta": param_deltas,
            "eml_node_hint": {
                "adjust_type": code.root_cause_type.value,
                "eta_at_excursion": code.eta_residual,
            },
            "timestamp": code.timestamp,
        }

    # ── Evidence Scoring Functions ──

    def _safe_mean(self, arr: np.ndarray) -> float:
        """Compute mean safely, handling empty arrays.

        Args:
            arr: Input array.

        Returns:
            Mean value, or 0.0 for empty array.
        """
        if arr is None or len(arr) == 0:
            return 0.0
        return float(np.mean(arr))

    def _safe_std(self, arr: np.ndarray) -> float:
        """Compute std safely, handling empty arrays.

        Args:
            arr: Input array.

        Returns:
            Std value, or 0.0 for empty array.
        """
        if arr is None or len(arr) == 0:
            return 0.0
        return float(np.std(arr))

    def _score_gas_contamination(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Gas_Contamination evidence.

        Indicators:
          - Gas flow below nominal (>15% deficit)
          - High porosity proxy (voltage variance)
          - Temperature below nominal

        Returns:
            (score [0,1], evidence dict).
        """
        gas_mean = self._safe_mean(snap.gas_flow_lpm)
        gas_deficit = max(0.0, (snap.nominal_gas_flow - gas_mean) / snap.nominal_gas_flow)
        v_std = self._safe_std(snap.voltage_v)
        voltage_instability = min(1.0, v_std / 2.0)
        temp_mean = self._safe_mean(snap.temperature_c)
        temp_low = max(0.0, (200.0 - temp_mean) / 200.0)

        score = 0.4 * gas_deficit + 0.3 * voltage_instability + 0.3 * temp_low
        score = min(1.0, score)

        return score, {
            "gas_flow_deficit": gas_deficit,
            "voltage_instability": voltage_instability,
            "temp_low_factor": temp_low,
        }

    def _score_wire_stick(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Wire_Stick evidence.

        Indicators:
          - Current spike (sudden high current = short circuit)
          - Voltage dip (low voltage = stubbing)
          - Wire feed erratic (high variance)

        Returns:
            (score [0,1], evidence dict).
        """
        i_mean = self._safe_mean(snap.current_a)
        current_spike = max(0.0, (i_mean - snap.nominal_current) / snap.nominal_current)
        v_mean = self._safe_mean(snap.voltage_v)
        voltage_dip = max(0.0, (snap.nominal_voltage - v_mean) / snap.nominal_voltage)
        wf_std = self._safe_std(snap.wire_feed_mpm)
        wf_erratic = min(1.0, wf_std / 2.0)

        score = 0.4 * current_spike + 0.4 * voltage_dip + 0.2 * wf_erratic
        score = min(1.0, score)

        return score, {
            "current_spike_ratio": current_spike,
            "voltage_dip_ratio": voltage_dip,
            "wire_feed_erratic": wf_erratic,
        }

    def _score_arc_instability(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Arc_Instability evidence.

        Indicators:
          - High arc length variance
          - High voltage variance
          - Arc length far from nominal

        Returns:
            (score [0,1], evidence dict).
        """
        arc_std = self._safe_std(snap.arc_length_mm)
        arc_var = min(1.0, arc_std / 1.5)
        v_std = self._safe_std(snap.voltage_v)
        voltage_var = min(1.0, v_std / 2.0)
        arc_mean = self._safe_mean(snap.arc_length_mm)
        arc_offset = min(1.0, abs(arc_mean - 5.0) / 5.0)

        score = 0.4 * arc_var + 0.4 * voltage_var + 0.2 * arc_offset
        score = min(1.0, score)

        return score, {
            "arc_length_variance": arc_var,
            "voltage_variance": voltage_var,
            "arc_offset": arc_offset,
        }

    def _score_low_penetration(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Low_Penetration evidence.

        Indicators:
          - Current below nominal
          - Low heat input proxy (I×V low)
          - Low temperature

        Returns:
            (score [0,1], evidence dict).
        """
        i_mean = self._safe_mean(snap.current_a)
        current_low = max(0.0, (snap.nominal_current - i_mean) / snap.nominal_current)
        v_mean = self._safe_mean(snap.voltage_v)
        heat_input_proxy = (i_mean * v_mean) / (snap.nominal_current * snap.nominal_voltage)
        heat_low = max(0.0, 1.0 - heat_input_proxy)
        temp_mean = self._safe_mean(snap.temperature_c)
        temp_low = max(0.0, (200.0 - temp_mean) / 200.0)

        score = 0.4 * current_low + 0.4 * heat_low + 0.2 * temp_low
        score = min(1.0, score)

        return score, {
            "current_low_ratio": current_low,
            "heat_input_low": heat_low,
            "temp_low_factor": temp_low,
        }

    def _score_excess_spatter(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Excess_Spatter evidence.

        Indicators:
          - Current above nominal (too high)
          - Voltage above nominal (too high)
          - High arc length

        Returns:
            (score [0,1], evidence dict).
        """
        i_mean = self._safe_mean(snap.current_a)
        current_high = max(0.0, (i_mean - snap.nominal_current) / snap.nominal_current)
        v_mean = self._safe_mean(snap.voltage_v)
        voltage_high = max(0.0, (v_mean - snap.nominal_voltage) / snap.nominal_voltage)
        arc_mean = self._safe_mean(snap.arc_length_mm)
        arc_high = max(0.0, (arc_mean - 5.0) / 5.0)

        score = 0.4 * current_high + 0.3 * voltage_high + 0.3 * arc_high
        score = min(1.0, score)

        return score, {
            "current_high_ratio": current_high,
            "voltage_high_ratio": voltage_high,
            "arc_high_factor": arc_high,
        }

    def _score_voltage_drop(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Voltage_Drop evidence.

        Indicators:
          - Voltage consistently below nominal
          - Current relatively stable (rules out wire stick)

        Returns:
            (score [0,1], evidence dict).
        """
        v_mean = self._safe_mean(snap.voltage_v)
        voltage_drop = max(0.0, (snap.nominal_voltage - v_mean) / snap.nominal_voltage)
        i_std = self._safe_std(snap.current_a)
        current_stable = 1.0 - min(1.0, i_std / 20.0)

        # Penalize when current is significantly above nominal (indicates wire stick, not voltage drop)
        i_mean = self._safe_mean(snap.current_a)
        current_spike_penalty = max(0.0, (i_mean - snap.nominal_current) / snap.nominal_current)

        score = 0.6 * voltage_drop + 0.4 * current_stable
        score = score * (1.0 - min(0.8, current_spike_penalty))  # Up to 80% penalty
        score = min(1.0, max(0.0, score))

        return score, {
            "voltage_drop_ratio": voltage_drop,
            "current_stability": current_stable,
            "current_spike_penalty": current_spike_penalty,
        }

    def _score_contact_tube_wear(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Contact_Tube_Wear evidence.

        Indicators:
          - Erratic wire feed (high variance)
          - Intermittent current fluctuations
          - Voltage jitter

        Returns:
            (score [0,1], evidence dict).
        """
        wf_std = self._safe_std(snap.wire_feed_mpm)
        wf_erratic = min(1.0, wf_std / 1.5)
        i_std = self._safe_std(snap.current_a)
        i_jitter = min(1.0, i_std / 30.0)
        v_std = self._safe_std(snap.voltage_v)
        v_jitter = min(1.0, v_std / 1.0)

        score = 0.4 * wf_erratic + 0.3 * i_jitter + 0.3 * v_jitter
        score = min(1.0, score)

        return score, {
            "wire_feed_erratic": wf_erratic,
            "current_jitter": i_jitter,
            "voltage_jitter": v_jitter,
        }

    def _score_plate_contamination(
        self, snap: MultimodalSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """Score Plate_Contamination evidence.

        Indicators:
          - Arc instability (contaminated surface causes arc wander)
          - Low temperature (energy lost to vaporizing contaminants)
          - Gas flow normal (rules out gas contamination)

        Returns:
            (score [0,1], evidence dict).
        """
        arc_std = self._safe_std(snap.arc_length_mm)
        arc_wander = min(1.0, arc_std / 2.0)
        temp_mean = self._safe_mean(snap.temperature_c)
        temp_low = max(0.0, (200.0 - temp_mean) / 200.0)
        gas_mean = self._safe_mean(snap.gas_flow_lpm)
        gas_ok = 1.0 if gas_mean >= snap.nominal_gas_flow * 0.9 else 0.0

        score = 0.4 * arc_wander + 0.3 * temp_low + 0.3 * gas_ok
        score = min(1.0, score)

        return score, {
            "arc_wander": arc_wander,
            "temp_low_factor": temp_low,
            "gas_flow_ok": gas_ok,
        }


def generate_root_cause(
    eta_residual: float,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    nominal_current: float = 200.0,
    nominal_voltage: float = 24.0,
    **kwargs: Any,
) -> RootCauseCode:
    """Convenience function: generate a root cause code from sensor data.

    Args:
        eta_residual: η residual that triggered the analysis.
        current_a: Current signal array.
        voltage_v: Voltage signal array.
        nominal_current: Nominal current (A).
        nominal_voltage: Nominal voltage (V).
        **kwargs: Additional sensor arrays (gas_flow_lpm, arc_length_mm, etc.).

    Returns:
        RootCauseCode with the identified cause.
    """
    generator = KSnapRootCauseGenerator()
    snapshot = MultimodalSnapshot(
        timestamp=time.time(),
        current_a=np.asarray(current_a, dtype=np.float64),
        voltage_v=np.asarray(voltage_v, dtype=np.float64),
        eta_residual=eta_residual,
        nominal_current=nominal_current,
        nominal_voltage=nominal_voltage,
        gas_flow_lpm=np.asarray(kwargs.get("gas_flow_lpm", []), dtype=np.float64),
        arc_length_mm=np.asarray(kwargs.get("arc_length_mm", []), dtype=np.float64),
        temperature_c=np.asarray(kwargs.get("temperature_c", []), dtype=np.float64),
        wire_feed_mpm=np.asarray(kwargs.get("wire_feed_mpm", []), dtype=np.float64),
        nominal_gas_flow=kwargs.get("nominal_gas_flow", 15.0),
    )
    return generator.analyze(snapshot)


def _make_synthetic_snapshot(
    cause: RootCauseType,
    n_samples: int = 100,
) -> MultimodalSnapshot:
    """Create a synthetic multimodal snapshot for a given root cause.

    Used by self-test to generate realistic test data.

    Args:
        cause: The root cause to simulate.
        n_samples: Number of samples in the ±100ms window.

    Returns:
        Synthetic MultimodalSnapshot.
    """
    np.random.seed(hash(cause.value) % 2**32)

    t = np.linspace(-0.1, 0.1, n_samples)
    nominal_i, nominal_v, nominal_gas = 200.0, 24.0, 15.0

    current = np.ones(n_samples) * nominal_i
    voltage = np.ones(n_samples) * nominal_v
    arc = np.ones(n_samples) * 5.0
    gas = np.ones(n_samples) * nominal_gas
    temp = np.ones(n_samples) * 250.0
    wf = np.ones(n_samples) * 8.0

    if cause == RootCauseType.GAS_CONTAMINATION:
        gas *= 0.7  # 30% deficit
        voltage += np.random.randn(n_samples) * 3.0
        temp *= 0.8
    elif cause == RootCauseType.WIRE_STICK:
        current *= 1.3  # 30% spike
        voltage *= 0.6  # dip
        wf += np.random.randn(n_samples) * 2.0
    elif cause == RootCauseType.ARC_INSTABILITY:
        arc += np.random.randn(n_samples) * 2.0
        voltage += np.random.randn(n_samples) * 3.0
    elif cause == RootCauseType.LOW_PENETRATION:
        current *= 0.7
        temp *= 0.7
    elif cause == RootCauseType.EXCESS_SPATTER:
        current *= 1.2
        voltage *= 1.15
        arc *= 1.4
    elif cause == RootCauseType.VOLTAGE_DROP:
        voltage *= 0.7
    elif cause == RootCauseType.CONTACT_TUBE_WEAR:
        wf += np.random.randn(n_samples) * 1.5
        current += np.random.randn(n_samples) * 40.0
        voltage += np.random.randn(n_samples) * 1.5
    elif cause == RootCauseType.PLATE_CONTAMINATION:
        arc += np.random.randn(n_samples) * 2.5
        temp *= 0.75

    return MultimodalSnapshot(
        timestamp=time.time(),
        current_a=current,
        voltage_v=voltage,
        arc_length_mm=arc,
        gas_flow_lpm=gas,
        temperature_c=temp,
        wire_feed_mpm=wf,
        eta_residual=0.8,
        nominal_current=nominal_i,
        nominal_voltage=nominal_v,
        nominal_gas_flow=nominal_gas,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Usage:
        python core/ksnap_root_cause.py [--demo] [--self-test]

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="κ-Snap root cause code generator"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic data for all root cause types",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run self-test and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if _self_test() else 1

    print("=" * 60)
    print("κ-Snap Root Cause Code Generator")
    print("=" * 60)

    generator = KSnapRootCauseGenerator()

    if args.demo or not args.self_test:
        # Demo: generate root cause codes for each type
        test_types = [
            RootCauseType.GAS_CONTAMINATION,
            RootCauseType.WIRE_STICK,
            RootCauseType.ARC_INSTABILITY,
            RootCauseType.LOW_PENETRATION,
            RootCauseType.EXCESS_SPATTER,
        ]

        print(f"\n{'Cause':<25} {'Action':<35} {'Conf':>6}")
        print("-" * 70)

        for rct in test_types:
            snap = _make_synthetic_snapshot(rct)
            code = generator.analyze(snap)
            feedback = generator.generate_feedback(code)
            print(f"{code.cause:<25} {code.action:<35} {code.confidence:>5.2f}")
            print(f"  -> Evidence: {code.evidence}")
            print(f"  -> Feedback: {feedback['process_param_delta']}")
            print(f"  -> {code.format_string()}")
            print()

    print("=" * 60)
    return 0


def _self_test() -> bool:
    """Self-test for KSnapRootCauseGenerator.

    Tests:
      1. Should-trigger threshold check
      2. Root cause identification for each type
      3. Confidence in valid range
      4. Feedback generation
      5. Format string output

    Returns:
        True if all tests pass.
    """
    print("[ksnap_root_cause] Running self-test...")

    generator = KSnapRootCauseGenerator(eta_threshold=0.5, min_confidence=0.3)

    # Test 1: Threshold check
    assert not generator.should_trigger(0.3), "η=0.3 should not trigger"
    assert generator.should_trigger(0.6), "η=0.6 should trigger"
    assert generator.should_trigger(1.0), "η=1.0 should trigger"

    # Test 2: Root cause identification for each type
    test_cases: List[Tuple[RootCauseType, RootCauseType]] = []
    all_causes = [
        RootCauseType.GAS_CONTAMINATION,
        RootCauseType.WIRE_STICK,
        RootCauseType.ARC_INSTABILITY,
        RootCauseType.LOW_PENETRATION,
        RootCauseType.EXCESS_SPATTER,
        RootCauseType.VOLTAGE_DROP,
        RootCauseType.CONTACT_TUBE_WEAR,
        RootCauseType.PLATE_CONTAMINATION,
    ]

    correct = 0
    for rct in all_causes:
        snap = _make_synthetic_snapshot(rct)
        code = generator.analyze(snap)
        test_cases.append((rct, code.root_cause_type))

        # Check confidence is in valid range
        assert 0.0 <= code.confidence <= 1.0, \
            f"Confidence {code.confidence} out of range for {rct.value}"

        # Check format string
        fmt = code.format_string()
        assert "RootCause:" in fmt, "Format string missing 'RootCause:'"
        assert "Action:" in fmt, "Format string missing 'Action:'"
        assert "Confidence:" in fmt, "Format string missing 'Confidence:'"

        if code.root_cause_type == rct:
            correct += 1

    print(f"  Root cause identification: {correct}/{len(all_causes)} correct")

    # At least 5/8 should be correctly identified (allowing for overlap)
    assert correct >= 5, \
        f"Expected ≥5/8 correct identifications, got {correct}/{len(all_causes)}"

    # Test 3: Gas contamination specific test (most important from SLOS paper)
    snap_gas = _make_synthetic_snapshot(RootCauseType.GAS_CONTAMINATION)
    code_gas = generator.analyze(snap_gas)
    assert code_gas.root_cause_type == RootCauseType.GAS_CONTAMINATION, \
        f"Gas contamination should be identified, got {code_gas.root_cause_type.value}"
    assert code_gas.confidence > 0.3, \
        f"Gas contamination confidence should be >0.3, got {code_gas.confidence}"
    assert "Increase" in code_gas.action, \
        f"Gas contamination action should include 'Increase', got '{code_gas.action}'"
    print(f"  Gas_Contamination: conf={code_gas.confidence:.2f} [OK]")

    # Test 4: Feedback generation
    feedback = generator.generate_feedback(code_gas)
    assert "process_param_delta" in feedback, "Feedback missing process_param_delta"
    assert "gas_flow" in feedback["process_param_delta"], \
        "Gas contamination feedback should include gas_flow delta"
    assert feedback["process_param_delta"]["gas_flow"] > 0, \
        "Gas flow delta should be positive (increase)"
    assert "eml_node_hint" in feedback, "Feedback missing eml_node_hint"

    # Test 5: Wire stick specific test
    snap_stick = _make_synthetic_snapshot(RootCauseType.WIRE_STICK)
    code_stick = generator.analyze(snap_stick)
    assert code_stick.root_cause_type == RootCauseType.WIRE_STICK, \
        f"Wire stick should be identified, got {code_stick.root_cause_type.value}"
    assert "Reduce" in code_stick.action or "Wire" in code_stick.action, \
        f"Wire stick action should include 'Reduce' or 'Wire', got '{code_stick.action}'"
    print(f"  Wire_Stick: conf={code_stick.confidence:.2f} [OK]")

    # Test 6: Empty data handling
    snap_empty = MultimodalSnapshot(eta_residual=0.6)
    code_empty = generator.analyze(snap_empty)
    assert code_empty.confidence >= 0.0, "Empty data should not crash"

    print(f"  All {len(all_causes)} root cause types tested [OK]")
    print("[ksnap_root_cause] Self-test PASSED.")
    return True


if __name__ == "__main__":
    sys.exit(main())
