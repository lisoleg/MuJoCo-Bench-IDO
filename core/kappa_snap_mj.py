"""
κ-Snap (GaussEx Residual) — Continuous-state IDO version
=========================================================

Computes the GaussEx residual η = Σ w_i · f_i(z_i, Goal-EML)^2 that
measures how far the current observation z_i deviates from the IDO
goal manifold defined by GoalEML invariants.

Components:
  - Position error:   ||ee_pos − target_pos||^2
  - Orientation error: tilt angle from upright z-axis ^2
  - Energy excess:    max(0, E_total − max_energy_inject)^2
  - Velocity error:   ||ee_vel[:3]||^2

v0.2.0 Upgrade: FlowMatchingEtaPredictor
  - Maintains a η trajectory buffer for trend analysis
  - Uses flow matching (linear extrapolation + residual correction) to predict future η
  - Detects hesitation (η plateau) and retry (η fallback)
  - Provides Hesitation-RMSE and Retry-VOC metrics

Inspired by:
  - ReinFlow (NeurIPS 2025): online RL fine-tuning of flow matching policy
  - FPO: on-policy flow matching training
  - IRP (RSS 2022 Best Paper): iterative residual policy
  - ResShift: residual shifting for efficient diffusion

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import numpy as np
from typing import Optional

IDO_KAPPA_SNAP_MJ_VERSION: str = "v0.2.0"


class FlowMatchingEtaPredictor:
    """Flow-matching enhanced η predictor for κ-Snap.

    Instead of just computing current η, this predictor:
    1. Maintains a η trajectory buffer
    2. Uses flow matching to predict future η trend
    3. Detects hesitation (η plateau) and retry (η fallback)
    4. Provides Hesitation-RMSE and Retry-VOC metrics

    The flow matching model is a lightweight linear extrapolation with
    residual correction, inspired by the iterative residual policy (IRP)
    approach from robotics and ResShift from image generation.

    Attributes:
        eta_buffer: Rolling buffer of recent η values.
        window_size: Size of the η trajectory window for prediction.
    """

    def __init__(self, window_size: int = 10) -> None:
        """Initialize the FlowMatchingEtaPredictor.

        Args:
            window_size: Number of η values kept in the trajectory buffer.
                         Default 10 provides enough history for trend analysis.
        """
        self.eta_buffer: list = []
        self.window_size: int = window_size

    def push(self, eta: float) -> None:
        """Push new η value into buffer.

        Maintains a bounded rolling buffer. When the buffer exceeds
        window_size, the oldest value is discarded.

        Args:
            eta: Current κ-Snap residual value from this decision step.
        """
        self.eta_buffer.append(eta)
        if len(self.eta_buffer) > self.window_size:
            self.eta_buffer = self.eta_buffer[-self.window_size:]

    def predict_next_eta(self) -> float:
        """Predict next η using simple flow matching.

        Flow = η(t+1) ≈ η(t) + Δη(t) + residual_correction

        The prediction uses:
        1. Last η value as base (η(t))
        2. Last delta η as velocity (Δη(t) = η(t) - η(t-1))
        3. Mean of delta-η errors as residual correction:
           For each pair in the window, compute Δη_error = actual_Δη - predicted_Δη
           Then residual_correction = mean(Δη_errors)

        This is a minimal "flow matching" that captures the trajectory
        dynamics without requiring external ML libraries.

        Returns:
            Predicted next η value. Returns current η if buffer has < 2 values.
        """
        if len(self.eta_buffer) < 2:
            # Insufficient data: return last η as prediction
            return self.eta_buffer[-1] if len(self.eta_buffer) > 0 else 0.0

        # Current η and most recent delta
        eta_t: float = self.eta_buffer[-1]
        delta_eta_t: float = self.eta_buffer[-1] - self.eta_buffer[-2]

        # Compute residual correction: mean of delta-η prediction errors
        # Over the window, compare predicted Δη (= previous Δη) vs actual Δη
        deltas: list = []
        for i in range(1, len(self.eta_buffer)):
            deltas.append(self.eta_buffer[i] - self.eta_buffer[i - 1])

        # Prediction errors: for each step, predicted delta = previous delta,
        # actual delta = current delta
        residual_errors: list = []
        for i in range(1, len(deltas)):
            predicted_delta: float = deltas[i - 1]
            actual_delta: float = deltas[i]
            residual_errors.append(actual_delta - predicted_delta)

        # Residual correction = mean of all prediction errors
        if len(residual_errors) > 0:
            residual_correction: float = float(np.mean(residual_errors))
        else:
            residual_correction = 0.0

        # Flow matching prediction: η(t+1) = η(t) + Δη(t) + correction
        predicted_eta: float = eta_t + delta_eta_t + residual_correction

        # Ensure predicted η is non-negative (η is a squared residual)
        predicted_eta = max(0.0, predicted_eta)

        return float(predicted_eta)

    def compute_hesitation_rmse(self) -> float:
        """Compute Hesitation-RMSE: measures η oscillation around a plateau.

        Hesitation-RMSE quantifies how much η oscillates around a local mean
        over the trajectory window. High Hesitation-RMSE indicates the agent
        is "hesitating" — making small oscillations without meaningful progress.

        Formula: RMSE = sqrt(mean((η_i - η_mean)^2)) over window

        Returns:
            Hesitation-RMSE value. Returns 0.0 if buffer is empty.
        """
        if len(self.eta_buffer) == 0:
            return 0.0

        eta_mean: float = float(np.mean(self.eta_buffer))
        deviations: np.ndarray = np.array(self.eta_buffer) - eta_mean
        rmse: float = float(np.sqrt(np.mean(deviations ** 2)))

        return rmse

    def compute_retry_voc(self) -> float:
        """Compute Retry-VOC: variance of change direction.

        Retry-VOC measures how often η changes direction (up vs down),
        quantifying the "retry" behavior where the agent alternates
        between improving and worsening states.

        Formula: VOC = variance(sign(Δη)) over window

        sign(Δη) ∈ {-1, 0, +1}:
        - High VOC means frequent direction changes (retry behavior)
        - Low VOC means consistent monotonic progress (no retries)

        Returns:
            Retry-VOC value. Returns 0.0 if buffer has < 2 values.
        """
        if len(self.eta_buffer) < 2:
            return 0.0

        # Compute sign of each delta η
        signs: list = []
        for i in range(1, len(self.eta_buffer)):
            delta: float = self.eta_buffer[i] - self.eta_buffer[i - 1]
            if delta > 1e-8:
                signs.append(1.0)
            elif delta < -1e-8:
                signs.append(-1.0)
            else:
                signs.append(0.0)

        if len(signs) == 0:
            return 0.0

        # Variance of sign values
        voc: float = float(np.var(signs))
        return voc

    def detect_stagnation(self, threshold: float = 0.01) -> bool:
        """Detect if η is stagnating (not making meaningful progress).

        Stagnation is detected when:
        1. The mean absolute delta η over the window < threshold
           (η changes are too small)
        OR
        2. The Hesitation-RMSE < threshold * 0.5
           (η oscillates within a very narrow band)

        Args:
            threshold: Minimum meaningful change in η. Default 0.01.

        Returns:
            True if η is stagnating, False if making progress.
        """
        if len(self.eta_buffer) < 2:
            return False

        # Mean absolute delta η
        abs_deltas: list = []
        for i in range(1, len(self.eta_buffer)):
            abs_deltas.append(abs(self.eta_buffer[i] - self.eta_buffer[i - 1]))

        mean_abs_delta: float = float(np.mean(abs_deltas))

        # Hesitation-RMSE check
        hesit_rmse: float = self.compute_hesitation_rmse()

        # Stagnation condition
        is_stagnant: bool = (mean_abs_delta < threshold
                             or hesit_rmse < threshold * 0.5)

        return is_stagnant

    def clear(self) -> None:
        """Clear the η trajectory buffer.

        Useful for resetting the predictor between episodes or phases.
        """
        self.eta_buffer = []


def _quat_to_z_axis(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to the body's local z-axis vector.

    Uses the rotation matrix formula for the third column (z-axis).

    Args:
        quat: Quaternion array of length ≥4 in [w, x, y, z] convention.
              If None or shorter than 4, returns default [0, 0, 1].

    Returns:
        3-vector representing the body's z-axis in world coordinates.
    """
    if quat is None or len(quat) < 4:
        return np.array([0.0, 0.0, 1.0])
    qw: float = quat[0]
    qx: float = quat[1]
    qy: float = quat[2]
    qz: float = quat[3]
    zx: float = 2.0 * (qx * qz + qw * qy)
    zy: float = 2.0 * (qy * qz - qw * qx)
    zz: float = qw * qw - qx * qx - qy * qy + qz * qz
    return np.array([zx, zy, zz])


def gauss_ex_residual(z_i: dict,
                      goal,
                      w_pos: float = 1.0,
                      w_ori: float = 0.3,
                      w_eng: float = 0.01,
                      w_vel: float = 0.05,
                      flow_predictor: Optional[FlowMatchingEtaPredictor] = None) -> float:
    """Compute GaussEx residual η for continuous-state IDO κ-Snap.

    η = w_pos * pos_err^2 + w_ori * tilt_err^2
        + w_eng * energy_excess^2 + w_vel * vel_mag^2

    v0.2.0: If flow_predictor is provided, the current η is pushed into
    the predictor's buffer and a trend-adjusted η is returned:
        η_adjusted = η_current + α * (η_predicted - η_current)
    where α = 0.1 (light blending of predicted trend). This allows
    the κ-Snap residual to be forward-looking while still grounded
    in the current observation.

    Args:
        z_i: EML observation dict with keys:
             'ee_pos' (3-vector), 'qpos' (nq-vector), 'E_total' (float),
             'ee_vel' (6-vector or 3-vector).
        goal: GoalEML instance with target_pos, max_energy_inject, etc.
        w_pos: Weight for position error component.
        w_ori: Weight for orientation (tilt) error component.
        w_eng: Weight for energy excess component.
        w_vel: Weight for velocity magnitude component.
        flow_predictor: Optional FlowMatchingEtaPredictor for trend enhancement.
                        If None, returns pure current η (backward compatible).

    Returns:
        Scalar residual η (float). Lower η means closer to goal manifold.
        If flow_predictor is provided, η is trend-adjusted with predicted future.
    """
    # Position error
    ee: np.ndarray = z_i.get('ee_pos', np.zeros(3))
    target: np.ndarray = goal.target_pos
    if ee is None or target is None:
        pos_err: float = 0.0
    else:
        pos_err = float(np.linalg.norm(np.asarray(ee) - np.asarray(target)))

    # Orientation (tilt) error via quaternion → z-axis
    quat: Optional[np.ndarray] = None
    qpos: Optional[np.ndarray] = z_i.get('qpos', None)
    if qpos is not None and len(qpos) >= 4:
        quat = qpos[3:7] if len(qpos) >= 7 else qpos[:4]
    z_body: np.ndarray = _quat_to_z_axis(quat)
    tilt_err: float = float(np.arccos(
        np.clip(np.dot(z_body, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))

    # Energy excess beyond budget
    E: float = float(z_i.get('E_total', 0.0))
    energy_excess: float = max(0.0, E - goal.max_energy_inject)

    # End-effector velocity magnitude
    ee_vel: np.ndarray = z_i.get('ee_vel', np.zeros(6))
    vel_mag: float = float(np.linalg.norm(ee_vel[:3])) if ee_vel is not None else 0.0

    # Weighted sum of squared residuals
    eta: float = (w_pos * pos_err ** 2
                  + w_ori * tilt_err ** 2
                  + w_eng * energy_excess ** 2
                  + w_vel * vel_mag ** 2)

    # ── v0.2.0: Flow-Matching Enhancement ──
    if flow_predictor is not None:
        # Push current η into trajectory buffer
        flow_predictor.push(eta)

        # Predict next η using flow matching
        predicted_eta: float = flow_predictor.predict_next_eta()

        # Blend current η with predicted trend (α = 0.1, light forward-looking)
        alpha: float = 0.1
        eta = eta + alpha * (predicted_eta - eta)

    return float(eta)
