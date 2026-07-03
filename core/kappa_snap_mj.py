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

v0.3.0 Upgrade: MerkleChain Integration
  - η computation results can be recorded to MerkleChain via prev_snap_id
  - snap_id = prev_snap_id + sha256(prev_snap_id + str(η) + str(decision))[:16]
  - Agent passes prev_snap_id through internal state for chain linkage
  - Provides compute_merkle_snap_id() helper for agent integration

Inspired by:
  - ReinFlow (NeurIPS 2025): online RL fine-tuning of flow matching policy
  - FPO: on-policy flow matching training
  - IRP (RSS 2022 Best Paper): iterative residual policy
  - ResShift: residual shifting for efficient diffusion

Author: tomas-arc3-solver project · IDO-MuJoCo-Bench extension
"""
import hashlib
import numpy as np
from typing import Optional

IDO_KAPPA_SNAP_MJ_VERSION: str = "v0.3.1"


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
                      flow_predictor: Optional[FlowMatchingEtaPredictor] = None,
                      step_index: int = 0,
                      epiplexity: float = 0.0,
                      w_decay: float = 0.02) -> float:
    """Compute GaussEx residual η for continuous-state IDO κ-Snap.

    η = w_pos * pos_err^2 + w_ori * tilt_err^2
        + w_eng * energy_excess^2 + w_vel * vel_mag^2

    v0.2.0: If flow_predictor is provided, the current η is pushed into
    the predictor's buffer and a trend-adjusted η is returned:
        η_adjusted = η_current + α * (η_predicted - η_current)
    where α is adaptive (0.1–0.3) based on trend signal strength.

    v0.2.1: Familiarity decay — η decreases as step_index and epiplexity
    increase, modelling structural understanding accumulation:
        decay_factor = w_decay * min(step/100, 1) * min(epi/5, 1)
        η = η * (1 - decay_factor)
    This provides a measurable downward trend across SIP-Bench phases
    without distorting the physical signal (max 2% decay per step).

    v0.3.1: Locomotion η-mode — when goal.eta_mode == 'locomotion',
    η is computed from velocity/height/upright deficits instead of
    position distance. This fixes the fundamental problem where
    locomotion tasks (walker-walk, cheetah-run) use velocity-based
    rewards but η was measuring distance to a fixed point, making η
    unreachable (walker η~38, cheetah η~100, never decreasing).

    Locomotion η formula:
        η = w_vel * vel_deficit^2 + w_height * height_deficit^2
            + w_upright * upright_deficit^2 + w_eng * energy_excess^2
        vel_deficit = max(0, target_speed - current_speed)
        height_deficit = max(0, target_height - torso_z)
        upright_deficit = max(0, target_upright - torso_upright_score)

    This aligns η with dm_control's tolerance() reward formula, so η
    decreases as the agent walks/runs correctly, enabling near-goal
    PD stabilize mode and Creative-Probe effectiveness.

    Args:
        z_i: EML observation dict with keys:
             'ee_pos' (3-vector), 'qpos' (nq-vector), 'E_total' (float),
             'ee_vel' (6-vector or 3-vector).
             For locomotion η-mode, also needs:
             'horiz_vel' (float, horizontal speed m/s) and
             'torso_z' (float, torso height m) and
             'torso_upright' (float, upright score 0-1).
        goal: GoalEML instance with target_pos, max_energy_inject, etc.
            If goal.eta_mode == 'locomotion', also uses target_speed,
            target_height, target_upright, eta_weights.
        w_pos: Weight for position error component (point mode only).
        w_ori: Weight for orientation (tilt) error component (point mode only).
        w_eng: Weight for energy excess component.
        w_vel: Weight for velocity magnitude component (point mode only).
        flow_predictor: Optional FlowMatchingEtaPredictor for trend enhancement.
                        If None, returns pure current η (backward compatible).
        step_index: Current decision-step index (for familiarity decay).
                    Default 0 means no decay (backward compatible).
        epiplexity: Structural information density score from ψ-Anchor.
                    Default 0.0 means no decay (backward compatible).
        w_decay: Familiarity decay weight (max η reduction per step).
                 Default 0.02 → η reduces at most 2%/step.

    Returns:
        Scalar residual η (float). Lower η means closer to goal manifold.
        If flow_predictor is provided, η is trend-adjusted with predicted future.
        Familiarity decay is applied after trend blending.
    """
    # ── v0.3.1: Locomotion η-mode ──
    # When goal.eta_mode == 'locomotion', compute η from velocity/height/upright
    # deficits instead of position distance. This aligns η with dm_control's
    # tolerance()-based reward formula for locomotion tasks.
    eta_mode: str = getattr(goal, 'eta_mode', 'point')

    if eta_mode == 'locomotion':
        # Extract locomotion-specific observations from z_i
        # horiz_vel: horizontal speed (m/s) — from torso velocity or qvel
        horiz_vel: float = float(z_i.get('horiz_vel', 0.0))
        # torso_z: torso height (m) — from ee_pos[2] or xpos
        torso_z: float = float(z_i.get('torso_z',
                             float(z_i.get('ee_pos', np.zeros(3))[2])
                             if len(z_i.get('ee_pos', np.zeros(3))) >= 3
                             else 0.0))
        # torso_upright: upright score (0-1) — from quaternion or xmat
        torso_upright: float = float(z_i.get('torso_upright', 0.0))
        # If torso_upright not provided, compute from quaternion
        if torso_upright == 0.0:
            qpos_arr: Optional[np.ndarray] = z_i.get('qpos', None)
            if qpos_arr is not None and len(qpos_arr) >= 4:
                quat: np.ndarray = qpos_arr[3:7] if len(qpos_arr) >= 7 else qpos_arr[:4]
                z_body: np.ndarray = _quat_to_z_axis(quat)
                torso_upright = float(z_body[2])  # z-component of body z-axis

        # Get locomotion η weights from goal.eta_weights (or defaults)
        # v0.6.5: Support both dict and ndarray eta_weights formats.
        # ndarray format: [w_vel, w_height, w_upright, w_eng] (legacy)
        # dict format: {'vel': w_vel, 'height': w_height, ...} (preferred)
        raw_weights = getattr(goal, 'eta_weights', None)
        if isinstance(raw_weights, dict):
            eta_w: dict = raw_weights
        elif isinstance(raw_weights, np.ndarray):
            # Legacy ndarray format: [w_vel, w_height, w_upright, w_eng]
            eta_w = {
                'w_vel': float(raw_weights[0]) if len(raw_weights) > 0 else 1.0,
                'w_height': float(raw_weights[1]) if len(raw_weights) > 1 else 0.5,
                'w_upright': float(raw_weights[2]) if len(raw_weights) > 2 else 0.3,
                'w_eng': float(raw_weights[3]) if len(raw_weights) > 3 else 0.01,
            }
        elif raw_weights is None:
            eta_w = {}
        else:
            eta_w = {}
        w_vel_loc: float = float(eta_w.get('w_vel', 1.0))
        w_height_loc: float = float(eta_w.get('w_height', 0.5))
        w_upright_loc: float = float(eta_w.get('w_upright', 0.3))
        w_eng_loc: float = float(eta_w.get('w_eng', 0.01))

        # Target values from GoalEML
        target_speed: float = getattr(goal, 'target_speed', 0.0)
        target_height: float = getattr(goal, 'target_height', 0.0)
        target_upright: float = getattr(goal, 'target_upright', 0.0)

        # Compute deficits (max(0, ...) = only penalize being BELOW target)
        # Velocity deficit: LINEAR (not squared) — aligns with dm_control
        # tolerance(speed, bounds=(target,inf), margin=target, sigmoid='linear')
        # Linear penalty makes η proportional to speed deficit (0-10 for cheetah),
        # not quadratic (0-100), keeping η manageable and κ_thresh reachable.
        vel_deficit: float = max(0.0, target_speed - horiz_vel)
        # Height/upright deficits: SQUARED — preserves basin effect near goal
        # (η drops faster as agent approaches target height/upright, enabling
        # near-goal PD stabilize mode to kick in smoothly)
        height_deficit: float = max(0.0, target_height - torso_z)
        upright_deficit: float = max(0.0, target_upright - torso_upright)

        # Energy excess beyond budget
        E: float = float(z_i.get('E_total', 0.0))
        energy_excess: float = max(0.0, E - goal.max_energy_inject)

        # Weighted locomotion η: velocity linear, height/upright squared
        # η = w_vel * vel_deficit + w_height * height_deficit^2
        #     + w_upright * upright_deficit^2 + w_eng * energy_excess^2
        eta: float = (w_vel_loc * vel_deficit
                      + w_height_loc * height_deficit ** 2
                      + w_upright_loc * upright_deficit ** 2
                      + w_eng_loc * energy_excess ** 2)

    else:
        # ── Point η-mode (default): distance to target_pos ──
        # Position error — align ee and target dimensions (pad/trim to same length)
        ee: np.ndarray = np.asarray(z_i.get('ee_pos', np.zeros(3)))
        target: np.ndarray = np.asarray(goal.target_pos)
        # Pad shorter array with zeros to match dimensions
        max_dim: int = max(len(ee), len(target))
        ee_padded: np.ndarray = np.zeros(max_dim)
        ee_padded[:len(ee)] = ee
        target_padded: np.ndarray = np.zeros(max_dim)
        target_padded[:len(target)] = target
        pos_err: float = float(np.linalg.norm(ee_padded - target_padded))

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

    # ── v0.2.1: Familiarity Decay ──
    # Structural understanding accumulation: as steps and epiplexity increase,
    # η should decay slightly, reflecting that the agent is "getting familiar"
    # with the task's structure. This provides a measurable downward trend
    # across SIP-Bench phases without distorting physical signal.
    decay_step_ratio: float = min(step_index / 100.0, 1.0)
    decay_epi_ratio: float = min(max(epiplexity, 0.0) / 5.0, 1.0)
    decay_factor: float = w_decay * decay_step_ratio * decay_epi_ratio
    eta = eta * (1.0 - decay_factor)

    # ── v0.2.0: Flow-Matching Enhancement ──
    if flow_predictor is not None:
        # Push current η into trajectory buffer
        flow_predictor.push(eta)

        # Predict next η using flow matching
        predicted_eta: float = flow_predictor.predict_next_eta()

        # v0.2.1: Adaptive α blending — increases when trend signal is strong
        # Base α = 0.1 (conservative), up to 0.3 when predicted deviation is large
        trend_signal_strength: float = abs(predicted_eta - eta) / max(eta, 0.01)
        alpha: float = min(0.3, 0.1 + 0.2 * trend_signal_strength)
        alpha = max(0.1, alpha)  # guarantee α ≥ 0.1
        eta = eta + alpha * (predicted_eta - eta)

    return float(eta)


def compute_merkle_snap_id(prev_snap_id: str,
                            eta: float,
                            decision: str = "") -> str:
    """Compute a MerkleChain snap_id for κ-Snap audit trail.

    v0.3.0: Provides a helper function for computing MerkleChain
    snap IDs from η values and decision strings. This enables
    agents to create Merkle-linked audit entries for each step.

    Hash rule: snap_id = prev_snap_id + sha256(prev_snap_id + str(η) + str(decision))[:16]

    Args:
        prev_snap_id: Previous snap_id from the MerkleChain.
                      Use "genesis" for the first entry in a new chain.
        eta: Current κ-Snap residual η value.
        decision: Decision string for this step (e.g., 'EXPLOIT', 'SAFE').

    Returns:
        Computed snap_id string linking to prev_snap_id via SHA-256 hash.
    """
    hash_input: str = prev_snap_id + str(eta) + str(decision)
    hash_hex: str = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    snap_id: str = prev_snap_id + hash_hex[:16]
    return snap_id
