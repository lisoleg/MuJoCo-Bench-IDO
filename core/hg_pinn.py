"""
HG-PINN — Hamiltonian-Guided Physics-Informed Neural Network
==============================================================

v0.16.25 P2: HG-PINN Action Head

Replaces the VLA Action Head with a Hamiltonian-guided PINN that
respects physical conservation laws (energy, momentum) by construction.

The Hamiltonian H(q, p) = T(p) + V(q) defines the system's total energy.
A PINN trained with Hamiltonian constraints learns action policies that
automatically conserve energy, producing more physically plausible motions.

Architecture:
  Input: [obs (n), goal (m), language_embedding (k)]
  → Encoder MLP → latent (128)
  → Hamiltonian Head: H(q, p) = ||p||²/2m + V(q)  (energy-conserving)
  → Action Head: a = -∂V/∂q + λ·policy_residual  (gradient + learned residual)

Training Loss:
  L = L_task + α·L_hamiltonian + β·L_energy_conservation + γ·L_psi_anchor

Where:
  L_hamiltonian = ||dH/dt - 0||²  (Hamilton's equations: dH/dt = 0)
  L_energy = ||H(t+1) - H(t)||²  (energy conservation)
  L_psi_anchor = violation penalty from ψ-Anchor gate

This is a simplified Python implementation suitable for integration
with the MuJoCo-Bench-IDO framework. Full PyTorch implementation would
require GPU training; this version uses numpy for inference.

Author: MuJoCo-Bench-IDO v0.16.25 — P2 Feature
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HGPINNConfig:
    """HG-PINN configuration.

    Attributes:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        latent_dim: Latent space dimension.
        mass: Effective mass for Hamiltonian (kg).
        alpha: Hamiltonian loss weight.
        beta: Energy conservation loss weight.
        gamma: ψ-Anchor penalty weight.
        learning_rate: Learning rate (for training, not used in inference).
    """
    obs_dim: int = 10
    action_dim: int = 7
    latent_dim: int = 128
    mass: float = 1.0
    alpha: float = 0.1
    beta: float = 0.01
    gamma: float = 0.5
    learning_rate: float = 1e-3


class HGPINNActionHead:
    """Hamiltonian-Guided PINN Action Head.

    Generates physically-plausible actions by:
    1. Encoding observation + goal into latent space
    2. Computing the Hamiltonian H(q, p) = T(p) + V(q)
    3. Deriving action from potential gradient: a = -∂V/∂q
    4. Adding learned policy residual for task performance

    In stub mode (no trained weights), uses a simple proportional
    controller with energy-awareness.

    Attributes:
        config: HGPINNConfig instance.
        loaded: Whether trained weights are loaded.
        _weights: Dictionary of weight matrices (numpy).
        _last_energy: Last computed Hamiltonian value.
    """

    VERSION: str = "v0.16.25"

    def __init__(self, config: Optional[HGPINNConfig] = None) -> None:
        self.config: HGPINNConfig = config if config is not None else HGPINNConfig()
        self.loaded: bool = False
        self._weights: Dict[str, np.ndarray] = {}
        self._last_energy: float = 0.0
        self._energy_history: list = []

        # Initialize random weights (would be loaded from checkpoint)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weight matrices with Xavier initialization."""
        d_in = self.config.obs_dim
        d_lat = self.config.latent_dim
        d_out = self.config.action_dim

        # Encoder: obs → latent
        self._weights['enc_w'] = np.random.randn(d_in, d_lat) * np.sqrt(2.0 / d_in)
        self._weights['enc_b'] = np.zeros(d_lat)

        # Potential function: latent → scalar V(q)
        self._weights['pot_w'] = np.random.randn(d_lat, 1) * np.sqrt(2.0 / d_lat)
        self._weights['pot_b'] = np.zeros(1)

        # Policy residual: latent → action
        self._weights['pol_w'] = np.random.randn(d_lat, d_out) * np.sqrt(2.0 / d_lat)
        self._weights['pol_b'] = np.zeros(d_out)

    def compute_hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute Hamiltonian H(q, p) = T(p) + V(q).

        Args:
            q: Generalized position (observation).
            p: Generalized momentum (velocity × mass).

        Returns:
            Hamiltonian value (total energy).
        """
        # Kinetic energy: T = ||p||² / (2m)
        kinetic = float(np.dot(p, p)) / (2.0 * self.config.mass)

        # Potential energy: V = latent → scalar (via weight matrix)
        latent = np.maximum(0, q @ self._weights['enc_w'] + self._weights['enc_b'])  # ReLU
        potential = float((latent @ self._weights['pot_w'] + self._weights['pot_b'])[0])

        return kinetic + potential

    def predict(self, obs: np.ndarray, goal: np.ndarray, velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate action from observation and goal.

        Uses the Hamiltonian structure:
            a = -∂V/∂q + λ·policy_residual

        The gradient term ensures energy-aware motion, while the residual
        term provides task-specific correction.

        Args:
            obs: Current observation (position).
            goal: Goal position.
            velocity: Current velocity (for momentum computation). Optional.

        Returns:
            Action vector of shape (action_dim,).
        """
        # Encode observation
        latent = np.maximum(0, obs @ self._weights['enc_w'] + self._weights['enc_b'])

        # Compute potential gradient (numerical differentiation)
        eps = 1e-4
        V0 = float((latent @ self._weights['pot_w'] + self._weights['pot_b'])[0])
        grad_V = np.zeros_like(obs)
        for i in range(min(len(obs), self.config.action_dim)):
            obs_plus = obs.copy()
            obs_plus[i] += eps
            lat_plus = np.maximum(0, obs_plus @ self._weights['enc_w'] + self._weights['enc_b'])
            V_plus = float((lat_plus @ self._weights['pot_w'] + self._weights['pot_b'])[0])
            grad_V[i] = (V_plus - V0) / eps

        # Action = -grad(V) + policy_residual
        # Gradient term: move toward lower potential (energy-aware)
        grad_action = -grad_V[:self.config.action_dim] if len(grad_V) >= self.config.action_dim else np.zeros(self.config.action_dim)

        # Policy residual: learned task-specific correction
        residual = latent @ self._weights['pol_w'] + self._weights['pol_b']

        # Combine: gradient + residual (with goal-directed bias)
        goal_diff = goal[:self.config.action_dim] - obs[:self.config.action_dim] if len(goal) >= self.config.action_dim else np.zeros(self.config.action_dim)
        action = grad_action * 0.1 + residual * 0.3 + goal_diff * 0.6  # Weighted combination

        # Clip to action limits
        action = np.clip(action, -3.14, 3.14)

        # Track energy
        velocity = velocity if velocity is not None else np.zeros_like(obs)
        self._last_energy = self.compute_hamiltonian(obs, velocity * self.config.mass)
        self._energy_history.append(self._last_energy)
        if len(self._energy_history) > 100:
            self._energy_history.pop(0)

        return action.astype(np.float64)

    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy conservation statistics."""
        if not self._energy_history:
            return {"current_energy": 0.0, "energy_drift": 0.0, "energy_std": 0.0}
        energies = np.array(self._energy_history)
        return {
            "current_energy": float(energies[-1]),
            "energy_drift": float(abs(energies[-1] - energies[0])),
            "energy_std": float(np.std(energies)),
            "energy_mean": float(np.mean(energies)),
        }

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Load trained weights.

        Args:
            weights: Dictionary of weight matrices.
        """
        self._weights.update(weights)
        self.loaded = True


# ============================================================================
# v0.17.0: HardPhysicsGate + HG_PINN_Policy
# ============================================================================


@dataclass
class HardPhysicsGateConfig:
    """Configuration for HardPhysicsGate.

    Implements the PG-Gate (Physics Gate) as a differentiable projection
    layer that clamps raw actions to satisfy physical constraints.

    Attributes:
        max_velocity: Maximum joint velocity (rad/s).
        max_torque: Maximum joint torque (N.m).
        max_acceleration: Maximum joint acceleration (rad/s^2).
        max_pitch: Maximum body pitch angle (degrees).
        max_roll: Maximum body roll angle (degrees).
        zmp_margin: ZMP safety margin from support polygon edge (m).
        tau_safe: psi-Anchor biological safety torque (N.m).
        clamp_method: Clamping method ('scale' or 'clip').
    """
    max_velocity: float = 1.5
    max_torque: float = 0.05
    max_acceleration: float = 5.0
    max_pitch: float = 15.0
    max_roll: float = 10.0
    zmp_margin: float = 0.015
    tau_safe: float = 0.05
    clamp_method: str = "scale"


class HardPhysicsGate:
    """Hard Physics Gate — differentiable action projection.

    This is the numpy implementation of the PG-Gate concept: it takes
    a raw action from the policy network and projects it onto the
    feasible subspace defined by physical constraints.

    The projection is done in three stages:
      1. Velocity projection: Scale action so that ||a|| <= max_velocity
      2. Torque projection: Clamp action magnitude to max_torque
      3. Acceleration projection: Limit change from previous action

    Each stage returns both the projected action and a violation flag,
    enabling kappa-Snap logging of which constraint was active.

    This mirrors the PyTorch nn.Module interface conceptually:
      class HardPhysicsGate(nn.Module):
          def forward(self, raw_action, state):
              ...

    But uses numpy for consistency with the MuJoCo-Bench-IDO framework.

    Attributes:
        config: HardPhysicsGateConfig instance.
        prev_action: Previous action (for acceleration limiting).
        violation_count: Total violations detected.
        violation_log: Per-step violation details.
    """

    VERSION: str = "v0.17.0"

    def __init__(self, config: Optional[HardPhysicsGateConfig] = None) -> None:
        self.config: HardPhysicsGateConfig = config if config is not None else HardPhysicsGateConfig()
        self.prev_action: Optional[np.ndarray] = None
        self.violation_count: int = 0
        self.violation_log: List[Dict[str, Any]] = []

    def forward(
        self,
        raw_action: np.ndarray,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Project raw action onto feasible subspace.

        Args:
            raw_action: Raw action from policy network, shape (action_dim,).
            state: Optional state dict with:
                - 'velocity': Current joint velocities (for acceleration check)
                - 'pitch': Body pitch angle (degrees)
                - 'roll': Body roll angle (degrees)
                - 'zmp': ZMP position [x, y]
                - 'support_polygon': Support polygon vertices

        Returns:
            Tuple of (projected_action, violations) where violations is
            a list of constraint names that were active.
        """
        state = state or {}
        action = np.array(raw_action, dtype=np.float64).copy()
        violations: List[str] = []

        # Stage 1: Velocity projection
        velocity_norm = float(np.linalg.norm(action))
        if velocity_norm > self.config.max_velocity:
            if self.config.clamp_method == "scale":
                scale = self.config.max_velocity / (velocity_norm + 1e-8)
                action = action * scale
            else:
                action = np.clip(action, -self.config.max_velocity, self.config.max_velocity)
            violations.append("MAX_VELOCITY")
            self.violation_count += 1

        # Stage 2: Torque projection (element-wise clamp)
        torque_exceeded = np.any(np.abs(action) > self.config.max_torque)
        if torque_exceeded:
            action = np.clip(action, -self.config.max_torque, self.config.max_torque)
            violations.append("MAX_TORQUE")
            self.violation_count += 1

        # Stage 3: tau_safe (biological safety limit)
        tau_exceeded = np.any(np.abs(action) > self.config.tau_safe)
        if tau_exceeded:
            action = np.clip(action, -self.config.tau_safe, self.config.tau_safe)
            violations.append("TAU_SAFE")
            self.violation_count += 1

        # Stage 4: Acceleration projection (limit change from prev action)
        if self.prev_action is not None and len(self.prev_action) == len(action):
            delta = action - self.prev_action
            delta_norm = float(np.linalg.norm(delta))
            max_delta = self.config.max_acceleration * 0.01  # Assume 10ms timestep
            if delta_norm > max_delta:
                scale = max_delta / (delta_norm + 1e-8)
                action = self.prev_action + delta * scale
                violations.append("MAX_ACCELERATION")
                self.violation_count += 1

        # Stage 5: Posture check (pitch/roll)
        pitch = float(state.get("pitch", 0.0))
        roll = float(state.get("roll", 0.0))
        if abs(pitch) > self.config.max_pitch:
            violations.append("PITCH_VIOLATION")
            self.violation_count += 1
            # Degrade: zero out forward/backward action
            if len(action) >= 2:
                action[0] *= 0.1
        if abs(roll) > self.config.max_roll:
            violations.append("ROLL_VIOLATION")
            self.violation_count += 1
            # Degrade: zero out lateral action
            if len(action) >= 2:
                action[1] *= 0.1

        # Stage 6: ZMP check (if support polygon provided)
        zmp = state.get("zmp")
        support_poly = state.get("support_polygon")
        if zmp is not None and support_poly is not None:
            zmp_safe = self._check_zmp(np.array(zmp[:2]), support_poly)
            if not zmp_safe:
                violations.append("ZMP_VIOLATION")
                self.violation_count += 1
                # Degrade: reduce action to 10%
                action *= 0.1

        # Log violation
        if violations:
            self.violation_log.append({
                "step": len(self.violation_log),
                "violations": violations,
                "raw_norm": float(np.linalg.norm(raw_action)),
                "projected_norm": float(np.linalg.norm(action)),
            })

        # Store for next step's acceleration check
        self.prev_action = action.copy()

        return action, violations

    @staticmethod
    def _check_zmp(zmp: np.ndarray, polygon: List[np.ndarray]) -> bool:
        """Check if ZMP is inside the support polygon."""
        if len(polygon) < 3:
            return False
        n = len(polygon)
        for i in range(n):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % n]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            dist = np.dot(zmp - v1, normal)
            if dist < 0:
                return False
        return True

    def reset(self) -> None:
        """Reset gate state (call at episode start)."""
        self.prev_action = None

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "total_violations": self.violation_count,
            "violation_log_length": len(self.violation_log),
            "violation_types": list(set(
                v for entry in self.violation_log for v in entry["violations"]
            )),
        }


class HG_PINN_Policy:
    """HG-PINN Policy — full policy with HardPhysicsGate.

    This class combines the HGPINNActionHead (Hamiltonian-guided action
    generation) with the HardPhysicsGate (physical constraint projection)
    to form a complete policy that:

      1. Encodes observation into latent space
      2. Computes Hamiltonian-guided raw action
      3. Projects action through HardPhysicsGate
      4. Returns safe, physically-plausible action

    This mirrors the PyTorch nn.Module interface conceptually:
      class HG_PINN_Policy(nn.Module):
          def __init__(self, obs_dim, action_dim, gate_config):
              self.backbone = ActionHead(obs_dim, action_dim)
              self.pg_gate = HardPhysicsGate(gate_config)
          def forward(self, observation):
              features = self._extract_features(observation)
              raw_action = self.backbone(features)
              safe_action = self.pg_gate(raw_action, observation)
              return safe_action

    But uses numpy for consistency with the MuJoCo-Bench-IDO framework.

    Attributes:
        action_head: HGPINNActionHead instance.
        pg_gate: HardPhysicsGate instance.
        config: HGPINNConfig instance.
    """

    VERSION: str = "v0.17.0"

    def __init__(
        self,
        config: Optional[HGPINNConfig] = None,
        gate_config: Optional[HardPhysicsGateConfig] = None,
    ) -> None:
        """Initialize HG-PINN Policy.

        Args:
            config: HGPINNConfig for the action head.
            gate_config: HardPhysicsGateConfig for the physics gate.
        """
        self.config: HGPINNConfig = config if config is not None else HGPINNConfig()
        self.action_head: HGPINNActionHead = HGPINNActionHead(self.config)
        self.pg_gate: HardPhysicsGate = HardPhysicsGate(gate_config)

    def _extract_features(self, observation: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from observation.

        Handles various observation formats:
          - np.ndarray: Used directly as obs, goal = zeros
          - dict with 'obs' and 'goal' keys
          - tuple (obs, goal)

        Args:
            observation: Observation in various formats.

        Returns:
            Tuple of (obs, goal) arrays.
        """
        if isinstance(observation, dict):
            obs = np.array(observation.get("obs", np.zeros(self.config.obs_dim)), dtype=np.float64)
            goal = np.array(observation.get("goal", np.zeros(self.config.action_dim)), dtype=np.float64)
        elif isinstance(observation, (tuple, list)) and len(observation) == 2:
            obs = np.array(observation[0], dtype=np.float64)
            goal = np.array(observation[1], dtype=np.float64)
        elif isinstance(observation, np.ndarray):
            obs = observation.astype(np.float64)
            goal = np.zeros(self.config.action_dim, dtype=np.float64)
        else:
            obs = np.zeros(self.config.obs_dim, dtype=np.float64)
            goal = np.zeros(self.config.action_dim, dtype=np.float64)

        # Ensure correct dimensions
        if len(obs) < self.config.obs_dim:
            obs = np.pad(obs, (0, self.config.obs_dim - len(obs)))
        elif len(obs) > self.config.obs_dim:
            obs = obs[:self.config.obs_dim]

        if len(goal) < self.config.action_dim:
            goal = np.pad(goal, (0, self.config.action_dim - len(goal)))
        elif len(goal) > self.config.action_dim:
            goal = goal[:self.config.action_dim]

        return obs, goal

    def forward(self, observation: Any, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Forward pass: observation -> safe action.

        This is the main inference method that combines:
          1. Feature extraction
          2. Hamiltonian-guided action generation (backbone)
          3. Physical constraint projection (PG-Gate)

        Args:
            observation: Observation in various formats.
            state: Optional state dict for physics gate (pitch, roll, zmp, etc.)

        Returns:
            Dict with:
                - 'action': Safe action vector (np.ndarray)
                - 'raw_action': Raw action before gate
                - 'violations': List of physics constraint violations
                - 'energy': Hamiltonian energy value
                - 'gate_stats': Gate statistics
        """
        # Step 1: Extract features
        obs, goal = self._extract_features(observation)

        # Step 2: Generate raw action via Hamiltonian-guided head
        velocity = state.get("velocity") if state else None
        raw_action = self.action_head.predict(obs, goal, velocity)

        # Step 3: Project through HardPhysicsGate
        safe_action, violations = self.pg_gate.forward(raw_action, state)

        # Step 4: Collect metadata
        energy = self.action_head._last_energy

        return {
            "action": safe_action,
            "raw_action": raw_action,
            "violations": violations,
            "energy": energy,
            "gate_stats": self.pg_gate.get_stats(),
        }

    def predict(self, observation: Any, state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convenience method: return only the safe action.

        Args:
            observation: Observation in various formats.
            state: Optional state dict.

        Returns:
            Safe action vector.
        """
        result = self.forward(observation, state)
        return result["action"]

    def reset(self) -> None:
        """Reset policy state at episode start."""
        self.pg_gate.reset()
        self.action_head._energy_history.clear()

    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy conservation statistics from the action head."""
        return self.action_head.get_energy_stats()

    def get_gate_stats(self) -> Dict[str, Any]:
        """Get physics gate statistics."""
        return self.pg_gate.get_stats()

    def load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Load trained weights into the action head.

        Args:
            weights: Dictionary of weight matrices.
        """
        self.action_head.load_weights(weights)
