"""
ψ-Anchor LoRA — DPO-Based ψ-Compliance Preference Training
============================================================

v0.16.25 P2: ψ-Anchor LoRA

Fine-tunes VLA/LLM models with ψ-Anchor compliance preferences using
Direct Preference Optimization (DPO). The goal is to make the model
prefer ψ-compliant actions over violating ones.

DPO Training:
  Given pairs (a_compliant, a_violating) from the κ-Snap audit trail,
  optimize the model to increase P(a_compliant) / P(a_violating).

  Loss = -log σ(β · (log π(a_compliant) - log π(a_violating)
                       - log π_ref(a_compliant) + log π_ref(a_violating)))

  Where:
    π = policy being trained
    π_ref = reference policy (frozen)
    β = temperature parameter
    σ = sigmoid

LoRA (Low-Rank Adaptation):
  Instead of full fine-tuning, injects low-rank matrices into the
  model's attention layers. This is memory-efficient and prevents
  catastrophic forgetting.

  ΔW = A · B  where A ∈ R^{d×r}, B ∈ R^{r×d}, r << d

This is a simplified Python implementation that:
  1. Collects preference pairs from κ-Snap audit trail
  2. Computes DPO loss
  3. Applies LoRA-style updates to a simple action preference model

Author: MuJoCo-Bench-IDO v0.16.25 — P2 Feature
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PreferencePair:
    """A preference pair for DPO training.

    Attributes:
        obs: Observation when the decision was made.
        action_compliant: ψ-Anchor compliant action.
        action_violating: ψ-Anchor violating action.
        violation_type: Type of violation in the violating action.
        eta_compliant: η after compliant action.
        eta_violating: η after violating action.
    """
    obs: np.ndarray = field(default_factory=lambda: np.zeros(10))
    action_compliant: np.ndarray = field(default_factory=lambda: np.zeros(7))
    action_violating: np.ndarray = field(default_factory=lambda: np.zeros(7))
    violation_type: str = ""
    eta_compliant: float = 0.0
    eta_violating: float = 0.0


class PsiLoRATrainer:
    """ψ-Anchor LoRA preference trainer.

    Collects preference pairs from the κ-Snap audit trail and trains
    a low-rank preference model using DPO. The trained model can then
    be used to bias VLA/LLM action generation toward ψ-compliance.

    Architecture:
      Preference Model: obs → action_score (scalar)
      LoRA layers: rank-4 decomposition of the scoring matrix

    Usage:
        trainer = PsiLoRATrainer(obs_dim=10, action_dim=7)

        # Collect preference pairs from audit trail
        trainer.add_preference_pair(pair)

        # Train
        loss = trainer.train_step()

        # Use trained model
        score = trainer.score_action(obs, action)  # Higher = more compliant

    Attributes:
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        rank: LoRA rank (default 4).
        beta: DPO temperature parameter.
    """

    VERSION: str = "v0.16.25"

    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 7,
        rank: int = 4,
        beta: float = 0.1,
        learning_rate: float = 1e-3,
    ) -> None:
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.rank: int = rank
        self.beta: float = beta
        self.lr: float = learning_rate

        # Preference model: (obs, action) → score
        # Main weights (frozen reference)
        d = obs_dim + action_dim
        self._ref_w: np.ndarray = np.random.randn(d, 1) * 0.01
        self._ref_b: np.ndarray = np.zeros(1)

        # LoRA weights (trainable): ΔW = A @ B
        self._lora_A: np.ndarray = np.random.randn(d, rank) * 0.01
        self._lora_B: np.ndarray = np.random.randn(rank, 1) * 0.01

        # Preference pair buffer
        self._pairs: deque = deque(maxlen=10000)
        self._train_steps: int = 0
        self._loss_history: List[float] = []

    def add_preference_pair(self, pair: PreferencePair) -> None:
        """Add a preference pair to the training buffer.

        Args:
            pair: PreferencePair from κ-Snap audit trail.
        """
        self._pairs.append(pair)

    def _score(self, obs: np.ndarray, action: np.ndarray, use_lora: bool = True) -> float:
        """Compute preference score for an (obs, action) pair.

        Args:
            obs: Observation vector.
            action: Action vector.
            use_lora: Whether to include LoRA adaptation.

        Returns:
            Preference score (higher = more ψ-compliant).
        """
        x = np.concatenate([obs, action]).reshape(1, -1)
        # Reference score
        score = float((x @ self._ref_w + self._ref_b)[0, 0])
        # LoRA adaptation
        if use_lora:
            score += float((x @ self._lora_A @ self._lora_B)[0, 0])
        return score

    def train_step(self, batch_size: int = 32) -> float:
        """Execute one DPO training step.

        Samples a batch of preference pairs and updates the LoRA weights
        to increase the score gap between compliant and violating actions.

        Args:
            batch_size: Number of preference pairs per training step.

        Returns:
            DPO loss value.
        """
        if len(self._pairs) < batch_size:
            return 0.0

        # Sample batch
        indices = np.random.choice(len(self._pairs), min(batch_size, len(self._pairs)), replace=False)
        batch = [self._pairs[i] for i in indices]

        total_loss = 0.0
        grad_A = np.zeros_like(self._lora_A)
        grad_B = np.zeros_like(self._lora_B)

        for pair in batch:
            # Scores
            s_compliant = self._score(pair.obs, pair.action_compliant)
            s_violating = self._score(pair.obs, pair.action_violating)

            # DPO loss: -log σ(β · (s_compliant - s_violating))
            diff = self.beta * (s_compliant - s_violating)
            sigmoid = 1.0 / (1.0 + np.exp(-diff))
            loss = -np.log(max(sigmoid, 1e-8))
            total_loss += loss

            # Gradient of loss w.r.t. LoRA weights
            # d_loss/d_diff = -(1 - sigmoid)
            # d_diff/d_s_compliant = beta, d_diff/d_s_violating = -beta
            d_loss_d_diff = -(1 - sigmoid)

            x_c = np.concatenate([pair.obs, pair.action_compliant])
            x_v = np.concatenate([pair.obs, pair.action_violating])

            # Gradient for LoRA: d_s/d_A = x @ B^T, d_s/d_B = A^T @ x
            grad_A += self.beta * d_loss_d_diff * np.outer(x_c, self._lora_B.flatten())
            grad_A -= self.beta * d_loss_d_diff * np.outer(x_v, self._lora_B.flatten())
            grad_B += self.beta * d_loss_d_diff * np.outer(self._lora_A.T @ x_c, np.array([1.0]))
            grad_B -= self.beta * d_loss_d_diff * np.outer(self._lora_A.T @ x_v, np.array([1.0]))

        # Update LoRA weights
        n = len(batch)
        self._lora_A -= self.lr * grad_A / n
        self._lora_B -= self.lr * grad_B / n

        avg_loss = total_loss / n
        self._loss_history.append(avg_loss)
        self._train_steps += 1
        return avg_loss

    def score_action(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Score an action for ψ-compliance.

        Higher score = more ψ-compliant. Can be used to bias VLA action
        generation toward safe, physically-consistent actions.

        Args:
            obs: Current observation.
            action: Proposed action.

        Returns:
            ψ-compliance score (higher = more compliant).
        """
        return self._score(obs, action)

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "train_steps": self._train_steps,
            "pairs_collected": len(self._pairs),
            "last_loss": self._loss_history[-1] if self._loss_history else 0.0,
            "avg_loss_10": float(np.mean(self._loss_history[-10:])) if len(self._loss_history) >= 10 else 0.0,
            "lora_rank": self.rank,
            "beta": self.beta,
        }

    def get_lora_weights(self) -> Dict[str, np.ndarray]:
        """Get LoRA weights for injection into VLA/LLM model.

        Returns:
            Dict with 'A' and 'B' matrices.
        """
        return {'A': self._lora_A.copy(), 'B': self._lora_B.copy()}
