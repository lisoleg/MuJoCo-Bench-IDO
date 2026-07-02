"""
BayesianIntent — Beta-Bernoulli Intent Clarity Model
======================================================

Implements a lightweight Bayesian intent clarity model using
Beta-Bernoulli distribution. The model tracks whether the agent's
actions are "intentional" (η decreasing) vs "random" (η increasing
or stagnant) and computes an intent clarity score ℐ ∈ [0, 1].

ℐ = 1: Agent's intent is crystal clear — η consistently decreasing
ℐ = 0: Agent's intent is completely unclear — η random/stagnant

The Beta-Bernoulli model:
  Prior: Beta(α=1, β=1) — uniform (no prior belief about intent)
  Likelihood: η decreasing → Bernoulli(1) (intentional)
              η not decreasing → Bernoulli(0) (random)
  Posterior: Beta(α+n_intentional, β+n_random) — updated after each step

Author: MuJoCo-Bench-IDO v0.6.0 — Machine Conscience Audit Framework
"""

import numpy as np
from typing import Dict, Any, List, Optional

IDO_BAYESIAN_INTENT_VERSION: str = "v0.1.0"

# Intent labels for classification
INTENT_LABELS: List[str] = [
    "goal_directed",    # η decreasing → intentional goal pursuit
    "exploratory",      # η plateau with variation → exploration
    "stagnant",         # η plateau without variation → stuck
    "divergent",        # η increasing → diverging from goal
]


class BayesianIntent:
    """Beta-Bernoulli intent clarity model for IDO agents.

    Tracks the agent's intent clarity ℐ using a Bayesian update
    on η trajectory. Each step, the model observes whether η
    decreased (intentional) or not (random/stagnant), updates
    the Beta posterior, and computes intent clarity.

    Attributes:
        VERSION: Model version string.
    """

    VERSION: str = IDO_BAYESIAN_INTENT_VERSION

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Initialize BayesianIntent with Beta prior parameters.

        Args:
            prior_alpha: Beta distribution α parameter (prior intentional count).
                        Default 1.0 (uniform prior).
            prior_beta: Beta distribution β parameter (prior random count).
                       Default 1.0 (uniform prior).
        """
        self._intent_posterior: Dict[str, np.ndarray] = {
            "goal_directed": np.array([prior_alpha, prior_beta]),
            "exploratory": np.array([prior_alpha, prior_beta]),
            "stagnant": np.array([prior_alpha, prior_beta]),
            "divergent": np.array([prior_alpha, prior_beta]),
        }
        self._eta_history: List[float] = []
        self._last_eta: Optional[float] = None

    def update(self,
               observation: Optional[Dict[str, Any]] = None,
               action: Optional[np.ndarray] = None,
               eta: float = 0.0) -> Dict[str, Any]:
        """Update intent posterior based on η observation.

        Observes whether η decreased (intentional) or not (random),
        updates the Beta posterior for each intent label, and
        computes the most likely intent.

        Args:
            observation: Optional observation dict (for context).
            action: Optional action array (for action analysis).
            eta: Current κ-Snap residual η value.

        Returns:
            Dict with keys:
            - intent: str — most likely intent label
            - clarity: float — intent clarity ℐ ∈ [0, 1]
            - posterior: Dict[str, np.ndarray] — full posterior state
        """
        self._eta_history.append(eta)
        if len(self._eta_history) > 100:
            self._eta_history = self._eta_history[-100:]

        # Determine η trend for intent classification
        if len(self._eta_history) >= 2:
            prev_eta: float = self._eta_history[-2]
            delta_eta: float = eta - prev_eta

            # Intent classification based on η trend:
            # η decreasing → goal_directed (intentional)
            # η plateau with small variation → exploratory
            # η plateau with no variation → stagnant
            # η increasing → divergent
            if delta_eta < -0.001:
                # η decreased → goal_directed
                self._bayesian_update("goal_directed", likelihood=1.0)
                self._bayesian_update("exploratory", likelihood=0.2)
                self._bayesian_update("stagnant", likelihood=0.0)
                self._bayesian_update("divergent", likelihood=0.0)
            elif abs(delta_eta) <= 0.001:
                # η plateau → exploratory or stagnant
                # Check variation in recent window
                if len(self._eta_history) >= 5:
                    recent_std: float = float(np.std(self._eta_history[-5:]))
                    if recent_std > 0.01:
                        # Variation exists → exploratory
                        self._bayesian_update("goal_directed", likelihood=0.2)
                        self._bayesian_update("exploratory", likelihood=1.0)
                        self._bayesian_update("stagnant", likelihood=0.1)
                        self._bayesian_update("divergent", likelihood=0.0)
                    else:
                        # No variation → stagnant
                        self._bayesian_update("goal_directed", likelihood=0.0)
                        self._bayesian_update("exploratory", likelihood=0.1)
                        self._bayesian_update("stagnant", likelihood=1.0)
                        self._bayesian_update("divergent", likelihood=0.1)
                else:
                    self._bayesian_update("exploratory", likelihood=0.5)
            else:
                # η increased → divergent
                self._bayesian_update("goal_directed", likelihood=0.0)
                self._bayesian_update("exploratory", likelihood=0.1)
                self._bayesian_update("stagnant", likelihood=0.0)
                self._bayesian_update("divergent", likelihood=1.0)

        # Compute intent clarity and most likely intent
        clarity: float = self.get_intent_clarity()
        most_likely: str = self.get_most_likely_intent()

        self._last_eta = eta

        return {
            "intent": most_likely,
            "clarity": clarity,
            "posterior": {k: v.tolist() for k, v in self._intent_posterior.items()},
        }

    def get_intent_clarity(self) -> float:
        """Compute intent clarity ℐ from posterior distribution.

        Intent clarity is computed as the normalized entropy difference:
        ℐ = 1 - H(posterior) / H_max

        Where H is the Shannon entropy of the posterior distribution
        over intent labels, and H_max is the maximum entropy (uniform).

        High ℐ → posterior concentrated on one intent (clear)
        Low ℐ → posterior spread across intents (unclear)

        Returns:
            Intent clarity score ℐ ∈ [0, 1].
        """
        # Compute probability of each intent from Beta posterior mean
        probs: np.ndarray = np.zeros(len(INTENT_LABELS))
        for i, label in enumerate(INTENT_LABELS):
            alpha: float = self._intent_posterior[label][0]
            beta: float = self._intent_posterior[label][1]
            probs[i] = alpha / (alpha + beta)  # Beta mean

        # Normalize probabilities
        prob_sum: float = float(np.sum(probs))
        if prob_sum > 0:
            probs = probs / prob_sum

        # Compute Shannon entropy
        entropy: float = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        # Maximum entropy (uniform distribution over 4 labels)
        h_max: float = np.log2(len(INTENT_LABELS))

        # Intent clarity = 1 - normalized entropy
        clarity: float = 1.0 - entropy / h_max if h_max > 0 else 0.0
        clarity = max(0.0, min(1.0, clarity))

        return float(clarity)

    def get_most_likely_intent(self) -> str:
        """Get the most likely intent label from posterior.

        Returns:
            Intent label string with highest posterior probability.
        """
        best_label: str = INTENT_LABELS[0]
        best_prob: float = 0.0

        for label in INTENT_LABELS:
            alpha: float = self._intent_posterior[label][0]
            beta: float = self._intent_posterior[label][1]
            prob: float = alpha / (alpha + beta)
            if prob > best_prob:
                best_prob = prob
                best_label = label

        return best_label

    def reset(self) -> None:
        """Reset posterior to uniform prior (Beta(1,1))."""
        for label in INTENT_LABELS:
            self._intent_posterior[label] = np.array([1.0, 1.0])
        self._eta_history = []
        self._last_eta = None

    def _bayesian_update(self, label: str, likelihood: float) -> None:
        """Apply Bayesian update to Beta posterior for a specific intent label.

        Beta-Bernoulli update:
          posterior_alpha += likelihood (number of "successes")
          posterior_beta += (1 - likelihood) (number of "failures")

        Args:
            label: Intent label to update.
            likelihood: Likelihood value ∈ [0, 1]. Higher = more evidence
                       for this intent.
        """
        if label not in self._intent_posterior:
            return

        # Beta-Bernoulli conjugate update
        self._intent_posterior[label][0] += likelihood
        self._intent_posterior[label][1] += (1.0 - likelihood)
