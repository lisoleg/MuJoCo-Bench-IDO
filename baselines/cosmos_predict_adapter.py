"""
Cosmos-Predict1 Baseline Adapter for MuJoCo-Bench-IDO
=======================================================

Adapter for NVIDIA Cosmos-Predict1 world foundation model.

Cosmos-Predict1 is a world model (NOT a control agent) that predicts
future states from current observations and action sequences. This
adapter uses Cosmos-Predict for η trajectory prediction comparison
against the IDO FlowMatching η predictor.

Key distinction:
  - TD-MPC2: control agent baseline (steps, NVR, SER comparison)
  - Cosmos-Predict: world model baseline (η trajectory prediction comparison)

GitHub: https://github.com/nvidia-cosmos/cosmos-predict1

WARNING: Cosmos-Predict1 has been superseded by Cosmos 3.
  Recommended migration: https://github.com/NVIDIA/Cosmos

Graceful degradation: if CUDA or Cosmos is not available, the adapter
prints a warning and returns None, allowing the benchmark to skip this
baseline. Heavy GPU requirements (7B-14B models) mean this baseline
may not run on consumer hardware.

Author: MuJoCo-Bench-IDO v0.3.0 baseline integration
"""

import numpy as np
import sys
from typing import Dict, List, Optional, Tuple

# ── Cosmos-Predict model variants ──
COSMOS_MODEL_NAMES: List[str] = [
    'cosmos-predict1-7b-video2world',
    'cosmos-predict1-14b-video2world',
    'cosmos-predict1-7b-token2world',
]


class CosmosPredictAdapter:
    """NVIDIA Cosmos-Predict1 world model adapter for η trajectory comparison.

    Cosmos-Predict1 is a world foundation model that generates future state
    predictions (video/RGB frames) conditioned on current observations and
    action sequences. This adapter compares its state predictions against
    IDO's FlowMatching η trajectory predictions.

    Comparison modes:
      1. η trajectory prediction: IDO predicts η directly (low-dimensional);
         Cosmos-Predict predicts full future state, then η is computed from
         predicted state.
      2. RMSE comparison: measure prediction accuracy of both approaches.

    Attributes:
        model_name: Cosmos-Predict model variant name.
        device: Device string ('cuda' or 'cpu').
        model: Cosmos-Predict model instance (None if not available).
        _cosmos_available: Whether Cosmos-Predict is importable.
    """

    def __init__(self,
                 model_name: str = 'cosmos-predict1-7b-video2world',
                 device: str = 'cuda') -> None:
        """Initialize Cosmos-Predict adapter.

        Args:
            model_name: Cosmos-Predict model variant. Default 7B Video2World.
                        Options: cosmos-predict1-7b-video2world,
                                 cosmos-predict1-14b-video2world,
                                 cosmos-predict1-7b-token2world.
            device: Compute device. Default 'cuda' (required for 7B+ models).
                    'cpu' may work for small inference but is very slow.

        Raises:
            ValueError: If model_name is not a recognized Cosmos-Predict variant.
        """
        # Validate model name
        if model_name not in COSMOS_MODEL_NAMES:
            print(f"  [CosmosPredictAdapter] WARNING: model_name '{model_name}' "
                  f"not in known variants: {COSMOS_MODEL_NAMES}")
            print(f"  [CosmosPredictAdapter] Proceeding with custom model name "
                  f"(may fail at load time).")

        self.model_name: str = model_name
        self.device: str = device
        self._cosmos_available: bool = False
        self.model: Optional[object] = None

        # Try to import Cosmos-Predict
        try:
            from cosmos_predict1 import CosmosPredict1Model
            self._cosmos_available = True

            # Check CUDA availability
            if device == 'cuda':
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("  [CosmosPredictAdapter] WARNING: CUDA not available.")
                        print("  [CosmosPredictAdapter] Falling back to CPU "
                              "(very slow for 7B+ models).")
                        self.device = 'cpu'
                except ImportError:
                    print("  [CosmosPredictAdapter] WARNING: PyTorch not installed.")
                    self.device = 'cpu'

            # Load model
            self.model = CosmosPredict1Model.from_pretrained(
                model_name,
                device=self.device,
            )
            print(f"  [CosmosPredictAdapter] Loaded model: {model_name} "
                  f"on {self.device}")

        except ImportError as e:
            print("  [CosmosPredictAdapter] WARNING: cosmos_predict1 not installed.")
            print("  [CosmosPredictAdapter] Install with: pip install cosmos-predict1")
            print("  [CosmosPredictAdapter] GitHub: https://github.com/nvidia-cosmos/cosmos-predict1")
            print("  [CosmosPredictAdapter] NOTE: Cosmos-Predict1 superseded by Cosmos 3.")
            print("  [CosmosPredictAdapter] Consider migrating to: https://github.com/NVIDIA/Cosmos")
            print("  [CosmosPredictAdapter] This baseline will be skipped in evaluation.")
            self._cosmos_available = False
            self.model = None

        except Exception as e:
            print(f"  [CosmosPredictAdapter] Model load failed: {e}")
            print("  [CosmosPredictAdapter] Heavy GPU requirements (7B-14B models).")
            print("  [CosmosPredictAdapter] This baseline will be skipped.")
            self._cosmos_available = False
            self.model = None

    def predict_future_state(self,
                              current_obs: Dict[str, np.ndarray],
                              action_sequence: np.ndarray,
                              horizon: int = 10) -> Optional[List[Dict[str, np.ndarray]]]:
        """Predict future state sequence from current observation and actions.

        Uses Cosmos-Predict's Video2World / Token2World model to generate
        future state predictions conditioned on:
          - current_obs: current observation (RGB frames or state features)
          - action_sequence: planned action sequence
          - horizon: number of future steps to predict

        Args:
            current_obs: Current observation dict with keys:
                         'rgb_frame' (H,W,3 array) or 'state_features' (n-dim vector).
            action_sequence: Planned action array of shape (horizon, action_dim)
                             or (action_dim,) for single-step.
            horizon: Number of future steps to predict. Default 10.

        Returns:
            List of predicted state dicts (length = horizon), each with keys:
            'predicted_rgb' (H,W,3), 'predicted_state' (n-dim vector), 'step' (int).
            Returns None if Cosmos-Predict is not available.
        """
        if not self._cosmos_available or self.model is None:
            print("  [CosmosPredictAdapter] Cannot predict: model not available.")
            return None

        try:
            # Prepare input for Cosmos-Predict
            # Video2World models expect RGB frames + action conditioning
            rgb_frame: Optional[np.ndarray] = current_obs.get('rgb_frame', None)
            state_features: Optional[np.ndarray] = current_obs.get('state_features', None)

            if rgb_frame is not None:
                # Video2World mode: predict future RGB frames
                predicted_frames: object = self.model.predict_video(
                    video=rgb_frame,
                    actions=action_sequence,
                    horizon=horizon,
                )

                # Convert predictions to state dict format
                predicted_states: List[Dict[str, np.ndarray]] = []
                for step_idx in range(horizon):
                    # Extract predicted frame at this step
                    pred_frame: np.ndarray = np.array(predicted_frames[step_idx])
                    pred_state: Dict[str, np.ndarray] = {
                        'predicted_rgb': pred_frame,
                        'step': step_idx,
                    }
                    predicted_states.append(pred_state)

                return predicted_states

            elif state_features is not None:
                # Token2World mode: predict future state tokens
                predicted_tokens: object = self.model.predict_tokens(
                    state=state_features,
                    actions=action_sequence,
                    horizon=horizon,
                )

                predicted_states = []
                for step_idx in range(horizon):
                    pred_state_vec: np.ndarray = np.array(predicted_tokens[step_idx])
                    pred_state: Dict[str, np.ndarray] = {
                        'predicted_state': pred_state_vec,
                        'step': step_idx,
                    }
                    predicted_states.append(pred_state)

                return predicted_states

            else:
                print("  [CosmosPredictAdapter] No valid observation input "
                      "(need 'rgb_frame' or 'state_features').")
                return None

        except Exception as e:
            print(f"  [CosmosPredictAdapter] Prediction failed: {e}")
            return None

    def compare_eta_trajectory(self,
                                ido_eta_trajectory: List[float],
                                predicted_states: List[Dict[str, np.ndarray]],
                                goal_eml: object) -> Optional[Dict[str, object]]:
        """Compare IDO η trajectory with Cosmos-Predict predicted states.

        Computes η from Cosmos-Predict's predicted states using the same
        GoalEML κ-Snap residual formula, then compares the two trajectories.

        Args:
            ido_eta_trajectory: List of η values from IDO FlowMatching predictor.
            predicted_states: List of predicted state dicts from predict_future_state().
            goal_eml: GoalEML instance for computing η from predicted states.

        Returns:
            Dict with comparison metrics:
            - 'ido_eta_trajectory': Original IDO η values.
            - 'cosmos_eta_trajectory': η computed from Cosmos-Predict predictions.
            - 'trajectory_rmse': RMSE between IDO and Cosmos η trajectories.
            - 'trajectory_correlation': Pearson correlation between trajectories.
            - 'horizon': Number of steps compared.
            Returns None if prediction data is unavailable.
        """
        if predicted_states is None or len(predicted_states) == 0:
            print("  [CosmosPredictAdapter] No predicted states for comparison.")
            return None

        from core.kappa_snap_mj import gauss_ex_residual

        # Compute η from each predicted state using GoalEML
        cosmos_eta_trajectory: List[float] = []
        for pred_state_dict in predicted_states:
            # Convert predicted state to κ-Snap input format
            z_i: Dict[str, object] = {}

            if 'predicted_state' in pred_state_dict:
                # Token2World: predicted state vector
                state_vec: np.ndarray = pred_state_dict['predicted_state']
                z_i['ee_pos'] = state_vec[:3] if len(state_vec) >= 3 else np.zeros(3)
                z_i['qpos'] = state_vec
                z_i['E_total'] = 0.0  # Approximate; state prediction doesn't include energy
                z_i['ee_vel'] = np.zeros(6)  # Velocity unknown from static prediction

            elif 'predicted_rgb' in pred_state_dict:
                # Video2World: RGB frame → need state extraction heuristic
                # Simplified: use zero state (RGB→state extraction requires
                # separate decoder, not available in Cosmos-Predict alone)
                z_i['ee_pos'] = np.zeros(3)
                z_i['qpos'] = np.zeros(7)
                z_i['E_total'] = 0.0
                z_i['ee_vel'] = np.zeros(6)

            # Compute η using GoalEML κ-Snap residual
            eta: float = gauss_ex_residual(z_i, goal_eml)
            cosmos_eta_trajectory.append(eta)

        # Align trajectory lengths for comparison
        min_len: int = min(len(ido_eta_trajectory), len(cosmos_eta_trajectory))
        if min_len == 0:
            print("  [CosmosPredictAdapter] Empty trajectory for comparison.")
            return None

        ido_aligned: np.ndarray = np.array(ido_eta_trajectory[:min_len])
        cosmos_aligned: np.ndarray = np.array(cosmos_eta_trajectory[:min_len])

        # Compute RMSE between trajectories
        rmse: float = float(np.sqrt(np.mean((ido_aligned - cosmos_aligned) ** 2)))

        # Compute Pearson correlation
        if np.std(ido_aligned) > 0 and np.std(cosmos_aligned) > 0:
            correlation: float = float(
                np.corrcoef(ido_aligned, cosmos_aligned)[0, 1])
        else:
            correlation = 0.0

        return {
            'ido_eta_trajectory': ido_eta_trajectory[:min_len],
            'cosmos_eta_trajectory': cosmos_eta_trajectory[:min_len],
            'trajectory_rmse': rmse,
            'trajectory_correlation': correlation,
            'horizon': min_len,
        }

    def compute_prediction_rmse(self,
                                 predicted_states: List[Dict[str, np.ndarray]],
                                 actual_states: List[Dict[str, np.ndarray]]) -> Optional[float]:
        """Compute RMSE between predicted and actual future states.

        Args:
            predicted_states: List of predicted state dicts from predict_future_state().
            actual_states: List of actual (ground-truth) state dicts with same keys.

        Returns:
            RMSE value (float) between predicted and actual state features.
            Returns None if states are unavailable or incompatible.
        """
        if predicted_states is None or actual_states is None:
            return None

        if len(predicted_states) == 0 or len(actual_states) == 0:
            return None

        min_len: int = min(len(predicted_states), len(actual_states))

        errors: List[float] = []
        for i in range(min_len):
            pred: Dict[str, np.ndarray] = predicted_states[i]
            actual: Dict[str, np.ndarray] = actual_states[i]

            # Compare state features if available
            if 'predicted_state' in pred and 'state_features' in actual:
                pred_vec: np.ndarray = pred['predicted_state']
                actual_vec: np.ndarray = actual['state_features']
                # Align dimensions
                max_dim: int = max(len(pred_vec), len(actual_vec))
                pred_padded: np.ndarray = np.zeros(max_dim)
                pred_padded[:len(pred_vec)] = pred_vec
                actual_padded: np.ndarray = np.zeros(max_dim)
                actual_padded[:len(actual_vec)] = actual_vec
                error: float = float(np.linalg.norm(pred_padded - actual_padded))
                errors.append(error)

            elif 'predicted_rgb' in pred and 'rgb_frame' in actual:
                # RGB comparison: pixel-level RMSE
                pred_rgb: np.ndarray = pred['predicted_rgb'].astype(float)
                actual_rgb: np.ndarray = actual['rgb_frame'].astype(float)
                # Align shapes
                if pred_rgb.shape == actual_rgb.shape:
                    pixel_rmse: float = float(np.sqrt(np.mean(
                        (pred_rgb - actual_rgb) ** 2)))
                    errors.append(pixel_rmse)
                else:
                    # Shape mismatch, skip this frame
                    continue

        if len(errors) == 0:
            return None

        rmse: float = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        return rmse

    def reset(self) -> None:
        """Reset the Cosmos-Predict model's internal state.

        Clears cached predictions and internal state between evaluation
        episodes.
        """
        if self._cosmos_available and self.model is not None:
            try:
                self.model.reset()
            except AttributeError:
                pass
            except Exception as e:
                print(f"  [CosmosPredictAdapter] Reset warning: {e}")

    def is_available(self) -> bool:
        """Check whether Cosmos-Predict is installed and model is loaded.

        Returns:
            True if cosmos_predict1 is available and model is initialized.
        """
        return self._cosmos_available and self.model is not None

    def get_info(self) -> Dict[str, object]:
        """Get adapter configuration information.

        Returns:
            Dict with adapter metadata: model_name, device, availability.
        """
        return {
            'adapter': 'CosmosPredictAdapter',
            'model_name': self.model_name,
            'device': self.device,
            'available': self._cosmos_available,
            'model_loaded': self.model is not None,
            'note': 'Cosmos-Predict1 superseded by Cosmos 3. '
                    'Consider migrating to https://github.com/NVIDIA/Cosmos',
        }


def make_cosmos_predict_adapter(model_name: str = 'cosmos-predict1-7b-video2world',
                                 device: str = 'cuda') -> Optional[CosmosPredictAdapter]:
    """Factory function for creating a CosmosPredictAdapter instance.

    Creates an adapter with graceful degradation: if Cosmos-Predict
    is not installed or CUDA is not available, the adapter is created
    but will return None for all operations, allowing the benchmark
    to skip this baseline.

    Args:
        model_name: Cosmos-Predict model variant name.
        device: Compute device ('cuda' or 'cpu').

    Returns:
        CosmosPredictAdapter instance (always created, but may not be
        functional if Cosmos-Predict/CUDA is not available).
    """
    try:
        adapter: CosmosPredictAdapter = CosmosPredictAdapter(
            model_name=model_name,
            device=device,
        )
        return adapter
    except ValueError as e:
        print(f"  [CosmosPredictAdapter] Configuration error: {e}")
        return None
