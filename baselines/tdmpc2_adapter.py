"""
TD-MPC2 Baseline Adapter for MuJoCo-Bench-IDO
===============================================

Adapter for the TD-MPC2 (Temporal Difference Model Predictive Control 2)
model-based RL baseline by Hansen et al. (2024).

TD-MPC2 is a scalable, robust world-model approach for continuous control
that supports dm_control tasks. This adapter provides a unified interface
compatible with the IDO agent evaluation framework.

GitHub: https://github.com/nicklashansen/tdmpc2

Supported dm_control tasks (DMControl domain, 39 tasks total):
  - humanoid-stand, hopper-stand, walker-run, reacher-easy

Model sizes: 1M / 5M / 19M / 48M / 317M parameters.
Default training budget: 1M steps.

Graceful degradation: if tdmpc2 is not installed, prints a warning
and returns None, allowing the benchmark to skip this baseline.

Author: MuJoCo-Bench-IDO v0.3.0 baseline integration
"""

import numpy as np
import sys
from typing import Dict, List, Optional, Tuple

# ── dm_control task name mapping ──
# TD-MPC2 uses hyphenated names; dm_control uses (domain, task) pairs.
DMCONTROL_TASK_MAP: Dict[str, str] = {
    'humanoid-stand': 'humanoid_stand',
    'hopper-stand':   'hopper_stand',
    'walker-run':     'walker_run',
    'reacher-easy':   'reacher_easy',
}

# Supported model sizes (config keys in TD-MPC2)
TDMPC2_MODEL_SIZES: List[int] = [1, 5, 19, 48, 317]


class TDMPC2Adapter:
    """TD-MPC2 baseline adapter for MuJoCo-Bench-IDO comparative evaluation.

    Provides a unified interface for training, evaluating, and querying
    TD-MPC2, compatible with the IDO agent evaluation framework.

    The adapter wraps TD-MPC2's train/eval/predict APIs and maps dm_control
    task names to TD-MPC2's internal naming convention.

    Attributes:
        task_name: dm_control task name (e.g., 'humanoid-stand').
        tdmpc2_task_name: TD-MPC2 internal task name (e.g., 'humanoid_stand').
        model_size: TD-MPC2 model size in millions of parameters.
        checkpoint_path: Optional path to a pre-trained checkpoint.
        model: TD-MPC2 model instance (None if tdmpc2 not installed).
        _tdmpc2_available: Whether the tdmpc2 package is importable.
    """

    def __init__(self,
                 task_name: str = 'humanoid-stand',
                 model_size: int = 5,
                 checkpoint_path: Optional[str] = None) -> None:
        """Initialize TD-MPC2 adapter with task configuration.

        Args:
            task_name: dm_control task name (e.g., 'humanoid-stand').
            model_size: TD-MPC2 model size (1/5/19/48/317 million parameters).
                        Default 5 = 5M parameters.
            checkpoint_path: Optional path to a pre-trained checkpoint file.
                             If None, the model must be trained before evaluation.

        Raises:
            ValueError: If task_name is not in DMCONTROL_TASK_MAP.
            ValueError: If model_size is not a supported value.
        """
        # Validate task name
        if task_name not in DMCONTROL_TASK_MAP:
            raise ValueError(
                f"Unsupported task '{task_name}'. "
                f"Supported tasks: {list(DMCONTROL_TASK_MAP.keys())}")

        # Validate model size
        if model_size not in TDMPC2_MODEL_SIZES:
            raise ValueError(
                f"Unsupported model_size={model_size}. "
                f"Supported sizes: {TDMPC2_MODEL_SIZES}")

        self.task_name: str = task_name
        self.tdmpc2_task_name: str = DMCONTROL_TASK_MAP[task_name]
        self.model_size: int = model_size
        self.checkpoint_path: Optional[str] = checkpoint_path

        # Try to import tdmpc2
        self._tdmpc2_available: bool = False
        self.model: Optional[object] = None

        try:
            import tdmpc2
            self._tdmpc2_available = True

            # Load from checkpoint if provided
            if checkpoint_path is not None:
                self.model = tdmpc2.TDMPC2.load(checkpoint_path)
                print(f"  [TDMPC2Adapter] Loaded checkpoint: {checkpoint_path}")
            else:
                # Create a new model instance for training
                self.model = tdmpc2.TDMPC2(
                    task=self.tdmpc2_task_name,
                    model_size=model_size,
                )
                print(f"  [TDMPC2Adapter] Created new model for task="
                      f"{self.tdmpc2_task_name}, size={model_size}M")

        except ImportError:
            print("  [TDMPC2Adapter] WARNING: tdmpc2 package not installed.")
            print("  [TDMPC2Adapter] Install with: pip install tdmpc2")
            print("  [TDMPC2Adapter] GitHub: https://github.com/nicklashansen/tdmpc2")
            print("  [TDMPC2Adapter] This baseline will be skipped in evaluation.")
            self._tdmpc2_available = False
            self.model = None

    def train(self, steps: int = 1000000) -> Optional[Dict[str, object]]:
        """Train the TD-MPC2 model on the configured task.

        Calls TD-MPC2's training API with the specified number of steps.
        Training logs are collected and returned as a metrics dict.

        Args:
            steps: Number of training steps. Default 1M (standard budget).

        Returns:
            Dict with training metrics (reward curve, episode lengths, etc.)
            or None if tdmpc2 is not available.
        """
        if not self._tdmpc2_available or self.model is None:
            print("  [TDMPC2Adapter] Cannot train: tdmpc2 not installed.")
            return None

        try:
            import tdmpc2
            # TD-MPC2 training API: train with step budget
            train_result: Dict[str, object] = self.model.train(
                task=self.tdmpc2_task_name,
                steps=steps,
                model_size=self.model_size,
            )
            print(f"  [TDMPC2Adapter] Training completed: {steps} steps "
                  f"on {self.tdmpc2_task_name}")
            return train_result

        except Exception as e:
            print(f"  [TDMPC2Adapter] Training failed: {e}")
            return None

    def evaluate(self, n_episodes: int = 5) -> Optional[Dict[str, object]]:
        """Evaluate the TD-MPC2 model on the configured task.

        Runs n_episodes evaluation episodes using TD-MPC2's evaluation API
        and returns aggregated metrics.

        Args:
            n_episodes: Number of evaluation episodes. Default 5 (matching
                        IDO evaluation protocol).

        Returns:
            Dict with evaluation metrics (avg_steps, avg_return, NVR, etc.)
            or None if tdmpc2 is not available.
        """
        if not self._tdmpc2_available or self.model is None:
            print("  [TDMPC2Adapter] Cannot evaluate: tdmpc2 not installed.")
            return None

        try:
            import tdmpc2
            # TD-MPC2 evaluation API: evaluate with SB3-equivalent protocol
            eval_result: Dict[str, object] = self.model.evaluate(
                task=self.tdmpc2_task_name,
                num_episodes=n_episodes,
            )
            print(f"  [TDMPC2Adapter] Evaluation completed: {n_episodes} episodes "
                  f"on {self.tdmpc2_task_name}")
            return eval_result

        except Exception as e:
            print(f"  [TDMPC2Adapter] Evaluation failed: {e}")
            return None

    def choose_action(self, obs: object) -> Optional[np.ndarray]:
        """Single-step decision interface (unified with IDO agent).

        Queries the TD-MPC2 model for an action given the current observation.
        This interface matches IDO's choose_action, enabling direct
        comparative evaluation in the same episode loop.

        Args:
            obs: Current observation (dm_control TimeStep or observation dict).

        Returns:
            Action array compatible with dm_control environment, or None
            if tdmpc2 is not available.
        """
        if not self._tdmpc2_available or self.model is None:
            return None

        try:
            # Extract observation features from dm_control timestep
            if hasattr(obs, 'observation'):
                obs_features = obs.observation
            else:
                obs_features = obs

            action: np.ndarray = self.model.act(obs_features)
            return action

        except Exception as e:
            print(f"  [TDMPC2Adapter] Action selection failed: {e}")
            # Fallback to random action (7-dim for humanoid, varies per task)
            return np.random.uniform(-1, 1, size=7)

    def reset(self) -> None:
        """Reset the TD-MPC2 model's internal state.

        Clears any cached internal state between episodes, ensuring
        clean evaluation conditions.
        """
        if self._tdmpc2_available and self.model is not None:
            try:
                self.model.reset()
            except AttributeError:
                # Some TD-MPC2 versions may not have reset()
                pass
            except Exception as e:
                print(f"  [TDMPC2Adapter] Reset warning: {e}")

    def is_available(self) -> bool:
        """Check whether the TD-MPC2 package is installed and model is loaded.

        Returns:
            True if tdmpc2 is available and model is initialized.
        """
        return self._tdmpc2_available and self.model is not None

    def get_info(self) -> Dict[str, object]:
        """Get adapter configuration information.

        Returns:
            Dict with adapter metadata: task, model_size, checkpoint, availability.
        """
        return {
            'adapter': 'TDMPC2Adapter',
            'task_name': self.task_name,
            'tdmpc2_task_name': self.tdmpc2_task_name,
            'model_size': self.model_size,
            'checkpoint_path': self.checkpoint_path,
            'available': self._tdmpc2_available,
            'model_loaded': self.model is not None,
        }


def make_tdmpc2_adapter(task_name: str = 'humanoid-stand',
                         model_size: int = 5,
                         checkpoint_path: Optional[str] = None) -> Optional[TDMPC2Adapter]:
    """Factory function for creating a TDMPC2Adapter instance.

    Creates an adapter with graceful degradation: if tdmpc2 is not installed,
    the adapter is created but will return None for all operations, allowing
    the benchmark to skip this baseline.

    Args:
        task_name: dm_control task name.
        model_size: Model size in million parameters (1/5/19/48/317).
        checkpoint_path: Optional path to pre-trained checkpoint.

    Returns:
        TDMPC2Adapter instance (always created, but may not be functional
        if tdmpc2 is not installed).
    """
    try:
        adapter: TDMPC2Adapter = TDMPC2Adapter(
            task_name=task_name,
            model_size=model_size,
            checkpoint_path=checkpoint_path,
        )
        return adapter
    except ValueError as e:
        print(f"  [TDMPC2Adapter] Configuration error: {e}")
        return None
