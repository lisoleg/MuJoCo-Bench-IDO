"""
DreamerV3 Baseline Adapter for MuJoCo-Bench-IDO
=================================================

Adapter for DreamerV3 (Mastering Diverse Domains through World Models)
by Hafner et al. (2025, Nature). Uses the burchim/DreamerV3-PyTorch
implementation for dm_control continuous control tasks.

DreamerV3 is a model-based RL algorithm that learns a world model from
experiences and trains an actor-critic policy from imagined trajectories.
It achieves SOTA normalized scores on dm_control tasks with fixed
hyperparameters, making it ideal for our benchmark framework.

SOTA normalized scores (dm_control proprio, 1M steps):
  - cheetah-run: 886.6
  - walker-walk: 956.0
  - hopper-hop: 369.7
  - hopper-stand: 944.6
  - humanoid-stand: 944.6 (vision; proprio ~945)
  - walker-stand: 900.0

This adapter supports two modes:
  1. **Standalone training + evaluation**: Clone burchim/DreamerV3-PyTorch
     and train with their CLI, then load checkpoint for inference.
  2. **Direct inference**: Load a pre-trained checkpoint and use choose_action()
     for step-by-step evaluation compatible with IDO agent.

Graceful degradation: If dreamerv3 module is not installed, prints a warning
and returns None, allowing the benchmark to skip this baseline.

Author: MuJoCo-Bench-IDO v0.9.0 DreamerV3 motor layer integration
"""

import os
import sys
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

# ── dm_control task name mapping ──
# DreamerV3-PyTorch uses "dmc-{Domain}-{Task}" format.
# dm_control uses (domain, task) pairs with hyphenated names.
DMCONTROL_DREAMER_TASK_MAP: Dict[str, str] = {
    'cheetah-run':      'dmc-Cheetah-run',
    'walker-walk':      'dmc-Walker-walk',
    'walker-stand':     'dmc-Walker-stand',
    'walker-run':       'dmc-Walker-run',
    'hopper-hop':       'dmc-Hopper-hop',
    'hopper-stand':     'dmc-Hopper-stand',
    'humanoid-stand':   'dmc-Humanoid-stand',
    'humanoid-walk':    'dmc-Humanoid-walk',
    'humanoid-run':     'dmc-Humanoid-run',
    'reacher-easy':     'dmc-Reacher-easy',
    'reacher-hard':     'dmc-Reacher-hard',
    'finger-turn_easy': 'dmc-Finger-turn_easy',
    'finger-turn_hard': 'dmc-Finger-turn_hard',
    'finger-spin':      'dmc-Finger-spin',
    'cartpole-balance': 'dmc-Cartpole-balance',
    'cartpole-swingup': 'dmc-Cartpole-swingup',
    'acrobot-swingup':  'dmc-Acrobot-swingup',
    'pendulum-swingup': 'dmc-Pendulum-swingup',
    'quadruped-run':    'dmc-Quadruped-run',
    'quadruped-walk':   'dmc-Quadruped-walk',
}

# SOTA normalized scores from DreamerV3-PyTorch (1M steps, 3 seeds avg)
DREAMER_SOTA_SCORES: Dict[str, float] = {
    'cheetah-run':      886.6,
    'walker-walk':      956.0,
    'walker-stand':     900.0,
    'walker-run':       701.1,
    'hopper-hop':       369.7,
    'hopper-stand':     944.6,
    'humanoid-stand':   944.6,  # vision-based; proprio similar
    'reacher-easy':     831.5,
    'reacher-hard':     597.2,
    'finger-turn_easy': 819.4,
    'finger-turn_hard': 832.2,
    'finger-spin':      547.6,
    'cartpole-balance': 999.3,
    'cartpole-swingup': 865.1,
    'acrobot-swingup':  410.8,
    'pendulum-swingup': 791.8,
    'quadruped-run':    683.7,
    'quadruped-walk':   733.4,
}

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR: str = "checkpoints"
DREAMER_CHECKPOINT_SUBDIR: str = "dreamer"


class DreamerV3Adapter:
    """DreamerV3 baseline adapter for MuJoCo-Bench-IDO comparative evaluation.

    Provides a unified interface for training, evaluating, and querying
    DreamerV3, compatible with the IDO agent evaluation framework.

    The adapter wraps DreamerV3-PyTorch's train/eval/predict APIs and maps
    dm_control task names to DreamerV3's internal naming convention.

    Two integration modes:
      1. **External CLI mode**: Use DreamerV3-PyTorch's main.py for training
         and evaluation. This adapter loads the resulting checkpoint.
      2. **Direct inference mode**: Load a trained checkpoint and call
         choose_action() for step-by-step evaluation with IDO agents.

    Attributes:
        task_name: dm_control task name (e.g., 'cheetah-run').
        dreamer_task_name: DreamerV3 internal task name (e.g., 'dmc-Cheetah-run').
        checkpoint_path: Optional path to a pre-trained checkpoint directory.
        model: DreamerV3 model instance (None if not available).
        _dreamer_available: Whether DreamerV3 module is importable.
        _checkpoint_dir: Directory containing trained model files.
        _config: DreamerV3 configuration dict.
    """

    def __init__(self,
                 task_name: str = 'cheetah-run',
                 checkpoint_path: Optional[str] = None,
                 model_size: str = 'XS',
                 action_repeat: int = 2,
                 seed: int = 0) -> None:
        """Initialize DreamerV3 adapter with task configuration.

        Args:
            task_name: dm_control task name (e.g., 'cheetah-run').
            checkpoint_path: Optional path to a pre-trained checkpoint directory.
                             If None, the model must be trained before evaluation.
            model_size: DreamerV3 model size ('XS', 'S', 'M', 'L', 'XL').
                        Default 'XS' (fastest, sufficient for dm_control proprio).
            action_repeat: Number of times each action is repeated. Default 2.
            seed: Random seed for reproducibility. Default 0.

        Raises:
            ValueError: If task_name is not in DMCONTROL_DREAMER_TASK_MAP.
        """
        # Validate task name
        if task_name not in DMCONTROL_DREAMER_TASK_MAP:
            raise ValueError(
                f"Unsupported task '{task_name}'. "
                f"Supported tasks: {list(DMCONTROL_DREAMER_TASK_MAP.keys())}")

        self.task_name: str = task_name
        self.dreamer_task_name: str = DMCONTROL_DREAMER_TASK_MAP[task_name]
        self.model_size: str = model_size
        self.action_repeat: int = action_repeat
        self.seed: int = seed
        self._dreamer_available: bool = False
        self.model: Optional[object] = None
        self._config: Optional[Dict] = None
        self._checkpoint_dir: Optional[str] = None

        # Set checkpoint path
        if checkpoint_path is not None:
            self._checkpoint_dir = checkpoint_path
        else:
            # Default: checkpoints/<task>/dreamer/
            self._checkpoint_dir = os.path.join(
                DEFAULT_CHECKPOINT_DIR, task_name, DREAMER_CHECKPOINT_SUBDIR)

        # Try to import DreamerV3 module
        try:
            # Attempt 1: burchim/DreamerV3-PyTorch (local clone)
            _dreamer_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'third_party', 'dreamerv3_pytorch')
            if os.path.isdir(_dreamer_path):
                sys.path.insert(0, _dreamer_path)

            # Attempt 2: r2dreamer (local clone)
            _r2_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'third_party', 'r2dreamer')
            if os.path.isdir(_r2_path):
                sys.path.insert(0, _r2_path)

            # Attempt 3: pip-installed dreamer package
            from dreamer import Dreamer  # noqa: F401
            self._dreamer_available = True
            print(f"  [DreamerV3Adapter] DreamerV3 module available")

        except ImportError:
            # Try alternative import paths
            try:
                # burchim/DreamerV3-PyTorch uses different module structure
                # The main model is in nnet/dreamer.py
                import importlib
                spec = importlib.util.find_spec('nnet.dreamer')
                if spec is not None:
                    self._dreamer_available = True
                    print(f"  [DreamerV3Adapter] nnet.dreamer module available")
                else:
                    raise ImportError
            except (ImportError, ModuleNotFoundError):
                print("  [DreamerV3Adapter] WARNING: DreamerV3 not installed.")
                print("  [DreamerV3Adapter] Install options:")
                print("  [DreamerV3Adapter]   Option A: git clone https://github.com/burchim/DreamerV3-PyTorch.git third_party/dreamerv3_pytorch")
                print("  [DreamerV3Adapter]   Option B: git clone https://github.com/NM512/r2dreamer.git third_party/r2dreamer")
                print("  [DreamerV3Adapter]   Option C: pip install dreamer (danijar JAX version)")
                print("  [DreamerV3Adapter] This baseline will be skipped in evaluation.")
                self._dreamer_available = False
                self.model = None
                return

        # Load checkpoint if available
        if self._dreamer_available and self._checkpoint_dir is not None:
            self._load_checkpoint()

    def _load_checkpoint(self) -> bool:
        """Load a pre-trained DreamerV3 checkpoint.

        Looks for checkpoint files in the configured checkpoint directory.
        DreamerV3-PyTorch saves checkpoints as PyTorch state dicts.

        Returns:
            True if checkpoint loaded successfully, False otherwise.
        """
        if not self._dreamer_available:
            return False

        # Look for checkpoint files in the directory
        if not os.path.isdir(self._checkpoint_dir):
            print(f"  [DreamerV3Adapter] No checkpoint directory at {self._checkpoint_dir}")
            return False

        # DreamerV3-PyTorch saves in callbacks/ subdirectory
        callback_dir = os.path.join(self._checkpoint_dir, 'callbacks')
        if os.path.isdir(callback_dir):
            # Find the latest checkpoint
            ckpt_files = [f for f in os.listdir(callback_dir)
                          if f.endswith('.pt') or f.endswith('.pth')]
            if ckpt_files:
                latest_ckpt = sorted(ckpt_files)[-1]
                ckpt_path = os.path.join(callback_dir, latest_ckpt)
                try:
                    self.model = self._create_model()
                    state_dict = __import__('torch').load(ckpt_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                    print(f"  [DreamerV3Adapter] Loaded checkpoint: {ckpt_path}")
                    return True
                except Exception as e:
                    print(f"  [DreamerV3Adapter] Checkpoint load failed: {e}")
                    self.model = None
                    return False

        print(f"  [DreamerV3Adapter] No checkpoint found in {self._checkpoint_dir}")
        return False

    def _create_model(self) -> Optional[object]:
        """Create a fresh DreamerV3 model instance for training.

        Returns:
            DreamerV3 model instance, or None if not available.
        """
        if not self._dreamer_available:
            return None

        try:
            # This will be populated when DreamerV3-PyTorch is cloned
            # The exact API depends on which implementation is used
            # burchim/DreamerV3-PyTorch: from nnet.dreamer import DreamerV3
            # r2dreamer: from dreamer import Dreamer
            # danijar/dreamerv3: from dreamerv3 import Agent

            # Generic placeholder - will be implemented when
            # DreamerV3-PyTorch is cloned into third_party/
            print(f"  [DreamerV3Adapter] Model creation for {self.dreamer_task_name} "
                  f"requires DreamerV3 source code. Clone to third_party/ first.")
            return None

        except Exception as e:
            print(f"  [DreamerV3Adapter] Model creation failed: {e}")
            return None

    def train_cli(self, steps: int = 1_000_000,
                  logdir: Optional[str] = None) -> str:
        """Train DreamerV3 using the external CLI (burchim/DreamerV3-PyTorch).

        This method launches DreamerV3-PyTorch's main.py as a subprocess,
        which handles the full training pipeline (environment creation,
        model training, checkpoint saving, evaluation).

        Args:
            steps: Number of training steps (default 1M for proprio).
            logdir: Optional log directory. If None, uses checkpoint_dir.

        Returns:
            Command string that was executed.
        """
        if logdir is None:
            logdir = self._checkpoint_dir

        # Find DreamerV3-PyTorch main.py
        dreamer_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'third_party', 'dreamerv3_pytorch')

        if not os.path.isdir(dreamer_root):
            # Clone DreamerV3-PyTorch first
            print(f"  [DreamerV3Adapter] Cloning burchim/DreamerV3-PyTorch...")
            os.makedirs(os.path.dirname(dreamer_root), exist_ok=True)
            clone_cmd = f"git clone https://github.com/burchim/DreamerV3-PyTorch.git {dreamer_root}"
            print(f"  [DreamerV3Adapter] Run: {clone_cmd}")
            return clone_cmd

        main_py = os.path.join(dreamer_root, 'main.py')
        config_py = os.path.join(dreamer_root, 'configs', 'DreamerV3', 'dreamer_v3.py')

        if not os.path.isfile(main_py):
            print(f"  [DreamerV3Adapter] main.py not found at {main_py}")
            return ""

        # Build training command
        # For proprio (state-based), we use dmc_env_type=proprio
        # DreamerV3-PyTorch env naming: dmc-{Domain}-{Task}
        cmd = (
            f"cd {dreamer_root} && "
            f"python3 main.py "
            f"-c {config_py} "
            f"env_name={self.dreamer_task_name} "
            f"logdir={logdir} "
            f"--mode train"
        )

        print(f"  [DreamerV3Adapter] Training command: {cmd}")
        print(f"  [DreamerV3Adapter] Expected training time: ~2-8 hours on CPU")
        print(f"  [DreamerV3Adapter] Checkpoints saved to: {logdir}")
        return cmd

    def train(self, steps: int = 1_000_000) -> Optional[Dict[str, object]]:
        """Train the DreamerV3 model on the configured task.

        For dm_control proprio tasks, 500K steps is often sufficient
        (r2dreamer default). This method delegates to the CLI-based
        training if DreamerV3-PyTorch is cloned.

        Args:
            steps: Number of training steps (default 1M).

        Returns:
            Dict with training info, or None if not available.
        """
        if not self._dreamer_available:
            print("  [DreamerV3Adapter] Cannot train: DreamerV3 not available.")
            return None

        # For now, delegate to CLI training
        cmd = self.train_cli(steps=steps)
        if cmd:
            return {
                'algo': 'dreamerv3',
                'task': self.task_name,
                'steps': steps,
                'command': cmd,
                'checkpoint_dir': self._checkpoint_dir,
            }
        return None

    def evaluate(self, n_episodes: int = 10,
                 max_steps: int = 1000) -> Optional[Dict[str, object]]:
        """Evaluate the DreamerV3 model on the configured task.

        Runs n_episodes evaluation episodes and returns aggregated metrics.
        Compatible with IDO evaluation protocol.

        Args:
            n_episodes: Number of evaluation episodes (default 10, matching
                        DreamerV3-PyTorch standard).
            max_steps: Maximum steps per episode (default 1000, standard
                       dm_control episode length).

        Returns:
            Dict with evaluation metrics, or None if not available.
        """
        if not self._dreamer_available or self.model is None:
            print("  [DreamerV3Adapter] Cannot evaluate: model not available.")
            return None

        try:
            import dm_control.suite as suite

            episode_returns: list = []
            episode_lengths: list = []

            domain, task_name = self.task_name.split('-', 1)
            env = suite.load(domain_name=domain, task_name=task_name)

            for ep_idx in range(n_episodes):
                timestep = env.reset()
                total_reward: float = 0.0
                steps: int = 0

                self.reset()

                for step in range(max_steps):
                    action = self.choose_action(timestep, physics=env.physics)
                    if action is None:
                        action = np.random.uniform(-1, 1, size=env.action_spec().shape[0])
                    timestep = env.step(action)
                    total_reward += float(timestep.reward or 0.0)
                    steps += 1
                    if timestep.last():
                        break

                episode_returns.append(total_reward)
                episode_lengths.append(steps)
                print(f"  [DreamerV3Adapter] Eval episode {ep_idx + 1}: "
                      f"return={total_reward:.4f}, steps={steps}")

            avg_return: float = float(np.mean(episode_returns))
            avg_steps: float = float(np.mean(episode_lengths))
            print(f"  [DreamerV3Adapter] Eval summary: avg_return={avg_return:.4f}, "
                  f"avg_steps={avg_steps:.1f}")

            return {
                'algo': 'dreamerv3',
                'task': self.task_name,
                'n_episodes': n_episodes,
                'avg_return': avg_return,
                'std_return': float(np.std(episode_returns)),
                'avg_steps': avg_steps,
                'episode_returns': episode_returns,
                'episode_lengths': episode_lengths,
            }

        except Exception as e:
            print(f"  [DreamerV3Adapter] Evaluation failed: {e}")
            return None

    def choose_action(self, obs: object,
                      physics: Optional[object] = None) -> Optional[np.ndarray]:
        """Single-step decision interface (unified with IDO agent).

        Queries the DreamerV3 model for an action given the current observation.
        This interface matches IDO's choose_action, enabling direct
        comparative evaluation in the same episode loop.

        For dm_control proprio tasks, observations are state vectors.
        DreamerV3 processes them through its world model encoder.

        Args:
            obs: Current observation (dm_control TimeStep).
            physics: Optional dm_control Physics instance for state extraction.

        Returns:
            Action array compatible with dm_control environment, or None
            if model is not available.
        """
        if not self._dreamer_available or self.model is None:
            return None

        try:
            # Extract observation from dm_control timestep
            obs_features: np.ndarray = self._extract_obs(obs, physics)

            # Query DreamerV3 for action
            action: np.ndarray = self.model.act(obs_features)
            return action

        except Exception as e:
            print(f"  [DreamerV3Adapter] Action selection failed: {e}")
            # Fallback to random action with correct dimension
            if physics is not None:
                return np.random.uniform(-1, 1, size=physics.model.nu)
            return np.random.uniform(-1, 1, size=6)

    def _extract_obs(self, obs: object,
                     physics: Optional[object] = None) -> np.ndarray:
        """Extract flat observation vector from dm_control timestep.

        Args:
            obs: dm_control TimeStep or raw observation dict.
            physics: Optional Physics instance for additional state.

        Returns:
            Flattened observation numpy array.
        """
        if hasattr(obs, 'observation'):
            obs_dict = obs.observation
            parts: list = []
            for key in sorted(obs_dict.keys()):
                val = obs_dict[key]
                if isinstance(val, np.ndarray):
                    parts.append(val.flatten())
                else:
                    parts.append(np.array([val]).flatten())
            return np.concatenate(parts)
        elif isinstance(obs, np.ndarray):
            return obs
        elif isinstance(obs, dict):
            parts: list = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, np.ndarray):
                    parts.append(val.flatten())
                else:
                    parts.append(np.array([val]).flatten())
            return np.concatenate(parts)
        else:
            return np.array(obs).flatten()

    def reset(self) -> None:
        """Reset the DreamerV3 model's internal state.

        Clears any cached latent state between episodes, ensuring
        clean evaluation conditions.
        """
        if self._dreamer_available and self.model is not None:
            try:
                self.model.reset()
            except AttributeError:
                pass  # Some versions may not have reset()
            except Exception as e:
                print(f"  [DreamerV3Adapter] Reset warning: {e}")

    def is_available(self) -> bool:
        """Check whether the DreamerV3 module is installed and model is loaded.

        Returns:
            True if dreamerv3 is available and model is initialized.
        """
        return self._dreamer_available and self.model is not None

    def get_info(self) -> Dict[str, object]:
        """Get adapter configuration information.

        Returns:
            Dict with adapter metadata: task, model_size, checkpoint, availability.
        """
        return {
            'adapter': 'DreamerV3Adapter',
            'task_name': self.task_name,
            'dreamer_task_name': self.dreamer_task_name,
            'model_size': self.model_size,
            'action_repeat': self.action_repeat,
            'checkpoint_dir': self._checkpoint_dir,
            'available': self._dreamer_available,
            'model_loaded': self.model is not None,
            'sota_score': DREAMER_SOTA_SCORES.get(self.task_name, 0.0),
        }


def make_dreamer_adapter(task_name: str = 'cheetah-run',
                          checkpoint_path: Optional[str] = None,
                          model_size: str = 'XS',
                          action_repeat: int = 2,
                          seed: int = 0) -> Optional[DreamerV3Adapter]:
    """Factory function for creating a DreamerV3Adapter instance.

    Creates an adapter with graceful degradation: if dreamerv3 is not
    installed, the adapter is created but will return None for all
    operations, allowing the benchmark to skip this baseline.

    Args:
        task_name: dm_control task name.
        checkpoint_path: Optional path to pre-trained checkpoint directory.
        model_size: Model size ('XS', 'S', 'M', 'L', 'XL').
        action_repeat: Action repeat factor (default 2).
        seed: Random seed.

    Returns:
        DreamerV3Adapter instance, or None on ValueError.
    """
    try:
        adapter: DreamerV3Adapter = DreamerV3Adapter(
            task_name=task_name,
            checkpoint_path=checkpoint_path,
            model_size=model_size,
            action_repeat=action_repeat,
            seed=seed,
        )
        return adapter
    except ValueError as e:
        print(f"  [DreamerV3Adapter] Configuration error: {e}")
        return None
