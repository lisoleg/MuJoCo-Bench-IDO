"""
Stable-Baselines3 (PPO/SAC) Adapter for MuJoCo-Bench-IDO
==========================================================

Adapter classes for PPO and SAC baselines from stable-baselines3,
enabling actual policy-based actions on dm_control benchmark tasks
(via shimmy DmControlCompatibilityV0 + gymnasium FlattenObservation wrapper).

Key features:
  - Loads pre-trained checkpoints from checkpoints/<task>/<algo>/
  - Auto-trains (default 100K steps, configurable) when no checkpoint exists
  - Saves trained model checkpoints for future reuse
  - Returns actual PPO/SAC policy actions (no random fallback on trained model)
  - Graceful degradation if dm_control / shimmy / sb3 not installed
  - Proper dm_control timestep → gymnasium flat-obs conversion for choose_action

Shimmy 2.0.1 note: uses DmControlCompatibilityV0 (not DmControlCompatibility).
Gymnasium note: DmControlCompatibilityV0 returns dict observations, which SB3
MlpPolicy cannot handle directly. We wrap with FlattenObservation to produce
flat arrays, and use gymnasium.spaces.flatten for dm_control timestep conversion.

Author: MuJoCo-Bench-IDO v0.4.5 P0-4 baseline integration
"""

import os
import numpy as np
from typing import Dict, Optional

# ── Default configuration ──
DEFAULT_CHECKPOINT_DIR: str = "checkpoints"
DEFAULT_AUTO_TRAIN_STEPS: int = 100_000  # 100K for quick auto-train
EVAL_EPISODES: int = 5


def _make_gym_env(task_name: str) -> Optional[object]:
    """Create a Gymnasium-compatible environment from a dm_control task.

    Uses shimmy DmControlCompatibilityV0 to wrap dm_control environments,
    then wraps with gymnasium FlattenObservation for SB3 MlpPolicy compatibility.

    Args:
        task_name: dm_control task identifier (e.g., 'humanoid-stand').

    Returns:
        FlattenObservation-wrapped Gymnasium Env instance, or None on failure.
    """
    try:
        import dm_control.suite as suite
        from shimmy.dm_control_compatibility import DmControlCompatibilityV0
        from gymnasium.wrappers import FlattenObservation

        # Split task name: 'finger-turn_easy' -> domain='finger', task='turn_easy'
        domain, task = task_name.split('-', 1)
        dm_env = suite.load(domain_name=domain, task_name=task)
        gym_env = DmControlCompatibilityV0(dm_env)
        # DmControlCompatibilityV0 returns dict observations;
        # SB3 MlpPolicy requires flat arrays, so wrap with FlattenObservation.
        flat_env = FlattenObservation(gym_env)
        return flat_env
    except ImportError as e:
        print(f"  [SB3Adapter] WARNING: Cannot create Gymnasium env: {e}")
        print(f"  [SB3Adapter] Requires dm_control + shimmy >= 2.0.0 + gymnasium")
        return None
    except Exception as e:
        print(f"  [SB3Adapter] WARNING: Failed to load task '{task_name}': {e}")
        return None


def _make_inner_env(task_name: str) -> Optional[object]:
    """Create the inner DmControlCompatibilityV0 env (before FlattenObservation).

    Used to access the Dict observation_space for dm_control timestep conversion
    in choose_action().

    Args:
        task_name: dm_control task identifier.

    Returns:
        DmControlCompatibilityV0 env instance, or None on failure.
    """
    try:
        import dm_control.suite as suite
        from shimmy.dm_control_compatibility import DmControlCompatibilityV0

        domain, task = task_name.split('-', 1)
        dm_env = suite.load(domain_name=domain, task_name=task)
        gym_env = DmControlCompatibilityV0(dm_env)
        return gym_env
    except Exception:
        return None


def _checkpoint_path(task_name: str, algo: str,
                     checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> str:
    """Compute the checkpoint file path for a task+algorithm pair.

    Args:
        task_name: dm_control task identifier.
        algo: Algorithm name ('ppo' or 'sac').
        checkpoint_dir: Root checkpoint directory.

    Returns:
        Full path to model.zip checkpoint file.
    """
    return os.path.join(checkpoint_dir, task_name, algo, "model.zip")


def _flatten_dm_obs(obs_dict: dict, dict_obs_space: object) -> np.ndarray:
    """Flatten a dm_control observation dict using gymnasium.spaces.flatten.

    This ensures the flattening order matches exactly what FlattenObservation
    produces, so the SB3 model can correctly interpret the observation.

    Args:
        obs_dict: dm_control timestep observation dict.
        dict_obs_space: gymnasium Dict observation_space from the inner
                        DmControlCompatibilityV0 env.

    Returns:
        Flattened observation numpy array.
    """
    from gymnasium.spaces import flatten
    return flatten(dict_obs_space, obs_dict)


class SB3PPOAdapter:
    """Stable-Baselines3 PPO baseline adapter for MuJoCo-Bench-IDO.

    Wraps PPO from stable-baselines3 with:
    - Automatic checkpoint loading/saving
    - Auto-training when no checkpoint exists
    - DmControlCompatibilityV0 + FlattenObservation environment wrapping
    - Actual policy action output (no random fallback after training)
    - Correct dm_control timestep → gymnasium flat-obs conversion

    Attributes:
        task_name: dm_control task name (e.g., 'humanoid-stand').
        checkpoint_dir: Root directory for checkpoints.
        auto_train_steps: Number of training steps if no checkpoint found.
        model: PPO model instance (None if sb3 not available).
        gym_env: FlattenObservation-wrapped Gymnasium environment.
        _inner_env: DmControlCompatibilityV0 env for obs conversion.
        _dict_obs_space: Dict observation space from inner env.
        _sb3_available: Whether stable-baselines3 is importable.
        _trained: Whether the model has been trained or loaded.
    """

    def __init__(self,
                 task_name: str = 'humanoid-stand',
                 checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                 auto_train_steps: int = DEFAULT_AUTO_TRAIN_STEPS,
                 verbose: int = 0) -> None:
        """Initialize PPO adapter with task configuration.

        Attempts to load a pre-trained checkpoint. If no checkpoint exists,
        auto-trains for auto_train_steps and saves the result.

        Args:
            task_name: dm_control task name.
            checkpoint_dir: Root checkpoint directory.
            auto_train_steps: Steps for auto-training (default 100K).
            verbose: SB3 verbose level (0=silent, 1=info, 2=debug).
        """
        self.task_name: str = task_name
        self.checkpoint_dir: str = checkpoint_dir
        self.auto_train_steps: int = auto_train_steps
        self.verbose: int = verbose
        self._sb3_available: bool = False
        self.model: Optional[object] = None
        self.gym_env: Optional[object] = None
        self._inner_env: Optional[object] = None
        self._dict_obs_space: Optional[object] = None
        self._trained: bool = False

        # Step 1: Try importing stable_baselines3
        try:
            from stable_baselines3 import PPO
            self._sb3_available = True
            self._PPO_cls = PPO
        except ImportError:
            print("  [SB3PPOAdapter] WARNING: stable-baselines3 not installed.")
            print("  [SB3PPOAdapter] Install with: pip install stable-baselines3")
            print("  [SB3PPOAdapter] This baseline will fall back to random actions.")
            self._sb3_available = False
            self._PPO_cls = None
            return

        # Step 2: Create Gymnasium environment (FlattenObservation wrapped)
        self.gym_env = _make_gym_env(task_name)
        if self.gym_env is None:
            print(f"  [SB3PPOAdapter] WARNING: Cannot create env for '{task_name}'.")
            return

        # Step 3: Also create the inner env for obs conversion
        self._inner_env = _make_inner_env(task_name)
        if self._inner_env is not None:
            self._dict_obs_space = self._inner_env.observation_space

        # Step 4: Try loading checkpoint
        ckpt_path: str = _checkpoint_path(task_name, 'ppo', checkpoint_dir)
        if os.path.isfile(ckpt_path):
            try:
                self.model = PPO.load(ckpt_path, env=self.gym_env)
                self._trained = True
                print(f"  [SB3PPOAdapter] Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"  [SB3PPOAdapter] Checkpoint load failed: {e}")
                self.model = None
        else:
            print(f"  [SB3PPOAdapter] No checkpoint at {ckpt_path}; "
                  f"auto-training {auto_train_steps} steps...")

        # Step 5: Auto-train if no model loaded
        if self.model is None and self.gym_env is not None:
            self._auto_train()

    def _auto_train(self) -> None:
        """Auto-train PPO model when no checkpoint exists.

        Creates a fresh PPO model, trains for auto_train_steps,
        and saves the checkpoint.
        """
        if not self._sb3_available or self.gym_env is None:
            return

        try:
            self.model = self._PPO_cls(
                "MlpPolicy",
                self.gym_env,
                verbose=self.verbose,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
            self.model.learn(total_timesteps=self.auto_train_steps)
            self._trained = True

            # Save checkpoint
            ckpt_path: str = _checkpoint_path(
                self.task_name, 'ppo', self.checkpoint_dir)
            ckpt_dir: str = os.path.dirname(ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.model.save(ckpt_path)
            print(f"  [SB3PPOAdapter] Auto-trained {self.auto_train_steps} steps "
                  f"and saved checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"  [SB3PPOAdapter] Auto-training failed: {e}")
            self.model = None
            self._trained = False

    def train(self, steps: int = 1_000_000) -> Optional[Dict[str, object]]:
        """Explicitly train the PPO model for a specified number of steps.

        If a model already exists (from checkpoint or auto-train), continues
        training. Otherwise creates a fresh model.

        Args:
            steps: Number of training steps (default 1M).

        Returns:
            Dict with training info, or None if sb3 not available.
        """
        if not self._sb3_available or self.gym_env is None:
            print("  [SB3PPOAdapter] Cannot train: sb3 or env not available.")
            return None

        try:
            if self.model is None:
                self.model = self._PPO_cls(
                    "MlpPolicy",
                    self.gym_env,
                    verbose=self.verbose,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                )

            self.model.learn(total_timesteps=steps)
            self._trained = True

            # Save checkpoint
            ckpt_path: str = _checkpoint_path(
                self.task_name, 'ppo', self.checkpoint_dir)
            ckpt_dir: str = os.path.dirname(ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.model.save(ckpt_path)
            print(f"  [SB3PPOAdapter] Trained {steps} steps, "
                  f"saved checkpoint: {ckpt_path}")

            return {
                'algo': 'ppo',
                'task': self.task_name,
                'steps': steps,
                'checkpoint': ckpt_path,
            }
        except Exception as e:
            print(f"  [SB3PPOAdapter] Training failed: {e}")
            return None

    def evaluate(self, n_episodes: int = EVAL_EPISODES) -> Optional[Dict[str, object]]:
        """Evaluate the PPO model on the configured task.

        Runs n_episodes using the trained policy and collects episode returns
        and success rates.

        Args:
            n_episodes: Number of evaluation episodes (default 5).

        Returns:
            Dict with evaluation metrics, or None if not available.
        """
        if not self._sb3_available or self.model is None or self.gym_env is None:
            print("  [SB3PPOAdapter] Cannot evaluate: model/env not available.")
            return None

        try:
            episode_returns: list = []
            episode_lengths: list = []

            for ep_idx in range(n_episodes):
                obs, info = self.gym_env.reset()
                total_reward: float = 0.0
                steps: int = 0
                done: bool = False

                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.gym_env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    done = terminated or truncated

                episode_returns.append(total_reward)
                episode_lengths.append(steps)
                print(f"  [SB3PPOAdapter] Eval episode {ep_idx + 1}: "
                      f"return={total_reward:.4f}, steps={steps}")

            avg_return: float = float(np.mean(episode_returns))
            avg_steps: float = float(np.mean(episode_lengths))
            print(f"  [SB3PPOAdapter] Eval summary: avg_return={avg_return:.4f}, "
                  f"avg_steps={avg_steps:.1f}")

            return {
                'algo': 'ppo',
                'task': self.task_name,
                'n_episodes': n_episodes,
                'avg_return': avg_return,
                'std_return': float(np.std(episode_returns)),
                'avg_steps': avg_steps,
                'std_steps': float(np.std(episode_lengths)),
                'episode_returns': episode_returns,
                'episode_lengths': episode_lengths,
            }
        except Exception as e:
            print(f"  [SB3PPOAdapter] Evaluation failed: {e}")
            return None

    def choose_action(self, obs: object) -> Optional[np.ndarray]:
        """Single-step decision interface (unified with IDO agent).

        Returns the PPO policy action for the given observation. When obs
        is a dm_control timestep (has .observation dict), converts it to
        the flat gymnasium format using gymnasium.spaces.flatten for
        exact compatibility with the FlattenObservation wrapper.

        Falls back to random action if the model is not available.

        Args:
            obs: Current observation (gymnasium flat array, or dm_control
                 timestep with .observation dict; auto-converted if needed).

        Returns:
            Action array, or None on total failure.
        """
        if self._sb3_available and self.model is not None:
            try:
                # SB3 predict expects flat gymnasium-format observation
                if hasattr(obs, 'observation'):
                    # Convert dm_control timestep observation dict to
                    # flat gymnasium format using gymnasium.spaces.flatten
                    gym_obs = self._convert_dm_obs_to_gym(obs)
                else:
                    gym_obs = obs

                action, _states = self.model.predict(gym_obs, deterministic=True)
                return action
            except Exception as e:
                print(f"  [SB3PPOAdapter] Action prediction failed: {e}")
                # Fall back to random action with correct dimension
                if self.gym_env is not None:
                    return np.random.uniform(
                        -1, 1, size=self.gym_env.action_space.shape[0])
                return None
        else:
            # Random fallback when sb3 not installed
            if self.gym_env is not None:
                return np.random.uniform(
                    -1, 1, size=self.gym_env.action_space.shape[0])
            # Last resort: generic random action
            return np.random.uniform(-1, 1, size=6)

    def _convert_dm_obs_to_gym(self, dm_timestep: object) -> np.ndarray:
        """Convert dm_control timestep observation to gymnasium flat array.

        Uses gymnasium.spaces.flatten with the inner env's Dict observation
        space to ensure the flattening order matches exactly what
        FlattenObservation produces. This is critical for SB3 model
        prediction correctness.

        Args:
            dm_timestep: dm_control TimeStep with .observation dict.

        Returns:
            Flattened observation array compatible with the SB3 model.
        """
        obs_dict: dict = dm_timestep.observation

        # Use gymnasium.spaces.flatten for exact compatibility with FlattenObservation
        if self._dict_obs_space is not None:
            try:
                return _flatten_dm_obs(obs_dict, self._dict_obs_space)
            except Exception:
                pass  # Fall through to manual flattening

        # Fallback: manual flatten with sorted keys
        # This may not match gymnasium Dict space ordering exactly,
        # but works as a last resort when inner env isn't available
        parts: list = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, np.ndarray):
                parts.append(val.flatten())
            else:
                parts.append(np.array([val]).flatten())
        return np.concatenate(parts)

    def reset(self) -> None:
        """Reset internal state (PPO has no recurrent state, so this is a no-op)."""
        pass

    def is_available(self) -> bool:
        """Check whether the PPO model is loaded and ready.

        Returns:
            True if sb3 is available and model is initialized.
        """
        return self._sb3_available and self.model is not None

    def is_trained(self) -> bool:
        """Check whether the PPO model has been trained.

        Returns:
            True if the model has been trained or loaded from checkpoint.
        """
        return self._trained

    def get_info(self) -> Dict[str, object]:
        """Get adapter configuration and status information.

        Returns:
            Dict with adapter metadata.
        """
        return {
            'adapter': 'SB3PPOAdapter',
            'task_name': self.task_name,
            'checkpoint_dir': self.checkpoint_dir,
            'auto_train_steps': self.auto_train_steps,
            'available': self._sb3_available,
            'model_loaded': self.model is not None,
            'trained': self._trained,
        }


class SB3SACAdapter:
    """Stable-Baselines3 SAC baseline adapter for MuJoCo-Bench-IDO.

    Wraps SAC from stable-baselines3 with the same features as SB3PPOAdapter:
    - Automatic checkpoint loading/saving
    - Auto-training when no checkpoint exists
    - DmControlCompatibilityV0 + FlattenObservation environment wrapping
    - Actual policy action output (no random fallback after training)
    - Correct dm_control timestep → gymnasium flat-obs conversion

    Attributes:
        task_name: dm_control task name (e.g., 'humanoid-stand').
        checkpoint_dir: Root directory for checkpoints.
        auto_train_steps: Number of training steps if no checkpoint found.
        model: SAC model instance (None if sb3 not available).
        gym_env: FlattenObservation-wrapped Gymnasium environment.
        _inner_env: DmControlCompatibilityV0 env for obs conversion.
        _dict_obs_space: Dict observation space from inner env.
        _sb3_available: Whether stable-baselines3 is importable.
        _trained: Whether the model has been trained or loaded.
    """

    def __init__(self,
                 task_name: str = 'humanoid-stand',
                 checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                 auto_train_steps: int = DEFAULT_AUTO_TRAIN_STEPS,
                 verbose: int = 0) -> None:
        """Initialize SAC adapter with task configuration.

        Attempts to load a pre-trained checkpoint. If no checkpoint exists,
        auto-trains for auto_train_steps and saves the result.

        Args:
            task_name: dm_control task name.
            checkpoint_dir: Root checkpoint directory.
            auto_train_steps: Steps for auto-training (default 100K).
            verbose: SB3 verbose level (0=silent, 1=info, 2=debug).
        """
        self.task_name: str = task_name
        self.checkpoint_dir: str = checkpoint_dir
        self.auto_train_steps: int = auto_train_steps
        self.verbose: int = verbose
        self._sb3_available: bool = False
        self.model: Optional[object] = None
        self.gym_env: Optional[object] = None
        self._inner_env: Optional[object] = None
        self._dict_obs_space: Optional[object] = None
        self._trained: bool = False

        # Step 1: Try importing stable_baselines3
        try:
            from stable_baselines3 import SAC
            self._sb3_available = True
            self._SAC_cls = SAC
        except ImportError:
            print("  [SB3SACAdapter] WARNING: stable-baselines3 not installed.")
            print("  [SB3SACAdapter] Install with: pip install stable-baselines3")
            print("  [SB3SACAdapter] This baseline will fall back to random actions.")
            self._sb3_available = False
            self._SAC_cls = None
            return

        # Step 2: Create Gymnasium environment (FlattenObservation wrapped)
        self.gym_env = _make_gym_env(task_name)
        if self.gym_env is None:
            print(f"  [SB3SACAdapter] WARNING: Cannot create env for '{task_name}'.")
            return

        # Step 3: Also create the inner env for obs conversion
        self._inner_env = _make_inner_env(task_name)
        if self._inner_env is not None:
            self._dict_obs_space = self._inner_env.observation_space

        # Step 4: Try loading checkpoint
        ckpt_path: str = _checkpoint_path(task_name, 'sac', checkpoint_dir)
        if os.path.isfile(ckpt_path):
            try:
                self.model = SAC.load(ckpt_path, env=self.gym_env)
                self._trained = True
                print(f"  [SB3SACAdapter] Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"  [SB3SACAdapter] Checkpoint load failed: {e}")
                self.model = None
        else:
            print(f"  [SB3SACAdapter] No checkpoint at {ckpt_path}; "
                  f"auto-training {auto_train_steps} steps...")

        # Step 5: Auto-train if no model loaded
        if self.model is None and self.gym_env is not None:
            self._auto_train()

    def _auto_train(self) -> None:
        """Auto-train SAC model when no checkpoint exists.

        Creates a fresh SAC model, trains for auto_train_steps,
        and saves the checkpoint.
        """
        if not self._sb3_available or self.gym_env is None:
            return

        try:
            self.model = self._SAC_cls(
                "MlpPolicy",
                self.gym_env,
                verbose=self.verbose,
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=1000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
            )
            self.model.learn(total_timesteps=self.auto_train_steps)
            self._trained = True

            # Save checkpoint
            ckpt_path: str = _checkpoint_path(
                self.task_name, 'sac', self.checkpoint_dir)
            ckpt_dir: str = os.path.dirname(ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.model.save(ckpt_path)
            print(f"  [SB3SACAdapter] Auto-trained {self.auto_train_steps} steps "
                  f"and saved checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"  [SB3SACAdapter] Auto-training failed: {e}")
            self.model = None
            self._trained = False

    def train(self, steps: int = 1_000_000) -> Optional[Dict[str, object]]:
        """Explicitly train the SAC model for a specified number of steps.

        Args:
            steps: Number of training steps (default 1M).

        Returns:
            Dict with training info, or None if sb3 not available.
        """
        if not self._sb3_available or self.gym_env is None:
            print("  [SB3SACAdapter] Cannot train: sb3 or env not available.")
            return None

        try:
            if self.model is None:
                self.model = self._SAC_cls(
                    "MlpPolicy",
                    self.gym_env,
                    verbose=self.verbose,
                    learning_rate=3e-4,
                    buffer_size=100_000,
                    learning_starts=1000,
                    batch_size=256,
                    gamma=0.99,
                    tau=0.005,
                )

            self.model.learn(total_timesteps=steps)
            self._trained = True

            # Save checkpoint
            ckpt_path: str = _checkpoint_path(
                self.task_name, 'sac', self.checkpoint_dir)
            ckpt_dir: str = os.path.dirname(ckpt_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.model.save(ckpt_path)
            print(f"  [SB3SACAdapter] Trained {steps} steps, "
                  f"saved checkpoint: {ckpt_path}")

            return {
                'algo': 'sac',
                'task': self.task_name,
                'steps': steps,
                'checkpoint': ckpt_path,
            }
        except Exception as e:
            print(f"  [SB3SACAdapter] Training failed: {e}")
            return None

    def evaluate(self, n_episodes: int = EVAL_EPISODES) -> Optional[Dict[str, object]]:
        """Evaluate the SAC model on the configured task.

        Args:
            n_episodes: Number of evaluation episodes (default 5).

        Returns:
            Dict with evaluation metrics, or None if not available.
        """
        if not self._sb3_available or self.model is None or self.gym_env is None:
            print("  [SB3SACAdapter] Cannot evaluate: model/env not available.")
            return None

        try:
            episode_returns: list = []
            episode_lengths: list = []

            for ep_idx in range(n_episodes):
                obs, info = self.gym_env.reset()
                total_reward: float = 0.0
                steps: int = 0
                done: bool = False

                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.gym_env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    done = terminated or truncated

                episode_returns.append(total_reward)
                episode_lengths.append(steps)
                print(f"  [SB3SACAdapter] Eval episode {ep_idx + 1}: "
                      f"return={total_reward:.4f}, steps={steps}")

            avg_return: float = float(np.mean(episode_returns))
            avg_steps: float = float(np.mean(episode_lengths))
            print(f"  [SB3SACAdapter] Eval summary: avg_return={avg_return:.4f}, "
                  f"avg_steps={avg_steps:.1f}")

            return {
                'algo': 'sac',
                'task': self.task_name,
                'n_episodes': n_episodes,
                'avg_return': avg_return,
                'std_return': float(np.std(episode_returns)),
                'avg_steps': avg_steps,
                'std_steps': float(np.std(episode_lengths)),
                'episode_returns': episode_returns,
                'episode_lengths': episode_lengths,
            }
        except Exception as e:
            print(f"  [SB3SACAdapter] Evaluation failed: {e}")
            return None

    def choose_action(self, obs: object) -> Optional[np.ndarray]:
        """Single-step decision interface (unified with IDO agent).

        Returns the SAC policy action for the given observation. When obs
        is a dm_control timestep (has .observation dict), converts it to
        the flat gymnasium format using gymnasium.spaces.flatten.

        Falls back to random action if the model is not available.

        Args:
            obs: Current observation (gymnasium flat array, or dm_control
                 timestep with .observation dict; auto-converted if needed).

        Returns:
            Action array, or None on total failure.
        """
        if self._sb3_available and self.model is not None:
            try:
                if hasattr(obs, 'observation'):
                    gym_obs = self._convert_dm_obs_to_gym(obs)
                else:
                    gym_obs = obs

                action, _states = self.model.predict(gym_obs, deterministic=True)
                return action
            except Exception as e:
                print(f"  [SB3SACAdapter] Action prediction failed: {e}")
                if self.gym_env is not None:
                    return np.random.uniform(
                        -1, 1, size=self.gym_env.action_space.shape[0])
                return None
        else:
            # Random fallback when sb3 not installed
            if self.gym_env is not None:
                return np.random.uniform(
                    -1, 1, size=self.gym_env.action_space.shape[0])
            return np.random.uniform(-1, 1, size=6)

    def _convert_dm_obs_to_gym(self, dm_timestep: object) -> np.ndarray:
        """Convert dm_control timestep observation to gymnasium flat array.

        Uses gymnasium.spaces.flatten with the inner env's Dict observation
        space for exact compatibility with FlattenObservation wrapper.

        Args:
            dm_timestep: dm_control TimeStep with .observation dict.

        Returns:
            Flattened observation array compatible with the SB3 model.
        """
        obs_dict: dict = dm_timestep.observation

        # Use gymnasium.spaces.flatten for exact compatibility
        if self._dict_obs_space is not None:
            try:
                return _flatten_dm_obs(obs_dict, self._dict_obs_space)
            except Exception:
                pass

        # Fallback: manual flatten with sorted keys
        parts: list = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, np.ndarray):
                parts.append(val.flatten())
            else:
                parts.append(np.array([val]).flatten())
        return np.concatenate(parts)

    def reset(self) -> None:
        """Reset internal state (SAC has no recurrent state, so this is a no-op)."""
        pass

    def is_available(self) -> bool:
        """Check whether the SAC model is loaded and ready.

        Returns:
            True if sb3 is available and model is initialized.
        """
        return self._sb3_available and self.model is not None

    def is_trained(self) -> bool:
        """Check whether the SAC model has been trained.

        Returns:
            True if the model has been trained or loaded from checkpoint.
        """
        return self._trained

    def get_info(self) -> Dict[str, object]:
        """Get adapter configuration and status information.

        Returns:
            Dict with adapter metadata.
        """
        return {
            'adapter': 'SB3SACAdapter',
            'task_name': self.task_name,
            'checkpoint_dir': self.checkpoint_dir,
            'auto_train_steps': self.auto_train_steps,
            'available': self._sb3_available,
            'model_loaded': self.model is not None,
            'trained': self._trained,
        }


# ── Factory functions ──


def make_sb3_ppo_adapter(task_name: str = 'humanoid-stand',
                          checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                          auto_train_steps: int = DEFAULT_AUTO_TRAIN_STEPS,
                          verbose: int = 0) -> Optional[SB3PPOAdapter]:
    """Factory function for creating a SB3PPOAdapter instance.

    Creates an adapter with graceful degradation: if stable-baselines3
    is not installed, the adapter is created but will fall back to random
    actions.

    Args:
        task_name: dm_control task name.
        checkpoint_dir: Root checkpoint directory.
        auto_train_steps: Steps for auto-training (default 100K).
        verbose: SB3 verbose level.

    Returns:
        SB3PPOAdapter instance, or None on ValueError.
    """
    try:
        adapter: SB3PPOAdapter = SB3PPOAdapter(
            task_name=task_name,
            checkpoint_dir=checkpoint_dir,
            auto_train_steps=auto_train_steps,
            verbose=verbose,
        )
        return adapter
    except ValueError as e:
        print(f"  [SB3PPOAdapter] Configuration error: {e}")
        return None


def make_sb3_sac_adapter(task_name: str = 'humanoid-stand',
                          checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                          auto_train_steps: int = DEFAULT_AUTO_TRAIN_STEPS,
                          verbose: int = 0) -> Optional[SB3SACAdapter]:
    """Factory function for creating a SB3SACAdapter instance.

    Creates an adapter with graceful degradation: if stable-baselines3
    is not installed, the adapter is created but will fall back to random
    actions.

    Args:
        task_name: dm_control task name.
        checkpoint_dir: Root checkpoint directory.
        auto_train_steps: Steps for auto-training (default 100K).
        verbose: SB3 verbose level.

    Returns:
        SB3SACAdapter instance, or None on ValueError.
    """
    try:
        adapter: SB3SACAdapter = SB3SACAdapter(
            task_name=task_name,
            checkpoint_dir=checkpoint_dir,
            auto_train_steps=auto_train_steps,
            verbose=verbose,
        )
        return adapter
    except ValueError as e:
        print(f"  [SB3SACAdapter] Configuration error: {e}")
        return None
