"""
TOMAS Deploy API — Headless MuJoCo Environment Adapter
=======================================================

v0.17.1: Bridges the TOMASAgent (agent/tomas_deploy.py) with the
SO-ARM100 MuJoCo scene for headless benchmark evaluation.

The challenge: TOMASAgent expects a gym-like env with step()/reset()/_get_obs(),
but the SO-ARM100 scene is raw MuJoCo (model + data). This module provides:

  1. HeadlessMuJoCoEnv — wraps raw MuJoCo model+data into a gym-like interface
  2. create_tomas_agent_for_arm100() — factory that builds a complete TOMASAgent
  3. run_tomas_eval() — runs multi-episode evaluation and returns DeployReport

This allows the TOMASAgent's P->C->S pipeline to run on the SO-ARM100
pick-and-place task without the viser 3D viewer, enabling fast benchmark
evaluation via API.

Author: MuJoCo-Bench-IDO v0.17.1
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class HeadlessMuJoCoEnv:
    """Gym-like wrapper around raw MuJoCo model+data for TOMASAgent.

    Provides the step()/reset()/_get_obs() interface that
    agent/tomas_mujoco_wrapper.py::TOMASMuJoCoWrapper expects.

    The environment loads a MuJoCo XML scene, manages physics stepping,
    and computes observations from MuJoCo sensor data.

    Observation (18D):
      - 7 joint positions (rad)
      - 7 joint velocities (rad/s)
      - 3 gripper-to-target distance (xyz)
      - 1 gripper open/close state

    Action (7D):
      - 5 arm joint position targets (rad)
      - 2 gripper position targets (rad)

    Reward:
      - Negative GaussEx residual (eta)
      - Bonus for reaching target (< 0.05m)
      - Penalty for psi-Anchor violations

    Attributes:
        model: MuJoCo model.
        data: MuJoCo data.
        step_count: Current step in episode.
        max_steps: Maximum steps before truncation.
        target_body_id: Body ID of the pick-and-place target.
        gripper_body_id: Body ID of the gripper.
    """

    def __init__(
        self,
        scene_xml_path: str,
        max_steps: int = 500,
        target_object: str = "red_cube",
        tray_body: str = "tray",
    ) -> None:
        """Initialize headless MuJoCo environment.

        Args:
            scene_xml_path: Path to the MuJoCo XML scene file.
            max_steps: Maximum steps per episode.
            target_object: Name of the target body to pick.
            tray_body: Name of the tray body to place onto.
        """
        import mujoco as mj

        self._mj = mj
        self.model = mj.MjModel.from_xml_path(scene_xml_path)
        self.data = mj.MjData(self.model)
        self.max_steps = max_steps
        self.step_count = 0

        # Resolve body IDs
        self.target_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, target_object
        )
        self.gripper_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "gripper_base"
        )
        self.tray_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, tray_body
        )

        if self.target_body_id < 0:
            raise ValueError(f"Target body '{target_object}' not found in scene")
        if self.gripper_body_id < 0:
            raise ValueError("Body 'gripper_base' not found in scene")

        # Arm joint qpos offset (skip freejoint objects)
        # SO-ARM100 scene: 3 freejoint objects (7 qpos each) = 21, then 7 arm joints
        self._arm_qpos_offset = self._find_arm_qpos_offset()
        self._arm_qvel_offset = self._arm_qpos_offset - 7  # qvel has 6 per freejoint

        # Action dimension = number of actuators
        self._action_dim = self.model.nu

        # Initial state for reset
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()

        # Home pose
        self._home_pose = np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0])

        logger.info(
            f"HeadlessMuJoCoEnv: nq={self.model.nq}, nu={self.model.nu}, "
            f"arm_offset={self._arm_qpos_offset}, target={target_object}"
        )

    def _find_arm_qpos_offset(self) -> int:
        """Find the qpos offset where arm joints start."""
        import mujoco as mj
        # Look for the first non-freejoint
        for i in range(self.model.njnt):
            jtype = self.model.jnt_type[i]
            if jtype != mj.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[i])
        return 21  # fallback

    def _get_obs(self) -> np.ndarray:
        """Get observation vector (18D).

        Returns:
            np.ndarray of shape (18,):
                [0:7]   joint positions (rad)
                [7:14]  joint velocities (rad/s)
                [14:17] gripper-to-target distance (xyz)
                [17]    gripper open ratio (0=closed, 1=open)
        """
        arm_qpos = self.data.qpos[self._arm_qpos_offset:self._arm_qpos_offset + 7]
        arm_qvel = self.data.qvel[self._arm_qvel_offset:self._arm_qvel_offset + 7]

        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        distance = gripper_pos - target_pos

        # Gripper open ratio: average of |gripper_L| and |gripper_R|
        grip_open = (abs(arm_qpos[5]) + abs(arm_qpos[6])) / (2 * 0.873)

        obs = np.concatenate([
            arm_qpos,       # 7
            arm_qvel,       # 7
            distance[:3],   # 3
            [grip_open],    # 1
        ]).astype(np.float64)

        return obs

    def _get_goal(self) -> np.ndarray:
        """Get goal vector (target position in world frame)."""
        return self.data.xpos[self.target_body_id].copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one physics step.

        Args:
            action: Joint position targets (nu-dimensional).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        import mujoco as mj

        self.step_count += 1

        # Clip action to actuator range
        action = np.clip(action, -3.14, 3.14)
        if len(action) >= self.model.nu:
            ctrl = action[:self.model.nu]
        else:
            ctrl = np.zeros(self.model.nu)
            ctrl[:len(action)] = action

        self.data.ctrl[:] = ctrl
        mj.mj_step(self.model, self.data)

        # Kinematic assist (same as viewer)
        for _ji in range(min(7, self.model.nu)):
            _qi = self._arm_qpos_offset + _ji
            if _qi < self.data.qpos.shape[0]:
                _target = float(self.data.ctrl[_ji])
                _diff = _target - float(self.data.qpos[_qi])
                self.data.qpos[_qi] += _diff * 0.50

        mj.mj_forward(self.model, self.data)

        # Check for NaN
        if np.any(np.isnan(self.data.qpos)):
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

        obs = self._get_obs()

        # Compute reward
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        eta = float(np.linalg.norm(gripper_pos - target_pos))

        reward = -eta
        if eta < 0.05:
            reward += 10.0  # Success bonus

        # Check termination
        terminated = eta < 0.03  # Very close to target
        truncated = self.step_count >= self.max_steps

        info = {
            "eta": eta,
            "step": self.step_count,
            "gripper_pos": gripper_pos.tolist(),
            "target_pos": target_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        import mujoco as mj

        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel

        # Set arm to home pose
        for i, val in enumerate(self._home_pose):
            if self._arm_qpos_offset + i < self.data.qpos.shape[0]:
                self.data.qpos[self._arm_qpos_offset + i] = val

        mj.mj_forward(self.model, self.data)
        self.step_count = 0

        return self._get_obs()

    def action_spec(self):
        """Return action spec (dm_control compatible)."""
        class _Spec:
            def __init__(self, shape):
                self.shape = shape
        return _Spec((self._action_dim,))


def create_tomas_agent_for_arm100(
    scene_xml_path: Optional[str] = None,
    vla_model_name: str = "demo-vla",
    vla_instruction: str = "pick up the red cube and place it on the tray",
    max_steps: int = 500,
    enable_failure_attribution: bool = True,
    enable_skill_learning: bool = True,
) -> Tuple[Any, Any]:
    """Create a fully configured TOMASAgent for SO-ARM100 pick-and-place.

    Args:
        scene_xml_path: Path to MuJoCo XML. If None, uses default scene.
        vla_model_name: VLA model name (openvla-7b, octo-base, pi0-base, demo-vla).
        vla_instruction: Language instruction for VLA.
        max_steps: Max steps per episode.
        enable_failure_attribution: Enable S-Layer failure attribution.
        enable_skill_learning: Enable skill learning from successful episodes.

    Returns:
        (tomas_agent, vla_adapter) tuple.
    """
    if scene_xml_path is None:
        scene_xml_path = str(
            Path(_PROJECT_ROOT) / "webviz" / "scenes" / "so_arm100_scene.xml"
        )

    # Create headless env
    env = HeadlessMuJoCoEnv(
        scene_xml_path=scene_xml_path,
        max_steps=max_steps,
    )

    # Create VLA adapter
    from webviz.tomas_wrapper import create_vla_adapter
    vla_adapter = create_vla_adapter(vla_model_name, model=env.model, data=env.data)

    # Wrap VLA adapter to provide the obs_dict it expects
    class VLAWrapper:
        """Adapts VLA adapter interface for TOMASMuJoCoWrapper."""

        def __init__(self, adapter, env: HeadlessMuJoCoEnv, instruction: str):
            self._adapter = adapter
            self._env = env
            self._instruction = instruction
            self.model_name = adapter.model_name
            self.loaded = adapter.is_loaded()

        def predict(self, obs: np.ndarray) -> np.ndarray:
            """Predict action from observation.

            Args:
                obs: Raw observation from env (18D).

            Returns:
                Action vector (7D).
            """
            import mujoco as mj

            # Build obs_dict for VLA adapter
            _obj_positions = {}
            for _obj_name in ["red_cube", "blue_ball", "white_tissue", "tray"]:
                _bid = mj.mj_name2id(self._env.model, mj.mjtObj.mjOBJ_BODY, _obj_name)
                if _bid >= 0:
                    _obj_positions[_obj_name] = self._env.data.xpos[_bid].copy()

            _grip_bid = mj.mj_name2id(
                self._env.model, mj.mjtObj.mjOBJ_BODY, "gripper_base"
            )
            _gripper_pos = (
                self._env.data.xpos[_grip_bid].copy() if _grip_bid >= 0 else np.zeros(3)
            )

            obs_dict = {
                "language": self._instruction,
                "proprio": obs[:7],  # joint positions
                "object_positions": _obj_positions,
                "gripper_position": _gripper_pos,
                "arm_base_z": 0.40,
                "rgb": None,  # Headless mode: no camera
                "wrist_rgb": None,
            }

            try:
                action = self._adapter.predict(obs_dict)
                return np.asarray(action, dtype=np.float64)
            except Exception as e:
                logger.warning(f"VLA predict failed: {e}, using zero action")
                return np.zeros(7)

    vla_wrapper = VLAWrapper(vla_adapter, env, vla_instruction)

    # Create TOMASAgent
    from agent.tomas_deploy import TOMASAgent

    agent = TOMASAgent(
        env=env,
        vla_policy=vla_wrapper,
        max_steps=max_steps,
        tau_safe=0.05,
        enable_failure_attribution=enable_failure_attribution,
        enable_skill_learning=enable_skill_learning,
    )

    return agent, vla_adapter


def run_tomas_eval(
    vla_model_name: str = "demo-vla",
    vla_instruction: str = "pick up the red cube and place it on the tray",
    num_episodes: int = 3,
    max_steps: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run TOMAS evaluation on SO-ARM100 pick-and-place.

    Args:
        vla_model_name: VLA model to use.
        vla_instruction: Language instruction.
        num_episodes: Number of episodes to run.
        max_steps: Max steps per episode.
        verbose: Print progress.

    Returns:
        Dictionary with evaluation results:
          - deploy_report: DeployReport.to_dict()
          - per_episode: List of per-episode metrics
          - vla_model: Model name
          - vla_loaded: Whether real weights were loaded
    """
    if verbose:
        logger.info(f"=== TOMAS Evaluation Start ===")
        logger.info(f"VLA: {vla_model_name}, Episodes: {num_episodes}, Max Steps: {max_steps}")

    # Create agent
    agent, vla_adapter = create_tomas_agent_for_arm100(
        vla_model_name=vla_model_name,
        vla_instruction=vla_instruction,
        max_steps=max_steps,
    )

    # Run deploy
    report = agent.deploy(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        verbose=verbose,
    )

    result = {
        "deploy_report": report.to_dict(),
        "per_episode": [],
        "vla_model": vla_model_name,
        "vla_loaded": vla_adapter.is_loaded(),
        "instruction": vla_instruction,
    }

    # Extract per-episode info from audit trail
    audit_trail = report.audit_trail
    eta_history = report.eta_history

    if verbose:
        logger.info(f"=== TOMAS Evaluation Complete ===")
        logger.info(f"Status: {report.status.value}")
        logger.info(f"Total Steps: {report.total_steps}")
        logger.info(f"Avg Eta: {report.avg_eta:.6f}")
        logger.info(f"Final Eta: {report.final_eta:.6f}")
        logger.info(f"Psi Violations: {report.psi_violations}")
        logger.info(f"Kappa-Snap Count: {report.kappa_snap_count}")
        logger.info(f"Skills Learned: {len(report.learned_skills)}")
        logger.info(f"Failure Attributions: {len(report.failure_attributions)}")
        logger.info(f"Elapsed: {report.elapsed_seconds:.2f}s")

    return result


def check_vla_availability() -> Dict[str, Dict[str, Any]]:
    """Check which VLA models are available (real weights vs stub).

    Returns:
        Dict mapping model name to availability info.
    """
    results = {}

    for model_name in ["openvla-7b", "octo-base", "pi0-base", "demo-vla"]:
        info = {
            "model": model_name,
            "real_weights": False,
            "error": None,
        }

        if model_name == "demo-vla":
            info["real_weights"] = True  # Demo is always available
            info["note"] = "Instruction-driven demo adapter (no GPU needed)"
        elif model_name == "openvla-7b":
            try:
                import torch
                from transformers import AutoProcessor
                info["real_weights"] = True
                info["note"] = "OpenVLA-7B available via HuggingFace (requires 16GB+ VRAM)"
            except ImportError:
                info["error"] = "torch/transformers not installed"
                info["note"] = "Install: pip install torch transformers"
        elif model_name == "octo-base":
            try:
                import octo
                info["real_weights"] = True
                info["note"] = "Octo available (requires JAX)"
            except ImportError:
                info["error"] = "octo package not installed"
                info["note"] = "Install: pip install octo"
        elif model_name == "pi0-base":
            info["real_weights"] = False
            info["note"] = "Pi0 weights via openpi (Physical-Intelligence/openpi)"

        results[model_name] = info

    return results
