"""
WeldingEnv 单元测试
====================

测试焊接机器人仿真环境的核心功能:
  - 环境创建
  - reset 返回 18 维观测
  - step 返回正确结构
  - 观测维度验证
  - 动作范围验证
  - 奖励为负数 (惩罚制)
  - 4 种焊接姿态初始化

Author: MuJoCo-Bench-IDO Welding Module v0.1.0
"""

import os
import sys
import numpy as np
import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.welding_env import WeldingEnv, ACTION_DIM, OBS_DIM, ACTION_LOW, ACTION_HIGH


class TestWeldingEnvCreation:
    """测试环境创建."""

    def test_env_creation(self):
        """创建 WeldingEnv, 检查 model/data 不为 None."""
        env = WeldingEnv(weld_type="flat")
        assert env.model is not None, "MuJoCo model should not be None"
        assert env.data is not None, "MuJoCo data should not be None"

    def test_env_creation_with_explicit_path(self):
        """使用显式 XML 路径创建环境."""
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", "envs", "assets", "mujoco_weld_robot.xml"
        )
        env = WeldingEnv(xml_path=xml_path, weld_type="flat")
        assert env.model is not None

    def test_invalid_weld_type(self):
        """无效焊接类型应抛出 ValueError."""
        with pytest.raises(ValueError):
            WeldingEnv(weld_type="invalid_type")


class TestWeldingEnvReset:
    """测试 reset 方法."""

    def test_reset_returns_18dim(self):
        """reset 返回 18 维 observation."""
        env = WeldingEnv(weld_type="flat")
        obs = env.reset()
        assert obs.shape == (OBS_DIM,), f"Expected shape ({OBS_DIM},), got {obs.shape}"

    def test_reset_returns_numpy_array(self):
        """reset 返回 numpy 数组."""
        env = WeldingEnv(weld_type="flat")
        obs = env.reset()
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"

    def test_reset_temperature_initialized(self):
        """reset 后温度初始化为 25°C."""
        env = WeldingEnv(weld_type="flat")
        obs = env.reset()
        # 温度在 obs[16]
        assert obs[16] == 25.0, f"Initial temperature should be 25.0, got {obs[16]}"


class TestWeldingEnvStep:
    """测试 step 方法."""

    def test_step_returns_correct_dict(self):
        """step 返回正确的字典结构."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)

        assert "observation" in result, "Result should contain 'observation'"
        assert "reward" in result, "Result should contain 'reward'"
        assert "done" in result, "Result should contain 'done'"
        assert "info" in result, "Result should contain 'info'"

    def test_step_observation_dim(self):
        """step 返回的 observation 是 18 维."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert result["observation"].shape == (OBS_DIM,)

    def test_step_reward_is_float(self):
        """step 返回的 reward 是 float 类型."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert isinstance(result["reward"], (float, np.floating)), \
            f"Reward should be float, got {type(result['reward'])}"

    def test_step_done_is_bool(self):
        """step 返回的 done 是 bool 类型."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert isinstance(result["done"], (bool, np.bool_)), \
            f"Done should be bool, got {type(result['done'])}"

    def test_step_info_contains_quality(self):
        """step 的 info 包含 quality 字典."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert "quality" in result["info"], "Info should contain 'quality'"
        assert "eta" in result["info"]["quality"]
        assert "porosity" in result["info"]["quality"]
        assert "distortion" in result["info"]["quality"]

    def test_step_info_contains_safety(self):
        """step 的 info 包含 safety 检查结果."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert "safety" in result["info"], "Info should contain 'safety'"
        assert "passed" in result["info"]["safety"]


class TestObservationDimension:
    """测试观测维度."""

    def test_observation_dim_constant(self):
        """OBS_DIM 常量为 18."""
        assert OBS_DIM == 18, f"OBS_DIM should be 18, got {OBS_DIM}"

    def test_get_observation_shape(self):
        """get_observation 返回 18 维向量."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        obs = env.get_observation()
        assert obs.shape == (18,)

    def test_observation_tcp_position(self):
        """观测前3维是 TCP 位置 (x, y, z)."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        obs = env.get_observation()
        # TCP 位置应该在合理范围内 (机器人工作空间内)
        tcp_pos = obs[0:3]
        assert np.all(np.isfinite(tcp_pos)), "TCP position should be finite"

    def test_observation_joints(self):
        """观测 [6:12] 是关节角度."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        obs = env.get_observation()
        joints = obs[6:12]
        assert len(joints) == 6, "Should have 6 joint values"
        # 关节角度应该在限位范围内
        assert np.all(np.abs(joints) < 3.15), "Joint angles should be within range"


class TestActionRange:
    """测试动作范围."""

    def test_action_dim_constant(self):
        """ACTION_DIM 常量为 4."""
        assert ACTION_DIM == 4, f"ACTION_DIM should be 4, got {ACTION_DIM}"

    def test_action_low_high(self):
        """动作上下界正确."""
        assert len(ACTION_LOW) == 4
        assert len(ACTION_HIGH) == 4
        assert ACTION_LOW[0] == 50.0   # current min
        assert ACTION_HIGH[0] == 350.0  # current max
        assert ACTION_LOW[1] == 14.0   # voltage min
        assert ACTION_HIGH[1] == 32.0   # voltage max

    def test_action_spec(self):
        """action_spec 返回正确规格."""
        env = WeldingEnv(weld_type="flat")
        spec = env.action_spec
        assert spec["shape"] == (4,)
        assert np.allclose(spec["low"], ACTION_LOW)
        assert np.allclose(spec["high"], ACTION_HIGH)

    def test_step_clips_action(self):
        """step 裁剪超出范围的动作."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        # 超出范围的动作
        action = np.array([500.0, 50.0, 10.0, 20.0])
        result = env.step(action)
        # 不应崩溃, 且 info 中有裁剪后的动作
        clipped = result["info"]["action_clipped"]
        assert clipped[0] <= ACTION_HIGH[0], "Current should be clipped"
        assert clipped[1] <= ACTION_HIGH[1], "Voltage should be clipped"


class TestReward:
    """测试奖励计算."""

    def test_reward_is_negative(self):
        """reward 为负数 (所有质量项都是惩罚)."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert result["reward"] <= 0.0, \
            f"Reward should be <= 0 (penalty-based), got {result['reward']}"

    def test_reward_with_high_current(self):
        """高电流 → 更低奖励 (更高 eta 惩罚)."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        normal_action = np.array([200.0, 24.0, 2.0, 8.0])
        result_normal = env.step(normal_action)

        env.reset()
        high_action = np.array([340.0, 30.0, 4.0, 3.0])
        result_high = env.step(high_action)

        # 高电流偏离标准值更远, eta 更高, 奖励更低
        assert result_high["reward"] <= result_normal["reward"] + 0.1, \
            "High current should produce lower or equal reward"

    def test_compute_welding_reward(self):
        """直接测试 compute_welding_reward 方法."""
        env = WeldingEnv(weld_type="flat")
        env.reset()
        quality = {"eta": 0.5, "porosity": 0.3, "distortion": 0.1}
        reward = env.compute_welding_reward(quality)
        # reward = -0.5*10 - 0.3*20 - 0.1*50 - stickout_penalty
        expected = -0.5 * 10 - 0.3 * 20 - 0.1 * 50
        # 减去可能的 stickout_penalty (0 or 1)
        assert reward <= expected + 0.01, \
            f"Reward {reward} should be <= {expected}"


class TestWeldTypes:
    """测试 4 种焊接姿态."""

    @pytest.mark.parametrize("weld_type", ["flat", "horizontal", "vertical", "overhead"])
    def test_weld_type_initialization(self, weld_type):
        """4 种焊接姿态都能正确初始化."""
        env = WeldingEnv(weld_type=weld_type)
        obs = env.reset()
        assert obs.shape == (OBS_DIM,), f"Failed for weld_type={weld_type}"
        assert env.weld_type == weld_type

    @pytest.mark.parametrize("weld_type", ["flat", "horizontal", "vertical", "overhead"])
    def test_weld_type_step(self, weld_type):
        """4 种焊接姿态都能执行 step."""
        env = WeldingEnv(weld_type=weld_type)
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert "observation" in result
        assert result["observation"].shape == (OBS_DIM,)

    def test_weld_type_in_info(self):
        """step info 包含 weld_type."""
        env = WeldingEnv(weld_type="vertical")
        env.reset()
        action = np.array([200.0, 24.0, 2.0, 8.0])
        result = env.step(action)
        assert result["info"]["weld_type"] == "vertical"


class TestWaypoints:
    """测试焊缝 waypoints."""

    def test_waypoints_generated(self):
        """环境生成焊缝 waypoints."""
        env = WeldingEnv(weld_type="flat")
        assert env.waypoints is not None
        assert env.waypoints.shape == (20, 3), \
            f"Expected (20, 3), got {env.waypoints.shape}"

    def test_waypoints_along_x(self):
        """焊缝 waypoints 沿 X 轴方向."""
        env = WeldingEnv(weld_type="flat")
        wp = env.waypoints
        # X 坐标应单调递增
        assert np.all(np.diff(wp[:, 0]) >= 0), "Waypoint X should be increasing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
