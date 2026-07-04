"""
WeldingController 单元测试
===========================

测试 6-DOF 焊缝跟踪控制器的核心功能:
  - IK 精度 (DLS 求解后 FK 误差 < 0.03mm)
  - 轨迹生成
  - 跟踪误差计算
  - 正运动学

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

from agent.welding_controller import WeldingController


def generate_test_waypoints(n: int = 20) -> np.ndarray:
    """生成测试用焊缝 waypoints.

    在工件表面上方沿 X 轴生成直线焊缝.

    Args:
        n: waypoint 数量.

    Returns:
        n×3 数组 (m).
    """
    waypoints = np.zeros((n, 3))
    x_start = 0.9
    x_end = 1.1
    waypoints[:, 0] = np.linspace(x_start, x_end, n)
    waypoints[:, 1] = 0.0
    waypoints[:, 2] = 0.28  # 工件上方
    return waypoints


@pytest.fixture
def controller():
    """创建 WeldingController 测试 fixture."""
    waypoints = generate_test_waypoints(20)
    return WeldingController(waypoints, xml_path=os.path.join(
        PROJECT_ROOT, "envs", "assets", "mujoco_weld_robot.xml"
    ))


class TestForwardKinematics:
    """测试正运动学."""

    def test_fk_returns_6d(self, controller):
        """FK 返回 6 维 TCP 位姿."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        tcp = controller._forward_kinematics(joint_angles)
        assert tcp.shape == (6,), f"Expected (6,), got {tcp.shape}"

    def test_fk_position_reasonable(self, controller):
        """FK 返回合理的 TCP 位置 (在工作空间内)."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        tcp = controller._forward_kinematics(joint_angles)
        # TCP 应该在合理范围内 (机器人前方, 高度>0)
        assert np.isfinite(tcp[0]), "TCP x should be finite"
        assert np.isfinite(tcp[1]), "TCP y should be finite"
        assert np.isfinite(tcp[2]), "TCP z should be finite"
        assert tcp[2] > 0.0, "TCP z should be above ground"

    def test_fk_different_angles_different_positions(self, controller):
        """不同关节角度产生不同 TCP 位置."""
        angles1 = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        angles2 = np.array([0.5, 0.3, -0.5, 0.0, 0.1, 0.0])
        tcp1 = controller._forward_kinematics(angles1)
        tcp2 = controller._forward_kinematics(angles2)
        assert not np.allclose(tcp1[:3], tcp2[:3]), \
            "Different joint angles should produce different TCP positions"

    def test_fk_restores_state(self, controller):
        """FK 执行后恢复原始状态."""
        original_qpos = controller.data.qpos[:6].copy()
        joint_angles = np.array([0.3, 0.3, -0.3, 0.3, 0.3, 0.3])
        controller._forward_kinematics(joint_angles)
        restored_qpos = controller.data.qpos[:6].copy()
        assert np.allclose(original_qpos, restored_qpos), \
            "FK should restore original qpos after computation"


class TestIKPrecision:
    """测试逆运动学精度."""

    def test_ik_precision(self, controller):
        """给定 TCP 目标, solve_ik 后 FK 的误差 < 0.03mm.

        注意: DLS IK 可能无法精确到达所有目标点 (工作空间限制),
        此测试选择一个可达的目标点.
        """
        # 先用 FK 得到一个可达的 TCP 位置
        seed_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        tcp_target = controller._forward_kinematics(seed_angles)[:3]

        # 从不同初始角度求解 IK
        controller.data.qpos[:6] = np.array([0.1, 0.3, -0.5, 0.1, 0.2, 0.1])
        solved_angles = controller.solve_ik(tcp_target)

        # FK 验证
        tcp_result = controller._forward_kinematics(solved_angles)[:3]
        error_mm = float(np.linalg.norm(tcp_result - tcp_target)) * 1000.0

        assert error_mm < controller.TRACKING_TOLERANCE, \
            f"IK precision error: {error_mm:.4f}mm > {controller.TRACKING_TOLERANCE}mm"

    def test_ik_returns_6d(self, controller):
        """solve_ik 返回 6 维关节角度."""
        tcp_target = np.array([1.0, 0.0, 0.3])
        angles = controller.solve_ik(tcp_target)
        assert angles.shape == (6,), f"Expected (6,), got {angles.shape}"

    def test_ik_within_joint_limits(self, controller):
        """IK 结果在关节限位范围内."""
        tcp_target = np.array([1.0, 0.0, 0.3])
        angles = controller.solve_ik(tcp_target)
        assert np.all(angles >= controller._joint_lower - 0.01), \
            f"Joint angles below lower limit: {angles}"
        assert np.all(angles <= controller._joint_upper + 0.01), \
            f"Joint angles above upper limit: {angles}"


class TestTrajectoryGeneration:
    """测试轨迹生成."""

    def test_trajectory_generation(self, controller):
        """generate_trajectory 返回正确数量的 waypoint."""
        n_steps = 50
        trajectory = controller.generate_trajectory(n_steps=n_steps)
        assert len(trajectory) == n_steps, \
            f"Expected {n_steps} waypoints, got {len(trajectory)}"

    def test_trajectory_element_structure(self, controller):
        """轨迹元素结构正确: (joint_angles, tcp_target)."""
        trajectory = controller.generate_trajectory(n_steps=10)
        first = trajectory[0]
        assert len(first) == 2, "Each trajectory element should be (joint_angles, tcp_target)"
        joint_angles, tcp_target = first
        assert joint_angles.shape == (6,), "Joint angles should be 6D"
        assert tcp_target.shape == (3,), "TCP target should be 3D"

    def test_trajectory_tcp_progression(self, controller):
        """轨迹中 TCP 目标沿焊缝方向递进."""
        trajectory = controller.generate_trajectory(n_steps=20)
        tcp_targets = np.array([t[1] for t in trajectory])
        # X 坐标应大致单调递增 (沿焊缝方向)
        x_coords = tcp_targets[:, 0]
        assert x_coords[-1] > x_coords[0], "TCP should progress along seam (X increasing)"

    def test_trajectory_default_steps(self, controller):
        """默认步数生成轨迹."""
        trajectory = controller.generate_trajectory()
        assert len(trajectory) == 200, "Default should be 200 steps"


class TestTrackingError:
    """测试跟踪误差计算."""

    def test_tracking_error_zero(self, controller):
        """相同位置 → 误差为 0."""
        tcp = np.array([1.0, 0.0, 0.3])
        error = controller.compute_tracking_error(tcp, tcp)
        assert error == 0.0, "Same position should have 0 error"

    def test_tracking_error_nonzero(self, controller):
        """不同位置 → 误差 > 0."""
        tcp_actual = np.array([1.0, 0.0, 0.3])
        tcp_target = np.array([1.0, 0.0, 0.31])  # 10mm 偏差
        error = controller.compute_tracking_error(tcp_actual, tcp_target)
        assert abs(error - 10.0) < 0.1, f"Expected ~10mm, got {error:.2f}mm"

    def test_tracking_error_6d_input(self, controller):
        """6D 输入也正确计算 (只取前3维位置)."""
        tcp_actual = np.array([1.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        tcp_target = np.array([1.0, 0.0, 0.3, 0.1, 0.1, 0.1])
        error = controller.compute_tracking_error(tcp_actual, tcp_target)
        assert error == 0.0, "Same position (different orientation) should have 0 error"

    def test_tracking_error_units_mm(self, controller):
        """跟踪误差单位为 mm."""
        tcp_actual = np.array([0.0, 0.0, 0.0])
        tcp_target = np.array([0.001, 0.0, 0.0])  # 1mm 偏差
        error = controller.compute_tracking_error(tcp_actual, tcp_target)
        assert abs(error - 1.0) < 0.01, f"1mm offset should give ~1mm error, got {error}"


class TestDLS:
    """测试阻尼最小二乘法."""

    def test_dls_output_shape(self, controller):
        """DLS 输出维度正确."""
        J = np.random.randn(3, 6)
        error = np.array([0.01, 0.0, 0.0])
        delta = controller._damped_least_squares(J, error, damping=0.1)
        assert delta.shape == (6,), f"Expected (6,), got {delta.shape}"

    def test_dls_finite(self, controller):
        """DLS 输出为有限值."""
        J = np.eye(3, 6)  # 3×6 矩阵
        error = np.array([0.01, 0.02, 0.03])
        delta = controller._damped_least_squares(J, error, damping=0.1)
        assert np.all(np.isfinite(delta)), "DLS output should be finite"

    def test_dls_singular_matrix(self, controller):
        """奇异雅可比矩阵不崩溃 (使用伪逆)."""
        J = np.zeros((3, 6))  # 全零矩阵 → 奇异
        error = np.array([0.01, 0.0, 0.0])
        delta = controller._damped_least_squares(J, error, damping=0.1)
        assert np.all(np.isfinite(delta)), "DLS should handle singular matrix gracefully"

    def test_dls_damping_effect(self, controller):
        """阻尼系数影响结果大小."""
        J = np.eye(3, 6)
        error = np.array([1.0, 0.0, 0.0])
        delta_low = controller._damped_least_squares(J, error, damping=0.01)
        delta_high = controller._damped_least_squares(J, error, damping=1.0)
        # 高阻尼应产生更小的增量
        assert np.linalg.norm(delta_high) <= np.linalg.norm(delta_low) + 1e-6, \
            "Higher damping should produce smaller delta"


class TestJacobian:
    """测试雅可比矩阵计算."""

    def test_jacobian_shape(self, controller):
        """雅可比矩阵形状为 3×6."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        J = controller._jacobian(joint_angles)
        assert J.shape == (3, 6), f"Expected (3, 6), got {J.shape}"

    def test_jacobian_finite(self, controller):
        """雅可比矩阵元素为有限值."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        J = controller._jacobian(joint_angles)
        assert np.all(np.isfinite(J)), "Jacobian should be finite"

    def test_numerical_jacobian_shape(self, controller):
        """数值雅可比矩阵形状正确."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        J = controller._numerical_jacobian(joint_angles)
        assert J.shape == (3, 6)

    def test_jacobian_nonzero(self, controller):
        """非奇异状态下雅可比矩阵非全零."""
        joint_angles = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0])
        J = controller._jacobian(joint_angles)
        assert np.any(J != 0.0), "Jacobian should not be all zeros for non-singular config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
