"""
WeldingController — 6-DOF 焊缝跟踪控制器
==========================================

实现阻尼最小二乘法 (DLS) 逆运动学求解，生成焊缝跟踪轨迹。

核心算法:
  - 正运动学 (FK): 设置关节角度 → mj_forward → 读取 wire_tip site 位置
  - 雅可比矩阵: 使用 MuJoCo mj_jacBody 或数值差分计算 3×6 位置雅可比
  - DLS IK: θ_delta = J^T (J J^T + λ²I)^{-1} · e
  - 轨迹生成: 三次样条插值 waypoints → 均匀采样 → IK 求解

Author: MuJoCo-Bench-IDO Welding Module v0.1.0
"""

import os
import numpy as np
import mujoco
from typing import Optional, List, Tuple

# 尝试导入 scipy 用于样条插值，回退到 numpy
try:
    from scipy.interpolate import CubicSpline
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class WeldingController:
    """6-DOF 焊缝跟踪控制器.

    使用阻尼最小二乘法 (DLS) 求解逆运动学，生成沿焊缝的关节空间轨迹。
    控制器直接操作 MuJoCo model/data 进行正运动学和雅可比计算。

    Attributes:
        TRACKING_TOLERANCE: 目标跟踪精度 (mm).
        waypoints: Nx3 焊缝 TCP 目标位置序列 (m).
        model: MuJoCo 模型.
        data: MuJoCo 数据.
    """

    TRACKING_TOLERANCE: float = 0.03  # mm

    def __init__(
        self,
        waypoints: np.ndarray,
        model: Optional[mujoco.MjModel] = None,
        data: Optional[mujoco.MjData] = None,
        xml_path: Optional[str] = None,
    ) -> None:
        """初始化焊缝跟踪控制器.

        Args:
            waypoints: Nx3 数组, 焊缝上的 TCP 目标位置序列 (单位: m).
            model: MuJoCo 模型. 如果为 None, 从 xml_path 加载.
            data: MuJoCo 数据. 如果为 None, 从 model 创建.
            xml_path: MuJoCo XML 路径. 当 model 为 None 时使用.
        """
        self.waypoints: np.ndarray = np.asarray(waypoints, dtype=np.float64)

        # 加载 MuJoCo 模型
        if model is None:
            if xml_path is None:
                xml_path = os.path.join(
                    os.path.dirname(__file__), "..", "envs", "assets",
                    "mujoco_weld_robot.xml"
                )
            try:
                self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(xml_path))
                self.data: mujoco.MjData = mujoco.MjData(self.model)
            except Exception as e:
                raise RuntimeError(f"Failed to load MuJoCo model: {e}")
        else:
            self.model = model
            if data is None:
                self.data = mujoco.MjData(self.model)
            else:
                self.data = data

        # 缓存 site/body ID
        self._wire_tip_id: int = self.model.site("wire_tip").id
        self._weld_gun_id: int = self.model.body("weld_gun").id

        # IK 参数
        self._max_ik_iterations: int = 100
        self._ik_tolerance_m: float = self.TRACKING_TOLERANCE / 1000.0  # mm → m
        self._damping: float = 0.1  # DLS 阻尼系数 λ

        # 关节限位
        self._joint_lower: np.ndarray = np.array([
            -3.14159, -1.5708, -2.3562, -3.14159, -1.5708, -3.14159
        ])
        self._joint_upper: np.ndarray = np.array([
            3.14159, 1.5708, 2.3562, 3.14159, 1.5708, 3.14159
        ])

    def solve_ik(self, tcp_target: np.ndarray) -> np.ndarray:
        """阻尼最小二乘法 (DLS) 逆运动学求解.

        使用迭代 DLS 方法求解使 TCP 到达目标位置的关节角度:
          θ_{k+1} = θ_k + J^T (J J^T + λ²I)^{-1} · e

        其中 e = target_pos - current_pos, J 是 3×6 位置雅可比.

        Args:
            tcp_target: 3D TCP 目标位置 (m).

        Returns:
            6 维关节角度向量 (rad).
        """
        tcp_target = np.asarray(tcp_target, dtype=np.float64).flatten()[:3]

        # 初始关节角度 (使用当前 data 中的值)
        joint_angles: np.ndarray = self.data.qpos[:6].copy()

        for iteration in range(self._max_ik_iterations):
            # 正运动学: 获取当前 TCP 位置
            current_tcp: np.ndarray = self._forward_kinematics(joint_angles)[:3]

            # 位置误差
            error: np.ndarray = tcp_target - current_tcp
            error_norm: float = float(np.linalg.norm(error))

            # 收敛判断
            if error_norm < self._ik_tolerance_m:
                break

            # 计算雅可比
            J: np.ndarray = self._jacobian(joint_angles)  # 3×6

            # DLS 求解
            delta_theta: np.ndarray = self._damped_least_squares(
                J, error, damping=self._damping
            )

            # 更新关节角度 (带步长限制)
            step_scale: float = min(1.0, 0.5 / max(np.linalg.norm(delta_theta), 1e-9))
            joint_angles = joint_angles + step_scale * delta_theta

            # 关节限位裁剪
            joint_angles = np.clip(joint_angles, self._joint_lower, self._joint_upper)

        return joint_angles

    def generate_trajectory(self, n_steps: int = 200) -> List[Tuple[np.ndarray, np.ndarray]]:
        """从 waypoints 生成关节空间轨迹.

        使用三次样条插值对 waypoints 进行平滑插值，然后均匀采样
        n_steps 个 TCP 目标点，对每个目标点调用 solve_ik 得到关节角度.

        Args:
            n_steps: 轨迹采样点数.

        Returns:
            轨迹列表, 每个元素为 (joint_angles(6,), tcp_target(3,)).
        """
        if len(self.waypoints) < 2:
            raise ValueError("Need at least 2 waypoints to generate trajectory")

        # 三次样条插值
        n_wp: int = len(self.waypoints)
        t_wp: np.ndarray = np.linspace(0.0, 1.0, n_wp)
        t_dense: np.ndarray = np.linspace(0.0, 1.0, n_steps)

        if _HAS_SCIPY:
            # 使用 scipy 三次样条
            splines = []
            for dim in range(3):
                cs = CubicSpline(t_wp, self.waypoints[:, dim])
                splines.append(cs(t_dense))
            tcp_targets: np.ndarray = np.stack(splines, axis=-1)
        else:
            # 回退: numpy 线性插值 + 平滑
            tcp_targets = np.zeros((n_steps, 3))
            for dim in range(3):
                tcp_targets[:, dim] = np.interp(t_dense, t_wp, self.waypoints[:, dim])
            # 简单平滑: 移动平均
            if n_steps > 5:
                kernel_size: int = 3
                for dim in range(3):
                    padded = np.pad(tcp_targets[:, dim], kernel_size // 2, mode='edge')
                    tcp_targets[:, dim] = np.convolve(
                        padded, np.ones(kernel_size) / kernel_size, mode='valid'
                    )[:n_steps]

        # 对每个 TCP 目标求解 IK
        trajectory: List[Tuple[np.ndarray, np.ndarray]] = []

        # 初始化: 重置到关键帧
        try:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass

        for i in range(n_steps):
            tcp_target: np.ndarray = tcp_targets[i]
            joint_angles: np.ndarray = self.solve_ik(tcp_target)
            trajectory.append((joint_angles.copy(), tcp_target.copy()))

        return trajectory

    def compute_tracking_error(
        self, tcp_actual: np.ndarray, tcp_target: np.ndarray
    ) -> float:
        """计算 TCP 跟踪误差 (mm).

        Args:
            tcp_actual: 实际 TCP 位置 (3D 或 6D).
            tcp_target: 目标 TCP 位置 (3D 或 6D).

        Returns:
            位置跟踪误差 (mm).
        """
        actual_pos: np.ndarray = np.asarray(tcp_actual, dtype=np.float64).flatten()[:3]
        target_pos: np.ndarray = np.asarray(tcp_target, dtype=np.float64).flatten()[:3]
        error_m: float = float(np.linalg.norm(actual_pos - target_pos))
        return error_m * 1000.0  # m → mm

    def _forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """正运动学: 关节角度 → TCP 位姿.

        设置关节角度到 MuJoCo data, 执行 mj_forward, 读取 wire_tip site 位置
        和 weld_gun body 四元数.

        Args:
            joint_angles: 6 维关节角度 (rad).

        Returns:
            TCP 位姿 [x, y, z, rx, ry, rz] (位置 m, 姿态 rad).
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64).flatten()[:6]

        # 保存原始状态
        original_qpos: np.ndarray = self.data.qpos[:6].copy()

        try:
            # 设置关节角度
            self.data.qpos[:6] = joint_angles
            # 前进物理
            mujoco.mj_forward(self.model, self.data)

            # 读取 TCP 位置 (wire_tip site)
            tcp_pos: np.ndarray = self.data.site_xpos[self._wire_tip_id].copy()

            # 读取 TCP 姿态 (weld_gun body 四元数 → 欧拉角)
            quat: np.ndarray = self.data.xquat[self._weld_gun_id].copy()
            euler: np.ndarray = self._quat_to_euler(quat)

            tcp_pose: np.ndarray = np.concatenate([tcp_pos, euler])
        except Exception:
            tcp_pose = np.zeros(6)
        finally:
            # 恢复原始状态
            self.data.qpos[:6] = original_qpos
            mujoco.mj_forward(self.model, self.data)

        return tcp_pose

    def _jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算 3×6 位置雅可比矩阵.

        使用 MuJoCo mj_jacBody 计算焊枪 body 的位置雅可比.
        如果 mj_jacBody 不可用, 回退到数值差分.

        Args:
            joint_angles: 6 维关节角度 (rad).

        Returns:
            3×6 位置雅可比矩阵 (∂pos/∂θ).
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64).flatten()[:6]

        # 尝试使用 MuJoCo 解析雅可比
        try:
            original_qpos: np.ndarray = self.data.qpos[:6].copy()
            self.data.qpos[:6] = joint_angles
            mujoco.mj_forward(self.model, self.data)

            jacp: np.ndarray = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, None, self._weld_gun_id)

            # 恢复状态
            self.data.qpos[:6] = original_qpos
            mujoco.mj_forward(self.model, self.data)

            # 取前6列 (6个关节)
            return jacp[:3, :6]
        except Exception:
            pass

        # 回退: 数值差分雅可比
        return self._numerical_jacobian(joint_angles)

    def _numerical_jacobian(
        self, joint_angles: np.ndarray, delta: float = 1e-6
    ) -> np.ndarray:
        """数值差分计算 3×6 位置雅可比.

        J[:, i] = (FK(θ + δ·e_i)[:3] - FK(θ - δ·e_i)[:3]) / (2δ)

        Args:
            joint_angles: 6 维关节角度 (rad).
            delta: 差分步长.

        Returns:
            3×6 位置雅可比矩阵.
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64).flatten()[:6]
        n_joints: int = 6
        jac: np.ndarray = np.zeros((3, n_joints))

        tcp_base: np.ndarray = self._forward_kinematics(joint_angles)[:3]

        for i in range(n_joints):
            theta_plus: np.ndarray = joint_angles.copy()
            theta_minus: np.ndarray = joint_angles.copy()
            theta_plus[i] += delta
            theta_minus[i] -= delta

            tcp_plus: np.ndarray = self._forward_kinematics(theta_plus)[:3]
            tcp_minus: np.ndarray = self._forward_kinematics(theta_minus)[:3]

            jac[:, i] = (tcp_plus - tcp_minus) / (2.0 * delta)

        return jac

    def _damped_least_squares(
        self, J: np.ndarray, error: np.ndarray, damping: float = 0.1
    ) -> np.ndarray:
        """阻尼最小二乘法 (DLS) 求解.

        θ_delta = J^T (J J^T + λ²I)^{-1} · error

        Args:
            J: 3×6 雅可比矩阵.
            error: 3D 位置误差向量.
            damping: 阻尼系数 λ.

        Returns:
            6 维关节角度增量.
        """
        J = np.asarray(J, dtype=np.float64)
        error = np.asarray(error, dtype=np.float64).flatten()[:3]

        lambda_sq: float = damping ** 2
        JJt: np.ndarray = J @ J.T  # 3×3
        damped: np.ndarray = JJt + lambda_sq * np.eye(3)

        try:
            damped_inv: np.ndarray = np.linalg.inv(damped)
        except np.linalg.LinAlgError:
            # 奇异矩阵: 使用伪逆
            damped_inv = np.linalg.pinv(damped)

        delta_theta: np.ndarray = J.T @ damped_inv @ error
        return delta_theta

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (ZYX 顺序).

        Args:
            quat: MuJoCo 四元数 [w, x, y, z].

        Returns:
            欧拉角 [rx, ry, rz] (rad).
        """
        w: float = float(quat[0])
        x: float = float(quat[1])
        y: float = float(quat[2])
        z: float = float(quat[3])

        sinr_cosp: float = 2.0 * (w * x + y * z)
        cosr_cosp: float = 1.0 - 2.0 * (x * x + y * y)
        rx: float = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp: float = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            ry: float = float(np.copysign(np.pi / 2.0, sinp))
        else:
            ry = float(np.arcsin(sinp))

        siny_cosp: float = 2.0 * (w * z + x * y)
        cosy_cosp: float = 1.0 - 2.0 * (y * y + z * z)
        rz: float = float(np.arctan2(siny_cosp, cosy_cosp))

        return np.array([rx, ry, rz])
