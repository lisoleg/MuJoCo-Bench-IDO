"""
WeldingSensorSuite — 7类多模态焊接传感器仿真
=============================================

支持7类焊接专用传感器, 实现自适应采样率:
当GaussEx残差增大时自动提高采样率。

传感器列表:
  1. tcp_pose       — TCP位姿 (500Hz)
  2. stickout       — 干伸长距离 (500Hz)
  3. joint_torque   — 关节力矩 (500Hz)
  4. contact_force  — 接触力 (500Hz)
  5. temperature    — 红外温度 (60Hz)
  6. arc_current    — 霍尔电流 (50kHz)
  7. seam_deviation — 焊缝偏差 (500Hz)

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


@dataclass
class SensorConfig:
    """传感器配置数据类.

    Attributes:
        name: 传感器名称.
        sample_rate_hz: 采样率.
        sensor_type: 传感器类型
            ("framepos", "framequat", "distance", "force", "custom").
        description: 描述.
    """
    name: str
    sample_rate_hz: float
    sensor_type: str
    description: str


# 7类传感器配置
SENSOR_CONFIGS: Dict[str, SensorConfig] = {
    "tcp_pose":       SensorConfig("tcp_pose", 500, "framepos", "TCP位姿传感器 (500Hz)"),
    "stickout":       SensorConfig("stickout", 500, "distance", "干伸长距离传感器 (500Hz)"),
    "joint_torque":   SensorConfig("joint_torque", 500, "joint", "关节力矩传感器 (500Hz)"),
    "contact_force":  SensorConfig("contact_force", 500, "force", "接触力传感器 (500Hz)"),
    "temperature":    SensorConfig("temperature", 60, "custom", "红外温度传感器 (60Hz)"),
    "arc_current":    SensorConfig("arc_current", 50000, "custom", "霍尔电流传感器 (50kHz)"),
    "seam_deviation": SensorConfig("seam_deviation", 500, "custom", "焊缝偏差传感器 (500Hz)"),
}


class WeldingSensorSuite:
    """7类多模态焊接传感器仿真套件.

    支持自适应采样率: 当GaussEx残差增大时自动提高采样率,
    当残差较小时降低采样率以节能。

    Attributes:
        sensor_types: 启用的传感器类型列表.
        adaptive_sampling: 是否启用自适应采样率.
        sample_rate_multipliers: 各传感器的采样率倍数.
    """

    def __init__(
        self,
        sensor_types: Optional[List[str]] = None,
        adaptive_sampling: bool = True,
    ) -> None:
        """初始化焊接传感器套件.

        Args:
            sensor_types: 要启用的传感器列表, None=全部.
            adaptive_sampling: 是否启用自适应采样率.
        """
        if sensor_types is None:
            self.sensor_types: List[str] = list(SENSOR_CONFIGS.keys())
        else:
            self.sensor_types = [
                s for s in sensor_types if s in SENSOR_CONFIGS
            ]

        self.adaptive_sampling: bool = adaptive_sampling
        self.sample_rate_multipliers: Dict[str, float] = {
            s: 1.0 for s in self.sensor_types
        }

        # 内部状态
        self._current_eta: float = 0.0
        self._step_count: int = 0
        self._last_temperature: float = 25.0
        self._last_arc_current: float = 0.0

    def read_all(self, model, data, env: Optional[Any] = None) -> Dict[str, Any]:
        """读取所有启用的传感器数据.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            env: 可选的 WeldingEnv 实例 (用于获取额外状态).

        Returns:
            传感器数据字典 {sensor_name: value}.
        """
        results: Dict[str, Any] = {}

        if not _HAS_MUJOCO:
            return results

        # 获取 site/body ID
        wire_tip_id: int = 0
        workpiece_id: int = 0
        weld_gun_id: int = 0
        try:
            wire_tip_id = model.site("wire_tip").id
            workpiece_id = model.body("workpiece").id
            weld_gun_id = model.body("weld_gun").id
        except Exception:
            pass

        for sensor_type in self.sensor_types:
            try:
                if sensor_type == "tcp_pose":
                    results[sensor_type] = self.read_tcp_pose(model, data)
                elif sensor_type == "stickout":
                    results[sensor_type] = self.read_stickout(
                        model, data, wire_tip_id, workpiece_id
                    )
                elif sensor_type == "joint_torque":
                    results[sensor_type] = self.read_joint_torques(model, data)
                elif sensor_type == "contact_force":
                    results[sensor_type] = self.read_contact_force(
                        model, data, weld_gun_id
                    )
                elif sensor_type == "temperature":
                    temp: float = self._last_temperature
                    if env is not None:
                        temp = float(getattr(env, "_temperature", 25.0))
                    results[sensor_type] = self.read_temperature(temp)
                elif sensor_type == "arc_current":
                    current_val: float = self._last_arc_current
                    if env is not None:
                        # 从 info 中获取当前电流
                        current_val = float(getattr(env, "_last_current", 200.0))
                    results[sensor_type] = self.read_arc_current(current_val)
                elif sensor_type == "seam_deviation":
                    results[sensor_type] = self.read_seam_deviation(
                        model, data, wire_tip_id,
                        np.array([1.0, 0.0, 0.265]), 0.28
                    )
            except Exception:
                results[sensor_type] = None

        self._step_count += 1
        return results

    def read_tcp_pose(self, model, data) -> np.ndarray:
        """读取TCP位姿 [x, y, z, rx, ry, rz].

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.

        Returns:
            6维 TCP 位姿向量 [位置(3), 欧拉角(3)].
        """
        pose: np.ndarray = np.zeros(6, dtype=np.float64)
        try:
            wire_tip_id: int = model.site("wire_tip").id
            pose[0:3] = data.site_xpos[wire_tip_id].copy()

            weld_gun_id: int = model.body("weld_gun").id
            quat: np.ndarray = data.xquat[weld_gun_id].copy()
            pose[3:6] = self._quat_to_euler(quat)
        except Exception:
            pass
        return pose

    def read_stickout(
        self,
        model,
        data,
        wire_tip_id: int,
        workpiece_id: int,
    ) -> float:
        """读取干伸长.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            wire_tip_id: wire_tip site ID.
            workpiece_id: workpiece body ID.

        Returns:
            干伸长.
        """
        try:
            tcp_pos: np.ndarray = data.site_xpos[wire_tip_id]
            wp_pos: np.ndarray = data.xpos[workpiece_id]
            dist_m: float = float(np.linalg.norm(tcp_pos - wp_pos))
            return max(0.0, dist_m * 1000.0)
        except Exception:
            return 15.0

    def read_joint_torques(self, model, data) -> np.ndarray:
        """读取6维关节力矩.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.

        Returns:
            6维关节力矩向量.
        """
        torques: np.ndarray = np.zeros(6, dtype=np.float64)
        try:
            n_joints: int = min(6, model.nv)
            if hasattr(data, "qfrc_actuator"):
                torques[:n_joints] = data.qfrc_actuator[:n_joints]
        except Exception:
            pass
        return torques

    def read_contact_force(self, model, data, body_id: int) -> np.ndarray:
        """读取3维接触力.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            body_id: 焊枪 body ID.

        Returns:
            3维接触力向量 [fx, fy, fz] (N).
        """
        force: np.ndarray = np.zeros(3, dtype=np.float64)
        try:
            if data.ncon > 0 and _HAS_MUJOCO:
                for i in range(data.ncon):
                    contact = data.contact[i]
                    if (contact.geom1 == body_id or contact.geom2 == body_id):
                        force_tmp: np.ndarray = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force_tmp)
                        force[:3] += force_tmp[:3]
        except Exception:
            pass
        return force

    def read_temperature(self, env_temp: Optional[float] = None) -> float:
        """读取温度.

        Args:
            env_temp: 环境温度 (°C), None=使用内部状态.

        Returns:
            温度 (°C).
        """
        if env_temp is not None:
            self._last_temperature = float(env_temp)
        return self._last_temperature

    def read_arc_current(self, action_current: Optional[float] = None) -> float:
        """读取电弧电流.

        Args:
            action_current: 动作中的电流值 (A), None=使用内部状态.

        Returns:
            电弧电流 (A).
        """
        if action_current is not None:
            self._last_arc_current = float(action_current)
        return self._last_arc_current

    def read_seam_deviation(
        self,
        model,
        data,
        wire_tip_id: int,
        seam_center: np.ndarray,
        seam_z: float,
    ) -> float:
        """读取焊缝偏差.

        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData.
            wire_tip_id: wire_tip site ID.
            seam_center: 焊缝中心坐标 (3D).
            seam_z: 焊缝表面高度.

        Returns:
            焊缝偏差.
        """
        try:
            tcp_pos: np.ndarray = data.site_xpos[wire_tip_id]
            # 偏差 = YZ 平面距离
            dev_m: float = float(np.sqrt(
                (tcp_pos[1] - seam_center[1]) ** 2
                + (tcp_pos[2] - seam_z) ** 2
            ))
            return dev_m * 1000.0
        except Exception:
            return 0.0

    def adapt_sample_rate(self, eta_residual: float) -> None:
        """根据η残差自适应调整采样率.

        eta > 0.5  → 2x 采样率
        eta > 1.0  → 4x 采样率
        eta < 0.1  → 0.5x 采样率 (节能)

        Args:
            eta_residual: 当前 GaussEx 残差 η.
        """
        self._current_eta = float(eta_residual)

        if not self.adaptive_sampling:
            return

        if eta_residual > 1.0:
            multiplier: float = 4.0
        elif eta_residual > 0.5:
            multiplier = 2.0
        elif eta_residual < 0.1:
            multiplier = 0.5
        else:
            multiplier = 1.0

        for sensor_type in self.sensor_types:
            self.sample_rate_multipliers[sensor_type] = multiplier

    def get_effective_sample_rates(self) -> Dict[str, float]:
        """获取各传感器当前有效采样率.

        Returns:
            {sensor_name: effective_sample_rate_hz}.
        """
        rates: Dict[str, float] = {}
        for sensor_type in self.sensor_types:
            config: SensorConfig = SENSOR_CONFIGS[sensor_type]
            mult: float = self.sample_rate_multipliers[sensor_type]
            rates[sensor_type] = config.sample_rate_hz * mult
        return rates

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
