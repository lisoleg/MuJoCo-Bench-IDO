"""
WeldingSensorSuite — 完整焊接传感器套件
========================================
模拟真实焊接系统中的传感器网络，提供全面的实时监控数据。

v0.19.0: 从7类传感器扩展到16种完整传感器套件, 每种传感器用经验公式
从当前焊接参数计算读数, 添加高斯噪声模拟, 并保留最近50步历史数据
用于趋势分析。

传感器列表 (16种):
1. arc_voltage       — 电弧电压传感器 (V)
2. arc_current       — 电弧电流传感器 (A)
3. wire_feed_speed   — 送丝速度传感器 (m/min)
4. gas_flow          — 保护气体流量传感器 (L/min)
5. contact_tip_temp  — 导电嘴温度传感器 (°C)
6. weld_pool_width   — 熔池宽度视觉传感器 (mm)
7. weld_pool_length  — 熔池长度视觉传感器 (mm)
8. arc_sound         — 电弧声发射传感器 (dB)
9. ir_temp_1         — 红外温度传感器1 (焊缝中心, °C)
10. ir_temp_2        — 红外温度传感器2 (HAZ, °C)
11. ir_temp_3        — 红外温度传感器3 (母材, °C)
12. magnetic_arc_blow — 磁偏吹传感器 (mT)
13. pool_oscillation — 熔池振荡频率传感器 (Hz)
14. spatter_count    — 飞溅计数传感器 (particles/s)
15. seam_tracking    — 焊缝跟踪传感器 (mm偏差)
16. bead_profile     — 焊缝轮廓扫描传感器 (3D profile)

Author: MuJoCo-Bench-IDO Welding Module v0.20.0
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

try:
    import mujoco
    _HAS_MUJOCO: bool = True
except ImportError:
    _HAS_MUJOCO = False


@dataclass
class SensorConfig:
    """传感器配置数据类.

    Attributes:
        name: 传感器名称.
        unit: 测量单位.
        sample_rate_hz: 采样率 (Hz).
        noise_pct: 噪声百分比 (±%).
        description: 描述.
    """
    name: str
    unit: str
    sample_rate_hz: float
    noise_pct: float
    description: str


# 16种传感器配置
SENSOR_CONFIGS: Dict[str, SensorConfig] = {
    "arc_voltage":       SensorConfig("arc_voltage", "V", 50000, 0.5, "电弧电压传感器"),
    "arc_current":       SensorConfig("arc_current", "A", 50000, 1.0, "电弧电流传感器"),
    "wire_feed_speed":   SensorConfig("wire_feed_speed", "m/min", 500, 2.0, "送丝速度传感器"),
    "gas_flow":          SensorConfig("gas_flow", "L/min", 60, 3.0, "保护气体流量传感器"),
    "contact_tip_temp":  SensorConfig("contact_tip_temp", "°C", 60, 5.0, "导电嘴温度传感器"),
    "weld_pool_width":   SensorConfig("weld_pool_width", "mm", 120, 3.0, "熔池宽度视觉传感器"),
    "weld_pool_length":  SensorConfig("weld_pool_length", "mm", 120, 3.0, "熔池长度视觉传感器"),
    "arc_sound":         SensorConfig("arc_sound", "dB", 44100, 2.0, "电弧声发射传感器"),
    "ir_temp_1":         SensorConfig("ir_temp_1", "°C", 60, 3.0, "红外温度传感器1 (焊缝中心)"),
    "ir_temp_2":         SensorConfig("ir_temp_2", "°C", 60, 4.0, "红外温度传感器2 (HAZ)"),
    "ir_temp_3":         SensorConfig("ir_temp_3", "°C", 60, 5.0, "红外温度传感器3 (母材)"),
    "magnetic_arc_blow": SensorConfig("magnetic_arc_blow", "mT", 500, 5.0, "磁偏吹传感器"),
    "pool_oscillation":  SensorConfig("pool_oscillation", "Hz", 1000, 5.0, "熔池振荡频率传感器"),
    "spatter_count":     SensorConfig("spatter_count", "particles/s", 500, 5.0, "飞溅计数传感器"),
    "seam_tracking":     SensorConfig("seam_tracking", "mm", 500, 5.0, "焊缝跟踪传感器"),
    "bead_profile":      SensorConfig("bead_profile", "dict", 120, 2.0, "焊缝轮廓扫描传感器 (3D profile)"),
}

# 所有传感器名称列表 (有序)
SENSOR_NAMES: List[str] = list(SENSOR_CONFIGS.keys())

# 历史记录最大长度
HISTORY_MAX_LEN: int = 50


class WeldingSensorSuite:
    """16种完整焊接传感器套件.

    模拟真实焊接系统中的传感器网络，提供全面的实时监控数据。
    每个传感器用经验公式从当前焊接参数计算读数，添加高斯噪声模拟。
    保留最近50步数据用于趋势分析。

    Attributes:
        sensor_types: 启用的传感器类型列表.
        _history: 各传感器的历史数据 (deque, 最近50步).
        _step_count: 采样步数计数.
    """

    def __init__(
        self,
        sensor_types: Optional[List[str]] = None,
    ) -> None:
        """初始化焊接传感器套件.

        Args:
            sensor_types: 要启用的传感器列表, None=全部16种.
        """
        if sensor_types is None:
            self.sensor_types: List[str] = list(SENSOR_NAMES)
        else:
            self.sensor_types = [
                s for s in sensor_types if s in SENSOR_CONFIGS
            ]

        # 历史记录: 每个传感器一个 deque
        self._history: Dict[str, deque] = {
            s: deque(maxlen=HISTORY_MAX_LEN) for s in self.sensor_types
        }

        # 内部状态
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng(42)

    def read_all(
        self,
        model: Any = None,
        data: Any = None,
        env: Any = None,
    ) -> Dict[str, Any]:
        """读取所有启用的传感器数据.

        从焊接环境参数计算每个传感器的读数, 添加高斯噪声,
        并将读数存入历史记录。

        Args:
            model: MuJoCo MjModel (可选, 用于几何传感器).
            data: MuJoCo MjData (可选, 用于几何传感器).
            env: WeldingEnv 实例 (用于获取焊接参数和状态).

        Returns:
            传感器数据字典 {sensor_name: value}.
        """
        results: Dict[str, Any] = {}

        # 从 env 提取焊接参数
        current: float = 200.0
        voltage: float = 24.0
        speed: float = 6.0
        stickout: float = 15.0
        temperature: float = 25.0
        arc_stability: float = 0.9
        spatter_rate: float = 0.01
        bead_width: float = 8.0
        bead_height: float = 2.0
        bead_area: float = 20.0
        seam_deviation: float = 0.0

        if env is not None:
            current = float(getattr(env, "_last_current", 200.0))
            voltage = float(getattr(env, "_last_voltage", 24.0))
            temperature = float(getattr(env, "_temperature", 25.0))
            # Try to get more params from quality info
            try:
                from core.welding_process_proxy import WeldingProcessProxy
                proxy = getattr(env, "_full_proxy", None)
                if proxy is not None:
                    quality = proxy.predict(
                        current=current, voltage=voltage,
                        travel_speed=speed, stickout=stickout, weave=2.0,
                    )
                    arc_stability = quality.arc_stability
                    spatter_rate = quality.spatter_rate
                    bead_width = quality.bead_width
                    bead_height = quality.bead_height
                    bead_area = quality.bead_area
            except Exception:
                pass

            # Stickout from observation
            try:
                obs = env.get_observation()
                stickout = float(obs[12])
                speed = float(getattr(env, "_last_action", np.array([200, 24, 2, 6]))[3])
                seam_deviation = float(obs[17])
            except Exception:
                pass

        # 从 MuJoCo data 获取几何信息 (如果有)
        tcp_pose: np.ndarray = np.zeros(6)
        if _HAS_MUJOCO and model is not None and data is not None:
            try:
                wire_tip_id = model.site("wire_tip").id
                tcp_pose[0:3] = data.site_xpos[wire_tip_id].copy()
                weld_gun_id = model.body("weld_gun").id
                quat = data.xquat[weld_gun_id].copy()
                tcp_pose[3:6] = self._quat_to_euler(quat)
            except Exception:
                pass

        # 计算每个传感器读数
        for sensor_name in self.sensor_types:
            try:
                value = self._compute_sensor_reading(
                    sensor_name=sensor_name,
                    current=current,
                    voltage=voltage,
                    speed=speed,
                    stickout=stickout,
                    temperature=temperature,
                    arc_stability=arc_stability,
                    spatter_rate=spatter_rate,
                    bead_width=bead_width,
                    bead_height=bead_height,
                    bead_area=bead_area,
                    seam_deviation=seam_deviation,
                )
                results[sensor_name] = value
                # 存入历史
                # For dict values (bead_profile), store a scalar summary
                hist_val = value if not isinstance(value, dict) else value.get("area", 0.0)
                self._history[sensor_name].append(float(hist_val))
            except Exception:
                results[sensor_name] = None

        self._step_count += 1
        return results

    def _compute_sensor_reading(
        self,
        sensor_name: str,
        current: float,
        voltage: float,
        speed: float,
        stickout: float,
        temperature: float,
        arc_stability: float,
        spatter_rate: float,
        bead_width: float,
        bead_height: float,
        bead_area: float,
        seam_deviation: float,
    ) -> Any:
        """计算单个传感器的读数 (含高斯噪声).

        Args:
            sensor_name: 传感器名称.
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            stickout: 干伸长 (mm).
            temperature: 当前温度 (°C).
            arc_stability: 电弧稳定性 (0-1).
            spatter_rate: 飞溅率 (0-1).
            bead_width: 焊缝宽度 (mm).
            bead_height: 焊缝余高 (mm).
            bead_area: 焊缝截面积 (mm²).
            seam_deviation: 焊缝偏差 (mm).

        Returns:
            传感器读数 (float 或 dict).
        """
        config = SENSOR_CONFIGS[sensor_name]
        noise_pct = config.noise_pct / 100.0

        if sensor_name == "arc_voltage":
            # 电弧电压 = voltage ± noise(0.1V)
            base: float = voltage
            noise: float = self._rng.normal(0.0, 0.1)
            return float(max(0.0, base + noise))

        elif sensor_name == "arc_current":
            # 电弧电流 = current ± noise(2A)
            base = current
            noise = self._rng.normal(0.0, 2.0)
            return float(max(0.0, base + noise))

        elif sensor_name == "wire_feed_speed":
            # 送丝速度 = current * 0.04 ± noise(0.1)
            base = current * 0.04
            noise = self._rng.normal(0.0, 0.1)
            return float(max(0.0, base + noise))

        elif sensor_name == "gas_flow":
            # 保护气体流量 = 15.0 ± noise(0.5) L/min
            base = 15.0
            noise = self._rng.normal(0.0, 0.5)
            return float(max(0.0, base + noise))

        elif sensor_name == "contact_tip_temp":
            # 导电嘴温度 = 25 + current * 0.3 + (stickout < 10 ? 30 : 0) ± noise(5)
            base = 25.0 + current * 0.3
            if stickout < 10.0:
                base += 30.0
            noise = self._rng.normal(0.0, 5.0)
            return float(max(25.0, base + noise))

        elif sensor_name == "weld_pool_width":
            # 熔池宽度 = bead_width * 1.2 ± noise(0.2)
            base = bead_width * 1.2
            noise = self._rng.normal(0.0, 0.2)
            return float(max(0.0, base + noise))

        elif sensor_name == "weld_pool_length":
            # 熔池长度 = bead_width * 1.5 ± noise(0.3)
            base = bead_width * 1.5
            noise = self._rng.normal(0.0, 0.3)
            return float(max(0.0, base + noise))

        elif sensor_name == "arc_sound":
            # 电弧声 = 80 + current * 0.15 + (1-arc_stability) * 20 ± noise(2) dB
            base = 80.0 + current * 0.15 + (1.0 - max(0.0, min(1.0, arc_stability))) * 20.0
            noise = self._rng.normal(0.0, 2.0)
            return float(max(0.0, base + noise))

        elif sensor_name == "ir_temp_1":
            # 红外温度1 (焊缝中心) = temperature ± noise(10)
            base = temperature
            noise = self._rng.normal(0.0, 10.0)
            return float(max(25.0, base + noise))

        elif sensor_name == "ir_temp_2":
            # 红外温度2 (HAZ) = temperature * 0.7 ± noise(8)
            base = temperature * 0.7
            noise = self._rng.normal(0.0, 8.0)
            return float(max(25.0, base + noise))

        elif sensor_name == "ir_temp_3":
            # 红外温度3 (母材) = temperature * 0.3 ± noise(5)
            base = temperature * 0.3
            noise = self._rng.normal(0.0, 5.0)
            return float(max(25.0, base + noise))

        elif sensor_name == "magnetic_arc_blow":
            # 磁偏吹 = current * 0.002 * sin(step * 0.1) ± noise(0.01) mT
            base = current * 0.002 * np.sin(self._step_count * 0.1)
            noise = self._rng.normal(0.0, 0.01)
            return float(base + noise)

        elif sensor_name == "pool_oscillation":
            # 熔池振荡频率 = 10 + current * 0.05 ± noise(1) Hz
            base = 10.0 + current * 0.05
            noise = self._rng.normal(0.0, 1.0)
            return float(max(0.0, base + noise))

        elif sensor_name == "spatter_count":
            # 飞溅计数 = spatter_rate * 1000 ± noise(5) particles/s
            base = spatter_rate * 1000.0
            noise = self._rng.normal(0.0, 5.0)
            return float(max(0.0, base + noise))

        elif sensor_name == "seam_tracking":
            # 焊缝跟踪 = seam_deviation ± noise(0.1) mm
            base = seam_deviation
            noise = self._rng.normal(0.0, 0.1)
            return float(max(0.0, base + noise))

        elif sensor_name == "bead_profile":
            # 焊缝轮廓 = {width, height, area} with noise
            w_noise = self._rng.normal(0.0, bead_width * noise_pct)
            h_noise = self._rng.normal(0.0, bead_height * noise_pct)
            a_noise = self._rng.normal(0.0, bead_area * noise_pct)
            return {
                "width": float(max(0.0, bead_width + w_noise)),
                "height": float(max(0.0, bead_height + h_noise)),
                "area": float(max(0.0, bead_area + a_noise)),
            }

        return 0.0

    def get_sensor_names(self) -> List[str]:
        """返回所有启用的传感器名称列表.

        Returns:
            传感器名称列表.
        """
        return list(self.sensor_types)

    def get_history(
        self,
        sensor_name: str,
        n: int = 10,
    ) -> np.ndarray:
        """返回指定传感器最近n步的历史数据.

        Args:
            sensor_name: 传感器名称.
            n: 返回的历史步数 (默认10).

        Returns:
            numpy 数组, 包含最近n步的读数。如果传感器不存在或无数据, 返回空数组。
        """
        if sensor_name not in self._history:
            return np.array([], dtype=np.float64)
        hist = list(self._history[sensor_name])
        if n <= 0:
            return np.array([], dtype=np.float64)
        return np.array(hist[-n:], dtype=np.float64)

    def get_trend(self, sensor_name: str) -> str:
        """返回指定传感器的趋势分析.

        比较最近5步和前5步的平均值, 判断趋势方向。

        Args:
            sensor_name: 传感器名称.

        Returns:
            "rising" (上升), "falling" (下降), "stable" (稳定),
            或 "unknown" (数据不足).
        """
        if sensor_name not in self._history:
            return "unknown"
        hist = list(self._history[sensor_name])
        if len(hist) < 10:
            return "unknown"

        recent: float = float(np.mean(hist[-5:]))
        earlier: float = float(np.mean(hist[-10:-5]))
        diff: float = recent - earlier
        threshold: float = max(abs(earlier) * 0.02, 1e-6)

        if diff > threshold:
            return "rising"
        elif diff < -threshold:
            return "falling"
        else:
            return "stable"

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
