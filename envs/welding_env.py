"""
WeldingEnv — 焊接机器人 MuJoCo 仿真环境
================================================

六轴焊接机器人 + 双轴变位机的 MuJoCo 环境包装器。
直接使用 mujoco.MjModel / MjData（不依赖 dm_control），
复用 PinchLeafEnv 的 physics 管理模式。

动作空间 (4维):
  - current:  焊接电流 [50, 350] A
  - voltage:  焊接电压 [14, 32] V
  - weave:    摆动幅度 [0, 5] mm
  - speed:    焊接速度 [2, 15] mm/s

观测空间 (18维):
  [0:6]   TCP 位姿 (x, y, z, rx, ry, rz)
  [6:12]  关节角度 joint1-6
  [12]    干伸长 stickout (mm)
  [13:16] 接触力 contact_force (fx, fy, fz)
  [16]    温度 temperature (°C)
  [17]    焊缝偏差 seam_deviation (mm)

Author: MuJoCo-Bench-IDO Welding Module v0.1.0
"""

import os
import numpy as np
import mujoco
from typing import Dict, Any, Optional, List, Tuple

# 尝试导入安全锚和控制器（延迟导入避免循环依赖）
try:
    from agent.welding_psi_anchor import WeldingPsiAnchor
    _HAS_PSI_ANCHOR = True
except ImportError:
    _HAS_PSI_ANCHOR = False

# 尝试导入完整焊接工艺代理模型 (T03), 回退到内置stub
try:
    from core.welding_process_proxy import WeldingProcessProxy as _FullProcessProxy
    _HAS_FULL_PROXY = True
except ImportError:
    _HAS_FULL_PROXY = False

WELDING_ENV_VERSION: str = "v0.1.0"

# ── 环境常量 ──
WELD_TIMESTEP: float = 0.002        # 2ms = 500Hz
ACTION_DIM: int = 4                 # current, voltage, weave, speed
OBS_DIM: int = 18                   # TCP(6) + joints(6) + stickout(1) + force(3) + temp(1) + dev(1)

# ── 动作范围 ──
ACTION_LOW: np.ndarray = np.array([50.0, 14.0, 0.0, 2.0])
ACTION_HIGH: np.ndarray = np.array([350.0, 32.0, 5.0, 15.0])

# ── 焊接类型 → keyframe 名称映射 ──
WELD_TYPE_KEYFRAMES: Dict[str, str] = {
    "flat": "flat",
    "horizontal": "horizontal",
    "vertical": "vertical",
    "overhead": "overhead",
}

# ── 焊缝 waypoint 配置 ──
WAYPOINT_SPACING_MM: float = 2.0   # 焊缝上相邻 waypoint 间距
NUM_WAYPOINTS: int = 20            # 焊缝上的 waypoint 数量 (20×2mm=40mm, 3500步可跑完)

# ── 工件参数 (与 XML 一致) ──
WORKPIECE_CENTER: np.ndarray = np.array([1.0, 0.0, 0.265])  # 工件中心世界坐标
WORKPIECE_SEAM_LENGTH: float = 0.20  # 焊缝长度 0.2m = 200mm
WORKPIECE_SEAM_WIDTH: float = 0.005  # 焊缝宽度 5mm

# ── 奖励权重 ──
REWARD_ETA_WEIGHT: float = 10.0
REWARD_POROSITY_WEIGHT: float = 20.0
REWARD_DISTORTION_WEIGHT: float = 50.0


class WeldingProcessProxy:
    """焊接过程预测代理 (内置 stub 版本).

    如果 core.welding_process_proxy.WeldingProcessProxy 可用 (T03),
    WeldingEnv会优先使用完整版本。此类作为回退 stub。

    Attributes:
        stub_mode: 是否处于 stub 模式。
    """

    def __init__(self) -> None:
        """初始化焊接过程代理 (stub 模式)."""
        self.stub_mode: bool = True
        # 如果完整代理可用, 委托给它
        if _HAS_FULL_PROXY:
            self._full_proxy = _FullProcessProxy()
            self.stub_mode = False
        else:
            self._full_proxy = None

    def predict_quality(
        self,
        current: float,
        voltage: float,
        speed: float,
        stickout: float,
    ) -> Dict[str, float]:
        """预测焊接质量指标.

        如果完整代理模型可用, 委托给完整版本; 否则使用stub.

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            stickout: 干伸长 (mm).

        Returns:
            质量指标字典，包含:
              - eta: 焊缝成形效率 (0-1, 越高越差)
              - porosity: 气孔率 (0-1)
              - distortion: 变形量 (mm)
        """
        if self._full_proxy is not None:
            return self._full_proxy.predict_quality(current, voltage, speed, stickout)

        # Stub: 返回基于物理直觉的简单估计
        eta: float = max(0.0, min(1.0, abs(current - 200.0) / 200.0))
        porosity: float = max(0.0, min(1.0, abs(stickout - 15.0) / 25.0))
        distortion: float = max(0.0, min(2.0, 15.0 / max(speed, 1.0) * 0.1))
        return {
            "eta": eta,
            "porosity": porosity,
            "distortion": distortion,
        }


class WeldingEnv:
    """六轴焊接机器人 MuJoCo 仿真环境.

    封装 MuJoCo 物理引擎，提供焊接专用的 reset/step 接口，
    集成 WeldingPsiAnchor 安全门控和 WeldingProcessProxy 质量预测。

    Attributes:
        ACTION_DIM: 动作维度 (4).
        OBS_DIM: 观测维度 (18).
        WELD_TIMESTEP: 仿真步长 (0.002s).
        VERSION: 环境版本.
    """

    ACTION_DIM: int = ACTION_DIM
    OBS_DIM: int = OBS_DIM
    WELD_TIMESTEP: float = WELD_TIMESTEP
    VERSION: str = WELDING_ENV_VERSION

    def __init__(
        self,
        xml_path: Optional[str] = None,
        weld_type: str = "flat",
        random_seed: int = 42,
    ) -> None:
        """初始化焊接环境.

        Args:
            xml_path: MuJoCo XML 文件路径. 如果为 None, 使用默认路径.
            weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead").
            random_seed: 随机种子.
        """
        self._random_seed: int = random_seed
        np.random.seed(random_seed)

        # 焊接类型验证
        if weld_type not in WELD_TYPE_KEYFRAMES:
            raise ValueError(
                f"Invalid weld_type '{weld_type}'. "
                f"Must be one of {list(WELD_TYPE_KEYFRAMES.keys())}"
            )
        self.weld_type: str = weld_type

        # 加载 MuJoCo 模型
        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(__file__), "assets", "mujoco_weld_robot.xml"
            )

        try:
            self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(str(xml_path))
            self.data: mujoco.MjData = mujoco.MjData(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {xml_path}: {e}")

        # 设置仿真步长
        self.model.opt.timestep = WELD_TIMESTEP

        # 获取 site/body ID (缓存以加速)
        self._wire_tip_id: int = self.model.site("wire_tip").id
        self._weld_gun_id: int = self.model.body("weld_gun").id
        self._workpiece_id: int = self.model.body("workpiece").id

        # 获取传感器数据地址 (缓存)
        self._sensor_adr: Dict[str, int] = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            self._sensor_adr[name] = self.model.sensor_adr[i]

        # 焊接过程代理 (T01 stub)
        self.process_proxy: WeldingProcessProxy = WeldingProcessProxy()

        # 安全锚
        if _HAS_PSI_ANCHOR:
            self.psi_anchor: Optional[WeldingPsiAnchor] = WeldingPsiAnchor()
        else:
            self.psi_anchor = None

        # 焊缝 waypoints
        self.waypoints: np.ndarray = self._generate_waypoints()

        # 内部状态
        self._step_count: int = 0
        self._current_waypoint_idx: int = 0
        self._temperature: float = 25.0  # 初始温度 25°C
        self._last_current: float = 200.0  # 最近一次焊接电流
        self._last_voltage: float = 24.0  # 最近一次焊接电压
        self._last_heat_input: float = 0.0  # 最近一次热输入 (J/mm)

        # 传感器套件
        try:
            from core.welding_sensors import WeldingSensorSuite
            self.sensor_suite: Optional[Any] = WeldingSensorSuite()
        except ImportError:
            self.sensor_suite = None

        # keyframe 索引
        self._keyframe_map: Dict[str, int] = {}
        for i in range(self.model.nkey):
            kf_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            self._keyframe_map[kf_name] = i

    def _generate_waypoints(self) -> np.ndarray:
        """生成焊缝上的 TCP 目标 waypoint 序列.

        焊缝沿 X 轴方向，从工件一端到另一端，间距 2mm。

        Returns:
            Nx3 数组，每行为一个 waypoint 的 (x, y, z) 世界坐标 (单位: m).
        """
        waypoints: List[np.ndarray] = []
        seam_half: float = WORKPIECE_SEAM_LENGTH / 2.0
        spacing_m: float = WAYPOINT_SPACING_MM / 1000.0

        for i in range(NUM_WAYPOINTS):
            t: float = i / max(NUM_WAYPOINTS - 1, 1)
            x: float = WORKPIECE_CENTER[0] - seam_half + t * WORKPIECE_SEAM_LENGTH
            y: float = WORKPIECE_CENTER[1]
            z: float = WORKPIECE_CENTER[2] + 0.015  # TCP 在焊缝上方 15mm
            waypoints.append(np.array([x, y, z]))

        return np.array(waypoints)

    def reset(self) -> np.ndarray:
        """重置环境到焊接姿态对应的关键帧.

        Returns:
            18 维初始观测向量.
        """
        self._step_count = 0
        self._current_waypoint_idx = 0
        self._temperature = 25.0

        # 重置 MuJoCo 到关键帧
        kf_name: str = WELD_TYPE_KEYFRAMES.get(self.weld_type, "flat")
        kf_id: int = self._keyframe_map.get(kf_name, 0)
        mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)

        # 前进物理一步以稳定传感器读数
        mujoco.mj_forward(self.model, self.data)

        # 重置安全锚历史
        if self.psi_anchor is not None:
            self.psi_anchor._arc_length_history = []

        return self.get_observation()

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """执行一步焊接环境仿真.

        Args:
            action: 4 维数组 [current(A), voltage(V), weave(mm), speed(mm/s)].

        Returns:
            字典包含:
              - observation: 18 维观测向量.
              - reward: 奖励值 (负数, 惩罚制).
              - done: 是否结束.
              - info: 额外信息字典.
        """
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64).flatten()

        # 裁剪动作到有效范围
        action_clipped = np.clip(action, ACTION_LOW, ACTION_HIGH)
        current: float = float(action_clipped[0])
        voltage: float = float(action_clipped[1])
        weave: float = float(action_clipped[2])
        speed: float = float(action_clipped[3])

        # ① 读取传感器
        obs: np.ndarray = self.get_observation()
        stickout: float = float(obs[12])
        contact_force: np.ndarray = obs[13:16].copy()

        # ② 调用 WeldingProcessProxy 预测质量
        quality: Dict[str, float] = self.process_proxy.predict_quality(
            current=current, voltage=voltage, speed=speed, stickout=stickout
        )

        # ③ Ψ-Anchor 安全检查
        welding_state: Dict[str, Any] = {
            "stickout": stickout,
            "current": current,
            "voltage": voltage,
            "arc_length_variance": 0.1,  # stub: 固定低方差
            "seam_deviation": float(obs[17]),
            "contact_force": contact_force.tolist(),
            "temperature": self._temperature,
        }

        safety_result: Dict[str, Any] = {"passed": True, "violations": [], "actions": []}
        if self.psi_anchor is not None:
            safety_result = self.psi_anchor.check_all(welding_state)

        # ④ 推进 MuJoCo 物理 (如果安全检查通过)
        if safety_result["passed"]:
            # 驱动关节让 TCP 沿焊缝移动 + 摆动 + 电弧可视化
            self._advance_along_seam(speed, weave, current)
            mujoco.mj_step(self.model, self.data)
        else:
            # 安全违规: 仍推进物理但不移动TCP
            mujoco.mj_step(self.model, self.data)

        # ⑤ 更新温度 (简化热模型: 电流×电压 → 热量, 自然冷却)
        heat_input_j: float = current * voltage * WELD_TIMESTEP  # J
        cooling: float = (self._temperature - 25.0) * 0.001  # 自然冷却
        self._temperature += heat_input_j * 0.0001 - cooling
        self._temperature = max(25.0, min(self._temperature, 2000.0))

        # 存储最近焊接参数 (供传感器套件读取)
        self._last_current = current
        self._last_voltage = voltage
        # 热输入 (J/mm) = 电流×电压 / 速度
        self._last_heat_input = (current * voltage) / max(speed, 0.1)

        # ⑥ 计算奖励
        reward: float = self.compute_welding_reward(quality)

        # ⑦ 判断是否完成
        done: bool = self._current_waypoint_idx >= len(self.waypoints) - 1

        # ⑧ 组装 info
        info: Dict[str, Any] = {
            "quality": quality,
            "safety": safety_result,
            "weld_type": self.weld_type,
            "step_count": self._step_count,
            "waypoint_idx": self._current_waypoint_idx,
            "waypoint_total": len(self.waypoints),
            "weld_progress": self._current_waypoint_idx / max(len(self.waypoints) - 1, 1),
            "action_clipped": action_clipped.tolist(),
            "heat_input": self._last_heat_input,
            "current": current,
            "voltage": voltage,
            "weave": weave,
            "speed": speed,
        }

        # 获取更新后的观测
        new_obs: np.ndarray = self.get_observation()

        return {
            "observation": new_obs,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def _advance_along_seam(
        self,
        speed_mm_s: float,
        weave_mm: float = 0.0,
        current_a: float = 200.0,
    ) -> None:
        """根据焊接速度推进 TCP 沿焊缝方向移动.

        通过设置 position actuator 的 ctrl 目标来驱动关节运动,
        使 TCP 沿焊缝 X 轴方向移动, 同时叠加 Y 方向摆动(weave)。

        Args:
            speed_mm_s: 焊接速度 (mm/s).
            weave_mm: 摆动幅度 (mm).
            current_a: 焊接电流 (A), 用于电弧可视化.
        """
        # 每步移动距离 (m)
        step_distance_m: float = speed_mm_s / 1000.0 * WELD_TIMESTEP
        waypoint_spacing_m: float = WAYPOINT_SPACING_MM / 1000.0
        steps_per_waypoint: float = waypoint_spacing_m / max(step_distance_m, 1e-9)

        # 累积推进 waypoint 索引
        if self._step_count % max(int(steps_per_waypoint), 1) == 0:
            self._current_waypoint_idx = min(
                self._current_waypoint_idx + 1, len(self.waypoints) - 1
            )

        # ── 计算关节控制目标 ──
        # 基础姿态来自关键帧 (keyframe qpos, 不是 qpos0)
        kf_name: str = WELD_TYPE_KEYFRAMES.get(self.weld_type, "flat")
        kf_id: int = self._keyframe_map.get(kf_name, 0)
        if self.model.nkey > 0 and kf_id < self.model.nkey:
            base_qpos = self.model.key_qpos[kf_id][:8].copy()
        else:
            base_qpos = np.array([0.0, 0.5, -0.8, 0.0, 0.3, 0.0, 0.0, 0.0])

        # 焊缝进度 (0.0 → 1.0)
        progress: float = self._current_waypoint_idx / max(len(self.waypoints) - 1, 1)

        # joint2 (shoulder pitch): 从 0.5 → 0.65, 使手臂前倾延伸 TCP 沿 +X
        joint2_offset: float = 0.15 * progress
        # joint3 (elbow pitch): 从 1.35 → 1.20, 肘部微调保持 TCP 高度
        joint3_offset: float = -0.15 * progress
        # joint1 (base rotation): 叠加摆动
        weave_rad: float = (weave_mm / 1000.0) * 0.5  # mm → rad (近似)
        weave_offset: float = weave_rad * np.sin(
            2.0 * np.pi * 2.0 * self._step_count * WELD_TIMESTEP
        )

        # 设置 8 个 position actuator 的控制目标
        ctrl_targets = np.zeros(8, dtype=np.float64)
        ctrl_targets[0] = base_qpos[0] + weave_offset       # joint1 + weave
        ctrl_targets[1] = base_qpos[1] + joint2_offset       # joint2 + sweep
        ctrl_targets[2] = base_qpos[2] + joint3_offset       # joint3 + sweep
        ctrl_targets[3] = base_qpos[3]                        # joint4
        ctrl_targets[4] = base_qpos[4]                        # joint5
        ctrl_targets[5] = base_qpos[5]                        # joint6
        ctrl_targets[6] = base_qpos[6]                        # pos_rot_z
        ctrl_targets[7] = base_qpos[7]                        # pos_tilt_x

        # 裁剪到 actuator ctrlrange
        for i in range(min(8, self.model.nu)):
            lo = self.model.actuator_ctrlrange[i, 0]
            hi = self.model.actuator_ctrlrange[i, 1]
            ctrl_targets[i] = np.clip(ctrl_targets[i], lo, hi)

        self.data.ctrl[:8] = ctrl_targets

        # ── 电弧可视化: 根据电流大小调整 arc_cone 透明度 ──
        try:
            arc_geom_id = self.model.geom("arc_cone").id
            # 电流越大, 电弧越亮 (alpha 0.0→0.8)
            arc_alpha: float = min(0.8, current_a / 350.0 * 0.8)
            # 添加闪烁效果 (高频随机)
            flicker: float = 0.85 + 0.15 * np.sin(self._step_count * 0.5)
            self.model.geom_rgba[arc_geom_id] = np.array(
                [0.95, 0.25, 0.08, arc_alpha * flicker]
            )
        except Exception:
            pass

    def get_observation(self) -> np.ndarray:
        """获取 18 维观测向量.

        观测布局:
          [0:6]   TCP 位姿 (x, y, z, rx, ry, rz) — 位置 + 欧拉角
          [6:12]  关节角度 joint1-6 (rad)
          [12]    干伸长 stickout (mm)
          [13:16] 接触力 contact_force (fx, fy, fz) (N)
          [16]    温度 temperature (°C)
          [17]    焊缝偏差 seam_deviation (mm)

        Returns:
            18 维 float64 观测向量.
        """
        obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float64)

        # TCP 位置 (wire_tip site 世界坐标)
        tcp_pos: np.ndarray = self.data.site_xpos[self._wire_tip_id].copy()
        obs[0:3] = tcp_pos

        # TCP 姿态 (weld_gun body 四元数 → 欧拉角)
        weld_gun_quat: np.ndarray = self.data.xquat[self._weld_gun_id].copy()
        euler: np.ndarray = self._quat_to_euler(weld_gun_quat)
        obs[3:6] = euler

        # 关节角度 joint1-6
        obs[6:12] = self.data.qpos[:6].copy()

        # 干伸长 (mm)
        obs[12] = self._compute_stickout()

        # 接触力 (N)
        obs[13:16] = self._read_contact_force()

        # 温度 (°C)
        obs[16] = self._temperature

        # 焊缝偏差 (mm)
        obs[17] = self._compute_seam_deviation()

        return obs

    def compute_welding_reward(self, quality: Dict[str, float]) -> float:
        """计算焊接奖励 (惩罚制).

        reward = -eta*10 - porosity*20 - distortion*50 - stickout_penalty

        所有项都是惩罚，因此 reward ≤ 0。

        Args:
            quality: 质量指标字典 (eta, porosity, distortion).

        Returns:
            奖励值 (负数).
        """
        eta: float = float(quality.get("eta", 0.0))
        porosity: float = float(quality.get("porosity", 0.0))
        distortion: float = float(quality.get("distortion", 0.0))

        # 干伸长惩罚
        stickout: float = self._compute_stickout()
        stickout_penalty: float = 1.0 if (stickout > 25.0 or stickout < 8.0) else 0.0

        reward: float = (
            -REWARD_ETA_WEIGHT * eta
            - REWARD_POROSITY_WEIGHT * porosity
            - REWARD_DISTORTION_WEIGHT * distortion
            - stickout_penalty
        )

        return float(reward)

    def _compute_stickout(self) -> float:
        """计算干伸长 (mm).

        干伸长 = wire_tip 到工件表面的垂直距离 (Z方向)。
        工件表面 = workpiece body Z + 工件半厚度 (0.025m)。

        Returns:
            干伸长 (mm).
        """
        # 尝试从 distance 传感器读取
        dist_adr: Optional[int] = self._sensor_adr.get("stickout_dist")
        if dist_adr is not None and dist_adr < len(self.data.sensordata):
            dist_m: float = float(self.data.sensordata[dist_adr])
            return max(0.0, dist_m * 1000.0)

        # 几何计算: Z方向距离 (wire_tip Z - workpiece surface Z)
        tcp_pos: np.ndarray = self.data.site_xpos[self._wire_tip_id]
        wp_pos: np.ndarray = self.data.xpos[self._workpiece_id]
        # 工件表面 = workpiece center Z + 半厚度 (0.025m)
        wp_surface_z: float = float(wp_pos[2]) + 0.025
        stickout_m: float = float(tcp_pos[2]) - wp_surface_z
        return max(0.0, stickout_m * 1000.0)

    def _compute_seam_deviation(self) -> float:
        """计算 TCP 到焊缝的横向偏差 (mm).

        焊缝沿 X 轴方向, 偏差仅为 Y 方向 (横向偏移),
        不含 Z 方向 (高度由 stickout 单独管理)。

        Returns:
            焊缝横向偏差 (mm).
        """
        tcp_pos: np.ndarray = self.data.site_xpos[self._wire_tip_id]

        # 焊缝中心线: 沿X轴, y=WORKPIECE_CENTER[1]
        seam_y: float = WORKPIECE_CENTER[1]

        # 偏差 = Y方向距离 (横向偏移)
        dev_m: float = float(abs(tcp_pos[1] - seam_y))
        return dev_m * 1000.0

    def _read_contact_force(self) -> np.ndarray:
        """读取焊枪接触力 (N).

        从 contactforce 传感器读取 3 维力向量。
        如果传感器不可用，返回零向量。

        Returns:
            3 维接触力向量 [fx, fy, fz] (N).
        """
        cf_adr: Optional[int] = self._sensor_adr.get("cf_weld_gun")
        if cf_adr is not None and cf_adr + 2 < len(self.data.sensordata):
            return self.data.sensordata[cf_adr:cf_adr + 3].copy()

        # 回退: 检查接触列表
        try:
            if self.data.ncon > 0:
                force: np.ndarray = np.zeros(3)
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    if contact.geom1 == self._weld_gun_id or contact.geom2 == self._weld_gun_id:
                        force_tmp = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, i, force_tmp)
                        force[:3] += force_tmp[:3]
                return force
        except Exception:
            pass

        return np.zeros(3)

    def _check_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """调用 WeldingPsiAnchor 进行安全检查.

        Args:
            state: 焊接状态字典.

        Returns:
            安全检查结果字典.
        """
        if self.psi_anchor is None:
            return {"passed": True, "violations": [], "actions": []}
        return self.psi_anchor.check_all(state)

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (ZYX 顺序, 即 roll-pitch-yaw).

        Args:
            quat: MuJoCo 四元数 [w, x, y, z].

        Returns:
            欧拉角 [rx, ry, rz] (rad).
        """
        w: float = float(quat[0])
        x: float = float(quat[1])
        y: float = float(quat[2])
        z: float = float(quat[3])

        # Roll (x-axis rotation)
        sinr_cosp: float = 2.0 * (w * x + y * z)
        cosr_cosp: float = 1.0 - 2.0 * (x * x + y * y)
        rx: float = float(np.arctan2(sinr_cosp, cosr_cosp))

        # Pitch (y-axis rotation)
        sinp: float = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            ry: float = float(np.copysign(np.pi / 2.0, sinp))
        else:
            ry = float(np.arcsin(sinp))

        # Yaw (z-axis rotation)
        siny_cosp: float = 2.0 * (w * z + x * y)
        cosy_cosp: float = 1.0 - 2.0 * (y * y + z * z)
        rz: float = float(np.arctan2(siny_cosp, cosy_cosp))

        return np.array([rx, ry, rz])

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取所有传感器实时数据 (供 API 调用).

        Returns:
            传感器数据字典, 包含:
              - tcp_pose: [x, y, z, rx, ry, rz]
              - stickout_mm: 干伸长
              - joint_torques: [6] 关节力矩
              - contact_force: [fx, fy, fz]
              - temperature_C: 温度
              - arc_current_A: 电弧电流
              - seam_deviation_mm: 焊缝偏差
              - joint_angles: [6] 关节角度
              - joint_velocities: [6] 关节速度
              - actuator_forces: [6] 执行器力
              - weld_progress: 焊接进度 0-1
              - heat_input: 热输入 J/mm
        """
        obs: np.ndarray = self.get_observation()

        # 从 MuJoCo 传感器读取
        joint_angles = np.zeros(6)
        joint_velocities = np.zeros(6)
        actuator_forces = np.zeros(6)

        # 直接从 data 读取
        joint_angles[:6] = self.data.qpos[:6].copy()
        if self.model.nv >= 6:
            joint_velocities[:6] = self.data.qvel[:6].copy()
        if self.model.nu >= 6:
            actuator_forces[:6] = self.data.actuator_force[:6].copy()

        # 如果有传感器套件, 用它读取
        sensor_readings: Dict[str, Any] = {}
        if self.sensor_suite is not None:
            try:
                sensor_readings = self.sensor_suite.read_all(
                    self.model, self.data, self
                )
            except Exception:
                pass

        return {
            "tcp_pose": obs[0:6].tolist(),
            "tcp_position_mm": (obs[0:3] * 1000.0).tolist(),
            "tcp_euler_deg": np.degrees(obs[3:6]).tolist(),
            "stickout_mm": float(obs[12]),
            "contact_force_N": obs[13:16].tolist(),
            "temperature_C": float(obs[16]),
            "arc_current_A": self._last_current,
            "arc_voltage_V": self._last_voltage,
            "seam_deviation_mm": float(obs[17]),
            "joint_angles_rad": joint_angles.tolist(),
            "joint_angles_deg": np.degrees(joint_angles).tolist(),
            "joint_velocities_rad_s": joint_velocities.tolist(),
            "actuator_forces_N": actuator_forces.tolist(),
            "weld_progress": float(
                self._current_waypoint_idx / max(len(self.waypoints) - 1, 1)
            ),
            "waypoint_idx": self._current_waypoint_idx,
            "waypoint_total": len(self.waypoints),
            "heat_input_J_mm": self._last_heat_input,
            "sensor_suite": sensor_readings if sensor_readings else {},
        }

    def render_camera(
        self,
        cam_name: str = "overview_cam",
        width: int = 640,
        height: int = 480,
    ) -> Optional[bytes]:
        """渲染 MuJoCo 相机为 JPEG bytes.

        Args:
            cam_name: 相机名称 ("overview_cam" 或 "weld_cam").
            width: 图像宽度.
            height: 图像高度.

        Returns:
            JPEG bytes, 如果渲染失败返回 None.
        """
        try:
            cam_id = self.model.camera(cam_name).id
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data, camera=cam_id)
            pixels = renderer.render()
            # numpy RGB → JPEG
            import cv2
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return buf.tobytes()
        except Exception:
            pass

        # 回退: 用 PIL
        try:
            from PIL import Image
            import io
            cam_id = self.model.camera(cam_name).id
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data, camera=cam_id)
            pixels = renderer.render()
            img = Image.fromarray(pixels)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            return None

    @property
    def action_spec(self) -> Dict[str, Any]:
        """返回动作空间规格.

        Returns:
            字典包含 shape, low, high.
        """
        return {
            "shape": (ACTION_DIM,),
            "low": ACTION_LOW.copy(),
            "high": ACTION_HIGH.copy(),
        }

    @property
    def observation_spec(self) -> Dict[str, Any]:
        """返回观测空间规格.

        Returns:
            字典包含 shape.
        """
        return {"shape": (OBS_DIM,)}

    @property
    def physics(self) -> mujoco.MjData:
        """访问底层 MuJoCo data (兼容 PinchLeafEnv.physics 模式)."""
        return self.data
