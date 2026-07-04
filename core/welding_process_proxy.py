"""
WeldingProcessProxy — 焊接工艺代理模型
======================================

用经验公式近似焊接物理，不模拟流体力学/MHD。
替换 welding_env.py 中的 stub 版本。

输入: 焊接参数 (current, voltage, speed, stickout, weave, weld_type)
输出: WeldingQuality (eta_residual, porosity_risk, angular_distortion,
       penetration_depth, arc_length, heat_input)

经验公式来源:
  - arc_length ≈ voltage - 14 (mm)
  - heat_input = (I × V) / (v × 1000) (kJ/mm)
  - penetration = k × sqrt(I × V / v) (mm)
  - porosity = base × (1 + gas_coverage_penalty + arc_stability_penalty)
  - angular_distortion = heat_input × material_factor × constraint

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class WeldingParams:
    """焊接参数数据类.

    Attributes:
        current: 焊接电流 (A).
        voltage: 焊接电压 (V).
        travel_speed: 焊接速度 (mm/s).
        wire_feed: 送丝速度 (m/min).
        stickout: 干伸长 (mm).
        weave: 摆动幅度 (mm).
        weld_type: 焊接姿态类型.
    """
    current: float = 200.0
    voltage: float = 24.0
    travel_speed: float = 6.0
    wire_feed: float = 8.0
    stickout: float = 15.0
    weave: float = 2.0
    weld_type: str = "flat"


@dataclass
class WeldingQuality:
    """焊接质量预测结果数据类.

    Attributes:
        eta_residual: 综合工艺偏差 (无量纲, 越低越好).
        porosity_risk: 气孔风险概率 (0-1).
        angular_distortion: 角变形量 (degrees).
        penetration_depth: 熔深 (mm).
        arc_length: 电弧长度 (mm).
        heat_input: 热输入 (kJ/mm).
    """
    eta_residual: float = 0.0
    porosity_risk: float = 0.0
    angular_distortion: float = 0.0
    penetration_depth: float = 0.0
    arc_length: float = 0.0
    heat_input: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WeldingQuality(eta={self.eta_residual:.4f}, "
            f"porosity={self.porosity_risk:.4f}, "
            f"distortion={self.angular_distortion:.4f}°, "
            f"penetration={self.penetration_depth:.4f}mm, "
            f"arc={self.arc_length:.4f}mm, "
            f"heat={self.heat_input:.4f}kJ/mm)"
        )

    def to_dict(self) -> dict:
        """转换为字典格式 (兼容 WeldingEnv.predict_quality 接口).

        Returns:
            包含 eta, porosity, distortion 键的字典.
        """
        return {
            "eta": self.eta_residual,
            "porosity": self.porosity_risk,
            "distortion": self.angular_distortion,
            "penetration": self.penetration_depth,
            "arc_length": self.arc_length,
            "heat_input": self.heat_input,
        }


class WeldingProcessProxy:
    """焊接工艺代理模型 — 用经验公式近似焊接物理.

    不模拟流体力学/MHD, 用经验公式快速预测焊接质量指标。
    输入焊接参数, 输出 WeldingQuality dataclass。

    经验公式:
      - arc_length = voltage - 14 (mm)
      - heat_input = (current × voltage) / (travel_speed × 1000) (kJ/mm)
      - eta_residual = sqrt(sum(((param - optimal) / range)²)) × 0.5
      - porosity = base × (1 + (1-gas_coverage)×2 + (1-arc_stability)×3)
      - angular_distortion = heat_input × material_factor × constraint
      - penetration = k × sqrt(current × voltage / travel_speed) (mm)

    Attributes:
        weld_type: 焊接姿态类型.
        EMPIRICAL_COEFFS: 经验系数表.
    """

    # 经验系数表
    EMPIRICAL_COEFFS: dict = {
        "arc_length_factor": 14.0,            # arc_length = voltage - 14 (近似)
        "heat_input_unit": 1000.0,            # heat_input = (I*V)/(v*1000) kJ/mm
        "penetration_coeff": 0.08,            # penetration = k * sqrt(I*V/v)
        "porosity_base": 0.02,                # 基础气孔率
        "distortion_material_factor": 1.2e-4, # 材料因子
        "distortion_constraint": 0.8,         # 约束因子
    }

    # 最优参数 (用于 eta_residual 计算)
    OPTIMAL_PARAMS: dict = {
        "current": 200.0,
        "voltage": 24.0,
        "travel_speed": 6.0,
        "stickout": 15.0,
    }

    # 参数范围 (用于归一化)
    PARAM_RANGES: dict = {
        "current": 300.0,       # 50-350
        "voltage": 18.0,        # 14-32
        "travel_speed": 13.0,   # 2-15
        "stickout": 17.0,       # 8-25
    }

    def __init__(self, weld_type: str = "flat") -> None:
        """初始化焊接工艺代理模型.

        Args:
            weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead").
        """
        self.weld_type: str = weld_type
        self._current_history: list[float] = []
        self.stub_mode: bool = False  # 完整版, 非 stub

    def predict(
        self,
        current: float,
        voltage: float,
        travel_speed: float,
        wire_feed: float = 8.0,
        stickout: float = 15.0,
        weave: float = 2.0,
    ) -> WeldingQuality:
        """主预测函数 — 返回 WeldingQuality dataclass.

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            travel_speed: 焊接速度 (mm/s).
            wire_feed: 送丝速度 (m/min).
            stickout: 干伸长 (mm).
            weave: 摆动幅度 (mm).

        Returns:
            WeldingQuality 实例.
        """
        # 更新电流历史 (用于电弧稳定性计算)
        self.update_current_history(current)
        current_variance: float = self.compute_current_variance()

        # 1. 电弧长度
        arc_length: float = self.compute_arc_length(voltage)

        # 2. 热输入
        heat_input: float = self.compute_heat_input(current, voltage, travel_speed)

        # 3. 综合工艺偏差 eta_residual
        # optimal: current=200, voltage=24, speed=6, stickout=15
        eta: float = self._compute_eta_residual(
            current, voltage, travel_speed, stickout
        )

        # 4. 气孔风险
        # 电弧稳定性 = 1 / (1 + arc_length_variance)
        arc_stability: float = 1.0 / (1.0 + current_variance / 100.0)
        # 气体保护覆盖率: 干伸长越长保护越差
        gas_coverage: float = 1.0 - max(0.0, (stickout - 15.0) / 10.0)
        gas_coverage = max(0.0, min(1.0, gas_coverage))

        porosity: float = self.compute_porosity(
            stickout, current_variance / 10.0, gas_flow=15.0
        )
        # 考虑电弧稳定性的额外气孔风险
        porosity += self.EMPIRICAL_COEFFS["porosity_base"] * (1.0 - arc_stability) * 2.0
        porosity = max(0.0, min(1.0, porosity))

        # 5. 角变形
        distortion: float = self.compute_distortion(heat_input)

        # 6. 熔深
        penetration: float = self.compute_penetration(current, voltage, travel_speed)

        return WeldingQuality(
            eta_residual=eta,
            porosity_risk=porosity,
            angular_distortion=distortion,
            penetration_depth=penetration,
            arc_length=arc_length,
            heat_input=heat_input,
        )

    def predict_quality(
        self,
        current: float,
        voltage: float,
        speed: float,
        stickout: float,
    ) -> dict:
        """预测焊接质量指标 (兼容 WeldingEnv 旧接口).

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            stickout: 干伸长 (mm).

        Returns:
            质量指标字典 {eta, porosity, distortion}.
        """
        quality: WeldingQuality = self.predict(
            current=current,
            voltage=voltage,
            travel_speed=speed,
            stickout=stickout,
        )
        return quality.to_dict()

    def _compute_eta_residual(
        self,
        current: float,
        voltage: float,
        travel_speed: float,
        stickout: float,
    ) -> float:
        """计算综合工艺偏差 eta_residual.

        eta = sqrt(sum(((param - optimal) / range)²)) × 0.5

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            travel_speed: 焊接速度 (mm/s).
            stickout: 干伸长.

        Returns:
            综合偏差值 (越低越好).
        """
        params: dict = {
            "current": current,
            "voltage": voltage,
            "travel_speed": travel_speed,
            "stickout": stickout,
        }

        sum_sq: float = 0.0
        for key, value in params.items():
            optimal: float = self.OPTIMAL_PARAMS[key]
            rng: float = self.PARAM_RANGES[key]
            normalized_dev: float = (value - optimal) / max(rng, 1e-9)
            sum_sq += normalized_dev ** 2

        eta: float = float(np.sqrt(sum_sq) * 0.5)
        return max(0.0, eta)

    def compute_heat_input(self, current: float, voltage: float, speed: float) -> float:
        """计算热输入.

        heat_input = (current × voltage) / (speed × 1000)  # kJ/mm

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).

        Returns:
            热输入 (kJ/mm).
        """
        if speed < 1e-9:
            return 0.0
        unit: float = self.EMPIRICAL_COEFFS["heat_input_unit"]
        return float((current * voltage) / (speed * unit))

    def compute_arc_length(self, voltage: float) -> float:
        """计算电弧长度.

        arc_length = voltage - 14  # mm (近似)

        Args:
            voltage: 焊接电压 (V).

        Returns:
            电弧长度 (mm).
        """
        factor: float = self.EMPIRICAL_COEFFS["arc_length_factor"]
        return float(max(0.0, voltage - factor))

    def compute_porosity(
        self,
        stickout: float,
        arc_variance: float,
        gas_flow: float = 15.0,
    ) -> float:
        """计算气孔率.

        porosity = base × (1 + (1-gas_coverage)×2 + (1-arc_stability)×3)

        Args:
            stickout: 干伸长.
            arc_variance: 电弧长度方差.
            gas_flow: 保护气体流量 (L/min).

        Returns:
            气孔率 (0-1).
        """
        base: float = self.EMPIRICAL_COEFFS["porosity_base"]

        # 气体保护覆盖率: 干伸长越长保护越差
        gas_coverage: float = 1.0 - max(0.0, (stickout - 15.0) / 10.0)
        gas_coverage = max(0.0, min(1.0, gas_coverage))

        # 气体流量影响: 流量不足时保护变差
        if gas_flow < 10.0:
            gas_coverage *= gas_flow / 10.0

        # 电弧稳定性: 方差越大越不稳定
        arc_stability: float = 1.0 / (1.0 + arc_variance)

        porosity: float = base * (1.0 + (1.0 - gas_coverage) * 2.0
                                  + (1.0 - arc_stability) * 3.0)
        return float(max(0.0, min(1.0, porosity)))

    def compute_distortion(
        self,
        heat_input: float,
        material_factor: float = 1.2e-4,
        constraint: float = 0.8,
    ) -> float:
        """计算角变形.

        angular_distortion = heat_input × material_factor × constraint  # degrees

        Args:
            heat_input: 热输入 (kJ/mm).
            material_factor: 材料因子 (默认低碳钢).
            constraint: 约束因子 (0-1, 越大约束越强).

        Returns:
            角变形量 (degrees).
        """
        return float(heat_input * material_factor * constraint * 1000.0)

    def compute_penetration(
        self,
        current: float,
        voltage: float,
        speed: float,
    ) -> float:
        """计算熔深.

        penetration = k × sqrt(current × voltage / speed)  # mm

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).

        Returns:
            熔深.
        """
        if speed < 1e-9:
            return 0.0
        k: float = self.EMPIRICAL_COEFFS["penetration_coeff"]
        return float(k * np.sqrt(current * voltage / speed))

    def update_current_history(self, current: float) -> None:
        """更新电流历史, 用于方差计算.

        Args:
            current: 焊接电流 (A).
        """
        self._current_history.append(float(current))
        if len(self._current_history) > 50:
            self._current_history = self._current_history[-50:]

    def compute_current_variance(self) -> float:
        """返回最近电流历史的方差.

        Returns:
            电流方差 (A²). 如果历史不足 2 个, 返回 0.
        """
        if len(self._current_history) < 2:
            return 0.0
        return float(np.var(self._current_history))

    # ═══════════════════════════════════════════════════════════════════
    # v0.3.0: 章锋2026-07-04论文焊接物理公式扩展
    # ═══════════════════════════════════════════════════════════════════

    # ── κ-Phase: Welding Physics Detail ──

    #: 电流系数 k_I — 用于目标熔深公式
    K_I_COEFF: float = 0.085

    #: 名义电压基准 — 厚度≤3mm时
    V_NOM_BASE: float = 16.0

    #: 名义电压增量 — 厚度>3mm时
    V_NOM_INCREMENT: float = 2.0

    #: 厚度阈值 (mm) — 影响名义电压
    THICKNESS_THRESHOLD: float = 3.0

    def evaluate_detailed(
        self,
        I: float,
        V: float,
        v_mms: float,
        t_mm: float,
        stick_out: float,
    ) -> Tuple[float, float, float]:
        """详细焊接评估 — 返回目标熔深、实际熔深和偏差.

        章锋论文公式:
          target_pen = k_I × I² / (v × t)
          actual_pen = k × sqrt(I × V / v)  (已有方法)
          deviation = |actual_pen - target_pen| / target_pen

        Args:
            I: 焊接电流 (A).
            V: 焊接电压 (V).
            v_mms: 焊接速度 (mm/s).
            t_mm: 板厚 (mm).
            stick_out: 干伸长 (mm).

        Returns:
            Tuple[float, float, float]:
              (target_penetration, actual_penetration, deviation_ratio)
              deviation_ratio = |actual - target| / max(target, 1e-9)
        """
        # 目标熔深
        target_pen: float = self.compute_target_penetration(I, v_mms, t_mm)

        # 实际熔深 (使用已有方法)
        actual_pen: float = self.compute_penetration(I, V, v_mms)

        # 偏差比率
        deviation: float = abs(actual_pen - target_pen) / max(target_pen, 1e-9)

        return (target_pen, actual_pen, deviation)

    def compute_target_penetration(
        self,
        I: float,
        v: float,
        t: float,
    ) -> float:
        """计算目标熔深.

        章锋论文公式:
            target_pen = k_I × I² / (v × t)

        其中 k_I 是电流系数 (默认0.085), I是电流, v是速度, t是板厚.

        Args:
            I: 焊接电流 (A).
            v: 焊接速度 (mm/s).
            t: 板厚 (mm).

        Returns:
            目标熔深 (mm). 如果 v 或 t 过小, 返回 0.
        """
        if v < 1e-9 or t < 1e-9:
            return 0.0
        target_pen: float = self.K_I_COEFF * I ** 2 / (v * t)
        return float(max(0.0, target_pen))

    def compute_nominal_voltage(self, thickness_mm: float) -> float:
        """计算名义电压.

        章锋论文公式:
            V_nom = 16.0 + 2.0 × (thickness > 3)

        厚度≤3mm时 V_nom=16V, 厚度>3mm时 V_nom=18V.

        Args:
            thickness_mm: 板厚 (mm).

        Returns:
            名义电压 (V).
        """
        if thickness_mm > self.THICKNESS_THRESHOLD:
            return self.V_NOM_BASE + self.V_NOM_INCREMENT
        return self.V_NOM_BASE
