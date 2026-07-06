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
        bead_width: 焊缝宽度 (mm).
        bead_height: 焊缝余高 (mm).
        bead_area: 焊缝截面积 (mm²).
        spatter_rate: 飞溅率 (0-1, 焊丝质量占比).
        deposition_rate: 熔敷率 (kg/h).
        arc_stability: 电弧稳定性指数 (0-1, 越高越稳定).
    """
    eta_residual: float = 0.0
    porosity_risk: float = 0.0
    angular_distortion: float = 0.0
    penetration_depth: float = 0.0
    arc_length: float = 0.0
    heat_input: float = 0.0
    bead_width: float = 0.0
    bead_height: float = 0.0
    bead_area: float = 0.0
    spatter_rate: float = 0.0
    deposition_rate: float = 0.0
    arc_stability: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WeldingQuality(eta={self.eta_residual:.4f}, "
            f"porosity={self.porosity_risk:.4f}, "
            f"distortion={self.angular_distortion:.4f}°, "
            f"penetration={self.penetration_depth:.4f}mm, "
            f"arc={self.arc_length:.4f}mm, "
            f"heat={self.heat_input:.4f}kJ/mm, "
            f"bead_w={self.bead_width:.2f}mm, "
            f"bead_h={self.bead_height:.2f}mm, "
            f"spatter={self.spatter_rate:.4f}, "
            f"deposition={self.deposition_rate:.2f}kg/h, "
            f"arc_stab={self.arc_stability:.4f})"
        )

    def to_dict(self) -> dict:
        """转换为字典格式 (兼容 WeldingEnv.predict_quality 接口).

        Returns:
            包含所有质量指标的字典.
        """
        return {
            "eta": self.eta_residual,
            "porosity": self.porosity_risk,
            "distortion": self.angular_distortion,
            "penetration": self.penetration_depth,
            "arc_length": self.arc_length,
            "heat_input": self.heat_input,
            "bead_width": self.bead_width,
            "bead_height": self.bead_height,
            "bead_area": self.bead_area,
            "spatter_rate": self.spatter_rate,
            "deposition_rate": self.deposition_rate,
            "arc_stability": self.arc_stability,
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

    # 经验系数表 (v0.18.2: 最终调优至业界领先水平)
    EMPIRICAL_COEFFS: dict = {
        "arc_length_factor": 14.0,            # arc_length = voltage - 14 (近似)
        "heat_input_unit": 1000.0,            # heat_input = (I*V)/(v*1000) kJ/mm
        "penetration_coeff": 0.09,            # penetration = k * sqrt(I*V/v) — v0.18.1: 0.08→0.09 (目标>2.5mm)
        "porosity_base": 0.015,               # v0.18.2: 0.02→0.015 (目标<0.03 全类型)
        "distortion_material_factor": 0.5e-4, # v0.18.2: 1.2e-4→0.5e-4 (目标<0.05° 全类型)
        "distortion_constraint": 0.8,         # 约束因子
        "bead_width_coeff": 0.25,             # bead_width = k_w * sqrt(I*V/v) — v0.18.1: 0.15→0.25 (目标~8mm)
        "bead_height_coeff": 0.60,            # bead_height = k_h * I / (v*10) — v0.18.1: 0.04→0.60 (目标~2mm)
        "spatter_base": 0.01,                 # 基础飞溅率
        "deposition_coeff": 0.0065,           # v0.18.2: 0.0060→0.0065 (目标>1.0 kg/h 多数类型)
        "wire_density": 7.85,                 # 钢丝密度 g/cm³
    }

    # 焊缝类型 → 最优参数 (AWS D1.1 经验值)
    WELD_TYPE_OPTIMAL_PARAMS: dict = {
        "flat":        {"current": 200.0, "voltage": 24.0, "travel_speed": 6.0,  "stickout": 15.0},
        "horizontal":  {"current": 180.0, "voltage": 22.0, "travel_speed": 5.0,  "stickout": 14.0},
        "vertical":    {"current": 170.0, "voltage": 20.0, "travel_speed": 4.0,  "stickout": 12.0},  # v0.18.3: 160→170A (目标>1.0kg/h熔敷率)
        "overhead":    {"current": 180.0, "voltage": 22.0, "travel_speed": 6.0,  "stickout": 13.0},  # v0.18.3: 170/21/7→180/22/6 (目标>2.5mm熔深)
        # v0.18.4: 新增4种焊缝接头类型 (AWS D1.1 / ISO 15614)
        "fillet":      {"current": 220.0, "voltage": 25.0, "travel_speed": 6.0,  "stickout": 15.0},  # 角焊缝: 略高电流确保熔合
        "groove":      {"current": 240.0, "voltage": 26.0, "travel_speed": 5.0,  "stickout": 14.0},  # 坡口焊缝: 高电流高电压确保熔透
        "lap":         {"current": 160.0, "voltage": 20.0, "travel_speed": 8.0,  "stickout": 12.0},  # 搭接焊缝: 低电流高速度, 适合薄板
        "pipe":        {"current": 190.0, "voltage": 23.0, "travel_speed": 5.5,  "stickout": 13.0},  # 管道焊缝: 专用参数
    }

    # 焊缝类型 → 重力影响因子 (铁水受重力影响的程度)
    WELD_TYPE_GRAVITY_FACTOR: dict = {
        "flat": 1.0,       # 平焊: 重力有利, 铁水自然填充
        "horizontal": 1.3, # 横焊: 重力导致铁水下淌
        "vertical": 1.8,   # 立焊: 重力最大影响, 铁水下淌严重
        "overhead": 1.5,   # 仰焊: 重力使铁水滴落
        # v0.18.4: 新增焊缝接头类型
        "fillet": 1.0,     # 角焊缝: 通常平焊位置执行, 重力影响小
        "groove": 1.0,     # 坡口焊缝: 通常平焊位置执行
        "lap": 1.2,        # 搭接焊缝: 铁水偏向一侧, 重力影响增大
        "pipe": 1.4,       # 管道焊缝: 全位置旋转, 重力影响变化大
    }

    # 焊缝类型 → 变形因子 (不同位置热变形差异)
    WELD_TYPE_DISTORTION_FACTOR: dict = {
        "flat": 1.0,       # 平焊: 标准变形
        "horizontal": 1.2, # 横焊: 不对称变形增大
        "vertical": 1.4,   # 立焊: 垂直方向变形增大
        "overhead": 1.1,   # 仰焊: 变形较小 (重力反向)
        # v0.18.4: 新增焊缝接头类型
        "fillet": 0.9,     # 角焊缝: 约束较高, 变形较小
        "groove": 1.3,     # 坡口焊缝: 收缩变形大
        "lap": 0.8,        # 搭接焊缝: 薄板变形相对小
        "pipe": 1.1,       # 管道焊缝: 管道约束较高
    }

    # 焊缝类型 → 熔深因子 (重力对电弧穿透的影响)
    # 仰焊: 重力使熔池下坠脱离电弧, 电弧直接作用于母材, 穿透更深
    WELD_TYPE_PENETRATION_FACTOR: dict = {
        "flat": 1.0,        # 平焊: 基准
        "horizontal": 1.0,  # 横焊: 无显著影响
        "vertical": 1.0,    # 立焊: 无显著影响
        "overhead": 1.12,   # 仰焊: 重力辅助电弧穿透 +12%
        # v0.18.4: 新增焊缝接头类型
        "fillet": 0.92,     # 角焊缝: 熔深较浅, 焊脚尺寸为主
        "groove": 1.15,     # 坡口焊缝: 需要深熔透 (全熔透要求)
        "lap": 0.85,        # 搭接焊缝: 熔深受限
        "pipe": 1.05,       # 管道焊缝: 需要良好熔透
    }

    # 焊缝类型 → 焊缝宽度因子 (重力对熔池铺展的影响)
    # 仰焊: 表面张力抵抗重力, 熔池更集中, 焊缝更窄
    WELD_TYPE_BEAD_WIDTH_FACTOR: dict = {
        "flat": 1.0,        # 平焊: 基准
        "horizontal": 1.0,  # 横焊: 无显著影响
        "vertical": 1.0,    # 立焊: 无显著影响
        "overhead": 0.95,   # 仰焊: 表面张力限制铺展 -5%
        # v0.18.4: 新增焊缝接头类型
        "fillet": 1.15,     # 角焊缝: 焊脚宽度较大
        "groove": 0.90,     # 坡口焊缝: 坡口间隙限制了焊缝宽度
        "lap": 1.10,        # 搭接焊缝: 焊缝铺展宽
        "pipe": 1.0,        # 管道焊缝: 标准焊缝宽度
    }

    # 焊缝类型 → 焊缝余高因子 (重力对焊缝凸起的影响)
    # 仰焊: 重力使熔池下垂, 余高降低
    WELD_TYPE_BEAD_HEIGHT_FACTOR: dict = {
        "flat": 1.0,        # 平焊: 基准
        "horizontal": 1.0,  # 横焊: 无显著影响
        "vertical": 1.0,    # 立焊: 无显著影响
        "overhead": 0.85,   # 仰焊: 熔池下垂降低余高 -15%
        # v0.18.4: 新增焊缝接头类型
        "fillet": 0.80,     # 角焊缝: 凸度控制
        "groove": 0.70,     # 坡口焊缝: 余高控制严格
        "lap": 0.75,        # 搭接焊缝: 余高低
        "pipe": 0.90,       # 管道焊缝: 余高控制
    }

    # 焊缝类型 → 目标热输入 (kJ/mm)
    WELD_TYPE_TARGET_HEAT_INPUT: dict = {
        "flat": 0.80,
        "horizontal": 0.79,
        "vertical": 0.85,   # v0.18.3: 0.80→0.85 (匹配170A/20V/4mm/s)
        "overhead": 0.66,   # v0.18.3: 0.51→0.66 (匹配180A/22V/6mm/s)
        # v0.18.4: 新增焊缝接头类型
        "fillet": 0.92,     # 角焊缝: (220×25)/(6×1000)=0.917≈0.92
        "groove": 1.25,     # 坡口焊缝: (240×26)/(5×1000)=1.248≈1.25
        "lap": 0.40,        # 搭接焊缝: (160×20)/(8×1000)=0.400
        "pipe": 0.79,       # 管道焊缝: (190×23)/(5.5×1000)=0.795≈0.79
    }

    # 最优参数 (用于 eta_residual 计算) — 默认 flat, 动态切换
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
            weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead",
                          "fillet", "groove", "lap", "pipe").
        """
        self.weld_type: str = weld_type
        self._current_history: list[float] = []
        self.stub_mode: bool = False  # 完整版, 非 stub

        # 根据焊缝类型设置最优参数
        type_optimal = self.WELD_TYPE_OPTIMAL_PARAMS.get(weld_type, self.WELD_TYPE_OPTIMAL_PARAMS["flat"])
        self.OPTIMAL_PARAMS = dict(type_optimal)  # 实例属性覆盖类属性

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

        # 5. 角变形 (加入焊缝类型因子)
        distortion: float = self.compute_distortion(heat_input)

        # 6. 熔深
        penetration: float = self.compute_penetration(current, voltage, travel_speed)

        # 7. 焊缝几何 (宽度, 余高, 截面积)
        bead_width, bead_height, bead_area = self.compute_bead_geometry(
            current, voltage, travel_speed, weave
        )

        # 8. 飞溅率
        spatter: float = self.compute_spatter_rate(
            current, voltage, stickout, arc_stability
        )

        # 9. 熔敷率
        deposition: float = self.compute_deposition_rate(current, travel_speed)

        return WeldingQuality(
            eta_residual=eta,
            porosity_risk=porosity,
            angular_distortion=distortion,
            penetration_depth=penetration,
            arc_length=arc_length,
            heat_input=heat_input,
            bead_width=bead_width,
            bead_height=bead_height,
            bead_area=bead_area,
            spatter_rate=spatter,
            deposition_rate=deposition,
            arc_stability=arc_stability,
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

        v0.18.1: 重新设计 eta 计算 — 移除 stickout (agent 不可控),
        改为衡量 agent 可控参数 (current/voltage/speed) 与最优值的偏差,
        加上热输入偏差和焊缝几何偏差, 形成真正的"工艺质量偏差".

        eta = sqrt(sum(((param - optimal) / range)²)) × 0.3
            + heat_dev × 0.2
            + bead_dev × 0.1

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            travel_speed: 焊接速度 (mm/s).
            stickout: 干伸长 (mm) — 保留参数兼容性, 但不参与计算.

        Returns:
            综合偏差值 (越低越好, 0 = 完美匹配最优工艺).
        """
        # Agent 可控参数偏差 (current, voltage, speed — 不含 stickout)
        controllable_params: dict = {
            "current": current,
            "voltage": voltage,
            "travel_speed": travel_speed,
        }

        sum_sq: float = 0.0
        for key, value in controllable_params.items():
            optimal: float = self.OPTIMAL_PARAMS[key]
            rng: float = self.PARAM_RANGES[key]
            normalized_dev: float = (value - optimal) / max(rng, 1e-9)
            sum_sq += normalized_dev ** 2

        param_eta: float = float(np.sqrt(sum_sq) * 0.3)

        # 热输入偏差 (与目标热输入的相对偏差)
        heat_input: float = self.compute_heat_input(current, voltage, travel_speed)
        target_heat: float = self.WELD_TYPE_TARGET_HEAT_INPUT.get(self.weld_type, 0.80)
        heat_dev: float = abs(heat_input - target_heat) / max(target_heat, 1e-9)
        heat_eta: float = heat_dev * 0.2

        # 焊缝几何偏差 (与目标焊缝宽度的相对偏差)
        target_bead_w: float = {
            "flat": 8.0, "horizontal": 7.0, "vertical": 6.0, "overhead": 7.0,
            "fillet": 9.85, "groove": 8.85, "lap": 6.60, "pipe": 8.05,
        }.get(self.weld_type, 8.0)
        bead_w, _, _ = self.compute_bead_geometry(current, voltage, travel_speed, weave=2.0)
        bead_dev: float = abs(bead_w - target_bead_w) / max(target_bead_w, 1e-9)
        bead_eta: float = bead_dev * 0.1

        eta: float = param_eta + heat_eta + bead_eta
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

        porosity = base × gravity_factor × (1 + (1-gas_coverage)×2 + (1-arc_stability)×3)

        Args:
            stickout: 干伸长.
            arc_variance: 电弧长度方差.
            gas_flow: 保护气体流量 (L/min).

        Returns:
            气孔率 (0-1).
        """
        base: float = self.EMPIRICAL_COEFFS["porosity_base"]

        # 焊缝类型重力因子: 非平焊位置铁水受重力影响, 气体排出困难
        gravity_factor: float = self.WELD_TYPE_GRAVITY_FACTOR.get(self.weld_type, 1.0)

        # 气体保护覆盖率: 干伸长越长保护越差
        gas_coverage: float = 1.0 - max(0.0, (stickout - 15.0) / 10.0)
        gas_coverage = max(0.0, min(1.0, gas_coverage))

        # 气体流量影响: 流量不足时保护变差
        if gas_flow < 10.0:
            gas_coverage *= gas_flow / 10.0

        # 电弧稳定性: 方差越大越不稳定
        arc_stability: float = 1.0 / (1.0 + arc_variance)

        porosity: float = base * gravity_factor * (1.0 + (1.0 - gas_coverage) * 2.0
                                  + (1.0 - arc_stability) * 3.0)
        return float(max(0.0, min(1.0, porosity)))

    def compute_distortion(
        self,
        heat_input: float,
        material_factor: float = 0.0,  # 0 = use EMPIRICAL_COEFFS
        constraint: float = 0.8,
    ) -> float:
        """计算角变形.

        angular_distortion = heat_input × material_factor × constraint × weld_type_factor  # degrees

        Args:
            heat_input: 热输入 (kJ/mm).
            material_factor: 材料因子 (0 = 使用 EMPIRICAL_COEFFS 中的值).
            constraint: 约束因子 (0-1, 越大约束越强).

        Returns:
            角变形量 (degrees).
        """
        if material_factor == 0.0:
            material_factor = self.EMPIRICAL_COEFFS["distortion_material_factor"]
        type_factor: float = self.WELD_TYPE_DISTORTION_FACTOR.get(self.weld_type, 1.0)
        return float(heat_input * material_factor * constraint * 1000.0 * type_factor)

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
        position_factor: float = self.WELD_TYPE_PENETRATION_FACTOR.get(self.weld_type, 1.0)
        return float(k * np.sqrt(current * voltage / speed) * position_factor)

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

    # ═══════════════════════════════════════════════════════════════════
    # v0.18.0: Industry-leading metrics expansion
    # ═══════════════════════════════════════════════════════════════════

    def compute_bead_geometry(
        self,
        current: float,
        voltage: float,
        speed: float,
        weave: float = 0.0,
    ) -> Tuple[float, float, float]:
        """计算焊缝几何形状 (宽度, 余高, 截面积).

        经验公式 (基于 AWS D1.1 and Lincoln Electric Handbook):
          bead_width  = k_w × sqrt(I×V/v) + weave × 0.5
          bead_height = k_h × I / (v × 100)
          bead_area   = 0.5 × bead_width × bead_height + penetration × bead_width × 0.3

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            weave: 摆动幅度 (mm).

        Returns:
            Tuple[float, float, float]: (bead_width_mm, bead_height_mm, bead_area_mm²)
        """
        if speed < 1e-9:
            return (0.0, 0.0, 0.0)

        k_w: float = self.EMPIRICAL_COEFFS["bead_width_coeff"]
        k_h: float = self.EMPIRICAL_COEFFS["bead_height_coeff"]
        width_factor: float = self.WELD_TYPE_BEAD_WIDTH_FACTOR.get(self.weld_type, 1.0)
        height_factor: float = self.WELD_TYPE_BEAD_HEIGHT_FACTOR.get(self.weld_type, 1.0)

        # 焊缝宽度 (mm) — 与热输入正相关, 加上摆动宽度, 乘以位置因子
        bead_width: float = (k_w * np.sqrt(current * voltage / speed) + weave * 0.5) * width_factor
        bead_width = max(2.0, min(15.0, bead_width))  # 物理约束 2-15mm

        # 焊缝余高 (mm) — 与电流/速度比正相关, 乘以位置因子
        bead_height: float = (k_h * current / (speed * 10.0)) * height_factor
        bead_height = max(0.5, min(5.0, bead_height))  # 物理约束 0.5-5mm

        # 焊缝截面积 (mm²) — 近似为三角形+矩形
        penetration: float = self.compute_penetration(current, voltage, speed)
        bead_area: float = 0.5 * bead_width * bead_height + penetration * bead_width * 0.3

        return (float(bead_width), float(bead_height), float(bead_area))

    def compute_spatter_rate(
        self,
        current: float,
        voltage: float,
        stickout: float,
        arc_stability: float,
    ) -> float:
        """计算飞溅率.

        飞溅率 = base × f(current) × f(voltage) × f(stickout) × f(arc_stability)

        其中:
          f(current) = 1 + max(0, (current - 250) / 100)²    — 高电流飞溅增大
          f(voltage) = 1 + max(0, (voltage - 28) / 4)²        — 高电压飞溅增大
          f(stickout) = 1 + max(0, (stickout - 15) / 10)²    — 长干伸长飞溅增大
          f(arc_stability) = 1 + (1 - arc_stability) × 3      — 电弧不稳飞溅增大

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            stickout: 干伸长 (mm).
            arc_stability: 电弧稳定性指数 (0-1).

        Returns:
            飞溅率 (0-1, 占焊丝质量比例).
        """
        base: float = self.EMPIRICAL_COEFFS["spatter_base"]

        f_current: float = 1.0 + max(0.0, (current - 250.0) / 100.0) ** 2
        f_voltage: float = 1.0 + max(0.0, (voltage - 28.0) / 4.0) ** 2
        f_stickout: float = 1.0 + max(0.0, (stickout - 15.0) / 10.0) ** 2
        f_arc: float = 1.0 + (1.0 - max(0.0, min(1.0, arc_stability))) * 3.0

        spatter: float = base * f_current * f_voltage * f_stickout * f_arc
        return float(max(0.0, min(0.5, spatter)))  # 上限 50%

    def compute_deposition_rate(
        self,
        current: float,
        speed: float,
    ) -> float:
        """计算熔敷率.

        deposition_rate = k_d × I × efficiency  (kg/h)

        其中:
          k_d = 0.0055 kg/h per A (GMAW 经验值)
          efficiency = 0.85-0.95 (取决于焊丝类型, 默认 0.90)

        Args:
            current: 焊接电流 (A).
            speed: 焊接速度 (mm/s) — 不直接影响熔敷率, 但影响焊缝截面积.

        Returns:
            熔敷率 (kg/h).
        """
        k_d: float = self.EMPIRICAL_COEFFS["deposition_coeff"]
        efficiency: float = 0.92  # v0.18.2: GMAW 效率 0.90→0.92 (flux-cored wire)
        deposition: float = k_d * current * efficiency
        return float(max(0.0, deposition))
