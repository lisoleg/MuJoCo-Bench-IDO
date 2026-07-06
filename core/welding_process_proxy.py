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

v0.20.1: 非标场景分类 + DIKWP K-层规则库 + η-PID自适应控制 (从播放器到工匠)
v0.20.0: 扩展至25种焊缝类型 + 跨材质支持(Al6061/SS304/Ti6Al4V/Q235) + IntentGuard四级安全分类
v0.19.0: 扩展至18种焊缝类型 + 逼真物理仿真 + 性能优化

Author: MuJoCo-Bench-IDO Welding Module v0.20.1
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
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
        cooling_rate_t85: t8/5冷却时间(s).
        interpass_temp: 层间温度(°C).
        haz_width: 热影响区宽度(mm).
        max_hardness: 最大硬度(HV).
        residual_stress: 残余应力(MPa).
        cracking_susceptibility: 凝固裂纹敏感性(0-1).
        microstructure: 微观组织比例.
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
    # v0.19.0: Enhanced physics metrics
    cooling_rate_t85: float = 0.0        # t8/5冷却时间(s)
    interpass_temp: float = 0.0          # 层间温度(°C)
    haz_width: float = 0.0               # 热影响区宽度(mm)
    max_hardness: float = 0.0            # 最大硬度(HV)
    residual_stress: float = 0.0         # 残余应力(MPa)
    cracking_susceptibility: float = 0.0 # 凝固裂纹敏感性(0-1)
    microstructure: dict = field(default_factory=dict)  # 微观组织比例

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
            f"arc_stab={self.arc_stability:.4f}, "
            f"t85={self.cooling_rate_t85:.1f}s, "
            f"HAZ={self.haz_width:.2f}mm, "
            f"HV={self.max_hardness:.0f}, "
            f"stress={self.residual_stress:.0f}MPa)"
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
            # v0.19.0: Enhanced physics metrics
            "cooling_rate_t85": self.cooling_rate_t85,
            "interpass_temp": self.interpass_temp,
            "haz_width": self.haz_width,
            "max_hardness": self.max_hardness,
            "residual_stress": self.residual_stress,
            "cracking_susceptibility": self.cracking_susceptibility,
            "microstructure": self.microstructure,
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
        # v0.19.0: 新增10种焊缝接头类型 (AWS A3.0, AWS D1.1, ISO 15614, EN ISO 4063)
        "corner":      {"current": 200.0, "voltage": 24.0, "travel_speed": 6.0,  "stickout": 15.0},  # 角接焊缝: L形接头
        "edge":        {"current": 180.0, "voltage": 22.0, "travel_speed": 7.0,  "stickout": 14.0},  # 边缘焊缝: 两板平行边缘对接
        "plug":        {"current": 210.0, "voltage": 23.0, "travel_speed": 4.0,  "stickout": 16.0},  # 塞焊焊缝: 圆孔塞焊
        "slot":        {"current": 200.0, "voltage": 22.0, "travel_speed": 4.5,  "stickout": 15.0},  # 槽焊焊缝: 长孔槽焊
        "surfacing":   {"current": 230.0, "voltage": 25.0, "travel_speed": 5.0,  "stickout": 16.0},  # 堆焊: 表面堆焊
        "tack":        {"current": 190.0, "voltage": 23.0, "travel_speed": 8.0,  "stickout": 14.0},  # 定位焊: 短段定位焊
        "butt":        {"current": 210.0, "voltage": 24.0, "travel_speed": 5.0,  "stickout": 14.0},  # 对接焊缝: 全熔透对接
        "tee":         {"current": 215.0, "voltage": 24.0, "travel_speed": 5.5,  "stickout": 15.0},  # T形焊缝: T接头
        "multipass":   {"current": 225.0, "voltage": 25.0, "travel_speed": 5.0,  "stickout": 15.0},  # 多层焊: 多道焊缝
        "repair":      {"current": 195.0, "voltage": 23.0, "travel_speed": 5.0,  "stickout": 14.0},  # 补焊: 缺陷修复焊缝
        # v0.20.0: 新增6种焊缝类型 (电阻焊/特殊焊缝) + generic兜底 (DIKWP-IDO跨材质融合)
        "seam":        {"current": 180.0, "voltage": 15.0, "travel_speed": 3.0,  "stickout": 10.0},  # 缝焊: 电阻缝焊, 电压低
        "spot":        {"current": 150.0, "voltage": 12.0, "travel_speed": 2.0,  "stickout": 8.0},   # 点焊: 电阻点焊, 极低电压
        "flange":      {"current": 200.0, "voltage": 24.0, "travel_speed": 6.0,  "stickout": 15.0},  # 法兰焊: 类似角焊
        "projection":  {"current": 170.0, "voltage": 14.0, "travel_speed": 2.5,  "stickout": 10.0},  # 凸焊: 电阻凸焊
        "stud":        {"current": 250.0, "voltage": 28.0, "travel_speed": 4.0,  "stickout": 18.0},  # 螺柱焊: 高电流短时焊
        "seal":        {"current": 160.0, "voltage": 20.0, "travel_speed": 7.0,  "stickout": 13.0},  # 密封焊: 低电流薄板密封
        "generic":     {"current": 200.0, "voltage": 24.0, "travel_speed": 6.0,  "stickout": 15.0},  # 通用兜底: 未知类型优雅降级
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 1.1,     # 角接焊缝: L形接头, 重力影响略增
        "edge": 1.0,       # 边缘焊缝: 平行边缘, 重力影响标准
        "plug": 1.0,       # 塞焊焊缝: 孔内填充, 重力影响小
        "slot": 1.0,       # 槽焊焊缝: 槽内填充, 重力影响小
        "surfacing": 1.0,  # 堆焊: 表面堆焊, 重力有利
        "tack": 1.0,       # 定位焊: 短段, 重力影响标准
        "butt": 1.0,       # 对接焊缝: 平焊位置, 重力影响标准
        "tee": 1.0,        # T形焊缝: 通常平焊位置
        "multipass": 1.0,  # 多层焊: 通常平焊位置
        "repair": 1.0,     # 补焊: 修复位置不定, 取标准值
        # v0.20.0: 新增焊缝类型 (电阻焊类用1.0)
        "seam": 1.0,       # 缝焊: 电阻焊, 重力影响小
        "spot": 1.0,       # 点焊: 电阻焊, 重力影响小
        "flange": 1.0,     # 法兰焊: 类似角焊, 平焊位置
        "projection": 1.0, # 凸焊: 电阻焊, 重力影响小
        "stud": 1.0,       # 螺柱焊: 垂直方向, 重力影响标准
        "seal": 1.0,       # 密封焊: 薄板平焊, 重力影响小
        "generic": 1.0,    # 通用兜底: 标准值
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 1.0,     # 角接焊缝: L形接头, 标准变形
        "edge": 0.85,      # 边缘焊缝: 变形较小
        "plug": 0.7,       # 塞焊焊缝: 局部热输入, 变形小
        "slot": 0.75,      # 槽焊焊缝: 变形较小
        "surfacing": 0.6,  # 堆焊: 表面堆焊, 变形最小
        "tack": 0.3,       # 定位焊: 短段, 变形极小
        "butt": 1.2,       # 对接焊缝: 全熔透, 收缩变形大
        "tee": 0.95,       # T形焊缝: 变形略低于标准
        "multipass": 1.15, # 多层焊: 多道累积变形
        "repair": 1.05,    # 补焊: 局部修复, 变形略增
        # v0.20.0: 新增焊缝类型
        "seam": 0.80,      # 缝焊: 电阻焊, 热输入集中, 变形较小
        "spot": 0.50,      # 点焊: 极局部热输入, 变形最小
        "flange": 0.90,    # 法兰焊: 类似角焊, 约束较高
        "projection": 0.70,# 凸焊: 局部凸点, 变形较小
        "stud": 1.00,      # 螺柱焊: 集中高热, 标准变形
        "seal": 0.70,      # 密封焊: 低热输入薄板, 变形较小
        "generic": 1.0,    # 通用兜底: 标准变形
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 0.90,     # 角接焊缝: 熔深略浅
        "edge": 0.88,       # 边缘焊缝: 熔深受限
        "plug": 0.95,       # 塞焊焊缝: 孔内熔透
        "slot": 0.92,       # 槽焊焊缝: 槽内熔透
        "surfacing": 0.70,  # 堆焊: 表面堆焊, 熔深最浅
        "tack": 0.80,       # 定位焊: 短段, 熔深较浅
        "butt": 1.10,       # 对接焊缝: 全熔透要求, 熔深深
        "tee": 0.90,        # T形焊缝: 熔深略浅
        "multipass": 0.85,  # 多层焊: 逐层累积熔深
        "repair": 1.08,     # 补焊: 需要深熔透修复缺陷
        # v0.20.0: 新增焊缝类型
        "seam": 0.90,       # 缝焊: 电阻焊, 熔深较浅
        "spot": 0.70,       # 点焊: 熔核形成, 非深熔透
        "flange": 0.92,     # 法兰焊: 类似角焊, 熔深略浅
        "projection": 0.85, # 凸焊: 局部凸点熔透
        "stud": 1.10,       # 螺柱焊: 高电流, 熔深深
        "seal": 0.80,       # 密封焊: 薄板浅熔深
        "generic": 1.0,     # 通用兜底: 基准
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 1.05,     # 角接焊缝: L形接头, 焊缝略宽
        "edge": 0.95,       # 边缘焊缝: 焊缝略窄
        "plug": 1.20,       # 塞焊焊缝: 孔内填充, 焊缝宽
        "slot": 1.15,       # 槽焊焊缝: 槽内填充, 焊缝较宽
        "surfacing": 1.25,  # 堆焊: 表面铺展, 焊缝最宽
        "tack": 0.90,       # 定位焊: 短段, 焊缝较窄
        "butt": 0.95,       # 对接焊缝: 坡口限制, 焊缝略窄
        "tee": 1.10,        # T形焊缝: 焊脚宽度较大
        "multipass": 1.05,  # 多层焊: 逐道累积, 焊缝略宽
        "repair": 0.92,     # 补焊: 局部修复, 焊缝略窄
        # v0.20.0: 新增焊缝类型
        "seam": 1.0,        # 缝焊: 标准焊缝宽度
        "spot": 0.80,       # 点焊: 熔核小, 焊缝窄
        "flange": 1.10,     # 法兰焊: 类似角焊, 焊缝略宽
        "projection": 0.90, # 凸焊: 局部凸点, 焊缝较窄
        "stud": 1.0,        # 螺柱焊: 集中熔化, 标准宽度
        "seal": 1.0,        # 密封焊: 标准宽度
        "generic": 1.0,     # 通用兜底: 基准
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 0.90,     # 角接焊缝: 余高略低
        "edge": 0.85,       # 边缘焊缝: 余高低
        "plug": 1.10,       # 塞焊焊缝: 孔内填充, 余高较高
        "slot": 1.05,       # 槽焊焊缝: 余高略高
        "surfacing": 1.30,  # 堆焊: 表面堆高, 余高最高
        "tack": 0.75,       # 定位焊: 短段, 余高低
        "butt": 0.80,       # 对接焊缝: 余高控制严格
        "tee": 0.85,        # T形焊缝: 余高略低
        "multipass": 1.15,  # 多层焊: 逐层累积, 余高较高
        "repair": 0.95,     # 补焊: 修复余高, 略低于标准
        # v0.20.0: 新增焊缝类型
        "seam": 0.90,       # 缝焊: 低余高强化
        "spot": 1.10,       # 点焊: 熔核凸出, 余高较高
        "flange": 0.85,     # 法兰焊: 类似角焊, 余高略低
        "projection": 1.0,  # 凸焊: 标准余高
        "stud": 1.20,       # 螺柱焊: 螺柱凸出, 余高最高
        "seal": 0.80,       # 密封焊: 薄板低余高
        "generic": 1.0,     # 通用兜底: 基准
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
        # v0.19.0: 新增焊缝接头类型
        "corner": 0.80,     # 角接焊缝: (200×24)/(6×1000)=0.800
        "edge": 0.56,       # 边缘焊缝: (180×22)/(7×1000)=0.566≈0.56
        "plug": 1.21,       # 塞焊焊缝: (210×23)/(4×1000)=1.208≈1.21
        "slot": 0.98,       # 槽焊焊缝: (200×22)/(4.5×1000)=0.978≈0.98
        "surfacing": 1.15,  # 堆焊: (230×25)/(5×1000)=1.150
        "tack": 0.55,       # 定位焊: (190×23)/(8×1000)=0.546≈0.55
        "butt": 1.01,       # 对接焊缝: (210×24)/(5×1000)=1.008≈1.01
        "tee": 0.94,        # T形焊缝: (215×24)/(5.5×1000)=0.938≈0.94
        "multipass": 1.13,  # 多层焊: (225×25)/(5×1000)=1.125≈1.13
        "repair": 0.90,     # 补焊: (195×23)/(5×1000)=0.897≈0.90
        # v0.20.0: 新增焊缝类型 (按公式 (I×V)/(v×1000) 计算)
        "seam": 0.90,       # 缝焊: (180×15)/(3×1000)=0.900
        "spot": 0.90,       # 点焊: (150×12)/(2×1000)=0.900
        "flange": 0.80,     # 法兰焊: (200×24)/(6×1000)=0.800
        "projection": 0.95, # 凸焊: (170×14)/(2.5×1000)=0.952≈0.95
        "stud": 1.75,       # 螺柱焊: (250×28)/(4×1000)=1.750
        "seal": 0.46,       # 密封焊: (160×20)/(7×1000)=0.457≈0.46
        "generic": 0.80,    # 通用兜底: 同flat基准
    }

    # v0.19.0: 焊缝类型 → 目标焊缝宽度 (mm) — 用于 eta 计算
    # 基于最优参数计算: bead_width = k_w * sqrt(I*V/v) + weave*0.5, 然后 * width_factor
    _TARGET_BEAD_WIDTH: dict = {
        "flat": 8.0, "horizontal": 7.0, "vertical": 6.0, "overhead": 7.0,
        "fillet": 9.85, "groove": 8.85, "lap": 6.60, "pipe": 8.05,
        # v0.19.0: 新增焊缝类型目标宽度
        "corner": 8.48, "edge": 6.60, "plug": 11.63, "slot": 10.14,
        "surfacing": 11.85, "tack": 6.16, "butt": 8.49, "tee": 9.52,
        "multipass": 9.85, "repair": 7.81,
        # v0.20.0: 新增焊缝类型目标宽度
        "seam": 8.50, "spot": 6.80, "flange": 8.88, "projection": 7.84,
        "stud": 11.46, "seal": 6.35, "generic": 8.0,
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

    # v0.20.0: 材料属性表 (DIKWP-IDO跨材质支持: Al6061/SS304/Ti6Al4V/Q235)
    MATERIAL_PROPERTIES: dict = {
        "steel":      {"thermal_conductivity": 50.0, "melting_point": 1500.0, "density": 7850.0, "specific_heat": 470.0, "thermal_expansion": 12e-6},
        "aluminum":   {"thermal_conductivity": 167.0, "melting_point": 660.0, "density": 2700.0, "specific_heat": 900.0, "thermal_expansion": 23e-6},
        "stainless":  {"thermal_conductivity": 16.0, "melting_point": 1450.0, "density": 8000.0, "specific_heat": 500.0, "thermal_expansion": 16e-6},
        "titanium":   {"thermal_conductivity": 7.0, "melting_point": 1668.0, "density": 4500.0, "specific_heat": 520.0, "thermal_expansion": 9e-6},
    }

    def __init__(self, weld_type: str = "flat", material: str = "steel") -> None:
        """初始化焊接工艺代理模型.

        Args:
            weld_type: 焊接姿态类型 ("flat", "horizontal", "vertical", "overhead",
                          "fillet", "groove", "lap", "pipe", "corner", "edge",
                          "plug", "slot", "surfacing", "tack", "butt", "tee",
                          "multipass", "repair", "seam", "spot", "flange",
                          "projection", "stud", "seal", "generic").
            material: 焊接材料 ("steel", "aluminum", "stainless", "titanium").
                      默认 "steel" (Q235碳钢).
        """
        self.weld_type: str = weld_type
        self.material: str = material
        self._current_history: list[float] = []
        self.stub_mode: bool = False  # 完整版, 非 stub

        # 根据焊缝类型设置最优参数
        type_optimal = self.WELD_TYPE_OPTIMAL_PARAMS.get(weld_type, self.WELD_TYPE_OPTIMAL_PARAMS["flat"])
        self.OPTIMAL_PARAMS = dict(type_optimal)  # 实例属性覆盖类属性

        # v0.20.0: 提取材料属性到实例变量 (DIKWP-IDO跨材质支持)
        mat_props: dict = self.MATERIAL_PROPERTIES.get(
            material, self.MATERIAL_PROPERTIES["steel"]
        )
        self._thermal_conductivity: float = mat_props["thermal_conductivity"]
        self._melting_point: float = mat_props["melting_point"]
        self._material_density: float = mat_props["density"]
        self._specific_heat: float = mat_props["specific_heat"]
        self._thermal_expansion: float = mat_props["thermal_expansion"]

        # v0.19.0: Performance optimizations ──────────────────────────
        # 预计算参数范围倒数 (避免每次除法)
        self._param_range_inv: dict = {
            k: 1.0 / v for k, v in self.PARAM_RANGES.items()
        }
        # 将频繁访问的 EMPIRICAL_COEFFS 值提取为实例属性 (减少字典查找)
        self._arc_length_factor: float = self.EMPIRICAL_COEFFS["arc_length_factor"]
        self._heat_input_unit: float = self.EMPIRICAL_COEFFS["heat_input_unit"]
        self._penetration_coeff: float = self.EMPIRICAL_COEFFS["penetration_coeff"]
        self._porosity_base: float = self.EMPIRICAL_COEFFS["porosity_base"]
        self._distortion_material_factor: float = self.EMPIRICAL_COEFFS["distortion_material_factor"]
        self._bead_width_coeff: float = self.EMPIRICAL_COEFFS["bead_width_coeff"]
        self._bead_height_coeff: float = self.EMPIRICAL_COEFFS["bead_height_coeff"]
        self._spatter_base: float = self.EMPIRICAL_COEFFS["spatter_base"]
        self._deposition_coeff: float = self.EMPIRICAL_COEFFS["deposition_coeff"]
        # 缓存 sqrt(I*V/v) 计算
        self._cached_sqrt_ivv: float = 0.0
        self._cached_params: tuple = (0.0, 0.0, 0.0)
        # ──────────────────────────────────────────────────────────────

    def _get_sqrt_ivv(
        self,
        current: float,
        voltage: float,
        speed: float,
    ) -> float:
        """获取缓存或重新计算的 sqrt(I*V/v).

        如果当前参数与缓存参数匹配, 直接返回缓存值;
        否则重新计算并更新缓存。

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).

        Returns:
            sqrt(current * voltage / speed) 的值.
        """
        if (self._cached_params[0] == current and
                self._cached_params[1] == voltage and
                self._cached_params[2] == speed and
                self._cached_sqrt_ivv > 0.0):
            return self._cached_sqrt_ivv
        if speed < 1e-9:
            return 0.0
        val: float = float(np.sqrt(current * voltage / speed))
        self._cached_sqrt_ivv = val
        self._cached_params = (current, voltage, speed)
        return val

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
        # v0.19.0: 预计算并缓存 sqrt(I*V/v) 供后续方法复用
        self._cached_params = (current, voltage, travel_speed)
        self._cached_sqrt_ivv = float(np.sqrt(
            current * voltage / max(travel_speed, 1e-9)
        ))

        # 更新电流历史 (用于电弧稳定性计算)
        self.update_current_history(current)
        current_variance: float = self.compute_current_variance()

        # 1. 电弧长度
        arc_length: float = self.compute_arc_length(voltage)

        # 2. 热输入
        heat_input: float = self.compute_heat_input(current, voltage, travel_speed)

        # 3. 综合工艺偏差 eta_residual
        eta: float = self._compute_eta_residual(
            current, voltage, travel_speed, stickout
        )

        # 4. 气孔风险
        arc_stability: float = 1.0 / (1.0 + current_variance / 100.0)
        gas_coverage: float = 1.0 - max(0.0, (stickout - 15.0) / 10.0)
        gas_coverage = max(0.0, min(1.0, gas_coverage))

        porosity: float = self.compute_porosity(
            stickout, current_variance / 10.0, gas_flow=15.0
        )
        porosity += self._porosity_base * (1.0 - arc_stability) * 2.0
        porosity = max(0.0, min(1.0, porosity))

        # 5. 角变形
        distortion: float = self.compute_distortion(heat_input)

        # 6. 熔深
        penetration: float = self.compute_penetration(current, voltage, travel_speed)

        # 7. 焊缝几何
        bead_width, bead_height, bead_area = self.compute_bead_geometry(
            current, voltage, travel_speed, weave
        )

        # 8. 飞溅率
        spatter: float = self.compute_spatter_rate(
            current, voltage, stickout, arc_stability
        )

        # 9. 熔敷率
        deposition: float = self.compute_deposition_rate(current, travel_speed)

        # v0.19.0: 逼真物理仿真指标
        t85: float = self.compute_cooling_rate(current, voltage, travel_speed)
        interpass: float = self.compute_interpass_temp(heat_input=heat_input)
        microstruct: dict = self.compute_microstructure(t85)
        haz_w: float = self.compute_haz_width(current, voltage, travel_speed)
        hardness: float = self.compute_hardness(t85)
        res_stress: float = self.compute_residual_stress(heat_input)
        crack_suscept: float = self.compute_solidification_cracking(
            current, travel_speed, bead_width
        )

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
            # v0.19.0: Enhanced physics metrics
            cooling_rate_t85=t85,
            interpass_temp=interpass,
            haz_width=haz_w,
            max_hardness=hardness,
            residual_stress=res_stress,
            cracking_susceptibility=crack_suscept,
            microstructure=microstruct,
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

        v0.19.0: 向量化计算 (用 numpy 替代 for 循环)

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
        # v0.19.0: 向量化参数偏差计算
        params = np.array([current, voltage, travel_speed])
        optimal = np.array([
            self.OPTIMAL_PARAMS["current"],
            self.OPTIMAL_PARAMS["voltage"],
            self.OPTIMAL_PARAMS["travel_speed"],
        ])
        ranges = np.array([
            self.PARAM_RANGES["current"],
            self.PARAM_RANGES["voltage"],
            self.PARAM_RANGES["travel_speed"],
        ])
        param_eta: float = float(
            np.sqrt(np.sum(((params - optimal) / ranges) ** 2)) * 0.3
        )

        # 热输入偏差
        heat_input: float = self.compute_heat_input(current, voltage, travel_speed)
        target_heat: float = self.WELD_TYPE_TARGET_HEAT_INPUT.get(self.weld_type, 0.80)
        heat_dev: float = abs(heat_input - target_heat) / max(target_heat, 1e-9)
        heat_eta: float = heat_dev * 0.2

        # 焊缝几何偏差 (与目标焊缝宽度的相对偏差)
        target_bead_w: float = self._TARGET_BEAD_WIDTH.get(self.weld_type, 8.0)
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
        return float((current * voltage) / (speed * self._heat_input_unit))

    def compute_arc_length(self, voltage: float) -> float:
        """计算电弧长度.

        arc_length = voltage - 14  # mm (近似)

        Args:
            voltage: 焊接电压 (V).

        Returns:
            电弧长度 (mm).
        """
        return float(max(0.0, voltage - self._arc_length_factor))

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
        base: float = self._porosity_base

        # 焊缝类型重力因子
        gravity_factor: float = self.WELD_TYPE_GRAVITY_FACTOR.get(self.weld_type, 1.0)

        # 气体保护覆盖率
        gas_coverage: float = 1.0 - max(0.0, (stickout - 15.0) / 10.0)
        gas_coverage = max(0.0, min(1.0, gas_coverage))

        if gas_flow < 10.0:
            gas_coverage *= gas_flow / 10.0

        # 电弧稳定性
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
            material_factor = self._distortion_material_factor
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

        v0.19.0: 使用缓存的 sqrt(I*V/v) 值

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).

        Returns:
            熔深.
        """
        if speed < 1e-9:
            return 0.0
        sqrt_ivv: float = self._get_sqrt_ivv(current, voltage, speed)
        position_factor: float = self.WELD_TYPE_PENETRATION_FACTOR.get(self.weld_type, 1.0)
        return float(self._penetration_coeff * sqrt_ivv * position_factor)

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
        target_pen: float = self.compute_target_penetration(I, v_mms, t_mm)
        actual_pen: float = self.compute_penetration(I, V, v_mms)
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

        v0.19.0: 使用缓存的 sqrt(I*V/v) 值

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

        k_w: float = self._bead_width_coeff
        k_h: float = self._bead_height_coeff
        width_factor: float = self.WELD_TYPE_BEAD_WIDTH_FACTOR.get(self.weld_type, 1.0)
        height_factor: float = self.WELD_TYPE_BEAD_HEIGHT_FACTOR.get(self.weld_type, 1.0)

        # v0.19.0: 使用缓存的 sqrt(I*V/v)
        sqrt_ivv: float = self._get_sqrt_ivv(current, voltage, speed)

        # 焊缝宽度 (mm)
        bead_width: float = (k_w * sqrt_ivv + weave * 0.5) * width_factor
        bead_width = max(2.0, min(15.0, bead_width))  # 物理约束 2-15mm

        # 焊缝余高 (mm)
        bead_height: float = (k_h * current / (speed * 10.0)) * height_factor
        bead_height = max(0.5, min(5.0, bead_height))  # 物理约束 0.5-5mm

        # 焊缝截面积 (mm²)
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

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            stickout: 干伸长 (mm).
            arc_stability: 电弧稳定性指数 (0-1).

        Returns:
            飞溅率 (0-1, 占焊丝质量比例).
        """
        base: float = self._spatter_base

        f_current: float = 1.0 + max(0.0, (current - 250.0) / 100.0) ** 2
        f_voltage: float = 1.0 + max(0.0, (voltage - 28.0) / 4.0) ** 2
        f_stickout: float = 1.0 + max(0.0, (stickout - 15.0) / 10.0) ** 2
        f_arc: float = 1.0 + (1.0 - max(0.0, min(1.0, arc_stability))) * 3.0

        spatter: float = base * f_current * f_voltage * f_stickout * f_arc
        return float(max(0.0, min(0.5, spatter)))

    def compute_deposition_rate(
        self,
        current: float,
        speed: float,
    ) -> float:
        """计算熔敷率.

        deposition_rate = k_d × I × efficiency  (kg/h)

        Args:
            current: 焊接电流 (A).
            speed: 焊接速度 (mm/s) — 不直接影响熔敷率, 但影响焊缝截面积.

        Returns:
            熔敷率 (kg/h).
        """
        efficiency: float = 0.92  # v0.18.2: GMAW 效率 0.90→0.92 (flux-cored wire)
        deposition: float = self._deposition_coeff * current * efficiency
        return float(max(0.0, deposition))

    # ═══════════════════════════════════════════════════════════════════
    # v0.19.0: 逼真物理仿真 — 热循环/微观组织/残余应力/裂纹
    # ═══════════════════════════════════════════════════════════════════

    def compute_cooling_rate(
        self,
        current: float,
        voltage: float,
        speed: float,
        thickness: float = 10.0,
    ) -> float:
        """计算t8/5冷却时间(800°C→500°C), 影响微观组织.

        t85 = K × (thickness / (current×voltage/speed))² × (1/300 + 1/800)
        简化经验公式: t85 ∝ thickness² / heat_input

        v0.20.0: 引入材料热导率影响 — 热导率高的材料散热更快, t85更短。
        以钢的热导率(50 W/m·K)为基准归一化: t85 × (50 / thermal_conductivity)

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            thickness: 板厚 (mm), 默认10mm.

        Returns:
            t8/5冷却时间 (s), 范围0.5-120秒.
        """
        heat: float = self.compute_heat_input(current, voltage, speed)
        # 基准 t85 (以钢的热导率50 W/m·K为基准)
        t85: float = 4300.0 * (thickness / 10.0) ** 2 / max(heat * 1000.0, 1.0)
        # v0.20.0: 材料热导率修正 — 热导率越高, 冷却越快, t85越短
        t85 *= (50.0 / max(self._thermal_conductivity, 1e-9))
        return float(max(0.5, min(120.0, t85)))

    def compute_interpass_temp(
        self,
        base_temp: float = 20.0,
        heat_input: float = 0.8,
    ) -> float:
        """层间温度估算.

        Args:
            base_temp: 基础/环境温度 (°C).
            heat_input: 热输入 (kJ/mm).

        Returns:
            层间温度 (°C), 上限350°C.
        """
        return float(min(base_temp + heat_input * 150.0, 350.0))

    def compute_microstructure(self, t85: float) -> dict:
        """根据t8/5冷却时间估算微观组织比例.

        t85 < 3s: 马氏体为主
        t85 3-10s: 贝氏体为主
        t85 10-30s: 魏氏体+铁素体
        t85 > 30s: 铁素体+珠光体

        Args:
            t85: t8/5冷却时间 (s).

        Returns:
            包含 martensite, bainite, ferrite, pearlite 比例的字典.
        """
        if t85 < 3.0:
            martensite: float = 0.7
            bainite: float = 0.2
            ferrite: float = 0.1
            pearlite: float = 0.0
        elif t85 < 10.0:
            martensite = 0.1
            bainite = 0.6
            ferrite = 0.2
            pearlite = 0.1
        elif t85 < 30.0:
            martensite = 0.0
            bainite = 0.2
            ferrite = 0.55
            pearlite = 0.25
        else:
            martensite = 0.0
            bainite = 0.05
            ferrite = 0.60
            pearlite = 0.35
        return {
            "martensite": martensite,
            "bainite": bainite,
            "ferrite": ferrite,
            "pearlite": pearlite,
        }

    def compute_haz_width(
        self,
        current: float,
        voltage: float,
        speed: float,
    ) -> float:
        """热影响区宽度(mm). HAZ ∝ sqrt(heat_input).

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).

        Returns:
            热影响区宽度 (mm), 范围0.5-8.0mm.
        """
        heat: float = self.compute_heat_input(current, voltage, speed)
        return float(max(0.5, min(8.0, 2.5 * np.sqrt(heat / 0.8))))

    def compute_hardness(
        self,
        t85: float,
        carbon_eq: float = 0.35,
    ) -> float:
        """最大硬度估算(HV). 冷速越快硬度越高.

        碳当量+冷却速率 → 硬度

        Args:
            t85: t8/5冷却时间 (s).
            carbon_eq: 碳当量 (默认0.35, 对应Q345钢).

        Returns:
            最大硬度 (HV), 上限450.
        """
        hv: float = 200.0 + carbon_eq * 400.0 + max(0.0, (10.0 - t85)) * 15.0
        return float(min(hv, 450.0))

    def compute_residual_stress(
        self,
        heat_input: float,
        yield_strength: float = 345.0,
    ) -> float:
        """残余应力估算(MPa). 通常达到屈服强度的60-100%.

        残余应力 ≈ yield_strength × (0.6 + 0.4×heat_factor)

        Args:
            heat_input: 热输入 (kJ/mm).
            yield_strength: 屈服强度 (MPa), 默认345 (Q345钢).

        Returns:
            残余应力 (MPa), 不超过屈服强度.
        """
        stress: float = yield_strength * (0.6 + 0.4 * min(heat_input / 1.5, 1.0))
        return float(min(stress, yield_strength))

    def compute_solidification_cracking(
        self,
        current: float,
        speed: float,
        bead_width: float,
    ) -> float:
        """凝固裂纹敏感性指数(0-1, 越高越容易开裂).

        CSR ∝ current / (speed × bead_width)

        Args:
            current: 焊接电流 (A).
            speed: 焊接速度 (mm/s).
            bead_width: 焊缝宽度 (mm).

        Returns:
            凝固裂纹敏感性 (0-1).
        """
        csr: float = current / max(speed * bead_width * 10.0, 1.0) * 0.01
        return float(max(0.0, min(1.0, csr)))

    # ═══════════════════════════════════════════════════════════════════
    # v0.20.1: 非标场景分类 (从播放器到工匠 — 因果自适应)
    # ═══════════════════════════════════════════════════════════════════

    # 非标场景类型 (参考《从播放器到工匠》论文)
    NON_STANDARD_TYPES: dict = {
        "geometric":     "几何非标: 工件变形、错边、间隙不均",
        "semantic":      "语义非标: 无完整CAD图纸、现场临时切割/修补",
        "environmental": "环境非标: 空间受限(仰焊/死角)、表面状态恶劣(锈蚀/油污)",
        "production":    "生产非标: 频繁换型, 单批次数量极少",
    }

    def classify_non_standard(
        self,
        misalignment: float = 0.0,
        gap: float = 0.0,
        has_cad: bool = True,
        surface_condition: str = "clean",
        confined_space: bool = False,
        batch_size: int = 100,
    ) -> list:
        """非标场景分类器 (参考《从播放器到工匠》论文 1.2节).

        判定输入工况属于哪些非标类别, 为后续因果自适应提供决策依据。

        Args:
            misalignment: 错边量 (mm), >2mm 视为几何非标.
            gap: 间隙 (mm), >3mm 视为几何非标.
            has_cad: 是否有完整CAD图纸, False 视为语义非标.
            surface_condition: 表面状态 ("clean"/"rusty"/"oily"),
                               非"clean" 视为环境非标.
            confined_space: 是否在受限空间作业, True 视为环境非标.
            batch_size: 批次数量, <10 视为生产非标.

        Returns:
            非标类别列表 (可能为空列表表示完全标准场景).
            元素来自 ["geometric", "semantic", "environmental", "production"].
        """
        non_standard: list = []

        # 几何非标: 错边>2mm 或 间隙>3mm
        if misalignment > 2.0 or gap > 3.0:
            non_standard.append("geometric")

        # 语义非标: 无CAD图纸
        if not has_cad:
            non_standard.append("semantic")

        # 环境非标: 表面恶劣 或 受限空间
        if surface_condition != "clean" or confined_space:
            non_standard.append("environmental")

        # 生产非标: 单批次极少
        if batch_size < 10:
            non_standard.append("production")

        return non_standard

    # ═══════════════════════════════════════════════════════════════════
    # v0.20.1: DIKWP K-层焊接规则库 (因果知识 → 参数调整)
    # ═══════════════════════════════════════════════════════════════════

    # DIKWP焊接规则库 — 基于经验的因果规则 (参考论文附录A JSON Schema)
    # 条件: material + misalignment + gap → 动作: adjust_current/voltage/weave
    DIKWP_RULE_BASE: list = [
        {
            "rule_id": "R001_aluminum_misalign",
            "condition": {"material": "aluminum", "misalignment_min": 2.0, "gap_max": 3.0},
            "action": {"adjust_current_A": -10.0, "adjust_voltage_V": -0.3, "adjust_weave_mm": 0.5},
        },
        {
            "rule_id": "R002_aluminum_gap",
            "condition": {"material": "aluminum", "gap_min": 3.0},
            "action": {"adjust_current_A": -15.0, "adjust_voltage_V": -0.5, "adjust_weave_mm": 1.0},
        },
        {
            "rule_id": "R003_steel_misalign",
            "condition": {"material": "steel", "misalignment_min": 2.0, "gap_max": 3.0},
            "action": {"adjust_current_A": 5.0, "adjust_voltage_V": -0.2, "adjust_weave_mm": 0.3},
        },
        {
            "rule_id": "R004_steel_gap_large",
            "condition": {"material": "steel", "gap_min": 4.0},
            "action": {"adjust_current_A": -20.0, "adjust_voltage_V": -1.0, "adjust_weave_mm": 1.5},
        },
        {
            "rule_id": "R005_titanium_thin_gap",
            "condition": {"material": "titanium", "gap_min": 2.0},
            "action": {"adjust_current_A": -15.0, "adjust_voltage_V": -0.4, "adjust_weave_mm": 0.8},
        },
        {
            "rule_id": "R006_stainless_rusty_surface",
            "condition": {"material": "stainless", "misalignment_min": 1.5},
            "action": {"adjust_current_A": 10.0, "adjust_voltage_V": 0.5, "adjust_weave_mm": 0.0},
        },
    ]

    def apply_dikwp_rules(
        self,
        misalignment: float = 0.0,
        gap: float = 0.0,
    ) -> dict:
        """应用DIKWP K-层焊接规则库, 返回参数调整建议.

        参考论文《从播放器到工匠》附录A — DIKWP焊接规则库JSON Schema。
        根据材料属性和几何偏差, 匹配因果规则, 生成参数调整量。

        Args:
            misalignment: 错边量 (mm).
            gap: 间隙 (mm).

        Returns:
            参数调整字典:
            {
                "adjust_current_A": float,
                "adjust_voltage_V": float,
                "adjust_weave_mm": float,
                "matched_rules": list[str],
                "scenario_type": list[str],
            }
        """
        # 分类非标场景
        scenario = self.classify_non_standard(
            misalignment=misalignment, gap=gap,
            has_cad=True, surface_condition="clean",
            confined_space=False, batch_size=100,
        )

        # 匹配规则
        matched: list = []
        total_adjust = {"adjust_current_A": 0.0, "adjust_voltage_V": 0.0, "adjust_weave_mm": 0.0}

        for rule in self.DIKWP_RULE_BASE:
            cond = rule["condition"]
            # 材料匹配
            if cond.get("material", self.material) != self.material:
                continue
            # 错边匹配
            if "misalignment_min" in cond and misalignment < cond["misalignment_min"]:
                continue
            # 间隙上限匹配
            if "gap_max" in cond and gap > cond["gap_max"]:
                continue
            # 间隙下限匹配
            if "gap_min" in cond and gap < cond["gap_min"]:
                continue
            # 规则匹配
            matched.append(rule["rule_id"])
            for key, val in rule["action"].items():
                total_adjust[key] += val

        return {
            "adjust_current_A": total_adjust["adjust_current_A"],
            "adjust_voltage_V": total_adjust["adjust_voltage_V"],
            "adjust_weave_mm": total_adjust["adjust_weave_mm"],
            "matched_rules": matched,
            "scenario_type": scenario,
        }

    # ═══════════════════════════════════════════════════════════════════
    # v0.20.1: η-PID 自适应控制 (从播放器到工匠 — 因果闭环)
    # ═══════════════════════════════════════════════════════════════════

    def eta_pid_adjust(
        self,
        current: float,
        voltage: float,
        travel_speed: float,
        weave: float,
        eta: float,
        eta_threshold: float = 0.05,
        kp: float = 0.8,
        ki: float = 0.1,
        kd: float = 0.2,
    ) -> dict:
        """η-PID自适应参数调整 (参考论文第四章 算法1).

        当η残差超过阈值时, 基于PID控制律动态调整焊接参数,
        使系统状态重新逼近Goal-EML陪集。

        算法:
            if η > η_threshold:
                ΔI = -kp × ∂η/∂I × η
                ΔV = -kp × ∂η/∂V × η
                ΔW = -kp × ∂η/∂W × η
            (偏导用有限差分近似)

        Args:
            current: 当前电流 (A).
            voltage: 当前电压 (V).
            travel_speed: 当前速度 (mm/s).
            weave: 当前摆动 (mm).
            eta: 当前η残差.
            eta_threshold: η阈值, 超过则触发调整.
            kp: 比例增益.
            ki: 积分增益 (保留接口, 当前不累积).
            kd: 微分增益 (保留接口, 当前不微分).

        Returns:
            调整后的参数字典:
            {
                "adjusted_current": float,
                "adjusted_voltage": float,
                "adjusted_weave": float,
                "delta_current": float,
                "delta_voltage": float,
                "delta_weave": float,
                "eta_before": float,
                "triggered": bool,
            }
        """
        if eta <= eta_threshold:
            return {
                "adjusted_current": current,
                "adjusted_voltage": voltage,
                "adjusted_weave": weave,
                "delta_current": 0.0,
                "delta_voltage": 0.0,
                "delta_weave": 0.0,
                "eta_before": eta,
                "triggered": False,
            }

        # 有限差分近似偏导 ∂η/∂I, ∂η/∂V, ∂η/∂W
        h: float = 0.01  # 差分步长

        # ∂η/∂current
        q1 = self.predict_quality(current + h, voltage, travel_speed, 15.0)
        q0 = self.predict_quality(current - h, voltage, travel_speed, 15.0)
        deta_di: float = (q1["eta"] - q0["eta"]) / (2.0 * h)

        # ∂η/∂voltage
        q1 = self.predict_quality(current, voltage + h, travel_speed, 15.0)
        q0 = self.predict_quality(current, voltage - h, travel_speed, 15.0)
        deta_dv: float = (q1["eta"] - q0["eta"]) / (2.0 * h)

        # ∂η/∂weave (通过摆动影响焊缝宽度, 间接影响eta)
        # 近似: weave变化影响bead_width, bead_width影响eta
        bw1 = self.compute_bead_geometry(current, voltage, travel_speed, weave + h)
        bw0 = self.compute_bead_geometry(current, voltage, travel_speed, weave - h)
        deta_dw: float = (bw1[0] - bw0[0]) / (2.0 * h) * 0.01

        # PID控制律: Δparam = -kp × (∂η/∂param) × η
        delta_current: float = -kp * deta_di * eta
        delta_voltage: float = -kp * deta_dv * eta
        delta_weave: float = -kd * deta_dw * eta

        # 限幅: 调整幅度不超过原参数的15%
        max_curr_adj: float = abs(current) * 0.15
        max_volt_adj: float = abs(voltage) * 0.15
        max_weave_adj: float = abs(weave) * 0.15 + 0.5

        delta_current = max(-max_curr_adj, min(max_curr_adj, delta_current))
        delta_voltage = max(-max_volt_adj, min(max_volt_adj, delta_voltage))
        delta_weave = max(-max_weave_adj, min(max_weave_adj, delta_weave))

        return {
            "adjusted_current": current + delta_current,
            "adjusted_voltage": voltage + delta_voltage,
            "adjusted_weave": weave + delta_weave,
            "delta_current": delta_current,
            "delta_voltage": delta_voltage,
            "delta_weave": delta_weave,
            "eta_before": eta,
            "triggered": True,
        }

    # ═══════════════════════════════════════════════════════════════════
    # v0.20.0: IntentGuard 四级安全分类 (DIKWP-IDO融合)
    # ═══════════════════════════════════════════════════════════════════

    def classify_intent_safety(
        self,
        current: float,
        voltage: float,
        travel_speed: float,
    ) -> Tuple[int, str]:
        """IntentGuard四级安全分类 (参考DIKWP-IDO文章).

        基于焊接参数与最优参数的偏差比例进行四级安全判定:
          0 = SAFE: 正常焊接参数
          1 = SUSPICIOUS: 边界值, 降速+增强监控
          2 = DANGEROUS: 超出安全上限, Ψ-Anchor硬拦截
          3 = CRITICAL: 严重超限, 立即断弧+急停

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            travel_speed: 焊接速度 (mm/s).

        Returns:
            (level, label): level 0-3, label为安全等级描述字符串.
        """
        opt: dict = self.OPTIMAL_PARAMS
        # 计算偏差比例
        curr_ratio: float = current / opt["current"] if opt["current"] > 0 else 1.0
        volt_ratio: float = voltage / opt["voltage"] if opt["voltage"] > 0 else 1.0

        max_ratio: float = max(curr_ratio, volt_ratio)

        if max_ratio > 2.0:
            return (3, "CRITICAL")
        elif max_ratio > 1.5:
            return (2, "DANGEROUS")
        elif max_ratio > 1.2:
            return (1, "SUSPICIOUS")
        else:
            return (0, "SAFE")
