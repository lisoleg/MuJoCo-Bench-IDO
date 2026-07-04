"""
WeldingPsiAnchor — 焊接安全门控 (ψ-Anchor 焊接扩展)
====================================================

实现焊接专用的安全约束检查，作为 IDO 架构中 Ψ-Check (C-Layer)
的焊接域扩展。

安全约束:
  1. 干伸长检查 (check_stick_out): stickout 在 [8, 25] mm 范围内
     - > 25mm → warning, 降低送丝速度
     - < 8mm  → critical, 紧急停止送丝

  2. 回烧检查 (check_burn_back): 防止焊丝回烧到导电嘴
     - current > MAX_CURRENT 且 voltage < MIN_VOLTAGE → critical

  3. 气孔风险检查 (check_porosity_risk): 电弧长度方差过大
     - arc_var > threshold → warning, 触发摆动焊接
     - arc_var > 2×threshold → critical, 降低电流

Author: MuJoCo-Bench-IDO Welding Module v0.1.0
"""

from typing import Dict, Any, Optional, List

# ── 焊接安全阈值 ──
WELDING_SAFETY_THRESHOLDS: Dict[str, float] = {
    "STICK_OUT_MIN": 2.0,         # 最小干伸长 (mm) — 仿真放宽
    "STICK_OUT_MAX": 35.0,        # 最大干伸长 (mm) — 仿真放宽
    "MAX_CURRENT": 350.0,        # 最大焊接电流 (A)
    "MIN_VOLTAGE": 5.0,          # 最小焊接电压 (V) — 低于此值视为回烧
    "ARC_VAR_THRESHOLD": 0.5,    # 电弧长度方差阈值
    "MAX_HEAT_INPUT": 2.5,       # 最大热输入 (kJ/mm)
    "SEAM_DEV_MAX": 2.0,         # 最大焊缝偏差 (mm) — 仿真放宽
}


class WeldingPsiAnchor:
    """焊接安全门控 — Ψ-Anchor 的焊接域扩展.

    在每次焊接 step 前执行安全约束检查，确保焊接参数在安全范围内。
    任何 critical 级别的违规都会阻止当前 step 的执行。

    Attributes:
        thresholds: 安全阈值字典.
        _arc_length_history: 电弧长度历史 (用于方差计算).
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        """初始化焊接安全锚.

        Args:
            thresholds: 自定义安全阈值. 如果为 None, 使用默认 WELDING_SAFETY_THRESHOLDS.
        """
        self.thresholds: Dict[str, float] = thresholds if thresholds is not None else WELDING_SAFETY_THRESHOLDS.copy()
        self._arc_length_history: List[float] = []

    def check_stick_out(self, stickout_mm: float) -> Dict[str, Any]:
        """检查干伸长是否在安全范围内.

        判定逻辑:
          - stickout > STICK_OUT_MAX (25mm) → warning, 降低送丝速度
          - stickout < STICK_OUT_MIN (8mm)  → critical, 紧急停止送丝
          - 否则 → passed

        Args:
            stickout_mm: 当前干伸长 (mm).

        Returns:
            检查结果字典:
              - passed: 是否通过
              - violation: 违规描述 (None 表示无违规)
              - severity: 严重程度 ("none" / "warning" / "critical")
        """
        max_stick: float = self.thresholds["STICK_OUT_MAX"]
        min_stick: float = self.thresholds["STICK_OUT_MIN"]

        if stickout_mm < min_stick:
            return {
                "passed": False,
                "violation": f"Stickout {stickout_mm:.1f}mm < MIN {min_stick}mm — "
                             f"emergency stop wire feed (回烧风险)",
                "severity": "critical",
            }

        if stickout_mm > max_stick:
            return {
                "passed": False,
                "violation": f"Stickout {stickout_mm:.1f}mm > MAX {max_stick}mm — "
                             f"reduce wire feed speed (电弧不稳风险)",
                "severity": "warning",
            }

        return {
            "passed": True,
            "violation": None,
            "severity": "none",
        }

    def check_burn_back(self, current: float, voltage: float, stickout_mm: float) -> Dict[str, Any]:
        """检查回烧风险 (焊丝熔断回烧到导电嘴).

        判定逻辑:
          - current > MAX_CURRENT 且 voltage < MIN_VOLTAGE → critical
            (高电流 + 低电压 = 焊丝在导电嘴处熔断 = 回烧)
          - 否则 → passed

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            stickout_mm: 干伸长 (mm) — 用于辅助判断.

        Returns:
            检查结果字典.
        """
        max_current: float = self.thresholds["MAX_CURRENT"]
        min_voltage: float = self.thresholds["MIN_VOLTAGE"]

        if current > max_current and voltage < min_voltage:
            return {
                "passed": False,
                "violation": f"Burn-back risk: current={current:.1f}A > {max_current}A "
                             f"AND voltage={voltage:.1f}V < {min_voltage}V — "
                             f"stop wire feed + emergency stop",
                "severity": "critical",
            }

        return {
            "passed": True,
            "violation": None,
            "severity": "none",
        }

    def check_porosity_risk(self, arc_length_variance: float) -> Dict[str, Any]:
        """检查气孔风险 (电弧长度方差过大导致保护气体紊乱).

        判定逻辑:
          - arc_var > 2 × ARC_VAR_THRESHOLD → critical, 降低电流
          - arc_var > ARC_VAR_THRESHOLD     → warning, 触发摆动焊接
          - 否则 → passed

        Args:
            arc_length_variance: 电弧长度方差.

        Returns:
            检查结果字典.
        """
        threshold: float = self.thresholds["ARC_VAR_THRESHOLD"]
        critical_threshold: float = 2.0 * threshold

        if arc_length_variance > critical_threshold:
            return {
                "passed": False,
                "violation": f"Porosity risk: arc_var={arc_length_variance:.3f} > "
                             f"{critical_threshold:.3f} (2×threshold) — "
                             f"reduce current to stabilize arc",
                "severity": "critical",
            }

        if arc_length_variance > threshold:
            return {
                "passed": False,
                "violation": f"Porosity risk: arc_var={arc_length_variance:.3f} > "
                             f"{threshold:.3f} — trigger weave welding",
                "severity": "warning",
            }

        return {
            "passed": True,
            "violation": None,
            "severity": "none",
        }

    def check_all(self, welding_state: Dict[str, Any]) -> Dict[str, Any]:
        """执行所有焊接安全检查并汇总结果.

        Args:
            welding_state: 焊接状态字典, 包含:
              - stickout: 干伸长 (mm)
              - current: 焊接电流 (A)
              - voltage: 焊接电压 (V)
              - arc_length_variance: 电弧长度方差
              - seam_deviation: 焊缝偏差 (mm)
              - temperature: 温度 (°C)
              - contact_force: 接触力 (list of 3 floats)

        Returns:
            汇总结果字典:
              - passed: 全部检查通过才为 True
              - violations: 违规描述列表
              - actions: 建议动作列表
              - details: 各项检查的详细结果
        """
        stickout: float = float(welding_state.get("stickout", 15.0))
        current: float = float(welding_state.get("current", 200.0))
        voltage: float = float(welding_state.get("voltage", 24.0))
        arc_var: float = float(welding_state.get("arc_length_variance", 0.0))
        seam_dev: float = float(welding_state.get("seam_deviation", 0.0))

        # 执行三项检查
        stick_result: Dict[str, Any] = self.check_stick_out(stickout)
        burn_result: Dict[str, Any] = self.check_burn_back(current, voltage, stickout)
        porosity_result: Dict[str, Any] = self.check_porosity_risk(arc_var)

        # 焊缝偏差检查 (附加)
        seam_result: Dict[str, Any] = {"passed": True, "violation": None, "severity": "none"}
        if seam_dev > self.thresholds["SEAM_DEV_MAX"]:
            seam_result = {
                "passed": False,
                "violation": f"Seam deviation {seam_dev:.2f}mm > "
                             f"{self.thresholds['SEAM_DEV_MAX']}mm — adjust tracking",
                "severity": "warning",
            }

        # 汇总
        all_results: List[Dict[str, Any]] = [
            ("stick_out", stick_result),
            ("burn_back", burn_result),
            ("porosity", porosity_result),
            ("seam_dev", seam_result),
        ]

        violations: List[str] = []
        actions: List[str] = []
        all_passed: bool = True

        for check_name, result in all_results:
            if not result["passed"]:
                violations.append(result["violation"])
                # 只有 critical 级别阻止焊接步骤, warning 仅记录
                if result["severity"] == "critical":
                    all_passed = False
                # 根据检查类型建议动作
                if check_name == "stick_out":
                    if result["severity"] == "critical":
                        actions.append("EMERGENCY_STOP_WIRE_FEED")
                    else:
                        actions.append("REDUCE_WIRE_SPEED")
                elif check_name == "burn_back":
                    actions.append("STOP_WIRE_FEED_AND_EMERGENCY_STOP")
                elif check_name == "porosity":
                    if result["severity"] == "critical":
                        actions.append("REDUCE_CURRENT")
                    else:
                        actions.append("TRIGGER_WEAVE_WELDING")
                elif check_name == "seam_dev":
                    actions.append("ADJUST_TRACKING")

        return {
            "passed": all_passed,
            "violations": violations,
            "actions": actions,
            "details": {
                "stick_out": stick_result,
                "burn_back": burn_result,
                "porosity": porosity_result,
                "seam_dev": seam_result,
            },
        }

    def update_arc_history(self, arc_length: float) -> None:
        """更新电弧长度历史记录.

        Args:
            arc_length: 当前电弧长度 (mm).
        """
        self._arc_length_history.append(arc_length)
        # 保留最近 50 个记录
        if len(self._arc_length_history) > 50:
            self._arc_length_history = self._arc_length_history[-50:]

    def compute_arc_variance(self) -> float:
        """计算电弧长度方差.

        Returns:
            电弧长度方差. 如果历史不足 2 个记录, 返回 0.
        """
        if len(self._arc_length_history) < 2:
            return 0.0
        import numpy as np
        return float(np.var(self._arc_length_history))
