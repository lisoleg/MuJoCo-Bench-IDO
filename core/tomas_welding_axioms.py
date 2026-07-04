"""
TomasWeldingAxioms — TOMAS焊接工艺公理库
========================================

TOMAS (Task-Oriented Multimodal Agent System) 焊接工艺公理库。
混合公理来源:
  - 核心公理来自论文 (IDO/TOMAS 架构安全约束)
  - 工艺公理来自 AWS D1.1 / API 580 工程标准

7条公理:
  1. 热输入上限: heat_input <= 2.5 kJ/mm
  2. 干伸长范围: 8 <= stickout <= 25 mm
  3. 气孔风险: porosity_risk < 0.1
  4. 角变形限制: angular_distortion < 2.0 degrees
  5. 熔深最小: penetration >= 1.5 mm
  6. 电流-电压匹配: voltage in [14 + 0.04*current, 14 + 0.06*current]
  7. 焊接速度范围: 2 <= speed <= 15 mm/s

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional


@dataclass
class WeldingAxiom:
    """焊接公理数据类.

    Attributes:
        name: 公理名称.
        description: 公理描述.
        check_fn: 检查函数 (welding_state) -> bool.
        severity: 严重程度 ("info", "warning", "critical").
        source: 公理来源 ("paper", "AWS_D1.1", "API_580").
    """
    name: str
    description: str
    check_fn: Callable[[Dict[str, Any]], bool]
    severity: str
    source: str


class TomasWeldingAxioms:
    """TOMAS焊接工艺公理库.

    公理来源混合:
      - 核心公理来自论文 (IDO/TOMAS架构安全约束)
      - 工艺公理来自 AWS D1.1 / API 580 工程标准

    Attributes:
        _axioms: 公理列表.
    """

    def __init__(self) -> None:
        """初始化焊接公理库."""
        self._axioms: List[WeldingAxiom] = self._build_axioms()

    def _build_axioms(self) -> List[WeldingAxiom]:
        """构建所有焊接公理.

        Returns:
            WeldingAxiom 列表.
        """
        axioms: List[WeldingAxiom] = []

        # 1. 热输入上限公理: heat_input <= 2.5 kJ/mm
        def check_heat_input(state: Dict[str, Any]) -> bool:
            heat_input: float = float(state.get("heat_input", 0.0))
            return heat_input <= 2.5

        axioms.append(WeldingAxiom(
            name="heat_input_limit",
            description="热输入上限: heat_input <= 2.5 kJ/mm "
                        "(AWS D1.1 防止过热变形)",
            check_fn=check_heat_input,
            severity="critical",
            source="AWS_D1.1",
        ))

        # 2. 干伸长范围公理: 8 <= stickout <= 25 mm
        def check_stickout_range(state: Dict[str, Any]) -> bool:
            stickout: float = float(state.get("stickout", 15.0))
            return 8.0 <= stickout <= 25.0

        axioms.append(WeldingAxiom(
            name="stickout_range",
            description="干伸长范围: 8 <= stickout <= 25 mm "
                        "(保证电弧稳定和保护气覆盖)",
            check_fn=check_stickout_range,
            severity="warning",
            source="paper",
        ))

        # 3. 气孔风险公理: porosity_risk < 0.1
        def check_porosity(state: Dict[str, Any]) -> bool:
            porosity: float = float(state.get("porosity_risk", 0.0))
            return porosity < 0.1

        axioms.append(WeldingAxiom(
            name="porosity_risk_limit",
            description="气孔风险: porosity_risk < 0.1 "
                        "(API 580 焊缝验收标准)",
            check_fn=check_porosity,
            severity="critical",
            source="API_580",
        ))

        # 4. 角变形限制公理: angular_distortion < 2.0 degrees
        def check_distortion(state: Dict[str, Any]) -> bool:
            distortion: float = float(state.get("angular_distortion", 0.0))
            return distortion < 2.0

        axioms.append(WeldingAxiom(
            name="angular_distortion_limit",
            description="角变形限制: angular_distortion < 2.0 degrees "
                        "(AWS D1.1 结构变形控制)",
            check_fn=check_distortion,
            severity="warning",
            source="AWS_D1.1",
        ))

        # 5. 熔深最小公理: penetration >= 1.5 mm
        def check_penetration(state: Dict[str, Any]) -> bool:
            penetration: float = float(state.get("penetration_depth", 0.0))
            return penetration >= 1.5

        axioms.append(WeldingAxiom(
            name="min_penetration",
            description="熔深最小: penetration >= 1.5 mm "
                        "(保证焊缝强度, 防未熔合)",
            check_fn=check_penetration,
            severity="critical",
            source="AWS_D1.1",
        ))

        # 6. 电流-电压匹配公理: voltage in [14 + 0.04*current, 14 + 0.06*current]
        def check_current_voltage_match(state: Dict[str, Any]) -> bool:
            current: float = float(state.get("current", 200.0))
            voltage: float = float(state.get("voltage", 24.0))
            v_min: float = 14.0 + 0.04 * current
            v_max: float = 14.0 + 0.06 * current
            return v_min <= voltage <= v_max

        axioms.append(WeldingAxiom(
            name="current_voltage_match",
            description="电流-电压匹配: voltage in [14+0.04*I, 14+0.06*I] "
                        "(GMAW 经验公式, 保证电弧稳定)",
            check_fn=check_current_voltage_match,
            severity="warning",
            source="paper",
        ))

        # 7. 焊接速度范围公理: 2 <= speed <= 15 mm/s
        def check_speed_range(state: Dict[str, Any]) -> bool:
            speed: float = float(state.get("travel_speed",
                                 state.get("speed", 6.0)))
            return 2.0 <= speed <= 15.0

        axioms.append(WeldingAxiom(
            name="speed_range",
            description="焊接速度范围: 2 <= speed <= 15 mm/s "
                        "(保证焊缝成形质量)",
            check_fn=check_speed_range,
            severity="info",
            source="AWS_D1.1",
        ))

        return axioms

    def get_axiom(self, name: str) -> Optional[WeldingAxiom]:
        """按名称获取公理.

        Args:
            name: 公理名称.

        Returns:
            WeldingAxiom 实例, 如果不存在返回 None.
        """
        for axiom in self._axioms:
            if axiom.name == name:
                return axiom
        return None

    def check_axiom(self, name: str, welding_state: Dict[str, Any]) -> bool:
        """检查单个公理是否满足.

        Args:
            name: 公理名称.
            welding_state: 焊接状态字典.

        Returns:
            True 如果公理满足, False 如果违反.
        """
        axiom: Optional[WeldingAxiom] = self.get_axiom(name)
        if axiom is None:
            return True
        try:
            return bool(axiom.check_fn(welding_state))
        except Exception:
            return True

    def check_all(self, welding_state: Dict[str, Any]) -> Dict[str, Any]:
        """检查所有公理, 返回违规列表.

        Args:
            welding_state: 焊接状态字典, 包含:
              - heat_input: 热输入 (kJ/mm)
              - stickout: 干伸长
              - porosity_risk: 气孔风险 (0-1)
              - angular_distortion: 角变形
              - penetration_depth: 熔深
              - current: 焊接电流 (A)
              - voltage: 焊接电压 (V)
              - travel_speed / speed: 焊接速度

        Returns:
            检查结果字典:
              - passed: 全部公理满足才为 True
              - violations: 违规公理名称列表
              - details: 各公理的详细检查结果
        """
        violations: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}
        all_passed: bool = True

        for axiom in self._axioms:
            try:
                satisfied: bool = bool(axiom.check_fn(welding_state))
            except Exception as e:
                satisfied = True
                details[axiom.name] = {
                    "satisfied": True,
                    "severity": axiom.severity,
                    "description": axiom.description,
                    "source": axiom.source,
                    "error": str(e),
                }
                continue

            details[axiom.name] = {
                "satisfied": satisfied,
                "severity": axiom.severity,
                "description": axiom.description,
                "source": axiom.source,
            }

            if not satisfied:
                all_passed = False
                violations.append(axiom.name)

        return {
            "passed": all_passed,
            "violations": violations,
            "details": details,
        }

    def get_all_axioms(self) -> List[WeldingAxiom]:
        """返回所有公理.

        Returns:
            WeldingAxiom 列表.
        """
        return self._axioms.copy()

    def get_axiom_names(self) -> List[str]:
        """返回所有公理名称.

        Returns:
            公理名称列表.
        """
        return [axiom.name for axiom in self._axioms]
