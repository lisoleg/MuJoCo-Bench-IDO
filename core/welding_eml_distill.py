"""
WeldingEMLDistiller — EML焊接域Pareto蒸馏
==========================================

从DreamerV3训练结果提取Pareto最优工艺参数集,
蒸馏为GoalEML超图节点。

Pareto目标: min(eta, porosity, distortion), max(penetration)

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# 添加项目根路径
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.welding_process_proxy import WeldingProcessProxy, WeldingQuality


class WeldingEMLDistiller:
    """EML焊接域蒸馏器.

    从DreamerV3训练结果提取Pareto最优工艺参数集,
    蒸馏为GoalEML超图节点。

    Pareto目标: min(eta, porosity, distortion), max(penetration)

    Attributes:
        proxy: WeldingProcessProxy 实例.
        eml: GoalEML 实例 (可选).
        _pareto_front: Pareto前沿参数列表.
    """

    def __init__(
        self,
        process_proxy: Optional[WeldingProcessProxy] = None,
        eml: Optional[Any] = None,
    ) -> None:
        """初始化EML蒸馏器.

        Args:
            process_proxy: WeldingProcessProxy 实例, None=新建.
            eml: GoalEML 实例 (可选).
        """
        self.proxy: WeldingProcessProxy = process_proxy or WeldingProcessProxy()
        self.eml: Optional[Any] = eml
        self._pareto_front: List[Dict[str, Any]] = []

    def search_pareto_optimal(
        self,
        n_trials: int = 1000,
        weld_type: str = "flat",
    ) -> List[Dict[str, Any]]:
        """Pareto最优参数搜索.

        网格搜索 + 随机搜索混合:
          - 网格: current(50-350, step 50) × voltage(14-32, step 2) × speed(2-15, step 1)
          - 随机: 在网格点附近随机扰动

        Args:
            n_trials: 总试验次数 (网格 + 随机).
            weld_type: 焊接姿态类型.

        Returns:
            Pareto前沿参数列表, 每个元素包含 params 和 quality.
        """
        self._pareto_front = []
        self.proxy.weld_type = weld_type

        # ── 网格搜索 ──
        current_grid: np.ndarray = np.arange(50, 351, 50)  # 50-350, step 50
        voltage_grid: np.ndarray = np.arange(14, 33, 2)     # 14-32, step 2
        speed_grid: np.ndarray = np.arange(2, 16, 1)        # 2-15, step 1
        stickout_options: List[float] = [10.0, 12.0, 15.0, 18.0, 20.0]

        # 计算网格点数量
        n_grid: int = (len(current_grid) * len(voltage_grid)
                       * len(speed_grid) * len(stickout_options))
        n_random: int = max(0, n_trials - n_grid)

        # 网格搜索
        for current in current_grid:
            for voltage in voltage_grid:
                for speed in speed_grid:
                    for stickout in stickout_options:
                        quality: WeldingQuality = self._evaluate_params(
                            float(current), float(voltage),
                            float(speed), float(stickout),
                            weld_type,
                        )
                        self._try_add_to_pareto(
                            float(current), float(voltage),
                            float(speed), float(stickout),
                            quality,
                        )

        # ── 随机搜索: 在网格点附近扰动 ──
        for _ in range(n_random):
            current: float = float(np.random.choice(current_grid))
            voltage: float = float(np.random.choice(voltage_grid))
            speed: float = float(np.random.choice(speed_grid))
            stickout: float = float(np.random.choice(stickout_options))

            # 添加随机扰动
            current += np.random.uniform(-25, 25)
            voltage += np.random.uniform(-1, 1)
            speed += np.random.uniform(-0.5, 0.5)
            stickout += np.random.uniform(-2, 2)

            # 裁剪到有效范围
            current = float(np.clip(current, 50, 350))
            voltage = float(np.clip(voltage, 14, 32))
            speed = float(np.clip(speed, 2, 15))
            stickout = float(np.clip(stickout, 8, 25))

            quality = self._evaluate_params(
                current, voltage, speed, stickout, weld_type,
            )
            self._try_add_to_pareto(
                current, voltage, speed, stickout, quality,
            )

        return self._pareto_front.copy()

    def _evaluate_params(
        self,
        current: float,
        voltage: float,
        speed: float,
        stickout: float,
        weld_type: str,
    ) -> WeldingQuality:
        """评估一组参数, 返回quality.

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度.
            stickout: 干伸长.
            weld_type: 焊接姿态类型.

        Returns:
            WeldingQuality 实例.
        """
        self.proxy.weld_type = weld_type
        return self.proxy.predict(
            current=current,
            voltage=voltage,
            travel_speed=speed,
            stickout=stickout,
        )

    def _try_add_to_pareto(
        self,
        current: float,
        voltage: float,
        speed: float,
        stickout: float,
        quality: WeldingQuality,
    ) -> None:
        """尝试将参数加入Pareto前沿.

        如果新参数Pareto支配现有前沿中的某些点, 则替换被支配的点。
        如果新参数被现有前沿中的任何点支配, 则不加入。

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度.
            stickout: 干伸长.
            quality: 焊接质量.
        """
        candidate: Dict[str, Any] = {
            "params": {
                "current": current,
                "voltage": voltage,
                "speed": speed,
                "stickout": stickout,
            },
            "quality": {
                "eta": quality.eta_residual,
                "porosity": quality.porosity_risk,
                "distortion": quality.angular_distortion,
                "penetration": quality.penetration_depth,
                "heat_input": quality.heat_input,
                "arc_length": quality.arc_length,
            },
        }

        # 检查是否被现有前沿中的任何点支配
        is_dominated: bool = False
        to_remove: List[int] = []

        for i, existing in enumerate(self._pareto_front):
            if self._is_dominant(existing, candidate):
                is_dominated = True
                break
            if self._is_dominant(candidate, existing):
                to_remove.append(i)

        if not is_dominated:
            # 移除被支配的点
            for i in sorted(to_remove, reverse=True):
                self._pareto_front.pop(i)
            self._pareto_front.append(candidate)

        # 限制前沿大小
        if len(self._pareto_front) > 100:
            # 保留eta最低的100个
            self._pareto_front.sort(
                key=lambda x: x["quality"]["eta"]
            )
            self._pareto_front = self._pareto_front[:100]

    def _is_dominant(
        self,
        candidate: Dict[str, Any],
        existing: Dict[str, Any],
    ) -> bool:
        """判断candidate是否Pareto支配existing.

        Pareto支配定义 (min eta, min porosity, min distortion, max penetration):
        candidate支配existing当且仅当:
        - candidate在所有目标上不劣于existing
        - candidate在至少一个目标上严格优于existing

        Args:
            candidate: 候选参数点.
            existing: 现有参数点.

        Returns:
            True 如果candidate支配existing.
        """
        c_q: Dict[str, float] = candidate["quality"]
        e_q: Dict[str, float] = existing["quality"]

        # min目标: eta, porosity, distortion (越小越好)
        c_min_better_or_equal: bool = (
            c_q["eta"] <= e_q["eta"]
            and c_q["porosity"] <= e_q["porosity"]
            and c_q["distortion"] <= e_q["distortion"]
        )
        # max目标: penetration (越大越好)
        c_max_better_or_equal: bool = c_q["penetration"] >= e_q["penetration"]

        # 至少一个严格更好
        c_strictly_better: bool = (
            c_q["eta"] < e_q["eta"]
            or c_q["porosity"] < e_q["porosity"]
            or c_q["distortion"] < e_q["distortion"]
            or c_q["penetration"] > e_q["penetration"]
        )

        return c_min_better_or_equal and c_max_better_or_equal and c_strictly_better

    def distill_to_eml(
        self,
        pareto_params: Optional[List[Dict[str, Any]]] = None,
        weld_type: str = "flat",
    ) -> List[Dict[str, Any]]:
        """将Pareto最优参数蒸馏到EML超图节点.

        每个Pareto最优点 → 一个EML节点:
        {
            "weld_type": weld_type,
            "params": {current, voltage, speed, stickout},
            "quality": {eta, porosity, distortion, penetration},
            "eml_node_id": "weld_pareto_{idx}",
        }

        Args:
            pareto_params: Pareto前沿参数列表, None=使用已搜索的.
            weld_type: 焊接姿态类型.

        Returns:
            EML节点列表.
        """
        if pareto_params is None:
            pareto_params = self._pareto_front

        if len(pareto_params) == 0:
            pareto_params = self.search_pareto_optimal(weld_type=weld_type)

        eml_nodes: List[Dict[str, Any]] = []
        for idx, point in enumerate(pareto_params):
            node: Dict[str, Any] = {
                "weld_type": weld_type,
                "params": point["params"].copy(),
                "quality": point["quality"].copy(),
                "eml_node_id": f"weld_pareto_{idx}",
            }
            eml_nodes.append(node)

        # 如果有 EML 实例, 尝试添加节点
        if self.eml is not None:
            # GoalEML 没有 add_node 方法, 但我们可以记录在 extra 字段
            if hasattr(self.eml, "invariants"):
                # 将Pareto最优点添加为不变量描述
                for node in eml_nodes:
                    inv_name: str = (
                        f"pareto_{node['eml_node_id']}_"
                        f"I{int(node['params']['current'])}_"
                        f"V{int(node['params']['voltage'])}_"
                        f"S{int(node['params']['speed'])}"
                    )
                    if inv_name not in self.eml.invariants:
                        self.eml.invariants.append(inv_name)

        return eml_nodes

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """返回当前Pareto前沿.

        Returns:
            Pareto前沿参数列表.
        """
        return self._pareto_front.copy()

    def compute_pareto_front(
        self,
        n_trials: int = 1000,
        weld_type: str = "flat",
    ) -> List[Dict[str, Any]]:
        """计算并返回Pareto前沿 (search + distill的便捷方法).

        Args:
            n_trials: 总试验次数.
            weld_type: 焊接姿态类型.

        Returns:
            EML节点列表.
        """
        self.search_pareto_optimal(n_trials=n_trials, weld_type=weld_type)
        return self.distill_to_eml(weld_type=weld_type)

    def get_best_params(
        self,
        weld_type: str = "flat",
        n_trials: int = 500,
    ) -> Dict[str, Any]:
        """获取最佳参数 (eta最低的Pareto点).

        Args:
            weld_type: 焊接姿态类型.
            n_trials: 搜索试验次数.

        Returns:
            最佳参数字典.
        """
        if len(self._pareto_front) == 0:
            self.search_pareto_optimal(n_trials=n_trials, weld_type=weld_type)

        if len(self._pareto_front) == 0:
            return {
                "params": {"current": 200.0, "voltage": 24.0,
                           "speed": 6.0, "stickout": 15.0},
                "quality": {"eta": 0.0, "porosity": 0.0,
                           "distortion": 0.0, "penetration": 0.0},
            }

        # 选择eta最低的点
        best: Dict[str, Any] = min(
            self._pareto_front, key=lambda x: x["quality"]["eta"]
        )
        return best

    def summarize_pareto(self) -> Dict[str, Any]:
        """生成Pareto前沿的统计摘要.

        Returns:
            统计摘要字典.
        """
        if len(self._pareto_front) == 0:
            return {"n_points": 0}

        etas: List[float] = [p["quality"]["eta"] for p in self._pareto_front]
        porosities: List[float] = [p["quality"]["porosity"] for p in self._pareto_front]
        penetrations: List[float] = [p["quality"]["penetration"] for p in self._pareto_front]

        return {
            "n_points": len(self._pareto_front),
            "eta_range": [float(min(etas)), float(max(etas))],
            "eta_mean": float(np.mean(etas)),
            "porosity_range": [float(min(porosities)), float(max(porosities))],
            "porosity_mean": float(np.mean(porosities)),
            "penetration_range": [float(min(penetrations)), float(max(penetrations))],
            "penetration_mean": float(np.mean(penetrations)),
        }
