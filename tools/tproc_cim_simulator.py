"""
TProcCIMSimulator — T-Processor CIM忆阻器模拟器
================================================

模拟基于忆阻器的存内计算(CIM)阵列，对比SRAM+ALU能耗。

CIM优势: 矩阵向量乘法在忆阻器交叉阵列中一步完成，无需读写SRAM。
对比基准: SRAM+ALU = 335.36 pJ (8x8矩阵向量乘法)

参考: 章锋论文附录D

Author: MuJoCo-Bench-IDO v0.3.0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "MemristorModel",
    "CrossbarArray",
    "SRAM_ALU_ENERGY_PJ",
    "run_energy_comparison",
    "main",
]


#: SRAM+ALU基准能耗 (8×8矩阵向量乘法, 28nm工艺)
SRAM_ALU_ENERGY_PJ: float = 335.36


@dataclass
class MemristorModel:
    """忆阻器模型.

    Attributes:
        g: 当前电导.
        g_on: 导通电导.
        g_off: 关断电导.
    """
    g_on: float = 1e-3   # S (Siemens)
    g_off: float = 1e-6  # S

    def __post_init__(self) -> None:
        self.g: float = self.g_off

    def set(self) -> None:
        """置位 (导通)."""
        self.g = self.g_on

    def reset(self) -> None:
        """复位 (关断)."""
        self.g = self.g_off

    def energy(self, v: float, dt: float) -> float:
        """计算单次读取能耗.

        E = G × V² × dt

        Args:
            v: 读取电压.
            dt: 读取时间.

        Returns:
            能量.
        """
        return self.g * v ** 2 * dt


class CrossbarArray:
    """忆阻器交叉阵列 (N×N).

    用于存内计算矩阵向量乘法: y = W × x
    其中W由忆阻器电导矩阵表示, x由输入电压向量表示.

    Attributes:
        n: 阵列维度 (N×N).
        memristors: N×N忆阻器矩阵.
    """

    def __init__(self, n: int = 8) -> None:
        """初始化N×N交叉阵列.

        Args:
            n: 阵列维度.
        """
        self.n: int = n
        self.memristors: list[list[MemristorModel]] = [
            [MemristorModel() for _ in range(n)]
            for _ in range(n)
        ]

    def set_weight(self, i: int, j: int, value: bool) -> None:
        """设置权重.

        Args:
            i: 行索引.
            j: 列索引.
            value: True=导通, False=关断.
        """
        if value:
            self.memristors[i][j].set()
        else:
            self.memristors[i][j].reset()

    def matrix_vector_mult(
        self,
        vec: np.ndarray,
        v_read: float = 0.1,
        dt: float = 1e-9,
    ) -> Tuple[np.ndarray, float]:
        """矩阵向量乘法 (存内计算).

        y[i] = Σ_j G[i,j] × x[j] × v_read
        E = Σ_{i,j} G[i,j] × v_read² × dt

        Args:
            vec: 输入向量, 长度为n.
            v_read: 读取电压.
            dt: 读取时间.

        Returns:
            (输出电流向量, 总能耗).
        """
        vec = np.asarray(vec, dtype=np.float64)
        if len(vec) != self.n:
            raise ValueError(f"Input vector length {len(vec)} != array size {self.n}")

        currents = np.zeros(self.n)
        total_energy = 0.0

        for i in range(self.n):
            for j in range(self.n):
                total_energy += self.memristors[i][j].energy(v_read, dt)
                currents[i] += self.memristors[i][j].g * vec[j] * v_read

        return currents, total_energy


def run_energy_comparison() -> dict:
    """运行能耗对比: SRAM+ALU vs CIM阵列.

    设置8×8单位矩阵（对角线导通），与8维向量相乘。

    Returns:
        包含能耗对比结果的字典.
    """
    print("=" * 50)
    print("CIM Energy Comparison: SRAM+ALU vs Memristor")
    print("=" * 50)

    # CIM阵列: 8×8, 对角线设为导通 (近似单位矩阵)
    cim = CrossbarArray(8)
    for i in range(8):
        cim.set_weight(i, i, True)  # 对角线导通

    q = np.ones(8) * 0.5  # 输入向量
    _, energy_cim = cim.matrix_vector_mult(q)

    energy_sram = SRAM_ALU_ENERGY_PJ * 1e-12  # pJ → J

    saving = SRAM_ALU_ENERGY_PJ / (energy_cim * 1e12) if energy_cim > 0 else float('inf')

    print(f"  SRAM+ALU Energy: {SRAM_ALU_ENERGY_PJ:.2f} pJ")
    print(f"  CIM Energy:      {energy_cim * 1e12:.2f} pJ")
    print(f"  Saving:          {saving:.1f}x")
    print("=" * 50)

    return {
        "sram_alu_pj": SRAM_ALU_ENERGY_PJ,
        "cim_pj": energy_cim * 1e12,
        "saving_ratio": saving,
    }


def main() -> None:
    """CLI入口."""
    run_energy_comparison()


def _self_test() -> bool:
    """自测."""
    print("[tproc_cim_simulator] Running self-test...")

    cim = CrossbarArray(8)

    # 验证全关断状态
    vec = np.ones(8) * 0.5
    currents, energy = cim.matrix_vector_mult(vec)
    assert np.allclose(currents, 0, atol=1e-5), "All-off array should produce ~0 current"
    assert energy > 0, "Energy should be positive even for off state"

    # 设置对角线
    for i in range(8):
        cim.set_weight(i, i, True)

    currents, energy = cim.matrix_vector_mult(vec)
    # 对角线g_on=1e-3, v_read=0.1, vec=0.5 → current[i] ≈ 1e-3 × 0.5 × 0.1 = 5e-5
    # (含off-state泄漏: 7 × g_off × vec × v_read ≈ 3.5e-7, 用atol=1e-6容差)
    expected_current = 1e-3 * 0.5 * 0.1
    assert np.allclose(currents, expected_current, atol=1e-6), f"Diagonal current should be ~{expected_current}, got {currents[0]}"

    # CIM能耗应远小于SRAM+ALU
    energy_cim_pj = energy * 1e12
    assert energy_cim_pj < SRAM_ALU_ENERGY_PJ, f"CIM ({energy_cim_pj:.2f}pJ) should be < SRAM+ALU ({SRAM_ALU_ENERGY_PJ:.2f}pJ)"

    print(f"  CIM energy: {energy_cim_pj:.4f} pJ")
    print(f"  SRAM+ALU:   {SRAM_ALU_ENERGY_PJ:.2f} pJ")
    print(f"  Saving:     {SRAM_ALU_ENERGY_PJ / energy_cim_pj:.1f}x")
    print("[tproc_cim_simulator] Self-test PASSED.")
    return True


if __name__ == "__main__":
    main()
