"""
TProcCIMSimulator — T-Processor CIM忆阻器/PCM模拟器
====================================================

模拟基于忆阻器(RRRIM)和相变存储器(PCM)的存内计算(CIM)阵列，
对比SRAM+ALU能耗。

v0.4.0 升级: 添加PCM (Phase Change Memory)模型
  - PCMModel: 电导演化 (SET→高电导, RESET→低电导, 部分SET→中间态)
  - pulse_verify_write(): 脉冲校验写入算法 (~7脉冲收敛)
  - PCMCrossbarArray: 继承CrossbarArray, 使用PCM单元
  - 保留原RRRIM模型向后兼容

CIM优势: 矩阵向量乘法在交叉阵列中一步完成，无需读写SRAM。
对比基准: SRAM+ALU = 335.36 pJ (8x8矩阵向量乘法)

参考: 章锋SLOS论文 (2026-07-04, 第二版) 附录D/E

Author: MuJoCo-Bench-IDO v0.4.0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional

__all__ = [
    "MemristorModel",
    "PCMModel",
    "CrossbarArray",
    "PCMCrossbarArray",
    "SRAM_ALU_ENERGY_PJ",
    "PCM_G_MAX_S",
    "PCM_G_MIN_S",
    "PCM_TARGET_CODE",
    "run_energy_comparison",
    "run_pcm_comparison",
    "main",
]


#: SRAM+ALU基准能耗 (8×8矩阵向量乘法, 28nm工艺)
SRAM_ALU_ENERGY_PJ: float = 335.36

#: PCM电导参数 (杨玉超团队可控相变忆阻器)
PCM_G_MAX_S: float = 100e-6   # 100 µS (完全结晶/SET态)
PCM_G_MIN_S: float = 1e-6     # 1 µS (完全非晶/RESET态)
PCM_CODE_MAX: int = 0xFFFF    # 16位电导码满量程
PCM_TARGET_CODE: int = 0x4000  # 默认目标电导码
PCM_TOLERANCE: int = 0x0200    # 电导容差 (±512码, ~0.78%)
PCM_MAX_PULSES: int = 16       # 最大SET脉冲次数


@dataclass
class MemristorModel:
    """忆阻器模型 (RRRIM — Redox-based Resistive RAM).

    基本二态器件: SET(导通) / RESET(关断).

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


@dataclass
class PCMModel:
    """PCM (Phase Change Memory) 相变存储器模型.

    基于杨玉超团队可控相变忆阻器, 电导通过结晶度连续可调.

    电导态映射:
      - RESET (完全非晶): G_min = 1 µS
      - SET (完全结晶): G_max = 100 µS
      - 部分SET (渐进结晶): G_min ~ G_max 之间连续可调

    16位电导码: 0x0000 (G_min) ~ 0xFFFF (G_max)

    Attributes:
        g: 当前电导 (Siemens).
        g_max: 最大电导 (SET态).
        g_min: 最小电导 (RESET态).
        code: 当前电导码 (16位).
        code_max: 电导码满量程.
    """
    g_max: float = PCM_G_MAX_S   # 100 µS
    g_min: float = PCM_G_MIN_S   # 1 µS
    code_max: int = PCM_CODE_MAX

    def __post_init__(self) -> None:
        self.code: int = 0  # 初始为RESET态 (code=0)
        self.g: float = self.g_min

    def _update_g(self) -> None:
        """根据code更新电导值."""
        frac = self.code / self.code_max
        self.g = self.g_min + frac * (self.g_max - self.g_min)

    def set(self) -> None:
        """完全SET (结晶化 → 高电导)."""
        self.code = self.code_max
        self._update_g()

    def reset(self) -> None:
        """完全RESET (非晶化 → 低电导)."""
        self.code = 0
        self._update_g()

    def partial_set(self, target_code: int) -> None:
        """部分SET (渐进结晶 → 中间电导态).

        Args:
            target_code: 目标电导码 (0 ~ code_max).
        """
        self.code = max(0, min(self.code_max, int(target_code)))
        self._update_g()

    def read_code(self) -> int:
        """读回当前电导码.

        Returns:
            当前电导码 (含±1%噪声).
        """
        noise = int(np.random.normal(0, self.code_max * 0.005))
        return max(0, min(self.code_max, self.code + noise))

    def code_to_conductance(self, code: int) -> float:
        """电导码 → 物理电导.

        Args:
            code: 16位电导码.

        Returns:
            电导值 (Siemens).
        """
        code_clamped = max(0, min(self.code_max, int(code)))
        frac = code_clamped / self.code_max
        return self.g_min + frac * (self.g_max - self.g_min)

    def conductance_to_code(self, g: float) -> int:
        """物理电导 → 电导码.

        Args:
            g: 电导值 (Siemens).

        Returns:
            16位电导码.
        """
        g_clamped = max(self.g_min, min(self.g_max, g))
        frac = (g_clamped - self.g_min) / (self.g_max - self.g_min)
        return int(round(frac * self.code_max))

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
        self.memristors: List[List[MemristorModel]] = [
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


class PCMCrossbarArray(CrossbarArray):
    """PCM交叉阵列 (N×N).

    继承CrossbarArray, 使用PCMModel替代MemristorModel,
    支持连续电导态编程和脉冲校验写入.

    Attributes:
        n: 阵列维度 (N×N).
        cells: N×N PCM单元矩阵.
    """

    def __init__(self, n: int = 8) -> None:
        """初始化N×N PCM交叉阵列.

        Args:
            n: 阵列维度.
        """
        super().__init__(n)
        # 用PCM单元替换忆阻器
        self.cells: List[List[PCMModel]] = [
            [PCMModel() for _ in range(n)]
            for _ in range(n)
        ]

    def set_weight(self, i: int, j: int, value: bool) -> None:
        """设置权重 (二态模式).

        Args:
            i: 行索引.
            j: 列索引.
            value: True=导通(SET), False=关断(RESET).
        """
        if value:
            self.cells[i][j].set()
        else:
            self.cells[i][j].reset()

    def set_weight_code(self, i: int, j: int, code: int) -> None:
        """设置权重 (连续电导态).

        Args:
            i: 行索引.
            j: 列索引.
            code: 16位电导码 (0 ~ 0xFFFF).
        """
        self.cells[i][j].partial_set(code)

    def pulse_verify_write(
        self,
        i: int,
        j: int,
        target_code: int,
        tolerance: int = PCM_TOLERANCE,
        max_pulses: int = PCM_MAX_PULSES,
    ) -> Dict[str, Any]:
        """脉冲校验写入单个PCM单元.

        实现SLOS论文的渐进SET脉冲算法:
          1. 从RESET态(code=0)开始
          2. 施加SET脉冲, 步进增加电导
          3. 读回验证: |actual - target| < tolerance?
          4. 若未收敛, 自适应步长后继续
          5. 若过冲, 施加小RESET修正

        目标0x4000的参考收敛序列(~7脉冲):
          0x2000 → 0x2800 → 0x3500 → 0x3E00 → 0x3F80 → 0x3FF0 → 0x4000

        Args:
            i: 行索引.
            j: 列索引.
            target_code: 目标电导码.
            tolerance: 收敛容差 (码单位).
            max_pulses: 最大脉冲数.

        Returns:
            包含收敛信息的字典:
              - converged: 是否收敛
              - pulses: 脉冲次数
              - final_code: 最终电导码
              - sequence: 每次脉冲后的电导码序列
              - error: 最终误差
        """
        cell = self.cells[i][j]
        cell.reset()  # 从RESET态开始

        step = 0x0800  # 初始步长 = 2048码
        sequence: List[int] = [cell.code]
        pulse_count = 0

        for _ in range(max_pulses):
            error = target_code - cell.code

            # 检查收敛
            if abs(error) <= tolerance:
                break

            # 自适应步长
            if abs(error) < step:
                step = max(abs(error) // 2, 0x0040)

            if error > 0:
                # 需要更高电导 → SET脉冲
                new_code = min(cell.code_max, cell.code + step)
                cell.partial_set(new_code)
            else:
                # 过冲 → 小RESET修正
                new_code = max(0, cell.code - step // 2)
                cell.partial_set(new_code)

            # 模拟PCM随机性 (~0.5%)
            noise = int(np.random.normal(0, cell.code_max * 0.003))
            actual_code = max(0, min(cell.code_max, cell.code + noise))
            cell.partial_set(actual_code)

            sequence.append(cell.code)
            pulse_count += 1

        final_error = abs(target_code - cell.code)
        converged = final_error <= tolerance

        return {
            "converged": converged,
            "pulses": pulse_count,
            "final_code": cell.code,
            "sequence": sequence,
            "error": final_error,
        }

    def matrix_vector_mult(
        self,
        vec: np.ndarray,
        v_read: float = 0.1,
        dt: float = 1e-9,
    ) -> Tuple[np.ndarray, float]:
        """矩阵向量乘法 (PCM存内计算).

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
                total_energy += self.cells[i][j].energy(v_read, dt)
                currents[i] += self.cells[i][j].g * vec[j] * v_read

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


def run_pcm_comparison() -> dict:
    """运行PCM对比: SRAM+ALU vs RRRIM vs PCM.

    Returns:
        包含三种方案能耗对比结果的字典.
    """
    print("=" * 60)
    print("CIM Energy Comparison: SRAM+ALU vs RRRIM vs PCM")
    print("=" * 60)

    q = np.ones(8) * 0.5  # 输入向量

    # RRRIM阵列 (对角线导通)
    rrrim = CrossbarArray(8)
    for i in range(8):
        rrrim.set_weight(i, i, True)
    _, energy_rrrim = rrrim.matrix_vector_mult(q)

    # PCM阵列 (对角线编程到中间电导态 0x8000)
    pcm = PCMCrossbarArray(8)
    for i in range(8):
        pcm.set_weight_code(i, i, 0x8000)
    _, energy_pcm = pcm.matrix_vector_mult(q)

    energy_sram = SRAM_ALU_ENERGY_PJ * 1e-12  # pJ → J

    saving_rrrim = SRAM_ALU_ENERGY_PJ / (energy_rrrim * 1e12) if energy_rrrim > 0 else float('inf')
    saving_pcm = SRAM_ALU_ENERGY_PJ / (energy_pcm * 1e12) if energy_pcm > 0 else float('inf')

    print(f"  SRAM+ALU Energy:  {SRAM_ALU_ENERGY_PJ:.2f} pJ")
    print(f"  RRRIM CIM Energy: {energy_rrrim * 1e12:.4f} pJ  (saving: {saving_rrrim:.1f}x)")
    print(f"  PCM CIM Energy:   {energy_pcm * 1e12:.4f} pJ  (saving: {saving_pcm:.1f}x)")
    print("=" * 60)

    return {
        "sram_alu_pj": SRAM_ALU_ENERGY_PJ,
        "rrrim_pj": energy_rrrim * 1e12,
        "pcm_pj": energy_pcm * 1e12,
        "saving_rrrim": saving_rrrim,
        "saving_pcm": saving_pcm,
    }


def main() -> None:
    """CLI入口."""
    run_energy_comparison()
    print()
    run_pcm_comparison()


def _self_test() -> bool:
    """自测."""
    print("[tproc_cim_simulator] Running self-test...")

    # ── 原RRRIM测试 (向后兼容) ──
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
    expected_current = 1e-3 * 0.5 * 0.1
    assert np.allclose(currents, expected_current, atol=1e-6), \
        f"Diagonal current should be ~{expected_current}, got {currents[0]}"

    # CIM能耗应远小于SRAM+ALU
    energy_cim_pj = energy * 1e12
    assert energy_cim_pj < SRAM_ALU_ENERGY_PJ, \
        f"CIM ({energy_cim_pj:.2f}pJ) should be < SRAM+ALU ({SRAM_ALU_ENERGY_PJ:.2f}pJ)"

    print(f"  RRRIM CIM: {energy_cim_pj:.4f} pJ vs SRAM {SRAM_ALU_ENERGY_PJ:.2f} pJ ✓")

    # ── PCM模型测试 ──
    pcm_cell = PCMModel()

    # RESET态: 低电导
    pcm_cell.reset()
    assert pcm_cell.code == 0, f"RESET code should be 0, got {pcm_cell.code}"
    assert abs(pcm_cell.g - PCM_G_MIN_S) < 1e-12, "RESET conductance should be G_min"

    # SET态: 高电导
    pcm_cell.set()
    assert pcm_cell.code == PCM_CODE_MAX, f"SET code should be 0xFFFF, got {pcm_cell.code}"
    assert abs(pcm_cell.g - PCM_G_MAX_S) < 1e-12, "SET conductance should be G_max"

    # 部分SET: 中间电导
    pcm_cell.partial_set(0x8000)  # 中间态
    assert pcm_cell.code == 0x8000, f"Partial SET code should be 0x8000, got {pcm_cell.code}"
    expected_g = PCM_G_MIN_S + 0x8000 / PCM_CODE_MAX * (PCM_G_MAX_S - PCM_G_MIN_S)
    assert abs(pcm_cell.g - expected_g) < 1e-12, \
        f"Partial SET conductance mismatch: {pcm_cell.g} vs {expected_g}"

    # 电导码↔电导 转换
    for test_code in [0, 0x2000, 0x4000, 0x8000, 0xFFFF]:
        g = pcm_cell.code_to_conductance(test_code)
        code_back = pcm_cell.conductance_to_code(g)
        assert abs(code_back - test_code) <= 1, \
            f"Round-trip failed: {test_code} → {g} → {code_back}"

    print(f"  PCM model: SET/RESET/partial_set/conversion ✓")

    # ── PCM阵列测试 ──
    pcm_array = PCMCrossbarArray(8)

    # 全RESET: 近似零电流 (G_min leakage)
    currents, energy_pcm = pcm_array.matrix_vector_mult(vec)
    assert np.allclose(currents, 0, atol=1e-6), "All-RESET PCM should produce ~0 current"

    # 对角线编程到0x8000
    for i in range(8):
        pcm_array.set_weight_code(i, i, 0x8000)

    currents, energy_pcm = pcm_array.matrix_vector_mult(vec)
    g_mid = PCM_G_MIN_S + 0x8000 / PCM_CODE_MAX * (PCM_G_MAX_S - PCM_G_MIN_S)
    # Expected: diagonal (g_mid) + 7 off-diagonal (g_min) leakage
    expected_current_pcm = (g_mid + 7 * PCM_G_MIN_S) * 0.5 * 0.1
    assert np.allclose(currents, expected_current_pcm, atol=1e-8), \
        f"PCM diagonal current should be ~{expected_current_pcm}, got {currents[0]}"

    # PCM能耗应远小于SRAM+ALU
    energy_pcm_pj = energy_pcm * 1e12
    assert energy_pcm_pj < SRAM_ALU_ENERGY_PJ, \
        f"PCM ({energy_pcm_pj:.2f}pJ) should be < SRAM+ALU ({SRAM_ALU_ENERGY_PJ:.2f}pJ)"

    print(f"  PCM array: {energy_pcm_pj:.4f} pJ vs SRAM {SRAM_ALU_ENERGY_PJ:.2f} pJ ✓")

    # ── 脉冲校验写入测试 ──
    np.random.seed(42)
    pcm_array2 = PCMCrossbarArray(8)
    result = pcm_array2.pulse_verify_write(0, 0, PCM_TARGET_CODE)

    assert result["converged"], \
        f"Pulse-verify should converge to 0x{PCM_TARGET_CODE:04X}, " \
        f"got 0x{result['final_code']:04X} after {result['pulses']} pulses"
    assert result["pulses"] <= 10, \
        f"Should converge in ≤10 pulses, got {result['pulses']}"
    assert result["error"] <= PCM_TOLERANCE, \
        f"Error {result['error']} exceeds tolerance {PCM_TOLERANCE}"

    # 序列应为单调递增 (SET脉冲渐进)
    seq = result["sequence"]
    for k in range(1, len(seq)):
        assert seq[k] >= seq[k-1] - PCM_TOLERANCE, \
            f"Sequence should be roughly monotonic at step {k}: {seq[k-1]} → {seq[k]}"

    print(f"  Pulse-verify: 0x{PCM_TARGET_CODE:04X} → 0x{result['final_code']:04X} "
          f"in {result['pulses']} pulses ✓")

    # ── PCM对比测试 ──
    comp = run_pcm_comparison()
    assert comp["saving_rrrim"] > 1.0, "RRRIM should save energy vs SRAM"
    assert comp["saving_pcm"] > 1.0, "PCM should save energy vs SRAM"

    print("[tproc_cim_simulator] Self-test PASSED.")
    return True


if __name__ == "__main__":
    main()
