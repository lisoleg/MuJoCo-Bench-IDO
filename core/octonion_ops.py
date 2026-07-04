"""
OctonionOps — 八元数非结合代数运算模块
========================================

实现章锋2026-07-04论文《面向硅基生命操作系统的焊接机器人多模态数据采集与因果蒸馏框架》
中的八元数代数核心，用于 EML 因果蒸馏的代数基础。

八元数（Octonion）是Cayley-Dickson构造的第三层超复数，具有8个分量:
    O = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇

关键性质:
  - 非交换: a·b ≠ b·a
  - 非结合: (a·b)·c ≠ a·(b·c)
  - 范数乘性: |a·b| = |a|·|b|
  - Fano平面对称群阶数: 168

Φ流贯演化算子: Φ(q, ω) = (q·ω)·q  (左结合约定)
η残差: ||Φ(q,ω) - ω||²

Author: MuJoCo-Bench-IDO Welding Module v0.3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np

__all__ = [
    "OctonionOps",
    "OctonionEMLNode",
    "FANO_PLANE_ORDER",
    "OCTONION_MUL_TABLE",
    "_self_test",
]

# ── κ-Phase: Octonion Algebra Foundation ──

#: Fano平面自同构群 (PSL(2,7)) 的阶数
FANO_PLANE_ORDER: int = 168

#: Cayley-Dickson 8×8 乘法表 (左结合约定)
#: 行索引 = 被乘元素 e_i, 列索引 = 乘元素 e_j
#: 值 = ±k 表示 e_i × e_j = ±e_k
#: 0 = 实部 e_0 (单位元)
OCTONION_MUL_TABLE: np.ndarray = np.array([
    #  e0   e1   e2   e3   e4   e5   e6   e7
    [  0,   1,   2,   3,   4,   5,   6,   7],  # e0 × e_j
    [  1,  -0,   3,  -2,   5,  -4,  -7,   6],  # e1 × e_j
    [  2,  -3,  -0,   1,   6,   7,  -4,  -5],  # e2 × e_j
    [  3,   2,  -1,  -0,   7,  -6,   5,  -4],  # e3 × e_j
    [  4,  -5,  -6,  -7,  -0,   1,   2,   3],  # e4 × e_j
    [  5,   4,  -7,   6,  -1,  -0,  -3,   2],  # e5 × e_j
    [  6,   7,   4,  -5,  -2,   3,  -0,  -1],  # e6 × e_j
    [  7,  -6,   5,   4,  -3,  -2,   1,  -0],  # e7 × e_j
], dtype=np.int8)


@dataclass
class OctonionEMLNode:
    """八元数EML节点 — 将焊接状态编码为八元数向量.

    将焊接工艺参数映射到八元数的8个分量:
        e0: 焊接电流归一化
        e1: 焊接电压归一化
        e2: 焊接速度归一化
        e3: 干伸长归一化
        e4: 热输入归一化
        e5: 熔深归一化
        e6: 气孔风险归一化
        e7: 角变形归一化

    Attributes:
        components: 8维浮点数组, 八元数的8个实系数.
        weld_type: 焊接姿态类型标签.
        snap_id: κ-Snap审计ID (可选).
    """
    components: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float64)
    )
    weld_type: str = "flat"
    snap_id: str = ""

    def __post_init__(self) -> None:
        """确保 components 是长度为8的 numpy 数组."""
        if len(self.components) != 8:
            raise ValueError(
                f"OctonionEMLNode requires 8 components, got {len(self.components)}"
            )

    def to_bytes(self) -> bytes:
        """将八元数节点序列化为字节串 (用于κ-Snap哈希).

        Returns:
            8×8=64 字节的二进制表示 (每个分量为 float64).
        """
        return self.components.astype(np.float64).tobytes()

    @classmethod
    def from_welding_state(
        cls,
        current: float = 200.0,
        voltage: float = 24.0,
        speed: float = 6.0,
        stickout: float = 15.0,
        heat_input: float = 0.8,
        penetration: float = 2.0,
        porosity: float = 0.02,
        distortion: float = 0.5,
        weld_type: str = "flat",
        snap_id: str = "",
    ) -> "OctonionEMLNode":
        """从焊接状态参数构造八元数EML节点.

        所有参数被归一化到 [−1, 1] 区间:
            current:    [50, 350]  → 归一化
            voltage:    [14, 32]   → 归一化
            speed:      [2, 15]    → 归一化
            stickout:   [8, 25]    → 归一化
            heat_input: [0, 3]     → 归一化
            penetration:[0, 5]     → 归一化
            porosity:   [0, 1]     → 归一化
            distortion: [0, 5]     → 归一化

        Args:
            current: 焊接电流 (A).
            voltage: 焊接电压 (V).
            speed: 焊接速度 (mm/s).
            stickout: 干伸长 (mm).
            heat_input: 热输入 (kJ/mm).
            penetration: 熔深 (mm).
            porosity: 气孔风险 (0-1).
            distortion: 角变形 (degrees).
            weld_type: 焊接姿态类型.
            snap_id: κ-Snap审计ID.

        Returns:
            OctonionEMLNode 实例, 分量归一化到 [−1, 1].
        """
        # 归一化函数: (val - mid) / (range / 2) → [−1, 1]
        def norm(val: float, lo: float, hi: float) -> float:
            """线性归一化到 [−1, 1]."""
            mid: float = (lo + hi) / 2.0
            half_range: float = max((hi - lo) / 2.0, 1e-9)
            return float(np.clip((val - mid) / half_range, -1.0, 1.0))

        comp: np.ndarray = np.array([
            norm(current, 50.0, 350.0),     # e0
            norm(voltage, 14.0, 32.0),      # e1
            norm(speed, 2.0, 15.0),         # e2
            norm(stickout, 8.0, 25.0),      # e3
            norm(heat_input, 0.0, 3.0),     # e4
            norm(penetration, 0.0, 5.0),    # e5
            norm(porosity, 0.0, 1.0),       # e6
            norm(distortion, 0.0, 5.0),     # e7
        ], dtype=np.float64)

        return cls(components=comp, weld_type=weld_type, snap_id=snap_id)

    def to_dict(self) -> dict:
        """转换为字典表示.

        Returns:
            包含 components, weld_type, snap_id 的字典.
        """
        return {
            "components": self.components.tolist(),
            "weld_type": self.weld_type,
            "snap_id": self.snap_id,
        }


class OctonionOps:
    """八元数代数运算静态工具类.

    提供八元数的基本运算:
        - mul: 八元数乘法 (Cayley-Dickson, 非交换非结合)
        - conjugate: 共轭
        - norm: 范数 (满足乘性 |a·b|=|a|·|b|)
        - normalize: 归一化到单位八元数
        - phi: Φ流贯演化算子 Φ(q,ω) = (q·ω)·q
        - eta_residual: η残差 ||Φ(q,ω) - ω||²

    所有方法均为静态方法, 接受 8 维 numpy 数组输入.
    """

    MUL_TABLE: np.ndarray = OCTONION_MUL_TABLE

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """八元数乘法 (Cayley-Dickson 构造, 显式公式).

        使用完整的8分量乘法公式进行运算.
        注意: 八元数乘法非交换 (a·b ≠ b·a) 且非结合 ((a·b)·c ≠ a·(b·c)).

        乘法公式 (Cayley-Dickson):
            res[0] = a0b0 - a1b1 - a2b2 - a3b3 - a4b4 - a5b5 - a6b6 - a7b7
            res[1] = a0b1 + a1b0 + a2b3 - a3b2 + a4b5 - a5b4 - a6b7 + a7b6
            res[2] = a0b2 - a1b3 + a2b0 + a3b1 + a4b6 + a5b7 - a6b4 - a7b5
            res[3] = a0b3 + a1b2 - a2b1 + a3b0 + a4b7 - a5b6 + a6b5 - a7b4
            res[4] = a0b4 - a1b5 - a2b6 - a3b7 + a4b0 + a5b1 + a6b2 + a7b3
            res[5] = a0b5 + a1b4 - a2b7 + a3b6 - a4b1 + a5b0 - a6b3 + a7b2
            res[6] = a0b6 + a1b7 + a2b4 - a3b5 - a4b2 + a5b3 + a6b0 - a7b1
            res[7] = a0b7 - a1b6 + a2b5 + a3b4 - a4b3 - a5b2 + a6b1 + a7b0

        Args:
            a: 左操作数, 长度为8的 numpy 数组.
            b: 右操作数, 长度为8的 numpy 数组.

        Returns:
            乘积 a·b, 长度为8的 numpy 数组.
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        if len(a) != 8 or len(b) != 8:
            raise ValueError(
                f"Octonion operands must have 8 components, "
                f"got {len(a)} and {len(b)}"
            )

        a0, a1, a2, a3, a4, a5, a6, a7 = a
        b0, b1, b2, b3, b4, b5, b6, b7 = b

        result: np.ndarray = np.array([
            a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7,
            a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6,
            a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5,
            a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4,
            a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3,
            a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2,
            a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1,
            a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0,
        ], dtype=np.float64)

        return result

    @staticmethod
    def conjugate(a: np.ndarray) -> np.ndarray:
        """八元数共轭.

        共轭定义: a* = (a₀, -a₁, -a₂, -a₃, -a₄, -a₅, -a₆, -a₇)
        即实部不变, 所有虚部分取反.

        性质: a · a* = a* · a = |a|² (实数)

        Args:
            a: 八元数, 长度为8的 numpy 数组.

        Returns:
            共轭八元数 a*.
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        if len(a) != 8:
            raise ValueError(f"Octonion must have 8 components, got {len(a)}")
        result: np.ndarray = a.copy()
        result[1:] = -result[1:]
        return result

    @staticmethod
    def norm(a: np.ndarray) -> float:
        """八元数范数 (欧几里得范数).

        |a| = sqrt(a₀² + a₁² + ... + a₇²)

        满足乘性: |a·b| = |a|·|b| (八元数的Normed Division Algebra性质)

        Args:
            a: 八元数, 长度为8的 numpy 数组.

        Returns:
            范数标量值.
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        if len(a) != 8:
            raise ValueError(f"Octonion must have 8 components, got {len(a)}")
        return float(np.sqrt(np.sum(a ** 2)))

    @staticmethod
    def normalize(a: np.ndarray) -> np.ndarray:
        """将八元数归一化为单位八元数.

        â = a / |a|

        Args:
            a: 八元数, 长度为8的 numpy 数组.

        Returns:
            单位八元数 (|â| = 1). 如果 |a| < 1e-12, 返回零向量.
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        if len(a) != 8:
            raise ValueError(f"Octonion must have 8 components, got {len(a)}")
        n: float = OctonionOps.norm(a)
        if n < 1e-12:
            return np.zeros(8, dtype=np.float64)
        return a / n

    @staticmethod
    def phi(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Φ流贯演化算子.

        Φ(q, ω) = (q · ω) · q

        左结合约定: 先计算 q·ω, 再将结果乘以 q.
        这是章锋论文中定义的"流贯"演化算子, 用于描述焊接工艺参数
        在八元代数空间中的因果演化路径.

        Args:
            q: 演化算子八元数 (建议归一化).
            omega: 初始状态八元数.

        Returns:
            演化后的八元数状态.
        """
        q = np.asarray(q, dtype=np.float64).flatten()
        omega = np.asarray(omega, dtype=np.float64).flatten()
        if len(q) != 8 or len(omega) != 8:
            raise ValueError(
                f"Octonion operands must have 8 components, "
                f"got {len(q)} and {len(omega)}"
            )
        # 左结合: (q · ω) · q
        q_omega: np.ndarray = OctonionOps.mul(q, omega)
        result: np.ndarray = OctonionOps.mul(q_omega, q)
        return result

    @staticmethod
    def eta_residual(q: np.ndarray, omega: np.ndarray) -> float:
        """η残差 — 量化Φ演化后的状态偏离.

        η = ||Φ(q, ω) - ω||²

        η 越小表示 q 对 ω 的演化越"保守" (接近恒等变换),
        η 越大表示演化偏离越大.

        在EML因果蒸馏中, η用于:
          1. 评估蒸馏后的工艺参数与原始状态的因果距离
          2. 作为Pareto前沿的优化目标之一
          3. 驱动κ-Snap审计触发阈值

        Args:
            q: 演化算子八元数.
            omega: 初始状态八元数.

        Returns:
            η残差标量值 (≥0).
        """
        phi_result: np.ndarray = OctonionOps.phi(q, omega)
        diff: np.ndarray = phi_result - omega
        eta: float = float(np.sum(diff ** 2))
        return max(0.0, eta)

    @staticmethod
    def is_alternative(a: np.ndarray, b: np.ndarray) -> bool:
        """检验交错律 (Alternative Law) 是否成立.

        八元数满足交错律 (比结合律弱):
          - 左交错律: (a·a)·b = a·(a·b)
          - 右交错律: (a·b)·b = a·(b·b)

        本方法检验左交错律.

        Args:
            a: 第一个八元数.
            b: 第二个八元数.

        Returns:
            True 如果左交错律成立 (在数值精度内).
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()

        lhs: np.ndarray = OctonionOps.mul(OctonionOps.mul(a, a), b)
        rhs: np.ndarray = OctonionOps.mul(a, OctonionOps.mul(a, b))
        diff: float = float(np.max(np.abs(lhs - rhs)))
        return diff < 1e-10

    @staticmethod
    def check_non_associative(
        a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> bool:
        """检验非结合性.

        验证 (a·b)·c ≠ a·(b·c) 对某些三元组成立.

        Args:
            a, b, c: 三个八元数.

        Returns:
            True 如果该三元组展现非结合性 (即 (a·b)·c ≠ a·(b·c)).
        """
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()
        c = np.asarray(c, dtype=np.float64).flatten()

        lhs: np.ndarray = OctonionOps.mul(OctonionOps.mul(a, b), c)
        rhs: np.ndarray = OctonionOps.mul(a, OctonionOps.mul(b, c))
        diff: float = float(np.max(np.abs(lhs - rhs)))
        return diff > 1e-10


def _self_test() -> bool:
    """八元数运算模块自测.

    验证:
      1. 乘法表正确性 (e1·e2 = e3, e2·e1 = -e3)
      2. 非交换性 (a·b ≠ b·a)
      3. 非结合性 ((a·b)·c ≠ a·(b·c))
      4. 共轭性质 (a · a* = |a|²)
      5. 范数乘性 (|a·b| = |a|·|b|)
      6. 交错律 (alternative law)
      7. Φ算子和η残差

    Returns:
        True 如果所有测试通过, False 否则.
    """
    rng = np.random.default_rng(42)

    # ── 测试1: 乘法表基本性质 ──
    e1: np.ndarray = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    e2: np.ndarray = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    e3: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)

    # e1 · e2 = e3
    prod_12: np.ndarray = OctonionOps.mul(e1, e2)
    assert np.allclose(prod_12, e3), f"e1·e2 should be e3, got {prod_12}"

    # e2 · e1 = -e3 (非交换)
    prod_21: np.ndarray = OctonionOps.mul(e2, e1)
    assert np.allclose(prod_21, -e3), f"e2·e1 should be -e3, got {prod_21}"

    # ── 测试2: 非交换性 ──
    a: np.ndarray = rng.standard_normal(8)
    b: np.ndarray = rng.standard_normal(8)
    ab: np.ndarray = OctonionOps.mul(a, b)
    ba: np.ndarray = OctonionOps.mul(b, a)
    assert not np.allclose(ab, ba), "Octonion multiplication should be non-commutative"

    # ── 测试3: 非结合性 ──
    c: np.ndarray = rng.standard_normal(8)
    is_nonassoc: bool = OctonionOps.check_non_associative(a, b, c)
    assert is_nonassoc, "Octonion should exhibit non-associativity for generic elements"

    # ── 测试4: 共轭性质 a · a* = |a|² ──
    a_conj: np.ndarray = OctonionOps.conjugate(a)
    a_ac: np.ndarray = OctonionOps.mul(a, a_conj)
    norm_sq: float = OctonionOps.norm(a) ** 2
    # a·a* 的虚部应该为零, 实部 = |a|²
    assert abs(a_ac[0] - norm_sq) < 1e-8, f"a·a* real part should be |a|²={norm_sq}, got {a_ac[0]}"
    assert np.max(np.abs(a_ac[1:])) < 1e-8, f"a·a* imaginary parts should be zero, got {a_ac[1:]}"

    # ── 测试5: 范数乘性 |a·b| = |a|·|b| ──
    norm_a: float = OctonionOps.norm(a)
    norm_b: float = OctonionOps.norm(b)
    norm_ab: float = OctonionOps.norm(ab)
    assert abs(norm_ab - norm_a * norm_b) < 1e-8 * max(norm_a * norm_b, 1.0), \
        f"|a·b|={norm_ab} should equal |a|·|b|={norm_a * norm_b}"

    # ── 测试6: 交错律 ──
    is_alt: bool = OctonionOps.is_alternative(a, b)
    assert is_alt, "Octonion should satisfy alternative law"

    # ── 测试7: Φ算子和η残差 ──
    q: np.ndarray = OctonionOps.normalize(rng.standard_normal(8))
    omega: np.ndarray = rng.standard_normal(8)
    phi_result: np.ndarray = OctonionOps.phi(q, omega)
    assert len(phi_result) == 8, "Φ should return 8-component array"

    eta: float = OctonionOps.eta_residual(q, omega)
    assert eta >= 0.0, f"η residual should be non-negative, got {eta}"

    # 当 q = 实单位 (1,0,...,0) 时, Φ(q,ω) = (1·ω)·1 = ω, η = 0
    q_identity: np.ndarray = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    eta_identity: float = OctonionOps.eta_residual(q_identity, omega)
    assert eta_identity < 1e-20, f"η with identity q should be ~0, got {eta_identity}"

    # ── 测试8: OctonionEMLNode ──
    node: OctonionEMLNode = OctonionEMLNode.from_welding_state(
        current=200.0, voltage=24.0, speed=6.0, stickout=15.0,
        heat_input=0.8, penetration=2.0, porosity=0.02, distortion=0.5,
        weld_type="flat",
    )
    assert len(node.components) == 8
    assert node.weld_type == "flat"
    # 最优参数应归一化到接近0
    assert abs(node.components[0]) < 0.1, f"Optimal current should normalize near 0, got {node.components[0]}"

    # to_bytes 测试
    raw: bytes = node.to_bytes()
    assert len(raw) == 64, f"to_bytes should produce 64 bytes, got {len(raw)}"

    print("[octonion_ops] All 8 self-tests passed.")
    return True


if __name__ == "__main__":
    _self_test()
