"""
test_octonion — 八元数代数模块测试
===================================

测试 core/octonion_ops.py 的八元数运算、EML节点和Φ流贯演化算子。

Author: MuJoCo-Bench-IDO Welding Module v0.3.0
"""

import numpy as np
import pytest
import sys
import os

# 添加项目根路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.octonion_ops import (
    OctonionOps,
    OctonionEMLNode,
    FANO_PLANE_ORDER,
    OCTONION_MUL_TABLE,
)


class TestOctonionMul:
    """八元数乘法测试."""

    def test_unit_element(self):
        """e0 (单位元) 乘以任意八元数 = 原八元数."""
        e0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        a = np.array([3, -2, 5, 1, -4, 7, -6, 2], dtype=np.float64)
        result = OctonionOps.mul(e0, a)
        np.testing.assert_allclose(result, a, atol=1e-10)

    def test_zero_element(self):
        """零八元数乘法 = 零."""
        zero = np.zeros(8, dtype=np.float64)
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        result = OctonionOps.mul(zero, a)
        np.testing.assert_allclose(result, zero, atol=1e-10)

    def test_e1_squared(self):
        """e1 × e1 = -e0 (虚单位平方 = -1)."""
        e1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.mul(e1, e1)
        expected = np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_e2_squared(self):
        """e2 × e2 = -e0."""
        e2 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.mul(e2, e2)
        expected = np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_non_commutative(self):
        """八元数乘法非交换: a·b ≠ b·a (对于非单位元)."""
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float64)
        ab = OctonionOps.mul(a, b)
        ba = OctonionOps.mul(b, a)
        assert not np.allclose(ab, ba), "Octonion multiplication should be non-commutative"

    def test_non_associative(self):
        """八元数乘法非结合: (a·b)·c ≠ a·(b·c).

        使用跨越 Cayley-Dickson 层的基元素 e1,e2,e4:
          (e1·e2)·e4 = e3·e4 = e7
          e1·(e2·e4) = e1·e6 = -e7
        """
        e1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        e2 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        e4 = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        # (e1·e2)·e4
        e1e2 = OctonionOps.mul(e1, e2)
        left = OctonionOps.mul(e1e2, e4)
        # e1·(e2·e4)
        e2e4 = OctonionOps.mul(e2, e4)
        right = OctonionOps.mul(e1, e2e4)
        assert not np.allclose(left, right), "Octonion multiplication should be non-associative"

    def test_e1_e2_equals_e3(self):
        """e1 × e2 = e3 (基本乘法表验证)."""
        e1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        e2 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.mul(e1, e2)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_e2_e1_equals_neg_e3(self):
        """e2 × e1 = -e3 (非交换性验证)."""
        e1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        e2 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.mul(e2, e1)
        expected = np.array([0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_scalar_multiplication(self):
        """实数部分乘法: (a0·e0) × (b0·e0) = (a0*b0)·e0."""
        a = np.array([5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        b = np.array([3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.mul(a, b)
        expected = np.array([15, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_invalid_dimension(self):
        """非8维输入应抛出ValueError."""
        with pytest.raises(ValueError):
            OctonionOps.mul(np.zeros(7), np.zeros(8))
        with pytest.raises(ValueError):
            OctonionOps.mul(np.zeros(8), np.zeros(10))


class TestOctonionConjugate:
    """八元数共轭测试."""

    def test_conjugate_definition(self):
        """共轭: a* = [a0, -a1, -a2, ..., -a7]."""
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        result = OctonionOps.conjugate(a)
        expected = np.array([1, -2, -3, -4, -5, -6, -7, -8], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_real_conjugate(self):
        """实数的共轭 = 自身."""
        a = np.array([42, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.conjugate(a)
        np.testing.assert_allclose(result, a, atol=1e-10)

    def test_double_conjugate(self):
        """a** = a (双重共轭 = 原始)."""
        a = np.array([3, -1, 2, 4, -5, 7, -3, 6], dtype=np.float64)
        result = OctonionOps.conjugate(OctonionOps.conjugate(a))
        np.testing.assert_allclose(result, a, atol=1e-10)


class TestOctonionNorm:
    """八元数范数测试."""

    def test_norm_definition(self):
        """||a|| = sqrt(sum(ai²))."""
        a = np.array([3, 4, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.norm(a)
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_zero_norm(self):
        """零八元数范数 = 0."""
        result = OctonionOps.norm(np.zeros(8))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_norm_multiplicativity(self):
        """范数乘性: ||a·b|| = ||a|| × ||b||."""
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float64)
        norm_a = OctonionOps.norm(a)
        norm_b = OctonionOps.norm(b)
        ab = OctonionOps.mul(a, b)
        norm_ab = OctonionOps.norm(ab)
        assert norm_ab == pytest.approx(norm_a * norm_b, rel=1e-6)


class TestOctonionNormalize:
    """八元数归一化测试."""

    def test_normalize_to_unit(self):
        """归一化后范数 = 1."""
        a = np.array([3, 4, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        result = OctonionOps.normalize(a)
        norm_result = OctonionOps.norm(result)
        assert norm_result == pytest.approx(1.0, abs=1e-10)

    def test_normalize_preserves_direction(self):
        """归一化保持方向 (比例不变)."""
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        result = OctonionOps.normalize(a)
        ratio = a / result
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)


class TestPhiOperator:
    """Φ流贯演化算子测试."""

    def test_phi_identity(self):
        """Φ(单位元e0, ω) ≈ ω (单位元流贯不变)."""
        e0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        omega = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        result = OctonionOps.phi(e0, omega)
        np.testing.assert_allclose(result, omega, atol=1e-6)

    def test_phi_returns_8d(self):
        """Φ返回8维向量."""
        q = np.random.randn(8)
        omega = np.random.randn(8)
        result = OctonionOps.phi(q, omega)
        assert len(result) == 8

    def test_phi_left_associative(self):
        """Φ(q,ω) = (q·ω)·q (左结合, 验证等于显式计算)."""
        q = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
        omega = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=np.float64)
        # 显式左结合
        qw = OctonionOps.mul(q, omega)
        expected = OctonionOps.mul(qw, q)
        result = OctonionOps.phi(q, omega)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestEtaResidual:
    """η残差测试."""

    def test_eta_zero_on_match(self):
        """完美匹配时 η = 0 (q=单位元, ω=单位元)."""
        e0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        eta = OctonionOps.eta_residual(e0, e0)
        assert eta == pytest.approx(0.0, abs=1e-10)

    def test_eta_positive_on_mismatch(self):
        """不匹配时 η > 0."""
        q = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        omega = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        eta = OctonionOps.eta_residual(q, omega)
        assert eta > 0.0

    def test_eta_is_squared_norm(self):
        """η = ||Φ(q,ω) - ω||² (验证公式)."""
        q = np.random.randn(8)
        omega = np.random.randn(8)
        phi_result = OctonionOps.phi(q, omega)
        diff = phi_result - omega
        expected_eta = float(np.dot(diff, diff))
        eta = OctonionOps.eta_residual(q, omega)
        assert eta == pytest.approx(expected_eta, rel=1e-6)


class TestOctonionEMLNode:
    """八元数EML节点测试."""

    def test_default_creation(self):
        """默认创建: 8个零分量."""
        node = OctonionEMLNode()
        assert len(node.components) == 8
        assert node.weld_type == "flat"

    def test_from_welding_state(self):
        """从焊接状态创建: 分量在[-1, 1]范围内."""
        node = OctonionEMLNode.from_welding_state(
            current=200.0, voltage=24.0, speed=6.0, stickout=15.0,
            heat_input=0.8, penetration=2.0, porosity=0.02, distortion=0.5,
        )
        assert len(node.components) == 8
        assert np.all(node.components >= -1.0 - 1e-6)
        assert np.all(node.components <= 1.0 + 1e-6)

    def test_from_welding_state_clipping(self):
        """超出范围的值被裁剪到[-1, 1]."""
        node = OctonionEMLNode.from_welding_state(
            current=500.0,  # 超出[50,350]
            voltage=50.0,   # 超出[14,32]
        )
        assert node.components[0] <= 1.0  # current被裁剪
        assert node.components[1] <= 1.0  # voltage被裁剪

    def test_optimal_params_near_center(self):
        """最优参数 (200A, 24V, 6mm/s, 15mm) 归一化后接近0."""
        node = OctonionEMLNode.from_welding_state(
            current=200.0, voltage=24.0, speed=6.0, stickout=15.0,
            heat_input=0.8, penetration=2.0, porosity=0.02, distortion=0.5,
        )
        # 200A在[50,350]中点=200 → 归一化为0
        assert node.components[0] == pytest.approx(0.0, abs=1e-6)
        # 24V在[14,32]中点=23 → 归一化接近0
        assert node.components[1] == pytest.approx(0.0, abs=0.12)

    def test_to_bytes(self):
        """序列化为字节串."""
        node = OctonionEMLNode(components=np.ones(8))
        b = node.to_bytes()
        assert len(b) == 64  # 8 × 8 bytes (float64)

    def test_to_dict(self):
        """转换为字典."""
        node = OctonionEMLNode.from_welding_state(weld_type="vertical")
        d = node.to_dict()
        assert "components" in d
        assert "weld_type" in d
        assert d["weld_type"] == "vertical"
        assert len(d["components"]) == 8

    def test_invalid_components(self):
        """非8维分量应抛出ValueError."""
        with pytest.raises(ValueError):
            OctonionEMLNode(components=np.zeros(7))


class TestFanoPlane:
    """Fano平面常数测试."""

    def test_fano_plane_order(self):
        """Fano平面自同构群阶数 = 168."""
        assert FANO_PLANE_ORDER == 168

    def test_mul_table_shape(self):
        """乘法表形状 = 8×8."""
        assert OCTONION_MUL_TABLE.shape == (8, 8)
