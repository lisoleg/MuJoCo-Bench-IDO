"""
WeldingProcessProxy 单元测试
=============================

测试焊接工艺代理模型的预测功能 (适配架构师版本API):
  - predict返回WeldingQuality
  - predict_quality返回dict (兼容env接口)
  - arc_length = voltage - 14
  - heat_input公式正确
  - 高干伸长→高气孔风险
  - 大热输入→大角变形
  - 质量评分在合理范围

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.welding_process_proxy import WeldingProcessProxy, WeldingQuality, WeldingParams


@pytest.fixture
def proxy():
    """创建 WeldingProcessProxy fixture."""
    return WeldingProcessProxy(weld_type="flat")


class TestPredictBasic:
    """测试 predict 基本功能."""

    def test_predict_returns_welding_quality(self, proxy):
        """predict返回WeldingQuality dataclass."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert isinstance(result, WeldingQuality), f"Expected WeldingQuality, got {type(result)}"

    def test_predict_all_fields_present(self, proxy):
        """predict返回所有字段."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert hasattr(result, "eta_residual")
        assert hasattr(result, "porosity_risk")
        assert hasattr(result, "angular_distortion")
        assert hasattr(result, "penetration_depth")
        assert hasattr(result, "arc_length")
        assert hasattr(result, "heat_input")

    def test_predict_all_values_finite(self, proxy):
        """predict返回的所有值为有限数."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert np.isfinite(result.eta_residual)
        assert np.isfinite(result.porosity_risk)
        assert np.isfinite(result.angular_distortion)
        assert np.isfinite(result.penetration_depth)
        assert np.isfinite(result.arc_length)
        assert np.isfinite(result.heat_input)

    def test_predict_porosity_range(self, proxy):
        """气孔风险在0-1之间."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert 0.0 <= result.porosity_risk <= 1.0

    def test_to_dict(self, proxy):
        """to_dict返回正确字典."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        d = result.to_dict()
        assert "eta" in d
        assert "porosity" in d
        assert "distortion" in d


class TestArcLengthCalc:
    """测试电弧长度计算."""

    def test_arc_length_calc(self, proxy):
        """arc_length = voltage - 14."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert abs(result.arc_length - 10.0) < 0.01, \
            f"arc_length should be 10.0 (24-14), got {result.arc_length}"

    def test_arc_length_zero_voltage(self, proxy):
        """电压<14时arc_length=0."""
        result = proxy.predict(current=200, voltage=10, travel_speed=6)
        assert result.arc_length >= 0.0, "arc_length should be >= 0"

    def test_compute_arc_length(self, proxy):
        """直接测试compute_arc_length方法."""
        assert proxy.compute_arc_length(24.0) == 10.0
        assert proxy.compute_arc_length(14.0) == 0.0
        assert proxy.compute_arc_length(10.0) == 0.0


class TestHeatInput:
    """测试热输入计算."""

    def test_heat_input_formula(self, proxy):
        """heat_input = (current * voltage) / (travel_speed * 1000)."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        expected = (200 * 24) / (6 * 1000)  # 0.8 kJ/mm
        assert abs(result.heat_input - expected) < 0.01, \
            f"heat_input should be {expected}, got {result.heat_input}"

    def test_heat_input_high_current(self, proxy):
        """高电流→高热输入."""
        result_low = proxy.predict(current=100, voltage=20, travel_speed=6)
        result_high = proxy.predict(current=300, voltage=28, travel_speed=6)
        assert result_high.heat_input > result_low.heat_input

    def test_compute_heat_input(self, proxy):
        """直接测试compute_heat_input方法."""
        hi = proxy.compute_heat_input(200, 24, 6)
        assert abs(hi - 0.8) < 0.01


class TestPorosity:
    """测试气孔风险."""

    def test_porosity_high_stickout(self, proxy):
        """干伸长25mm→高气孔风险."""
        result_normal = proxy.predict(current=200, voltage=24, travel_speed=6, stickout=15)
        result_high = proxy.predict(current=200, voltage=24, travel_speed=6, stickout=25)
        assert result_high.porosity_risk >= result_normal.porosity_risk, \
            "Higher stickout should increase porosity risk"

    def test_porosity_range(self, proxy):
        """气孔风险在0-1之间."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6, stickout=15)
        assert 0.0 <= result.porosity_risk <= 1.0


class TestDistortion:
    """测试角变形."""

    def test_distortion_high_heat(self, proxy):
        """大热输入→大角变形."""
        result_low = proxy.predict(current=100, voltage=18, travel_speed=10)
        result_high = proxy.predict(current=280, voltage=28, travel_speed=3)
        assert result_high.angular_distortion > result_low.angular_distortion, \
            "Higher heat input should increase distortion"

    def test_distortion_nonnegative(self, proxy):
        """角变形非负."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert result.angular_distortion >= 0.0


class TestPredictQuality:
    """测试兼容接口 predict_quality."""

    def test_predict_quality_returns_dict(self, proxy):
        """predict_quality返回dict."""
        result = proxy.predict_quality(current=200, voltage=24, speed=6, stickout=15)
        assert isinstance(result, dict)
        assert "eta" in result
        assert "porosity" in result
        assert "distortion" in result

    def test_predict_quality_values_finite(self, proxy):
        """predict_quality值有限."""
        result = proxy.predict_quality(current=200, voltage=24, speed=6, stickout=15)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} should be finite"


class TestCurrentVariance:
    """测试电流方差计算."""

    def test_update_and_compute(self, proxy):
        """更新电流历史并计算方差."""
        proxy.update_current_history(200.0)
        proxy.update_current_history(200.0)
        var = proxy.compute_current_variance()
        assert var == 0.0, "Constant current should have 0 variance"

    def test_variance_with_variation(self, proxy):
        """有波动的电流方差>0."""
        for val in [190, 200, 210, 195, 205]:
            proxy.update_current_history(float(val))
        var = proxy.compute_current_variance()
        assert var > 0.0


class TestPenetration:
    """测试熔深计算."""

    def test_penetration_positive(self, proxy):
        """熔深为正数."""
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert result.penetration_depth > 0.0

    def test_penetration_high_current(self, proxy):
        """高电流→深熔深."""
        result_low = proxy.predict(current=100, voltage=20, travel_speed=6)
        result_high = proxy.predict(current=300, voltage=28, travel_speed=6)
        assert result_high.penetration_depth > result_low.penetration_depth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
