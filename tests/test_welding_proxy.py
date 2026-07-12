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
            # Skip non-numeric values (e.g., microstructure dict)
            if isinstance(v, dict):
                continue
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

    # ═══════════════════════════════════════════════════════════════════
    # v0.20.1: 非标场景 + DIKWP规则库 + η-PID自适应 测试
    # ═══════════════════════════════════════════════════════════════════

    def test_classify_non_standard_clean(self, proxy):
        """标准场景→空列表."""
        result = proxy.classify_non_standard(misalignment=0.0, gap=0.0, has_cad=True,
                                              surface_condition="clean", confined_space=False,
                                              batch_size=100)
        assert result == []

    def test_classify_non_standard_geometric(self, proxy):
        """错边>2mm→geometric非标."""
        result = proxy.classify_non_standard(misalignment=4.0, gap=1.0)
        assert "geometric" in result

    def test_classify_non_standard_semantic(self, proxy):
        """无CAD→semantic非标."""
        result = proxy.classify_non_standard(has_cad=False)
        assert "semantic" in result

    def test_classify_non_standard_environmental(self, proxy):
        """锈蚀表面→environmental非标."""
        result = proxy.classify_non_standard(surface_condition="rusty")
        assert "environmental" in result

    def test_classify_non_standard_production(self, proxy):
        """批次<10→production非标."""
        result = proxy.classify_non_standard(batch_size=5)
        assert "production" in result

    def test_classify_non_standard_multiple(self, proxy):
        """多维度非标→多类别."""
        result = proxy.classify_non_standard(
            misalignment=5.0, gap=5.0, has_cad=False,
            surface_condition="oily", confined_space=True, batch_size=3,
        )
        assert len(result) == 4  # 全部四种

    def test_dikwp_rules_no_match(self, proxy):
        """无偏差→无规则匹配."""
        result = proxy.apply_dikwp_rules(misalignment=0.0, gap=0.0)
        assert len(result["matched_rules"]) == 0
        assert result["adjust_current_A"] == 0.0

    def test_dikwp_rules_steel_misalign(self):
        """钢错边→R003匹配."""
        proxy = WeldingProcessProxy("flat", material="steel")
        result = proxy.apply_dikwp_rules(misalignment=3.0, gap=1.0)
        assert "R003_steel_misalign" in result["matched_rules"]

    def test_dikwp_rules_aluminum_gap(self):
        """铝间隙→R002匹配."""
        proxy = WeldingProcessProxy("flat", material="aluminum")
        result = proxy.apply_dikwp_rules(misalignment=0.0, gap=4.0)
        assert "R002_aluminum_gap" in result["matched_rules"]

    def test_dikwp_rules_returns_adjustments(self, proxy):
        """规则返回参数调整量."""
        result = proxy.apply_dikwp_rules(misalignment=3.0, gap=1.0)
        assert "adjust_current_A" in result
        assert "adjust_voltage_V" in result
        assert "adjust_weave_mm" in result
        assert "scenario_type" in result

    def test_eta_pid_no_trigger(self, proxy):
        """η低于阈值→不触发."""
        # 最优参数下η≈0
        result = proxy.eta_pid_adjust(200.0, 24.0, 6.0, 2.0, eta=0.001)
        assert result["triggered"] is False
        assert result["delta_current"] == 0.0

    def test_eta_pid_trigger(self, proxy):
        """η高于阈值→触发调整."""
        # 大偏差参数→高η
        result = proxy.eta_pid_adjust(300.0, 32.0, 3.0, 1.0, eta=0.5)
        assert result["triggered"] is True
        assert result["delta_current"] != 0.0

    def test_eta_pid_adjustment_direction(self, proxy):
        """η-PID调整方向: 参数向最优值靠拢."""
        # 电流过高(300 vs 200), 调整应为负
        result = proxy.eta_pid_adjust(300.0, 24.0, 6.0, 2.0, eta=0.3)
        if result["triggered"]:
            # 电流过高时, 调整应使电流降低 (向最优200靠拢)
            assert result["adjusted_current"] <= 300.0

    def test_eta_pid_limited_adjustment(self, proxy):
        """调整幅度受限: 不超过原参数的15%."""
        result = proxy.eta_pid_adjust(200.0, 24.0, 6.0, 2.0, eta=1.0)
        if result["triggered"]:
            assert abs(result["delta_current"]) <= 200.0 * 0.15 + 0.01
            assert abs(result["delta_voltage"]) <= 24.0 * 0.15 + 0.01

    def test_intent_safety_safe(self, proxy):
        """正常参数→SAFE."""
        level, label = proxy.classify_intent_safety(200.0, 24.0, 6.0)
        assert level == 0
        assert label == "SAFE"

    def test_intent_safety_suspicious(self, proxy):
        """边界参数→SUSPICIOUS."""
        level, label = proxy.classify_intent_safety(250.0, 30.0, 6.0)
        assert level >= 1

    def test_intent_safety_dangerous(self, proxy):
        """超限参数→DANGEROUS."""
        level, label = proxy.classify_intent_safety(350.0, 40.0, 6.0)
        assert level >= 2

    def test_intent_safety_critical(self, proxy):
        """严重超限→CRITICAL."""
        level, label = proxy.classify_intent_safety(500.0, 60.0, 6.0)
        assert level == 3
        assert label == "CRITICAL"

    def test_material_affects_cooling(self):
        """不同材料→不同冷却速率."""
        steel = WeldingProcessProxy("flat", material="steel")
        aluminum = WeldingProcessProxy("flat", material="aluminum")
        t85_steel = steel.compute_cooling_rate(200, 24, 6)
        t85_al = aluminum.compute_cooling_rate(200, 24, 6)
        # 铝热导率(167) > 钢(50), 所以铝冷却更快(t85更短)
        assert t85_al < t85_steel

    def test_weld_type_count_25(self, proxy):
        """焊缝类型总数=25."""
        assert len(proxy.WELD_TYPE_OPTIMAL_PARAMS) == 25

    def test_generic_fallback_exists(self, proxy):
        """generic类型存在."""
        assert "generic" in proxy.WELD_TYPE_OPTIMAL_PARAMS

    # ═══════════════════════════════════════════════════════════════════
    # v0.21.0: 材质库扩展 + MUS多假设材质辨识器 测试
    # ═══════════════════════════════════════════════════════════════════

    def test_material_count_11(self, proxy):
        """材料总数=11 (10种+generic兜底)."""
        assert len(proxy.MATERIAL_PROPERTIES) == 11

    def test_new_materials_present(self, proxy):
        """新增6种材料均存在."""
        expected_new = ["copper", "nickel", "cast_iron", "inconel",
                        "magnesium", "bronze", "generic"]
        for mat in expected_new:
            assert mat in proxy.MATERIAL_PROPERTIES, f"Missing material: {mat}"

    def test_copper_cooling_fast(self):
        """铜热导率高(398), t85应比钢(50)短 — 散热更快."""
        steel = WeldingProcessProxy("flat", material="steel")
        copper = WeldingProcessProxy("flat", material="copper")
        t85_steel = steel.compute_cooling_rate(200, 24, 6)
        t85_copper = copper.compute_cooling_rate(200, 24, 6)
        assert t85_copper < t85_steel, \
            f"Copper t85 ({t85_copper:.1f}s) should be < steel t85 ({t85_steel:.1f}s)"

    def test_inconel_cooling_slow(self):
        """Inconel热导率低(9.8), t85应比钢(50)长 — 散热更慢."""
        steel = WeldingProcessProxy("flat", material="steel")
        inconel = WeldingProcessProxy("flat", material="inconel")
        t85_steel = steel.compute_cooling_rate(200, 24, 6)
        t85_inconel = inconel.compute_cooling_rate(200, 24, 6)
        assert t85_inconel > t85_steel, \
            f"Inconel t85 ({t85_inconel:.1f}s) should be > steel t85 ({t85_steel:.1f}s)"

    def test_generic_material_fallback(self):
        """未知材料名不报错, 用generic兜底."""
        proxy = WeldingProcessProxy("flat", material="unknown_alloy_xyz")
        # 应该不报错, 且热导率应等于generic (=50.0)
        assert proxy._thermal_conductivity == 50.0
        # 应该能正常预测
        result = proxy.predict(current=200, voltage=24, travel_speed=6)
        assert np.isfinite(result.eta_residual)

    def test_identify_material_exact(self):
        """精确匹配已知材料属性 — 高置信度, mus_mode=False."""
        proxy = WeldingProcessProxy("flat")
        # 精确提供copper的属性值
        result = proxy.identify_material(
            observed_thermal_conductivity=398.0,
            observed_density=8960.0,
            observed_melting_point=1085.0,
        )
        assert result["best_match"] == "copper"
        assert result["confidence"] > 0.9
        assert result["mus_mode"] is False
        # mus_mode=False 时只保留1个假设
        assert len(result["hypotheses"]) == 1

    def test_identify_material_mus_mode(self):
        """低置信度时mus_mode=True, 保留top-3假设."""
        proxy = WeldingProcessProxy("flat")
        # 提供远离所有已知材料的极端值, 确保最高置信度 < 0.6
        # k=1000 远超铜(398), density=1500 低于所有材料, mp=200 远低于所有材料
        result = proxy.identify_material(
            observed_thermal_conductivity=1000.0,
            observed_density=1500.0,
            observed_melting_point=200.0,
        )
        assert result["mus_mode"] is True
        assert result["confidence"] < 0.6
        assert len(result["hypotheses"]) <= 3
        assert len(result["hypotheses"]) >= 1

    def test_identify_material_no_input(self):
        """无输入返回generic + mus_mode=True."""
        proxy = WeldingProcessProxy("flat")
        result = proxy.identify_material()
        assert result["best_match"] == "generic"
        assert result["confidence"] == 0.0
        assert result["mus_mode"] is True
        assert result["hypotheses"] == []

    def test_identify_material_partial_input(self):
        """仅提供部分属性也能辨识."""
        proxy = WeldingProcessProxy("flat")
        # 仅提供铝的热导率
        result = proxy.identify_material(
            observed_thermal_conductivity=167.0,
        )
        assert result["best_match"] == "aluminum"
        assert result["confidence"] > 0.5

    def test_identify_material_generic_excluded(self):
        """generic不出现在hypotheses中."""
        proxy = WeldingProcessProxy("flat")
        result = proxy.identify_material(
            observed_thermal_conductivity=50.0,
            observed_density=7850.0,
            observed_melting_point=1500.0,
        )
        mat_names = [h["material"] for h in result["hypotheses"]]
        assert "generic" not in mat_names

    def test_identify_material_returns_correct_structure(self):
        """identify_material返回正确的字典结构."""
        proxy = WeldingProcessProxy("flat")
        result = proxy.identify_material(
            observed_thermal_conductivity=50.0,
            observed_density=7850.0,
        )
        assert "best_match" in result
        assert "confidence" in result
        assert "mus_mode" in result
        assert "hypotheses" in result
        assert isinstance(result["best_match"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["mus_mode"], bool)
        assert isinstance(result["hypotheses"], list)

    def test_dikwp_rule_copper(self):
        """铜材料规则R007匹配."""
        proxy = WeldingProcessProxy("flat", material="copper")
        result = proxy.apply_dikwp_rules(misalignment=2.0, gap=1.0)
        assert "R007_copper_high_k" in result["matched_rules"]
        assert result["adjust_current_A"] == 15.0

    def test_dikwp_rule_inconel(self):
        """Inconel规则R009匹配."""
        proxy = WeldingProcessProxy("flat", material="inconel")
        result = proxy.apply_dikwp_rules(misalignment=1.5, gap=1.0)
        assert "R009_inconel_low_k" in result["matched_rules"]
        assert result["adjust_current_A"] == -5.0

    def test_dikwp_rule_count_10(self, proxy):
        """DIKWP规则库应有10条规则."""
        assert len(proxy.DIKWP_RULE_BASE) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
