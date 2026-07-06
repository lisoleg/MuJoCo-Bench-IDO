"""
焊接安全约束测试
================

测试 WeldingPsiAnchor 的安全检查功能:
  - 干伸长检查 (正常/过高/过低)
  - 回烧检查 (正常/触发)
  - 气孔风险检查 (正常/警告/严重)
  - check_all 综合检查 (全通过/多违规)

Author: MuJoCo-Bench-IDO Welding Module v0.1.0
"""

import os
import sys
import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.welding_psi_anchor import (
    WeldingPsiAnchor,
    WELDING_SAFETY_THRESHOLDS,
)


class TestStickOut:
    """测试干伸长检查."""

    def setup_method(self):
        """每个测试前创建新的 WeldingPsiAnchor."""
        self.anchor = WeldingPsiAnchor()

    def test_stick_out_normal(self):
        """stickout=15mm → passed."""
        result = self.anchor.check_stick_out(15.0)
        assert result["passed"] is True, "15mm stickout should pass"
        assert result["severity"] == "none"

    def test_stick_out_too_high(self):
        """stickout=26mm → violation (warning). 业界标准阈值 MAX=25mm."""
        result = self.anchor.check_stick_out(26.0)
        assert result["passed"] is False, "26mm stickout should fail"
        assert result["severity"] == "warning"
        assert result["violation"] is not None

    def test_stick_out_too_low(self):
        """stickout=1mm → critical violation. 业界标准阈值 MIN=8mm."""
        result = self.anchor.check_stick_out(1.0)
        assert result["passed"] is False, "1mm stickout should fail"
        assert result["severity"] == "critical"
        assert result["violation"] is not None

    def test_stick_out_at_boundary_max(self):
        """stickout=25mm (边界) → passed. 业界标准 MAX=25mm."""
        result = self.anchor.check_stick_out(25.0)
        assert result["passed"] is True, "25mm (boundary) should pass"

    def test_stick_out_at_boundary_min(self):
        """stickout=8mm (边界) → passed. 业界标准 MIN=8mm."""
        result = self.anchor.check_stick_out(8.0)
        assert result["passed"] is True, "8mm (boundary) should pass"

    def test_stick_out_just_over_max(self):
        """stickout=25.1mm → warning."""
        result = self.anchor.check_stick_out(25.1)
        assert result["passed"] is False
        assert result["severity"] == "warning"

    def test_stick_out_just_under_min(self):
        """stickout=7.9mm → critical."""
        result = self.anchor.check_stick_out(7.9)
        assert result["passed"] is False
        assert result["severity"] == "critical"


class TestBurnBack:
    """测试回烧检查."""

    def setup_method(self):
        """每个测试前创建新的 WeldingPsiAnchor."""
        self.anchor = WeldingPsiAnchor()

    def test_burn_back_normal(self):
        """current=200, voltage=24 → passed."""
        result = self.anchor.check_burn_back(200.0, 24.0, 15.0)
        assert result["passed"] is True
        assert result["severity"] == "none"

    def test_burn_back_triggered(self):
        """current=360, voltage=4 → critical."""
        result = self.anchor.check_burn_back(360.0, 4.0, 15.0)
        assert result["passed"] is False, "High current + low voltage should trigger burn-back"
        assert result["severity"] == "critical"
        assert result["violation"] is not None

    def test_burn_back_high_current_normal_voltage(self):
        """current=360, voltage=24 (高电流但电压正常) → passed."""
        result = self.anchor.check_burn_back(360.0, 24.0, 15.0)
        assert result["passed"] is True, "High current with normal voltage should pass"

    def test_burn_back_normal_current_low_voltage(self):
        """current=200, voltage=4 (低电压但电流正常) → passed."""
        result = self.anchor.check_burn_back(200.0, 4.0, 15.0)
        assert result["passed"] is True, "Low voltage with normal current should pass"

    def test_burn_back_boundary_current(self):
        """current=350 (边界), voltage=4 → passed (current not > MAX)."""
        result = self.anchor.check_burn_back(350.0, 4.0, 15.0)
        assert result["passed"] is True, "current=350 (boundary) should not trigger"


class TestPorosity:
    """测试气孔风险检查."""

    def setup_method(self):
        """每个测试前创建新的 WeldingPsiAnchor."""
        self.anchor = WeldingPsiAnchor()

    def test_porosity_normal(self):
        """arc_var=0.3 → passed."""
        result = self.anchor.check_porosity_risk(0.3)
        assert result["passed"] is True
        assert result["severity"] == "none"

    def test_porosity_warning(self):
        """arc_var=0.6 → warning."""
        result = self.anchor.check_porosity_risk(0.6)
        assert result["passed"] is False, "arc_var=0.6 > threshold 0.5 → warning"
        assert result["severity"] == "warning"

    def test_porosity_critical(self):
        """arc_var=1.2 → critical (> 2× threshold)."""
        result = self.anchor.check_porosity_risk(1.2)
        assert result["passed"] is False, "arc_var=1.2 > 2×threshold → critical"
        assert result["severity"] == "critical"

    def test_porosity_at_threshold(self):
        """arc_var=0.5 (边界) → passed."""
        result = self.anchor.check_porosity_risk(0.5)
        assert result["passed"] is True, "arc_var=0.5 (at threshold) should pass"

    def test_porosity_at_critical_boundary(self):
        """arc_var=1.0 (= 2×threshold) → warning (not > 2×threshold, but > threshold)."""
        result = self.anchor.check_porosity_risk(1.0)
        assert result["passed"] is False, "arc_var=1.0 (> threshold 0.5) → warning"
        assert result["severity"] == "warning"

    def test_porosity_zero_variance(self):
        """arc_var=0.0 → passed."""
        result = self.anchor.check_porosity_risk(0.0)
        assert result["passed"] is True


class TestCheckAll:
    """测试 check_all 综合检查."""

    def test_check_all_passed(self):
        """全部正常 → passed."""
        anchor = WeldingPsiAnchor()
        state = {
            "stickout": 15.0,
            "current": 200.0,
            "voltage": 24.0,
            "arc_length_variance": 0.3,
            "seam_deviation": 0.1,
        }
        result = anchor.check_all(state)
        assert result["passed"] is True, "All normal should pass"
        assert len(result["violations"]) == 0
        assert len(result["actions"]) == 0

    def test_check_all_multiple_violations(self):
        """多个违规同时触发. 业界标准阈值: MAX=25mm, SEAM_DEV_MAX=2.0mm."""
        anchor = WeldingPsiAnchor()
        state = {
            "stickout": 40.0,          # > 25 → warning
            "current": 360.0,          # > 350
            "voltage": 4.0,            # < 5 → burn-back critical
            "arc_length_variance": 1.2,  # > 2×threshold → critical
            "seam_deviation": 3.0,     # > 2.0 → warning
        }
        result = anchor.check_all(state)
        assert result["passed"] is False, "Multiple violations should fail"
        assert len(result["violations"]) >= 3, "Should have multiple violations"
        assert len(result["actions"]) >= 3, "Should have multiple actions"

    def test_check_all_stickout_critical(self):
        """stickout 过低 → check_all 返回 critical. 业界阈值 MIN=8mm."""
        anchor = WeldingPsiAnchor()
        state = {
            "stickout": 1.0,
            "current": 200.0,
            "voltage": 24.0,
            "arc_length_variance": 0.1,
            "seam_deviation": 0.1,
        }
        result = anchor.check_all(state)
        assert result["passed"] is False
        assert result["details"]["stick_out"]["severity"] == "critical"

    def test_check_all_seam_deviation(self):
        """焊缝偏差过大 → warning (不阻止焊接, 但记录违规). 仿真阈值 SEAM_DEV_MAX=2.0mm."""
        anchor = WeldingPsiAnchor()
        state = {
            "stickout": 15.0,
            "current": 200.0,
            "voltage": 24.0,
            "arc_length_variance": 0.1,
            "seam_deviation": 3.0,  # > 2.0 → warning
        }
        result = anchor.check_all(state)
        # warning 级别不阻止焊接步骤, 但记录违规和动作
        assert result["details"]["seam_dev"]["severity"] == "warning"
        assert len(result["violations"]) >= 1, "Should record seam deviation violation"
        assert any("ADJUST_TRACKING" in a for a in result["actions"]), "Should suggest tracking adjustment"

    def test_check_all_missing_keys(self):
        """缺少键时使用默认值, 不应崩溃."""
        anchor = WeldingPsiAnchor()
        state = {}  # 空字典
        result = anchor.check_all(state)
        # 默认值都是正常的, 应该通过
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_check_all_actions_content(self):
        """check_all 返回的 actions 包含正确的动作建议."""
        anchor = WeldingPsiAnchor()
        state = {
            "stickout": 1.0,          # critical (< 8mm) → EMERGENCY_STOP
            "current": 360.0,
            "voltage": 4.0,           # critical → STOP_WIRE_FEED
            "arc_length_variance": 0.1,
            "seam_deviation": 0.1,
        }
        result = anchor.check_all(state)
        assert not result["passed"]
        actions_str = " ".join(result["actions"])
        assert "EMERGENCY_STOP" in actions_str or "STOP_WIRE" in actions_str


class TestArcHistory:
    """测试电弧历史和方差计算."""

    def test_update_arc_history(self):
        """更新电弧长度历史."""
        anchor = WeldingPsiAnchor()
        anchor.update_arc_history(15.0)
        anchor.update_arc_history(16.0)
        assert len(anchor._arc_length_history) == 2

    def test_compute_arc_variance(self):
        """计算电弧长度方差."""
        anchor = WeldingPsiAnchor()
        for val in [15.0, 15.0, 15.0, 15.0]:
            anchor.update_arc_history(val)
        var = anchor.compute_arc_variance()
        assert var == 0.0, "Constant arc length should have 0 variance"

    def test_arc_history_max_size(self):
        """电弧历史最多保留 50 个记录."""
        anchor = WeldingPsiAnchor()
        for i in range(60):
            anchor.update_arc_history(float(i))
        assert len(anchor._arc_length_history) <= 50

    def test_compute_arc_variance_empty(self):
        """空历史方差为 0."""
        anchor = WeldingPsiAnchor()
        assert anchor.compute_arc_variance() == 0.0


class TestCustomThresholds:
    """测试自定义阈值."""

    def test_custom_thresholds(self):
        """自定义阈值生效."""
        custom = {
            "STICK_OUT_MIN": 10.0,
            "STICK_OUT_MAX": 20.0,
            "MAX_CURRENT": 300.0,
            "MIN_VOLTAGE": 6.0,
            "ARC_VAR_THRESHOLD": 0.3,
            "MAX_HEAT_INPUT": 2.0,
            "SEAM_DEV_MAX": 0.3,
        }
        anchor = WeldingPsiAnchor(thresholds=custom)
        # stickout=9 → < custom MIN(10) → critical
        result = anchor.check_stick_out(9.0)
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_default_thresholds(self):
        """默认阈值与 WELDING_SAFETY_THRESHOLDS 一致."""
        anchor = WeldingPsiAnchor()
        assert anchor.thresholds["STICK_OUT_MIN"] == WELDING_SAFETY_THRESHOLDS["STICK_OUT_MIN"]
        assert anchor.thresholds["STICK_OUT_MAX"] == WELDING_SAFETY_THRESHOLDS["STICK_OUT_MAX"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
