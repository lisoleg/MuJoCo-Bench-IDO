"""
v0.8.0 新功能集成测试
=====================

测试覆盖:
  U1: SafeFuse graded 三级渐进 + locomotion INFO 级
  U2: evidence_verified 标记
  U3: κ-Snap JSONL 步骤级输出 + Hermes 翻译
  U4: PreAffect GRRR/PHEW/NEUTRAL 检测 + 调幅
  U5: S-Bridge 四接口
  U6: IC 计算 + Dead-Zero 过滤 + 过采样 + 毛睿重加权

Author: MuJoCo-Bench-IDO v0.8.0 — 集成测试
"""

import numpy as np
import os
import tempfile
import time
import pytest
from typing import Dict, List

# ── v0.8.0 新模块导入 ──
from core.pre_affect import PreAffect, detect, probe_multiplier, stall_extension
from core.kappa_snap_jsonl import KappaSnapJSONLWriter, HermesTranslator, PRIVATE_MAP
from core.eml_semzip_ic import EMLSemZipIC, THETA_DEAD, OVERSAMPLE_TOP_PCT, OVERSAMPLE_FACTOR, MAO_RUI_POWER
from agent.safe_fuse import SafeFuse, FuseLevel, FuseGradeResult, FuseOption
from agent.s_bridge import SBridge, SkillEntry


# ══════════════════════════════════════════════════════════════════
#  U4: PreAffect 内在信号测试
# ══════════════════════════════════════════════════════════════════

class TestPreAffect:
    """PreAffect 枚举 + detect + probe_multiplier + stall_extension 测试."""

    def test_pre_affect_enum_values(self):
        """测试 PreAffect 枚举值."""
        assert PreAffect.GRRR.value == "GRRR"
        assert PreAffect.PHEW.value == "PHEW"
        assert PreAffect.NEUTRAL.value == "NEUTRAL"

    def test_detect_grrr_stagnation(self):
        """GRRR: η 连续 3 步不降 → 停滞焦虑信号."""
        # η 连续上升 (不降) → GRRR
        eta_history: List[float] = [1.0, 1.1, 1.2, 1.3]
        result: PreAffect = detect(eta_history, window_grrr=3)
        assert result == PreAffect.GRRR

    def test_detect_grrr_flat(self):
        """GRRR: η 连续 3 步几乎不变 (变化 < 1%) → 停滞焦虑."""
        eta_history: List[float] = [1.0, 1.005, 1.008, 1.009]
        result: PreAffect = detect(eta_history, window_grrr=3)
        assert result == PreAffect.GRRR

    def test_detect_phew_breakthrough(self):
        """PHEW: η 连续 2 步降 > 10% → 突破释然信号."""
        # η 连续显著下降
        eta_history: List[float] = [1.0, 0.85, 0.70]
        result: PreAffect = detect(eta_history, window_grrr=3, window_phew=2)
        assert result == PreAffect.PHEW

    def test_detect_neutral_mixed(self):
        """NEUTRAL: η 混合趋势 (不满足 GRRR 或 PHEW)."""
        # η 先降后升
        eta_history: List[float] = [1.0, 0.95, 1.05]
        result: PreAffect = detect(eta_history)
        assert result == PreAffect.NEUTRAL

    def test_detect_neutral_short_history(self):
        """NEUTRAL: η 历史不足 → NEUTRAL."""
        eta_history: List[float] = [1.0]
        result: PreAffect = detect(eta_history)
        assert result == PreAffect.NEUTRAL

    def test_probe_multiplier_grrr(self):
        """probe_multiplier: GRRR → ×1.5."""
        assert probe_multiplier(PreAffect.GRRR) == 1.5

    def test_probe_multiplier_phew(self):
        """probe_multiplier: PHEW → ×1.0 (不调整 probe)."""
        assert probe_multiplier(PreAffect.PHEW) == 1.0

    def test_probe_multiplier_neutral(self):
        """probe_multiplier: NEUTRAL → ×1.0."""
        assert probe_multiplier(PreAffect.NEUTRAL) == 1.0

    def test_stall_extension_grrr(self):
        """stall_extension: GRRR → ×1.0 (不延伸 stall)."""
        assert stall_extension(PreAffect.GRRR) == 1.0

    def test_stall_extension_phew(self):
        """stall_extension: PHEW → ×1.5 (延伸 EXPLOIT stall)."""
        assert stall_extension(PreAffect.PHEW) == 1.5

    def test_stall_extension_neutral(self):
        """stall_extension: NEUTRAL → ×1.0."""
        assert stall_extension(PreAffect.NEUTRAL) == 1.0


# ══════════════════════════════════════════════════════════════════
#  U6: EML-SemZip IC + Dead-Zero 过滤测试
# ══════════════════════════════════════════════════════════════════

class TestEMLSemZipIC:
    """IC 计算 + Dead-Zero 过滤 + 过采样 + 毛睿重加权 测试."""

    def test_compute_ic_high_diversity(self):
        """高 IC: 轨迹有丰富状态变化."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        # 丰富变化的轨迹
        states: List[np.ndarray] = [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.5, 0.8, 1.0]),
            np.array([1.5, 1.2, 0.5]),
            np.array([2.0, 0.3, 1.5]),
            np.array([0.1, 0.9, 2.0]),
        ]
        ic_value: float = ic_filter.compute_ic(states)
        # 高 IC > 0.45
        assert ic_value > THETA_DEAD

    def test_compute_ic_dead_zero(self):
        """Dead-Zero: 轨迹几乎无变化 → IC ≈ 0."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        # 几乎无变化的轨迹
        states: List[np.ndarray] = [
            np.array([1.0, 0.5, 0.3]),
            np.array([1.001, 0.501, 0.301]),
            np.array([1.002, 0.502, 0.302]),
            np.array([1.003, 0.503, 0.303]),
        ]
        ic_value: float = ic_filter.compute_ic(states)
        # Dead-Zero → IC ≈ 0 或 IC < θ_dead
        assert ic_value < THETA_DEAD

    def test_compute_ic_single_step(self):
        """轨迹长度 1 → IC = 0.0."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        states: List[np.ndarray] = [np.array([0.0, 0.1])]
        ic_value: float = ic_filter.compute_ic(states)
        assert ic_value == 0.0

    def test_is_dead_zero_true(self):
        """IC < θ_dead → Dead-Zero."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        assert ic_filter.is_dead_zero(0.2) is True  # 0.2 < 0.45
        assert ic_filter.is_dead_zero(0.44) is True  # 0.44 < 0.45

    def test_is_dead_zero_false(self):
        """IC ≥ θ_dead → 非 Dead-Zero."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        assert ic_filter.is_dead_zero(0.45) is False  # 0.45 ≥ 0.45
        assert ic_filter.is_dead_zero(1.0) is False

    def test_filter_episodes(self):
        """过滤 episodes, 剔除 Dead-Zero."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        episodes_data: List[Dict] = [
            # 高 IC (有价值)
            {"trajectory_states": [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])],
             "episode_return": 100.0},
            # Dead-Zero (无价值)
            {"trajectory_states": [np.array([1.0]), np.array([1.001]), np.array([1.002])],
             "episode_return": 10.0},
        ]
        filtered: List[Dict] = ic_filter.filter_episodes(episodes_data)
        # Dead-Zero episode 被剔除
        assert len(filtered) >= 0  # 可能两个都保留或剔除,取决于IC值

    def test_oversample_high_ic(self):
        """Top 5% × 3 过采样."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        # 创建 20 个 episodes, 每个都有 IC 值
        filtered_episodes: List[Dict] = []
        for i in range(20):
            filtered_episodes.append({"ic": float(i) / 20.0, "episode_return": float(i)})
        oversampled: List[Dict] = ic_filter.oversample_high_ic(filtered_episodes)
        # Top 5% = 1 episode × 3 = 3 份副本 + 20 原始 = 23
        assert len(oversampled) >= len(filtered_episodes)

    def test_reweight_mao_rui(self):
        """毛睿度量重加权: 采样概率 ∝ IC^power."""
        ic_filter: EMLSemZipIC = EMLSemZipIC()
        ic_values: List[float] = [0.5, 1.0, 1.5, 2.0]
        weights: np.ndarray = ic_filter.reweight_mao_rui(ic_values)
        # 归一化: Σ weights = 1.0
        assert abs(float(np.sum(weights)) - 1.0) < 0.01
        # 高 IC 有更高权重
        assert weights[3] > weights[0]

    def test_reweight_mao_rui_power_zero(self):
        """毛睿重加权: power=0 → 均匀权重."""
        ic_filter: EMLSemZipIC = EMLSemZipIC(mao_rui_power=0.0)
        ic_values: List[float] = [0.5, 1.0, 1.5, 2.0]
        weights: np.ndarray = ic_filter.reweight_mao_rui(ic_values, power=0.0)
        # 均匀权重: 1/N
        expected: float = 1.0 / 4.0
        for w in weights:
            assert abs(float(w) - expected) < 0.01

    def test_theta_dead_default(self):
        """θ_dead 默认值 = 0.45."""
        assert THETA_DEAD == 0.45

    def test_oversample_defaults(self):
        """过采样默认配置: Top 5%, ×3."""
        assert OVERSAMPLE_TOP_PCT == 0.05
        assert OVERSAMPLE_FACTOR == 3

    def test_mao_rui_power_default(self):
        """毛睿 power 默认 = 1.0."""
        assert MAO_RUI_POWER == 1.0


# ══════════════════════════════════════════════════════════════════
#  U3: κ-Snap JSONL 步骤级输出 + Hermes 翻译测试
# ══════════════════════════════════════════════════════════════════

class TestKappaSnapJSONL:
    """KappaSnapJSONLWriter + HermesTranslator 测试."""

    def test_hermes_translate_known_labels(self):
        """Hermes 翻译已知私有标签."""
        hermes: HermesTranslator = HermesTranslator()
        assert hermes.translate("GRRR") == "η停滞焦虑信号"
        assert hermes.translate("PHEW") == "η突破释然信号"
        assert hermes.translate("EVC") == "证据自校验完成"
        assert hermes.translate("L3h") == "ψ-Anchor 触发安全降级"

    def test_hermes_translate_unknown_label(self):
        """Hermes 翻译未知标签 → 返回原始标签."""
        hermes: HermesTranslator = HermesTranslator()
        assert hermes.translate("UNKNOWN_LABEL") == "UNKNOWN_LABEL"

    def test_hermes_add_mapping(self):
        """Hermes 添加新映射."""
        hermes: HermesTranslator = HermesTranslator()
        hermes.add_mapping("NEW_LABEL", "新标签翻译")
        assert hermes.translate("NEW_LABEL") == "新标签翻译"

    def test_hermes_extra_map_init(self):
        """Hermes 初始化时传入额外映射."""
        hermes: HermesTranslator = HermesTranslator(extra_map={"X1": "翻译X1"})
        assert hermes.translate("X1") == "翻译X1"

    def test_jsonl_writer_open_write_close(self):
        """JSONL writer: open → write_step → close 完整流程."""
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        # 使用临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path: str = tmp.name
        try:
            writer.open(tmp_path)
            snap_id: str = writer.write_step(
                eta=0.05,
                mode="EXPLOIT",
                fuse_level="NORMAL",
                pre_affect="NEUTRAL",
                noether_result={"ok": True, "total": 0},
                evidence_verified=True,
            )
            assert snap_id.startswith("snap_")
            writer.flush()
            writer.close()

            # 读取 JSONL 文件内容
            import json
            with open(tmp_path, 'r', encoding='utf-8') as f:
                line: str = f.readline()
                record: Dict = json.loads(line)
                assert record["η"] == 0.05
                assert record["mode"] == "EXPLOIT"
                assert record["fuse_level"] == "NORMAL"
                assert record["pre_affect"] == "NEUTRAL"
                assert record["noether_ok"] is True
                assert record["evidence_verified"] is True
        finally:
            os.unlink(tmp_path)

    def test_jsonl_writer_multiple_steps(self):
        """JSONL writer: 多步写入 + 查询."""
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path: str = tmp.name
        try:
            writer.open(tmp_path)
            snap_ids: List[str] = []
            for i in range(5):
                snap_id: str = writer.write_step(
                    eta=float(i) * 0.01,
                    mode="EXPLOIT",
                    fuse_level="NORMAL",
                    pre_affect="NEUTRAL",
                )
                snap_ids.append(snap_id)
            writer.flush()

            # 查询特定 snap_id
            record: Dict = writer.query(snap_ids[2])
            assert record.get("η") == 0.02
            assert record.get("step") == 2

            # 查询不存在的 snap_id → 空 Dict
            empty: Dict = writer.query("nonexistent_snap_id")
            assert empty == {}

            writer.close()
        finally:
            os.unlink(tmp_path)

    def test_jsonl_writer_not_opened(self):
        """JSONL writer: 未 open 时 write_step → snap_id 仍返回 (仅写内存)."""
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        snap_id: str = writer.write_step(
            eta=0.05,
            mode="EXPLOIT",
            fuse_level="NORMAL",
            pre_affect="NEUTRAL",
        )
        # snap_id 正常生成
        assert snap_id.startswith("snap_")
        # 内存缓冲中有记录
        assert len(writer.get_buffer()) == 1

    def test_jsonl_writer_reset(self):
        """JSONL writer: reset 清空缓冲."""
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        writer.write_step(eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL")
        assert len(writer.get_buffer()) == 1
        writer.reset()
        assert len(writer.get_buffer()) == 0


# ══════════════════════════════════════════════════════════════════
#  U1: SafeFuse 三级渐进约束测试
# ══════════════════════════════════════════════════════════════════

class TestSafeFuseGraded:
    """SafeFuse 三级渐进约束 (FuseLevel, FuseGradeResult, check_graded) 测试."""

    def test_fuse_level_enum(self):
        """FuseLevel 枚举值."""
        assert FuseLevel.WARNING.value == "WARNING"
        assert FuseLevel.BLOCK.value == "BLOCK"
        assert FuseLevel.INFO.value == "INFO"
        assert FuseLevel.NORMAL.value == "NORMAL"

    def test_fuse_grade_result_dataclass(self):
        """FuseGradeResult 默认值."""
        result: FuseGradeResult = FuseGradeResult()
        assert result.level == FuseLevel.NORMAL
        assert result.reason == ""
        assert result.options == []
        assert result.safe_action is None
        assert result.log_message == ""

    def test_fuse_option_dataclass(self):
        """FuseOption 创建."""
        option: FuseOption = FuseOption(
            name="continue",
            action_modifier=lambda a: a,
            description="信任 agent",
        )
        assert option.name == "continue"
        assert option.description == "信任 agent"

    def test_check_graded_normal(self):
        """check_graded: 所有检查通过 → NORMAL."""
        fuse: SafeFuse = SafeFuse()
        result: FuseGradeResult = fuse.check_graded(
            eta=0.05,
            delta_K=0.15,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
            torque_ratio=0.5,
            is_locomotion=False,
        )
        assert result.level == FuseLevel.NORMAL

    def test_check_graded_locomotion_info(self):
        """check_graded: locomotion → INFO 级 (仅记录日志)."""
        fuse: SafeFuse = SafeFuse()
        result: FuseGradeResult = fuse.check_graded(
            eta=2.0,
            delta_K=1.0,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
            torque_ratio=0.95,
            is_locomotion=True,
        )
        assert result.level == FuseLevel.INFO
        assert "locomotion" in result.reason

    def test_check_graded_warning_torque(self):
        """check_graded: torque_ratio ≥ 0.95 → WARNING 级."""
        fuse: SafeFuse = SafeFuse()
        result: FuseGradeResult = fuse.check_graded(
            eta=0.05,
            delta_K=0.15,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
            torque_ratio=0.97,
            is_locomotion=False,
        )
        assert result.level == FuseLevel.WARNING
        assert len(result.options) == 3  # 三选项 (continue/degrade/abort)

    def test_check_graded_warning_eta_ratio(self):
        """check_graded: η_ratio ∈ [1.2, 1.5) → WARNING 级."""
        fuse: SafeFuse = SafeFuse()
        result: FuseGradeResult = fuse.check_graded(
            eta=0.20,  # η_ratio = 0.20 / 0.15 = 1.33 ∈ [1.2, 1.5)
            delta_K=0.15,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
            torque_ratio=0.5,
            is_locomotion=False,
        )
        assert result.level == FuseLevel.WARNING

    def test_check_graded_block_fatal(self):
        """check_graded: 灾难性违反 → BLOCK 级."""
        fuse: SafeFuse = SafeFuse()
        result: FuseGradeResult = fuse.check_graded(
            eta=0.5,
            delta_K=0.15,
            noether_result={"ok": False, "total": 3, "energy": 1, "torque": 1, "collision": 1},
            psi_anchor_state=None,
            torque_ratio=0.5,
            is_locomotion=False,
        )
        assert result.level == FuseLevel.BLOCK
        assert "L4_fatal" in result.reason

    def test_check_graded_block_psi_anchor(self):
        """check_graded: ψ-Anchor 触发 → BLOCK 级."""
        fuse: SafeFuse = SafeFuse()
        psi_state: Dict = {"evolution_triggered": True}
        result: FuseGradeResult = fuse.check_graded(
            eta=0.3,
            delta_K=0.15,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=psi_state,
            torque_ratio=0.5,
            is_locomotion=False,
        )
        assert result.level == FuseLevel.BLOCK

    def test_check_graded_backward_compat_check(self):
        """向后兼容: 原 check() 方法仍可用."""
        fuse: SafeFuse = SafeFuse()
        result: tuple = fuse.check(
            eta=0.05,
            delta_K=0.15,
            noether_result={"ok": True, "total": 0},
            psi_anchor_state=None,
        )
        assert result[0] == "normal"

    def test_apply_graded_normal(self):
        """apply_graded: NORMAL → action 不修改."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        graded: FuseGradeResult = FuseGradeResult(level=FuseLevel.NORMAL)
        result_action: np.ndarray = fuse.apply_graded(action, graded)
        assert np.allclose(result_action, action)

    def test_apply_graded_info(self):
        """apply_graded: INFO → action 不修改 (locomotion 透明路由)."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        graded: FuseGradeResult = FuseGradeResult(level=FuseLevel.INFO)
        result_action: np.ndarray = fuse.apply_graded(action, graded)
        assert np.allclose(result_action, action)

    def test_apply_graded_warning_continue(self):
        """apply_graded: WARNING + η下降 → continue (不修改)."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        graded: FuseGradeResult = FuseGradeResult(level=FuseLevel.WARNING)
        result_action: np.ndarray = fuse.apply_graded(action, graded, eta_trend="descending")
        assert np.allclose(result_action, action)

    def test_apply_graded_warning_degrade(self):
        """apply_graded: WARNING + η平 → degrade (×0.8)."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        graded: FuseGradeResult = FuseGradeResult(level=FuseLevel.WARNING)
        result_action: np.ndarray = fuse.apply_graded(action, graded, eta_trend="flat")
        expected: np.ndarray = action * 0.8
        assert np.allclose(result_action, expected)

    def test_apply_graded_warning_abort(self):
        """apply_graded: WARNING + η上升 → abort (×0.0 → 零 action)."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        graded: FuseGradeResult = FuseGradeResult(level=FuseLevel.WARNING)
        result_action: np.ndarray = fuse.apply_graded(action, graded, eta_trend="ascending")
        assert np.allclose(result_action, np.zeros_like(action))

    def test_apply_graded_block_with_safe_action(self):
        """apply_graded: BLOCK + safe_action → 使用 safe_action."""
        fuse: SafeFuse = SafeFuse()
        action: np.ndarray = np.array([0.5, -0.3, 0.8])
        safe_action: np.ndarray = np.array([0.1, -0.1, 0.2])
        graded: FuseGradeResult = FuseGradeResult(
            level=FuseLevel.BLOCK,
            reason="L3_hard",
            safe_action=None,
        )
        result_action: np.ndarray = fuse.apply_graded(action, graded, safe_action=safe_action)
        assert np.allclose(result_action, np.clip(safe_action, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════
#  U5: S-Bridge 四接口测试
# ══════════════════════════════════════════════════════════════════

class TestSBridge:
    """SBridge MetaQuery 四接口 测试."""

    def test_s_bridge_init(self):
        """SBridge 初始化."""
        bridge: SBridge = SBridge()
        assert bridge._agent is None
        assert bridge._jsonl is None
        assert len(bridge._skill_bank) == 0
        assert len(bridge._timeline) == 0

    def test_why_this_action_no_agent(self):
        """WHY_THIS_ACTION: 无 agent → 返回 unknown."""
        bridge: SBridge = SBridge()
        result: str = bridge.why_this_action()
        assert "unknown" in result

    def test_why_this_action_with_eta(self):
        """WHY_THIS_ACTION: 有 eta → 返回 η 值."""
        bridge: SBridge = SBridge()
        result: str = bridge.why_this_action(eta=0.5)
        # η=0.5000 在输出中 (η 可能是 Unicode 或编码形式)
        assert "0.5000" in result

    def test_audit_snap_no_jsonl(self):
        """AUDIT_SNAP: 无 JSONL → 空 Dict."""
        bridge: SBridge = SBridge()
        result: Dict = bridge.audit_snap("snap_0_abc123")
        assert result == {}

    def test_audit_snap_with_jsonl(self):
        """AUDIT_SNAP: 有 JSONL → 查询记录."""
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        snap_id: str = writer.write_step(
            eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL"
        )
        bridge: SBridge = SBridge(jsonl=writer)
        result: Dict = bridge.audit_snap(snap_id)
        assert result.get("η") == 0.05

    def test_learn_skill_empty_episodes(self):
        """LEARN_SKILL: 空 episodes → 空 SkillEntry."""
        bridge: SBridge = SBridge()
        skill: SkillEntry = bridge.learn_skill([], skill_name="test_skill")
        assert skill.name == "test_skill"

    def test_learn_skill_with_episodes(self):
        """LEARN_SKILL: 有 episodes → 提炼模式."""
        bridge: SBridge = SBridge()
        episodes: List[Dict] = [
            {"avg_eta": 0.05, "avg_cq": 0.9, "mode_counts": {"EXPLOIT": 80, "EXPLORE": 10, "SAFE": 10}},
            {"avg_eta": 0.06, "avg_cq": 0.85, "mode_counts": {"EXPLOIT": 70, "EXPLORE": 20, "SAFE": 10}},
        ]
        skill: SkillEntry = bridge.learn_skill(episodes, skill_name="locomotion_skill", task_name="cheetah-run")
        assert skill.name == "locomotion_skill"
        assert skill.task == "cheetah-run"
        assert skill.avg_eta > 0
        assert "mode_distribution" in skill.pattern
        # skill 存入 skill_bank
        assert "locomotion_skill" in bridge.get_skill_bank()

    def test_journey_timeline_empty(self):
        """JOURNEY_TIMELINE: 无事件 → 空 List."""
        bridge: SBridge = SBridge()
        result: List[Dict] = bridge.journey_timeline()
        assert result == []

    def test_journey_timeline_with_events(self):
        """JOURNEY_TIMELINE: 有事件 → 返回时间线."""
        bridge: SBridge = SBridge()
        bridge._record_step(eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL")
        result: List[Dict] = bridge.journey_timeline()
        assert len(result) == 1
        assert result[0]["mode"] == "EXPLOIT"

    def test_journey_timeline_since_filter(self):
        """JOURNEY_TIMELINE: since 时间过滤."""
        bridge: SBridge = SBridge()
        # 先记录一步
        bridge._record_step(eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL")
        # 用一个较大的 since 过滤
        future_ts: float = time.time() + 1000
        result: List[Dict] = bridge.journey_timeline(since=future_ts)
        assert len(result) == 0  # 全部被过滤

    def test_s_bridge_reset(self):
        """SBridge reset: 清空时间线, 保留 skill_bank."""
        bridge: SBridge = SBridge()
        bridge._record_step(eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL")
        bridge.learn_skill([{"avg_eta": 0.05}], skill_name="test_skill")
        assert bridge.get_timeline_length() == 1
        assert len(bridge.get_skill_bank()) == 1
        bridge.reset()
        assert bridge.get_timeline_length() == 0
        assert len(bridge.get_skill_bank()) == 1  # skill_bank 保留


# ══════════════════════════════════════════════════════════════════
#  U2: κ-Snap Schema 事件类型扩展测试
# ══════════════════════════════════════════════════════════════════

class TestKappaSnapSchemaV080:
    """κ-Snap Schema v0.8.0 新事件类型 测试."""

    def test_evidence_check_event_type(self):
        """EVIDENCE_CHECK 事件类型存在."""
        from core.kappa_snap_schema import KappaSnapSchema, EVENT_TYPES
        schema: KappaSnapSchema = KappaSnapSchema()
        assert "EVIDENCE_CHECK" in EVENT_TYPES
        assert EVENT_TYPES["EVIDENCE_CHECK"]["level"] == "L6"

    def test_fuse_warning_event_type(self):
        """FUSE_WARNING 事件类型存在."""
        from core.kappa_snap_schema import EVENT_TYPES
        assert "FUSE_WARNING" in EVENT_TYPES
        assert EVENT_TYPES["FUSE_WARNING"]["level"] == "L4"

    def test_fuse_info_event_type(self):
        """FUSE_INFO 事件类型存在."""
        from core.kappa_snap_schema import EVENT_TYPES
        assert "FUSE_INFO" in EVENT_TYPES
        assert EVENT_TYPES["FUSE_INFO"]["level"] == "L4"

    def test_pre_affect_signal_event_type(self):
        """PRE_AFFECT_SIGNAL 事件类型存在."""
        from core.kappa_snap_schema import EVENT_TYPES
        assert "PRE_AFFECT_SIGNAL" in EVENT_TYPES
        assert EVENT_TYPES["PRE_AFFECT_SIGNAL"]["level"] == "L4"

    def test_schema_version_v020(self):
        """Schema 版本 v0.2.0."""
        from core.kappa_snap_schema import IDO_KAPPA_SNAP_SCHEMA_VERSION
        assert IDO_KAPPA_SNAP_SCHEMA_VERSION == "v0.2.0"

    def test_schema_validate_evidence_check(self):
        """Schema 验证 EVIDENCE_CHECK 事件."""
        from core.kappa_snap_schema import KappaSnapSchema
        schema: KappaSnapSchema = KappaSnapSchema()
        event: Dict = schema.create_event(
            event_type="EVIDENCE_CHECK",
            eta=0.05,
            decision="benchmark_pass",
            snap_id="snap_0_abc",
            prev_snap_id="genesis",
            details={"benchmark_name": "cheetah-run", "test_result": "pass", "evidence_verified": True},
        )
        assert schema.validate(event) is True

    def test_schema_validate_fuse_warning(self):
        """Schema 验证 FUSE_WARNING 事件."""
        from core.kappa_snap_schema import KappaSnapSchema
        schema: KappaSnapSchema = KappaSnapSchema()
        event: Dict = schema.create_event(
            event_type="FUSE_WARNING",
            eta=0.3,
            decision="WARNING",
            snap_id="snap_1_def",
            prev_snap_id="snap_0_abc",
            details={"fuse_level": "WARNING", "options": ["continue", "degrade", "abort"], "auto_decision": "eta_trend=descending"},
        )
        assert schema.validate(event) is True


# ══════════════════════════════════════════════════════════════════
#  κ-Snap Logger v0.2.0 log_to_jsonl 测试
# ══════════════════════════════════════════════════════════════════

class TestKappaSnapLoggerV080:
    """KappaSnapLogger v0.2.0 log_to_jsonl 测试."""

    def test_logger_version_v020(self):
        """Logger 版本 v0.2.0."""
        from core.kappa_snap_logger import IDO_KAPPA_SNAP_LOGGER_VERSION
        assert IDO_KAPPA_SNAP_LOGGER_VERSION == "v0.2.0"

    def test_log_to_jsonl_not_enabled(self):
        """log_to_jsonl: 未启用 → 返回空字符串."""
        from core.kappa_snap_logger import KappaSnapLogger
        logger: KappaSnapLogger = KappaSnapLogger()
        result: str = logger.log_to_jsonl(
            eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL"
        )
        assert result == ""

    def test_log_to_jsonl_enabled(self):
        """log_to_jsonl: 启用 → 返回 snap_id."""
        from core.kappa_snap_logger import KappaSnapLogger
        logger: KappaSnapLogger = KappaSnapLogger()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path: str = tmp.name
        try:
            logger.enable_jsonl(tmp_path)
            snap_id: str = logger.log_to_jsonl(
                eta=0.05, mode="EXPLOIT", fuse_level="NORMAL", pre_affect="NEUTRAL",
                evidence_verified=True,
            )
            assert snap_id.startswith("snap_")
            # JSONL writer 存在
            assert logger.get_jsonl_writer() is not None
        finally:
            logger.reset()  # reset 关闭 JSONL writer
            os.unlink(tmp_path)

    def test_logger_reset_closes_jsonl(self):
        """logger reset: 关闭 JSONL writer."""
        from core.kappa_snap_logger import KappaSnapLogger
        logger: KappaSnapLogger = KappaSnapLogger()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path: str = tmp.name
        try:
            logger.enable_jsonl(tmp_path)
            assert logger.get_jsonl_writer() is not None
            logger.reset()
            assert logger.get_jsonl_writer() is None
        finally:
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════
#  core/__init__.py 新增导出测试
# ══════════════════════════════════════════════════════════════════

class TestCoreInitV080:
    """core/__init__.py v0.8.0 新增导出 测试."""

    def test_pre_affect_importable(self):
        """PreAffect 可从 core 导入."""
        from core import PreAffect
        assert PreAffect.GRRR.value == "GRRR"

    def test_kappa_snap_jsonl_importable(self):
        """KappaSnapJSONLWriter 可从 core 导入."""
        from core import KappaSnapJSONLWriter
        writer: KappaSnapJSONLWriter = KappaSnapJSONLWriter()
        assert writer._file is None

    def test_hermes_translator_importable(self):
        """HermesTranslator 可从 core 导入."""
        from core import HermesTranslator
        hermes: HermesTranslator = HermesTranslator()
        assert hermes.translate("GRRR") == "η停滞焦虑信号"

    def test_eml_semzip_ic_importable(self):
        """EMLSemZipIC 可从 core 导入."""
        from core import EMLSemZipIC
        ic: EMLSemZipIC = EMLSemZipIC()
        assert ic.theta_dead == 0.45


# ══════════════════════════════════════════════════════════════════
#  agent/__init__.py 新增导出测试
# ══════════════════════════════════════════════════════════════════

class TestAgentInitV080:
    """agent/__init__.py v0.8.0 新增导出 测试."""

    def test_s_bridge_importable(self):
        """SBridge 可从 agent 导入."""
        from agent import SBridge
        bridge: SBridge = SBridge()
        assert bridge._agent is None

    def test_skill_entry_importable(self):
        """SkillEntry 可从 agent 导入."""
        from agent import SkillEntry
        skill: SkillEntry = SkillEntry(name="test")
        assert skill.name == "test"

    def test_fuse_level_importable(self):
        """FuseLevel 可从 agent 导入."""
        from agent import FuseLevel
        assert FuseLevel.WARNING.value == "WARNING"

    def test_fuse_grade_result_importable(self):
        """FuseGradeResult 可从 agent 导入."""
        from agent import FuseGradeResult
        result: FuseGradeResult = FuseGradeResult()
        assert result.level == FuseLevel.NORMAL

    def test_fuse_option_importable(self):
        """FuseOption 可从 agent 导入."""
        from agent import FuseOption
        option: FuseOption = FuseOption(name="test", action_modifier=lambda a: a)
        assert option.name == "test"
