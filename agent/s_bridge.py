"""
S-Bridge — MetaQuery 自我归因接口 (可选插件)
=============================================

v0.8.0 升级项 U5: S-Bridge MetaQuery
  来源: 文12 S-Layer interface specification

四接口 MetaQuery:
  WHY_THIS_ACTION: 返回决策理由 — mode + fuse_level + pre_affect + η
  AUDIT_SNAP: 查询 κ-Snap JSONL 日志 — snap_id → 步骤审计记录
  LEARN_SKILL: 将成功经验提炼为 skill — 模式 Dict (mode 分布 + fuse 统计)
  JOURNEY_TIMELINE: 返回时间线事件列表 — since_timestamp 以来的事件

SkillEntry dataclass:
  name: skill 名称
  task: 任务名称
  avg_eta: 平均 η 值
  avg_cq: 平均 CQ 值
  ic: IC 值
  pattern: 决策模式 Dict
  created_at: 创建时间戳

可选插件约定:
  - s_bridge=None 默认不启用 (不侵入主循环)
  - 仅在 choose_action 末尾 _record_step()
  - 外部调用四接口查询/归因

Author: MuJoCo-Bench-IDO v0.8.0 — 升级项 U5
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.kappa_snap_jsonl import KappaSnapJSONLWriter


@dataclass
class SkillEntry:
    """S-Bridge skill 条目 — 成功经验提炼.

    Attributes:
        name: skill 名称.
        task: 任务名称.
        avg_eta: 平均 η 值.
        avg_cq: 平均 CQ 值.
        ic: IC (Information Cardinality) 值.
        pattern: 决策模式 Dict (mode 分布 + fuse 统计 + η 范围).
        created_at: 创建时间戳.
    """
    name: str = ""
    task: str = ""
    avg_eta: float = 0.0
    avg_cq: float = 0.0
    ic: float = 0.0
    pattern: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


class SBridge:
    """S-Bridge MetaQuery 自我归因接口 — 可选插件 v0.8.0.

    提供 4 种 MetaQuery 接口, 让 agent 可以自我归因、
    审计查询、提炼 skill、查看时间线.

    S-Bridge 作为可选插件, 不侵入 choose_action 主循环.
    仅在 choose_action 末尾调用 _record_step() 记录步骤信息.

    Attributes:
        _agent: HybridSB3IDOAgent 引用 (用于访问 agent 状态).
        _jsonl: KappaSnapJSONLWriter 引用 (用于审计查询).
        _skill_bank: Dict[str, SkillEntry] skill 存储.
        _timeline: List[Dict] 时间线事件列表.
    """

    def __init__(self,
                 agent: Optional[Any] = None,
                 jsonl: Optional[KappaSnapJSONLWriter] = None) -> None:
        """初始化 S-Bridge.

        Args:
            agent: HybridSB3IDOAgent 引用 (可选, 用于访问 agent 状态).
            jsonl: KappaSnapJSONLWriter 引用 (可选, 用于审计查询).
        """
        self._agent: Optional[Any] = agent
        self._jsonl: Optional[KappaSnapJSONLWriter] = jsonl
        self._skill_bank: Dict[str, SkillEntry] = {}
        self._timeline: List[Dict[str, Any]] = []

    # ── WHY_THIS_ACTION: 返回决策理由 ──

    def why_this_action(self,
                        state: Optional[Dict] = None,
                        action: Optional[Any] = None,
                        eta: Optional[float] = None) -> str:
        """WHY_THIS_ACTION: 返回决策理由字符串.

        格式: "mode={mode}, fuse={fuse_level}, affect={pre_affect}, η={eta:.4f}"

        Args:
            state: 当前状态 Dict (可选).
            action: 当前 action (可选).
            eta: 当前 η 值 (可选, 若 None 则从 agent 获取).

        Returns:
            决策理由字符串.
        """
        # 从 agent 获取最近决策信息
        mode_str: str = "unknown"
        fuse_level_str: str = "NORMAL"
        pre_affect_str: str = "NEUTRAL"
        eta_val: float = eta if eta is not None else 0.0

        if self._agent is not None:
            try:
                mode_str = self._agent._mode.value if hasattr(self._agent, '_mode') else "unknown"
                # eta_val: 优先使用传入参数, 其次从 agent 获取
                if eta is None:
                    eta_val = self._agent._last_eta if hasattr(self._agent, '_last_eta') and self._agent._last_eta is not None else 0.0
                if hasattr(self._agent, '_last_fuse_grade') and self._agent._last_fuse_grade is not None:
                    fuse_level_str = self._agent._last_fuse_grade.level.value
                if hasattr(self._agent, '_last_pre_affect') and self._agent._last_pre_affect is not None:
                    pre_affect_str = self._agent._last_pre_affect.value
            except Exception:
                pass

        return f"mode={mode_str}, fuse={fuse_level_str}, affect={pre_affect_str}, η={eta_val:.4f}"

    # ── AUDIT_SNAP: 查询 κ-Snap JSONL 日志 ──

    def audit_snap(self, snap_id: str) -> Dict[str, Any]:
        """AUDIT_SNAP: 查询 κ-Snap JSONL 日志.

        查询特定 snap_id 的步骤审计记录.

        Args:
            snap_id: 要查询的 snap_id 字符串.

        Returns:
            匹配的审计记录 Dict. 若未找到, 返回空 Dict.
        """
        if self._jsonl is not None:
            return self._jsonl.query(snap_id)
        # 若 JSONL 未启用, 从 KappaSnapLogger 的 merkle chain 查找
        if self._agent is not None and hasattr(self._agent, '_logger'):
            merkle_chain: List[Dict] = self._agent._logger.get_merkle_chain()
            for entry in merkle_chain:
                if entry.get("snap_id") == snap_id:
                    return dict(entry)
        return {}

    # ── LEARN_SKILL: 将成功经验提炼为 skill ──

    def learn_skill(self,
                    episodes: List[Dict[str, Any]],
                    skill_name: str = "",
                    task_name: str = "") -> SkillEntry:
        """LEARN_SKILL: 将成功经验提炼为 skill.

        从成功 episodes 中提炼决策模式:
          - mode 分布 (EXPLOIT/EXPLORE/SAFE 比例)
          - fuse 统计 (WARNING/BLOCK/INFO/NORMAL 比例)
          - η 范围 (min/max/avg)

        Args:
            episodes: 成功 episode 数据列表.
            skill_name: skill 名称. 若空, 自动生成.
            task_name: 任务名称. 若空, 从 episodes 获取.

        Returns:
            SkillEntry dataclass.
        """
        if len(episodes) == 0:
            return SkillEntry(
                name=skill_name or "empty_skill",
                task=task_name,
                created_at=time.time(),
            )

        # 自动生成 skill 名称
        if not skill_name:
            skill_name = f"skill_{task_name or 'unknown'}_{int(time.time())}"

        # 从 episodes 提取统计信息
        eta_values: List[float] = []
        cq_values: List[float] = []
        mode_counts: Dict[str, int] = {"EXPLOIT": 0, "EXPLORE": 0, "SAFE": 0}
        fuse_counts: Dict[str, int] = {"NORMAL": 0, "WARNING": 0, "BLOCK": 0, "INFO": 0}

        for ep in episodes:
            eta_values.append(ep.get("avg_eta", 0.0))
            cq_values.append(ep.get("avg_cq", 0.0))
            # mode 分布
            mc: Dict = ep.get("mode_counts", {})
            for mode_key, count in mc.items():
                mode_counts[mode_key] = mode_counts.get(mode_key, 0) + count
            # fuse 统计
            fc: Dict = ep.get("fuse_counts", {})
            for fuse_key, count in fc.items():
                fuse_counts[fuse_key] = fuse_counts.get(fuse_key, 0) + count

        avg_eta: float = sum(eta_values) / max(len(eta_values), 1)
        avg_cq: float = sum(cq_values) / max(len(cq_values), 1)

        # 决策模式 Dict
        pattern: Dict[str, Any] = {
            "mode_distribution": mode_counts,
            "fuse_statistics": fuse_counts,
            "eta_range": {
                "min": min(eta_values) if eta_values else 0.0,
                "max": max(eta_values) if eta_values else 0.0,
                "avg": avg_eta,
            },
            "n_episodes": len(episodes),
        }

        # IC 值 (从 episodes 获取, 如有)
        ic_value: float = 0.0
        ic_list: List[float] = [ep.get("ic", 0.0) for ep in episodes if "ic" in ep]
        if ic_list:
            ic_value = sum(ic_list) / len(ic_list)

        skill = SkillEntry(
            name=skill_name,
            task=task_name,
            avg_eta=avg_eta,
            avg_cq=avg_cq,
            ic=ic_value,
            pattern=pattern,
            created_at=time.time(),
        )

        # 存储 skill
        self._skill_bank[skill_name] = skill

        return skill

    # ── JOURNEY_TIMELINE: 返回时间线事件列表 ──

    def journey_timeline(self, since: float = 0.0) -> List[Dict[str, Any]]:
        """JOURNEY_TIMELINE: 返回 since 以来的时间线事件列表.

        Args:
            since: 时间戳起点 (Unix timestamp). 默认 0.0 (返回全部).

        Returns:
            时间线事件列表 (每个事件为 Dict 包含 type, timestamp, data).
        """
        timeline_events: List[Dict[str, Any]] = []
        for event in self._timeline:
            event_ts: float = event.get("timestamp", 0.0)
            if event_ts >= since:
                timeline_events.append(dict(event))
        return timeline_events

    # ── 内部方法: _record_step (主循环末尾调用) ──

    def _record_step(self,
                     state: Optional[Dict] = None,
                     action: Optional[Any] = None,
                     eta: Optional[float] = None,
                     mode: str = "",
                     fuse_level: str = "",
                     pre_affect: str = "",
                     noether_result: Optional[Dict] = None,
                     evidence_verified: Optional[bool] = None) -> None:
        """记录一步信息到时间线 — choose_action 末尾调用.

        不侵入主循环, 仅记录信息供 MetaQuery 查询.

        Args:
            state: 当前状态 Dict.
            action: 当前 action.
            eta: 当前 η 值.
            mode: Agent 模式.
            fuse_level: SafeFuse 级别.
            pre_affect: PreAffect 信号.
            noether_result: Noether 检查结果.
            evidence_verified: 证据校验标记.
        """
        event: Dict[str, Any] = {
            "type": "step",
            "timestamp": time.time(),
            "eta": eta,
            "mode": mode,
            "fuse_level": fuse_level,
            "pre_affect": pre_affect,
            "noether_ok": noether_result.get("ok", True) if noether_result else True,
            "evidence_verified": evidence_verified,
        }
        self._timeline.append(event)

    # ── 辅助方法 ──

    def get_skill_bank(self) -> Dict[str, SkillEntry]:
        """返回 skill 存储.

        Returns:
            Dict[str, SkillEntry] skill 存储字典.
        """
        return dict(self._skill_bank)

    def get_timeline_length(self) -> int:
        """返回时间线长度.

        Returns:
            时间线事件数量.
        """
        return len(self._timeline)

    def reset(self) -> None:
        """重置 S-Bridge 状态 (清空时间线, 保留 skill_bank)."""
        self._timeline = []
