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
import json
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

    # ── v0.16.25 P0: LLM-based Natural Language Attribution ──

    def ask_why_llm(
        self,
        snap_id: Optional[str] = None,
        kappa_tokens: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ) -> str:
        """v0.16.25 P0: LLM-powered natural language causal attribution.

        Translates the κ-Snap audit trail into a natural language causal
        explanation using few-shot prompting. If an LLM client is provided,
        performs real LLM inference; otherwise, falls back to a rule-based
        template that generates a structured explanation from the audit data.

        This implements the S-Bridge → LLM attribution pipeline from the
        IDO/TOMAS nine-layer architecture:
          - S-Layer (κ-Snap) provides the causal audit trail
          - L1 (LLM/Brain) performs natural language causal analysis
          - S-Bridge bridges the gap between machine audit and human-readable explanation

        Args:
            snap_id: Optional snap_id to query. If provided, retrieves that
                    specific event from the JSONL/Merkle chain for analysis.
            kappa_tokens: Optional κ-Snap token string from KappaSnapTokenizer.
                         If provided, used as context for the LLM prompt.
            llm_client: Optional LLM client with a .complete(prompt) → str interface.
                       If None, uses rule-based template fallback.

        Returns:
            Natural language causal explanation string. Example:
            "The agent chose EXPLOIT mode because η=0.03 (converging), Noether
            gates passed, and the previous SAFE action at step 42 reduced η by
            0.15. The κ-Snap audit trail shows no violations in the last 16
            steps, indicating the agent is in a stable exploitation phase."
        """
        # Gather context from audit trail
        audit_data = self._gather_attribution_context(snap_id)

        if llm_client is not None:
            return self._llm_attribution(audit_data, kappa_tokens, llm_client)
        else:
            return self._template_attribution(audit_data, kappa_tokens)

    def _gather_attribution_context(self, snap_id: Optional[str] = None) -> Dict[str, Any]:
        """Gather context from timeline and audit trail for attribution.

        Args:
            snap_id: Optional specific snap_id to focus on.

        Returns:
            Context dict with keys:
            - 'recent_steps': List of recent timeline events
            - 'snap_detail': Specific snap event if snap_id provided
            - 'eta_trend': Recent η values and trend direction
            - 'violation_count': Number of violations in recent steps
            - 'mode_distribution': Decision mode distribution
            - 'fuse_distribution': SafeFuse level distribution
        """
        recent_steps = self._timeline[-20:] if self._timeline else []

        # Get specific snap detail if requested
        snap_detail: Dict[str, Any] = {}
        if snap_id:
            snap_detail = self.audit_snap(snap_id)

        # Analyze η trend
        eta_values = [s.get("eta", 0.0) for s in recent_steps if s.get("eta") is not None]
        eta_trend = "unknown"
        eta_trend_desc = "insufficient data"
        if len(eta_values) >= 3:
            recent_avg = sum(eta_values[-5:]) / min(len(eta_values[-5:]), 5)
            older_avg = sum(eta_values[:-5]) / max(len(eta_values[:-5]), 1)
            if recent_avg < older_avg * 0.9:
                eta_trend = "descending"
                eta_trend_desc = f"η is decreasing ({older_avg:.4f} → {recent_avg:.4f}), agent is converging"
            elif recent_avg > older_avg * 1.1:
                eta_trend = "ascending"
                eta_trend_desc = f"η is increasing ({older_avg:.4f} → {recent_avg:.4f}), agent is diverging"
            else:
                eta_trend = "plateau"
                eta_trend_desc = f"η is stable (~{recent_avg:.4f}), agent may be stalled"

        # Count violations
        violation_count = sum(1 for s in recent_steps if not s.get("noether_ok", True))

        # Mode distribution
        mode_counts: Dict[str, int] = {}
        for s in recent_steps:
            mode = s.get("mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Fuse distribution
        fuse_counts: Dict[str, int] = {}
        for s in recent_steps:
            fuse = s.get("fuse_level", "NORMAL")
            fuse_counts[fuse] = fuse_counts.get(fuse, 0) + 1

        return {
            "recent_steps": recent_steps,
            "snap_detail": snap_detail,
            "eta_trend": eta_trend,
            "eta_trend_desc": eta_trend_desc,
            "eta_values": eta_values,
            "violation_count": violation_count,
            "mode_distribution": mode_counts,
            "fuse_distribution": fuse_counts,
            "n_steps": len(recent_steps),
        }

    def _llm_attribution(
        self,
        context: Dict[str, Any],
        kappa_tokens: Optional[str],
        llm_client: Any,
    ) -> str:
        """Perform real LLM-based causal attribution.

        Constructs a few-shot prompt with the audit context and κ-Snap tokens,
        then calls the LLM client to generate a natural language explanation.

        Args:
            context: Attribution context from _gather_attribution_context.
            kappa_tokens: κ-Snap token string for additional context.
            llm_client: LLM client with .complete(prompt) → str interface.

        Returns:
            LLM-generated natural language attribution.
        """
        # Build few-shot prompt
        prompt_parts: List[str] = [
            "You are an IDO/TOMAS S-Bridge attribution analyst.",
            "Your job is to explain WHY the embodied AI agent made its recent decisions,",
            "based on the κ-Snap audit trail (causal step-level snapshots).",
            "",
            "## Few-Shot Examples",
            "",
            "Example 1:",
            "Audit: η=0.03 (descending), mode=EXPLOIT, fuse=NORMAL, violations=0",
            "Explanation: The agent is in stable exploitation mode. η has been",
            "decreasing over the last 20 steps (0.12 → 0.03), indicating convergence",
            "toward the goal. No Noether violations occurred, and SafeFuse is in",
            "NORMAL mode, meaning the action is safe to execute. The agent should",
            "continue exploiting its current policy.",
            "",
            "Example 2:",
            "Audit: η=0.85 (ascending), mode=SAFE, fuse=WARNING, violations=3",
            "Explanation: The agent is struggling. η is increasing (diverging),",
            "and 3 Noether violations occurred in the last 20 steps. SafeFuse",
            "triggered WARNING level, indicating the agent's actions are approaching",
            "safety limits. The agent switched to SAFE mode to prioritize stability",
            "over performance. The S-Bridge recommends exploring alternative strategies",
            "or triggering Creative-Probe to break out of the divergence.",
            "",
            "## Current Audit Context",
            "",
        ]

        # Add context details
        prompt_parts.append(f"η trend: {context['eta_trend_desc']}")
        prompt_parts.append(f"Violations in last {context['n_steps']} steps: {context['violation_count']}")

        # Mode distribution
        mode_dist = context.get("mode_distribution", {})
        if mode_dist:
            mode_str = ", ".join(f"{k}={v}" for k, v in sorted(mode_dist.items(), key=lambda x: -x[1]))
            prompt_parts.append(f"Mode distribution: {mode_str}")

        # Fuse distribution
        fuse_dist = context.get("fuse_distribution", {})
        if fuse_dist:
            fuse_str = ", ".join(f"{k}={v}" for k, v in sorted(fuse_dist.items(), key=lambda x: -x[1]))
            prompt_parts.append(f"Fuse distribution: {fuse_str}")

        # κ-Snap tokens
        if kappa_tokens:
            prompt_parts.append(f"\nκ-Snap audit tokens:\n{kappa_tokens}")

        # Specific snap detail
        snap_detail = context.get("snap_detail", {})
        if snap_detail:
            prompt_parts.append(f"\nFocused snap event:\n{json.dumps(snap_detail, indent=2, default=str)[:500]}")

        prompt_parts.append("\n## Explanation")
        prompt_parts.append("Provide a concise (3-5 sentence) natural language explanation of WHY")
        prompt_parts.append("the agent made its recent decisions, based on the audit context above.")

        prompt = "\n".join(prompt_parts)

        try:
            result = llm_client.complete(prompt)
            return str(result).strip()
        except Exception as e:
            return f"[LLM attribution failed: {e}]\n{self._template_attribution(context, kappa_tokens)}"

    def _template_attribution(
        self,
        context: Dict[str, Any],
        kappa_tokens: Optional[str],
    ) -> str:
        """Rule-based template attribution (fallback when no LLM available).

        Generates a structured natural language explanation from the audit
        context using deterministic rules. Not as fluent as LLM output but
        provides the same causal information.

        Args:
            context: Attribution context from _gather_attribution_context.
            kappa_tokens: κ-Snap token string (for reference).

        Returns:
            Structured natural language attribution string.
        """
        parts: List[str] = []

        # η trend analysis
        parts.append(f"η Trend: {context['eta_trend_desc']}.")

        # Mode analysis
        mode_dist = context.get("mode_distribution", {})
        if mode_dist:
            dominant_mode = max(mode_dist, key=mode_dist.get)
            dominant_pct = mode_dist[dominant_mode] / max(context["n_steps"], 1) * 100
            parts.append(
                f"Decision Mode: The agent primarily used {dominant_mode} mode "
                f"({dominant_pct:.0f}% of last {context['n_steps']} steps)."
            )

        # Violation analysis
        violation_count = context["violation_count"]
        if violation_count == 0:
            parts.append("Safety: No Noether conservation violations in the recent audit window — the agent's actions are physically consistent.")
        elif violation_count <= 2:
            parts.append(f"Safety: {violation_count} Noether violation(s) detected — the agent occasionally pushed physical limits but recovered.")
        else:
            parts.append(f"Safety: {violation_count} Noether violations — the agent is frequently violating conservation constraints, suggesting instability.")

        # Fuse analysis
        fuse_dist = context.get("fuse_distribution", {})
        if "WARNING" in fuse_dist or "BLOCK" in fuse_dist:
            parts.append(f"SafeFuse: Triggered WARNING/BLOCK {fuse_dist.get('WARNING', 0) + fuse_dist.get('BLOCK', 0)} time(s), indicating the agent's actions approached safety limits.")

        # Causal chain
        eta_trend = context["eta_trend"]
        if eta_trend == "descending" and violation_count == 0:
            parts.append("Causal Chain: η convergence + no violations → the agent's current exploitation policy is effective and should continue.")
        elif eta_trend == "descending" and violation_count > 0:
            parts.append("Causal Chain: η is converging despite violations → the agent is making progress but needs tighter safety compliance.")
        elif eta_trend == "ascending":
            parts.append("Causal Chain: η divergence → the current policy is not working. Recommend Creative-Probe or strategy switch.")
        elif eta_trend == "plateau":
            parts.append("Causal Chain: η plateau → the agent may be stuck. Consider relaxing δ_K or triggering evolution via ψ-Anchor.")

        # κ-Snap token reference
        if kappa_tokens:
            parts.append(f"κ-Snap Tokens: {kappa_tokens[:120]}{'...' if len(kappa_tokens) > 120 else ''}")

        return " ".join(parts)
