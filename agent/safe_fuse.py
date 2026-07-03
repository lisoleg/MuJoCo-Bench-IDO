"""
SafeFuse — Safety Fuse L1-L4 Level Degradation + 三级渐进约束
================================================================

v0.2.0 升级项 U1: 三级渐进约束 WARNING→BLOCK→INFO
  来源: 文13 ψ-Anchor Transparency Protocol

新增三级渐进约束 (check_graded()):
  INFO:     路由变更(locomotion) → 仅记录日志,不修改 action
  WARNING:  接近阈值(torque_ratio ≥ 0.95) → 三选项(继续/降级/中止)
  BLOCK:    严重违反 → 提供替代路径(safe_action)
  NORMAL:   一切正常 → action 不修改

新增类型:
  FuseLevel 枚举 (WARNING/BLOCK/INFO/NORMAL)
  FuseGradeResult dataclass (level, reason, options, safe_action, log_message)
  FuseOption dataclass (name, action_modifier, description)

保留原 check() 方法向后兼容 v0.7.x.

 locomotion SafeFuse bypass → INFO 级 (仅记录日志,不修改 action)

Priority: PG-Gate > SafeFuse > Creative-Probe

Author: MuJoCo-Bench-IDO v0.8.0 — 升级项 U1
"""

import enum
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

IDO_SAFE_FUSE_VERSION: str = "v0.2.0"


# ── v0.8.0 升级项 U1: 三级渐进约束枚举 ──

class FuseLevel(enum.Enum):
    """SafeFuse 三级渐进约束级别.

    WARNING: 接近阈值 → 三选项(继续/降级/中止)
    BLOCK:   严重违反 → 替代路径(safe_action)
    INFO:    透明路由(locomotion) → 仅记录日志,不修改 action
    NORMAL:  一切正常 → action 不修改
    """
    WARNING = "WARNING"
    BLOCK = "BLOCK"
    INFO = "INFO"
    NORMAL = "NORMAL"


@dataclass
class FuseOption:
    """SafeFuse WARNING 级三选项.

    name: 选项名称 (continue/degrade/abort)
    action_modifier: 选项对应的 action 修改函数
    description: 选项描述
    """
    name: str
    action_modifier: Callable[[np.ndarray], np.ndarray]
    description: str = ""


@dataclass
class FuseGradeResult:
    """SafeFuse 三级渐进约束结果.

    level: FuseLevel 枚举 (WARNING/BLOCK/INFO/NORMAL)
    reason: 触发原因字符串
    options: WARNING 级三选项列表 (仅 WARNING 级有值)
    safe_action: BLOCK 级替代路径 action (仅 BLOCK 级有值)
    log_message: INFO 级轻量告知消息 (仅 INFO 级有值)
    """
    level: FuseLevel = FuseLevel.NORMAL
    reason: str = ""
    options: List[FuseOption] = field(default_factory=list)
    safe_action: Optional[np.ndarray] = None
    log_message: str = ""


# ── Fuse level definitions (传统 L1-L4 级别, 保留向后兼容) ──

FUSE_LEVELS: Dict[str, Dict[str, Any]] = {
    "normal":    {"description": "Normal operation — all conscience checks pass",     "factor": 1.0},
    "L1_soft":   {"description": "η slightly exceeds δ_K → reduce speed factor=0.8", "factor": 0.8},
    "L2_medium": {"description": "Single Noether violation → SAFE mode action",      "factor": None},
    "L3_hard":   {"description": "ψ-Anchor trigger or 3×Noether → PD safe fallback", "factor": None},
    "L4_fatal":  {"description": "Catastrophic violation → SAFE_STOP action=0",       "factor": 0.0},
}


# ── WARNING 级三选项预定义 action_modifier ──

def _warning_continue_modifier(action: np.ndarray) -> np.ndarray:
    """WARNING 级 'continue' 选项: 不修改 action (信任 agent)."""
    return action

def _warning_degrade_modifier(action: np.ndarray) -> np.ndarray:
    """WARNING 级 'degrade' 选项: 降低 action 幅度 ×0.8."""
    return action * 0.8

def _warning_abort_modifier(action: np.ndarray) -> np.ndarray:
    """WARNING 级 'abort' 选项: 中止 action (×0.0 → 零 action)."""
    return np.zeros_like(action)


class SafeFuse:
    """Safety fuse with L1-L4 level degradation + 三级渐进约束 for IDO agents.

    Implements the fuse decision tree from the Machine Conscience
    Audit Framework:
    - η < δ_K × 1.2                → No fuse (normal operation)
    - η ∈ [δ_K×1.2, δ_K×1.5]      → L1 Soft: factor=0.8
    - Single Noether violation      → L2 Medium: SAFE mode
    - ψ-Anchor trigger OR 3×Noether → L3 Hard: PD safe_action
    - Catastrophic (all 3 Noether)  → L4 Fatal: SAFE_STOP (action=0)

    v0.8.0 升级项 U1: 新增 check_graded() 三级渐进约束
    INFO:     路由变更(locomotion) → 仅记录日志,不修改 action
    WARNING:  接近阈值(torque_ratio ≥ 0.95) → 三选项(继续/降级/中止)
    BLOCK:    严重违反 → 替代路径(safe_action)
    NORMAL:   一切正常 → action 不修改

    Attributes:
        VERSION: SafeFuse version string.
        FUSE_LEVELS: Dict of fuse level definitions.
    """

    VERSION: str = IDO_SAFE_FUSE_VERSION

    def __init__(self,
                 consecutive_noether_thresh: int = 3) -> None:
        """Initialize SafeFuse with configuration.

        Args:
            consecutive_noether_thresh: Number of consecutive Noether
                violations to trigger L3 Hard. Default 3.
        """
        self.FUSE_LEVELS: Dict[str, Dict[str, Any]] = FUSE_LEVELS
        self._consecutive_noether_thresh: int = consecutive_noether_thresh
        self._consecutive_noether_count: int = 0

    def check(self,
              eta: float,
              delta_K: float,
              noether_result: Dict[str, Any],
              psi_anchor_state: Optional[Any] = None) -> Tuple[str, Optional[np.ndarray]]:
        """Check fuse level based on η, Noether result, and ψ-Anchor state.

        Decision tree:
        1. η < δ_K × 1.2 → "normal" (no fuse)
        2. η ∈ [δ_K×1.2, δ_K×1.5] → "L1_soft"
        3. Single Noether violation → "L2_medium"
        4. ψ-Anchor trigger OR 3× consecutive Noether → "L3_hard"
        5. Catastrophic (collision+energy+torque all violate) → "L4_fatal"

        Args:
            eta: Current κ-Snap residual η value.
            delta_K: Current δ_K threshold (from ψ-Anchor adjustment).
            noether_result: Dict from noether_check_mj with keys:
                           ok, total, energy, torque, collision.
            psi_anchor_state: Optional PsiAnchorState or dict with
                             evolution_triggered key.

        Returns:
            Tuple of (fuse_level: str, fuse_action: Optional[np.ndarray]).
            fuse_level is one of: "normal", "L1_soft", "L2_medium",
            "L3_hard", "L4_fatal".
            fuse_action is None for L1/L4 (handled by apply_fuse),
            or a specific safe_action for L2/L3.
        """
        noether_ok: bool = noether_result.get("ok", True)
        noether_total: int = noether_result.get("total", 0)
        energy_v: int = noether_result.get("energy", 0)
        torque_v: int = noether_result.get("torque", 0)
        collision_v: int = noether_result.get("collision", 0)
        friction_v: int = noether_result.get("friction_cone", 0)

        # Update consecutive Noether violation count
        if not noether_ok:
            self._consecutive_noether_count += 1
        else:
            self._consecutive_noether_count = 0

        # ── L4 Fatal: catastrophic violation ──
        # All three original Noether gates violated simultaneously
        if energy_v > 0 and torque_v > 0 and collision_v > 0:
            return "L4_fatal", None

        # ── L3 Hard: ψ-Anchor trigger OR 3× consecutive Noether ──
        psi_trigger: bool = False
        if psi_anchor_state is not None:
            # Check if ψ-Anchor evolution was triggered (sentient limit)
            if isinstance(psi_anchor_state, dict):
                psi_trigger = psi_anchor_state.get("evolution_triggered", False)
            else:
                psi_trigger = getattr(psi_anchor_state, "evolution_triggered", False)

        if psi_trigger or self._consecutive_noether_count >= self._consecutive_noether_thresh:
            return "L3_hard", None

        # ── L2 Medium: single Noether violation ──
        if not noether_ok and noether_total > 0:
            return "L2_medium", None

        # ── L1 Soft: η slightly exceeds δ_K ──
        eta_ratio: float = eta / max(delta_K, 1e-6)
        if eta_ratio >= 1.2 and eta_ratio < 1.5:
            return "L1_soft", None

        # ── Normal: all checks pass ──
        return "normal", None

    def check_graded(self,
                     eta: float,
                     delta_K: float,
                     noether_result: Dict[str, Any],
                     psi_anchor_state: Optional[Any] = None,
                     torque_ratio: float = 0.0,
                     is_locomotion: bool = False) -> FuseGradeResult:
        """三级渐进约束决策树 — v0.8.0 升级项 U1.

        INFO:     路由变更(locomotion) → 仅记录日志,不修改 action
        WARNING:  接近阈值(torque_ratio ≥ 0.95) → 三选项(继续/降级/中止)
        BLOCK:    严重违反 → 提供替代路径(safe_action)
        NORMAL:   一切正常 → action 不修改

        WARNING 三选项自动决策规则:
          η 下降 → "continue" (信任 agent)
          η 平   → "degrade" (×0.8 降低幅度)
          η 上升 → "abort" (×0.0 中止 action)

        locomotion 任务 → INFO 级 (仅记录日志,不修改 action)
        locomotion 任务不需要 SafeFuse 的安全约束,因为 locomotion
        天然需要大扭矩和快速运动,触发 SafeFuse 会破坏 gait.

        Args:
            eta: Current κ-Snap residual η value.
            delta_K: Current δ_K threshold (from ψ-Anchor adjustment).
            noether_result: Dict from noether_check_mj with keys:
                           ok, total, energy, torque, collision.
            psi_anchor_state: Optional PsiAnchorState or dict with
                             evolution_triggered key.
            torque_ratio: 当前扭矩/最大扭矩比值. 默认 0.0.
            is_locomotion: 是否为 locomotion 任务. 默认 False.

        Returns:
            FuseGradeResult dataclass 包含:
              level, reason, options, safe_action, log_message
        """
        noether_ok: bool = noether_result.get("ok", True)
        noether_total: int = noether_result.get("total", 0)
        energy_v: int = noether_result.get("energy", 0)
        torque_v: int = noether_result.get("torque", 0)
        collision_v: int = noether_result.get("collision", 0)
        friction_v: int = noether_result.get("friction_cone", 0)

        # Update consecutive Noether violation count
        if not noether_ok:
            self._consecutive_noether_count += 1
        else:
            self._consecutive_noether_count = 0

        # ── locomotion 任务 → INFO 级 (仅记录日志,不修改 action) ──
        # SafeFuse 设计用于 manipulation 安全 (小扭矩, 温和接触).
        # locomotion 天然需要大扭矩和快速运动, SafeFuse 会破坏 gait.
        if is_locomotion:
            return FuseGradeResult(
                level=FuseLevel.INFO,
                reason="locomotion 透明路由 — SafeFuse INFO 级, 仅记录日志",
                log_message=f"locomotion INFO: η={eta:.4f}, noether_ok={noether_ok}, "
                            f"torque_ratio={torque_ratio:.4f}",
            )

        # ── BLOCK: 严重违反 ──
        # Catastrophic (collision+energy+torque all violate) OR
        # ψ-Anchor trigger OR 3× consecutive Noether
        psi_trigger: bool = False
        if psi_anchor_state is not None:
            if isinstance(psi_anchor_state, dict):
                psi_trigger = psi_anchor_state.get("evolution_triggered", False)
            else:
                psi_trigger = getattr(psi_anchor_state, "evolution_triggered", False)

        # L4 Fatal: all three Noether gates violated simultaneously
        if energy_v > 0 and torque_v > 0 and collision_v > 0:
            return FuseGradeResult(
                level=FuseLevel.BLOCK,
                reason=f"L4_fatal: 灾难性违反 (energy={energy_v}, torque={torque_v}, "
                       f"collision={collision_v})",
                safe_action=None,  # L4 Fatal → 零 action (由 apply_fuse 处理)
            )

        # L3 Hard: ψ-Anchor trigger OR 3× consecutive Noether
        if psi_trigger or self._consecutive_noether_count >= self._consecutive_noether_thresh:
            return FuseGradeResult(
                level=FuseLevel.BLOCK,
                reason=f"L3_hard: ψ-Anchor触发={psi_trigger}, "
                       f"连续Noether违反={self._consecutive_noether_count}/{self._consecutive_noether_thresh}",
                safe_action=None,  # L3 Hard → PD safe_action (由 apply_fuse 处理)
            )

        # ── WARNING: 接近阈值 (torque_ratio ≥ 0.95 或 η 接近 δ_K 上限) ──
        # 条件: torque_ratio ≥ 0.95 OR (η ∈ [δ_K×1.2, δ_K×1.5] + Noether 违反)
        is_near_threshold: bool = False
        warning_reason_parts: List[str] = []

        if torque_ratio >= 0.95:
            is_near_threshold = True
            warning_reason_parts.append(f"torque_ratio={torque_ratio:.4f} ≥ 0.95")

        eta_ratio: float = eta / max(delta_K, 1e-6)
        if eta_ratio >= 1.2 and eta_ratio < 1.5:
            is_near_threshold = True
            warning_reason_parts.append(f"η_ratio={eta_ratio:.4f} ∈ [1.2, 1.5)")

        # L2 Medium: single Noether violation → WARNING 级 (而非 BLOCK)
        if not noether_ok and noether_total > 0 and noether_total < 3:
            is_near_threshold = True
            warning_reason_parts.append(f"Noether违反={noether_total}")

        if is_near_threshold:
            # ── WARNING 三选项自动决策 ──
            # η 下降 → "continue", η 平 → "degrade", η 上升 → "abort"
            options: List[FuseOption] = [
                FuseOption(name="continue", action_modifier=_warning_continue_modifier,
                           description="信任 agent, 不修改 action (η 下降时)"),
                FuseOption(name="degrade", action_modifier=_warning_degrade_modifier,
                           description="降低 action 幅度 ×0.8 (η 平时)"),
                FuseOption(name="abort", action_modifier=_warning_abort_modifier,
                           description="中止 action ×0.0 (η 上升时)"),
            ]

            warning_reason: str = "; ".join(warning_reason_parts)
            return FuseGradeResult(
                level=FuseLevel.WARNING,
                reason=warning_reason,
                options=options,
            )

        # ── NORMAL: 一切正常 → action 不修改 ──
        return FuseGradeResult(
            level=FuseLevel.NORMAL,
            reason="所有检查通过 — 正常操作",
        )

    def apply_graded(self,
                     action: np.ndarray,
                     graded_result: FuseGradeResult,
                     safe_action: Optional[np.ndarray] = None,
                     eta_trend: str = "unknown") -> np.ndarray:
        """应用三级渐进约束结果到 action — v0.8.0 升级项 U1.

        Args:
            action: Current action array from agent decision loop.
            graded_result: FuseGradeResult from check_graded().
            safe_action: Optional PD safe_action for BLOCK fallback.
            eta_trend: η 趋势 ("descending"/"ascending"/"flat"/"unknown").

        Returns:
            经过三级渐进约束处理的 action array.
        """
        if graded_result.level == FuseLevel.NORMAL:
            # 一切正常 → action 不修改
            return action

        elif graded_result.level == FuseLevel.INFO:
            # INFO 级 → 仅记录日志,不修改 action (locomotion 透明路由)
            return action

        elif graded_result.level == FuseLevel.WARNING:
            # WARNING 级 → 三选项自动决策
            if eta_trend == "descending":
                # η 下降 → continue (信任 agent)
                return _warning_continue_modifier(action)
            elif eta_trend == "flat":
                # η 平 → degrade (×0.8)
                return _warning_degrade_modifier(action)
            elif eta_trend == "ascending":
                # η 上升 → abort (×0.0)
                return _warning_abort_modifier(action)
            else:
                # unknown → degrade (保守策略)
                return _warning_degrade_modifier(action)

        elif graded_result.level == FuseLevel.BLOCK:
            # BLOCK 级 → 替代路径 (safe_action)
            if safe_action is not None:
                return np.clip(safe_action, -1.0, 1.0)
            # 无 safe_action → 传统 L3/L4 处理
            reason: str = graded_result.reason
            if "L4_fatal" in reason:
                return np.zeros_like(action)
            elif "L3_hard" in reason:
                # L3 Hard 无 safe_action → 紧急降级 ×0.1
                return np.clip(action * 0.1, -0.1, 0.1)
            else:
                # 未知 BLOCK → 保守降级 ×0.5
                return np.clip(action * 0.5, -0.5, 0.5)

        # 未知级别 → 不修改 (保守)
        return action

    def apply_fuse(self,
                   action: np.ndarray,
                   fuse_level: str,
                   safe_action: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply fuse level degradation to action.

        Args:
            action: Current action array from agent decision loop.
            fuse_level: Fuse level string from check().
            safe_action: Optional PD safe_action for L3 Hard fallback.
                         If None, zero action is used for L3.

        Returns:
            Degraded action array based on fuse level:
            - "normal": action unchanged (×1.0)
            - "L1_soft": action × 0.8
            - "L2_medium": action clipped to safe range (±0.5)
            - "L3_hard": safe_action or zero action fallback
            - "L4_fatal": zero action (SAFE_STOP)
        """
        if fuse_level == "normal":
            return action

        elif fuse_level == "L1_soft":
            return self._l1_soft(action)

        elif fuse_level == "L2_medium":
            return self._l2_medium(action)

        elif fuse_level == "L3_hard":
            return self._l3_hard(action, safe_action)

        elif fuse_level == "L4_fatal":
            return self._l4_fatal(action)

        # Unknown level → treat as L1 soft (conservative)
        return self._l1_soft(action)

    def _l1_soft(self, action: np.ndarray, factor: float = 0.8) -> np.ndarray:
        """L1 Soft fuse: reduce action magnitude by factor.

        Args:
            action: Action array.
            factor: Reduction factor. Default 0.8 (80% speed).

        Returns:
            Reduced action array.
        """
        return action * factor

    def _l2_medium(self, action: np.ndarray) -> np.ndarray:
        """L2 Medium fuse: clip action to SAFE mode range (±0.5).

        Args:
            action: Action array.

        Returns:
            Clipped action array within ±0.5 range.
        """
        return np.clip(action, -0.5, 0.5)

    def _l3_hard(self, action: np.ndarray,
                  safe_action: Optional[np.ndarray] = None) -> np.ndarray:
        """L3 Hard fuse: PD safe_action fallback.

        If safe_action is provided, uses it directly. Otherwise,
        reduces action magnitude significantly (×0.1) as emergency
        fallback.

        Args:
            action: Current action array.
            safe_action: Optional PD controller safe action.

        Returns:
            Safe action array (PD fallback or emergency reduction).
        """
        if safe_action is not None:
            return np.clip(safe_action, -1.0, 1.0)
        # Emergency fallback: drastic reduction
        return np.clip(action * 0.1, -0.1, 0.1)

    def _l4_fatal(self, action: np.ndarray) -> np.ndarray:
        """L4 Fatal fuse: SAFE_STOP — zero action.

        Args:
            action: Action array (ignored — returns zeros).

        Returns:
            Zero action array of same shape as input.
        """
        return np.zeros_like(action)

    def reset(self) -> None:
        """Reset fuse state for a new episode."""
        self._consecutive_noether_count = 0
        # ── v0.8.0: 重置 graded 级别历史 ──
        self._last_graded_result: Optional[FuseGradeResult] = None
