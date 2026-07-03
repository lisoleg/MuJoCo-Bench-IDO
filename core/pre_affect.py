"""
PreAffect — η 窗口内在信号检测
================================

v0.8.0 升级项 U4: Pre-Affect 内在信号
  来源: 文13 Pre-Affect as intrinsic reward signal

检测 η 趋势中的内在情绪信号:
  - GRRR: η 连续停滞焦虑 — 加强 Creative-Probe perturbation ×1.5
  - PHEW: η 连续突破释然 — 延伸 EXPLOIT max_stall ×1.5
  - NEUTRAL: 无特殊信号 — 正常操作

检测逻辑:
  GRRR: 连续 window_grrr 步 η 不降 (η[t+1] >= η[t] 或变化 < 1%)
  PHEW: 连续 window_phew 步 η 降 > phew_threshold_pct (10%)
  NEUTRAL: 不满足上述条件

调幅因子:
  probe_multiplier(GRRR) → ×1.5 (加强 Creative-Probe perturbation)
  stall_extension(PHEW) → max_stall ×1.5 (延伸 EXPLOIT 稳定区间)

Author: MuJoCo-Bench-IDO v0.8.0 — 升级项 U4
"""

import enum
from typing import List


class PreAffect(enum.Enum):
    """η 窗口内在信号枚举 — 表示 agent 内在情绪状态.

    GRRR: η 连续停滞焦虑信号 — 需要更强 perturbation 来突破
    PHEW: η 连续突破释然信号 — 可以延长当前策略稳定区间
    NEUTRAL: 无特殊信号 — 正常操作
    """
    GRRR = "GRRR"
    PHEW = "PHEW"
    NEUTRAL = "NEUTRAL"


# ── 调幅因子常量 ──
PROBE_MULTIPLIER_GRRR: float = 1.5  # GRRR → Creative-Probe perturbation ×1.5
STALL_EXTENSION_PHEW: float = 1.5   # PHEW → EXPLOIT max_stall ×1.5
PROBE_MULTIPLIER_NEUTRAL: float = 1.0  # NEUTRAL → 无调幅
STALL_EXTENSION_NEUTRAL: float = 1.0   # NEUTRAL → 无延伸

# ── 默认检测窗口 ──
DEFAULT_WINDOW_GRRR: int = 3         # GRRR 检测窗口: 连续 3 步 η 不降
DEFAULT_WINDOW_PHEW: int = 2         # PHEW 检测窗口: 连续 2 步 η 降 > 10%
DEFAULT_PHEW_THRESHOLD_PCT: float = 0.10  # PHEW 阈值: η 降 > 10%


def detect(eta_history: List[float],
           window_grrr: int = DEFAULT_WINDOW_GRRR,
           window_phew: int = DEFAULT_WINDOW_PHEW,
           phew_threshold_pct: float = DEFAULT_PHEW_THRESHOLD_PCT) -> PreAffect:
    """检测 η 趋势中的内在情绪信号.

    分析 η 历史窗口，判断当前内在情绪状态:
      GRRR: η 连续 window_grrr 步不降 (停滞焦虑)
      PHEW: η 连续 window_phew 步降 > phew_threshold_pct (突破释然)
      NEUTRAL: 不满足上述条件

    Args:
        eta_history: η 历史值列表 (最近步的 η 值, 按时间顺序排列).
        window_grrr: GRRR 检测窗口大小. 默认 3 步.
        window_phew: PHEW 检测窗口大小. 默认 2 步.
        phew_threshold_pct: PHEW 下降阈值百分比. 默认 0.10 (10%).

    Returns:
        PreAffect 枚举值 (GRRR / PHEW / NEUTRAL).
    """
    # 历史不足时返回 NEUTRAL
    if len(eta_history) < max(window_grrr, window_phew):
        return PreAffect.NEUTRAL

    # ── GRRR 检测: η 连续 window_grrr 步不降 ──
    # η[t+1] >= η[t] 或变化 < 1% 视为"不降"
    if len(eta_history) >= window_grrr:
        grrr_window: List[float] = eta_history[-window_grrr:]
        all_not_descending: bool = True
        for i in range(len(grrr_window) - 1):
            # η 不降: 后一步 >= 前一步，或变化幅度 < 1%
            if grrr_window[i + 1] < grrr_window[i] * (1.0 - 0.01):
                all_not_descending = False
                break
        if all_not_descending:
            return PreAffect.GRRR

    # ── PHEW 检测: η 连续 window_phew 步降 > phew_threshold_pct ──
    # η[t+1] < η[t] × (1 - phew_threshold_pct) 视为"显著下降"
    if len(eta_history) >= window_phew:
        phew_window: List[float] = eta_history[-window_phew:]
        all_significant_drop: bool = True
        for i in range(len(phew_window) - 1):
            relative_drop: float = (phew_window[i] - phew_window[i + 1]) / max(abs(phew_window[i]), 1e-6)
            if relative_drop < phew_threshold_pct:
                all_significant_drop = False
                break
        if all_significant_drop:
            return PreAffect.PHEW

    # ── 默认: NEUTRAL ──
    return PreAffect.NEUTRAL


def probe_multiplier(affect: PreAffect) -> float:
    """返回 Creative-Probe perturbation 调幅因子.

    GRRR: ×1.5 (加强 Creative-Probe perturbation, noise_scale/phase_offset/gain_multiplier)
    PHEW: ×1.0 (PHEW 不调整 probe, 而是延伸 EXPLOIT stall)
    NEUTRAL: ×1.0 (无调幅)

    Args:
        affect: PreAffect 枚举值.

    Returns:
        perturbation 调幅因子 (float).
    """
    if affect == PreAffect.GRRR:
        return PROBE_MULTIPLIER_GRRR
    elif affect == PreAffect.PHEW:
        return PROBE_MULTIPLIER_NEUTRAL  # PHEW 不调幅 probe
    else:
        return PROBE_MULTIPLIER_NEUTRAL


def stall_extension(affect: PreAffect) -> float:
    """返回 EXPLOIT max_stall 延伸因子.

    GRRR: ×1.0 (GRRR 不延伸 stall, 而是加强 probe)
    PHEW: ×1.5 (延伸 EXPLOIT max_stall 稳定区间)
    NEUTRAL: ×1.0 (无延伸)

    Args:
        affect: PreAffect 枚举值.

    Returns:
        max_stall 延伸因子 (float).
    """
    if affect == PreAffect.GRRR:
        return STALL_EXTENSION_NEUTRAL  # GRRR 不延伸 stall
    elif affect == PreAffect.PHEW:
        return STALL_EXTENSION_PHEW
    else:
        return STALL_EXTENSION_NEUTRAL
