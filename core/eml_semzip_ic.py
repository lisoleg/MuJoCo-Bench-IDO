"""
EML-SemZip IC — Information Cardinality 计算 + Dead-Zero 过滤 + 毛睿重加权
===========================================================================

v0.8.0 升级项 U6: EML-SemZip Dead-Zero 数据治理
  来源: 文12 EML-SemZip data governance + Dead-Zero pruning

Information Cardinality (IC):
  IC = Shannon 熵(Δstate_i 归一化分布)
  Δstate_i = state[t+1] - state[t] (归一化后)
  对 Δstate_i 的分布计算 H = -Σ p_k * log2(p_k)

  高 IC = 轨迹有丰富状态变化(有价值, 应保留/过采样)
  低 IC = 轨迹几乎无变化(噪声/dead-zero, 应剔除)

Dead-Zero 过滤:
  θ_dead = 0.45 (Dead-Zero 过滤阈值)
  IC < θ_dead → 剔除 (轨迹无价值)
  IC ≥ θ_dead → 保留

高 IC 过采样:
  Top 5% × 3 过采样 (让高 IC 轨迹在训练/分析中更频繁出现)

毛睿度量重加权:
  采样概率 ∝ IC^power, power=1.0 默认
  (来自 v0.6.4 MaoRuiMetric, 延续其重加权思想)

Author: MuJoCo-Bench-IDO v0.8.0 — 升级项 U6
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── 默认配置常量 ──
THETA_DEAD: float = 0.45             # Dead-Zero 过滤阈值
OVERSAMPLE_TOP_PCT: float = 0.05     # 高 IC 过采样: Top 5%
OVERSAMPLE_FACTOR: int = 3           # 高 IC 过采样因子: ×3
MAO_RUI_POWER: float = 1.0           # 毛睿度量重加权: IC^power


class EMLSemZipIC:
    """EML-SemZip IC 计算 + Dead-Zero 过滤 + 毛睿重加权.

    Information Cardinality (IC) = Shannon 熵(轨迹状态变化分布).
    用于判断轨迹的信息密度, 剔除低价值(dead-zero)轨迹,
    过采样高价值轨迹, 并用毛睿度量重加权采样概率.

    Attributes:
        theta_dead: Dead-Zero 过滤阈值. 默认 0.45.
        oversample_top_pct: 高 IC 过采样百分比. 默认 0.05 (Top 5%).
        oversample_factor: 高 IC 过采样因子. 默认 3 (×3).
        mao_rui_power: 毛睿度量重加权指数. 默认 1.0.
    """

    def __init__(self,
                 theta_dead: float = THETA_DEAD,
                 oversample_top_pct: float = OVERSAMPLE_TOP_PCT,
                 oversample_factor: int = OVERSAMPLE_FACTOR,
                 mao_rui_power: float = MAO_RUI_POWER) -> None:
        """初始化 EMLSemZipIC.

        Args:
            theta_dead: Dead-Zero 过滤阈值.
            oversample_top_pct: 高 IC 过采样百分比 (Top N%).
            oversample_factor: 高 IC 过采样因子.
            mao_rui_power: 毛睿度量重加权指数.
        """
        self.theta_dead: float = theta_dead
        self.oversample_top_pct: float = oversample_top_pct
        self.oversample_factor: int = oversample_factor
        self.mao_rui_power: float = mao_rui_power

    def compute_ic(self, trajectory_states: List[np.ndarray]) -> float:
        """计算轨迹的 Information Cardinality (IC).

        IC = Shannon 熵(Δstate_i 归一化分布)
        Δstate_i = state[t+1] - state[t] (归一化后)
        对 Δstate_i 的分布计算 H = -Σ p_k * log2(p_k)

        Args:
            trajectory_states: 轨迹状态列表, 每个元素为 np.ndarray.
                               长度至少为 2 (才能计算差分).

        Returns:
            IC 值 (float). 轨迹长度不足时返回 0.0.
        """
        if len(trajectory_states) < 2:
            return 0.0

        # ── 计算 Δstate_i ──
        # 将状态列表转为矩阵 (每行一个状态)
        states_matrix: np.ndarray = np.array(trajectory_states)
        # 差分: Δstate_i = state[t+1] - state[t]
        deltas: np.ndarray = states_matrix[1:] - states_matrix[:-1]

        # ── 归一化 Δstate ──
        # 展平为 1D 向量便于熵计算
        delta_flat: np.ndarray = deltas.flatten()

        # 归一化到 [0, 1] 区间 (用于离散化熵计算)
        delta_range: float = float(np.ptp(delta_flat))  # ptp = max - min
        if delta_range < 1e-10:
            # Δstate 几乎为零 → IC = 0.0 (dead-zero)
            return 0.0

        delta_normalized: np.ndarray = (delta_flat - np.min(delta_flat)) / max(delta_range, 1e-10)

        # ── Shannon 熵计算 ──
        # 使用 scipy.stats.entropy (如有), 否则手动计算
        if HAS_SCIPY:
            # 离散化归一化值到 bins, 计算概率分布, 然后计算熵
            n_bins: int = min(50, max(10, len(delta_normalized) // 10))
            hist, _ = np.histogram(delta_normalized, bins=n_bins, range=(0.0, 1.0))
            # 概率分布 (归一化 histogram)
            prob_dist: np.ndarray = hist / max(float(np.sum(hist)), 1.0)
            # scipy.stats.entropy 使用自然对数, base=2 → Shannon 熵 (bits)
            ic_value: float = float(scipy_entropy(prob_dist, base=2))
        else:
            # 手动 Shannon 熵计算: H = -Σ p_k * log2(p_k)
            n_bins: int = min(50, max(10, len(delta_normalized) // 10))
            hist, _ = np.histogram(delta_normalized, bins=n_bins, range=(0.0, 1.0))
            prob_dist: np.ndarray = hist / max(float(np.sum(hist)), 1.0)
            # 去除零概率项 (避免 log2(0))
            nonzero_probs: np.ndarray = prob_dist[prob_dist > 0]
            ic_value: float = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

        return ic_value

    def compute_ic_single(self, episode_trajectory: List[np.ndarray]) -> float:
        """计算单个 episode 轨迹的 IC.

        与 compute_ic 相同, 但更明确地为单个 episode 设计.

        Args:
            episode_trajectory: 单个 episode 的轨迹状态列表.

        Returns:
            IC 值 (float).
        """
        return self.compute_ic(episode_trajectory)

    def is_dead_zero(self, ic: float, theta_dead: Optional[float] = None) -> bool:
        """判断 IC 是否为 Dead-Zero (无价值轨迹).

        IC < θ_dead → Dead-Zero (剔除)
        IC ≥ θ_dead → 有价值 (保留)

        Args:
            ic: IC 值.
            theta_dead: 过滤阈值. 若 None, 使用 self.theta_dead.

        Returns:
            True 表示 Dead-Zero (应剔除), False 表示有价值 (应保留).
        """
        threshold: float = theta_dead if theta_dead is not None else self.theta_dead
        return ic < threshold

    def filter_episodes(self,
                        episodes_data: List[Dict[str, Any]],
                        trajectory_key: str = "trajectory_states") -> List[Dict[str, Any]]:
        """过滤 episodes, 剔除 Dead-Zero (IC < θ_dead).

        Args:
            episodes_data: Episode 数据列表, 每个元素为 Dict.
                           必须包含 trajectory_key 对应的轨迹状态列表.
            trajectory_key: Dict 中轨迹状态列表的键名. 默认 "trajectory_states".

        Returns:
            过滤后的 episode 数据列表 (IC ≥ θ_dead).
        """
        filtered: List[Dict[str, Any]] = []
        for ep_data in episodes_data:
            trajectory: List[np.ndarray] = ep_data.get(trajectory_key, [])
            if len(trajectory) < 2:
                # 轨迹长度不足 → 视为 Dead-Zero
                continue
            ic_value: float = self.compute_ic(trajectory)
            # 将 IC 值写入 episode 数据
            ep_data["ic"] = ic_value
            if not self.is_dead_zero(ic_value):
                filtered.append(ep_data)
        return filtered

    def oversample_high_ic(self,
                           filtered_episodes: List[Dict[str, Any]],
                           top_pct: Optional[float] = None,
                           factor: Optional[int] = None) -> List[Dict[str, Any]]:
        """高 IC 过采样: Top N% × factor 过采样.

        让高 IC (高信息密度) 轨迹在训练/分析中更频繁出现.

        Args:
            filtered_episodes: 已过滤的 episode 数据列表 (IC ≥ θ_dead).
            top_pct: 过采样百分比. 若 None, 使用 self.oversample_top_pct.
            factor: 过采样因子. 若 None, 使用 self.oversample_factor.

        Returns:
            过采样后的 episode 数据列表 (原列表 + Top N% × factor 份副本).
        """
        if len(filtered_episodes) == 0:
            return []

        pct: float = top_pct if top_pct is not None else self.oversample_top_pct
        fac: int = factor if factor is not None else self.oversample_factor

        # 按 IC 值排序
        sorted_episodes: List[Dict[str, Any]] = sorted(
            filtered_episodes, key=lambda ep: ep.get("ic", 0.0), reverse=True
        )

        # 取 Top N%
        n_top: int = max(1, int(len(sorted_episodes) * pct))
        top_episodes: List[Dict[str, Any]] = sorted_episodes[:n_top]

        # 过采样: 每个 top episode 复制 factor 份
        oversampled: List[Dict[str, Any]] = list(filtered_episodes)
        for ep in top_episodes:
            for _ in range(fac):
                # 创建副本 (标记为过采样)
                copy_ep: Dict[str, Any] = dict(ep)
                copy_ep["_oversampled"] = True
                oversampled.append(copy_ep)

        return oversampled

    def reweight_mao_rui(self,
                         ic_values: List[float],
                         power: Optional[float] = None) -> np.ndarray:
        """毛睿度量重加权: 采样概率 ∝ IC^power.

        来自 v0.6.4 MaoRuiMetric 的重加权思想, 用 IC 值
        作为权重基础, 使得高 IC 轨迹有更高采样概率.

        Args:
            ic_values: IC 值列表.
            power: 重加权指数. 若 None, 使用 self.mao_rui_power.

        Returns:
            归一化权重数组 (numpy.ndarray), 每个元素对应一个 IC 值的采样概率.
        """
        p: float = power if power is not None else self.mao_rui_power
        ic_arr: np.ndarray = np.array(ic_values, dtype=float)

        # 防止零值 (加小常数)
        ic_shifted: np.ndarray = ic_arr + 1e-6

        # 采样概率 ∝ IC^power
        weights: np.ndarray = np.power(ic_shifted, p)

        # 归一化 (确保 Σ weights = 1.0)
        total: float = float(np.sum(weights))
        if total < 1e-10:
            # 全部接近零 → 均匀权重
            weights = np.ones_like(ic_arr) / max(len(ic_arr), 1)
        else:
            weights = weights / total

        return weights
