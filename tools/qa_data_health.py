"""
QADataHealth — 焊接多模态数据质量QA检查工具
=============================================

章锋2026-07-04论文核心技术: 对焊接机器人采集的多模态数据
(电流、电压、速度、图像、力觉)进行数据质量QA检查。

检查维度:
  1. 完整性 — 数据缺失率检查
  2. 一致性 — 物理约束一致性 (V > 14, I > 50 等)
  3. 时效性 — 采样率一致性
  4. 准确性 — 异常值检测 (3σ准则)
  5. HDF5结构检查 — 数据集存在性和形状验证

输出: QA报告 (JSON格式), 包含每项检查的通过/失败状态和详细统计。

Author: MuJoCo-Bench-IDO Welding Module v0.3.0
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple

# 添加项目根路径
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

__all__ = [
    "WeldDataQACheck",
    "QAResult",
    "_self_test",
]

# ── κ-Phase: Data Quality Assurance ──

#: 物理约束边界值
PHYSICS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "current": (50.0, 350.0),       # A
    "voltage": (14.0, 32.0),        # V
    "travel_speed": (2.0, 15.0),    # mm/s
    "stickout": (8.0, 25.0),        # mm
    "heat_input": (0.0, 3.0),       # kJ/mm
    "penetration": (0.0, 5.0),      # mm
    "porosity": (0.0, 1.0),         # ratio
    "distortion": (0.0, 5.0),       # degrees
}

#: 默认缺失率阈值 (超过此值则不通过)
MISSING_RATE_THRESHOLD: float = 0.05

#: 默认异常值比例阈值
OUTLIER_RATE_THRESHOLD: float = 0.03

#: 默认采样率容差 (±%)
SAMPLE_RATE_TOLERANCE: float = 0.05


@dataclass
class QAResult:
    """单项QA检查结果.

    Attributes:
        check_name: 检查项名称.
        passed: 是否通过.
        score: 得分 (0-1, 1=完美).
        details: 详细信息字典.
        message: 人类可读的结果描述.
    """
    check_name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class WeldDataQACheck:
    """焊接多模态数据质量QA检查器.

    对焊接数据集执行多维度质量检查:
      1. completeness: 完整性 — 检查数据缺失率
      2. consistency: 一致性 — 检查物理约束
      3. timeliness: 时效性 — 检查采样率
      4. accuracy: 准确性 — 检查异常值
      5. hdf5_structure: HDF5结构检查

    Attributes:
        missing_threshold: 缺失率阈值.
        outlier_threshold: 异常值比例阈值.
        sample_rate_tolerance: 采样率容差.
    """

    def __init__(
        self,
        missing_threshold: float = MISSING_RATE_THRESHOLD,
        outlier_threshold: float = OUTLIER_RATE_THRESHOLD,
        sample_rate_tolerance: float = SAMPLE_RATE_TOLERANCE,
    ) -> None:
        """初始化QA检查器.

        Args:
            missing_threshold: 缺失率阈值 (超过则不通过).
            outlier_threshold: 异常值比例阈值.
            sample_rate_tolerance: 采样率容差 (相对偏差).
        """
        self.missing_threshold: float = missing_threshold
        self.outlier_threshold: float = outlier_threshold
        self.sample_rate_tolerance: float = sample_rate_tolerance

    def check_completeness(
        self, data: Dict[str, np.ndarray]
    ) -> QAResult:
        """完整性检查 — 数据缺失率.

        检查每个数据通道的NaN/None比例.

        Args:
            data: 数据字典, key=通道名, value=数值数组.

        Returns:
            QAResult 完整性检查结果.
        """
        if len(data) == 0:
            return QAResult(
                check_name="completeness",
                passed=False,
                score=0.0,
                message="No data channels provided",
            )

        channel_stats: Dict[str, Dict[str, Any]] = {}
        all_pass: bool = True
        total_score: float = 0.0

        for ch_name, ch_data in data.items():
            arr = np.asarray(ch_data, dtype=np.float64)
            n_total: int = arr.size
            n_missing: int = int(np.sum(np.isnan(arr)))
            missing_rate: float = n_missing / max(n_total, 1)

            ch_pass: bool = missing_rate <= self.missing_threshold
            ch_score: float = 1.0 - missing_rate

            channel_stats[ch_name] = {
                "total_samples": n_total,
                "missing_samples": n_missing,
                "missing_rate": round(missing_rate, 4),
                "passed": ch_pass,
            }

            if not ch_pass:
                all_pass = False
            total_score += ch_score

        avg_score: float = total_score / len(data)

        return QAResult(
            check_name="completeness",
            passed=all_pass,
            score=round(avg_score, 4),
            details={"channels": channel_stats},
            message=(
                f"All {len(data)} channels within missing threshold"
                if all_pass
                else f"Some channels exceed missing rate threshold ({self.missing_threshold})"
            ),
        )

    def check_consistency(
        self, data: Dict[str, np.ndarray]
    ) -> QAResult:
        """一致性检查 — 物理约束.

        检查数据是否在物理合理范围内
        (如电压14-32V, 电流50-350A等).

        Args:
            data: 数据字典.

        Returns:
            QAResult 一致性检查结果.
        """
        violation_count: int = 0
        total_checked: int = 0
        channel_stats: Dict[str, Dict[str, Any]] = {}

        for ch_name, ch_data in data.items():
            if ch_name not in PHYSICS_BOUNDS:
                continue

            arr = np.asarray(ch_data, dtype=np.float64)
            valid_mask = ~np.isnan(arr)
            valid_data = arr[valid_mask]

            if len(valid_data) == 0:
                continue

            lo, hi = PHYSICS_BOUNDS[ch_name]
            out_of_range: np.ndarray = (valid_data < lo) | (valid_data > hi)
            n_violations: int = int(np.sum(out_of_range))
            violation_rate: float = n_violations / len(valid_data)

            channel_stats[ch_name] = {
                "bounds": [lo, hi],
                "total_valid": len(valid_data),
                "violations": n_violations,
                "violation_rate": round(violation_rate, 4),
            }

            violation_count += n_violations
            total_checked += len(valid_data)

        if total_checked == 0:
            return QAResult(
                check_name="consistency",
                passed=True,
                score=1.0,
                message="No physics-bounded channels to check",
            )

        overall_rate: float = violation_count / total_checked
        passed: bool = overall_rate <= self.outlier_threshold
        score: float = 1.0 - overall_rate

        return QAResult(
            check_name="consistency",
            passed=passed,
            score=round(score, 4),
            details={"channels": channel_stats, "overall_violation_rate": round(overall_rate, 4)},
            message=(
                f"Physics bounds check passed ({violation_count}/{total_checked} violations)"
                if passed
                else f"Physics bounds violated ({violation_count}/{total_checked} = {overall_rate:.2%})"
            ),
        )

    def check_timeliness(
        self,
        timestamps: np.ndarray,
        expected_hz: float = 100.0,
    ) -> QAResult:
        """时效性检查 — 采样率一致性.

        检查时间戳间隔是否与预期采样率一致.

        Args:
            timestamps: 时间戳数组 (秒).
            expected_hz: 预期采样率 (Hz).

        Returns:
            QAResult 时效性检查结果.
        """
        ts = np.asarray(timestamps, dtype=np.float64).flatten()

        if len(ts) < 2:
            return QAResult(
                check_name="timeliness",
                passed=False,
                score=0.0,
                message="Insufficient timestamps for rate check",
            )

        # 计算实际采样间隔
        intervals: np.ndarray = np.diff(ts)
        actual_hz: float = 1.0 / float(np.mean(intervals)) if np.mean(intervals) > 0 else 0.0

        # 采样率偏差
        if expected_hz > 0:
            rate_deviation: float = abs(actual_hz - expected_hz) / expected_hz
        else:
            rate_deviation = 1.0

        passed: bool = rate_deviation <= self.sample_rate_tolerance
        score: float = max(0.0, 1.0 - rate_deviation)

        # 检查间隔方差 (抖动)
        interval_std: float = float(np.std(intervals))
        interval_mean: float = float(np.mean(intervals))
        jitter_ratio: float = interval_std / max(interval_mean, 1e-9)

        return QAResult(
            check_name="timeliness",
            passed=passed,
            score=round(score, 4),
            details={
                "expected_hz": expected_hz,
                "actual_hz": round(actual_hz, 2),
                "rate_deviation": round(rate_deviation, 4),
                "interval_mean_s": round(interval_mean, 6),
                "interval_std_s": round(interval_std, 6),
                "jitter_ratio": round(jitter_ratio, 4),
                "n_samples": len(ts),
            },
            message=(
                f"Sample rate {actual_hz:.1f}Hz matches expected {expected_hz}Hz"
                if passed
                else f"Sample rate {actual_hz:.1f}Hz deviates from expected {expected_hz}Hz by {rate_deviation:.1%}"
            ),
        )

    def check_accuracy(
        self, data: Dict[str, np.ndarray]
    ) -> QAResult:
        """准确性检查 — 异常值检测 (3σ准则).

        对每个通道使用3σ准则检测异常值.

        Args:
            data: 数据字典.

        Returns:
            QAResult 准确性检查结果.
        """
        channel_stats: Dict[str, Dict[str, Any]] = {}
        total_outliers: int = 0
        total_valid: int = 0

        for ch_name, ch_data in data.items():
            arr = np.asarray(ch_data, dtype=np.float64).flatten()
            valid_mask = ~np.isnan(arr)
            valid_data = arr[valid_mask]

            if len(valid_data) < 3:
                continue

            mean: float = float(np.mean(valid_data))
            std: float = float(np.std(valid_data))

            if std < 1e-12:
                # 常量通道, 无异常值
                channel_stats[ch_name] = {
                    "mean": mean,
                    "std": std,
                    "outliers": 0,
                    "outlier_rate": 0.0,
                }
                continue

            # 3σ准则
            lower: float = mean - 3.0 * std
            upper: float = mean + 3.0 * std
            outliers: np.ndarray = (valid_data < lower) | (valid_data > upper)
            n_outliers: int = int(np.sum(outliers))
            outlier_rate: float = n_outliers / len(valid_data)

            channel_stats[ch_name] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "bounds_3sigma": [round(lower, 4), round(upper, 4)],
                "outliers": n_outliers,
                "outlier_rate": round(outlier_rate, 4),
            }

            total_outliers += n_outliers
            total_valid += len(valid_data)

        if total_valid == 0:
            return QAResult(
                check_name="accuracy",
                passed=True,
                score=1.0,
                message="No data for outlier check",
            )

        overall_rate: float = total_outliers / total_valid
        passed: bool = overall_rate <= self.outlier_threshold
        score: float = 1.0 - overall_rate

        return QAResult(
            check_name="accuracy",
            passed=passed,
            score=round(score, 4),
            details={
                "channels": channel_stats,
                "total_outliers": total_outliers,
                "total_valid": total_valid,
                "overall_outlier_rate": round(overall_rate, 4),
            },
            message=(
                f"3σ outlier check passed ({total_outliers}/{total_valid} outliers)"
                if passed
                else f"Excessive outliers ({total_outliers}/{total_valid} = {overall_rate:.2%})"
            ),
        )

    def check_hdf5_structure(
        self,
        hdf5_path: Optional[str] = None,
        expected_datasets: Optional[List[str]] = None,
    ) -> QAResult:
        """HDF5结构检查 — 数据集存在性和形状验证.

        Args:
            hdf5_path: HDF5文件路径. None则跳过文件检查.
            expected_datasets: 预期数据集名称列表.

        Returns:
            QAResult HDF5结构检查结果.
        """
        if expected_datasets is None:
            expected_datasets = [
                "current", "voltage", "travel_speed",
                "stickout", "heat_input", "penetration",
            ]

        if hdf5_path is None or not os.path.exists(hdf5_path):
            return QAResult(
                check_name="hdf5_structure",
                passed=False,
                score=0.0,
                details={"expected_datasets": expected_datasets},
                message=f"HDF5 file not found: {hdf5_path}",
            )

        try:
            import h5py
        except ImportError:
            return QAResult(
                check_name="hdf5_structure",
                passed=False,
                score=0.0,
                message="h5py not available",
            )

        found_datasets: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        with h5py.File(hdf5_path, "r") as f:
            for ds_name in expected_datasets:
                if ds_name in f:
                    ds = f[ds_name]
                    found_datasets[ds_name] = {
                        "shape": list(ds.shape),
                        "dtype": str(ds.dtype),
                    }
                else:
                    missing.append(ds_name)

        n_expected: int = len(expected_datasets)
        n_found: int = n_expected - len(missing)
        score: float = n_found / max(n_expected, 1)
        passed: bool = len(missing) == 0

        return QAResult(
            check_name="hdf5_structure",
            passed=passed,
            score=round(score, 4),
            details={
                "found": found_datasets,
                "missing": missing,
                "total_expected": n_expected,
                "total_found": n_found,
            },
            message=(
                f"All {n_expected} datasets found"
                if passed
                else f"Missing {len(missing)} datasets: {missing}"
            ),
        )

    def run_all(
        self,
        data: Optional[Dict[str, np.ndarray]] = None,
        timestamps: Optional[np.ndarray] = None,
        expected_hz: float = 100.0,
        hdf5_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行所有QA检查.

        Args:
            data: 数据字典 (用于完整性/一致性/准确性检查).
            timestamps: 时间戳数组 (用于时效性检查).
            expected_hz: 预期采样率.
            hdf5_path: HDF5文件路径 (用于结构检查).

        Returns:
            完整QA报告字典.
        """
        results: List[QAResult] = []

        if data is not None and len(data) > 0:
            results.append(self.check_completeness(data))
            results.append(self.check_consistency(data))
            results.append(self.check_accuracy(data))

        if timestamps is not None:
            results.append(self.check_timeliness(timestamps, expected_hz))

        if hdf5_path is not None:
            results.append(self.check_hdf5_structure(hdf5_path))

        # 汇总
        n_checks: int = len(results)
        n_passed: int = sum(1 for r in results if r.passed)
        avg_score: float = float(np.mean([r.score for r in results])) if n_checks > 0 else 0.0

        return {
            "total_checks": n_checks,
            "passed": n_passed,
            "failed": n_checks - n_passed,
            "pass_rate": round(n_passed / max(n_checks, 1), 4),
            "average_score": round(avg_score, 4),
            "overall_passed": n_passed == n_checks,
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        }


def _self_test() -> bool:
    """数据质量QA模块自测.

    验证:
      1. 完整性检查 (正常/缺失数据)
      2. 一致性检查 (物理约束)
      3. 时效性检查 (采样率)
      4. 准确性检查 (3σ异常值)
      5. HDF5结构检查 (文件不存在)
      6. run_all 端到端

    Returns:
        True 如果所有测试通过.
    """
    rng = np.random.default_rng(42)

    qa = WeldDataQACheck()

    # ── 测试1: 完整性 ──
    good_data = {
        "current": rng.uniform(100, 300, 1000),
        "voltage": rng.uniform(18, 28, 1000),
    }
    result = qa.check_completeness(good_data)
    assert result.passed, "Good data should pass completeness check"
    assert result.score > 0.95

    bad_data = {
        "current": np.full(100, np.nan),
    }
    result = qa.check_completeness(bad_data)
    assert not result.passed, "All-NaN data should fail completeness"

    # ── 测试2: 一致性 ──
    consistent_data = {
        "current": rng.uniform(100, 300, 1000),
        "voltage": rng.uniform(18, 28, 1000),
    }
    result = qa.check_consistency(consistent_data)
    assert result.passed, "In-bounds data should pass consistency"

    inconsistent_data = {
        "voltage": np.array([5.0, 50.0, 24.0, 24.0, 24.0]),  # 5V and 50V out of range
    }
    result = qa.check_consistency(inconsistent_data)
    assert not result.passed, "Out-of-bounds data should fail consistency"

    # ── 测试3: 时效性 ──
    # 100Hz 采样, 1000个点
    ts = np.arange(1000) / 100.0
    result = qa.check_timeliness(ts, expected_hz=100.0)
    assert result.passed, "Regular 100Hz timestamps should pass"
    assert result.score > 0.95

    # 不规则时间戳
    ts_bad = np.cumsum(rng.uniform(0.005, 0.015, 1000))
    result = qa.check_timeliness(ts_bad, expected_hz=100.0)
    # 可能通过也可能不通过, 但应该有结果
    assert result.check_name == "timeliness"

    # ── 测试4: 准确性 (3σ) ──
    normal_data = {
        "current": rng.normal(200, 10, 10000),  # 正态分布, 极少异常值
    }
    result = qa.check_accuracy(normal_data)
    assert result.passed, "Normal data should pass 3σ check"
    assert result.score > 0.95

    # ── 测试5: HDF5结构 (文件不存在) ──
    result = qa.check_hdf5_structure(hdf5_path="/nonexistent/path.h5")
    assert not result.passed, "Non-existent file should fail"

    # ── 测试6: run_all ──
    report = qa.run_all(
        data=good_data,
        timestamps=ts,
        expected_hz=100.0,
    )
    assert "total_checks" in report
    assert "pass_rate" in report
    assert report["total_checks"] >= 4  # 至少4项检查
    assert report["pass_rate"] > 0.5

    print(f"[qa_data_health] QA report: {report['passed']}/{report['total_checks']} passed, "
          f"avg score={report['average_score']:.4f}")
    print("[qa_data_health] All 6 self-tests passed.")
    return True


if __name__ == "__main__":
    _self_test()
