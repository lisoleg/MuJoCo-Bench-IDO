"""
HeteroBenchmark — 异构计算基准 (GPU vs GPU+T-Processor)
=========================================================

章锋2026-07-04论文核心技术: 对比纯GPU方案与GPU+T-Processor异构方案
在焊接机器人多模态数据处理中的能耗、延迟和事故成本。

基准维度:
  1. 能耗 (J): GPU独占 vs GPU+T-Processor协同
  2. 延迟 (ms): η计算 + Ψ检查 + κ-Snap记录
  3. 事故成本 ($): 因Ψ-Check延迟导致的安全事故预期成本

T-Processor优势:
  - η-ALU: 10ns/calc (vs GPU ~100μs including kernel launch)
  - Ψ-Check: 50ns (vs GPU ~500μs)
  - 能耗: 3.3mW (vs GPU ~250W)

CLI: python -m tools.hetero_benchmark [--steps N] [--json]

Author: MuJoCo-Bench-IDO Welding Module v0.3.0
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

# 添加项目根路径
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

__all__ = [
    "SimConfig",
    "BenchmarkResult",
    "simulate_bare_gpu",
    "simulate_gpu_tproc",
    "summarize",
    "main",
    "_self_test",
]

# ── κ-Phase: Heterogeneous Compute Benchmark ──

#: GPU功耗 (W) — NVIDIA RTX 3060典型值
GPU_POWER_W: float = 170.0

#: T-Processor功耗 (mW)
TPROC_POWER_MW: float = 3.3

#: GPU η计算延迟 (μs, 含kernel launch)
GPU_ETA_LATENCY_US: float = 100.0

#: T-Processor η计算延迟 (ns)
TPROC_ETA_LATENCY_NS: float = 10.0

#: GPU Ψ-Check延迟 (μs)
GPU_PSI_LATENCY_US: float = 500.0

#: T-Processor Ψ-Check延迟 (ns)
TPROC_PSI_LATENCY_NS: float = 50.0

#: GPU κ-Snap记录延迟 (μs)
GPU_SNAP_LATENCY_US: float = 50.0

#: T-Processor κ-Snap记录延迟 (ns)
TPROC_SNAP_LATENCY_NS: float = 20.0

#: 安全事故预期成本 ($/event)
ACCIDENT_COST_PER_EVENT: float = 50000.0

#: GPU方案Ψ-Check跳过率 (因延迟导致来不及检查)
GPU_PSI_SKIP_RATE: float = 0.02

#: T-Processor方案Ψ-Check跳过率 (几乎为零)
TPROC_PSI_SKIP_RATE: float = 0.0001


@dataclass
class SimConfig:
    """仿真配置参数.

    Attributes:
        n_steps: 仿真步数 (控制循环迭代次数).
        obs_dim: 观测向量维度.
        goal_dim: 目标向量维度.
        action_dim: 动作向量维度.
        control_hz: 控制循环频率 (Hz).
        gpu_power_w: GPU功耗 (W).
        tproc_power_mw: T-Processor功耗 (mW).
        accident_cost: 单次安全事故成本 ($).
        gpu_psi_skip_rate: GPU方案Ψ-Check跳过率.
        tproc_psi_skip_rate: T-Processor方案Ψ-Check跳过率.
    """
    n_steps: int = 10000
    obs_dim: int = 24
    goal_dim: int = 24
    action_dim: int = 12
    control_hz: float = 100.0
    gpu_power_w: float = GPU_POWER_W
    tproc_power_mw: float = TPROC_POWER_MW
    accident_cost: float = ACCIDENT_COST_PER_EVENT
    gpu_psi_skip_rate: float = GPU_PSI_SKIP_RATE
    tproc_psi_skip_rate: float = TPROC_PSI_SKIP_RATE


@dataclass
class BenchmarkResult:
    """基准测试结果.

    Attributes:
        config: 仿真配置.
        total_energy_j: 总能耗 (J).
        total_latency_ms: 总延迟 (ms).
        avg_latency_us: 平均每步延迟 (μs).
        psi_skip_count: Ψ-Check跳过次数.
        accident_count: 预期安全事故次数.
        accident_cost: 预期事故成本 ($).
        throughput: 吞吐量 (steps/s).
        backend: 后端名称 ("bare_gpu" 或 "gpu_tproc").
    """
    config: SimConfig
    total_energy_j: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_us: float = 0.0
    psi_skip_count: int = 0
    accident_count: float = 0.0
    accident_cost: float = 0.0
    throughput: float = 0.0
    backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于JSON序列化).

        Returns:
            结果字典.
        """
        d = asdict(self)
        return d


def simulate_bare_gpu(config: SimConfig) -> BenchmarkResult:
    """仿真纯GPU方案.

    纯GPU方案中, 所有计算 (η, Ψ-Check, κ-Snap) 都在GPU上执行:
      - η计算: ~100μs/step (含kernel launch开销)
      - Ψ-Check: ~500μs/step
      - κ-Snap: ~50μs/step
      - 总延迟: ~650μs/step → 超过10ms控制周期不是问题,
        但在高频场景下Ψ-Check可能被跳过

    能耗: GPU功耗 × 总时间
    事故: Ψ-Check跳过率 × 步数 → 预期事故次数

    Args:
        config: 仿真配置.

    Returns:
        BenchmarkResult 纯GPU方案结果.
    """
    n: int = config.n_steps

    # 每步延迟 (μs)
    per_step_latency_us: float = (
        GPU_ETA_LATENCY_US + GPU_PSI_LATENCY_US + GPU_SNAP_LATENCY_US
    )
    total_latency_us: float = per_step_latency_us * n
    total_latency_ms: float = total_latency_us / 1000.0
    avg_latency_us: float = per_step_latency_us

    # 总时间 (s)
    total_time_s: float = total_latency_us / 1e6

    # 能耗 (J) = 功率(W) × 时间(s)
    total_energy_j: float = config.gpu_power_w * total_time_s

    # Ψ-Check跳过 (GPU延迟导致来不及检查)
    psi_skip_count: int = int(np.random.binomial(n, config.gpu_psi_skip_rate))

    # 预期事故: 跳过Ψ-Check时有一定概率发生事故
    accident_probability: float = 0.001  # 每次跳过0.1%概率出事故
    accident_count: float = psi_skip_count * accident_probability
    accident_cost: float = accident_count * config.accident_cost

    # 吞吐量
    throughput: float = n / max(total_time_s, 1e-9)

    return BenchmarkResult(
        config=config,
        total_energy_j=total_energy_j,
        total_latency_ms=total_latency_ms,
        avg_latency_us=avg_latency_us,
        psi_skip_count=psi_skip_count,
        accident_count=accident_count,
        accident_cost=accident_cost,
        throughput=throughput,
        backend="bare_gpu",
    )


def simulate_gpu_tproc(config: SimConfig) -> BenchmarkResult:
    """仿真GPU+T-Processor异构方案.

    异构方案中, T-Processor负责实时安全关键计算:
      - η计算: ~10ns/step (T-Processor η-ALU)
      - Ψ-Check: ~50ns/step (T-Processor Ψ-Checker)
      - κ-Snap: ~20ns/step (T-Processor κ-FIFO)
      - GPU仅负责非实时ML推理 (能耗分摊)

    T-Processor在100MHz下运行, 所有安全检查在10ns内完成,
    Ψ-Check跳过率接近零。

    能耗: T-Processor 3.3mW + GPU分摊 (假设GPU负载降低50%)
    事故: 几乎为零

    Args:
        config: 仿真配置.

    Returns:
        BenchmarkResult GPU+T-Processor方案结果.
    """
    n: int = config.n_steps

    # T-Processor每步延迟 (ns → μs)
    per_step_latency_us: float = (
        TPROC_ETA_LATENCY_NS + TPROC_PSI_LATENCY_NS + TPROC_SNAP_LATENCY_NS
    ) / 1000.0  # ns → μs
    total_latency_us: float = per_step_latency_us * n
    total_latency_ms: float = total_latency_us / 1000.0
    avg_latency_us: float = per_step_latency_us

    # 总时间 (s)
    total_time_s: float = total_latency_us / 1e6

    # 能耗: T-Processor + GPU分摊 (GPU负载减半)
    tproc_power_w: float = config.tproc_power_mw / 1000.0
    gpu_energy: float = config.gpu_power_w * 0.5 * total_time_s  # GPU负载减半
    tproc_energy: float = tproc_power_w * total_time_s
    total_energy_j: float = gpu_energy + tproc_energy

    # Ψ-Check跳过 (T-Processor几乎不跳过)
    psi_skip_count: int = int(np.random.binomial(n, config.tproc_psi_skip_rate))

    # 预期事故
    accident_probability: float = 0.001
    accident_count: float = psi_skip_count * accident_probability
    accident_cost: float = accident_count * config.accident_cost

    # 吞吐量
    throughput: float = n / max(total_time_s, 1e-9)

    return BenchmarkResult(
        config=config,
        total_energy_j=total_energy_j,
        total_latency_ms=total_latency_ms,
        avg_latency_us=avg_latency_us,
        psi_skip_count=psi_skip_count,
        accident_count=accident_count,
        accident_cost=accident_cost,
        throughput=throughput,
        backend="gpu_tproc",
    )


def summarize(
    gpu_result: BenchmarkResult,
    tproc_result: BenchmarkResult,
) -> Dict[str, Any]:
    """生成对比摘要.

    Args:
        gpu_result: 纯GPU方案结果.
        tproc_result: GPU+T-Processor方案结果.

    Returns:
        对比摘要字典, 包含各指标的改善率.
    """
    # 能耗改善率
    energy_reduction: float = 0.0
    if gpu_result.total_energy_j > 1e-9:
        energy_reduction = (
            (gpu_result.total_energy_j - tproc_result.total_energy_j)
            / gpu_result.total_energy_j * 100.0
        )

    # 延迟改善率
    latency_reduction: float = 0.0
    if gpu_result.avg_latency_us > 1e-9:
        latency_reduction = (
            (gpu_result.avg_latency_us - tproc_result.avg_latency_us)
            / gpu_result.avg_latency_us * 100.0
        )

    # 事故成本改善率
    cost_reduction: float = 0.0
    if gpu_result.accident_cost > 1e-9:
        cost_reduction = (
            (gpu_result.accident_cost - tproc_result.accident_cost)
            / gpu_result.accident_cost * 100.0
        )

    # 吞吐量提升
    throughput_boost: float = 0.0
    if gpu_result.throughput > 1e-9:
        throughput_boost = (
            (tproc_result.throughput - gpu_result.throughput)
            / gpu_result.throughput * 100.0
        )

    return {
        "energy_reduction_pct": round(energy_reduction, 2),
        "latency_reduction_pct": round(latency_reduction, 2),
        "accident_cost_reduction_pct": round(cost_reduction, 2),
        "throughput_boost_pct": round(throughput_boost, 2),
        "gpu_energy_j": round(gpu_result.total_energy_j, 4),
        "tproc_energy_j": round(tproc_result.total_energy_j, 4),
        "gpu_latency_us": round(gpu_result.avg_latency_us, 4),
        "tproc_latency_us": round(tproc_result.avg_latency_us, 4),
        "gpu_accident_cost": round(gpu_result.accident_cost, 2),
        "tproc_accident_cost": round(tproc_result.accident_cost, 2),
        "gpu_psi_skips": gpu_result.psi_skip_count,
        "tproc_psi_skips": tproc_result.psi_skip_count,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI入口函数.

    用法:
        python -m tools.hetero_benchmark [--steps N] [--json]

    Args:
        argv: 命令行参数 (None则从sys.argv读取).

    Returns:
        0 表示成功.
    """
    parser = argparse.ArgumentParser(
        description="异构计算基准: 纯GPU vs GPU+T-Processor"
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="仿真步数 (默认10000)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="输出JSON格式"
    )
    args = parser.parse_args(argv)

    np.random.seed(42)
    config = SimConfig(n_steps=args.steps)

    gpu_result = simulate_bare_gpu(config)
    tproc_result = simulate_gpu_tproc(config)
    summary = summarize(gpu_result, tproc_result)

    if args.json:
        output = {
            "config": asdict(config),
            "gpu_result": gpu_result.to_dict(),
            "tproc_result": tproc_result.to_dict(),
            "summary": summary,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print("=" * 60)
        print("异构计算基准: 纯GPU vs GPU+T-Processor")
        print("=" * 60)
        print(f"仿真步数: {config.n_steps}")
        print()
        print(f"{'指标':<25} {'纯GPU':>15} {'GPU+T-Proc':>15} {'改善率':>10}")
        print("-" * 65)
        print(f"{'总能耗 (J)':<25} {gpu_result.total_energy_j:>15.4f} "
              f"{tproc_result.total_energy_j:>15.4f} "
              f"{summary['energy_reduction_pct']:>9.1f}%")
        print(f"{'平均延迟 (μs)':<25} {gpu_result.avg_latency_us:>15.4f} "
              f"{tproc_result.avg_latency_us:>15.4f} "
              f"{summary['latency_reduction_pct']:>9.1f}%")
        print(f"{'Ψ-Check跳过次数':<25} {gpu_result.psi_skip_count:>15d} "
              f"{tproc_result.psi_skip_count:>15d}")
        print(f"{'预期事故成本 ($)':<25} {gpu_result.accident_cost:>15.2f} "
              f"{tproc_result.accident_cost:>15.2f} "
              f"{summary['accident_cost_reduction_pct']:>9.1f}%")
        print(f"{'吞吐量 (steps/s)':<25} {gpu_result.throughput:>15.1f} "
              f"{tproc_result.throughput:>15.1f} "
              f"{summary['throughput_boost_pct']:>9.1f}%")
        print("=" * 60)

    return 0


def _self_test() -> bool:
    """异构基准模块自测.

    验证:
      1. simulate_bare_gpu 产生合理结果
      2. simulate_gpu_tproc 产生合理结果
      3. T-Processor方案能耗低于纯GPU
      4. T-Processor方案延迟低于纯GPU
      5. summarize 计算正确
      6. CLI main 函数正常运行

    Returns:
        True 如果所有测试通过.
    """
    np.random.seed(42)
    config = SimConfig(n_steps=1000)

    # ── 测试1: 纯GPU仿真 ──
    gpu_result = simulate_bare_gpu(config)
    assert gpu_result.backend == "bare_gpu"
    assert gpu_result.total_energy_j > 0.0, "GPU energy should be positive"
    assert gpu_result.avg_latency_us > 0.0, "GPU latency should be positive"
    assert gpu_result.psi_skip_count >= 0

    # ── 测试2: GPU+T-Processor仿真 ──
    tproc_result = simulate_gpu_tproc(config)
    assert tproc_result.backend == "gpu_tproc"
    assert tproc_result.total_energy_j > 0.0
    assert tproc_result.avg_latency_us > 0.0

    # ── 测试3: T-Processor能耗更低 ──
    assert tproc_result.total_energy_j < gpu_result.total_energy_j, \
        f"T-Proc energy ({tproc_result.total_energy_j}) should be < GPU ({gpu_result.total_energy_j})"

    # ── 测试4: T-Processor延迟更低 ──
    assert tproc_result.avg_latency_us < gpu_result.avg_latency_us, \
        f"T-Proc latency ({tproc_result.avg_latency_us}) should be < GPU ({gpu_result.avg_latency_us})"

    # ── 测试5: summarize ──
    summary = summarize(gpu_result, tproc_result)
    assert summary["energy_reduction_pct"] > 0.0, "Energy reduction should be positive"
    assert summary["latency_reduction_pct"] > 0.0, "Latency reduction should be positive"
    assert "gpu_energy_j" in summary
    assert "tproc_energy_j" in summary

    # ── 测试6: CLI main ──
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        ret = main(["--steps", "100", "--json"])
    assert ret == 0, "CLI should return 0"
    output = f.getvalue()
    assert "summary" in output, "JSON output should contain summary"

    print("[hetero_benchmark] All 6 self-tests passed.")
    return True


if __name__ == "__main__":
    main()
