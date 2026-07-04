"""
HeteroBenchmark 单元测试
=========================

测试异构计算基准模块 (tools/hetero_benchmark.py):
  - simulate_bare_gpu 产生合理结果
  - simulate_gpu_tproc 产生合理结果
  - T-Processor方案能耗低于纯GPU
  - T-Processor方案延迟低于纯GPU
  - summarize 计算正确
  - CLI main 函数正常运行
  - CIM忆阻器模拟器 (tools/tproc_cim_simulator.py)
  - 焊接物理公式 (core/welding_process_proxy.py 新增方法)
  - EML蒸馏网络 (core/welding_eml_distillation.py)
  - κ-Snap统计聚合 (tools/wps_pqr_generator.py 新增方法)

Author: MuJoCo-Bench-IDO Welding Module v0.3.0
"""

import os
import sys
import io
import json
import numpy as np
import pytest
from contextlib import redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.hetero_benchmark import (
    SimConfig,
    BenchmarkResult,
    simulate_bare_gpu,
    simulate_gpu_tproc,
    summarize,
    main,
    GPU_POWER_W,
    TPROC_POWER_MW,
    GPU_ETA_LATENCY_US,
    TPROC_ETA_LATENCY_NS,
)


# ── Fixtures ──

@pytest.fixture
def small_config():
    """小规模仿真配置."""
    return SimConfig(n_steps=1000)


# ── SimConfig 测试 ──

class TestSimConfig:
    """测试仿真配置."""

    def test_default_values(self):
        """默认值合理."""
        config = SimConfig()
        assert config.n_steps == 10000
        assert config.gpu_power_w == 170.0
        assert config.tproc_power_mw == 3.3
        assert config.control_hz == 100.0

    def test_custom_steps(self):
        """自定义步数."""
        config = SimConfig(n_steps=500)
        assert config.n_steps == 500


# ── simulate_bare_gpu 测试 ──

class TestSimulateBareGPU:
    """测试纯GPU仿真."""

    def test_returns_benchmark_result(self, small_config):
        """返回BenchmarkResult."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert isinstance(result, BenchmarkResult)

    def test_backend_name(self, small_config):
        """后端名称为bare_gpu."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert result.backend == "bare_gpu"

    def test_positive_energy(self, small_config):
        """能耗为正."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert result.total_energy_j > 0.0

    def test_positive_latency(self, small_config):
        """延迟为正."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert result.avg_latency_us > 0.0

    def test_expected_latency(self, small_config):
        """平均延迟 = GPU_ETA + GPU_PSI + GPU_SNAP."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        expected = 100.0 + 500.0 + 50.0  # 650 μs
        assert abs(result.avg_latency_us - expected) < 0.01

    def test_psi_skip_non_negative(self, small_config):
        """Ψ-Check跳过次数非负."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert result.psi_skip_count >= 0

    def test_positive_throughput(self, small_config):
        """吞吐量为正."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        assert result.throughput > 0.0

    def test_to_dict(self, small_config):
        """to_dict 可序列化."""
        np.random.seed(42)
        result = simulate_bare_gpu(small_config)
        d = result.to_dict()
        assert "backend" in d
        assert "total_energy_j" in d
        assert d["backend"] == "bare_gpu"


# ── simulate_gpu_tproc 测试 ──

class TestSimulateGpuTProc:
    """测试GPU+T-Processor仿真."""

    def test_returns_benchmark_result(self, small_config):
        """返回BenchmarkResult."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        assert isinstance(result, BenchmarkResult)

    def test_backend_name(self, small_config):
        """后端名称为gpu_tproc."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        assert result.backend == "gpu_tproc"

    def test_positive_energy(self, small_config):
        """能耗为正."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        assert result.total_energy_j > 0.0

    def test_positive_latency(self, small_config):
        """延迟为正."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        assert result.avg_latency_us > 0.0

    def test_expected_latency(self, small_config):
        """平均延迟 = (TPROC_ETA + TPROC_PSI + TPROC_SNAP) / 1000 (ns→μs)."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        expected = (10.0 + 50.0 + 20.0) / 1000.0  # 0.08 μs
        assert abs(result.avg_latency_us - expected) < 0.001

    def test_positive_throughput(self, small_config):
        """吞吐量为正."""
        np.random.seed(42)
        result = simulate_gpu_tproc(small_config)
        assert result.throughput > 0.0


# ── 对比测试 ──

class TestComparison:
    """测试两种方案的对比."""

    def test_tproc_lower_energy(self, small_config):
        """T-Processor方案能耗低于纯GPU."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        assert tproc.total_energy_j < gpu.total_energy_j, \
            f"T-Proc energy ({tproc.total_energy_j}) should be < GPU ({gpu.total_energy_j})"

    def test_tproc_lower_latency(self, small_config):
        """T-Processor方案延迟低于纯GPU."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        assert tproc.avg_latency_us < gpu.avg_latency_us, \
            f"T-Proc latency ({tproc.avg_latency_us}) should be < GPU ({gpu.avg_latency_us})"

    def test_tproc_higher_throughput(self, small_config):
        """T-Processor方案吞吐量高于纯GPU."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        assert tproc.throughput > gpu.throughput

    def test_energy_reduction_significant(self, small_config):
        """能耗降低幅度显著 (>40%)."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        summary = summarize(gpu, tproc)
        assert summary["energy_reduction_pct"] > 40.0


# ── summarize 测试 ──

class TestSummarize:
    """测试对比摘要."""

    def test_contains_all_keys(self, small_config):
        """摘要包含所有必要字段."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        summary = summarize(gpu, tproc)
        expected_keys = {
            "energy_reduction_pct", "latency_reduction_pct",
            "accident_cost_reduction_pct", "throughput_boost_pct",
            "gpu_energy_j", "tproc_energy_j",
            "gpu_latency_us", "tproc_latency_us",
            "gpu_accident_cost", "tproc_accident_cost",
            "gpu_psi_skips", "tproc_psi_skips",
        }
        assert expected_keys.issubset(set(summary.keys())), \
            f"Missing keys: {expected_keys - set(summary.keys())}"

    def test_positive_reductions(self, small_config):
        """改善率为正."""
        np.random.seed(42)
        gpu = simulate_bare_gpu(small_config)
        tproc = simulate_gpu_tproc(small_config)
        summary = summarize(gpu, tproc)
        assert summary["energy_reduction_pct"] > 0.0
        assert summary["latency_reduction_pct"] > 0.0

    def test_zero_division_safe(self):
        """GPU结果为零时不崩溃."""
        zero_config = SimConfig(n_steps=0)
        np.random.seed(42)
        gpu = simulate_bare_gpu(zero_config)
        tproc = simulate_gpu_tproc(zero_config)
        summary = summarize(gpu, tproc)
        assert isinstance(summary, dict)


# ── CLI main 测试 ──

class TestCLI:
    """测试CLI入口."""

    def test_table_output(self):
        """表格输出返回0."""
        f = io.StringIO()
        with redirect_stdout(f):
            ret = main(["--steps", "100"])
        assert ret == 0
        output = f.getvalue()
        assert "异构计算基准" in output

    def test_json_output(self):
        """JSON输出包含summary."""
        f = io.StringIO()
        with redirect_stdout(f):
            ret = main(["--steps", "100", "--json"])
        assert ret == 0
        output = f.getvalue()
        data = json.loads(output)
        assert "summary" in data
        assert "gpu_result" in data
        assert "tproc_result" in data

    def test_custom_steps(self):
        """自定义步数."""
        f = io.StringIO()
        with redirect_stdout(f):
            ret = main(["--steps", "50"])
        assert ret == 0
        output = f.getvalue()
        assert "50" in output


# ── 自测集成 ──

class TestSelfTest:
    """测试模块自测函数."""

    def test_hetero_benchmark_self_test(self):
        """hetero_benchmark._self_test() 通过."""
        from tools.hetero_benchmark import _self_test
        assert _self_test() is True


# ── CIM 模拟器测试 ──

class TestCIMSimulator:
    """测试CIM忆阻器交叉阵列模拟器."""

    def test_crossbar_all_off_zero_current(self):
        """全关断阵列输出零电流."""
        from tools.tproc_cim_simulator import CrossbarArray
        cim = CrossbarArray(8)
        vec = np.ones(8) * 0.5
        currents, energy = cim.matrix_vector_mult(vec)
        assert np.allclose(currents, 0, atol=1e-5)

    def test_crossbar_diagonal_current(self):
        """对角线导通产生预期电流."""
        from tools.tproc_cim_simulator import CrossbarArray
        cim = CrossbarArray(8)
        for i in range(8):
            cim.set_weight(i, i, True)
        vec = np.ones(8) * 0.5
        currents, energy = cim.matrix_vector_mult(vec)
        expected = 1e-3 * 0.5 * 0.1  # g_on × vec × v_read
        assert np.allclose(currents, expected, atol=1e-6)

    def test_cim_energy_less_than_sram(self):
        """CIM能耗远小于SRAM+ALU."""
        from tools.tproc_cim_simulator import CrossbarArray, SRAM_ALU_ENERGY_PJ
        cim = CrossbarArray(8)
        for i in range(8):
            cim.set_weight(i, i, True)
        vec = np.ones(8) * 0.5
        _, energy = cim.matrix_vector_mult(vec)
        energy_pj = energy * 1e12
        assert energy_pj < SRAM_ALU_ENERGY_PJ

    def test_run_energy_comparison(self):
        """run_energy_comparison 返回正确结构."""
        from tools.tproc_cim_simulator import run_energy_comparison
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_energy_comparison()
        assert "sram_alu_pj" in result
        assert "cim_pj" in result
        assert "saving_ratio" in result
        assert result["saving_ratio"] > 1.0

    def test_cim_self_test(self):
        """tproc_cim_simulator._self_test() 通过."""
        from tools.tproc_cim_simulator import _self_test
        assert _self_test() is True


# ── 焊接物理公式测试 ──

class TestWeldingPhysicsFormulas:
    """测试章锋论文新增焊接物理公式."""

    def test_compute_target_penetration(self):
        """目标熔深: target_pen = k_I × I² / (v × t)."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        # I=200, v=6, t=2, k_I=0.085
        # target = 0.085 × 40000 / (6 × 2) = 3400 / 12 ≈ 283.33
        result = proxy.compute_target_penetration(I=200.0, v=6.0, t=2.0)
        expected = 0.085 * 200.0 ** 2 / (6.0 * 2.0)
        assert abs(result - expected) < 0.01

    def test_compute_target_penetration_zero_speed(self):
        """速度为零时返回0."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        result = proxy.compute_target_penetration(I=200.0, v=0.0, t=2.0)
        assert result == 0.0

    def test_compute_target_penetration_zero_thickness(self):
        """板厚为零时返回0."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        result = proxy.compute_target_penetration(I=200.0, v=6.0, t=0.0)
        assert result == 0.0

    def test_compute_nominal_voltage_thin(self):
        """薄板(≤3mm)名义电压为16V."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        assert proxy.compute_nominal_voltage(thickness_mm=2.0) == 16.0
        assert proxy.compute_nominal_voltage(thickness_mm=3.0) == 16.0

    def test_compute_nominal_voltage_thick(self):
        """厚板(>3mm)名义电压为18V."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        assert proxy.compute_nominal_voltage(thickness_mm=4.0) == 18.0
        assert proxy.compute_nominal_voltage(thickness_mm=10.0) == 18.0

    def test_evaluate_detailed_returns_tuple(self):
        """evaluate_detailed 返回三元组."""
        from core.welding_process_proxy import WeldingProcessProxy
        proxy = WeldingProcessProxy()
        result = proxy.evaluate_detailed(I=200, V=24, v_mms=6, t_mm=2, stick_out=15)
        assert len(result) == 3
        target, actual, deviation = result
        assert target > 0.0
        assert actual > 0.0
        assert deviation >= 0.0


# ── EML 蒸馏网络测试 ──

class TestEMLDistillation:
    """测试EML八元数蒸馏网络."""

    def test_distiller_forward_shape(self):
        """前向传播输出形状正确."""
        from core.welding_eml_distillation import WeldingEMLDistiller, HAS_TORCH
        distiller = WeldingEMLDistiller(hidden_dim=64)
        if HAS_TORCH:
            import torch
            x = torch.randn(4, 8)
            q, omega, phi_result = distiller(x)
            assert q.shape == (4, 8)
            assert omega.shape == (4, 8)
            assert phi_result.shape == (4, 8)
        else:
            x = np.random.randn(4, 8)
            q, omega, phi_result = distiller(x)
            assert q.shape == (4, 8)
            assert omega.shape == (4, 8)
            assert phi_result.shape == (4, 8)

    def test_distillation_loss_positive(self):
        """蒸馏损失为正."""
        from core.welding_eml_distillation import WeldingEMLDistiller, DistillationLoss, HAS_TORCH
        distiller = WeldingEMLDistiller(hidden_dim=64)
        loss_fn = DistillationLoss()
        if HAS_TORCH:
            import torch
            x = torch.randn(4, 8)
            q, omega, phi_result = distiller(x)
            y_eta = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
            y_d = torch.tensor([[0.5], [-0.3], [0.8], [-0.1]])
            loss = loss_fn(q, omega, phi_result, y_eta, y_d)
            assert loss.item() > 0.0
        else:
            x = np.random.randn(4, 8)
            q, omega, phi_result = distiller(x)
            y_eta = np.array([[1.0], [0.0], [1.0], [0.0]])
            y_d = np.array([[0.5], [-0.3], [0.8], [-0.1]])
            loss = loss_fn(q, omega, phi_result, y_eta, y_d)
            assert loss > 0.0

    def test_generate_eml_candidates(self):
        """从统计生成EML候选节点."""
        from core.welding_eml_distillation import generate_eml_candidates_from_stats
        stats = [
            {"episode": i, "final_I": 120 + i, "final_V": 18,
             "final_eta": 0.1 - i * 0.01, "final_porosity": 0.05,
             "final_distortion": 0.01, "stick_out": 8,
             "steps": 200, "reward": -10 + i}
            for i in range(20)
        ]
        candidates = generate_eml_candidates_from_stats(stats)
        assert len(candidates) == 2  # top 10% of 20 = 2
        assert candidates[0]["q_state"][2] <= candidates[1]["q_state"][2]

    def test_generate_eml_candidates_empty(self):
        """空输入返回空列表."""
        from core.welding_eml_distillation import generate_eml_candidates_from_stats
        assert generate_eml_candidates_from_stats([]) == []

    def test_eml_self_test(self):
        """welding_eml_distillation._self_test() 通过."""
        from core.welding_eml_distillation import _self_test
        assert _self_test() is True


# ── κ-Snap 统计聚合测试 ──

class TestKSnapStats:
    """测试κ-Snap审计统计聚合."""

    def test_aggregate_empty(self):
        """空列表返回零值统计."""
        from tools.wps_pqr_generator import WpsPqrGenerator
        gen = WpsPqrGenerator()
        stats = gen.aggregate_ksnap_stats([])
        assert stats["total_entries"] == 0
        assert stats["eta_mean"] == 0.0
        assert stats["psi_pass_rate"] == 0.0

    def test_aggregate_basic_stats(self):
        """基本统计计算正确."""
        from tools.wps_pqr_generator import WpsPqrGenerator
        gen = WpsPqrGenerator()
        entries = [
            {"eta": 0.1, "psi_passed": True, "violation": "", "step": 0, "timestamp": 0.0},
            {"eta": 0.3, "psi_passed": True, "violation": "", "step": 1, "timestamp": 0.1},
            {"eta": 0.5, "psi_passed": False, "violation": "BURN_BACK", "step": 2, "timestamp": 0.2},
        ]
        stats = gen.aggregate_ksnap_stats(entries)
        assert stats["total_entries"] == 3
        assert abs(stats["eta_mean"] - 0.3) < 0.01
        assert stats["eta_max"] == 0.5
        assert stats["eta_min"] == 0.1
        assert abs(stats["psi_pass_rate"] - 2.0 / 3.0) < 0.01

    def test_aggregate_violation_types(self):
        """违规类型分布正确."""
        from tools.wps_pqr_generator import WpsPqrGenerator
        gen = WpsPqrGenerator()
        entries = [
            {"eta": 0.1, "psi_passed": True, "violation": "", "step": 0, "timestamp": 0.0},
            {"eta": 0.2, "psi_passed": False, "violation": "BURN_BACK", "step": 1, "timestamp": 0.1},
            {"eta": 0.3, "psi_passed": False, "violation": "OVERHEAT", "step": 2, "timestamp": 0.2},
            {"eta": 0.4, "psi_passed": False, "violation": "BURN_BACK", "step": 3, "timestamp": 0.3},
        ]
        stats = gen.aggregate_ksnap_stats(entries)
        assert stats["violation_count"] == 3
        assert stats["violation_types"]["BURN_BACK"] == 2
        assert stats["violation_types"]["OVERHEAT"] == 1

    def test_aggregate_time_span(self):
        """时间跨度计算正确."""
        from tools.wps_pqr_generator import WpsPqrGenerator
        gen = WpsPqrGenerator()
        entries = [
            {"eta": 0.1, "psi_passed": True, "violation": "", "step": 0, "timestamp": 1.0},
            {"eta": 0.2, "psi_passed": True, "violation": "", "step": 1, "timestamp": 5.0},
        ]
        stats = gen.aggregate_ksnap_stats(entries)
        assert abs(stats["time_span_s"] - 4.0) < 0.01


# ── QA 数据健康检查测试 ──

class TestQADataHealth:
    """测试数据质量QA检查."""

    def test_qa_check_completeness(self):
        """完整性检查."""
        from tools.qa_data_health import WeldDataQACheck
        qa = WeldDataQACheck()
        data = {"current": np.array([200, np.nan, 180, 190])}
        result = qa.check_completeness(data)
        assert result is not None

    def test_qa_check_consistency(self):
        """一致性检查."""
        from tools.qa_data_health import WeldDataQACheck
        qa = WeldDataQACheck()
        data = {"current": np.array([200, 250, 180]), "voltage": np.array([24, 26, 22])}
        result = qa.check_consistency(data)
        assert result is not None

    def test_qa_self_test(self):
        """qa_data_health._self_test() 通过."""
        from tools.qa_data_health import _self_test
        assert _self_test() is True


# ── PCM CIM 模拟器测试 ──

class TestPCMModel:
    """测试PCM (Phase Change Memory)相变存储器模型."""

    def test_pcm_model_basic(self):
        """PCM SET/RESET/部分SET电导态."""
        from tools.tproc_cim_simulator import PCMModel, PCM_G_MAX_S, PCM_G_MIN_S
        cell = PCMModel()

        # RESET态: 低电导
        cell.reset()
        assert cell.code == 0, f"RESET code should be 0, got {cell.code}"
        assert abs(cell.g - PCM_G_MIN_S) < 1e-12, "RESET conductance should be G_min"

        # SET态: 高电导
        cell.set()
        assert cell.code == 0xFFFF, f"SET code should be 0xFFFF, got {cell.code}"
        assert abs(cell.g - PCM_G_MAX_S) < 1e-12, "SET conductance should be G_max"

        # 部分SET: 中间电导 (0x8000 = 50%)
        cell.partial_set(0x8000)
        assert cell.code == 0x8000
        expected_g = PCM_G_MIN_S + 0x8000 / 0xFFFF * (PCM_G_MAX_S - PCM_G_MIN_S)
        assert abs(cell.g - expected_g) < 1e-12, \
            f"Partial SET conductance mismatch: {cell.g} vs {expected_g}"

    def test_pcm_model_conversion(self):
        """PCM电导码↔电导转换."""
        from tools.tproc_cim_simulator import PCMModel
        cell = PCMModel()
        for test_code in [0, 0x2000, 0x4000, 0x8000, 0xC000, 0xFFFF]:
            g = cell.code_to_conductance(test_code)
            code_back = cell.conductance_to_code(g)
            assert abs(code_back - test_code) <= 1, \
                f"Round-trip failed: {test_code} → {g} → {code_back}"

    def test_pcm_pulse_verify(self):
        """PCM脉冲校验写入收敛."""
        from tools.tproc_cim_simulator import PCMCrossbarArray, PCM_TARGET_CODE, PCM_TOLERANCE
        np.random.seed(42)
        pcm = PCMCrossbarArray(8)
        result = pcm.pulse_verify_write(0, 0, PCM_TARGET_CODE)

        assert result["converged"], \
            f"Pulse-verify should converge to 0x{PCM_TARGET_CODE:04X}, " \
            f"got 0x{result['final_code']:04X}"
        assert result["pulses"] <= 10, \
            f"Should converge in ≤10 pulses, got {result['pulses']}"
        assert result["error"] <= PCM_TOLERANCE, \
            f"Error {result['error']} exceeds tolerance {PCM_TOLERANCE}"
        # Sequence should have at least 2 entries (initial + at least 1 pulse)
        assert len(result["sequence"]) >= 2, \
            f"Sequence should have ≥2 entries, got {len(result['sequence'])}"

    def test_pcm_crossbar(self):
        """PCM阵列MAC运算."""
        from tools.tproc_cim_simulator import PCMCrossbarArray, PCM_G_MIN_S, PCM_G_MAX_S, PCM_CODE_MAX
        pcm = PCMCrossbarArray(8)

        # 全RESET: 近似零电流 (G_min leakage)
        vec = np.ones(8) * 0.5
        currents, _ = pcm.matrix_vector_mult(vec)
        assert np.allclose(currents, 0, atol=1e-6), \
            "All-RESET PCM array should produce ~0 current"

        # 对角线编程到0x8000
        for i in range(8):
            pcm.set_weight_code(i, i, 0x8000)

        currents, energy = pcm.matrix_vector_mult(vec)
        g_mid = PCM_G_MIN_S + 0x8000 / PCM_CODE_MAX * (PCM_G_MAX_S - PCM_G_MIN_S)
        # Expected: diagonal (g_mid) + 7 off-diagonal (g_min) leakage
        expected_current = (g_mid + 7 * PCM_G_MIN_S) * 0.5 * 0.1
        assert np.allclose(currents, expected_current, atol=1e-8), \
            f"PCM diagonal current should be ~{expected_current}, got {currents[0]}"
        assert energy > 0, "PCM energy should be positive"

    def test_pcm_energy(self):
        """PCM能耗对比SRAM+ALU."""
        from tools.tproc_cim_simulator import PCMCrossbarArray, SRAM_ALU_ENERGY_PJ
        pcm = PCMCrossbarArray(8)
        for i in range(8):
            pcm.set_weight_code(i, i, 0x8000)
        vec = np.ones(8) * 0.5
        _, energy_pcm = pcm.matrix_vector_mult(vec)
        energy_pcm_pj = energy_pcm * 1e12
        assert energy_pcm_pj < SRAM_ALU_ENERGY_PJ, \
            f"PCM ({energy_pcm_pj:.2f}pJ) should be < SRAM+ALU ({SRAM_ALU_ENERGY_PJ:.2f}pJ)"

    def test_pcm_run_comparison(self):
        """run_pcm_comparison返回正确结构."""
        from tools.tproc_cim_simulator import run_pcm_comparison
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_pcm_comparison()
        assert "sram_alu_pj" in result
        assert "rrrim_pj" in result
        assert "pcm_pj" in result
        assert "saving_rrrim" in result
        assert "saving_pcm" in result
        assert result["saving_pcm"] > 1.0, "PCM should save energy vs SRAM"

    def test_pcm_cim_self_test(self):
        """tproc_cim_simulator._self_test() 通过(含PCM测试)."""
        from tools.tproc_cim_simulator import _self_test
        assert _self_test() is True


# ── EML→PCM标定测试 ──

class TestEMLPCMCalibration:
    """测试EML→PCM电导标定."""

    def test_eml_calibration(self):
        """EML→PCM标定基本流程."""
        from tools.eml_to_pcm_calibration import EMLPCMCalibrator
        calibrator = EMLPCMCalibrator()

        # 标准八元数EML节点
        components = np.array([0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.1, 0.8])
        result = calibrator.calibrate_eml_node(components, method="outer")

        assert "weight_matrix" in result
        assert "target_codes" in result
        assert "actual_codes" in result
        assert result["target_codes"].shape == (8, 8)
        assert result["actual_codes"].shape == (8, 8)
        assert result["num_cells"] == 64

    def test_eml_calibration_convergence(self):
        """EML标定脉冲收敛."""
        from tools.eml_to_pcm_calibration import EMLPCMCalibrator
        np.random.seed(42)
        calibrator = EMLPCMCalibrator()

        # 目标码0x4000的脉冲校验写入
        result = calibrator.pulse_verify_write(0x4000)
        assert result.converged, "Should converge to 0x4000"
        assert result.pulse_count <= 10, "Should converge in ≤10 pulses"
        assert result.error_code <= calibrator.tolerance

    def test_eml_calibration_verification(self):
        """EML标定验证精度."""
        from tools.eml_to_pcm_calibration import EMLPCMCalibrator
        np.random.seed(42)
        calibrator = EMLPCMCalibrator()

        components = np.array([0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.1, 0.8])
        cal = calibrator.calibrate_eml_node(components)
        ver = calibrator.verify_calibration(cal["target_codes"], cal["actual_codes"])

        assert "max_error" in ver
        assert "pass_rate" in ver
        assert ver["pass_rate"] > 0.8, f"Pass rate should be >80%, got {ver['pass_rate']*100:.1f}%"

    def test_eml_calibration_self_test(self):
        """eml_to_pcm_calibration._self_test() 通过."""
        from tools.eml_to_pcm_calibration import _self_test
        assert _self_test() is True


# ── κ-Snap根因代码测试 ──

class TestKSnapRootCause:
    """测试κ-Snap根因代码生成."""

    def test_ksnap_root_cause_gas_contamination(self):
        """气体污染根因识别."""
        from core.ksnap_root_cause import KSnapRootCauseGenerator, RootCauseType, _make_synthetic_snapshot
        generator = KSnapRootCauseGenerator(eta_threshold=0.5)
        snap = _make_synthetic_snapshot(RootCauseType.GAS_CONTAMINATION)
        code = generator.analyze(snap)

        assert code.root_cause_type == RootCauseType.GAS_CONTAMINATION, \
            f"Should identify Gas_Contamination, got {code.root_cause_type.value}"
        assert code.confidence > 0.3
        assert "Increase" in code.action

    def test_ksnap_root_cause_wire_stick(self):
        """粘丝根因识别."""
        from core.ksnap_root_cause import KSnapRootCauseGenerator, RootCauseType, _make_synthetic_snapshot
        generator = KSnapRootCauseGenerator(eta_threshold=0.5)
        snap = _make_synthetic_snapshot(RootCauseType.WIRE_STICK)
        code = generator.analyze(snap)

        assert code.root_cause_type == RootCauseType.WIRE_STICK, \
            f"Should identify Wire_Stick, got {code.root_cause_type.value}"
        assert code.confidence > 0.3

    def test_ksnap_root_cause_format(self):
        """根因代码格式字符串."""
        from core.ksnap_root_cause import RootCauseCode, RootCauseType
        code = RootCauseCode(
            cause="Gas_Contamination",
            action="Increase_Flow_20%",
            confidence=0.94,
            root_cause_type=RootCauseType.GAS_CONTAMINATION,
        )
        fmt = code.format_string()
        assert "RootCause: Gas_Contamination" in fmt
        assert "Action: Increase_Flow_20%" in fmt
        assert "Confidence: 0.94" in fmt

    def test_ksnap_root_cause_feedback(self):
        """工艺反哺建议生成."""
        from core.ksnap_root_cause import KSnapRootCauseGenerator, RootCauseType, _make_synthetic_snapshot
        generator = KSnapRootCauseGenerator()
        snap = _make_synthetic_snapshot(RootCauseType.GAS_CONTAMINATION)
        code = generator.analyze(snap)
        feedback = generator.generate_feedback(code)

        assert "process_param_delta" in feedback
        assert "eml_node_hint" in feedback
        assert "gas_flow" in feedback["process_param_delta"]
        assert feedback["process_param_delta"]["gas_flow"] > 0

    def test_ksnap_root_cause_self_test(self):
        """ksnap_root_cause._self_test() 通过."""
        from core.ksnap_root_cause import _self_test
        assert _self_test() is True


# ── SAC焊接训练测试 ──

class TestSACWeldTrain:
    """测试SAC焊接训练脚本."""

    def test_training_stats(self):
        """TrainingStats数据结构."""
        from baselines.sac_weld_train import TrainingStats
        stats = TrainingStats()
        stats.episode_returns = [1.0, 2.0, 3.0]
        stats.episode_lengths = [100, 200, 300]
        summary = stats.summary()
        assert summary["n_episodes"] == 3
        assert abs(summary["mean_return"] - 2.0) < 0.01

    def test_numpy_sac_stub(self):
        """NumpySAC Stub基本功能."""
        from baselines.sac_weld_train import NumpySACStub
        agent = NumpySACStub(obs_dim=18, act_dim=4)
        obs = np.random.randn(18).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_sac_weld_self_test(self):
        """sac_weld_train._self_test() 通过."""
        from baselines.sac_weld_train import _self_test
        assert _self_test() is True
