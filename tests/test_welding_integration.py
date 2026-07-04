"""
焊接系统端到端集成测试
=====================

测试 T03+T04+T05 所有模块的集成功能。

Author: MuJoCo-Bench-IDO Welding Module v0.2.0
"""

import os
import sys
import numpy as np
import pytest

# 添加项目根路径
_PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestWeldingIntegration:
    """焊接系统端到端集成测试."""

    def test_full_pipeline(self):
        """完整流程: XML加载→环境创建→轨迹规划→焊接执行→安全检查→质量评估."""
        # 1. 创建焊接环境
        from envs.welding_env import WeldingEnv
        try:
            env = WeldingEnv(weld_type="flat")
        except Exception as e:
            pytest.skip(f"WeldingEnv 创建失败 (可能MuJoCo未安装): {e}")

        # 2. 重置环境
        obs = env.reset()
        assert obs.shape == (18,), f"观测维度应为18, 实际为 {obs.shape}"

        # 3. 执行焊接步骤
        action = np.array([200.0, 24.0, 2.0, 6.0])
        result = env.step(action)

        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        assert result["observation"].shape == (18,)

        # 4. 安全检查
        safety = result["info"].get("safety", {})
        assert "passed" in safety

        # 5. 质量评估
        quality = result["info"].get("quality", {})
        assert "eta" in quality
        assert "porosity" in quality
        assert "distortion" in quality

    def test_process_proxy(self):
        """WeldingProcessProxy预测功能."""
        from core.welding_process_proxy import WeldingProcessProxy, WeldingQuality

        proxy = WeldingProcessProxy(weld_type="flat")

        # 测试主预测函数
        quality = proxy.predict(
            current=200.0,
            voltage=24.0,
            travel_speed=6.0,
            wire_feed=8.0,
            stickout=15.0,
            weave=2.0,
        )

        assert isinstance(quality, WeldingQuality)
        assert quality.eta_residual >= 0.0
        assert 0.0 <= quality.porosity_risk <= 1.0
        assert quality.angular_distortion >= 0.0
        assert quality.penetration_depth >= 0.0
        assert quality.arc_length >= 0.0
        assert quality.heat_input >= 0.0

        # 最优参数应产生较低的 eta
        quality_optimal = proxy.predict(200.0, 24.0, 6.0, stickout=15.0)
        quality_bad = proxy.predict(350.0, 32.0, 2.0, stickout=25.0)
        assert quality_optimal.eta_residual < quality_bad.eta_residual

        # 测试兼容接口
        quality_dict = proxy.predict_quality(
            current=200.0, voltage=24.0, speed=6.0, stickout=15.0
        )
        assert "eta" in quality_dict
        assert "porosity" in quality_dict
        assert "distortion" in quality_dict

        # 测试各计算函数
        assert proxy.compute_heat_input(200.0, 24.0, 6.0) > 0.0
        assert proxy.compute_arc_length(24.0) == 10.0
        porosity = proxy.compute_porosity(15.0, 0.1)
        assert 0.0 <= porosity <= 1.0
        distortion = proxy.compute_distortion(1.0)
        assert distortion >= 0.0
        penetration = proxy.compute_penetration(200.0, 24.0, 6.0)
        assert penetration > 0.0

        # 测试电流历史和方差
        proxy.update_current_history(200.0)
        proxy.update_current_history(205.0)
        variance = proxy.compute_current_variance()
        assert variance >= 0.0

    def test_sensor_suite(self):
        """WeldingSensorSuite读取功能."""
        from core.welding_sensors import WeldingSensorSuite, SENSOR_CONFIGS

        # 验证7类传感器配置
        assert len(SENSOR_CONFIGS) == 7

        suite = WeldingSensorSuite()

        # 验证默认启用所有传感器
        assert len(suite.sensor_types) == 7

        # 测试自适应采样率
        suite.adapt_sample_rate(0.8)  # > 0.5 → 2x
        rates = suite.get_effective_sample_rates()
        for sensor_name, rate in rates.items():
            base_rate = SENSOR_CONFIGS[sensor_name].sample_rate_hz
            assert rate >= base_rate * 1.5  # 应该被提升 (2x)

        suite.adapt_sample_rate(0.05)  # < 0.1 → 0.5x
        rates = suite.get_effective_sample_rates()
        for sensor_name, rate in rates.items():
            base_rate = SENSOR_CONFIGS[sensor_name].sample_rate_hz
            assert rate <= base_rate * 0.6  # 应该被降低 (0.5x)

        # 测试温度和电流读取
        temp = suite.read_temperature(150.0)
        assert temp == 150.0

        current = suite.read_arc_current(200.0)
        assert current == 200.0

        # 测试部分传感器启用
        suite_partial = WeldingSensorSuite(
            sensor_types=["tcp_pose", "temperature"]
        )
        assert len(suite_partial.sensor_types) == 2

    def test_tomas_axioms(self):
        """TOMAS公理检查功能."""
        from core.tomas_welding_axioms import TomasWeldingAxioms, WeldingAxiom

        axioms = TomasWeldingAxioms()

        # 验证7条公理
        all_axioms = axioms.get_all_axioms()
        assert len(all_axioms) == 7

        # 验证公理名称
        names = axioms.get_axiom_names()
        assert "heat_input_limit" in names
        assert "stickout_range" in names
        assert "porosity_risk_limit" in names
        assert "angular_distortion_limit" in names
        assert "min_penetration" in names
        assert "current_voltage_match" in names
        assert "speed_range" in names

        # 测试良好焊接状态 (全部满足)
        good_state = {
            "heat_input": 1.5,
            "stickout": 15.0,
            "porosity_risk": 0.02,
            "angular_distortion": 0.5,
            "penetration_depth": 3.0,
            "current": 200.0,
            "voltage": 24.0,
            "travel_speed": 6.0,
        }
        result = axioms.check_all(good_state)
        assert result["passed"] is True
        assert len(result["violations"]) == 0

        # 测试不良焊接状态 (有违规)
        bad_state = {
            "heat_input": 3.0,  # > 2.5
            "stickout": 5.0,    # < 8
            "porosity_risk": 0.2,  # > 0.1
            "angular_distortion": 3.0,  # > 2.0
            "penetration_depth": 0.5,   # < 1.5
            "current": 200.0,
            "voltage": 24.0,
            "travel_speed": 6.0,
        }
        result = axioms.check_all(bad_state)
        assert result["passed"] is False
        assert len(result["violations"]) > 0

        # 测试单个公理检查
        assert axioms.check_axiom("heat_input_limit", good_state) is True
        assert axioms.check_axiom("heat_input_limit", bad_state) is False

        # 测试获取公理
        axiom = axioms.get_axiom("stickout_range")
        assert axiom is not None
        assert isinstance(axiom, WeldingAxiom)
        assert axiom.name == "stickout_range"

    def test_wps_pqr_generator(self):
        """WPS/PQR文档生成."""
        from tools.wps_pqr_generator import WpsPqrGenerator

        gen = WpsPqrGenerator()

        params = {
            "current": 200.0,
            "voltage": 24.0,
            "travel_speed": 6.0,
            "stickout": 15.0,
            "weave": 2.0,
            "wire_feed": 8.0,
            "gas_flow": 15.0,
            "gas_type": "Ar+CO2 80/20",
            "wire_diameter": 1.2,
        }
        quality = {
            "eta_residual": 0.1,
            "porosity_risk": 0.02,
            "angular_distortion": 0.5,
            "penetration_depth": 3.0,
            "heat_input": 0.8,
            "arc_length": 10.0,
        }

        # 测试 WPS LaTeX
        wps_latex = gen.generate_wps(params, quality, "flat", "latex")
        assert "Welding Procedure Specification" in wps_latex
        assert "flat" in wps_latex

        # 测试 WPS HTML
        wps_html = gen.generate_wps(params, quality, "flat", "html")
        assert "<html>" in wps_html
        assert "flat" in wps_html

        # 测试 PQR LaTeX
        pqr_latex = gen.generate_pqr(params, quality, "flat", "latex")
        assert "Procedure Qualification Record" in pqr_latex
        assert "QUALIFIED" in pqr_latex or "NOT QUALIFIED" in pqr_latex

        # 测试 PQR HTML
        pqr_html = gen.generate_pqr(params, quality, "flat", "html")
        assert "<html>" in pqr_html

        # 测试同时生成
        both = gen.generate_both(params, quality, "flat", "latex")
        assert "wps" in both
        assert "pqr" in both

        # 测试参数验证
        assert gen._validate_params(params) is True
        assert gen._validate_params({}) is False

    def test_eml_distiller(self):
        """EML蒸馏Pareto搜索."""
        from core.welding_eml_distill import WeldingEMLDistiller
        from core.welding_process_proxy import WeldingProcessProxy

        proxy = WeldingProcessProxy()
        distiller = WeldingEMLDistiller(process_proxy=proxy)

        # 测试 Pareto 搜索 (少量试验以加速测试)
        pareto_front = distiller.search_pareto_optimal(
            n_trials=50, weld_type="flat"
        )
        assert len(pareto_front) > 0

        # 每个 Pareto 点应有 params 和 quality
        for point in pareto_front:
            assert "params" in point
            assert "quality" in point
            assert "current" in point["params"]
            assert "voltage" in point["params"]
            assert "speed" in point["params"]
            assert "stickout" in point["params"]
            assert "eta" in point["quality"]

        # 测试蒸馏到 EML
        eml_nodes = distiller.distill_to_eml(weld_type="flat")
        assert len(eml_nodes) > 0
        for node in eml_nodes:
            assert "eml_node_id" in node
            assert node["eml_node_id"].startswith("weld_pareto_")

        # 测试便捷方法
        distiller2 = WeldingEMLDistiller(process_proxy=WeldingProcessProxy())
        nodes = distiller2.compute_pareto_front(n_trials=30, weld_type="flat")
        assert len(nodes) > 0

        # 测试最佳参数
        best = distiller2.get_best_params(weld_type="flat")
        assert "params" in best
        assert "quality" in best

        # 测试摘要
        summary = distiller.summarize_pareto()
        assert summary["n_points"] > 0

    def test_welding_compare(self):
        """对比评估LaTeX表格生成."""
        from benchmarks.welding_compare import WeldingCompare, BENCHMARK_DATA

        # 验证基准数据
        assert "PID" in BENCHMARK_DATA
        assert "VLA" in BENCHMARK_DATA
        assert "IDO/TOMAS" in BENCHMARK_DATA

        compare = WeldingCompare()

        # 运行所有方法对比 (不使用env, 使用基准数据)
        results = compare.run_all(n_episodes=3)
        assert len(results) >= 3  # 至少 PID, VLA, IDO/TOMAS

        for method, metrics in results.items():
            assert "tracking_error_mm" in metrics
            assert "current_fluctuation_A" in metrics
            assert "stick_rate_pct" in metrics
            assert "defect_rate_pct" in metrics

        # 验证 IDO/TOMAS 性能最好
        ido_defect = results.get("IDO/TOMAS", {}).get("defect_rate_pct", 100)
        vla_defect = results.get("VLA", {}).get("defect_rate_pct", 0)
        assert ido_defect < vla_defect

        # 测试 LaTeX 表格生成
        latex_table = compare.generate_latex_table(results)
        assert "\\begin{table}" in latex_table
        assert "\\end{table}" in latex_table
        assert "轨迹误差" in latex_table or "tracking" in latex_table.lower()

        # 测试 Markdown 表格生成
        md_table = compare.generate_markdown_table(results)
        assert "|" in md_table
        assert "轨迹误差" in md_table or "tracking" in md_table.lower()

    def test_dreamer_trainer_init(self):
        """DreamerV3训练器初始化（不实际训练）."""
        from baselines.dreamer_weld_train import (
            WeldingDreamerTrainer, RSSM, Actor, Critic, ReplayBuffer
        )

        # 测试 RSSM 初始化
        rssm = RSSM(obs_dim=18, action_dim=4)
        obs = np.random.randn(18)
        latent = rssm.encode(obs)
        assert latent.shape[0] == 512  # hidden_dim

        action = np.array([200.0, 24.0, 2.0, 6.0])
        next_latent = rssm.transition(latent, action)
        assert next_latent.shape[0] == 512

        recon = rssm.decode(latent)
        assert recon.shape[0] == 18

        reward = rssm.predict_reward(latent)
        assert isinstance(reward, float)

        loss = rssm.compute_loss({
            "observations": np.random.randn(5, 18),
            "actions": np.random.randn(4, 4),
            "rewards": np.random.randn(5),
        })
        assert isinstance(loss, float)

        # 测试 Actor
        actor = Actor(latent_dim=512, action_dim=4)
        action_out = actor.act(latent, explore=True)
        assert action_out.shape == (4,)
        assert 50.0 <= action_out[0] <= 350.0  # current 范围

        # 测试 Critic
        critic = Critic(latent_dim=512)
        value = critic.value(latent)
        assert isinstance(value, float)

        # 测试 ReplayBuffer
        buffer = ReplayBuffer(capacity=100)
        buffer.add(
            [np.random.randn(18) for _ in range(10)],
            [np.random.randn(4) for _ in range(9)],
            [float(np.random.randn()) for _ in range(10)],
        )
        assert len(buffer) == 1
        batch = buffer.sample(1)
        assert len(batch) == 1

        # 测试训练器初始化 (不实际训练)
        # 使用 mock env
        class MockEnv:
            OBS_DIM = 18
            ACTION_DIM = 4
            def reset(self):
                return np.zeros(18)
            def step(self, action):
                return {
                    "observation": np.zeros(18),
                    "reward": -1.0,
                    "done": False,
                    "info": {"quality": {"eta": 0.1, "porosity": 0.02}},
                }
            @property
            def action_spec(self):
                return {
                    "shape": (4,),
                    "low": np.array([50.0, 14.0, 0.0, 2.0]),
                    "high": np.array([350.0, 32.0, 5.0, 15.0]),
                }

        trainer = WeldingDreamerTrainer(MockEnv())
        assert trainer.rssm is not None
        assert trainer.actor is not None
        assert trainer.critic is not None
        assert trainer.buffer is not None

        # 测试收集一个短 episode
        episode_data = trainer.collect_episode()
        assert "observations" in episode_data
        assert "actions" in episode_data
        assert "rewards" in episode_data

        # 测试短训练 (2 episodes)
        history = trainer.train(num_episodes=2)
        assert len(history) == 2
        assert "reward" in history[0]
        assert "eta" in history[0]
