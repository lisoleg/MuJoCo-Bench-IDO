"""
Tests for TOMAS Deploy API — Headless MuJoCo + TOMASAgent Integration
=====================================================================

v0.17.1: Integration tests for the TOMAS deployment pipeline on SO-ARM100.

Test coverage:
  1. HeadlessMuJoCoEnv: scene loading, obs shape, step/reset
  2. TOMASMuJoCoWrapper: P->C->S pipeline, info dict enrichment
  3. TOMASAgent.deploy(): multi-episode rollout, DeployReport
  4. MetaQuery: WHY_THIS_ACTION, AUDIT_SNAP
  5. VLA Loader: system requirements check, demo-vla loading
  6. run_tomas_eval: end-to-end evaluation entry point

Author: MuJoCo-Bench-IDO v0.17.1
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Scene XML path
_SCENE_XML = str(Path(_PROJECT_ROOT) / "webviz" / "scenes" / "so_arm100_scene.xml")


# ─── HeadlessMuJoCoEnv Tests ───


class TestHeadlessMuJoCoEnv:
    """Test the headless MuJoCo environment wrapper."""

    def test_scene_loading(self):
        """Verify SO-ARM100 scene XML loads correctly."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        assert env.model is not None
        assert env.data is not None
        assert env.model.nq > 0
        assert env.model.nu > 0

    def test_obs_shape(self):
        """Observation should be 18D."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        obs = env._get_obs()
        assert obs.shape == (18,), f"Expected (18,), got {obs.shape}"

    def test_step_returns_5tuple(self):
        """step() should return (obs, reward, terminated, truncated, info)."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        env.reset()
        action = np.zeros(env.model.nu)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (18,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset(self):
        """reset() should restore initial state and return obs."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        obs = env.reset()
        assert obs.shape == (18,)
        assert env.step_count == 0

    def test_action_spec(self):
        """action_spec() should return spec with correct shape."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        spec = env.action_spec()
        assert spec.shape == (env.model.nu,)

    def test_kinematic_assist(self):
        """Kinematic assist should move qpos toward ctrl target."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        env.reset()

        # Set a non-zero action and step
        action = np.array([0.5, 0.3, -0.3, 0.2, 0.0, 0.5, 0.5])
        if len(action) < env.model.nu:
            full = np.zeros(env.model.nu)
            full[:len(action)] = action
            action = full

        env.step(action)

        # Check that arm joints moved toward target (kinematic assist = 50%)
        arm_offset = env._arm_qpos_offset
        target = float(env.data.ctrl[0])
        actual = float(env.data.qpos[arm_offset])
        # Should have moved at least 40% of the way (50% assist minus physics drift)
        assert abs(actual - target) < abs(target) * 0.7 + 0.1, (
            f"Kinematic assist not working: target={target}, actual={actual}"
        )

    def test_body_ids_resolved(self):
        """Target and gripper body IDs should be valid."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=100)
        assert env.target_body_id >= 0, "red_cube body not found"
        assert env.gripper_body_id >= 0, "gripper_base body not found"


# ─── TOMASMuJoCoWrapper Tests ───


class TestTOMASMuJoCoWrapper:
    """Test the three-layer P->C->S wrapper."""

    def test_wrapper_step(self):
        """Wrapper step() should enrich info with TOMAS metadata."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper

        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=50)
        wrapper = TOMASMuJoCoWrapper(base_env=env, max_steps=50)
        wrapper.reset()

        result = wrapper.step(action=np.zeros(7))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        # Check TOMAS metadata in info
        assert "eta" in info
        assert "step" in info
        assert "psi_state" in info
        assert "safety_violations" in info
        assert "psi_violations" in info
        assert "raw_action" in info
        assert "snap_chain_verified" in info
        assert "tomas_version" in info
        assert info["step"] == 1

    def test_wrapper_audit_trail(self):
        """get_audit_trail() should return log buffer with details."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper

        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=50)
        wrapper = TOMASMuJoCoWrapper(base_env=env, max_steps=50)
        wrapper.reset()
        wrapper.step(action=np.zeros(7))
        wrapper.step(action=np.ones(7) * 0.1)

        trail = wrapper.get_audit_trail()
        assert len(trail) >= 2
        # Each entry should have details dict
        entry = trail[-1]
        assert "details" in entry
        details = entry["details"]
        assert "step" in details
        assert "action_norm" in details
        assert "psi_violations" in details

    def test_wrapper_safety_report(self):
        """get_safety_report() should return valid structure."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper

        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=50)
        wrapper = TOMASMuJoCoWrapper(base_env=env, max_steps=50)
        wrapper.reset()
        wrapper.step(action=np.zeros(7))

        report = wrapper.get_safety_report()
        assert "total_violations" in report
        assert "violation_breakdown" in report
        assert "chain_integrity" in report
        assert "steps_executed" in report
        assert report["chain_integrity"] is True

    def test_wrapper_reset_clears_state(self):
        """reset() should clear step count and violations."""
        from webviz.tomas_deploy_api import HeadlessMuJoCoEnv
        from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper

        env = HeadlessMuJoCoEnv(scene_xml_path=_SCENE_XML, max_steps=50)
        wrapper = TOMASMuJoCoWrapper(base_env=env, max_steps=50)
        wrapper.reset()
        wrapper.step(action=np.zeros(7))
        wrapper.step(action=np.zeros(7))
        assert wrapper.step_count == 2

        wrapper.reset()
        assert wrapper.step_count == 0
        assert len(wrapper._safety_violations) == 0


# ─── TOMASAgent.deploy() Tests ───


class TestTOMASAgentDeploy:
    """Test the TOMASAgent deployment orchestrator."""

    def test_deploy_single_episode(self):
        """deploy() should complete at least one episode."""
        from webviz.tomas_deploy_api import create_tomas_agent_for_arm100

        agent, _ = create_tomas_agent_for_arm100(
            vla_model_name="demo-vla",
            max_steps=50,
        )
        report = agent.deploy(num_episodes=1, max_steps_per_episode=50)

        assert report.total_episodes >= 1
        assert report.total_steps > 0
        assert report.avg_eta >= 0.0
        assert report.final_eta >= 0.0
        assert report.kappa_snap_count > 0
        assert report.elapsed_seconds > 0

    def test_deploy_report_to_dict(self):
        """DeployReport.to_dict() should be JSON-serializable."""
        import json
        from webviz.tomas_deploy_api import create_tomas_agent_for_arm100

        agent, _ = create_tomas_agent_for_arm100(
            vla_model_name="demo-vla",
            max_steps=30,
        )
        report = agent.deploy(num_episodes=1, max_steps_per_episode=30)
        d = report.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["status"] in ["success", "partial", "failed"]

    def test_deploy_multi_episode(self):
        """deploy() should handle multiple episodes."""
        from webviz.tomas_deploy_api import create_tomas_agent_for_arm100

        agent, _ = create_tomas_agent_for_arm100(
            vla_model_name="demo-vla",
            max_steps=30,
        )
        report = agent.deploy(num_episodes=2, max_steps_per_episode=30)
        assert report.total_episodes == 2

    def test_meta_query_why_this_action(self):
        """MetaQuery WHY_THIS_ACTION should return evidence."""
        from webviz.tomas_deploy_api import create_tomas_agent_for_arm100
        from agent.tomas_deploy import MetaQueryType

        agent, _ = create_tomas_agent_for_arm100(
            vla_model_name="demo-vla",
            max_steps=30,
        )
        agent.deploy(num_episodes=1, max_steps_per_episode=10)

        result = agent.meta_query(MetaQueryType.WHY_THIS_ACTION)
        assert result is not None
        assert "eta" in result.evidence
        assert result.confidence > 0.0

    def test_meta_query_audit_snap(self):
        """MetaQuery AUDIT_SNAP should return audit trail summary."""
        from webviz.tomas_deploy_api import create_tomas_agent_for_arm100
        from agent.tomas_deploy import MetaQueryType

        agent, _ = create_tomas_agent_for_arm100(
            vla_model_name="demo-vla",
            max_steps=30,
        )
        agent.deploy(num_episodes=1, max_steps_per_episode=10)

        result = agent.meta_query(MetaQueryType.AUDIT_SNAP)
        assert result is not None
        assert "total_steps" in result.evidence
        assert "eta_trend" in result.evidence
        assert result.confidence > 0.0


# ─── VLA Loader Tests ───


class TestVLALoader:
    """Test the VLA weight loader."""

    def test_model_specs_completeness(self):
        """All 4 VLA models should have complete specs."""
        from webviz.vla_loader import VLALoader
        loader = VLALoader()
        for name in ["openvla-7b", "octo-base", "pi0-base", "demo-vla"]:
            assert name in loader.MODEL_SPECS
            spec = loader.MODEL_SPECS[name]
            assert "params" in spec
            assert "min_vram_gb" in spec
            assert "hf_repo" in spec
            assert "ido_role" in spec

    def test_demo_vla_always_available(self):
        """demo-vla should always pass requirements check."""
        from webviz.vla_loader import VLALoader
        loader = VLALoader()
        req = loader.check_system_requirements("demo-vla")
        assert req["can_load"] is True

    def test_verify_action_output_demo(self):
        """Demo VLA should produce valid 7-DOF actions."""
        from webviz.vla_loader import VLALoader
        from webviz.tomas_wrapper import create_vla_adapter

        adapter = create_vla_adapter("demo-vla")
        loader = VLALoader()
        result = loader.verify_action_output(adapter)
        assert result["valid"] is True
        assert result["action_shape"] == [7]

    def test_full_report(self):
        """full_report() should return all model statuses."""
        from webviz.vla_loader import VLALoader
        loader = VLALoader()
        report = loader.full_report()
        assert "models" in report
        assert len(report["models"]) == 4


# ─── run_tomas_eval Integration Test ───


class TestRunTomasEval:
    """Test the end-to-end evaluation entry point."""

    def test_run_tomas_eval_demo(self):
        """run_tomas_eval() with demo-vla should produce a report."""
        from webviz.tomas_deploy_api import run_tomas_eval

        result = run_tomas_eval(
            vla_model_name="demo-vla",
            num_episodes=1,
            max_steps=30,
            verbose=False,
        )

        assert "deploy_report" in result
        assert "vla_model" in result
        assert result["vla_model"] == "demo-vla"
        assert result["deploy_report"]["total_steps"] > 0
        assert result["deploy_report"]["kappa_snap_count"] > 0

    def test_check_vla_availability(self):
        """check_vla_availability() should return all 4 models."""
        from webviz.tomas_deploy_api import check_vla_availability

        result = check_vla_availability()
        assert "demo-vla" in result
        assert "openvla-7b" in result
        assert "octo-base" in result
        assert "pi0-base" in result
        assert result["demo-vla"]["real_weights"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
