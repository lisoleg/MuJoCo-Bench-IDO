"""
TOMAS VLA Weight Loader — Real VLA Model Integration
=====================================================

v0.17.1: VLA weight loading verification and integration for TOMAS Agent.

This module provides utilities to:
  1. Check which VLA models can load real weights on this system
  2. Load real VLA weights (OpenVLA-7B, Octo-Base, Pi0-Base)
  3. Verify VLA output is compatible with TOMASMuJoCoWrapper
  4. Benchmark VLA inference latency

Supported VLA Models:
  ┌─────────────┬──────────┬───────────┬────────────────────┬───────┐
  │ Model       │ Faction  │ Params    │ Core Tech          │ Freq  │
  ├─────────────┼──────────┼───────────┼────────────────────┼───────┤
  │ OpenVLA-7B  │ Academic │ 7B        │ DINOv2+SigLIP+LLaMA│ 2-10Hz│
  │ Octo-Base   │ Academic │ 93M       │ Multi-view+Chunk   │ ~10Hz │
  │ Pi0-Base    │ Tech     │ PaliGemma │ Flow Matching      │ 50Hz  │
  │ DemoVLA     │ Built-in │ N/A       │ Instruction-driven │ 30Hz  │
  └─────────────┴──────────┴───────────┴────────────────────┴───────┘

Author: MuJoCo-Bench-IDO v0.17.1
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class VLALoader:
    """VLA model weight loader and verifier.

    Handles loading real VLA model weights from HuggingFace/GitHub
    and verifying they produce valid actions for the SO-ARM100 task.

    Attributes:
        loaded_models: Dict of successfully loaded model names to adapters.
    """

    # VLA model specifications
    MODEL_SPECS: Dict[str, Dict[str, Any]] = {
        "openvla-7b": {
            "params": "7B",
            "faction": "Academic",
            "architecture": "DINOv2 + SigLIP + Llama-2",
            "control_freq": "2-10Hz",
            "min_vram_gb": 16,
            "hf_repo": "openvla/openvla-7b",
            "input": "image + language + proprio",
            "output": "7-DOF discretized joint positions (256 bins/DOF)",
            "ido_role": "P-Layer (mimicry)",
            "missing_in_ido": ["κ-Snap audit", "ψ-Anchor safety", "EML-SemZip"],
        },
        "octo-base": {
            "params": "93M",
            "faction": "Academic",
            "architecture": "Transformer + Action Chunking",
            "control_freq": "~10Hz",
            "min_vram_gb": 4,
            "hf_repo": "octo-models/octo",
            "input": "image (primary + wrist) + language",
            "output": "7-DOF continuous actions (action chunk)",
            "ido_role": "P-Layer (mimicry)",
            "missing_in_ido": ["κ-Snap audit", "ψ-Anchor safety", "EML-SemZip"],
        },
        "pi0-base": {
            "params": "PaliGemma 2B + 300M",
            "faction": "Tech-Extreme",
            "architecture": "PaliGemma + Flow Matching Action Expert",
            "control_freq": "50Hz",
            "min_vram_gb": 8,
            "hf_repo": "Physical-Intelligence/openpi",
            "input": "image + language + proprio",
            "output": "50-step action chunk via Flow Matching",
            "ido_role": "P-Layer (highest φ-flow density)",
            "missing_in_ido": ["κ-Snap audit", "ψ-Anchor safety", "EML-SemZip"],
        },
        "demo-vla": {
            "params": "N/A",
            "faction": "Built-in",
            "architecture": "Instruction-driven state machine + FK-based IK",
            "control_freq": "30Hz",
            "min_vram_gb": 0,
            "hf_repo": None,
            "input": "language + proprio + object_positions",
            "output": "7-DOF joint position targets",
            "ido_role": "P-Layer (demo mode)",
            "missing_in_ido": [],
        },
    }

    def __init__(self) -> None:
        self.loaded_models: Dict[str, Any] = {}

    def check_system_requirements(self, model_name: str) -> Dict[str, Any]:
        """Check if the system meets requirements for a VLA model.

        Args:
            model_name: VLA model name.

        Returns:
            Dict with requirement check results.
        """
        spec = self.MODEL_SPECS.get(model_name, {})
        result = {
            "model": model_name,
            "can_load": False,
            "checks": {},
            "missing": [],
            "spec": spec,
        }

        if model_name == "demo-vla":
            result["can_load"] = True
            result["checks"]["always_available"] = True
            return result

        # Check torch
        try:
            import torch
            result["checks"]["torch"] = {
                "available": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
                result["checks"]["torch"]["vram_gb"] = round(vram_gb, 1)
                if vram_gb < spec.get("min_vram_gb", 0):
                    result["missing"].append(
                        f"VRAM {vram_gb:.1f}GB < required {spec['min_vram_gb']}GB"
                    )
            else:
                result["missing"].append("CUDA not available (CPU inference too slow)")
        except ImportError:
            result["checks"]["torch"] = {"available": False}
            result["missing"].append("torch not installed: pip install torch")

        # Check model-specific packages
        if model_name == "openvla-7b":
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                result["checks"]["transformers"] = {"available": True}
            except ImportError:
                result["checks"]["transformers"] = {"available": False}
                result["missing"].append("pip install transformers")
        elif model_name == "octo-base":
            try:
                import octo
                result["checks"]["octo"] = {"available": True}
            except ImportError:
                result["checks"]["octo"] = {"available": False}
                result["missing"].append("pip install octo")
        elif model_name == "pi0-base":
            try:
                import openpi
                result["checks"]["openpi"] = {"available": True}
            except ImportError:
                result["checks"]["openpi"] = {"available": False}
                result["missing"].append("pip install openpi (Physical-Intelligence/openpi)")

        result["can_load"] = len(result["missing"]) == 0
        return result

    def load_model(self, model_name: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load a VLA model with real weights.

        Args:
            model_name: VLA model name.
            **kwargs: Additional arguments for the adapter.

        Returns:
            (adapter, info) tuple. adapter is the VLA adapter instance,
            info contains loading details.
        """
        from webviz.tomas_wrapper import create_vla_adapter

        info = {
            "model": model_name,
            "loaded": False,
            "load_time": 0.0,
            "error": None,
            "real_weights": False,
        }

        # Check requirements first
        req = self.check_system_requirements(model_name)
        if not req["can_load"]:
            info["error"] = "Requirements not met: " + "; ".join(req["missing"])
            logger.warning(f"VLA {model_name}: {info['error']}")
            # Fall back to stub
            adapter = create_vla_adapter(model_name, **kwargs)
            info["loaded"] = adapter.is_loaded()
            return adapter, info

        start_time = time.time()

        try:
            adapter = create_vla_adapter(model_name, **kwargs)

            # For real VLA models, trigger _try_load()
            if hasattr(adapter, "_try_load"):
                loaded = adapter._try_load()
                info["real_weights"] = loaded
                if not loaded:
                    info["error"] = "_try_load() returned False (weights not found or OOM)"
                else:
                    info["loaded"] = True
            else:
                info["loaded"] = adapter.is_loaded()
                info["real_weights"] = adapter.is_loaded()

        except Exception as e:
            info["error"] = str(e)
            logger.error(f"VLA {model_name} load failed: {e}")
            adapter = create_vla_adapter(model_name, **kwargs)

        info["load_time"] = round(time.time() - start_time, 2)
        self.loaded_models[model_name] = adapter

        return adapter, info

    def verify_action_output(
        self,
        adapter: Any,
        obs_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verify VLA adapter produces valid actions.

        Args:
            adapter: VLA adapter instance.
            obs_dict: Test observation dict. If None, uses dummy.

        Returns:
            Verification result dict.
        """
        if obs_dict is None:
            obs_dict = {
                "language": "pick up the red cube",
                "proprio": np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0]),
                "object_positions": {
                    "red_cube": np.array([0.22, 0.18, 0.435]),
                    "tray": np.array([0.22, -0.18, 0.41]),
                },
                "gripper_position": np.array([0.0, 0.0, 0.40]),
                "arm_base_z": 0.40,
                "rgb": None,
                "wrist_rgb": None,
            }

        result = {
            "valid": False,
            "action": None,
            "action_shape": None,
            "action_range": None,
            "inference_time": 0.0,
            "error": None,
        }

        start = time.time()
        try:
            action = adapter.predict(obs_dict)
            elapsed = time.time() - start

            action = np.asarray(action, dtype=np.float64)
            result["action"] = action.tolist()
            result["action_shape"] = list(action.shape)
            result["action_range"] = [float(action.min()), float(action.max())]
            result["inference_time"] = round(elapsed * 1000, 1)  # ms

            # Check validity
            if action.shape == (7,) and not np.any(np.isnan(action)):
                result["valid"] = True
            else:
                result["error"] = f"Invalid action: shape={action.shape}, has_nan={np.any(np.isnan(action))}"

        except Exception as e:
            result["error"] = str(e)

        return result

    def benchmark_inference(
        self,
        adapter: Any,
        num_trials: int = 10,
        obs_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Benchmark VLA inference latency.

        Args:
            adapter: VLA adapter instance.
            num_trials: Number of inference trials.
            obs_dict: Test observation.

        Returns:
            Benchmark results with timing statistics.
        """
        if obs_dict is None:
            obs_dict = {
                "language": "pick up the red cube",
                "proprio": np.zeros(7),
                "rgb": None,
                "wrist_rgb": None,
            }

        times: List[float] = []

        for i in range(num_trials):
            start = time.time()
            try:
                action = adapter.predict(obs_dict)
                elapsed = time.time() - start
                times.append(elapsed * 1000)  # ms
            except Exception as e:
                logger.error(f"Benchmark trial {i} failed: {e}")
                times.append(-1.0)

        valid_times = [t for t in times if t > 0]

        if not valid_times:
            return {"error": "All trials failed", "model": adapter.model_name}

        return {
            "model": adapter.model_name,
            "trials": num_trials,
            "successful": len(valid_times),
            "avg_ms": round(np.mean(valid_times), 2),
            "min_ms": round(np.min(valid_times), 2),
            "max_ms": round(np.max(valid_times), 2),
            "p50_ms": round(np.percentile(valid_times, 50), 2),
            "p95_ms": round(np.percentile(valid_times, 95), 2),
            "effective_hz": round(1000.0 / np.mean(valid_times), 1) if np.mean(valid_times) > 0 else 0,
        }

    def full_report(self) -> Dict[str, Any]:
        """Generate a full VLA availability report.

        Returns:
            Dict with all model specs, requirement checks, and loading status.
        """
        report = {
            "version": "v0.17.1",
            "models": {},
        }

        for model_name in self.MODEL_SPECS:
            spec = self.MODEL_SPECS[model_name]
            req = self.check_system_requirements(model_name)

            report["models"][model_name] = {
                "spec": spec,
                "requirements": req,
                "loaded": model_name in self.loaded_models,
            }

            if model_name in self.loaded_models:
                adapter = self.loaded_models[model_name]
                verify = self.verify_action_output(adapter)
                report["models"][model_name]["verification"] = verify

        return report


def load_vla_for_tomas(
    model_name: str = "demo-vla",
    instruction: str = "pick up the red cube",
    model=None,
    data=None,
) -> Tuple[Any, Dict[str, Any]]:
    """Convenience function: load VLA adapter for TOMAS Agent.

    This is the main entry point for loading a VLA model to be used
    with TOMASAgent. It handles all the complexity of:
      1. Checking system requirements
      2. Attempting real weight loading
      3. Falling back to stub mode if real weights unavailable
      4. Verifying action output compatibility

    Args:
        model_name: VLA model name (openvla-7b, octo-base, pi0-base, demo-vla).
        instruction: Language instruction for the VLA.
        model: MuJoCo model (for DemoVLA IK).
        data: MuJoCo data (for DemoVLA IK).

    Returns:
        (adapter, info) tuple.
    """
    loader = VLALoader()

    kwargs = {}
    if model_name == "demo-vla":
        kwargs["model"] = model
        kwargs["data"] = data

    adapter, info = loader.load_model(model_name, **kwargs)

    # Verify
    if info.get("loaded", False) or model_name == "demo-vla":
        obs_dict = {
            "language": instruction,
            "proprio": np.array([0.0, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0]),
            "object_positions": {
                "red_cube": np.array([0.22, 0.18, 0.435]),
                "tray": np.array([0.22, -0.18, 0.41]),
            },
            "gripper_position": np.array([0.0, 0.0, 0.40]),
            "arm_base_z": 0.40,
            "rgb": None,
            "wrist_rgb": None,
        }
        verify = loader.verify_action_output(adapter, obs_dict)
        info["verification"] = verify

    logger.info(f"VLA loaded: {model_name} — real_weights={info.get('real_weights', False)}")
    return adapter, info
