"""
Nine-Layer Cognitive Architecture — L0-L8 Mapping
===================================================

v0.16.25 P2: Nine-Layer L0-L8 Mapping

Maps the IDO/TOMAS nine-layer cognitive architecture to code modules
in the MuJoCo-Bench-IDO framework. Each layer corresponds to a biological
analogue and is implemented by specific code modules.

Layer Mapping:
  L0 心脏 (Heart)     → T-Processor (η-ALU + Ψ-Check + κ-FIFO)
                        core/t_processor.py
  L1 大脑 (Brain)     → LLM/VLA (OpenVLA, Octo, π₀) + S-Bridge LLM Attribution
                        webviz/tomas_wrapper.py (VLA adapters)
                        agent/s_bridge.py (ask_why_llm)
  L2 骨架 (Skeleton)  → Agent Framework (Hybrid-SAC, DreamerV3, IDO Agent)
                        agent/mujoco_ido_agent.py
  L3 性格 (Personality) → PreAffect + SafeFuse (intrinsic signals + graded response)
                        agent/pre_affect.py
                        agent/safe_fuse.py
  L4 感知 (Perception) → CAMKit (dual cameras) + Proprioception + κ-Snap Tokenizer
                        webviz/tomas_wrapper.py (CAMKit)
                        core/kappa_snap_tokenizer.py
  L5 学识 (Knowledge)  → EML-SemZip (RT-X data reweighting) + Skill Bank
                        agent/s_bridge.py (learn_skill, get_skill_bank)
  L6 手脚 (Hands/Feet) → C-Gate (ψ-Anchor) + PG-Gate + SO-ARM100 Controller
                        webviz/tomas_wrapper.py (PsiAnchorGate)
                        core/pg_gate.py
                        webviz/tomas_wrapper.py (SOArm100Controller)
  L7 嘴 (Mouth)        → S-Bridge MetaQuery (why_this_action, audit_snap, journey)
                        agent/s_bridge.py
  L8 复盘 (Review)     → DPO/LoRA (ψ-anchor preference training) + Evolution
                        core/psi_lora.py
                        agent/psi_anchor.py (should_trigger_evolution)

This module provides a runtime registry that maps layers to their
implementing modules, enabling introspection and cross-layer coordination.

Author: MuJoCo-Bench-IDO v0.16.25 — P2 Feature
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class LayerInfo:
    """Information about a cognitive layer.

    Attributes:
        id: Layer ID (L0-L8).
        name_cn: Chinese name (心脏, 大脑, etc.).
        name_en: English name.
        biological_analogue: Biological system this layer models.
        modules: List of code module paths implementing this layer.
        description: What this layer does.
        active: Whether the layer is currently active in the system.
    """
    id: str = ""
    name_cn: str = ""
    name_en: str = ""
    biological_analogue: str = ""
    modules: List[str] = field(default_factory=list)
    description: str = ""
    active: bool = False


# ── Nine-Layer Definitions ──

NINE_LAYERS: List[LayerInfo] = [
    LayerInfo(
        id="L0",
        name_cn="心脏",
        name_en="Heart",
        biological_analogue="Cardiovascular system — constant rhythm, vital signs",
        modules=["core/t_processor.py"],
        description=(
            "T-Processor: η-ALU computes GaussEx residual, Ψ-Check evaluates "
            "physical constraints, κ-Snap FIFO buffers audit trail. "
            "Runs at 100Hz constant rhythm — the heartbeat of the system."
        ),
        active=True,
    ),
    LayerInfo(
        id="L1",
        name_cn="大脑",
        name_en="Brain",
        biological_analogue="Cerebral cortex — reasoning, language, planning",
        modules=[
            "webviz/tomas_wrapper.py (VLAAdapter, OpenVLAAdapter, OctoAdapter, Pi0Adapter)",
            "agent/s_bridge.py (ask_why_llm)",
        ],
        description=(
            "LLM/VLA models (OpenVLA-7B, Octo-93M, π₀) generate action plans "
            "from vision + language. S-Bridge LLM attribution translates "
            "κ-Snap audit trail into natural language causal explanations."
        ),
        active=True,
    ),
    LayerInfo(
        id="L2",
        name_cn="骨架",
        name_en="Skeleton",
        biological_analogue="Skeletal system — structure, support, agent framework",
        modules=["agent/mujoco_ido_agent.py"],
        description=(
            "Agent framework: Hybrid-SAC, DreamerV3, IDO Agent. "
            "Provides the structural backbone for decision-making, "
            "connecting perception (L4) to action (L6)."
        ),
        active=True,
    ),
    LayerInfo(
        id="L3",
        name_cn="性格",
        name_en="Personality",
        biological_analogue="Limbic system — emotions, instincts, temperament",
        modules=["agent/pre_affect.py", "agent/safe_fuse.py"],
        description=(
            "PreAffect: intrinsic signals (curiosity, boredom, anxiety) that "
            "bias decision-making. SafeFuse: graded response (NORMAL/WARNING/"
            "BLOCK/INFO) that modulates action safety based on PreAffect signals."
        ),
        active=True,
    ),
    LayerInfo(
        id="L4",
        name_cn="感知",
        name_en="Perception",
        biological_analogue="Sensory system — vision, touch, proprioception",
        modules=[
            "webviz/tomas_wrapper.py (CAMKit dual cameras)",
            "core/kappa_snap_tokenizer.py",
        ],
        description=(
            "CAMKit: dual camera simulation (top_cam + wrist_cam) for VLA input. "
            "KappaSnapTokenizer: encodes κ-Snap audit trail as special tokens "
            "for VLA/LLM context — the agent 'perceives' its own causal history."
        ),
        active=True,
    ),
    LayerInfo(
        id="L5",
        name_cn="学识",
        name_en="Knowledge",
        biological_analogue="Hippocampus + cortex — memory, learning, skills",
        modules=["agent/s_bridge.py (learn_skill, get_skill_bank)", "core/gel_loss.py"],
        description=(
            "S-Bridge skill bank: learns successful decision patterns from "
            "episodes and stores them as SkillEntry objects. EML-SemZip: "
            "reweights RT-X training data based on IC values."
        ),
        active=True,
    ),
    LayerInfo(
        id="L6",
        name_cn="手脚",
        name_en="Hands/Feet",
        biological_analogue="Motor system — muscles, tendons, actuators",
        modules=[
            "webviz/tomas_wrapper.py (PsiAnchorGate, SOArm100Controller)",
            "core/pg_gate.py",
            "core/three_body.py (PhysicalBody)",
        ],
        description=(
            "C-Gate: ψ-Anchor physical constraints (MAX_TORQUE, MAX_VELOCITY, "
            "ZMP, ENERGY_DRIFT, NO_SPILL, MAX_GRIP_FORCE). PG-Gate: sentient "
            "finger protection. SO-ARM100 controller: pick-and-place execution."
        ),
        active=True,
    ),
    LayerInfo(
        id="L7",
        name_cn="嘴",
        name_en="Mouth",
        biological_analogue="Language center — communication, expression",
        modules=["agent/s_bridge.py (why_this_action, audit_snap, journey_timeline)"],
        description=(
            "S-Bridge MetaQuery: four interfaces for self-attribution and "
            "communication — WHY_THIS_ACTION, AUDIT_SNAP, LEARN_SKILL, "
            "JOURNEY_TIMELINE. The system 'speaks' about its own decisions."
        ),
        active=True,
    ),
    LayerInfo(
        id="L8",
        name_cn="复盘",
        name_en="Review",
        biological_analogue="Prefrontal cortex — metacognition, self-improvement",
        modules=[
            "core/psi_lora.py (PsiLoRATrainer)",
            "agent/psi_anchor.py (should_trigger_evolution, decide_evolution_policy)",
            "core/hg_pinn.py",
        ],
        description=(
            "DPO/LoRA: trains ψ-compliance preferences from audit trail. "
            "ψ-Anchor evolution: decides when to trigger self-improvement "
            "(light/freeze policy). HG-PINN: Hamiltonian-guided action head "
            "that conserves energy by construction."
        ),
        active=True,
    ),
]


class NineLayerRegistry:
    """Runtime registry for the nine-layer cognitive architecture.

    Provides:
      1. Layer lookup (by ID, name, or biological analogue)
      2. Module-to-layer mapping (which layer implements a given module)
      3. Cross-layer coordination (e.g., L4 perception feeds L1 brain)
      4. Health monitoring (which layers are active/healthy)

    Usage:
        registry = NineLayerRegistry()
        registry.print_architecture()  # Pretty-print all layers
        layer = registry.get_layer("L0")  # Get layer by ID
        modules = registry.get_modules_for_layer("L6")  # Get implementing modules
    """

    VERSION: str = "v0.16.25"

    def __init__(self) -> None:
        self._layers: Dict[str, LayerInfo] = {l.id: l for l in NINE_LAYERS}
        self._module_map: Dict[str, str] = {}  # module_path → layer_id
        for layer in NINE_LAYERS:
            for module in layer.modules:
                self._module_map[module] = layer.id

    def get_layer(self, layer_id: str) -> Optional[LayerInfo]:
        """Get layer info by ID (L0-L8)."""
        return self._layers.get(layer_id)

    def get_all_layers(self) -> List[LayerInfo]:
        """Get all nine layers."""
        return list(self._layers.values())

    def get_modules_for_layer(self, layer_id: str) -> List[str]:
        """Get implementing modules for a layer."""
        layer = self._layers.get(layer_id)
        return layer.modules if layer else []

    def get_layer_for_module(self, module_path: str) -> Optional[str]:
        """Get which layer a module belongs to."""
        return self._module_map.get(module_path)

    def get_active_layers(self) -> List[LayerInfo]:
        """Get all active layers."""
        return [l for l in self._layers.values() if l.active]

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get a summary of the architecture."""
        return {
            "total_layers": len(self._layers),
            "active_layers": sum(1 for l in self._layers.values() if l.active),
            "total_modules": len(self._module_map),
            "layers": {
                lid: {
                    "name_cn": l.name_cn,
                    "name_en": l.name_en,
                    "active": l.active,
                    "n_modules": len(l.modules),
                }
                for lid, l in self._layers.items()
            },
        }

    def print_architecture(self) -> str:
        """Pretty-print the nine-layer architecture."""
        lines = [
            "=" * 70,
            "IDO/TOMAS 九层认知架构 (Nine-Layer Cognitive Architecture)",
            "=" * 70,
        ]
        for layer in NINE_LAYERS:
            status = "✅" if layer.active else "❌"
            lines.append(f"\n{status} {layer.id} {layer.name_cn} ({layer.name_en})")
            lines.append(f"   生物类比: {layer.biological_analogue}")
            lines.append(f"   功能: {layer.description[:100]}...")
            lines.append(f"   实现模块:")
            for mod in layer.modules:
                lines.append(f"     • {mod}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
