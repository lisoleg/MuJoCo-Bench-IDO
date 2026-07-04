# MuJoCo-Bench-IDO

IDO/TOMAS Architecture Upgraded to MuJoCo Continuous Physics Control Domain.

**Current Version: v0.4.0** — SLOS Three-Brain Architecture + PCM CIM + Psi-Anchor Gate + kappa-Snap Root Cause + MPW Tapeout + SAC Welding Training (章锋 2026-07-04 v2)

## Overview

This project upgrades the ARC discrete-symbol solver (tomas-arc3-solver) IDO/TOMAS architecture to the MuJoCo continuous physics control domain, preserving the IDO Harness philosophy:

| ARC Discrete | MuJoCo Continuous |
|---|---|
| Pixel grid | `mjData.qpos/qvel/actuator_force/sensor` |
| GaussEx residual η = pixel diff | Continuous state distance to Goal-EML coset squared distance |
| Noether-Check = Trigger prune | Physics conservation check (torque≤limit, energy no phantom increase, self-collision reject) |
| NARLA macro = discrete tile macro | Motor Primitive (IC-Value gated) |
| Oracle Replay = known trajectory replay | Expert Demonstration Replay |

## Nine-Layer Cognitive Architecture (v0.16.26)

| Layer | Biological Analogue | Modules |
|-------|-------------------|---------|
| L0 心脏 | Heart | T-Processor (η-ALU + ψ-Checker + κ-Snap FIFO) |
| L1 大脑 | Brain | VLA (OpenVLA/Octo/π₀) + LLM Attribution |
| L2 骨架 | Skeleton | Agent (IDOMuJoCoAgent + MotorPrimitives) |
| L3 性格 | Personality | PreAffect + SafeFuse |
| L4 感知 | Perception | CAMKit (dual camera) + KappaSnapTokenizer |
| L5 学识 | Knowledge | SkillBank + EML-SemZip |
| L6 手脚 | Hands/Feet | PsiAnchorGate (ZMP+Energy) + PG-Gate |
| L7 嘴 | Mouth | S-Bridge (MetaQuery + LLM Attribution) |
| L8 复盘 | Review | DPO + Evolution (ψ-LoRA) |

## Project Structure

```
MuJoCo-Bench-IDO/
├── agent/
│   ├── __init__.py
│   ├── mujoco_ido_agent.py     # IDOMuJoCoAgent + MotorPrimitives
│   └── s_bridge.py             # S-Bridge: κ-Snap audit + LLM attribution
├── core/
│   ├── __init__.py
│   ├── goal_eml_mj.py          # GoalEML dataclass + task factory functions
│   ├── kappa_snap_mj.py        # GaussEx residual η computation
│   ├── noether_check_mj.py     # Physics Noether-Check (Energy/Force/Collision)
│   ├── kappa_snap_tokenizer.py # κ-Snap → token encoding for VLA/LLM (P0)
│   ├── t_processor.py          # T-Processor: η-ALU + ψ-Checker + κ-Snap FIFO (P1)
│   ├── three_body.py           # Three-Body: Virtual→Software→Physical (P2)
│   ├── hg_pinn.py              # Hamiltonian-Guided PINN Action Head (P2)
│   ├── psi_lora.py             # ψ-Anchor LoRA DPO Preference Trainer (P2)
│   ├── nine_layer.py           # Nine-Layer L0-L8 Mapping Registry (P2)
│   └── gel_loss.py             # GEL auxiliary loss
├── webviz/
│   ├── server.py               # FastAPI + mjviser 3D viewer (v0.16.26)
│   ├── dashboard.html          # Web dashboard with architecture panel
│   └── tomas_wrapper.py        # TOMAS wrapper + VLA adapters + DemoVLAAdapter
├── benchmarks/
│   ├── __init__.py
│   ├── run_mujoco_bench.py     # Main benchmark runner
│   ├── evaluate_vs_baseline.py # IDO vs PPO/SAC/TD-MPC2 comparison
│   └── results/.gitkeep
├── envs/
│   ├── __init__.py
│   ├── welding_env.py          # WeldingEnv (6-axis welding robot)
│   └── assets/
│       └── mujoco_weld_robot.xml  # Welding robot MuJoCo scene
├── core/
│   ├── __init__.py
│   ├── goal_eml_mj.py          # GoalEML dataclass + task factory functions
│   ├── kappa_snap_mj.py        # GaussEx residual η computation
│   ├── noether_check_mj.py     # Physics Noether-Check (Energy/Force/Collision)
│   ├── kappa_snap_tokenizer.py # κ-Snap → token encoding for VLA/LLM (P0)
│   ├── t_processor.py          # T-Processor: η-ALU + ψ-Checker + κ-Snap FIFO (P1)
│   ├── three_body.py           # Three-Body: Virtual→Software→Physical (P2)
│   ├── hg_pinn.py              # Hamiltonian-Guided PINN Action Head (P2)
│   ├── psi_lora.py             # ψ-Anchor LoRA DPO Preference Trainer (P2)
│   ├── nine_layer.py           # Nine-Layer L0-L8 Mapping Registry (P2)
│   ├── octonion_ops.py         # Octonion non-associative algebra (v0.3.0)
│   ├── welding_eml_distillation.py  # EML octonion distillation network (v0.3.0)
│   ├── welding_process_proxy.py # Welding physics formula proxy (v0.3.0)
│   ├── welding_sensors.py      # 7-type multimodal sensors (v0.3.0)
│   ├── welding_eml_distill.py  # Pareto-optimal parameter search (v0.3.0)
│   ├── tomas_welding_axioms.py # 7 TOMAS welding axioms (v0.3.0)
│   └── gel_loss.py             # GEL auxiliary loss
├── tools/
│   ├── hetero_benchmark.py     # Heterogeneous GPU vs GPU+T-Proc benchmark (v0.3.0)
│   ├── tproc_cim_simulator.py  # CIM memristor crossbar simulator (v0.3.0)
│   ├── qa_data_health.py       # Welding data quality QA tool (v0.3.0)
│   └── wps_pqr_generator.py    # WPS/PQR DOCX generator + κ-Snap stats (v0.3.0)
├── hardware/                   # T-Proc hardware reference (v0.3.0)
│   ├── kintex_ultrascale_pins.xdc  # KCU105 pin constraints
│   ├── kria_k26_pin_constraints.xdc  # Kria K26 pin constraints
│   └── README.md               # Hardware architecture overview
├── docs/
│   ├── welding_robot_prd.md    # Welding robot PRD
│   ├── welding_architecture.md # Welding system architecture
│   ├── welding_delivery_summary.md  # Delivery summary (v0.2.0 + v0.3.0)
│   ├── welding_eml_annotation_schema.json  # EML annotation JSON Schema
│   ├── welding_sensor_selection.md  # 7-type sensor selection guide
│   └── welding_eml_reference.md  # EML reference documentation
├── papers/
│   ├── mujoco_bench_ido_validation.md  # Paper Appendix C (C.1-C.30)
│   └── mujoco_bench_ido_中文论文.md     # Chinese paper (§1-§9)
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_agent.py
│   ├── test_octonion.py        # 32 octonion algebra tests (v0.3.0)
│   ├── test_hetero_benchmark.py  # 51 hetero+CIM+EML tests (v0.3.0)
│   ├── test_welding_env.py     # 34 welding env tests
│   ├── test_welding_safety.py  # 30 safety gate tests
│   ├── test_welding_controller.py  # 23 controller tests
│   ├── test_welding_proxy.py   # 21 proxy model tests
│   └── test_welding_integration.py  # 8 integration tests
└── .gitignore
```

## Quick Start

```bash
pip install dm_control mujoco numpy

# Run benchmark
python benchmarks/run_mujoco_bench.py --task humanoid-reach --episodes 5

# Run tests
python -m pytest tests/ -v

# Start web dashboard
uvicorn webviz.server:app --host 0.0.0.0 --port 8080
# Dashboard: http://localhost:8080
# 3D Viewer: http://localhost:8081
# ARM100:   http://localhost:8091
```

## IDO Prophecy Verification Targets

| Prophecy | Metric | Target |
|----------|--------|--------|
| P1 | κ-Snap directedness > BFS-discretize | IDO steps ↓ 30%+ |
| P2 | Noether prevents reward hack | IDO NVR=0; PPO NVR>0 |
| P4 | Step Efficiency Ratio | SER≥1.2 (p<.05) |

## Key APIs (v0.3.0)

| Endpoint | Description |
|----------|-------------|
| `GET /api/architecture` | Nine-layer L0-L8 architecture mapping |
| `GET /api/t_processor` | T-Processor hardware spec (65k gates, 3.3mW) |
| `GET /api/cq` | Conscience Quotient (CQ) metrics |
| `GET /api/merkle` | κ-Snap Merkle chain audit trail |
| `GET /api/arm100/status` | SO-ARM100 viewer status |
| `POST /api/arm100/start` | Start ARM100 pick-and-place viewer |
| `GET /api/welding/status` | Welding robot status (v0.3.0) |

## v0.3.0 Test Suite (199 tests, 100% pass)

```bash
# Run all tests
python -m pytest tests/ -v

# Run v0.3.0 specific tests
python -m pytest tests/test_octonion.py tests/test_hetero_benchmark.py -v

# CLI tools
python tools/hetero_benchmark.py --steps 100   # Heterogeneous benchmark
python tools/tproc_cim_simulator.py             # CIM energy comparison
python tools/qa_data_health.py --file data.h5   # Data quality check
```

## License

MIT
