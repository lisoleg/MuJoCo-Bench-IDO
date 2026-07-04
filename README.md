# MuJoCo-Bench-IDO

IDO/TOMAS Architecture Upgraded to MuJoCo Continuous Physics Control Domain.

**Current Version: v0.16.26** — Nine-Layer Cognitive Architecture + T-Processor + Three-Body + HG-PINN + ψ-LoRA

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
│   └── __init__.py
├── papers/
│   └── mujoco_bench_ido_validation.md  # Paper Appendix C
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_agent.py
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

## Key APIs (v0.16.26)

| Endpoint | Description |
|----------|-------------|
| `GET /api/architecture` | Nine-layer L0-L8 architecture mapping |
| `GET /api/t_processor` | T-Processor hardware spec (65k gates, 3.3mW) |
| `GET /api/cq` | Conscience Quotient (CQ) metrics |
| `GET /api/merkle` | κ-Snap Merkle chain audit trail |
| `GET /api/arm100/status` | SO-ARM100 viewer status |
| `POST /api/arm100/start` | Start ARM100 pick-and-place viewer |

## License

MIT
