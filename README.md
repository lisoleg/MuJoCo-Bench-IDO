# MuJoCo-Bench-IDO

IDO/TOMAS Architecture Upgraded to MuJoCo Continuous Physics Control Domain.

**Current Version: v0.17.2** — TOMAS Agent Deploy API + VLA Loader + eta Computation Fix + Welding Viewer Port Lifecycle (2026-07-06)

## Overview

This project upgrades the ARC discrete-symbol solver (tomas-arc3-solver) IDO/TOMAS architecture to the MuJoCo continuous physics control domain, preserving the IDO Harness philosophy:

| ARC Discrete | MuJoCo Continuous |
|---|---|
| Pixel grid | `mjData.qpos/qvel/actuator_force/sensor` |
| GaussEx residual η = pixel diff | Continuous state distance to Goal-EML coset squared distance |
| Noether-Check = Trigger prune | Physics conservation check (torque<=limit, energy no phantom increase, self-collision reject) |
| NARLA macro = discrete tile macro | Motor Primitive (IC-Value gated) |
| Oracle Replay = known trajectory replay | Expert Demonstration Replay |

## Nine-Layer Cognitive Architecture (v0.17.x)

| Layer | Biological Analogue | Modules |
|-------|---------------------|---------|
| L0 Heart | T-Processor (eta-ALU + psi-Checker + kappa-Snap FIFO) |
| L1 Brain | VLA (OpenVLA/Octo/pi0) + LLM Attribution |
| L2 Skeleton | Agent (IDOMuJoCoAgent + TaskPDControllers + TOMASMuJoCoWrapper) |
| L3 Personality | PreAffect + SafeFuse |
| L4 Perception | CAMKit (dual camera) + KappaSnapTokenizer |
| L5 Knowledge | SkillBank + EML-SemZip |
| L6 Hands/Feet | PsiAnchorGate (ZMP+Energy) + PG-Gate + HardPhysicsGate |
| L7 Mouth | S-Bridge (MetaQuery + LLM Attribution) |
| L8 Review | DPO + Evolution (psi-LoRA) |

## Project Structure

```
MuJoCo-Bench-IDO/
├── agent/
│   ├── __init__.py
│   ├── mujoco_ido_agent.py          # IDOMuJoCoAgent + MotorPrimitives
│   ├── task_pd_controllers.py       # Per-task PD controllers (v0.5.0+)
│   ├── hybrid_sb3_ido_agent.py      # Hybrid IDO + SB3 (PPO/SAC) agent
│   ├── hybrid_dreamer_ido_agent.py  # Hybrid IDO + DreamerV3 agent
│   ├── tomas_mujoco_wrapper.py      # TOMAS MuJoCo env wrapper (v0.17.0+)
│   ├── tomas_deploy.py              # TOMAS Agent deployment orchestrator (v0.17.1+)
│   ├── footstep_planner.py          # Footstep trajectory planner (v0.17.0+)
│   ├── failure_attribution.py       # TOMAS failure attribution prompt
│   ├── psi_anchor.py                # Psi-Anchor gate
│   ├── welding_psi_anchor.py        # Welding safety Psi-Anchor
│   ├── welding_controller.py        # Welding PID controller
│   ├── safe_fuse.py                 # SafeFuse graded safety
│   └── s_bridge.py                  # S-Bridge: kappa-Snap audit + LLM attribution
├── core/
│   ├── __init__.py
│   ├── goal_eml_mj.py               # GoalEML dataclass + task factory functions
│   ├── kappa_snap_mj.py             # GaussEx residual eta computation
│   ├── noether_check_mj.py          # Physics Noether-Check (Energy/Force/Collision)
│   ├── kappa_snap_tokenizer.py      # kappa-Snap -> token encoding for VLA/LLM
│   ├── t_processor.py               # T-Processor: eta-ALU + psi-Checker + kappa-Snap FIFO
│   ├── three_body.py                # Three-Body: Virtual->Software->Physical
│   ├── hg_pinn.py                   # Hamiltonian-Guided PINN + HardPhysicsGate (PG-Gate)
│   ├── psi_lora.py                  # psi-Anchor LoRA DPO Preference Trainer
│   ├── nine_layer.py                # Nine-Layer L0-L8 Mapping Registry
│   ├── gel_loss.py                  # GEL auxiliary loss
│   ├── octonion_ops.py              # Octonion non-associative algebra
│   ├── welding_eml_distillation.py  # EML octonion distillation network
│   ├── welding_process_proxy.py     # Welding physics formula proxy
│   ├── welding_sensors.py           # 7-type multimodal sensors
│   └── tomas_welding_axioms.py      # 7 TOMAS welding axioms
├── webviz/
│   ├── server.py                    # FastAPI + mjviser 3D viewer (v0.17.2)
│   ├── dashboard.html               # Web dashboard with architecture panel
│   ├── tomas_deploy_api.py          # TOMAS deploy API endpoints (v0.17.1+)
│   ├── vla_loader.py                # VLA model loader (OpenVLA/Octo/pi0) (v0.17.1+)
│   ├── tomas_wrapper.py             # TOMAS wrapper + VLA adapters + DemoVLAAdapter
│   ├── user_manual.html             # Interactive user manual
│   ├── mujoco_docs_cn.html          # Chinese docs page
│   ├── run_webviz.py                # Server launch helper
│   └── ws_regression_test.py        # WebSocket regression tests
├── benchmarks/
│   ├── __init__.py
│   ├── run_mujoco_bench.py          # Main benchmark runner
│   ├── run_tomas_eval.py            # TOMAS end-to-end evaluation (v0.17.0+)
│   ├── welding_eval.py              # Real welding baseline eval engine (v0.4.0+)
│   ├── welding_compare.py           # Welding comparison (legacy)
│   ├── evaluate_vs_baseline.py      # IDO vs PPO/SAC/TD-MPC2 comparison
│   ├── train_baselines.py           # SB3 baseline training scripts
│   ├── full_benchmark_1000.py       # Full 1000-step benchmark
│   ├── compare_hybrid.py            # Hybrid agent comparison
│   ├── tomas_eval_report.json       # Latest TOMAS eval results
│   └── results/
├── envs/
│   ├── __init__.py
│   ├── welding_env.py               # WeldingEnv (4-axis welding robot)
│   └── assets/
│       ├── mujoco_weld_robot.xml    # Welding robot MuJoCo scene
│       └── so_arm100_mujoco_ido.xml # SO-ARM100 MuJoCo scene (v0.17.0+)
├── config/
│   └── psi_anchor_defaults.yaml     # SO-ARM100 psi-Anchor config (v0.17.0+)
├── tools/
│   ├── hetero_benchmark.py          # Heterogeneous GPU vs GPU+T-Proc benchmark
│   ├── tproc_cim_simulator.py       # CIM memristor crossbar simulator
│   ├── qa_data_health.py            # Welding data quality QA tool
│   └── wps_pqr_generator.py         # WPS/PQR DOCX generator + kappa-Snap stats
├── hardware/                        # T-Proc hardware reference
│   ├── kintex_ultrascale_pins.xdc   # KCU105 pin constraints
│   ├── kria_k26_pin_constraints.xdc # Kria K26 pin constraints
│   └── README.md                    # Hardware architecture overview
├── docs/
│   ├── welding_robot_prd.md         # Welding robot PRD
│   ├── welding_architecture.md      # Welding system architecture
│   └── ...
├── papers/
│   ├── mujoco_bench_ido_validation.md  # Paper Appendix C (C.1-C.39)
│   └── mujoco_bench_ido_中文论文.md     # Chinese paper (sec.1-sec.10)
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_agent.py
│   ├── test_octonion.py             # 32 octonion algebra tests
│   ├── test_hetero_benchmark.py     # 51 hetero+CIM+EML tests
│   ├── test_welding_env.py          # 34 welding env tests
│   ├── test_welding_safety.py       # 30 safety gate tests
│   ├── test_welding_controller.py   # 23 controller tests
│   ├── test_welding_proxy.py        # 21 proxy model tests
│   ├── test_welding_integration.py  # 8 integration tests
│   └── test_tomas_wrapper.py        # TOMAS wrapper tests (v0.17.0+)
├── checkpoints/
│   └── sac_weld/
│       └── sac_weld_flat.zip        # SAC welding checkpoint (1.47MB)
└── .gitignore
```

## Quick Start

```bash
pip install dm_control mujoco numpy stable-baselines3

# Run benchmark
python benchmarks/run_mujoco_bench.py --task humanoid-reach --episodes 5

# Run TOMAS end-to-end evaluation on SO-ARM100
python benchmarks/run_tomas_eval.py

# Run tests (681 tests, 100% pass)
python -m pytest tests/ -v

# Start web dashboard
uvicorn webviz.server:app --host 0.0.0.0 --port 8080
# Dashboard: http://localhost:8080
# 3D Viewer: http://localhost:8081
# ARM100:   http://localhost:8091

# SAC welding training
python sac_weld_train.py --episodes 500 --steps 1000 --weld-type flat
```

## IDO Prophecy Verification Targets

| Prophecy | Metric | Target |
|----------|--------|--------|
| P1 | kappa-Snap directedness > BFS-discretize | IDO steps down 30%+ |
| P2 | Noether prevents reward hack | IDO NVR=0; PPO NVR>0 |
| P4 | Step Efficiency Ratio | SER>=1.2 (p<.05) |

## Key APIs (v0.17.2)

| Endpoint | Description |
|----------|-------------|
| `GET /api/architecture` | Nine-layer L0-L8 architecture mapping |
| `GET /api/t_processor` | T-Processor hardware spec (65k gates, 3.3mW) |
| `GET /api/cq` | Conscience Quotient (CQ) metrics |
| `GET /api/merkle` | kappa-Snap Merkle chain audit trail |
| `GET /api/arm100/status` | SO-ARM100 viewer status |
| `POST /api/arm100/start` | Start ARM100 pick-and-place viewer |
| `POST /api/tomas/deploy` | Deploy TOMAS Agent with VLA model (v0.17.1+) |
| `GET /api/tomas/deploy_status` | Check deployment progress (v0.17.1+) |
| `GET /api/tomas/deploy_result` | Get deployment eval results (v0.17.1+) |
| `GET /api/tomas/vla_available` | List available VLA models (v0.17.1+) |
| `POST /api/tomas/quick_eval` | Quick TOMAS evaluation (v0.17.1+) |
| `GET /api/welding/status` | Welding robot status |
| `POST /api/welding/start` | Start welding viewer (v0.4.1+ port lifecycle) |
| `POST /api/welding/stop` | Stop welding viewer + cleanup ports (v0.4.1+) |
| `GET /api/welding/trajectory` | Welding trajectory data |
| `GET /api/welding/quality` | Welding quality metrics |
| `GET /api/welding/safety` | Welding safety gate status |
| `GET /api/welding/sensors` | Welding sensor readings |
| `GET /api/welding/camera_info` | Welding camera info |

## VLA Model Support (v0.17.1+)

| Model | Size | VRAM | Status |
|-------|------|------|--------|
| openvla-7b | 7B | 16GB | Supported (requires GPU) |
| octo-base | 93M | 4GB | Supported |
| pi0-base | PaliGemma | 8GB | Supported |
| demo-vla | Built-in | 0GB | Default (no download needed) |

## Test Suite (681 tests, 100% pass)

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_octonion.py tests/test_hetero_benchmark.py -v
python -m pytest tests/test_tomas_wrapper.py -v

# CLI tools
python tools/hetero_benchmark.py --steps 100   # Heterogeneous benchmark
python tools/tproc_cim_simulator.py             # CIM energy comparison
python tools/qa_data_health.py --file data.h5   # Data quality check
```

## Version History (Recent)

| Version | Date | Highlights |
|---------|------|------------|
| v0.17.2 | 2026-07-06 | eta computation fix (avg_eta 1.463->0.103, -93%), welding viewer port lifecycle fix |
| v0.17.1 | 2026-07-06 | TOMAS deploy API + VLA loader + end-to-end eval on SO-ARM100 |
| v0.17.0 | 2026-07-05 | TOMAS Agent full-stack: wrapper + deploy + footstep + HardPhysicsGate + HG_PINN_Policy |
| v0.4.0 | 2026-07-04 | SLOS three-brain + PCM CIM + Psi-Anchor + kappa-Snap root cause + SAC welding |

## License

MIT
