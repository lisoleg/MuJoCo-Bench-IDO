# MuJoCo-Bench-IDO

IDO/TOMAS Architecture Upgraded to MuJoCo Continuous Physics Control Domain.

## Overview

This project upgrades the ARC discrete-symbol solver (tomas-arc3-solver) IDO/TOMAS architecture to the MuJoCo continuous physics control domain, preserving the IDO Harness philosophy:

| ARC Discrete | MuJoCo Continuous |
|---|---|
| Pixel grid | `mjData.qpos/qvel/actuator_force/sensor` |
| GaussEx residual η = pixel diff | Continuous state distance to Goal-EML coset squared distance |
| Noether-Check = Trigger prune | Physics conservation check (torque≤limit, energy no phantom increase, self-collision reject) |
| NARLA macro = discrete tile macro | Motor Primitive (IC-Value gated) |
| Oracle Replay = known trajectory replay | Expert Demonstration Replay |

## Project Structure

```
MuJoCo-Bench-IDO/
├── agent/
│   ├── __init__.py
│   └── mujoco_ido_agent.py     # IDOMuJoCoAgent + MotorPrimitives
├── core/
│   ├── __init__.py
│   ├── goal_eml_mj.py          # GoalEML dataclass + task factory functions
│   ├── kappa_snap_mj.py        # GaussEx residual η computation
│   └── noether_check_mj.py     # Physics Noether-Check (Energy/Force/Collision)
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
```

## IDO Prophecy Verification Targets

| Prophecy | Metric | Target |
|----------|--------|--------|
| P1 | κ-Snap directedness > BFS-discretize | IDO steps ↓ 30%+ |
| P2 | Noether prevents reward hack | IDO NVR=0; PPO NVR>0 |
| P4 | Step Efficiency Ratio | SER≥1.2 (p<.05) |

## License

MIT
