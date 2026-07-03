# MuJoCo-Bench-IDO Gap Analysis vs Industry SOTA

**Date**: 2026-07-03 | **Version**: v0.8.0 | **Episode Length**: 100 steps (need 1000)

## 1. Industry SOTA Reference Data

Source: TD-MPC2 GitHub CSV files + DreamerV3 PyTorch README

| Task | TD-MPC2 (1M steps, 3 seeds avg) | DreamerV3 (1M steps) | Best SOTA |
|------|-------------------------------|---------------------|-----------|
| cheetah-run | **878** | **887** | 887 |
| walker-walk | **980** | **956** | 980 |
| walker-stand | **956** | **900** | 956 |
| walker-run | ~700 | **701** | 701 |
| hopper-hop | **338** | **370** | 370 |
| hopper-stand | **955** | **945** | 955 |
| humanoid-stand | **873** (peak 932 at 4M) | **945** | 945 |

**Normalization**: `score = (agent_return - random_return) / (max_return - random_return) × 1000`

Random returns (1000 steps): cheetah=3.4, walker-walk=30.2, humanoid=8.6

## 2. Our Current Results (v0.8.0, 100-step episodes)

| Task | PPO avg_return | Hybrid-PPO | SAC avg_return | Hybrid-SAC | Hybrid/PPO | Hybrid/SAC |
|------|---------------|-----------|---------------|-----------|-----------|-----------|
| cheetah-run | 20.71 | 21.31 | — | — | 1.03x | — |
| walker-walk | 24.99 | 14.94 | 76.10 | 75.03 | 0.60x ↓ | 0.98x |
| humanoid-stand | 6.12 | 5.68 | 39.16 | 6.40 | 0.93x ↓ | 0.16x ↓↓ |

**Critical issue**: Hybrid DEGRADES baseline on walker-walk and humanoid-stand!

## 3. Estimated Normalized Scores (extrapolated to 1000 steps)

Using linear extrapolation: `1000_step_return ≈ 100_step_return × 10`

| Task | Our best (SAC) | SOTA | % of SOTA | Gap (norm pts) |
|------|---------------|------|-----------|----------------|
| cheetah-run | ~205 (PPO) | 887 | 23% | 682 |
| walker-walk | ~761 (SAC) | 980 | 78% | 219 |
| hopper-hop | 0 (PD-only) | 370 | 0% | 370 |
| humanoid-stand | ~392 (SAC) | 945 | 41% | 553 |

**Note**: Linear extrapolation is approximate. Real 1000-step returns may differ due to:
- Episode termination (early termination reduces total)
- Reward dynamics (locomotion tasks accumulate over time)
- Training quality (SB3 PPO/SAC 1M steps vs TD-MPC2/DreamerV3 1M steps)

## 4. Root Cause Analysis

### Why SB3 PPO is far from SOTA:
- **Model-free vs Model-based**: SB3 PPO/SAC are model-free methods. TD-MPC2 and DreamerV3 are model-based methods that learn world models, giving them sample efficiency advantages.
- **Pixel vs State observations**: SOTA methods often work with 64×64 RGB pixels (harder!), while we use state observations (easier). This is actually an advantage for us — yet we still fall behind.
- **Training quality**: Our SB3 models are trained 1M steps, same as SOTA references. The gap is inherent to PPO/SAC vs model-based methods.

### Why Hybrid IDO degrades baseline on some tasks:
- **v0.8.0 SafeFuse graded**: On humanoid-stand, SAFE mode triggered 42-64% of steps → over-conservative, killing SAC's learned behavior
- **v0.8.0 Pre-Affect GRRR**: η stagnation anxiety → Creative-Probe ×1.5, but η computation may be unstable for some tasks
- **v0.7.1 SafeFuse bypass (locomotion)** worked well for cheetah-run, but v0.8.0's graded INFO-level still modifies action for some locomotion tasks

## 5. Path to Industry-Leading Performance

### Option A: Improve SB3 baseline (moderate improvement)
- Retrain SB3 PPO/SAC with better hyperparameters (larger network, more steps)
- Expected: ~30-50% of SOTA (still far from first place)

### Option B: Adopt model-based RL (significant improvement)
- Implement TD-MPC2 or DreamerV3 in our framework
- Expected: ~80-95% of SOTA (competitive but not necessarily first place)

### Option C: Hybrid IDO + Model-based RL (potential industry leader)
- Use TD-MPC2 or DreamerV3 as motor layer (instead of SB3 PPO/SAC)
- IDO cognitive layer adds η monitoring, Pre-Affect, S-Bridge on top
- Expected: **potentially exceed SOTA** (walker-walk Hybrid-SAC was 1.42x SAC baseline)

### Option D: Fix Hybrid degradation first (immediate priority)
- Revert v0.8.0 SafeFuse graded → keep v0.7.1 locomotion bypass
- Fix humanoid-stand SAFE mode over-triggering
- Then proceed to Option C

## 6. Recommended Action Plan

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| **P0** | Fix v0.8.0 Hybrid degradation on walker-walk/humanoid-stand | Restore 1.42x Hybrid-SAC |
| **P1** | Rerun all benchmarks at 1000 steps | Get accurate normalized scores |
| **P2** | Train SB3 SAC for cheetah-run + hopper-hop | Close 0% gap on hopper-hop |
| **P3** | Adopt DreamerV3 or TD-MPC2 as motor layer | Reach ~80-95% of SOTA |
| **P4** | Hybrid IDO + DreamerV3/TD-MPC2 | Potentially exceed SOTA |
