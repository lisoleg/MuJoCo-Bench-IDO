# MuJoCo-Bench-IDO v0.5.4 Roadmap

## Version: v0.5.4 — dm_control Reward-Aligned PD Controller Optimization

## Key Discovery: dm_control small_control Floor = 0.8

From dm_control source code (humanoid.py):
```python
small_control = (4 + tolerance(norm(ctrl), margin=1, sigmoid='quadratic')) / 5
# small_control ∈ [0.8, 1.0] — FLOOR at 0.8 regardless of ctrl magnitude!
```

**Critical implication**: ctrl_clip=0.08 (v0.5.3) was far too conservative. The penalty for moderate controls is minimal (small_control ≈ 0.9 at ctrl_clip=0.3). The real bottleneck is **standing/upright** being too low because the humanoid can't actually stand up with ultra-small controls.

## dm_control Reward Formulas (Source-Level Confirmation)

### Walker-walk (walker.py)
```python
_STAND_HEIGHT = 1.2
_WALK_SPEED = 1  # m/s
_CONTROL_TIMESTEP = 0.025  # 25ms

standing = tolerance(torso_height(), bounds=(1.2, inf), margin=0.6)
upright = (1 + torso_upright()) / 2  # xmat['torso','zz']
stand_reward = (3*standing + upright) / 4
move_reward = tolerance(horizontal_velocity(), bounds=(1, inf), margin=0.5,
                        value_at_margin=0.5, sigmoid='linear')
reward = stand_reward * (5*move_reward + 1) / 6
# Max per step = 1.0 at standing=1, upright=1, speed≥1 m/s
# Episode: 1000 steps (25s × 0.025s)
```

### Cheetah-run (cheetah.py)
```python
_RUN_SPEED = 10  # m/s
_DEFAULT_TIME_LIMIT = 10  # seconds → ~400 steps

reward = tolerance(speed(), bounds=(10, inf), margin=10,
                  value_at_margin=0, sigmoid='linear')
# Max per step = 1.0 at speed ≥ 10 m/s
# Linear: speed=3 → reward=0.3, speed=5 → reward=0.5
# Episode starts with 200 stabilization steps (physics.step(nstep=200))
```

### Humanoid-stand (humanoid.py)
```python
_STAND_HEIGHT = 1.4  # head height above ground
_CONTROL_TIMESTEP = 0.025  # 25ms

standing = tolerance(head_height(), bounds=(1.4, inf), margin=0.35)
upright = tolerance(torso_upright(), bounds=(0.9, inf), sigmoid='linear',
                    margin=1.9, value_at_margin=0)
stand_reward = standing * upright
small_control = (4 + tolerance(norm(ctrl), margin=1, sigmoid='quadratic').mean()) / 5
# small_control ∈ [0.8, 1.0] — FLOOR at 0.8!
horizontal_velocity = center_of_mass_velocity()[[0, 1]]
dont_move = tolerance(horizontal_velocity, margin=2).mean()
reward = small_control * stand_reward * dont_move
# Max per step = 1.0 (standing=1, upright=1, small_control=1, dont_move=1)
# Episode: 1000 steps (25s × 0.025s)
```

## TOMAS RSI Article Analysis

The yb.tencent.com article (TOMAS RSI) provides:
- **PG-Gate**: Prevents self-modification that conflicts with hard anchors
- **MUS Dual-Storage**: Maintain multiple strategies, switch based on environment
- **κ-Snap**: Rollback mechanism for failed modifications
- **Noether-Check**: Physics validation after code changes

Key actionable insight: **Strategy Switching** — maintain multiple control strategies and switch between them based on state. This inspired our multi-phase approach.

## v0.5.4 Implementation Plan

### P0: HumanoidStandPD — ctrl_clip Optimization (CRITICAL)

**Root cause**: ctrl_clip=0.08 yields small_control≈0.99 but standing≈0.3, upright≈0.5 → reward≈0.15/step
**Fix**: ctrl_clip=0.3 yields small_control≈0.9 but standing≈1.0, upright≈1.0 → reward≈0.9/step (6x improvement!)

- ctrl_clip: 0.08 → 0.3 (Phase 2) / 0.4 (Phase 1 recovery)
- 2-phase control: Recovery (ctrl_clip=0.4, stronger PD) → Standing (ctrl_clip=0.2)
- Better standing pose targets: knees slightly bent, hips slightly posterior
- Expected: avg_return 8.65 → 500+

### P1: WalkerWalkPD — 3-Phase Gait Enhancement

- Phase 1 (Recovery): More aggressive PD (kp=2.0, kd=0.8)
- Phase 2 (Stabilize): Hold standing pose for 30 steps before gait
- Phase 3 (Walking): Stronger gait (amplitude 0.55, vel_kp 0.5, forward lean)
- Expected: avg_return 28 → 200+

### P1: CheetahRunPD — Stronger Bounding Gait

- Initial stabilization phase (30 steps)
- Much stronger amplitudes (thigh 1.0-1.2, shin 0.7-0.8)
- Faster gait_freq (24 rad/s)
- Stronger velocity feedback (vel_kp 0.8)
- Expected: avg_return 5 → 50+

## v0.5.3 → v0.5.4 Benchmark Progress

| Task | v0.5.3 avg_return | v0.5.4 avg_return (target) | Δ |
|------|-------------------|---------------------------|---|
| humanoid-stand | 8.65 | 500+ | ↑58x |
| walker-walk | 28.31 | 200+ | ↑7x |
| cheetah-run | 5.32 | 50+ | ↑9x |

## Reference Table

| ID | Source | Key Takeaway |
|----|--------|-------------|
| R10 | dm_control humanoid.py source | small_control floor = 0.8, ctrl_clip=0.3 is optimal |
| R11 | dm_control walker.py source | move_reward = linear, value_at_margin=0.5, need speed≥1 |
| R12 | dm_control cheetah.py source | 200-step stabilization at start, reward linear at margin=10 |
| R13 | TOMAS RSI (yb.tencent.com) | Strategy switching → multi-phase control |
| R14 | dm_control rewards.py | tolerance() function: gaussian default, linear/quadratic options |
