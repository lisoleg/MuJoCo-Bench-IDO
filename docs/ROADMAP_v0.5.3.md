# MuJoCo-Bench-IDO v0.5.3 — PHL/iLQR-MPC 启发步态生成优化路线图

> **日期**: 2026-07-02
> **版本**: v0.5.3
> **核心目标**: 从互联网调研中借鉴 MuJoCo locomotion 控制最佳实践，实现真正的步态生成 PD 控制器

---

## 0. v0.5.2 基准现状

| 任务 | avg_return | success | NVR/ep | η avg |
|------|-----------|---------|--------|-------|
| humanoid-stand | 6.63 | 100% | 46 | 2.72 |
| walker-walk | 11.26 | 0% | 0 | 22 |
| cheetah-run | 0.31 | 0% | 0 | 101 |
| reacher-easy | 93.3 | 100% | 0 | 0.002 |

**关键差距**: walker/cheetah η=22-101, success=0%。根因：PD 控制器只有增益+正弦振荡，缺少真正的步态生成逻辑。

---

## 1. 互联网调研发现

### 1.1 dm_control 奖励公式（源码级精确）

**Walker-walk reward** (`dm_control/suite/walker.py`):
```
standing = tolerance(torso_height, bounds=(1.2, inf), margin=0.6)
upright = (1 + torso_upright) / 2  # torso_upright = xmat['torso','zz']
stand_reward = (3*standing + upright) / 4
move_reward = tolerance(horizontal_velocity, bounds=(1.0, inf), margin=0.5, value_at_margin=0.5, sigmoid='linear')
total = stand_reward * (5*move_reward + 1) / 6
```
- 最大每步 reward = 1.0（standing=1, upright=1, move_reward=1）
- 需要: torso_height ≥ 1.2m, torso_upright ≥ 1.0, horizontal_velocity ≥ 1.0 m/s
- 每集 ~1000 步 → 最大 ~1000

**Cheetah-run reward** (`dm_control/suite/cheetah.py`):
```
total = tolerance(speed, bounds=(10, inf), margin=10, value_at_margin=0, sigmoid='linear')
```
- 速度 ≥ 10 m/s → reward = 1.0/step
- 线性映射：speed=3 → reward≈0.3/step

**Humanoid-stand reward** (`dm_control/suite/humanoid.py`):
```
standing = tolerance(head_height, bounds=(1.4, inf), margin=0.35)
upright = tolerance(torso_upright, bounds=(0.9, inf), sigmoid='linear', margin=1.9, value_at_margin=0)
stand_reward = standing * upright
small_control = tolerance(ctrl, margin=1, value_at_margin=0, sigmoid='quadratic').mean()
small_control = (4 + small_control) / 5
dont_move = tolerance(horizontal_velocity, margin=2).mean()
total = small_control * stand_reward * dont_move
```
- **关键**: reward 惩罚大 ctrl 值！ctrl 接近 0 → small_control ≈ 1.0
- **关键**: reward 惩罚水平移动！velocity ≈ 0 → dont_move ≈ 1.0

### 1.2 iLQR MPC 论文启发 (arxiv 2503.04613)

**Whole-Body MPC with MuJoCo** (CMU + DeepMind, 2025):
- iLQR + MuJoCo dynamics + 有限差分导数 → 实时 whole-body MPC
- Gait 作为 **soft cost residual**（不是硬约束）
- 残差项: Upright, Height, Position, Gait, Balance, Effort, Posture, Yaw
- 控制频率: iLQR ~50 Hz, TV-LQR feedback ~300 Hz
- 关节 PD 控制器跟踪 iLQR 输出的关节角度参考值
- 开源代码: https://github.com/johnzhang3/mujoco_mpc_deploy/

**核心映射**:
- 我们的 PD 控制器 = iLQR 的关节 PD 层（跟踪参考角度）
- 我们的 gait phase = iLQR 的 Gait residual（每足一个相位信号）
- 我们缺少 iLQR 的高层规划 → 用固定 gait 模式替代

### 1.3 LearningHumanoidWalking (GitHub)

- PPO + imitation learning
- swing_duration=0.75s, stance_duration=0.35s
- 3D velocity command: [yaw_vel, vx, vy]
- footstep planning for bipedal locomotion

### 1.4 dm_control 基准分数

- Walker-walk: CURL score ≈ 403 (D4PG 最高约 ~900)
- Cheetah-run: RL 基线 ~500-700
- Humanoid-stand: RL 基线 ~800-900

---

## 2. v0.5.3 优化方案

### 2.1 WalkerWalkPD — 2-phase 控制 (iLQR 启发)

```
Phase 1 (Recovery): torso_z < 1.0 or upright < 0.7
  → Joint-level PD toward standing pose
  → standing_targets = [0, -0.5, 0, 0, -0.5, 0]
  → kp=0.8, kd=0.3

Phase 2 (Walking gait): upright achieved
  → Sinusoidal gait with velocity feedback
  → right_push = max(sin(freq*t), 0)
  → left_push = max(-sin(freq*t), 0)
  → forward_bias = clip(vel_error * 0.15, -0.2, 0.4)
  → target: horizontal_velocity ≥ 1.0 m/s
```

### 2.2 CheetahRunPD — Bounding gait

```
Bounding gait: back/front legs alternate with quarter-phase offset
  → back_phase = sin(freq*t)
  → front_phase = sin(freq*t + pi/2)
  → bthigh positive = forward, bshin negative = extend
  → Mild torso pitch stabilization
  → target: speed ≥ 10 m/s (or at least >3 for reward>0.3)
```

### 2.3 HumanoidStandPD — Small control (dm_control reward-aligned)

```
KEY: reward = small_control * standing * upright * dont_move
  → Reduce ALL ctrl clips from [-0.3, 0.3] → [-0.08, 0.08]
  → Reduce ALL scaling factors by 5-10x
  → Add horizontal velocity damping (dont_move factor)
  → This alone should boost return from ~6.6 → >500
```

### 2.4 ee_pos Cartesian 修复 (PHL 启发)

```
For walker/cheetah WITHOUT to_target:
  → ee_pos = phys.named.data.xpos['torso', :]  (Cartesian world position)
  → NOT qpos[:3] (mixed position+angle, no physical meaning)
  → Makes η = ||torso_pos - target_pos||² decrease as body advances
```

---

## 3. 预期效果

| 任务 | 当前 avg_return | 目标 avg_return | 当前 η | 目标 η |
|------|---------------|----------------|--------|--------|
| humanoid-stand | 6.63 | >500 | 2.72 | <2 |
| walker-walk | 11.26 | >50 | 22 | <10 |
| cheetah-run | 0.31 | >30 | 101 | <50 |
| reacher-easy | 93.3 | >400 | 0.002 | <0.001 |

---

## 4. 调研参考文献

| 编号 | 来源 | URL | 核心启发 |
|------|------|-----|---------|
| R1 | dm_control walker.py | github.com/google-deepmind/dm_control | 精确奖励公式: standing*upright*(5*move+1)/6 |
| R2 | dm_control cheetah.py | 同上 | tolerance(speed, bounds=(10,inf)), linear sigmoid |
| R3 | dm_control humanoid.py | 同上 | small_control*standing*upright*dont_move |
| R4 | iLQR MPC paper | arxiv.org/abs/2503.04613 | Gait as soft cost residual, iLQR whole-body MPC |
| R5 | iLQR MPC code | github.com/johnzhang3/mujoco_mpc_deploy | C++ iLQR + Python GUI, real hardware deployment |
| R6 | LearningHumanoidWalking | github.com/rohanpsingh/LearningHumanoidWalking | PPO+imitation, swing=0.75s stance=0.35s |
| R7 | PHL (物理启发式学习) | 微信公众号文章 | 策略=代码, Walker2d闭环: 脚滑→patch→原地踏步→patch |
| R8 | SAI (超人类适应性智能) | 微信公众号文章 | Creative-Probe: η停滞→八元数非结合性→宏序列扰动 |
| R9 | TOMAS RSI 安全治理 | yb.tencent.com/s/zzvf6DZJpCpn | 理论框架(无locomotion实现), PG-Gate硬锚点 |

---

## 5. 实现优先级

1. **P0**: HumanoidStandPD ctrl 缩减 (最简单、预期收益最大: 6.6→500+)
2. **P0**: Walker/cheetah ee_pos = torso xpos (PHL Cartesian 修复)
3. **P1**: WalkerWalkPD 2-phase gait (recovery + oscillating)
4. **P1**: CheetahRunPD bounding gait
5. **P2**: 论文附录 C.21 (dm_control 奖励分析 + iLQR 启发)
6. **P2**: user_manual.html 更新到 v0.5.3
7. **P3**: GitHub 提交
