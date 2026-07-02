# MuJoCo-Bench-IDO v0.6.3 — WalkerWalkPD Torque-Actuator 修复 & WalkerStandPD 索引修正

> **日期**: 2026-07-03
> **版本**: v0.6.3
> **核心目标**: 修复 WalkerWalkPD/WalkerStandPD 的 3 个关键 bug，诊断 PD 控制器的根本局限性

---

## 0. v0.6.1 基准现状

| 任务 | η avg | Return avg | NVR/ep | success |
|------|-------|-----------|--------|---------|
| humanoid-stand | ~2.6 | 6.02 | 46 | 67% |
| walker-walk | ~1.82 | 25.13 | 0 | 0% |
| cheetah-run | ~9.84 | 5.00 | 0 | 0% |
| reacher-easy | ~0.002 | 93.3 | 0 | 100% |

**关键问题**: walker-walk/cheetah-run η 虽然大幅改善（25x/10x），但实际速度仍然极低（~0.2 m/s），PD 控制器无法让 walker 稳定站立。

---

## 1. WalkerWalkPD v0.6.3 — 3 个关键 Bug 修复

### Bug 1: Torque vs Position Actuator Confusion

**根因**: Walker 的 6 个 actuator 全部是纯力矩 (torque) actuator:
- `dyntype=0` (general actuator)
- `gainprm=[1, 0, ...]` (gain=1, force = ctrl × 1)
- `biasprm=[0, 0, ...]` (zero bias)
- `ctrlrange=[-1, 1]` (torque range in Newton-meters)

但 WalkerWalkPD/WalkerStandPD 之前的代码用关节角度范围（如 hip [-0.35, 1.75], knee [-2.62, 0]) 作为 ctrl clip bounds。这导致 ctrl 值被限制在远超过 [-1, 1] 的范围，实际上没有任何有效限制。

**修复**: 所有 Walker ctrl 统一 clip 到 [-1, 1]。

### Bug 2: WalkerStandPD qpos/qvel 索引错误

Walker model 结构 (diagnostic 确认):
- nq=9: qpos[0]=rootz(slide)=0, qpos[1]=rootx(slide), qpos[2]=rooty(hinge), qpos[3:9]=joint angles
- nv=9: qvel[0:3]=root velocities, qvel[3:9]=joint velocities
- nu=6: 6 torque actuators

WalkerStandPD 之前的 3 个索引错误:

| 错误代码 | 问题 | 正确代码 |
|----------|------|----------|
| `root_z = qpos[2]` | qpos[2]=rooty (pitch angle, NOT height!) | `root_z = xpos['torso','z']` |
| `upright via qpos[3], qpos[4]` | Walker 无 quaternion | `upright = xmat['torso',8]` (zz component) |
| `qvel_idx = i + 5` | Walker 有 3 个 root DOFs | `qvel_idx = i + 3` |

**诊断方法**:
- `qpos[0]` = rootz = 0 (2D walker convention, not actual height)
- 实际高度从 `xpos['torso',z]` 获取 (如 1.3000)
- `xmat['torso',8]` 是旋转矩阵 zz 组件，范围 [-1, 1]，-1=倒立，1=直立
- Walker 没有 quaternion 在 qpos 中，只有 3 个 root entries (rootz, rootx, rooty)

### Bug 3: Height PD 双向推力

**根因**: `height_error = target_height - torso_z`
- Walker 初始高度 ≈ 1.3 > target 1.2
- 产生 height_error = 1.2 - 1.3 = -0.1 → **向下推力**
- WalkerStandPD-style 控制器把 walker 从已经站起的位置**推回地面**

**修复**: `height_error = max(0, target_height - torso_z)` — one-sided:
- 只在 height < target 时推上
- 在 height > target 时 height_torque = 0（不推下）
- 仍然保留 velocity damping (`-height_kd * root_vz`)

---

## 2. WalkerWalkPD v0.6.3 策略

### Phase 1: Recovery (height < 1.0 or upright < 0.5)

不使用关节角度目标 — 初始状态随机变化导致角度目标不稳定。

使用固定力矩推 walker 站起:
```
ctrl[0] = recovery_hip_torque (0.5)   → push hip forward → lift torso
ctrl[1] = recovery_knee_torque (-0.3)  → bend knee → support weight
ctrl[2] = recovery_ankle_torque (0.2)  → push ankle forward → balance
ctrl[3:6] = same for left leg
```

加上 one-sided height PD + upright PD + velocity damping。

### Phase 2: Stabilize (upright ≥ 0.5 for 30 steps)

WalkerStandPD-style damping:
```python
joint_ctrl = -stabilize_kp * joint_vel * 0.1  # damping only, no target tracking
```

加上 one-sided height PD + upright PD。

### Phase 3: Walking gait (stabilized 30+ steps)

Full-sine oscillation + velocity feedback:
```
phase = step_counter / gait_freq
hip_ctrl = gait_amp * sin(phase) + forward_lean_bias
knee_ctrl = knee_amp * sin(phase + pi)  # anti-phase
ankle_ctrl = ankle_amp * sin(phase)
```

加上 one-sided height PD + upright PD + velocity PD。

**Fallback**: height < 0.8 or upright < 0.4 → reset to Phase 1。

---

## 3. PD 控制器局限性诊断

### WalkerStandPD 测试结果

修复索引后，WalkerStandPD 可以让 walker **短暂站起**:
- upright 峰值可达 0.98 (接近完全直立)
- 但无法维持稳定站立 — walker 在 ~1 秒内摔倒

### Torque sweep 测试

扫描固定力矩 (0.01 ~ 0.3)，确认没有任何简单固定力矩可以让 walker 稳定站立超过 1 秒。

### 根本原因

Walker 2D 动力学是**underactuated system**:
- 6 个力矩 actuator 控制 9 个 DOF
- 没有 root_y (pitch) actuator — torso orientation 只能通过 leg joints **间接**控制
- 简单 PD 控制本质上无法处理这种 indirect dynamics

### 诊断结论

这确认了"聪明的残废"诊断在更深层面上成立:
1. 不仅 IDO 的 open-loop PD vs PPO 有差距
2. **PD 控制本身**就有 inherent limitations on underactuated locomotion
3. IDO 的价值不在于 PD motor layer 的绝对性能
4. 而在于 η-mode + ψ-Anchor + SafeFuse + CQ 提供的**adaptivity/interpretability/conservation/conscience guarantees**

### 后续方向

Locomotion PD 控制器作为 IDO 认知循环的**初始化层**:
- 提供结构化的 recovery → stabilize → gait phases
- η-mode 可以跟踪 PD 的性能趋势
- 实际 locomotion 需要:
  1. SB3 PPO/SAC trained policies (Phase 2 baseline)
  2. Hybrid IDO+SB3 agent (Phase 3 — η-aware decision loop switching between PD fallback and SB3 policy)

---

## 4. QA 测试验证

- 292/292 tests pass (pytest, 7.41s)
- No regressions from WalkerWalkPD v0.6.3 refactor
- WalkerStandPD index fixes verified

---

## 5. 下一步计划

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | Hybrid IDO+SB3 agent benchmark 验证 | 待完成 |
| P1 | CheetahRunPD 大幅调优 (speed 0.2 → 10.0) | 待完成 |
| P1 | WalkerRunPD 调优 | 待完成 |
| P2 | Webviz dashboard η 模式显示 (point vs locomotion) | 待完成 |
| P2 | Hopper-hop locomotion η 任务添加 | 待完成 |
| P3 | 论文 §7.9-7.11 最终版本更新 | 本轮完成 |
