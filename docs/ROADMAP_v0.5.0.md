# MuJoCo-Bench-IDO — 实验与优化路线图
> **日期**: 2026-07-02
> **核心问题**: 不知道该干什么、怎么做实验、怎么持续优化、怎么评估是否达到物理AI合格水平

---

## 0. 残酷现状

先看数据，不要逃避：

| 指标 | IDO humanoid-stand | IDO reacher-easy | 期望 (合格线) |
|------|-------------------|-----------------|-------------|
| episode_return | 1.5e-12 | 0.0 | > 0 (任何正值) |
| η | 2.6 (不下降) | 9763 (极大) | < 1.0 且下降趋势 |
| NVR | 981/episode | 0 | ≈ 0 (大多数任务) |
| success_rate | 0% | 0% | > 10% (最低合格线) |
| steps_to_goal | 1000 (跑完没达标) | 30 (极短) | < 500 |

**结论**: IDO 当前本质上 ≈ Random Agent。episode_return ≈ 0 意味着 IDO 没有产生任何有效控制信号。

根因不在 η 计算或评估框架（这些已修复），而在 **Motor 层**——5 个固定动作 (step_forward/step_left/step_right/squat/torque_explore) 对 dm_control 任务根本不产生 goal-directed progress。

---

## 1. 优化循环: "跑→看→改→再跑"

物理AI实验不是一次性的，而是迭代循环：

```
跑实验 → 看数据 → 诊断瓶颈 → 改代码 → 再跑 → 再看 → ...
```

### 1.1 跑什么实验

每次实验必须回答一个具体问题：

| 实验编号 | 问题 | 方法 | 预期结果 |
|---------|------|------|---------|
| E1 | IDO 当前能跑多好？ | `python benchmarks/run_mujoco_bench.py --task humanoid-stand --episodes 5` | episode_return, η趋势, NVR, success |
| E2 | Random agent 能跑多好？ | evaluate_vs_baseline.py --task humanoid-stand --baselines random | floor线: return ≈ 0 |
| E3 | PPO 能跑多好？ | train_baselines.py --task humanoid-stand --algo ppo --steps 1M | 期望: return > 0 |
| E4 | SAC 能跑多好？ | train_baselines.py --task humanoid-stand --algo sac --steps 1M | 期望: return > PPO |
| E5 | 改了Motor后IDO能跑多好？ | 修改后重新跑E1 | 期望: return > Random, η下降 |

### 1.2 看什么数据

每次实验后检查 5 个核心指标：

| 指标 | 含义 | 合格线 | 优秀线 |
|------|------|--------|--------|
| **episode_return** | dm_control累计reward（P0-1已修复） | > Random return | > SAC return |
| **η 趋势** | κ-Snap残差是否下降 | η下降 > 10% | η下降 > 50% 且收敛 |
| **NVR** | Noether违规率 | < Random NVR | ≈ 0 |
| **success_rate** | per-task goal-achieved (P0-2已修复) | > 10% | > 80% |
| **episode_length** | 活多久不摔倒 | > Random | > SAC |

### 1.3 诊断瓶颈

看数据 → 问"为什么这指标不行？" → 定位代码层：

| 症状 | 诊断 | 代码层 | 改什么 |
|------|------|--------|--------|
| return ≈ 0 | 没有goal-directed action | Motor层 | MotorPrimitives |
| η 不下降 | 距目标没进步 | κ-Snap + GoalEML | target_pos + 权重 |
| NVR 高 | 物理守恒被破坏 | Noether门 | 门阈值 + 检查方式 |
| success_rate 0% | goal判断太严/太宽 | TASK_SUCCESS_CRITERIA | 阈值调整 |

---

## 2. Phase 1: Motor层重构 (最紧急, 2-3天)

这是唯一能让 IDO 从 ≈Random 变成 ≈合格 的改动。

### 2.1 问题根因

当前 MotorPrimitives 只有 5 个固定动作：
```python
(self.step_forward,   0.70),   # ctrl[:2] += 0.3 — 对humanoid无效
(self.step_left,      0.65),   # ctrl[0] -= 0.3 — 对所有任务都粗暴
(self.step_right,     0.65),   # ctrl[0] += 0.3
(self.squat,          0.50),   # ctrl[1] -= 0.2 — 对非humanoid无效
(self.torque_explore, 0.40),   # uniform(-0.1,0.1) noise — ≈random
```

**问题**: 这些动作对 27 个 dm_control 任务没有一个产生有效的 goal-directed progress。原因是：
1. ctrl[:2] 只覆盖前2个关节，但 humanoid 有21个关节、reacher有2个关节（这恰好OK）、walker有6个关节
2. +0.3/-0.3 是固定偏移，不根据目标方向调整
3. 对 reacher 来说 step_forward 的 ctrl[:2] += 0.3 确实能移动臂，但方向不对目标

### 2.2 修复方案

**方案A: Task-specific PD controller library** (推荐)

为每个 dm_control 任务实现一个专门的 PD 控制器，替换固定动作库：

```python
TASK_CONTROLLER_MAP = {
    'humanoid-stand': HumanoidStandPD,   # 站直: root高度PD + 关节平衡
    'humanoid-walk': HumanoidWalkPD,     # 走路: 步态生成 + 朝目标方向
    'reacher-easy': ReacherTargetPD,     # 到目标: 2关节角度PD
    'walker-walk': WalkerWalkPD,         # 行走: 速度控制 + 站直
    'cheetah-run': CheetahRunPD,         # 奔跑: 前向速度 + 关节时序
    'hopper-stand': HopperStandPD,       # 单腿站: 高度PD + 关节稳定
    'cartpole-balance': CartpoleBalancePD, # 平衡杆: 角度PD
    'finger-turn_easy': FingerTurnPD,    # 旋转手指: 目标角度PD
    'fish-swim': FishSwimPD,             # 游泳: 朝目标方向 + 站直
    # ... 其余任务类似
}
```

每个 PD 控制器需要：
- 从 dm_control task 源码提取 **真实 target**（位置、角度、速度等）
- 用 **观测空间** 中的 relevant dimensions 计算 error
- PD增益 **per-task 调优**（不是全局 KP=30 KD=3）

**方案B: 通用 PD + dm_control reward 引导** (次优)

保留通用结构，但让 Motor 层利用 dm_control reward 信号做 action selection：

```python
# 在 choose_action 中增加 reward feedback
if timestep.reward > self._prev_reward:
    # 上一步action导致reward上升 → 继续那个action方向
    ctrl = self._prev_ctrl * 1.1  # 加强
else:
    # reward下降 → 反方向或换动作
    ctrl = -self._prev_ctrl * 0.5  # 减弱/反向
```

这个方案更简单但不那么 IDO-style。

### 2.3 具体步骤

1. **读取 dm_control task 源码**：`dm_control/suite/humanoid.py`, `reacher.py`, `walker.py` 等
   - 提取每个任务的 `get_reward()` 和 `get_observation()` 定义
   - 找到真正的 goal (target_pos, target_angle, target_velocity)
   
2. **实现 TASK_CONTROLLER_MAP**：每任务一个 PD controller class
   - `HumanoidStandPD`: root_xpos[2] PD → 站直 + joint position PD → 稳定姿态
   - `ReacherTargetPD`: to_target direction → 2关节角度PD → arm朝目标
   - `WalkerWalkPD`: speed PD → 前向速度 + upright PD → 不倒
   
3. **修改 IDOMuJoCoAgent.choose_action()**: 
   - 根据 task_name 选择对应 controller
   - controller 生成 action → IDO 层仍做 κ-Snap/Noether 判断是否执行
   
4. **修改 GoalEML**: 从 dm_control 真实 goal 构建 target_pos

### 2.4 验证

跑 E1 实验 (humanoid-stand 5 episodes)，期望：
- episode_return 从 ≈0 变为 > 0.1 (至少比Random好)
- η 开始有下降趋势
- success_rate > 0%

---

## 3. Phase 2: Baseline训练 + 水平标定 (3-5天)

### 3.1 训练 baseline

PPO/SAC 训练已在后台启动。完成后记录水平标定数据：

| 方法 | humanoid-stand | walker-walk | reacher-easy | cheetah-run | ... |
|------|---------------|-------------|-------------|-------------|-----|
| Random | ~0 | ~0 | ~0 | ~0 | ... |
| PPO (1M) | ? | ? | ? | ? | ... |
| SAC (1M) | ? | ? | ? | ? | ... |
| IDO (Phase 1后) | ? | ? | ? | ? | ... |

**合格线**: IDO return > Random return (最低要求)
**优秀线**: IDO return > SAC return (需要 Phase 3 的认知层对接)

### 3.2 水平标定表

每个任务建立3条线：

| 线 | 含义 | 用途 |
|----|------|------|
| 地板 (floor) | Random agent return | IDO 至少要超过这条线 |
| 天花板 (ceiling) | SAC 1M return | IDO 最终目标 |
| 当前 (current) | IDO 最新 return | 追踪迭代进度 |

---

## 4. Phase 3: IDO认知层对接 (5-7天)

Motor 层修好后，IDO 的认知层（κ-Snap + Noether + ψ-Anchor）才能真正发挥作用。

### 4.1 κ-Snap + dm_control reward 联合信号

当前 η 只反映物理距离（pos_err, tilt_err, energy_excess, vel_mag），不反映 task progress。

**改进**: η 应结合 dm_control reward：

```python
# 联合η = κ-Snap残差 + reward惩罚
eta_combined = eta_kappa + w_reward * (1.0 - normalized_reward)
```

当 reward 上升（做得更好）→ eta_combined 下降 → κ-Snap 说"更近目标了"。

### 4.2 Noether门 → per-task守恒约束

当前 3 个通用守恒门（能量/力矩/碰撞）对 humanoid 粗暴：981 violations/episode。

**改进**: 为每个任务定制守恒约束：
- humanoid-stand: 只检查总能量不爆炸 + 不自碰撞（不检查力矩，因为站立需要力矩）
- reacher-easy: 检查关节力矩不超限（不需要检查碰撞，reacher没有碰撞体）
- cartpole-balance: 检查角动量守恒（不检查线动量）

### 4.3 GoalEML → dm_control真实目标

当前很多任务的 `target_pos` 是零向量或默认值，不是 dm_control 真实目标。

**改进**: 从 dm_control task 的 `observation_spec()` 和 `get_reward()` 提取真实 goal：
- reacher: `timestep.observation['to_target']` → 目标方向
- humanoid-stand: upright = root_z > threshold → 站直
- walker-walk: velocity > target_speed → 走到目标速度

---

## 5. Phase 4: 论文实验矩阵 (5天)

当 Phase 1-3 完成后，运行完整实验矩阵：

| 维度 | 内容 |
|------|------|
| **任务** | 8 个核心 dm_control 任务 |
| **方法** | Random, PPO, SAC, IDO |
| **指标** | episode_return, η趋势, NVR, success_rate |
| **纵向** | SIP-Bench T0→T1→T2 演化验证 |
| **扰动** | 噪声注入 ±10%/±20%/±50% 观测噪声 |
| **迁移** | train on A, eval on B (跨任务泛化) |

产出论文级表格：

| Task | Random | PPO | SAC | IDO | IDO η↓ | IDO NVR |
|------|--------|-----|-----|-----|---------|---------|
| humanoid-stand | ... | ... | ... | ... | ... | ... |
| walker-walk | ... | ... | ... | ... | ... | ... |
| ... | | | | | | |

---

## 6. Phase 5: 物理AI最高水平 (持续)

**合格标准**: IDO return > Random, η下降, NVR < Random
**优秀标准**: IDO return > SAC, η收敛, NVR ≈ 0, 跨任务迁移

理论保证（VG-Pair + Pick's Theorem）：
- VG-Pair ≠ GAN: C-IPP theorem guarantees Soundness
- Pick's Theorem = Discrete Gauss-Bonnet: Pick-Check ↔ Noether-Check
- 这些理论提供的是 **结构保证**（不会比Random差），不是 **性能保证**（一定比SAC好）

要达到最高水平，还需要：
1. Motor层足够task-specific
2. κ-Snap足够反映真实progress
3. ψ-Anchor演化策略产生有效adaptation
4. 实验数据支撑所有claim

---

## 7. 今天该做什么 (立即行动)

1. **读取 dm_control task 源码** — 理解每个任务的真正目标
2. **实现 humanoid-stand PD controller** — 最简单的起始任务
3. **跑 E1 实验** — humanoid-stand 5 episodes, 看 return 是否 > 0
4. **对比 Random** — 确认 IDO 至少 > Random
5. **迭代** — 如果不够好，调整PD增益或添加新motor primitive

核心循环:
```
改Motor → 跑实验 → 看return → 如果>Random就继续优化 → 如果<=Random就回退重改
```

**不要**花时间优化 η 计算或 Noether 门——这些是认知层，在 Motor 层修好之前没有意义。
