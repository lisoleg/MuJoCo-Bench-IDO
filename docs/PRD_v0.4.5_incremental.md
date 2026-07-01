# MuJoCo-Bench-IDO — 增量 PRD v0.4.5

> **产品经理**: 许清楚（Alice） — Product Manager
> **日期**: 2025-07-02
> **版本**: v0.4.5 incremental
> **触发问题**: 不知道如何评价 IDO 模型的性能，下一步需要什么改进

---

## 1. 现状分析

### 1.1 当前 IDO 已有的评估指标

| 指标类别 | 指标名 | 实现位置 | 数据来源 | 当前状态 |
|---------|--------|----------|----------|---------|
| **IDO 专属** | κ-Snap 残差 η (GaussEx residual) | `core/kappa_snap_mj.py` → `gauss_ex_residual()` | 4维加权平方距 (pos/ori/eng/vel) | ✅ 已实现，每步计算 |
| **IDO 专属** | Noether 违规率 NVR | `benchmarks/run_mujoco_bench.py` → `noether_check_mj()` | 三重守恒门 (能量/力矩/碰撞) | ✅ 已实现，reacher-easy NVR≡0 |
| **IDO 专属** | η 趋势方向 (descending/plateau/ascending) | `agent/psi_anchor.py` → `analyze_eta_trend()` | η_history 窗口内均值Δη | ✅ 已实现 |
| **IDO 专属** | ψ-Anchor δ_K 动态调整 | `agent/psi_anchor.py` → `adjust_delta_K()` | η趋势 → ×0.8/×1.2/不变 | ✅ 已实现 |
| **IDO 专属** | ψ-Anchor 演化策略 (light/freeze) | `agent/psi_anchor.py` → `decide_evolution_policy()` | η趋势 + epiplexity | ✅ 已实现 |
| **IDO 专属** | ψ-Anchor conservation_score | `agent/psi_anchor.py` → `inject_conservation_anchor()` | Noether门结果 → 1.0 - 0.3×n_violations | ✅ 已实现 |
| **IDO 专属** | Epiplexity | `agent/psi_anchor.py` → `compute_epiplexity()` | n_invariants × (1/δ_K) × log(max_energy) | ✅ 已实现 |
| **纵向评估** | SIP-Bench T0/T1/T2 | `benchmarks/run_mujoco_bench.py` → `run_sip_benchmark()` | 3阶段 + evolution rounds | ✅ 已实现 |
| **纵向评估** | Retention Gain | `run_sip_benchmark()` 内计算 | T0_avg_steps / T2_avg_steps | ✅ 已实现，但当前值=1.0（无法区分） |
| **纵向评估** | Stability Index | `run_sip_benchmark()` 内计算 | T2_std_steps / T0_std_steps | ✅ 已实现，但当前值=0.0 |
| **纵向评估** | Hesitation-RMSE | `core/kappa_snap_mj.py` → `FlowMatchingEtaPredictor` | η窗口内围绕局部均值震荡 | ✅ 已实现 |
| **纵向评估** | Retry-VOC | `core/kappa_snap_mj.py` → `FlowMatchingEtaPredictor` | η方向翻转频率 | ✅ 已实现 |
| **基础指标** | steps_to_goal | `run_single_episode()` 返回值 | 到达目标的步数（或max_steps） | ✅ 已实现，但全部≈2000（未到达目标） |
| **基础指标** | avg_return (dm_control reward) | `run_single_episode()` → `getattr(timestep, 'reward', 0.0)` | dm_control timestep.reward | ⚠️ 部分实现：仅取最终timestep的reward，非累计 |
| **基础指标** | 成功率 (goal achievement) | `run_single_episode()` → dist < pos_tol 判断 | ee到target距离 < tolerance | ⚠️ 仅humanoid-stand可检查right_hand，其他任务缺失 |
| **对比指标** | SER (Step-Efficiency Ratio) | `benchmarks/evaluate_vs_baseline.py` → `compute_ser()` | baseline_avg_steps / IDO_avg_steps | ✅ 框架已实现，但baseline未实际运行 |
| **对比指标** | Wilcoxon统计检验 | `ARCHITECTURE.md` 设计提到 | IDO vs baseline 显著性 | ❌ 未实现 |
| **对比指标** | IDO Prophecy Verification | `evaluate_vs_baseline.py` 内打印 | P1(NVR≡0)/P2(SER≥1.2)/P3(Baseline NVR>0) | ⚠️ 打印框架存在，但无实际baseline数据 |

### 1.2 当前 Baseline 适配器能力

| Baseline | 实现文件 | 控制能力 | 预测能力 | 实际可运行? | 关键局限 |
|---------|----------|---------|---------|-----------|---------|
| TD-MPC2 (model-based RL) | `baselines/tdmpc2_adapter.py` | ✅ `choose_action()`, `evaluate()` | ❌ 无η预测 | ❌ **不可运行** — tdmpc2 package未安装，所有调用fallback为random agent |
| Cosmos-Predict (world model) | `baselines/cosmos_predict_adapter.py` | ❌ 不是控制agent | ✅ `predict_future_state()`, `compare_eta_trajectory()` | ❌ **不可运行** — 需GPU+CUDA+cosmos_predict1，未安装 |
| PPO (SB3) | `evaluate_vs_baseline.py` → `register_baseline("ppo")` | ✅ `PPO.load()` | ❌ | ❌ **不可运行** — SB3未安装，fallback为random |
| SAC (SB3) | `evaluate_vs_baseline.py` → `register_baseline("sac")` | ✅ `SAC.load()` | ❌ | ❌ **不可运行** — SB3未安装，fallback为random |
| Random Agent | `evaluate_vs_baseline.py` → `get_random_agent()` | ✅ uniform random in [-1,1] | ❌ | ✅ **可运行** — 无外部依赖 |

**关键发现**: 当前所有非random baseline在实际环境中都无法运行（依赖包未安装），所有对比评估实际退化为 IDO vs Random Agent。这是最大的评估盲区。

### 1.3 dm_control 标准奖励函数

dm_control.suite 每个任务都有**任务特定的标准奖励函数**（在 `dm_control/suite/<domain>.py` 中定义），例如：

| 任务 | 标准奖励组成 | IDO是否使用? |
|------|-------------|-------------|
| humanoid-stand | upright bonus + standing bonus + small_control penalty | ❌ **未使用** — IDO只用η和Noether门，不累计dm_control reward |
| walker-walk | walking speed bonus + upright bonus + small_control penalty | ❌ 未使用 |
| cheetah-run | forward velocity reward + control penalty | ❌ 未使用 |
| reacher-easy | -distance(target, ee) + control_cost | ❌ 未使用 |
| hopper-stand | upright + standing + foot_contact + control_cost | ❌ 未使用 |

当前 `run_single_episode()` 的 `avg_return` 仅取 **最终timestep的单步reward** (`getattr(timestep, 'reward', 0.0)`)，而非episode累计reward。dm_control的timestep.reward是单步奖励（float），标准评估应累计所有步的reward得到 **episode return**。

**这是一个重大盲区**: IDO目前不计算也不输出dm_control标准累计reward，无法与RL baseline做公平的reward对比。

### 1.4 当前评估盲区汇总

| # | 盲区 | 影响 | 优先级 |
|---|------|------|--------|
| B1 | **dm_control累计reward未正确计算** | 无法与任何baseline做reward维度公平对比 | P0 |
| B2 | **所有baseline不可运行（退化为random）** | SER/NVR对比数据全部无效，Prophecy验证无法执行 | P0 |
| B3 | **成功率检查仅覆盖humanoid（right_hand）** | 其他24个任务的成功率判断缺失或错误 | P0 |
| B4 | **η不下降（所有任务≈停滞）** | SIP-Bench Retention Gain=1.0/Stability Index=0.0 无法区分 | P1 |
| B5 | **Noether违规分析不区分类型** | 只记录总违规数，不区分能量/力矩/碰撞的分布 | P1 |
| B6 | **无跨任务泛化评估** | 所有任务独立跑分，无任务迁移测试 | P2 |
| B7 | **无抗扰动评估** | 无噪声注入/环境扰动测试 | P2 |
| B8 | **Dashboard无评估结果对比图表** | IDO vs Baseline reward曲线无法可视化 | P1 |
| B9 | **无CSV/JSON结构化结果导出** | evaluate_vs_baseline.py有CSV输出，但run_mujoco_bench.py只输出JSON | P1 |
| B10 | **无论文用实验结果表格自动生成** | report_ido_advantage.py存在但未实现LaTeX表格生成 | P2 |
| B11 | **FlowMatching η预测器过于简单** | 线性外推+残差修正，无法与Cosmos-Predict的7B-14B模型公平对比 | P2 |
| B12 | **GoalEML target_pos部分为硬编码默认值** | 未从dm_control环境的task.get_reward()或observation提取真实目标 | P1 |

---

## 2. IDO 性能评估指标体系

### 2.1 基础指标（与dm_control标准对齐）

| 指标 | 定义 | 计算方式 | 当前实现 | 需改进 |
|------|------|---------|---------|--------|
| **Episode Return** | dm_control标准累计奖励 | Σ timestep.reward over all steps | ⚠️ 仅取最后一步reward | **P0: 改为累计求和** |
| **成功率** | 任务是否完成（到达目标） | ee_dist < pos_tol → 1, else → 0 | ⚠️ 仅humanoid有ee提取 | **P0: 每任务定制goal-achieved判断** |
| **Episode Length** | 到达目标或timeout的步数 | steps_to_goal (≤max_steps = success) | ✅ 已实现 | 无需改进 |
| **Survival Rate** | 5个episode中成功比例 | n_success / n_episodes | ✅ evaluate_vs_baseline有 | 无需改进 |

### 2.2 IDO 专属指标

| 指标 | 定义 | 物理含义 | 当前实现 | 需改进 |
|------|------|---------|---------|--------|
| **κ-Snap η** | 4维加权平方距残差 | 当前状态距Goal-EML陪集的距离 | ✅ gauss_ex_residual | 需per-task权重校准 |
| **η 趋势** | descending/plateau/ascending | 收敛速率的方向和幅度 | ✅ psi_anchor.analyze_eta_trend | 无需改进 |
| **NVR** | Noether Violation Rate = NV / total_steps | 守恒约束违规密度 | ✅ 每步检查 | **需按类型细分统计** |
| **δ_K 动态轨迹** | ψ-Anchor调整后的δ_K序列 | 自适应收敛阈值的演化路径 | ✅ adjusted_delta_K | 需可视化 |
| **conservation_score** | 1.0 - 0.3 × n_violations | Noether锚定分数 | ✅ psi_anchor | 无需改进 |
| **Epiplexity** | n_inv × (1/δ_K) × log(max_energy) | 策略有效复杂度 | ✅ compute_epiplexity | 无需改进 |
| **Hesitation-RMSE** | η围绕局部均值的震荡幅度 | 策略犹豫程度 | ✅ FlowMatchingEtaPredictor | 无需改进 |
| **Retry-VOC** | η方向翻转频率方差 | 策略反复尝试程度 | ✅ FlowMatchingEtaPredictor | 无需改进 |

### 2.3 纵向指标（SIP-Bench）

| 指标 | 定义 | 理论预期 | 当前值 | 需改进 |
|------|------|---------|--------|--------|
| **Retention Gain** | T0_avg / T2_avg | >1 表示改善持久 | 1.000（无法区分） | 需Motor Primitive优化后才有效 |
| **Stability Index** | T2_std / T0_std | <1 表示更稳定 | 0.000 | 需Motor Primitive优化后才有效 |
| **T0→T1→T2 η下降** | 各阶段平均η | T1 η < T0 η, T2 η ≤ T1 η | T0:2.545 → T1:2.615 → T2:2.409 | η变化极小，需Primitive覆盖度提升 |

### 2.4 对比指标（IDO vs Baseline）

| 指标 | 定义 | 计算方式 | 当前状态 |
|------|------|---------|---------|
| **Episode Return差** | IDO_return - baseline_return | 同任务同条件下累计reward对比 | ❌ 无法计算（IDO reward未累计，baseline未运行） |
| **成功率差** | IDO_success_rate - baseline_success_rate | 成功比例对比 | ❌ baseline未运行 |
| **收敛速度** | 达到阈值的步数 | steps_to_η_threshold | ❌ 无baseline参照 |
| **SER** | baseline_avg_steps / IDO_avg_steps | ≥1.2表示IDO更快 | ❌ baseline未运行 |
| **NVR差** | IDO_NVR - baseline_NVR | IDO NVR≡0 vs baseline NVR>0 | ❌ baseline未运行 |
| **η RMSE差** | IDO η轨迹 vs Cosmos η轨迹RMSE | η预测精度对比 | ❌ Cosmos未运行 |

---

## 3. Baseline 对比方案设计

### 3.1 对比维度

#### 3.1.1 性能维度

| 子维度 | IDO指标 | Baseline指标 | 对比方法 |
|--------|---------|-------------|---------|
| Episode Return | 累计dm_control reward | 累计dm_control reward | 同任务5 episode均值对比 |
| 成功率 | goal_achieved / n_episodes | goal_achieved / n_episodes | 比例对比 |
| 收敛速度 | steps_to η < δ_K | steps_to reward plateau | 曲线对比 |

#### 3.1.2 效率维度

| 子维度 | IDO指标 | Baseline指标 | 对比方法 |
|--------|---------|-------------|---------|
| 训练需求 | 零样本部署（无需训练） | 1M step训练预算 | 人力成本对比 |
| 推理延迟 | 每步计算时间(ms) | 每步推理时间(ms) | wall-clock时间对比 |
| 参数量 | Motor Primitives ~0个可训练参数 | TD-MPC2 1M-317M, PPO/SAC ~1M | 资源效率对比 |

#### 3.1.3 泛化维度

| 子维度 | 测试方法 | IDO预期 | Baseline预期 |
|--------|---------|---------|-------------|
| 跨任务迁移 | 在A任务训练，在B任务测试（零样本） | IDO零样本→直接部署，可能需重新定义GoalEML | RL需重新训练 |
| 抗扰动 | 注入观测噪声(σ=0.01/0.05/0.1) | Noether门应仍能过滤违规动作 | RL策略可能崩溃 |

#### 3.1.4 IDO 专属维度

| 子维度 | 指标 | IDO | Baseline |
|--------|------|-----|---------|
| κ-Snap η残差 | η绝对值+趋势 | 有（核心度量） | 无（baseline不计算η） |
| NVR | 每步违规率 | ≡0（理论保证） | >0（无守恒门） |
| ψ-Anchor效果 | Retention Gain + Stability Index | 有（SIP-Bench） | 无（无自演化机制） |

### 3.2 Baseline 选择与实现策略

| Baseline | 类型 | 对比目的 | 当前状态 | 实现策略 |
|---------|------|---------|---------|---------|
| **Random Agent** | 下界 | reward floor, NVR baseline | ✅ 可运行 | 保持现有实现 |
| **TD-MPC2** | model-based RL | 模型学习vs IDO适配 | ❌ 未安装 | P1: 需安装tdmpc2 + 训练checkpoint |
| **PPO (SB3)** | on-policy RL | 端到端训练vs IDO循环 | ❌ SB3未安装 | P1: 需安装SB3 + dm_control2gymnasium wrapper |
| **SAC (SB3)** | off-policy RL | 端到端训练vs IDO循环 | ❌ SB3未安装 | P1: 同PPO |
| **Cosmos-Predict1** | world model | η轨迹预测vs全状态预测 | ❌ 需GPU | P2: 需CUDA + cosmos_predict1 |

**优先级排序**: Random Agent → PPO/SAC (SB3, 可CPU训练) → TD-MPC2 → Cosmos-Predict (GPU)

### 3.3 对比实验设计

#### 3.3.1 任务集选择（≥8个任务）

| # | 任务 | 模态 | 维度 | 选择理由 |
|---|------|------|------|---------|
| 1 | humanoid-stand | 站立平衡 | 21 DOF | 最复杂自由度，IDO NVR≈1.47 |
| 2 | walker-walk | 行走运动 | 6 DOF | 经典运动任务 |
| 3 | cheetah-run | 前进奔跑 | 2 DOF | 简单运动，baseline应高分 |
| 4 | hopper-stand | 单腿站立 | 3 DOF | IDO NVR≈1.47 |
| 5 | reacher-easy | 机械臂到达 | 2 DOF | IDO NVR≡0，最简单 |
| 6 | cartpole-balance | 倒立摆平衡 | 1 DOF | 经典控制任务 |
| 7 | finger-turn_easy | 手指旋转 | 4 DOF | 精细操控任务 |
| 8 | fish-swim | 鱼游泳 | 6 DOF | 游泳模态 |

#### 3.3.2 每任务实验配置

| 参数 | IDO | PPO/SAC | TD-MPC2 | Random |
|------|-----|---------|---------|--------|
| Episodes | 5 | 5 (eval) | 5 (eval) | 5 |
| Max steps | 1000 | 1000 | 1000 | 1000 |
| 训练预算 | 0 (零样本) | 1M steps | 1M steps | 0 |
| 随机种子 | 42 (固定) | 42 (固定) | 42 (固定) | 42 (固定) |
| 评估种子 | 0-4 (5 seeds) | 0-4 (5 seeds) | 0-4 (5 seeds) | 0-4 |

#### 3.3.3 评估流程

```
1. 安装并配置baseline依赖 (SB3, dm_control2gymnasium)
2. 训练PPO/SAC/TD-MPC2 (1M steps per task, 8 tasks)
3. 同条件评估所有agent (5 episodes × 8 tasks)
4. 计算对比指标 (Return, 成功率, NVR, SER)
5. 生成结果表格和图表
```

#### 3.3.4 dm_control Reward 累计修复

当前 `avg_return` 仅取最终timestep的单步reward。修复方案：

```python
# 修改 run_single_episode() 和 evaluate_vs_baseline.py
episode_return: float = 0.0
for step_idx in range(max_steps):
    ...
    timestep = env.step(action)
    episode_return += float(timestep.reward)  # 累计求和
    ...
return {'avg_return': episode_return, ...}  # 返回累计reward
```

同时在 `_aggregate_metrics()` 中计算 mean episode return 而非 mean 单步 reward。

---

## 4. 下一步改进建议

### 4.1 P0 改进（必须立即实施）

| # | 改进项 | 描述 | 实现位置 | 预估工作量 |
|---|--------|------|---------|-----------|
| P0-1 | **修复dm_control累计reward** | `run_single_episode()`改为累计求和timestep.reward，返回episode_return而非单步reward | `benchmarks/run_mujoco_bench.py`, `webviz/server.py`, `benchmarks/evaluate_vs_baseline.py` | 0.5天 |
| P0-2 | **修复成功率检查** | 每个任务定制goal-achieved判断：reacher用to_target距离，walker/hopper/cheetah用dm_control reward阈值，humanoid用right_hand距离 | `benchmarks/run_mujoco_bench.py` | 0.5天 |
| P0-3 | **NVR类型细分统计** | noether_check_mj返回violation_type (energy/torque/collision)，run_single_episode记录各类型计数 | `core/noether_check_mj.py`, `benchmarks/run_mujoco_bench.py` | 0.5天 |
| P0-4 | **安装SB3 + dm_control wrapper** | 安装stable-baselines3 + shimmy(dm_control2gymnasium)，使PPO/SAC baseline实际可运行 | 环境配置 + `benchmarks/evaluate_vs_baseline.py` | 1天 |
| P0-5 | **PPO/SAC训练脚本** | 每任务训练1M steps，保存checkpoint，评估5 episodes | 新文件 `benchmarks/train_baselines.py` | 1天 |

### 4.2 P1 改进（应该实施）

| # | 改进项 | 描述 | 实现位置 | 预估工作量 |
|---|--------|------|---------|-----------|
| P1-1 | **Dashboard增加对比图表** | IDO vs Baseline reward曲线（Chart.js line chart），NVR对比柱状图，SER对比表 | `webviz/dashboard.html`, `webviz/server.py` | 1天 |
| P1-2 | **CSV/JSON结果统一导出** | run_mujoco_bench.py也输出CSV（当前仅JSON），格式与evaluate_vs_baseline.py统一 | `benchmarks/run_mujoco_bench.py` | 0.5天 |
| P1-3 | **GoalEML目标提取改进** | 从dm_control task的observation或reward函数提取真实target_pos（而非硬编码） | `core/goal_eml_mj.py` | 1天 |
| P1-4 | **η权重per-task校准** | 实验确定每个任务的w_pos/w_ori/w_eng/w_vel最优权重 | `core/kappa_snap_mj.py` | 2天 |
| P1-5 | **抗扰动评估** | 注入观测噪声(σ=0.01/0.05)，测试IDO/RL策略鲁棒性 | 新文件 `benchmarks/robustness_eval.py` | 1天 |

### 4.3 P2 改进（Nice to have）

| # | 改进项 | 描述 | 实现位置 | 预估工作量 |
|---|--------|------|---------|-----------|
| P2-1 | **论文表格自动生成** | report_ido_advantage.py实际实现LaTeX表格 | `benchmarks/report_ido_advantage.py` | 1天 |
| P2-2 | **跨任务迁移评估** | IDO在A任务学到的δ_K/macros迁移到B任务（零样本） | `benchmarks/transfer_eval.py` | 2天 |
| P2-3 | **Cosmos-Predict安装** | 需GPU环境，安装cosmos_predict1 | 环境配置 | 2天+ |
| P2-4 | **TD-MPC2 checkpoint** | 需tdmpc2已训练模型或自行训练 | `baselines/tdmpc2_adapter.py` + 训练脚本 | 3天+ |
| P2-5 | **Motor Primitive覆盖度提升** | 增加arm_swing/leg_coordination等任务特定元动作 | `agent/mujoco_ido_agent.py` MotorPrimitives | 3天 |

---

## 5. 待确认问题

| # | 问题 | 影响评估 | 建议确认方式 |
|---|------|---------|-------------|
| Q1 | **IDO当前是否能跑完整IDO循环？** | 确认Sense→κ-Snap→Noether→Motor→Critique完整闭环是否工作 | 运行 `python benchmarks/run_mujoco_bench.py --task reacher-easy` 检查输出 |
| Q2 | **TD-MPC2 adapter是否能实际运行并输出reward？** | 决定是否需要先安装tdmpc2 | 尝试 `import tdmpc2`，确认安装状态 |
| Q3 | **Cosmos-Predict adapter的实际运行状态？** | 决定η轨迹对比是否可行 | 检查CUDA + cosmos_predict1安装状态 |
| Q4 | **dm_control reward计算逻辑是否已集成？** | 影响累计reward修复方案 | 检查 `env.step(action)` 返回的 `timestep.reward` 是否为单步奖励 |
| Q5 | **PPO/SAC训练环境是否可搭建？** | 决定baseline对比能否执行 | 检查 gymnasium + shimmy 安装可行性 |
| Q6 | **Motor Primitive η停滞的根因？** | 所有任务η≈停滞是Primitive局限还是δ_K阈值问题 | 实验：降低δ_K到0.001，观察η是否开始下降 |
| Q7 | **GoalEML target_pos是否应从环境动态提取？** | 硬编码目标可能不准确 | 检查dm_control task.get_reward()或observation是否包含目标信息 |
| Q8 | **Webviz WebSocket是否支持baseline评估数据流？** | Dashboard对比图表的数据来源 | 检查server.py的WebSocket broadcast是否支持baseline metrics |

---

## 6. 实施路线图

### Phase 1: 评估基础设施修复（P0，预估3天）

```
Day 1: P0-1 修复累计reward + P0-2 修复成功率判断 + P0-3 NVR细分
Day 2: P0-4 安装SB3 + dm_control wrapper + 修改evaluate_vs_baseline.py
Day 3: P0-5 训练PPO/SAC on 2-3个简单任务(reacher-easy, cartpole-balance, cheetah-run)
```

### Phase 2: 8任务完整对比实验（P0+P1，预估5天）

```
Day 4: 扩展PPO/SAC训练到8个任务
Day 5: 运行完整IDO vs PPO/SAC/Random对比评估
Day 6: P1-1 Dashboard对比图表 + P1-2 CSV导出
Day 7: P1-3 GoalEML目标提取 + P1-4 η权重校准实验
Day 8: 结果汇总 + 初步论文表格
```

### Phase 3: 深度评估（P2，预估5天+）

```
Day 9+: 抗扰动评估 + 跨任务迁移 + TD-MPC2训练
```

---

## 7. 评估结果输出格式设计

### 7.1 JSON 输出格式（v0.4.5）

```json
{
  "version": "v0.4.5",
  "task": "humanoid-stand",
  "episodes": 5,
  "max_steps": 1000,
  "agents": {
    "IDO": {
      "avg_episode_return": 123.45,
      "avg_steps_to_goal": 856,
      "success_rate": 0.6,
      "avg_final_eta": 2.60,
      "nvr_total": 2950,
      "nvr_by_type": {"energy": 1800, "torque": 800, "collision": 350},
      "avg_hesit_rmse": 0.05,
      "avg_retry_voc": 0.8,
      "avg_epiplexity": 3.2,
      "retention_gain": 1.05,
      "stability_index": 0.95
    },
    "PPO": {
      "avg_episode_return": 456.78,
      "avg_steps_to_goal": 200,
      "success_rate": 0.8,
      "nvr_total": 150,
      "nvr_by_type": {"energy": 80, "torque": 50, "collision": 20},
      "episode_return_curve": [10.5, 12.3, ...]
    },
    "Random": {
      "avg_episode_return": 5.0,
      "avg_steps_to_goal": 1000,
      "success_rate": 0.0,
      "nvr_total": 4500
    }
  },
  "comparison": {
    "SER_PPO_vs_IDO": 0.23,
    "SER_Random_vs_IDO": 1.17,
    "return_diff_PPO_vs_IDO": 333.33,
    "nvr_diff_PPO_vs_IDO": -2800
  },
  "prophecy_verification": {
    "P1_NVR_IDO_0": {"status": "CHECK", "detail": "reacher-easy PASS, others TBD"},
    "P2_SER_ge_1.2": {"status": "TBD", "detail": "needs trained baselines"},
    "P3_Baseline_NVR_gt_0": {"status": "TBD", "detail": "needs trained baselines"}
  }
}
```

### 7.2 CSV 输出格式

```
agent,task,avg_return,avg_steps,success_rate,nvr,nvr_energy,nvr_torque,nvr_collision,avg_eta,SER
IDO,humanoid-stand,123.45,856,0.6,2950,1800,800,350,2.60,1.00
PPO,humanoid-stand,456.78,200,0.8,150,80,50,20,0.00,0.23
Random,humanoid-stand,5.00,1000,0.0,4500,0,0,0,0.00,1.17
```

### 7.3 LaTeX 论文表格格式

```latex
\begin{table}[h]
\centering
\caption{IDO vs Baseline on dm\_control tasks (5 episodes, 1K steps)}
\begin{tabular}{lcccccc}
\toprule
Task & Agent & Return & Success & Steps & NVR & SER \\
\midrule
humanoid-stand & IDO & 123.4 & 0.6 & 856 & 2.95 & 1.00 \\
               & PPO  & 456.8 & 0.8 & 200 & 0.15 & 0.23 \\
               & Rand & 5.0   & 0.0 & 1000 & 4.50 & 1.17 \\
...
\bottomrule
\end{tabular}
\end{table}
```

---

*Document by 许清楚（Alice） — Product Manager*
*Date: 2025-07-02*
