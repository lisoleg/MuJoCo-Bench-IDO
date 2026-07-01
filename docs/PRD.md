# MuJoCo-Bench-IDO Product Requirement Document

## 1. 项目信息

- **Language**: 中文
- **Programming Language**: Python (dm_control + MuJoCo + numpy)
- **Project Name**: mujoco_bench_ido
- **原始需求复述**: 将 ARC 离散符号求解器（tomas-arc3-solver）的 IDO/TOMAS 架构升级到 MuJoCo 连续物理控制域，保留 IDO Harness 哲学（L2壳感知→经验→双路径更新→κ-Snap→Noether→Critique），将离散符号映射替换为连续物理状态映射，并在 dm_control suite 上验证 IDO 优于传统 RL baseline。

## 2. 产品定义

### Product Goals

1. **架构迁移验证**：证明 IDO Harness 哲学可从离散符号域（ARC）迁移到连续物理控制域（MuJoCo），核心五环节（κ-Snap / Noether / Goal-EML / Motor Primitive / Expert Replay）均可物理化实现。
2. **效率预言兑现**：在 dm_control suite 标准任务上，IDO Agent 的 κ-Snap 方向引导使求解步数较 BFS-discretize 降低 ≥30%（P1），Noether 物理守恒校验使 IDO NVR=0 而 PPO NVR>0（P2），SER≥1.2（P4）。
3. **可复现基准框架**：提供一键跑分脚本与 baseline 对比评估工具，使第三方可在标准 dm_control 任务上复现 IDO vs PPO/SAC/TD-MPC2 的对比结果。

### User Stories

1. **As a RL researcher**, I want to run a single command to benchmark IDO against PPO/SAC/TD-MPC2 on dm_control tasks, so that I can objectively evaluate whether IDO's Harness philosophy outperforms standard RL in continuous control.
2. **As an IDO framework developer**, I want to see the κ-Snap module compute continuous GaussEx residual η from MuJoCo state vectors, so that I can verify the direction-guidance mechanism works in continuous domains as it did in discrete ARC tasks.
3. **As a safety-conscious roboticist**, I want Noether-Check to reject trajectories that violate physical conservation laws (excess torque, energy creation, self-collision), so that I can trust IDO won't produce reward-hacking solutions that are physically invalid.

## 3. 技术规范

### Requirements Pool

#### P0 — Must Have（核心交付）

| ID | Requirement | Description |
|----|-------------|-------------|
| R01 | IDO MuJoCo Agent | `agent/mujoco_ido_agent.py`：实现 IDO Harness 五环节循环（感知→经验→双路径→κ-Snap→Noether→Critique），适配 MuJoCo mjData 连续状态输入 |
| R02 | κ-Snap 连续残差 | `core/kappa_snap_mj.py`：计算 η = 连续状态距 Goal-EML 陪集的平方距，替代 ARC 的像素 diff 残差 |
| R03 | Goal-EML 物理陪集 | `core/goal_eml_mj.py`：为每个 dm_control 任务定义 Goal-EML 不变量陪集（如 target pose、energy band、contact pattern），使 κ-Snap 可计算方向残差 |
| R04 | Noether 物理校验 | `core/noether_check_mj.py`：三重校验——力矩≤actuator limit、能量不凭空增（ΔE ≤ external work）、自碰撞拒 |
| R05 | 跑分脚本 | `benchmarks/run_mujoco_bench.py`：支持 Humanoid-stand/walk、Reacher-easy、Hopper-stand、Walker2d-run 四类任务的一键运行 |

#### P1 — Should Have（预言验证）

| ID | Requirement | Description |
|----|-------------|-------------|
| R06 | κ-Snap 步数缩减预言 | P1 验证：IDO κ-Snap 引方向 > BFS-discretize，IDO steps ↓ ≥30%，需在 ≥3 个任务上统计显著 |
| R07 | Noether NVR=0 预言 | P2 预言：IDO NVR（物理违规率）=0，PPO NVR>0，需在 ≥3 个任务上验证 |
| R08 | SER≥1.2 预言 | P4 预言：IDO SER（Solution Efficiency Ratio）≥1.2 on reach/walk，p<.05 |
| R09 | Baseline 对比评估 | `benchmarks/evaluate_vs_baseline.py`：IDO vs PPO/SAC/TD-MPC2 标准化对比，输出表格与统计检验 |

#### P2 — Nice to Have（增强与扩展）

| ID | Requirement | Description |
|----|-------------|-------------|
| R10 | Motor Primitive | IC_Value_Score=ΔIC 门控的 Motor Primitive，替代 NARLA discrete tile macro |
| R11 | Expert Demonstration Replay | Oracle Replay 的物理版本——已知轨迹回放作为 IDO 经验初始化 |
| R12 | 论文 Appendix C | `papers/mujoco_bench_ido_validation.md`：完整实验结果与预言验证的论文级附录 |

### UI Design Draft

本项目为纯 Python 库 + CLI 跑分工具，无 GUI。核心交互流程：

```
run_mujoco_bench.py --task Humanoid-stand --agent ido --seed 42 --episodes 100
  → 加载 mujoco_ido_agent.py + goal_eml_mj.py + kappa_snap_mj.py + noether_check_mj.py
  → 循环 episode：
      1. 感知：从 mjData 提取 qpos/qvel/actuator_force/sensor
      2. κ-Snap：计算 η（距离 Goal-EML 陪集）
      3. 双路径更新：经验路径 + 规则路径
      4. Noether-Check：物理守恒校验
      5. Critique：评估轨迹合法性
  → 输出 metrics：steps, NVR, SER, success_rate

evaluate_vs_baseline.py --tasks Humanoid-stand,Reacher-easy --baselines ppo,sac,tdmpc2
  → 逐任务逐 agent 运行
  → 聚合结果表格 + Wilcoxon / t-test 统计检验
  → 输出 CSV + console summary
```

### Open Questions

1. **Goal-EML 陪集粒度**：每个 dm_control 任务的 Goal-EML 不变量如何定义？Humanoid-stand 的陪集是否仅包含目标站立姿态，还是包含允许的身体摆动容差？需要明确各任务的陪集边界。
2. **κ-Snap η 的归一化**：连续状态向量各维度量纲不同（角度 vs 速度 vs 力矩），η 的平方距如何归一化以避免维度偏置？
3. **Noeter 能量校验的边界**：ΔE ≤ external work 的计算中，external work 是否应包含碰撞耗散？MuJoCo 的 contact 力如何纳入？
4. **Baseline 训练资源**：PPO/SAC/TD-MPC2 baseline 的训练预算如何设定？是否使用 stable_baselines3 默认超参，还是需专门调优以保证公平对比？
5. **Motor Primitive 与 Expert Replay 的依赖**：R10/R11 是否为 P0 IDO Agent 的必需组件？若缺失，Agent 的经验初始化与动作空间如何处理？

---

*Document by 许清楚（Xu） — Product Manager*
*Date: 2025-07-01*
