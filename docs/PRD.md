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

#### v0.3.0 — Baseline集成 + Web可视化（新增）

| ID | Requirement | Description |
|----|-------------|-------------|
| R13 | TD-MPC2 Baseline Adapter | `baselines/tdmpc2_adapter.py`：TD-MPC2 v2 baseline adapter，统一choose_action/evaluate/reset接口，1M步训练预算，优雅降级 |
| R14 | Cosmos-Predict Adapter | `baselines/cosmos_predict_adapter.py`：Cosmos-Predict世界模型adapter，η轨迹预测对比（7B/14B video2world），需GPU，优雅降级 |
| R15 | Baseline评估框架扩展 | `benchmarks/evaluate_vs_baseline.py`支持--eval-mode control/cosmos-predict两种评估模式，BASELINE_REGISTRY扩展 |
| R16 | Web可视化仪表盘 | `webviz/server.py` + `webviz/dashboard.html`：FastAPI REST API + WebSocket实时流 + Chart.js仪表盘，端口8080 |
| R17 | mjviser 3D Viewer | `webviz/server.py` launch_viewer()：mjviser Viewer + ViserServer在端口8081启动3D可视化（v0.3.0修复3个bug） |
| R18 | SIP-Bench Web支持 | Web仪表盘支持SIP-Bench toggle，实时显示T0/T1/T2阶段结果和Retention Gain/Stability Index |
| R19 | 中文顶刊论文 | `papers/mujoco_bench_ido_中文论文.md`：面向国内顶级学术期刊的完整论文（8,000-12,000字），含VG-Pair≠GAN理论框架 |
| R20 | 用户手册 | `docs/用户手册.md`：通俗易懂的中文用户手册，含快速开始、概念通俗解释、故障排查 |
| R21 | 用户手册HTML版 | `webviz/user_manual.html`：用户手册的交互式HTML版本，深色主题，左侧目录导航，挂到仪表盘首页 |
| R22 | MuJoCo中文文档 | `webviz/mujoco_docs_cn.html`：MuJoCo Overview完整中文翻译，深色主题，左侧目录导航，挂到仪表盘首页 |

### UI Design Draft

v0.3.0 提供两种交互界面：

#### CLI 交互（保留）

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

# v0.3.0 新增
evaluate_vs_baseline.py --task humanoid-stand --eval-mode cosmos-predict
  → IDO FlowMatching η vs Cosmos-Predict η轨迹对比
  → 输出 trajectory RMSE + correlation

evaluate_vs_baseline.py --task humanoid-stand --baseline tdmpc2_v2
  → TD-MPC2 v2 baseline对比
```

#### Web 仪表盘交互（新增）

```
python webviz/run_webviz.py
  → 打开 localhost:8080
  → 顶部导航栏：项目名称 + 用户手册链接 + MuJoCo中文文档链接 + 版本号(v0.3.0) + 运行状态
  → 左侧：任务选择、Episode配置、SIP-Bench toggle、mjviser按钮、Run/Stop
  → 右侧：η轨迹图、Noether计数器、κ-Snap状态、ψ-Anchor面板、IC-Value柱状图、SIP-Bench结果
  → 底部：状态栏
  → mjviser 3D Viewer：点击按钮 → localhost:8081 → 3D交互操作

用户手册HTML版：导航栏"📖 用户手册"链接 → /user_manual.html（新窗口）
  → 左侧：11章节目录（可点击跳转）
  → 右侧：完整用户手册内容（深色主题）
  → 底部版权：MuJoCo-Bench-IDO v0.3.0

MuJoCo中文文档：导航栏"📘 MuJoCo中文文档"链接 → /mujoco_docs_cn.html（新窗口）
  → 左侧：所有章节目录（可点击跳转）
  → 右侧：MuJoCo Overview完整中文翻译（深色主题）
  → 底部标注：中文翻译版 · 原文链接 · MuJoCo由Google DeepMind开源
```

### Open Questions

1. **Goal-EML 陪集粒度**：每个 dm_control 任务的 Goal-EML 不变量如何定义？Humanoid-stand 的陪集是否仅包含目标站立姿态，还是包含允许的身体摆动容差？需要明确各任务的陪集边界。
2. **κ-Snap η 的归一化**：连续状态向量各维度量纲不同（角度 vs 速度 vs 力矩），η 的平方距如何归一化以避免维度偏置？
3. **Noeter 能量校验的边界**：ΔE ≤ external work 的计算中，external work 是否应包含碰撞耗散？MuJoCo 的 contact 力如何纳入？
4. **Baseline 训练资源**：PPO/SAC/TD-MPC2 baseline 的训练预算如何设定？是否使用 stable_baselines3 默认超参，还是需专门调优以保证公平对比？
5. **Motor Primitive 与 Expert Replay 的依赖**：R10/R11 是否为 P0 IDO Agent 的必需组件？若缺失，Agent 的经验初始化与动作空间如何处理？
6. **Cosmos-Predict 生命周期**：Cosmos-Predict1已被Cosmos 3取代，是否需要迁移到Cosmos 3？
7. **mjviser 稳定性**：mjviser Viewer 在长时间运行中的稳定性如何？是否需要心跳检测？

---

*Document by 许清楚（Xu） — Product Manager*
*Date: 2025-07-01*
*v0.3.0 update: 2025-07-01*
*v0.3.1 update: 2025-07-01 — R23 语言切换, R24 实时3D仿真, R25 障碍物场景*

#### v0.3.1 — 语言切换 + 实时3D仿真 + 障碍物场景（新增）

| ID | Requirement | Description |
|----|-------------|-------------|
| R23 | 中英文界面切换 | 三个HTML页面（dashboard.html、user_manual.html、mujoco_docs_cn.html）加右上角中/EN切换按钮，使用 data-i18n 属性 + i18nDict 字典，localStorage key: mujoco-bench-lang |
| R24 | 实时3D仿真循环 | `launch_viewer()` 增加后台仿真线程，随机动作让机器人运动，viewer关闭时自动停止。plain场景用dm_control env.step，obstacle场景用mj.mj_step |
| R25 | 障碍物场景 | `webviz/scenes/humanoid_obstacle_arena.xml` 新增障碍物竞技场场景，dashboard新增3D场景下拉框，server.py新增 `/api/mjviser/scene` API endpoint，全局变量 mjviser_scene_type |

#### v0.3.1 Bug修复

| ID | Bug | Fix |
|----|-----|-----|
| BF1 | 导航链接是相对路径，从8081端口打开时404 | 改为绝对URL `http://localhost:8080/user_manual.html` 和 `http://localhost:8080/mujoco_docs_cn.html` |
