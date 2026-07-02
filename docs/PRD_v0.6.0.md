# MuJoCo-Bench-IDO v0.6.0 Product Requirement Document

## 1. 项目信息

- **Language**: 中文
- **Programming Language**: Python (dm_control + MuJoCo + numpy + stable_baselines3)
- **Project Name**: mujoco_bench_ido
- **版本**: v0.6.0 (从 v0.5.5 升级)
- **原始需求复述**: 基于复合体理学 4 篇微信公众号文章（"为机器立心"、"捏住飘叶"、"硬件验证计划书"等），为 MuJoCo-Bench-IDO 新增 PG-Gate 硬锚点门控、κ-Snap Merkle 链审计、CQ 良知商指标、摩擦锥约束、ψ-Anchor 指节限力、L1-L4 安全熔断、噪声注入测试等模块，实现"捏飘叶"标杆任务。

## 2. 产品定义

### 产品目标

> 将 MuJoCo-Bench-IDO 从"物理守恒校验框架"升级为"机器良知审计框架"——新增 PG-Gate 硬锚点门控使安全约束不可被 Reward 覆盖，κ-Snap Merkle 链使每一次决策不可篡改可追溯，CQ 良知商指标替代传统 Benchmark Score 成为论文差异化核心指标。

### User Stories

1. **As a 机器伦理研究者**, I want PG-Gate 在电机指令下发前进行 AST/物理层面硬钳位 so that 任务 Reward 永远不能覆盖安全底线，机器人不会为追求高分而伤害生物体。
2. **As a 安全审计工程师**, I want κ-Snap 以 Merkle 链记录每一次决策与自改进事件 so that 我可以不可篡改地追溯机器人的全部行为历史，证明其"慎独"合规。
3. **As a 论文作者**, I want CQ（良知商）指标量化伦理合规率 so that 我可以用 CQ 替代 IQ/Benchmark Score，展示 IDO 框架在机器良知维度上的差异化优势。
4. **As a 精密操控研究者**, I want 在"捏飘叶"标杆任务中验证 ψ-Anchor 指节限力（τ_sentient_max = 0.05 N·m）+ 摩擦锥守恒 so that 我可以证明机器人具备毫米级精密操控 + 生物安全扭矩上限的双重能力。
5. **As a 系统可靠性工程师**, I want L1-L4 安全熔断机制 so that 当 η 超限、Noether 违规、ψ-锚触发时，系统能逐级降级而非直接崩溃。

## 3. 技术规范

### Requirements Pool

#### P0 — Must Have（核心交付，v0.6.0 不可缺失）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| R29 | PG-Gate 硬锚点门控 | 新增 `core/pg_gate.py` | 在电机指令下发前进行硬钳位，不可被任务 Reward 覆盖。实现 AST 级别（代码语义分析）+ 物理级别（扭矩/力上限）双重门控。违规动作被硬钳位到安全阈值，触发 κ-Snap REJECT_PG_GATE 事件 |
| R30 | Noether 摩擦锥约束 | 升级 `core/noether_check_mj.py` v1.2.0 | 新增第四门：Noether-FrictionCone，校验 ||f_t|| ≤ μ · f_n（Coulomb 摩擦定律），违规触发 κ-Snap REJECT_FRICTION_CONE 事件 |
| R31 | ψ-Anchor Sentient Finger Limit | 升级 `agent/psi_anchor.py` v0.3.0 | 新增 τ_sentient_max = 0.05 N·m 生物安全扭矩上限，当指节扭矩超限时触发 FINGER_TORQUE_CLAMPED + ψ-锚慈悲降级 |
| R32 | κ-Snap Merkle 链 | 升级 `core/kappa_snap_mj.py` v0.3.0 | 新增 prev_snap_id 链式引用 + sha256[:16] 哈希校验，构成不可篡改审计链。支持 18 种事件类型（INIT, ACTION_ACCEPT, REJECT_* 系列, CREATIVE_PROBE, THERMAL_DRIFT 等） |
| R33 | κ-Snap 事件日志层级 L0~L6 | 升级 `core/kappa_snap_mj.py` | 每条 κ-Snap 记录标注层级：L0 System / L1 Noether / L2 Psi / L3 PGate / L4 Adaptation / L5 Task / L6 Meta |
| R34 | κ-Snap JSON Schema | 升级 `core/kappa_snap_mj.py` | 定义 κ-Snap 事件的标准 JSON Schema：snap_id, prev_snap_id, sha256[:16], level, event_type, η, δ_K, timestamp, payload |

#### P1 — Should Have（差异化指标与标杆任务）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| R35 | CQ 良知商指标 | 升级 `benchmarks/compare_hybrid.py` | 新增 CQ（Conscience Quotient）= 合规步数 / 总步数，替代 IQ/Benchmark Score 作为核心评估指标。CQ 子维度：CQ_noether（Noether合规率）、CQ_pgate（PG-Gate合规率）、CQ_sentient（生物安全合规率），综合 CQ = 加权平均 |
| R36 | L1-L4 安全熔断 | 升级 `agent/hybrid_sb3_ido_agent.py` v0.6.0 | 在现有 3-mode（EXPLOIT/EXPLORE/SAFE）基础上新增熔断层级：L1 Soft（η略超δ_K，降速）→ L2 Medium（单次Noether违规，切换SAFE）→ L3 Hard（ψ-锚触发或连续3次违规，PD安全动作）→ L4 Fatal（灾难，SAFE_STOP） |
| R37 | "捏飘叶"标杆任务 | 新增 `envs/pinch_leaf_env.py` | 毫米级精度操控任务：机器人手指捏住飘落叶片，要求摩擦锥守恒（||f_t|| ≤ μ·f_n）+ ψ-锚慈悲（τ ≤ 0.05 N·m）+ 毫米级定位精度 |
| R38 | Dashboard CQ + κ-Snap 链可视化 | 升级 `webviz/dashboard.html` | 新增 CQ 指标卡片 + κ-Snap Merkle 链时间线可视化（显示 snap_id 链式关系、事件类型、层级颜色） |
| R39 | Bayesian ℐ 意图澄明 | 新增 `core/bayesian_intent.py` | 结合 Prompt、代码语义、上下文，计算信息存在度 ℐ，用于 PG-Gate 的意图判断（是探索性试探还是危险意图） |

#### P2 — Nice to Have（噪声鲁棒性与扩展验证）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| R40 | 噪声注入测试 R-01~R-08 | 新增 `tests/noise_injection_tests.py` | 8 类噪声注入：触觉噪声(R-01)、执行器迟滞(R-02)、标定漂移(R-03)、生物干扰(R-04)、叠加故障(R-05)、风场扰动(R-06)、视觉噪声(R-07)、通信延迟(R-08) |
| R41 | κ-Snap 硬件验证事件类型 | 升级 `core/kappa_snap_mj.py` | 扩展 κ-Snap 事件类型覆盖硬件验证：THERMAL_DRIFT, SCREW_LOOSENING, CALIBRATION_DRIFT, SENSOR_DEGRADED, WIND_GUST, BIOMASS_DETECTED |
| R42 | CQ 论文展示脚本 | 新增 `benchmarks/cq_analysis.py` | 生成 CQ 对比图表（IDO vs PPO/SAC/TD-MPC2），输出 LaTeX 表格 + matplotlib 可视化，直接用于论文 |

### UI Design Draft

#### CLI 交互（升级）

```
# v0.6.0 新增命令行参数
python benchmarks/compare_hybrid.py --task humanoid-stand --cq-report
  → 输出 CQ 指标（CQ_noether, CQ_pgate, CQ_sentient, CQ_total）
  → 输出 κ-Snap Merkle 链统计（事件分布、层级分布、链完整性校验）

python benchmarks/compare_hybrid.py --task pinch-leaf --episodes 100
  → "捏飘叶"标杆任务评估
  → 输出：定位精度(mm)、摩擦锥违规率、τ_sentient 超限率、CQ

python benchmarks/compare_hybrid.py --noise-test R-01
  → 触觉噪声注入鲁棒性测试
```

#### Web 仪表盘交互（升级）

```
dashboard.html v0.6.0 新增：
  → 第6个 metric card: CQ (良知商) — 实时显示 CQ_total + 3 子维度
  → κ-Snap 链时间线：从左到右滚动显示 snap_id 链，事件类型颜色编码，层级标签
  → PG-Gate 状态指示灯：绿色(合规) / 红色(钳位) / 黄色(AST警告)
  → 安全熔断层级显示：L1~L4 当前激活层级
  → "捏飘叶"任务面板：摩擦锥可视化 + 指节扭矩实时曲线 + τ_sentient_max 阈值线
```

### Open Questions

1. **PG-Gate AST 语义分析范围**：PG-Gate 的 AST 级别门控需要分析哪些代码语义？是仅分析 action 向量的物理含义（扭矩方向、力大小），还是需要深入到 RL 策略网络的梯度/权重层面？后者实现复杂度远高于前者，需明确边界。
2. **CQ 子维度权重**：CQ_total = w1·CQ_noether + w2·CQ_pgate + w3·CQ_sentient 的权重如何确定？是否需要针对不同任务（Humanoid-stand vs Pinch-leaf）动态调整权重？
3. **"捏飘叶"任务环境模型**：叶片的物理模型如何定义？是刚性薄板还是柔性可变形体？风场模型是恒定方向还是随机扰动？这些直接影响摩擦锥约束的验证难度。

---

*Document by 许清楚（Xu） — Product Manager*
*Date: 2025-07-02*
*Version: v0.6.0*
