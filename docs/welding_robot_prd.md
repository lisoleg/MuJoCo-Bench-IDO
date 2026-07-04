# 焊接机器人仿真系统 Product Requirement Document

## 1. 项目信息

- **Language**: 中文
- **Programming Language**: Python (MuJoCo + dm_control + PyTorch/DreamerV3 + numpy)
- **Project Name**: mujoco_bench_ido (焊接场景扩展模块)
- **版本**: v0.10.0 (焊接机器人仿真扩展)
- **原始需求复述**: 在 MuJoCo-Bench-IDO 仿真平台中新增 6-DOF 工业焊接机器人场景，支持焊缝跟踪、DreamerV3 工艺参数优化训练，并集成 IDO/TOMAS 安全约束体系（Ψ-Anchor 焊接安全门控、κ-Snap 焊接因果快照、EML 工艺知识蒸馏）。参考论文《焊接机器人多模态数据采集及因果归因系统》的双层链架构与传感器方案，以及《硅基生命操作系统》附录Q（MuJoCo 焊接 XML 模板）与附录R（DreamerV3+MuJoCo 焊接训练代码）。

---

## 2. 产品定义

### 产品目标

> **目标一 — 焊接物理仿真闭环**：基于论文附录Q的 MuJoCo 焊接 XML 模板构建六轴机器人+双轴变位机+焊枪的完整焊接仿真场景，实现焊缝跟踪轨迹控制与焊接工艺参数（电流/电压/摆动/速度）的闭环仿真，使焊接过程在 MuJoCo 中可渲染、可测量、可复现。

> **目标二 — DreamerV3 工艺优化**：基于论文附录R训练 DreamerV3 世界模型学习焊接工艺参数→焊接质量的映射关系，通过 RSSM 编码器+转移模型预测 η 残差、气孔率、角变形，实现 Pareto 最优工艺参数的自动搜索与 EML 蒸馏，使 IDO/TOMAS 框架在焊接场景下的废品率目标 ≤ 0.1%（对比传统PID 5.0%、VLA 15.0%）。

> **目标三 — 焊接安全约束合规**：将 IDO 的 Ψ-Anchor 安全约束体系扩展到焊接域，实现 STICK_OUT（粘丝急停）、BURN_BACK（回烧停丝）、POROSITY_RISK（气孔风险摆动焊）三类焊接安全门控，确保焊接过程中每一次动作决策都经过 η-PID(1kHz) + Ψ-Anchor Gate 双重校验，并通过 κ-Snap 审计链不可篡改记录。

### User Stories

1. **As a 焊接工艺工程师**, I want 在 MuJoCo 中加载六轴焊接机器人场景并可视化焊缝跟踪轨迹 so that 我可以在不消耗实际焊材和工件的情况下验证焊接路径规划与工艺参数。

2. **As a RL 研究员**, I want 用 DreamerV3 在焊接环境中训练世界模型并自动搜索 Pareto 最优工艺参数 so that 我可以获得比传统PID和VLA更低的废品率（目标≤0.1%）和电流波动（目标±1.5A）。

3. **As a 安全审计工程师**, I want 焊接过程中的 STICK_OUT/BURN_BACK/POROSITY_RISK 危险工况被 Ψ-Anchor 实时检测并触发急停/停丝/摆动焊 so that 焊接机器人在仿真中就具备硬件级安全约束，不会在实机部署时发生粘丝烧焊事故。

4. **As a 论文作者**, I want IDO/TOMAS 框架在焊接场景下的对比实验数据（轨迹跟踪误差/电流波动/粘丝率/废品率）自动生成 LaTeX 表格 so that 我可以直接展示 IDO 相对于传统PID和VLA的差异化优势。

5. **As a 焊接认证工程师**, I want 从训练好的 DreamerV3 模型中蒸馏 Pareto 最优工艺参数到 EML 超图节点，并自动生成 WPS/PQR 文档 so that 焊接工艺参数可通过 CCS 船级社认证流程。

---

## 3. 技术规范

### Requirements Pool

#### P0 — Must Have（核心交付，焊接仿真闭环不可缺失）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| W01 | MuJoCo 焊接场景 XML | 新增 `envs/assets/mujoco_weld_robot.xml` | 按论文附录Q定义六轴机器人(joint1-6) + 双轴变位机(pos_rot_z, pos_tilt_x) + 焊枪(weld_gun) + 焊丝尖端(wire_tip) + 电弧区域(arc_cone) + 工件。含 TCP 位姿、干伸长距离、关节力矩、接触力传感器定义。4种焊接姿态关键帧：平焊/横焊/立焊/仰焊。干伸长约束(8-25mm)。EML 自定义数据接口预留 |
| W02 | WeldingEnv 焊接环境 | 新增 `envs/welding_env.py` | MuJoCo 焊接环境包装器。action_dim=4（焊接电流/电弧电压/摆动幅度/焊接速度），obs_dim=18（TCP位姿6 + 关节角6 + 干伸长1 + 接触力3 + 温度1 + 焊缝偏差1）。继承现有 `PinchLeafEnv`/`dmctrl_wrapper` 模式。提供 `reset()`/`step()`/`render()` 标准接口。焊缝跟踪轨迹以 waypoint 序列定义 |
| W03 | 焊缝跟踪控制器 | 新增 `agent/welding_controller.py` | 6-DOF 工业臂焊缝跟踪控制器。读取焊缝 waypoint 序列，生成关节空间轨迹（PD/轨迹插值），输出焊枪 TCP 位姿+速度指令。支持直线焊缝、圆弧焊缝、搭接焊缝。跟踪误差目标 ≤ 0.03mm（对标 IDO/TOMAS 实验数据） |
| W04 | 焊接 Ψ-Anchor 安全门控 | 升级 `core/t_processor.py` + 新增 `agent/welding_psi_anchor.py` | 扩展 T-Processor 的 Ψ-Check 新增三类焊接安全约束：① STICK_OUT 检测：current>MAX && voltage<5V → 急停；② BURN_BACK 检测：stick_out<MIN(8mm) → 停止送丝；③ POROSITY_RISK 检测：arc_length_variance>THRESH → 触发摆动焊模式。违规触发 κ-Snap REJECT_WELDING_* 事件 |
| W05 | 焊接 κ-Snap 事件类型 | 升级 `core/kappa_snap_mj.py` | 扩展 κ-Snap 事件类型覆盖焊接域：WELDING_STICK_OUT, WELDING_BURN_BACK, WELDING_POROSITY_RISK, WELDING_ARC_STABLE, WELDING_PASS_COMPLETE, WELDING_DEFECT_DETECTED。每条焊接 κ-Snap 记录含焊接参数快照(current/voltage/speed/stickout) |
| W06 | 焊接 η 残差计算 | 升级 `core/kappa_snap_mj.py` | 定义焊接域 GaussEx 残差 η_weld：焊缝偏差(位置) + TCP 姿态偏差(角度) + 干伸长偏差 + 电流波动。权重可配置，支持 per-weld-type 归一化 |

#### P1 — Should Have（DreamerV3 工艺优化与差异化指标）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| W07 | WeldingProcessProxy 工艺代理模型 | 新增 `core/welding_process_proxy.py` | 焊接工艺代理模型（不模拟流体力学/MHD）。输入：焊接参数(current/voltage/weave/speed) + 焊缝几何。输出：η 残差预测、气孔率(porosity)预测、角变形(ang_distortion)预测、熔深估计。基于经验公式+查表+轻量MLP混合模型。为 DreamerV3 提供快速 reward 信号 |
| W08 | DreamerV3 焊接训练 | 新增 `baselines/dreamer_weld_train.py` | 按论文附录R实现 DreamerV3 焊接训练。核心：RSSM(编码器+转移模型+解码器+奖励预测) + Actor + Critic。训练循环 1000 episodes。奖励函数 = -η_weld·10 - porosity·20 - ang_distortion·50 - stickout_penalty。复用现有 `hybrid_dreamer_ido_agent.py` 的 IDO 认知层（κ-Snap/Ψ-Anchor/Noether）作为 meta-management |
| W09 | EML 工艺参数蒸馏 | 升级 `core/goal_eml_mj.py` + 新增 `core/welding_eml_distill.py` | 从 DreamerV3 训练结果中提取 Pareto 最优工艺参数集（η最小/气孔率最低/角变形最小），蒸馏为 EML 超图节点。每个节点记录：焊接类型、参数向量、η值、质量预测、适用条件。支持 EML 节点的检索与复用 |
| W10 | 焊接对比评估脚本 | 新增 `benchmarks/welding_compare.py` | IDO/DreamerV3 vs 传统PID vs VLA 焊接对比评估。输出指标：轨迹跟踪误差(mm)、电流波动(±A)、粘丝率(%)、废品率(%)。对标论文实验数据：传统PID(0.05mm/±5.0A/2.1%/5.0%)、VLA(0.12mm/±15.0A/12.4%/15.0%)、IDO/TOMAS(0.03mm/±1.5A/0.0%/0.1%)。自动生成 LaTeX 表格 |

#### P2 — Nice to Have（多模态传感器与认证扩展）

| ID | Requirement | 涉及模块 | Description |
|----|-------------|----------|-------------|
| W11 | 多模态焊接传感器仿真 | 新增 `core/welding_sensors.py` | 仿真论文1的7类传感器：霍尔电流(LEM LAH 50-P, 50kHz)、电弧电压、声发射(100kHz)、同轴视觉(Basler ace 2, 1000fps)、红外温度(60Hz)、激光轮廓仪(Keyence IL-300, 2kHz)、气体流量。自适应采样率（GaussEx 残差驱动：η大时提高采样率） |
| W12 | 焊接缺陷 κ-Snap 因果快照 | 升级 `core/kappa_snap_mj.py` | 3种焊接缺陷的 κ-Snap 因果归因样本：气孔(Porosity) — 因果链：保护气体不足→熔池扰动→气孔；咬边(Undercut) — 因果链：电流过大/速度过慢→母材过熔→咬边；飞溅(Spatter) — 因果链：电压过高/干伸长过长→电弧不稳→飞溅。每个缺陷样本记录完整因果链 |
| W13 | WPS/PQR 文档生成 | 新增 `tools/wps_pqr_generator.py` | 从 EML Pareto 最优工艺参数自动生成 WPS(焊接工艺规程) 和 PQR(工艺评定记录) 文档。支持 CCS(中国船级社) 认证流程所需的文档格式。输出 PDF/DOCX |
| W14 | TOMAS 焊接工艺公理库 | 新增 `core/tomas_welding_axioms.py` | 焊接工艺公理知识库：平焊/横焊/立焊/仰焊的参数范围公理、材料-电流-电压匹配公理、干伸长-焊丝直径关系公理。为 Ψ-Anchor 和 DreamerV3 reward 提供先验约束 |

### UI Design Draft

#### CLI 交互

```bash
# 焊接场景渲染与轨迹跟踪
python -m envs.welding_env --weld-type flat --render --trajectory seam_straight
  → 加载 mujoco_weld_robot.xml，渲染六轴机器人焊接平焊场景
  → 输出：TCP 轨迹跟踪误差曲线、干伸长曲线、关节力矩

# DreamerV3 焊接训练
python baselines/dreamer_weld_train.py --episodes 1000 --weld-type flat
  → 训练 DreamerV3 世界模型
  → 输出：η 残差收敛曲线、气孔率下降曲线、Pareto 前沿图
  → 训练完成后自动蒸馏 EML 节点

# 焊接对比评估
python benchmarks/welding_compare.py --weld-type flat --report latex
  → IDO/DreamerV3 vs PID vs VLA 对比
  → 输出：4指标对比表(轨迹误差/电流波动/粘丝率/废品率) + LaTeX 表格

# WPS/PQR 生成
python tools/wps_pqr_generator.py --eml-node pareto_001 --format pdf
  → 从 EML 最优节点生成 WPS/PQR 文档
```

#### Web 仪表盘交互

```
dashboard.html 焊接面板新增：
  → 焊接场景 3D 可视化：六轴机器人 + 变位机 + 焊枪 + 焊缝 + 电弧
  → 焊接参数实时监控面板：电流(A)/电压(V)/速度(mm/s)/干伸长(mm) 实时曲线
  → η 残差实时曲线 + 阈值线（η→0 目标可视化）
  → Ψ-Anchor 焊接安全状态灯：
     - STICK_OUT: 🟢安全 / 🔴急停
     - BURN_BACK: 🟢安全 / 🟡停丝
     - POROSITY_RISK: 🟢安全 / 🔵摆动焊
  → κ-Snap 焊接事件时间线：焊接过程事件链可视化
  → Pareto 前沿散点图：η vs 气孔率 vs 角变形 三维权衡
  → EML 工艺参数节点浏览器
```

### Open Questions

1. **焊缝跟踪精度对标**：论文给出 IDO/TOMAS 轨迹跟踪误差 0.03mm 是在 SO-ARM100 6-DOF 机械臂上的实验数据。MuJoCo 仿真中由于物理引擎离散化精度（默认 2ms timestep），能否真实达到 0.03mm 级跟踪精度？是否需要将焊接场景的仿真步长降至 0.5ms 或更低？

2. **WeldingProcessProxy 代理模型保真度**：论文附录R明确指出"不模拟流体力学"，用代理模型预测 η/气孔/角变形。代理模型的预测精度需要达到什么水平才能支撑 DreamerV3 的有效训练？是否需要与真实焊接实验数据做交叉验证，还是纯仿真内闭环即可？

3. **DreamerV3 与 IDO 认知层的集成方式**：现有 `hybrid_dreamer_ido_agent.py` 的 IDO 认知层（κ-Snap/Ψ-Anchor/Noether）在 locomotion 任务中会跳过 SafeFuse 和 PreAffect。焊接场景是否需要启用完整的 IDO 认知层（包括 SafeFuse 降级），还是焊接安全约束仅通过 Ψ-Anchor 焊接门控（W04）处理？

4. **EML 蒸馏与现有 GoalEML 的关系**：现有 `goal_eml_mj.py` 定义了 locomotion 任务的 GoalEML 结构。焊接 EML 节点的 Pareto 最优工艺参数是否需要作为新的 GoalEML 子类实现，还是作为独立的 EML 超图节点存储？两者如何统一检索？

5. **4种焊接姿态的关键帧定义**：附录Q提到平焊/横焊/立焊/仰焊4种姿态关键帧。这些关键帧是作为 MuJoCo XML 中的 keyframe 定义，还是由 `welding_controller.py` 在运行时计算？不同姿态下变位机(pos_rot_z/pos_tilt_x)的初始角度如何确定？

---

*Document by 许清楚（Xu） — Product Manager*
*Date: 2025-07-04*
*Version: v0.10.0 (焊接机器人仿真扩展)*
