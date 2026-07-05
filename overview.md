# v0.17.0 — TOMAS Agent Full-Stack Integration

## TL;DR
完全吸收腾讯文档1.37M字符超长对话内容，升级 MuJoCo-Bench-IDO 到 v0.17.0，新增7个文件、升级2个文件，659/659 测试全部通过，零回归。

## 交付概览
- **交付状态**: ✅ 完成
- **测试通过率**: 659/659 (100%)
- **已知问题数**: 0
- **新测试**: 39个 (tests/test_v0170.py)
- **执行时间**: 26.50s

## 新建文件 (7个)

### 1. `agent/tomas_mujoco_wrapper.py`
TOMAS MuJoCo 环境集成包装器，实现 P→C→S 三层管线：
- **P-Layer**: VLA 推理 (`_vla_infer`)
- **C-Layer**: PG-Gate 物理钳位 + ψ-Anchor 硬约束检查 (`_degrade_action`)
- **S-Layer**: κ-Snap 因果快照审计
- 接口: `step()`, `reset()`, `get_audit_trail()`, `get_safety_report()`, `get_eta_history()`

### 2. `agent/failure_attribution.py`
TOMAS S-Layer 失败归因引擎：
- 6种流贯病理: `local_optimum_trap`, `eta_false_convergence`, `premature_release`, `eta_escape`, `psi_anchor_overkill`, `validation_gap`
- LLM prompt 构建 + JSON 解析
- 离线模式回退（启发式模式匹配）
- `offline_attribuate()` 方法：无需 LLM 即可分类失败类型

### 3. `agent/tomas_deploy.py`
TOMAS Agent 部署编排器：
- `TOMASAgent.deploy()` — 多 episode 部署管理
- **MetaQuery** 自我归因系统: `WHY_THIS_ACTION()`, `AUDIT_SNAP()`, `LEARN_SKILL()`
- **SkillRecord** — EML-SemZip 技能学习 (Dead-Zone 剪枝 IC<0.45, 高 IC 过采样)
- `DeployReport` — 完整部署报告 (JSON 序列化)
- 6种病理自动检测 + 自适应调整

### 4. `agent/footstep_planner.py`
足端轨迹规划器：
- **SupportPolygon** — 支撑多边形 + ZMP 包含检测 (凸包 + 安全裕量)
- **FootstepPlanner** — 世界坐标系→语义目标→步序生成→ZMP校验→ψ-锚审计
- Bezier 曲线 swing 轨迹 + CoM 轨迹生成
- 障碍物避让 (势场法)

### 5. `config/psi_anchor_defaults.yaml`
SO-ARM100/101 ψ-锚完整配置：
- physics: max_torque=0.050N·m, max_velocity=1.5rad/s, max_pitch=15°
- walking: zmp_safety_margin=0.015m, max_step_length=0.08m
- gripper: sentient_finger_limit=0.05N·m
- kappa_snap: enabled, jsonl_output, merkle_chain_verify
- safe_fuse: GREEN/YELLOW/RED 三级渐进
- task_overrides: pick_and_place, pinch_leaf, welding

### 6. `envs/assets/so_arm100_mujoco_ido.xml`
SO-ARM100 MuJoCo 仿真场景：
- 5个铰链关节: Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll
- 2个夹持关节: Gripper_Left/Gripper_Right (sentient finger)
- ST3215标定: motor gear="50", position actuator kp/kv
- 7×jointpos + 7×jointvel + 2×touch 传感器
- 桌面 + 目标方块 + 目标位置标记 + 3个相机

### 7. `tests/test_v0170.py`
39个集成测试，8个测试类：
- TestHardPhysicsGate (8 tests)
- TestHGPINNPolicy (6 tests)
- TestSupportPolygon (4 tests)
- TestFootstepPlanner (7 tests)
- TestTOMASFailureAttribution (3 tests)
- TestTOMASAgentDeploy (6 tests)
- TestModuleImports (5 tests)

## 升级文件 (2个)

### 1. `core/hg_pinn.py`
新增两个类：
- **HardPhysicsGate** — 6阶段物理约束投影 (velocity→torque→tau_safe→acceleration→pitch/roll→ZMP)
- **HG_PINN_Policy** — 完整策略 (backbone + PG-Gate), 支持 dict/tuple/array 观测格式

### 2. `agent/__init__.py`
新增15个导出：
- TOMASMuJoCoWrapper, TOMASFailureAttributor, FailureAttributionResult
- TOMASAgent, DeployStatus, DeployReport, MetaQueryType, MetaQueryResult, SkillRecord
- FootstepPlanner, SupportPolygon, Footstep, FootstepPlan, FootSide, StepPhase

## 架构总览

```
TOMASAgent.deploy()
    │
    ├── P-Layer: VLA Policy → raw_action
    │
    ├── C-Layer: PG-Gate clamp → ψ-Anchor check
    │   ├── HardPhysicsGate (6-stage projection)
    │   └── PsiAnchor (η trend + evolution)
    │
    ├── S-Layer: κ-Snap audit → MetaQuery
    │   ├── KappaSnapLogger (MerkleChain)
    │   ├── TOMASFailureAttributor (6 pathology types)
    │   └── MetaQuery (WHY_THIS_ACTION / AUDIT_SNAP / LEARN_SKILL)
    │
    └── DeployReport → JSON
```

## 下一步建议
1. 将 TOMASAgent 接入 webviz/server.py 的 benchmark 循环
2. 加载真实 VLA 权重 (OpenVLA/Octo/π₀)
3. 在 SO-ARM100 场景上运行端到端 pick-and-place 评估
4. 提交代码到 GitHub
