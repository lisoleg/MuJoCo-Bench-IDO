# MuJoCo-Bench-IDO 系统架构设计

> **架构师**: 高见远（Gao） — Architect  
> **日期**: 2025-07-01  
> **版本**: v1.0  

---

## Part A: System Design

### 1. Implementation Approach

#### 1.1 核心技术挑战

| # | 挑战 | 分析 | 解决策略 |
|---|------|------|----------|
| C1 | **连续状态映射** | ARC 的 Inflow 是离散像素网格，MuJoCo 的 Inflow 是连续 mjData（qpos/qvel/sensor），维度与量纲混杂 | 定义统一的 `sense()` 方法，将 mjData 各字段提取为标准化 `obs_dict`，加权组合为 η |
| C2 | **η 量纲归一化** | 位置(m)、角度(rad)、能量(J)、速度(m/s) 量纲不同，平方距 η 可能偏置 | `gauss_ex_residual` 采用可调权重 (w_pos, w_ori, w_eng, w_vel)，并支持 per-task 归一化系数；权重默认值已在代码骨架中给出 |
| C3 | **Noether 能量校验边界** | MuJoCo 接触力产生耗散，ΔE 的 "external work" 界定模糊 | 第一版采用保守策略：`ΔE ≤ max_energy_inject`（GoalEML 定义），不计碰撞耗散；碰撞单独由 `_min_geom_distance` 门控 |
| C4 | **Motor Primitive 与 IC 门控** | 原有 NARLA tile macro 是离散符号组合，连续域需参数化元动作 | 内嵌 `MotorPrimitives` 类，5 个元动作 + `pd_stabilize`，IC_Value_Score=ΔIC 门控决定是否执行或降级为 PD 探索 |
| C5 | **Baseline 公平对比** | PPO/SAC/TD-MPC2 训练预算与超参需统一 | `evaluate_vs_baseline.py` 提供 `register_baseline` decorator，统一 episode 数与 max_steps，baseline 使用 SB3 默认超参 + 1M step 训练预算 |

#### 1.2 框架与库选型

| 库 | 版本 | 用途 | 选型理由 |
|----|------|------|----------|
| `dm_control` | ≥1.0.0 | MuJoCo 环境封装 | dm_control 是 DeepMind 官方 MuJoCo Python wrapper，提供标准任务 suite（Humanoid, Hopper, Walker, Reacher），是学术基准主流 |
| `mujoco` | ≥2.3.0 | 物理引擎底层 | dm_control 的底层依赖，直接访问 `mjData`/`mjModel` |
| `numpy` | ≥1.24.0 | 数值计算 | η 计算、向量运算、矩阵操作 |
| `stable_baselines3` | ≥2.0.0 (可选) | PPO/SAC baseline | SB3 是 RL baseline 标准库，PPO/SAC 实现成熟，`evaluate_vs_baseline.py` 可直接调用 |
| `tdmpc2` | ≥1.0.0 (可选) | TD-MPC2 baseline | 论文对比所需的 model-based baseline |
| `dataclasses` | 标准库 | 数据结构定义 | GoalEML, NoetherViolation 等结构体 |

#### 1.3 架构模式

本项目为 **模块化管线架构**（Modular Pipeline），不采用 MVC/MVVM（无 GUI）。核心模式为：

- **Strategy Pattern**: GoalEML 定义任务策略，`make_*_eml()` factory 函数产出不同任务配置
- **Pipeline Pattern**: IDO Harness 五环节串行管线：Sense → κ-Snap → Dual-Path → Noether → Critique
- **Decorator Pattern**: `register_baseline` 允许第三方注册新 baseline
- **State Machine**: Agent 每步在 `(exploring / κ-snapping / noether-pruning / critiquing)` 状态间切换

---

### 2. File List

| # | 相对路径 | 类型 | 说明 |
|---|---------|------|------|
| 1 | `agent/mujoco_ido_agent.py` | ★ 新 | IDO MuJoCo Agent 核心类 + 内嵌 MotorPrimitives |
| 2 | `core/kappa_snap_mj.py` | ★ 新 | 连续 GaussEx 残差计算 |
| 3 | `core/noether_check_mj.py` | ★ 新 | 物理 Noether-Check 三重门 |
| 4 | `core/goal_eml_mj.py` | ★ 新 | Goal-EML 物理任务定义 + factory |
| 5 | `benchmarks/run_mujoco_bench.py` | ★ 新 | 一键跑分 CLI 脚本 |
| 6 | `benchmarks/evaluate_vs_baseline.py` | ★ 新 | IDO vs Baseline 对比评估 |
| 7 | `papers/mujoco_bench_ido_validation.md` | ★ 新 | 论文 Appendix C |
| 8 | `envs/dmctrl_wrapper.py` | ★ 新 | dm_control 环境统一封装（提取 obs/action/reward） |
| 9 | `envs/mujoco_visual.py` | ★ 新 | MuJoCo 可视化工具（可选渲染 + 录屏） |
| 10 | `benchmarks/report_ido_advantage.py` | ★ 新 | 结果汇总与 LaTeX 表格生成 |
| 11 | `agent/__init__.py` | ★ 新(更新) | 添加 mujoco_ido_agent 导出 |
| 12 | `core/__init__.py` | ★ 新(更新) | 添加 mj 模块导出 |
| 13 | `benchmarks/__init__.py` | ★ 新 | 包初始化 |
| 14 | `envs/__init__.py` | ★ 新 | 包初始化 |
| 15 | `benchmarks/results/.gitkeep` | ★ 新 | 结果输出目录占位 |

**不变的已有文件**：`agent/my_agent.py`, `core/kappa_snap.py`, `core/noether_check.py`, `core/goal_eml.py`, `core/narla_value.py`, `reference_notebooks/`, `papers/`（原有内容）

---

### 3. Data Structures and Interfaces

```mermaid
classDiagram
    class IDOMuJoCoAgent {
        +str agent_id
        +GoalEML goal_eml
        +MotorPrimitives motor_prims
        +float kappa_thresh
        +bool enable_critique
        +list~dict~ experience_buffer
        +list~dict~ replay_buffer
        +int step_count
        +__init__(goal_eml: GoalEML, kappa_thresh: float, enable_critique: bool)
        +sense(data: mjData) dict
        +compute_eta(data: mjData) float
        +dual_path_update(eta: float, data: mjData) ndarray
        +noether_filter(prev_data: mjData, cur_data: mjData) Tuple~bool, str~
        +critique_self_loop(trajectory: list) dict
        +step(data: mjData) ndarray
        +reset() None
        +load_oracle_replay(path: str) None
    }

    class MotorPrimitives {
        +dict primitives_registry
        +__init__()
        +step_forward(data: mjData, step_size: float) ndarray
        +step_left(data: mjData, step_size: float) ndarray
        +step_right(data: mjData, step_size: float) ndarray
        +squat(data: mjData, depth: float) ndarray
        +torque_explore(data: mjData, magnitude: float) ndarray
        +pd_stabilize(data: mjData, target_qpos: ndarray, kp: float, kd: float) ndarray
        +ic_gate(data: mjData, primitive_name: str) float
        +select_primitive(data: mjData, eta: float) Tuple~str, ndarray~
    }

    class GoalEML {
        +str name
        +dict invariants
        +ndarray target_pos
        +float delta_K
        +float max_energy_inject
        +float pos_tol
        +float ori_tol
    }

    class kappa_snap_mj {
        <<module>>
        +gauss_ex_residual(z_i: dict, goal: GoalEML, w_pos: float, w_ori: float, w_eng: float, w_vel: float) float
        +_quat_to_z_axis(quat: ndarray) ndarray
    }

    class noether_check_mj {
        <<module>>
        +noether_check_mj(prev_data: mjData, cur_data: mjData, goal: GoalEML) Tuple~bool, str~
        +_min_geom_distance(data: mjData) float
    }

    class NoetherViolation {
        +str gate_name
        +float magnitude
        +str detail
    }

    class DmCtrlWrapper {
        +str task_name
        +Env env
        +mjModel model
        +__init__(task_name: str, seed: int)
        +reset() Tuple~mjData, dict~
        +step(action: ndarray) Tuple~mjData, float, bool, dict~
        +get_obs(data: mjData) dict
        +get_reward(timestep: TimeStep) float
    }

    class MuJoCoVisual {
        +__init__(mode: str)
        +render(data: mjData) None
        +save_frame(path: str) None
        +close() None
    }

    IDOMuJoCoAgent --> GoalEML : uses
    IDOMuJoCoAgent --> MotorPrimitives : uses
    IDOMuJoCoAgent --> kappa_snap_mj : calls gauss_ex_residual
    IDOMuJoCoAgent --> noether_check_mj : calls noether_check_mj
    MotorPrimitives --> GoalEML : references invariants
    kappa_snap_mj --> GoalEML : reads target_pos + invariants
    noether_check_mj --> GoalEML : reads max_energy_inject + constraints
    DmCtrlWrapper --> IDOMuJoCoAgent : provides mjData
    NoetherViolation --> noether_check_mj : produced by
```

#### 关键接口详述

**`IDOMuJoCoAgent.sense(data: mjData) → dict`**

```python
def sense(self, data: mjData) -> dict:
    """从 mjData 提取标准化观测字典"""
    return {
        'qpos': np.array(data.qpos),          # 位置/角度
        'qvel': np.array(data.qvel),          # 速度
        'actuator_force': np.array(data.actuator_force),  # 执行器力
        'sensor_data': np.array(data.sensordata),         # 传感器
        'ee_pos': self._get_ee_pos(data),     # 末端执行器位置
        'ee_quat': self._get_ee_quat(data),   # 末端执行器姿态
        'energy': self._compute_energy(data), # 系统动能+势能
    }
```

**`kappa_snap_mj.gauss_ex_residual() → float`**

```python
def gauss_ex_residual(z_i: dict, goal: GoalEML,
                      w_pos=1.0, w_ori=0.3, w_eng=0.01, w_vel=0.05) -> float:
    """η = w_pos·‖ee−target‖² + w_ori·tilt² + w_eng·max(0,E−E_budget)² + w_vel·‖v_ee‖²"""
    pos_err = np.linalg.norm(z_i['ee_pos'] - goal.target_pos)
    z_axis = _quat_to_z_axis(z_i['ee_quat'])
    tilt = np.linalg.norm(z_axis - np.array([0, 0, 1]))
    energy_err = max(0, z_i['energy'] - goal.max_energy_inject)
    vel_err = np.linalg.norm(z_i['ee_vel'] if 'ee_vel' in z_i else z_i['qvel'][:3])
    return w_pos * pos_err**2 + w_ori * tilt**2 + w_eng * energy_err**2 + w_vel * vel_err**2
```

**`noether_check_mj.noether_check_mj() → Tuple[bool, str]`**

```python
def noether_check_mj(prev_data, cur_data, goal: GoalEML) -> Tuple[bool, str]:
    """三重守恒门：能量漂移 / 力矩限幅 / 自碰撞"""
    violations = []
    # Gate 1: Energy drift
    delta_E = cur_energy - prev_energy
    if delta_E > goal.max_energy_inject + ENERGY_TOLERANCE:
        violations.append(NoetherViolation("energy_drift", delta_E, ...))
    # Gate 2: Torque limit
    if np.any(np.abs(cur_data.actuator_force) > torque_limits + TORQUE_TOLERANCE):
        violations.append(NoetherViolation("torque_exceed", ...))
    # Gate 3: Self-collision
    min_dist = _min_geom_distance(cur_data)
    if min_dist < COLLISION_THRESHOLD:
        violations.append(NoetherViolation("self_collision", min_dist, ...))
    passed = len(violations) == 0
    detail = "; ".join(v.gate_name for v in violations) if violations else "OK"
    return (passed, detail)
```

**`GoalEML factory functions`**

```python
@dataclass
class GoalEML:
    name: str
    invariants: dict          # e.g. {'contact': ['feet_on_ground']}
    target_pos: np.ndarray    # target end-effector / body position
    delta_K: float            # allowed kinetic energy variation
    max_energy_inject: float  # max energy injection per step
    pos_tol: float            # position tolerance (m)
    ori_tol: float            # orientation tolerance (rad)

def make_humanoid_reach_eml() -> GoalEML: ...
def make_hopper_stand_eml() -> GoalEML: ...
def make_walker_run_eml() -> GoalEML: ...
def make_reacher_easy_eml() -> GoalEML: ...
```

**`DmCtrlWrapper`**

```python
class DmCtrlWrapper:
    """dm_control 环境统一封装"""
    def __init__(self, task_name: str, seed: int = 0):
        self.env = _import_env(task_name)(random=seed)
        self.model = self.env.physics.model
    
    def get_obs(self, data) -> dict:
        """提取 qpos/qvel/sensor/ee_pos/ee_quat/energy"""
        ...
```

---

### 4. Program Call Flow

```mermaid
sequenceDiagram
    participant CLI as run_mujoco_bench
    participant Wrap as DmCtrlWrapper
    participant Agent as IDOMuJoCoAgent
    participant Kappa as kappa_snap_mj
    participant Noether as noether_check_mj
    participant Goal as GoalEML
    participant Motor as MotorPrimitives

    CLI->>Goal: make_humanoid_reach_eml()
    Goal-->>CLI: goal_eml (target_pos, invariants, constraints)

    CLI->>Agent: IDOMuJoCoAgent(goal_eml, kappa_thresh=0.5, enable_critique=True)
    CLI->>Wrap: DmCtrlWrapper("Humanoid-stand", seed=42)
    CLI->>Wrap: reset()
    Wrap-->>CLI: mjData_0, obs_0

    loop Episode Steps (max_steps)
        CLI->>Agent: step(mjData_cur)
        Agent->>Agent: sense(mjData_cur) → obs_dict
        Agent->>Kappa: gauss_ex_residual(obs_dict, goal_eml)
        Kappa->>Goal: read target_pos, max_energy_inject
        Kappa-->>Agent: η (float)

        alt η > κ_thresh (exploring)
            Agent->>Motor: select_primitive(mjData_cur, η)
            Motor->>Motor: ic_gate(mjData_cur, primitive_name) → ΔIC
            Motor-->>Agent: gated_action (ndarray)
        else η ≤ κ_thresh (κ-snapping)
            Agent->>Agent: dual_path_update(η, mjData_cur) → PD refine action
        end

        Agent->>Wrap: step(action)
        Wrap-->>Agent: mjData_new, reward, done

        Agent->>Noether: noether_check_mj(mjData_cur, mjData_new, goal_eml)
        Noether->>Goal: read max_energy_inject, constraints
        Noether-->>Agent: (passed: bool, detail: str)

        alt Noether passed
            Agent->>Agent: accept step, append to trajectory
        else Noether violated
            Agent->>Agent: reject action, fallback to pd_stabilize
        end

        opt enable_critique
            Agent->>Agent: critique_self_loop(trajectory) → adjust weights
        end
    end

    CLI->>CLI: compute_metrics(episode_log)
    CLI-->>CLI: {steps, NVR, SER, success_rate}
```

#### Baseline 对比评估流程

```mermaid
sequenceDiagram
    participant Eval as evaluate_vs_baseline
    participant IDO as IDOMuJoCoAgent
    participant BL as BaselineAgent (PPO/SAC/TD-MPC2)
    participant Wrap as DmCtrlWrapper

    Eval->>Wrap: DmCtrlWrapper(task_name)
    Eval->>IDO: run N episodes → ido_logs
    Eval->>BL: train 1M steps → run N episodes → bl_logs

    loop Per Episode
        Eval->>IDO: collect (steps, violations, success)
        Eval->>BL: collect (steps, violations, success)
    end

    Eval->>Eval: compute_metrics(ido_logs) → ido_summary
    Eval->>Eval: compute_metrics(bl_logs) → bl_summary
    Eval->>Eval: compute_ser(ido_steps, bl_steps) → SER
    Eval->>Eval: Wilcoxon test → p_value

    Eval->>Eval: IDO Prophecy Verification
    Note over Eval: P1: IDO steps ↓ ≥30%? 
    Note over Eval: P2: IDO NVR=0, PPO NVR>0?
    Note over Eval: P4: SER≥1.2, p<.05?

    Eval-->>Eval: output JSON + CSV + console summary
```

---

### 5. Anything UNCLEAR

| # | 问题 | 当前假设 | 建议 |
|---|------|----------|------|
| U1 | **Goal-EML 陪集粒度**：各任务的 invariants 与 target_pos 精确值 | 默认值由 factory 函数硬编码（参考 dm_control reward 定义），pos_tol=0.05m, ori_tol=0.1rad | 需实验校准，factory 函数支持参数覆盖 |
| U2 | **η 归一化系数**：各任务 η 的权重 w_pos/w_ori/w_eng/w_vel | 使用代码骨架默认值 (1.0/0.3/0.01/0.05)，不做额外归一化 | 实验后可能需要 per-task 调权；架构预留 `GoalEML.invariants['weights']` 字段 |
| U3 | **Noether 能量校验边界**：碰撞耗散是否计入 external work | 保守策略：不计碰撞耗散，仅校验 `ΔE ≤ max_energy_inject` | 第一版简化，后续可增加 `contact_energy_loss` 项 |
| U4 | **Baseline 训练预算**：PPO/SAC/TD-MPC2 的超参与训练量 | SB3 默认超参 + 1M step 训练，dm_control 标准 reward | 可通过 `register_baseline` 的 `train_config` 参数调整 |
| U5 | **MotorPrimitives 与 Expert Replay 依赖关系**：是否为 P0 必需 | MotorPrimitives 内嵌于 Agent（P0），Expert Replay 为 P1 可选 | Expert Replay 缺失时 Agent 用 `pd_stabilize` 替代初始化 |
| U6 | **narla_value.py 复用方式**：IC_Value_Score=ΔIC 如何在连续域计算 | 复用原有 narla_value.py 的 ΔIC 计算，输入从离散 grid 改为连续 obs_dict | 需确认 narla_value.py 的接口是否可接受连续输入 |

---

## Part B: Task Decomposition

### 6. Required Packages

```
- dm_control@>=1.0.0: MuJoCo 环境封装（DeepMind 官方 wrapper）
- mujoco@>=2.3.0: 物理引擎底层
- numpy@>=1.24.0: 数值计算核心
- dataclasses: 标准库（GoalEML, NoetherViolation 定义）
- argparse: 标准库（CLI 参数解析）
- json: 标准库（结果输出）
- csv: 标准库（CSV 输出）
- scipy@>=1.10.0: 统计检验（Wilcoxon, t-test）
- stable_baselines3@>=2.0.0 (可选): PPO/SAC baseline 实现
- tdmpc2@>=1.0.0 (可选): TD-MPC2 baseline 实现
```

### 7. Task List (ordered by dependency)

| Task ID | Task Name | Source Files | Dependencies | Priority |
|---------|-----------|-------------|--------------|----------|
| T01 | **项目基础设施与环境封装** | `envs/__init__.py`, `envs/dmctrl_wrapper.py`, `envs/mujoco_visual.py`, `benchmarks/__init__.py`, `benchmarks/results/.gitkeep`, `agent/__init__.py`(更新), `core/__init__.py`(更新), `pyproject.toml`(更新依赖) | 无 | P0 |
| T02 | **核心数据层（Goal-EML + κ-Snap + Noether-Check）** | `core/goal_eml_mj.py`, `core/kappa_snap_mj.py`, `core/noether_check_mj.py` | T01 | P0 |
| T03 | **IDO Agent + Motor Primitives** | `agent/mujoco_ido_agent.py`, `core/narla_value.py`(复用适配), `envs/dmctrl_wrapper.py`(集成测试) | T02 | P0 |
| T04 | **跑分与评估管线** | `benchmarks/run_mujoco_bench.py`, `benchmarks/evaluate_vs_baseline.py`, `benchmarks/report_ido_advantage.py` | T03 | P1 |
| T05 | **论文附录与集成验证** | `papers/mujoco_bench_ido_validation.md`, `ARCHITECTURE.md`(追加), 全项目集成联调 | T04 | P2 |

#### Task 详情

**T01: 项目基础设施与环境封装**

- 创建 `envs/` 子包与 `benchmarks/` 子包的 `__init__.py`
- 实现 `DmCtrlWrapper`：统一加载 dm_control 环境，封装 `reset()`/`step()`/`get_obs()`/`get_reward()`，为 Agent 和 Benchmark 提供标准化接口
- 实现 `MuJoCoVisual`：可选渲染与帧保存，用于 debug 和论文素材
- 更新 `pyproject.toml` 添加 dm_control/mujoco/numpy/scipy 等依赖
- 更新 `agent/__init__.py` 和 `core/__init__.py` 添加 mj 模块导出

**T02: 核心数据层（Goal-EML + κ-Snap + Noether-Check）**

- 实现 `GoalEML` dataclass 与 4 个 factory 函数（`make_humanoid_reach_eml`, `make_hopper_stand_eml`, `make_walker_run_eml`, `make_reacher_easy_eml`）
- 实现 `gauss_ex_residual()`：加权平方距 η 计算 + `_quat_to_z_axis()` 辅助
- 实现 `noether_check_mj()`：三重守恒门（能量漂移/力矩限幅/自碰撞）+ `NoetherViolation` dataclass + `_min_geom_distance()` 辅助
- 单元测试验证各模块独立正确性

**T03: IDO Agent + Motor Primitives**

- 实现 `IDOMuJoCoAgent` 类：sense → compute_eta → dual_path_update → noether_filter → critique_self_loop 五环节循环
- 内嵌 `MotorPrimitives` 类：5 个元动作 + `pd_stabilize` + `ic_gate` + `select_primitive`
- 复用/适配 `core/narla_value.py` 的 ΔIC 计算接口
- 集成测试：Agent + DmCtrlWrapper 在至少 1 个 dm_control 任务上跑通完整 episode

**T04: 跑分与评估管线**

- 实现 `run_mujoco_bench.py`：TASK_REGISTRY, `_import_env()`, `run_single_episode()`, `run_benchmark()`, argparse CLI
- 实现 `evaluate_vs_baseline.py`：BASELINE_REGISTRY, `register_baseline` decorator, `compute_metrics()`, `compute_ser()`, `run_evaluation()`, IDO Prophecy Verification, JSON + CSV 输出
- 实现 `report_ido_advantage.py`：汇总结果生成 LaTeX 表格与 console summary
- 在 4 个标准任务上运行 IDO，确认跑分脚本可用

**T05: 论文附录与集成验证**

- 撰写 `papers/mujoco_bench_ido_validation.md`：完整实验结果、预言验证分析、方法论描述
- 追加 `ARCHITECTURE.md` MuJoCo-Bench-IDO 章节
- 全项目集成联调：确认 IDO Agent + 跑分 + 评估 + 可视化全链路可用
- 最终验证：IDO 在 ≥3 个任务上 P1/P2/P4 预言通过

### 8. Shared Knowledge

```
- 所有 mjData 提取的 obs_dict 格式统一：{'qpos', 'qvel', 'actuator_force', 'sensor_data', 'ee_pos', 'ee_quat', 'energy'}
- η 计算默认权重：w_pos=1.0, w_ori=0.3, w_eng=0.01, w_vel=0.05，可通过 GoalEML.invariants['weights'] 覆盖
- Noether-Check 容差：ENERGY_TOLERANCE=0.1J, TORQUE_TOLERANCE=0.05*N·m, COLLISION_THRESHOLD=0.01m
- 所有 baseline 统一训练预算：1M steps，SB3 默认超参
- 评估输出格式：JSON {task, agent, episodes, metrics: {steps, NVR, SER, success_rate}}
- 日期时间存储格式：ISO 8601 UTC
- GoalEML factory 函数参数可覆盖：make_*_eml(**kwargs) 支持自定义 target_pos/tol
- motor primitive action 输出维度与 dm_control env.action_spec() 一致
- Critique self-loop 仅在 enable_critique=True 时激活，默认 False
```

### 9. Task Dependency Graph

```mermaid
graph TD
    T01["T01: 项目基础设施与环境封装<br/>(envs, __init__, pyproject)"]
    T02["T02: 核心数据层<br/>(goal_eml_mj, kappa_snap_mj, noether_check_mj)"]
    T03["T03: IDO Agent + Motor Primitives<br/>(mujoco_ido_agent, narla_value适配)"]
    T04["T04: 跑分与评估管线<br/>(run_bench, evaluate_vs_baseline, report)"]
    T05["T05: 论文附录与集成验证<br/>(validation.md, ARCHITECTURE.md)"]
    
    T01 --> T02
    T02 --> T03
    T01 --> T03
    T03 --> T04
    T04 --> T05
```

---

### v0.3.0 新增模块（Baseline集成 + Web可视化）

#### baselines/ 目录

| 文件 | 类型 | 说明 |
|------|------|------|
| `baselines/__init__.py` | ★ 新 | 包初始化 |
| `baselines/tdmpc2_adapter.py` | ★ 新 | TD-MPC2 v2 baseline adapter（model-based RL控制对比） |
| `baselines/cosmos_predict_adapter.py` | ★ 新 | Cosmos-Predict世界模型adapter（η轨迹预测对比） |

**TDMPC2Adapter**：
- 统一接口：choose_action(obs), evaluate(n_episodes), reset()
- 任务名映射：humanoid-stand → humanoid_stand, 等
- 模型尺寸：1M/5M/19M/48M/317M参数
- 注册为"tdmpc2_v2"在BASELINE_REGISTRY
- 优雅降级：tdmpc2未安装时返回None

**CosmosPredictAdapter**：
- 世界模型baseline（非控制agent）——η轨迹预测对比
- 模型变体：7B/14B video2world, 7B token2world
- 注册为"cosmos-predict"在BASELINE_REGISTRY
- 需GPU + CUDA（7B-14B参数模型）
- 优雅降级：未安装时跳过

#### webviz/ 目录

| 文件 | 类型 | 说明 |
|------|------|------|
| `webviz/__init__.py` | ★ 新 | 包初始化 |
| `webviz/server.py` | ★ 新(v0.3.0修复) | FastAPI REST API + WebSocket + mjviser服务 |
| `webviz/dashboard.html` | ★ 新 | 实时监控仪表盘HTML（Chart.js + WebSocket） |
| `webviz/user_manual.html` | ★ 新(v0.3.0) | 用户手册HTML版本（从Markdown转换，深色主题单文件，左侧目录导航） |
| `webviz/mujoco_docs_cn.html` | ★ 新(v0.3.0) | MuJoCo官方文档Overview中文翻译版（深色主题单文件，左侧目录导航） |
| `webviz/run_webviz.py` | ★ 新 | uvicorn启动脚本 |

**server.py v0.3.0 mjviser Bug修复**：
- Bug A：Viewer.__init__不接受port参数 → 先创建ViserServer(port=8081)
- Bug B：env.model不存在 → env.physics.model._model
- Bug C：缺少data参数 → env.physics.data._data

**评估模式**：

| 模式 | CLI参数 | 对比内容 |
|------|---------|----------|
| control | --eval-mode control | IDO vs TD-MPC2/PPO/SAC（步数, NVR, SER） |
| cosmos-predict | --eval-mode cosmos-predict | IDO FlowMatching η vs Cosmos-Predict η轨迹 |

#### v0.3.0 模块关系图

```mermaid
graph TB
    subgraph Core
        GoalEML["GoalEML<br/>core/goal_eml_mj.py"]
        KappaSnap["κ-Snap + FlowMatching<br/>core/kappa_snap_mj.py"]
        Noether["Noether-Check<br/>core/noether_check_mj.py"]
    end
    
    subgraph Agent
        IDOAgent["IDOMuJoCoAgent<br/>agent/mujoco_ido_agent.py"]
        PsiAnchor["ψ-Anchor<br/>agent/psi_anchor.py"]
    end
    
    subgraph Baselines
        TDMPC2["TDMPC2Adapter<br/>baselines/tdmpc2_adapter.py"]
        Cosmos["CosmosPredictAdapter<br/>baselines/cosmos_predict_adapter.py"]
    end
    
    subgraph Webviz
        FastAPI["FastAPI Server<br/>webviz/server.py"]
        Dashboard["Dashboard<br/>webviz/dashboard.html"]
        Mjviser["mjviser Viewer<br/>webviz/server.py (launch_viewer)"]
        UserManual["用户手册HTML<br/>webviz/user_manual.html"]
        MuJoCoDocs["MuJoCo中文文档<br/>webviz/mujoco_docs_cn.html"]
    end
    
    subgraph Benchmarks
        RunBench["run_mujoco_bench.py"]
        EvalVS["evaluate_vs_baseline.py"]
    end
    
    IDOAgent --> GoalEML
    IDOAgent --> KappaSnap
    IDOAgent --> Noether
    IDOAgent --> PsiAnchor
    PsiAnchor --> KappaSnap
    KappaSnap --> GoalEML
    Noether --> GoalEML
    
    RunBench --> IDOAgent
    EvalVS --> IDOAgent
    EvalVS --> TDMPC2
    EvalVS --> Cosmos
    
    FastAPI --> Dashboard
    FastAPI --> RunBench
    FastAPI --> UserManual
    FastAPI --> MuJoCoDocs
    Dashboard --> UserManual
    Dashboard --> MuJoCoDocs
    Mjviser --> FastAPI
```

---

*Document by 高见远（Gao） — Architect*
*Date: 2025-07-01*
*v0.3.0 update: 2025-07-01*
*v0.3.1 update: 2025-07-01 — Language switching, real-time 3D simulation, obstacle scene*
*v0.4.2 update: 2025-07-01 — Three-layer PD standing controller (gravity comp + root orientation + joint PD)*
*v0.4.3 update: 2025-07-01 — Random walking controller + waypoint navigation + obstacle arena (replaces hard-lock root)*
*v0.4.4 update: 2025-07-02 — Expanded dashboard task/scene selection + explanatory tooltips + 4 new 3D arenas*
*v0.4.5 update: 2025-07-02 — Floating bug fix (root vertical force → adaptive gravity assist + leg joint ground support)*

---

### v0.3.1 新增模块（语言切换 + 实时3D仿真 + 障碍物场景）

#### webviz/scenes/ 目录

| 文件 | 类型 | 说明 |
|------|------|------|
| `webviz/scenes/__init__.py` | ★ 新 | 包初始化 |
| `webviz/scenes/humanoid_obstacle_arena.xml` | ★ 新 | MuJoCo XML障碍物竞技场场景（包含墙体、圆柱、方块障碍物+红色目标球+火柴人形机器人） |

#### 语言切换模块

| 修改文件 | 说明 |
|---------|------|
| `webviz/dashboard.html` | 新增 i18n 模块：data-i18n 属性 + i18nDict 字典 + toggleLanguage() 函数，localStorage key: `mujoco-bench-lang` |
| `webviz/user_manual.html` | 新增 i18n 模块：所有标题/段落/列表/表格加 data-i18n，右上角中/EN切换按钮 |
| `webviz/mujoco_docs_cn.html` | 新增 i18n 模块：侧边栏+节标题加 data-i18n，右上角中/EN切换按钮 |

**语言切换设计**：
- localStorage key统一为 `mujoco-bench-lang`，值 `zh`/`en`
- 页面加载时读取 localStorage，默认中文
- 切换按钮文案："中/EN"，放在右上角固定位置
- 代码块（XML/C）不翻译，只翻译周围说明文字
- 翻译质量面向学术/工程师

#### 实时3D仿真循环（v0.4.3 随机走动控制器 + 航点导航）

| 修改文件 | 说明 |
|---------|------|
| `webviz/server.py` | `launch_viewer()` 重写：随机走动控制器 + 航点导航 + 障碍物竞技场 + 手动 viewer loop + 暂停启动 |
| `webviz/scenes/humanoid_obstacle_arena.xml` | 障碍物改为静态（移除free joint），调整位置适配走动范围，增加地面尺寸 |

**仿真循环设计（v0.4.3 随机走动控制器）**：
- `viewer.run()` 在后台线程中调用 `signal.signal()` 会抛 ValueError → 手动 `_setup_gui()+_render()+_tick()` 循环规避
- **统一走动控制器** `step_fn`：同时适用于 plain 和 obstacle 场景
  - 不再硬锁定root — 机器人可自由移动
  - 7层控制架构：
    1. **Root高度**: 重力补偿 + PD弹簧 → `qfrc_applied[2] = m·g + KP·(h_target - z) - KD·vz`
    2. **Root水平移动**: 速度PD朝航点 → `qfrc_applied[0:2] = KP·(v_desired - v_current)`
    3. **Root朝向稳定**: z轴倾斜PD → `qfrc_applied[3:4] = -KP·tilt - KD·ω`
    4. **Root航向转向**: heading PD朝航点 → `qfrc_applied[5] = KP·Δθ - KD·ω_yaw`
    5. **行走步态**: 正弦髋/膝振荡（左右交替相位），手臂对侧摆动
    6. **关节稳定**: 非行走关节PD锁定初始姿态
    7. **安全恢复**: 高度过低时施加强向上力
  - **航点机制**: 每5秒随机更换目标点（x,y ∈ [-6, 6]），随机种子默认42（可通过 `MUJOCO_BENCH_WALK_SEED` 环境变量覆盖）
  - **PD增益质量比例**: root增益 = mass × 常数，关节增益固定 KP=50 KD=15
  - **球关节控制**: 使用四元数相对旋转 + 小角度近似（θ ≈ 2·quat_rel[1:4]）实现PD
  - **dm_control humanoid**: 21个弱执行器绕过 → 全部通过 `qfrc_applied` 驱动
  - **obstacle场景火柴人**: 同样通过 `qfrc_applied` 驱动，16kg总质量
  - 启动暂停模式：用户看到直立姿态，点 Play 后机器人开始走动
- obstacle 场景改进：
  - 障碍物改为静态（移除free joint），防止漂移
  - 地面尺寸从10×10增大到12×12，适配航点范围 [-6, 6]
  - 障碍物位置重新分布：墙、圆柱、方块散布在整个走动区域
  - 移除障碍物间 `<exclude>`（静态物体不需要）
- ViserServer 端口：从 `viser_server._websock_server._port` 读取实际端口
- viewer 关闭时 `viser_server.stop()` + `mjviser_viewer_running = False`

> **Bug 历史**：
> - v0.3.1：独立 `sim_loop` daemon 线程与 Viewer `_tick()` 并发写入同一 MjData → 数据竞争
> - v0.4.1：`viewer.run()` 在后台线程崩溃 + 随机动作导致机器人疯狂抽搐
> - v0.4.2a：仅执行器关节PD → 重力使free root关节下坠，机器人倒地
> - v0.4.2b：三层PD站立控制器 → 重力补偿+朝向稳定+关节锁定，但root_xy漂移+周期动作太闪
> - v0.4.2c：硬锁定root（每步重置qpos/qvel）+ 关节PD → 零漂移零抽搐，稳如磐石20s，但机器人被钉在原地不动
> - v0.4.3：随机走动控制器 + 航点导航 → 机器人直立走动，朝随机航点移动，不抽搐不躺倒

#### 3D场景选择

| 修改文件 | 说明 |
|---------|------|
| `webviz/server.py` | 新增全局变量 `mjviser_scene_type: str = "plain"`，新增 `SceneRequest` Pydantic model，新增 `/api/mjviser/scene` POST endpoint |
| `webviz/dashboard.html` | 左侧控制面板新增"3D场景"下拉框（选项：plain/obstacle），通过 `/api/mjviser/scene` API保存 |

#### v0.4.4 Dashboard 扩展

| 修改文件 | 说明 |
|---------|------|
| `webviz/dashboard.html` | 任务下拉扩展为分组选择（Humanoid/Walker/Cheetah/Hopper/Cartpole/Reacher/Manipulator/Swimmer/Classic）；3D场景下拉扩展为 6 个（plain/obstacle/ramp/stairs/floating/maze）；新增"当前场景"显示卡片；为每个控制区增加说明文字（中英文 i18n） |
| `webviz/server.py` | `/api/mjviser/scene` 支持 6 种场景；`launch_viewer()` 通过场景文件映射加载 XML；`/api/tasks` 返回完整任务描述；版本号 v0.4.4 |
| `webviz/scenes/humanoid_ramp_arena.xml` | 斜坡竞技场：倾斜坡道 + 通道侧墙 |
| `webviz/scenes/humanoid_stairs_arena.xml` | 阶梯地形：5 级上升台阶 |
| `webviz/scenes/humanoid_floating_platforms.xml` | 浮台地形：6 个高低错落的平台 |
| `webviz/scenes/humanoid_maze_arena.xml` | 迷宫场景：墙体围成的 corridors |

**新增 3D 场景映射**（`webviz/server.py` `launch_viewer`）：
- `plain` → dm_control `humanoid-stand`
- `obstacle` / `ramp` / `stairs` / `floating` / `maze` → `webviz/scenes/humanoid_*.xml`

**新增任务分组**（按 `benchmarks/run_mujoco_bench.py` 的 `TASK_REGISTRY`）：
- Humanoid: humanoid-stand, humanoid-walk, humanoid-run
- Walker: walker-stand, walker-walk, walker-run
- Cheetah: cheetah-run
- Hopper: hopper-stand, hopper-hop
- Cartpole: cartpole-balance, cartpole-swingup, cartpole-balance_sparse, cartpole-swingup_sparse
- Reacher: reacher-easy, reacher-hard
- Manipulator / Hand: manipulator-bring_ball, finger-spin, finger-turn_easy, finger-turn_hard, ball_in_cup-catch
- Swimmer / Fish: fish-swim, swimmer-swim6, swimmer-swim15
- Classic Control: pendulum-swingup, acrobot-swingup

#### 导航链接修复

| 修改文件 | 说明 |
|---------|------|
| `webviz/dashboard.html` | 导航链接从相对路径改为绝对URL `http://localhost:8080/user_manual.html` 和 `http://localhost:8080/mujoco_docs_cn.html` |

#### v0.4.5 漂浮Bug修复 + 3D场景中文i18n

| 修改文件 | 说明 |
|---------|------|
| `webviz/server.py` | Bug G（漂浮）修复：root垂直力从100%重力补偿+强PD弹簧改为自适应重力辅助（近地面50%/漂浮10%/过渡30%）+温和PD；腿部关节KP从50提至100、KD从15提至25；添加支撑相位逻辑（近地面boost腿伸展、漂浮时减弱）；水平移动KP从mass×5降至mass×1.5（步态驱动而非漂浮推力）；安全恢复力从mass×50降至mass×20、MIN_HEIGHT从0.5×target降至0.3×target；target_height从1.4降至1.28（dm_control）和1.0降至0.85（stick-figure）；版本号v0.4.5 |
| `webviz/dashboard.html` | 3D场景选项value与API端点对齐（obstacle_arena→obstacle等）；i18n dict新增label_current_scene键和短名scene键（scene_obstacle/scene_ramp/scene_stairs/scene_floating/scene_maze）及对应badge键；版本号v0.4.5 |
| `webviz/run_webviz.py` | 版本号v0.4.5 |

**Bug G（漂浮）根因与修复**：

根因：v0.4.3的step_fn在L1层使用`qfrc_applied[2] = m·g + KP·(h-z) - KD·vz`，完全抵消重力并加强PD弹簧推向target_height=1.4m，导致机器人像气球悬浮，脚不接触地面。

修复策略（v0.4.5）：
1. **自适应重力辅助**：不再100%抵消重力。近地面时50%辅助（站立稳定），漂浮时10%辅助（惩罚悬浮），过渡区30%
2. **温和高度PD**：KP从mass×100降至mass×15，不再强弹簧推至target
3. **腿部关节增强+支撑相位**：Hip/Knee KP从50→100、KD从15→25；支撑相位（root_z < target+0.05）boost腿伸展力×1.5、减少步态摆动；漂浮（root_z > target+0.1）减弱腿力×0.5
4. **水平移动步态驱动**：KP_MOVE从mass×5降至mass×1.5，机器人通过步态前进而非被推着飘
5. **安全恢复温和化**：恢复力从mass×50降至mass×20，MIN_HEIGHT从0.5×target降至0.3×target
6. **target_height修正**：dm_control humanoid 1.28（自然站立高度），stick-figure 0.85

#### v0.4.5 P0 评估基础设施修复（PRD盲区修复）

基于 PRD v0.4.5_incremental.md 识别的4大评估盲区（B1-B4），实施5个P0修复：

**P0-1: 修复 dm_control 累计 reward 计算**

| 修改文件 | 说明 |
|---------|------|
| `benchmarks/run_mujoco_bench.py` | `run_single_episode()` 中初始化 `episode_return=0.0`，每步累计 `episode_return += float(timestep.reward or 0.0)`，返回 `episode_return` 替代旧 `avg_return`（仅取最后一步） |
| `benchmarks/evaluate_vs_baseline.py` | `_aggregate_metrics()` 中 `avg_return` 改为 mean episode_return |
| `webviz/server.py` | 流式 episode endpoint 也累计 episode_return |

根因：dm_control 的 `timestep.reward` 是单步奖励（float），而非 episode 累计。旧代码仅取最后一步 reward，无法与 baseline 做 episode return 对比。

**P0-2: 修复成功率 per-task 定制判断**

| 修改文件 | 说明 |
|---------|------|
| `benchmarks/run_mujoco_bench.py` | 新增 `TASK_SUCCESS_CRITERIA` 字典，为 27 个 dm_control 任务定制 success lambda（如 `reacher-easy: reward > -0.01`，`humanoid-stand: reward > 0.5`，`cartpole-balance: reward > 0.95` 等）；`run_single_episode()` 中每步判断 `success_fn(obs, step_reward)` 并记录 `success: bool` 字段 |

根因：旧代码仅 humanoid-stand 有 right_hand 目标距离判断，其他 24 个任务缺失 goal-achieved 判断，成功率统计不可信。

**P0-3: NVR 类型细分统计**

| 修改文件 | 说明 |
|---------|------|
| `core/noether_check_mj.py` | `noether_check_mj()` 返回值从 `(bool, str)` tuple 改为 `{ok, total, energy, torque, collision, message}` dict；每个守恒门检查分别记录各类型违规数 |
| `benchmarks/run_mujoco_bench.py` | 每步收集 NVR breakdown，episode 结果增加 `nvr_breakdown: {energy, torque, collision}` 字段 |
| `benchmarks/evaluate_vs_baseline.py` | NVR 统计使用新 breakdown 数据 |

根因：旧代码仅返回总违规数 `n_violations`，无法区分 energy/torque/collision 各类型的违规比例，NVR 指标缺乏诊断价值。

**P0-4: 安装 SB3 + dm_control wrapper**

| 修改项 | 说明 |
|--------|------|
| stable-baselines3 | 安装 v2.9.0 到 managed venv |
| shimmy[gymnasium] | 安装 v2.0.1（`DmControlCompatibilityV0` API） |

根因：旧代码中 PPO/SAC baseline 因未安装 SB3 而退化为 random fallback，评估退化为 IDO vs Random。

**P0-5: SB3PPOAdapter/SB3SACAdapter + train_baselines.py**

| 新增文件 | 说明 |
|---------|------|
| `baselines/sb3_adapter.py` | SB3PPOAdapter + SB3SACAdapter：auto-train（默认100K steps可配置）、checkpoint save/load、5 episodes evaluate、`choose_action()` 返回实际策略动作（不再fallback random）；dm_control obs 通过 gymnasium.spaces.flatten 转换 |
| `benchmarks/train_baselines.py` | 批量训练脚本：支持8个核心任务（humanoid-stand/walker-walk/cheetah-run/hopper-stand/reacher-easy/cartpole-balance/finger-turn_easy/fish-swim），PPO+SAC，checkpoint 保存到 `checkpoints/<task>/<algo>/`，自动评估5 episodes，JSON结果输出 |
| `baselines/__init__.py` | 导出 SB3PPOAdapter/SB3SACAdapter + factory 函数 |

| 修改文件 | 说明 |
|---------|------|
| `benchmarks/evaluate_vs_baseline.py` | PPO/SAC baseline 使用 SB3PPOAdapter/SB3SACAdapter 替代旧 PPO.load()（旧代码因 SB3 未安装而 fallback random） |

根因：旧 evaluate_vs_baseline.py 中 PPO/SAC 实际不可运行，退化为 random baseline。

#### v0.3.1 模块关系图更新

```mermaid
graph TB
    subgraph Core
        GoalEML["GoalEML"]
        KappaSnap["κ-Snap"]
        Noether["Noether-Check"]
    end

    subgraph Agent
        IDOAgent["IDOMuJoCoAgent"]
        PsiAnchor["ψ-Anchor"]
    end

    subgraph Webviz
        FastAPI["FastAPI Server<br/>webviz/server.py"]
        Dashboard["Dashboard<br/>webviz/dashboard.html"]
        Mjviser["mjviser Viewer<br/>+ random walking + waypoint"]
        UserManual["用户手册<br/>webviz/user_manual.html"]
        MuJoCoDocs["MuJoCo文档<br/>webviz/mujoco_docs_cn.html"]
        ObstacleScene["障碍物场景<br/>webviz/scenes/humanoid_obstacle_arena.xml"]
        I18nModule["语言切换模块<br/>data-i18n + i18nDict"]
    end

    FastAPI --> Dashboard
    FastAPI --> Mjviser
    FastAPI --> ObstacleScene
    Dashboard --> I18nModule
    UserManual --> I18nModule
    MuJoCoDocs --> I18nModule
    Dashboard -->|"http://localhost:8080"| UserManual
    Dashboard -->|"http://localhost:8080"| MuJoCoDocs
    Mjviser --> FastAPI
```

---

### v0.4.0 理论模块 (规划中)

| 文件 | 状态 | 说明 |
|------|------|------|
| `papers/mujoco_bench_ido_validation.md` C.20 | ★ 新增 | 皮克定理作为离散几何先验 (Pick↔Noether理论桥、Hex推广、加权Pick↔ψ-Anchor) |
| `papers/mujoco_bench_ido_中文论文.md` §3.9 | ★ 新增 | 皮克定理：离散几何先验与IDO理论桥 |
| `docs/用户手册.md` §7 Pick子节 | ★ 新增 | 皮克定理通俗解释 + IDO连接表格 |

#### v0.4.0 模块关系图更新

```mermaid
graph TB
    subgraph Core
        GoalEML["GoalEML"]
        KappaSnap["κ-Snap"]
        Noether["Noether-Check"]
    end

    subgraph Theory_v040
        PickCheck["Pick-Check<br/>格点守恒校验"]
    end

    subgraph Agent
        IDOAgent["IDOMuJoCoAgent"]
        PsiAnchor["ψ-Anchor"]
    end

    subgraph Webviz
        FastAPI["FastAPI Server<br/>webviz/server.py"]
        Dashboard["Dashboard<br/>webviz/dashboard.html"]
        Mjviser["mjviser Viewer<br/>+ random walking + waypoint"]
        UserManual["用户手册<br/>webviz/user_manual.html"]
        MuJoCoDocs["MuJoCo文档<br/>webviz/mujoco_docs_cn.html"]
        ObstacleScene["障碍物场景<br/>webviz/scenes/humanoid_obstacle_arena.xml"]
        I18nModule["语言切换模块<br/>data-i18n + i18nDict"]
    end

    Noether --> PickCheck
    KappaSnap --> PickCheck
    IDOAgent --> GoalEML
    IDOAgent --> KappaSnap
    IDOAgent --> Noether
    IDOAgent --> PsiAnchor
    IDOAgent --> PickCheck
    PsiAnchor --> KappaSnap
    KappaSnap --> GoalEML
    Noether --> GoalEML

    FastAPI --> Dashboard
    FastAPI --> Mjviser
    FastAPI --> ObstacleScene
    Dashboard --> I18nModule
    UserManual --> I18nModule
    MuJoCoDocs --> I18nModule
    Dashboard -->|"http://localhost:8080"| UserManual
    Dashboard -->|"http://localhost:8080"| MuJoCoDocs
    Mjviser --> FastAPI
```
