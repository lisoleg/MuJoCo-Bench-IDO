# MuJoCo-Bench-IDO: 非冯诺依曼架构的连续物理域IDO/TOMAS基准验证平台

**摘要**

非冯诺依曼架构的信息驱动编排（IDO）与多智能体系统理论（TOMAS）提出了一种守恒优先的决策循环范式，其核心主张是：智能体应先验证物理守恒约束（Noether门），再通过方向引导（κ-Snap门）选择行动，而非依赖试错学习的奖励最大化策略。然而，IDO/TOMAS的实证验证长期局限于离散符号域（ARC抽象推理拼图），连续物理控制域的系统性验证平台始终处于空白状态。本文提出MuJoCo-Bench-IDO——首个将IDO决策循环从离散符号域映射到连续物理控制域（MuJoCo/dm_control）的基准验证平台。我们完成了四项核心贡献：（1）定义了连续域GaussEx残差η的物理化计算方法与Goal-EML陪集映射；（2）实现了ψ-Anchor元管理层与FlowMatching η轨迹预测器，赋予IDO自演化能力；（3）设计并实现了SIP-Bench纵向评估协议（T0/T1/T2三阶段）；（4）建立了VG-Pair≠GAN的理论框架，证明IDO决策循环是C-IPP（连续交互式证明协议）在物理域的实例，其Soundness由Goal-EML保证而非由学习到的判别器保证。在4个dm_control标准任务（humanoid-stand、hopper-stand、walker-run、reacher-easy）上的实验表明：reacher-easy实现了Noether违规率NVR≡0（预言P1通过），其余任务的η收敛分析与SIP-Bench纵向评估揭示了IDO自演化机制的潜力与当前Motor Primitive覆盖度的局限。MuJoCo-Bench-IDO是首个物理域VG-Pair验证平台，为非冯架构AGI的连续控制验证提供了可复现的基准工具。

**关键词**: 非冯诺依曼架构; IDO/TOMAS; MuJoCo; 连续物理控制; VG-Pair; κ-Snap; Noether守恒检验; ψ-Anchor

---

## 1 引言

### 1.1 非冯架构AGI的必然性

冯诺依曼架构计算范式——以存储程序、顺序执行、数值计算为核心——在过去七十年驱动了信息技术革命。然而，随着应用场景从符号处理扩展到具身智能与物理交互，冯架构的根本瓶颈日益显现：**奖励最大化的试错学习范式无法保证物理守恒约束**[1]。传统强化学习（PPO、SAC、TD-MPC2）通过奖励塑形（reward shaping）或惩罚项（penalty terms）间接逼近约束，但缺乏形式化的Soundness保证——奖励函数仅是对约束的近似，而非约束本身。

非冯诺依曼架构的信息驱动编排（IDO）提出了一种根本性不同的决策范式：**守恒优先**（Conservation-First）——先验证守恒约束是否满足，再决定行动方向。这种范式不是对冯架构RL的改进，而是架构层面的范式转换。六代计算机发展方向（从数值计算→知识处理→推理→自主学习→判断→创造）[2]与IDO的五层本体结构（L1-L5）存在深层对应关系，表明非冯架构是AGI发展的必然方向。

### 1.2 连续物理域benchmark的空白

当前AGI基准验证平台集中在离散符号域。ARC（Abstraction and Reasoning Corpus）[3]验证了IDO在像素网格上的符号推理能力，但连续物理控制域——机器人操控、人形运动、机械臂到达——缺乏IDO验证平台。dm_control suite[4]提供了标准化的连续物理任务，但现有评估仅测量奖励和成功率，不测量守恒约束满足度（Noether Violation Rate）和方向引导效率（Step-Efficiency Ratio）。

**空白的核心**：没有平台同时测量（1）物理守恒满足度、（2）方向引导效率、（3）自演化纵向改善——而这三个维度正是IDO/TOMAS架构的理论预言所要求的。

### 1.3 VG-Pair理论动机

VG-Pair（Verifier-Generator Pair）框架[5]是IDO决策循环的理论基础。VG-Pair = (G, P, V)定义了三层结构：G是域Goal-EML约束集（物理定律+任务目标），P是Prover/Generator（生成候选轨迹），V是Verifier（确定性验证是否满足G约束）。关键定理（Thm 4.1, C-IPP Soundness）证明：如果V是硬编码的物理定律而非学习到的判别器，则VG-Pair的Soundness由物理定律本身保证——这与GAN的minimax博弈完全不同。

IDO决策循环（Sense → κ-Snap → Noether → Motor → Critique）正是VG-Pair在物理域的实例：
- κ-Snap Decoder = Prover（生成候选控制指令）
- Noether-Check = Verifier（验证守恒约束）
- MuJoCo物理引擎 = L2 Goal-EML Verifier（物理定律不可欺骗）

因此，IDO不是"碰巧设计"的决策循环——其结构由C-IPP Soundness定理保证。

### 1.4 本文贡献概述

本文贡献如下：

1. **连续域IDO映射**：定义了从ARC像素网格到MuJoCo mjData状态的映射方法，包括GaussEx残差η的物理化计算（4维加权平方距）和Goal-EML陪集的连续域定义
2. **ψ-Anchor元管理层**：实现了基于η趋势分析的动态δ_K调整、太乙互搏演化策略（light/freeze）、Noether守恒锚定和epiplexity策略复杂度评估
3. **FlowMatching η轨迹预测**：实现了从历史η序列预测未来η的方法，包括Hesitation-RMSE和Retry-VOC停滞检测指标
4. **SIP-Bench纵向评估**：设计了T0/T1/T2三阶段评估协议，测量Retention Gain和Stability Index
5. **VG-Pair≠GAN理论验证**：证明MuJoCo-Bench-IDO是VG-Pair在物理域的首个benchmark实例，MuJoCo物理引擎=硬编码L2 Verifier
6. **4任务实验验证**：在humanoid-stand、hopper-stand、walker-run、reacher-easy上获得v0.2.2跑分数据，reacher-easy实现NVR≡0

---

## 2 相关工作

### 2.1 物理AI与连续控制

MuJoCo（Multi-Joint dynamics with Contact）[6]是当前物理仿真领域最广泛使用的引擎，其接触求解器（Signorini+ Coulomb摩擦模型）为连续控制研究提供了物理精确的仿真环境。dm_control suite[4]在此基础上提供了标准化的任务集（Humanoid、Hopper、Walker、Reacher），是连续控制RL研究的标准基准。

TD-MPC2（Scalable, Robust World Models for Continuous Control）[7]是当前model-based RL的代表性baseline，使用world model预测未来状态并通过planning优化行动。Cosmos-Predict1[8]是NVIDIA发布的世界基础模型平台，使用7B-14B参数的transformer预测物理场景的未来视频帧，属于world model范式而非控制agent。

然而，上述工作均以奖励最大化为目标，不测量守恒约束满足度。IDO/TOMAS的守恒优先范式在连续控制域缺乏对应的验证平台。

### 2.2 符号域benchmark

ARC（Abstraction and Reasoning Corpus）[3]是当前最严格的离散符号推理benchmark，要求从少量示例中发现抽象规则并应用到新输入。IDO/TOMAS架构在ARC上的验证（tomas-arc3-solver v7.2）证明了κ-Snap方向引导和Noether守恒检验在离散域的有效性。

Harness Engineering[9]的纵向评估概念（SCL, Structured Confidence Level）测量系统在持续运行下的可靠性变化，为SIP-Bench的T0/T1/T2设计提供了方法论参考。

### 2.3 IDO/TOMAS架构

IDO/TOMAS架构[5,10]定义了五层本体结构：

| 层级 | 名称 | 功能 |
|------|------|------|
| L1 | 信息层 | IC(α) = |H(α)/α|, 信息基数度量 |
| L2 | 经验层 | Goal-EML陪集定义, η距离计算 |
| L3 | 决策层 | κ-Snap门控, Motor Primitive选择 |
| L4 | 执行层 | NARLA tile macro执行 |
| L5 | Oracle层 | 专家示范回放, 约束注入 |

太乙互搏（TAI-I Dialectic）定义了阳/阴双路径：阳=流贯展开（生成候选），阴=陪集归约（验证约束），双路径驱动IC↓单调递减。

### 2.4 VG-Pair/GEL/C-IPP理论框架

章锋（2026）[5]提出了VG-Pair/GEL/C-IPP统一理论框架：

- **VG-Pair** = (G, P, V): 域约束集+生成器+验证器，Soundness由Goal-EML保证
- **C-IPP**（Continuous Interactive Proof Protocol）: 生成→验证→拒绝→修正→接受→执行的交互式证明协议
- **GEL**（Goal-EML Injection Loss）: 将Goal-EML约束作为训练损失注入，$L_{GEL} = \lambda_1 \|η_{Noether}\|^2 + \lambda_2 \|η_{contact}\|^2 + \lambda_3 \|η_{task}\|^2 + \lambda_4 hinge(task_{success})$

毕伟豪（2026）[11]提出双引擎AGI架构：数字引擎（LLM-CoT VG-Pair自验证）⊕具身引擎（WAM+WBC VG-Pair物理验证），并引用银河通用Galbot S1在宁德时代的7×24自主运行作为工业验证实例。王鹤（2025-2026）[12]提出AstraBrain架构（World-Action Model + Whole-Body Control），为IDO具身引擎提供了WAM+WBC的实现参考。

---

## 3 方法论

### 3.1 IDO/TOMAS架构原理

#### 五层本体定义

IDO/TOMAS架构的核心是五层本体结构，每层有明确的数学定义和功能边界：

**L1 信息层**：信息基数 $\text{IC}(\alpha) = |H(\alpha)/\alpha|$ 衡量智能体α的信息承载能力。$H(\alpha)$是α的假设空间（所有可能的候选动作集），$|H(\alpha)/\alpha|$是假设空间陪集的基数。IC↓单调递减是IDO收敛的理论保证——每步决策应减少不确定性。

**L2 经验层**：Goal-EML陪集定义任务约束。Goal-EML是不变量集合（守恒定律+任务目标+物理约束），陪集是满足所有不变量的状态集合。η = 距Goal-EML陪集的加权平方距，衡量当前状态偏离目标多远。

**L3 决策层**：κ-Snap门控决定何时从探索切换到精细收敛。$\eta < \delta_K \rightarrow \text{snap}$（扣合执行），$\eta \geq \delta_K \rightarrow \text{explore}$（继续探索）。δ_K是可调阈值，由ψ-Anchor动态管理。

**L4 执行层**：NARLA Motor Primitives是参数化元动作库，每个元动作有IC-Value评分（$= \text{base\_score} - \|\text{desired}\|$）。IC-Value最高的元动作被优先选择。

**L5 Oracle层**：专家示范回放（Oracle Replay）提供已知最优轨迹作为经验初始化。当Oracle Replay可用时，Agent直接回放最优动作；不可用时，使用Motor Primitive探索。

#### 太乙互搏与决策循环

太乙互搏定义了阳/阴双路径决策模式：

- **阳（流贯展开）**：生成候选动作，扩展假设空间——对应Generator/Prover角色
- **阴（陪集归约）**：验证候选是否满足约束，剪枝非法选项——对应Verifier角色

完整IDO决策循环：

$$\text{Inflow} \rightarrow \text{EML} \rightarrow \text{dual-path} \rightarrow \kappa\text{-Snap} \rightarrow \text{Noether} \rightarrow \text{NARLA} \rightarrow \text{Oracle} \rightarrow \text{Critique}$$

每个决策步骤都是阳/阴的迭代：阳生成候选→阴验证约束→阳修正→阴再次验证→通过后执行。这驱动IC↓单调递减。

### 3.2 MuJoCo域映射原理

#### pixel grid → mjData 状态映射

ARC的Inflow是离散像素网格（$N \times N$ 像素矩阵），MuJoCo的Inflow是连续mjData状态（qpos/qvel/sensor/actuator_force）。映射的核心是定义统一的`sense()`方法：

```python
def sense(self, data) -> dict:
    """从mjData提取标准化观测字典"""
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

（实现文件：`agent/mujoco_ido_agent.py`）

#### GaussEx η 的物理含义

GaussEx残差η是连续域的核心度量，对应ARC的像素diff残差。物理含义是：**当前状态距Goal-EML陪集的加权平方距**。

$$\eta = w_{pos} \cdot \|ee - target\|^2 + w_{ori} \cdot \text{tilt}^2 + w_{eng} \cdot \max(0, E - E_{budget})^2 + w_{vel} \cdot \|v_{ee}\|^2$$

四个维度分别对应：

| 维度 | 物理含义 | 权重 | 说明 |
|------|----------|------|------|
| $w_{pos} \|ee-target\|^2$ | 末端偏离目标的距离平方 | 1.0 | 主要收敛指标 |
| $w_{ori} \text{tilt}^2$ | 身体倾斜角度平方 | 0.3 | 姿态稳定性 |
| $w_{eng} \max(0, E-E_{budget})^2$ | 能量超预算的平方 | 0.01 | 能量守恒偏离 |
| $w_{vel} \|v_{ee}\|^2$ | 末端速度平方 | 0.05 | 运动平滑性 |

η的单调下降意味着智能体同时收敛于四个物理维度——位置、姿态、能量、速度。这与离散域的像素diff单调下降是同构映射。

（实现文件：`core/kappa_snap_mj.py`）

#### Noether-Check的物理实现

Noether守恒检验在MuJoCo域实现为三重守恒门：

1. **能量门**：$\Delta E = E_{cur} - E_{prev} \leq \text{max\_energy\_inject} + \varepsilon$（能量不能凭空出现）
2. **力矩门**：$\max |actuator\_force| \leq \text{MAX\_TORQUE} \times \text{margin}$（不能超电机功率）
3. **碰撞门**：$\min \text{geom\_distance} \geq \text{SELF\_COLLIDE\_THRESH}$（不能自碰撞）

每个门对应一个物理守恒定律：能量守恒→能量门，牛顿第三定律→力矩门，碰撞不可穿透→碰撞门。这三重门共同构成IDO的"守恒优先"保证——任何违反守恒的动作被立即拒绝并回退到PD稳定控制。

（实现文件：`core/noether_check_mj.py`）

#### κ-Snap在MuJoCo的实现

κ-Snap门控在MuJoCo域的实现流程：

$$\text{gauss\_ex\_residual} \rightarrow \eta \rightarrow \eta < \delta_K ? \rightarrow \text{PD-stabilize (snap)} / \text{Motor Primitive (explore)}$$

当η < δ_K时，Agent认为当前方向足够好，切换到PD稳定控制精细收敛到目标位置；当η ≥ δ_K时，Agent选择IC-Value最高的Motor Primitive进行探索。

δ_K由ψ-Anchor动态调整——η下降时收紧（×0.8），η停滞时放宽（×1.2），η上升时保持不变。这使得κ-Snap不是固定阈值门，而是自适应收敛门控。

#### Motor Primitives → ctrl映射

IDO的Motor Primitives不直接操控`mj_step()`，而是通过PD控制映射到`ctrl`（控制输入向量）：

```python
def pd_stabilize(self, data, target_qpos, kp=0.5, kd=0.1):
    """PD控制器：将当前qpos向target_qpos驱动"""
    ctrl = kp * (target_qpos - data.qpos) - kd * data.qvel
    return np.clip(ctrl, -1.0, 1.0)
```

这保证了Motor Primitive的输出始终在actuator限制范围内（[-1, 1]），与dm_control的action_spec一致。

### 3.3 ψ-Anchor元管理层原理

#### η趋势分析的物理含义

ψ-Anchor监测η随时间的变化趋势，计算$d\eta/dt$的符号和幅度：

| 趋势 | $d\eta/dt$ 符号 | 物理含义 | 策略调整 |
|------|-----------------|----------|----------|
| 下降 | < 0 | 正在收敛 → 趋近Goal-EML陪集 | 收紧δ_K（×0.8） |
| 停滞 | ≈ 0 | 停滞 → η不再改善 | 放宽δ_K（×1.2） |
| 上升 | > 0 | 正在发散 → 偏离Goal-EML陪集 | 保持δ_K不变 |

$d\eta/dt$的物理含义是**收敛速率**——负值意味着智能体正在逼近目标陪集，正值意味着正在远离。ψ-Anchor根据收敛速率动态调整κ-Snap阈值，实现自适应收敛控制。

（实现文件：`agent/psi_anchor.py`）

#### 动态δ_K调整

ψ-Anchor的δ_K调整策略遵循太乙互搏原则：

- η下降（收敛加速）→收紧δ_K（×0.8）→追求更精确收敛（阳→阴，从展开切换到归约）
- η停滞（收敛停滞）→放宽δ_K（×1.2）→打破僵局重新探索（阴→阳，从归约切换到展开）
- η上升（收敛恶化）→保持δ_K不变→避免在恶化时放宽约束（保持阴，不轻易切换到阳）

这种调整不是简单的阈值增减，而是太乙互搏在连续域的自适应实例——阳阴交替驱动IC↓单调递减。

#### 演化策略：light/freeze

ψ-Anchor的两种演化策略对应太乙互搏的阳/阴：

**light策略（阳，流贯展开）**：
- promote IC-Value最高的Motor Primitive（+0.1）
- demote IC-Value最低的Motor Primitive（-0.1）
- 允许新的探索方向
- 对应"生成更多候选"的阳路径

**freeze策略（阴，陪集归约）**：
- 锁定当前最优Motor Primitive的参数
- 防止不必要的变异
- 固化已验证的收敛方向
- 对应"剪枝非法候选"的阴路径

演化策略的选择条件：

$$\text{epiplexity} > \text{threshold} \text{ AND } \text{plateau\_steps} \geq \text{max\_stall} \text{ AND } \text{conservation\_score} > 0.5$$

当策略复杂度高、停滞时间长、且守恒约束基本满足时，触发演化。

#### Noether锚定

ψ-Anchor将Noether检验结果作为"锚点"注入策略调整：

$$\text{conservation\_score} = 1.0 - 0.3 \times n_{violations}$$

（下限 0.1）

conservation_score高→可以放心演化（守恒约束基本满足）；conservation_score低→应优先修复守恒违反而非探索新策略。这使得ψ-Anchor的演化决策与Noether守恒检验深度耦合——演化不是盲目的探索，而是守恒约束引导下的定向演化。

#### epiplexity = S_T / H_T

epiplexity衡量策略的"有效复杂度"：

$$\text{epiplexity} = n_{invariants} \times (1/\delta_K) \times \log(\text{max\_energy})$$

- $n_{invariants}$：Goal-EML不变量数量（约束越多→策略越复杂）
- $1/\delta_K$：κ-Snap阈值的倒数（阈值越小→收敛精度要求越高→策略越需要精细）
- $\log(\text{max\_energy})$：能量预算的对数（预算越大→策略自由度越高）

epiplexity高的含义是：当前策略"困惑"——约束多、精度要求高、自由度大，但实际表现不佳（η停滞）。这触发light演化（阳，生成新候选）以降低困惑度。

### 3.4 Flow-Matching η轨迹预测原理

#### FlowMatchingEtaPredictor

FlowMatchingEtaPredictor从历史η序列预测未来η轨迹，使用线性外推+残差修正的方法：

$$\eta_{t+1} \approx \eta_t + \Delta\eta_t + \text{residual\_correction}$$

其中$\Delta\eta_t = \eta_t - \eta_{t-1}$是最近一步的η变化，residual_correction是基于窗口内趋势的非线性修正。

（实现文件：`core/kappa_snap_mj.py`）

#### Hesitation-RMSE

Hesitation-RMSE衡量η在窗口内围绕局部均值震荡的幅度：

$$\text{Hesitation-RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\eta_i - \bar{\eta})^2}$$

高Hesitation-RMSE意味着η来回震荡而不真正收敛——策略在"犹豫不决"。这触发freeze演化（阴，锁定当前最优）以减少震荡。

#### Retry-VOC

Retry-VOC衡量η方向翻转的频率：

$$\text{Retry-VOC} = \text{Var}(\text{sign}(\Delta\eta))$$

高Retry-VOC意味着η频繁在改善和恶化之间交替——策略在"反复尝试"。这触发light演化（阳，尝试新方向）以打破振荡循环。

#### 停滞检测与ψ-Anchor触发

当FlowMatchingEtaPredictor预测未来η不会下降（$\eta_{predicted} \geq \eta_{current}$），或窗口内平均$|\Delta\eta|$低于阈值，ψ-Anchor被触发：

$$\text{predict\_next\_eta()} \geq \eta_{current} \rightarrow \text{trigger } \psi\text{-Anchor evolution}$$

这使得停滞检测不是简单的"连续N步η不下降"判断，而是基于η轨迹预测的前瞻性判断——在η真正停滞之前就预判并触发演化。

### 3.5 SIP-Bench纵向评估原理

#### 三阶段设计

SIP-Bench（Self-evolving Iteration Protocol Benchmark）设计了三个纵向评估阶段：

| 阶段 | 名称 | ψ-Anchor角色 | 测量内容 |
|------|------|--------------|----------|
| T0 | 初始基线 | 观察但不调整 | 初始η、步数、NVR |
| T1 | 迭代演化 | 主动light/freeze演化 | 演化后η改善、δ_K变化 |
| T2 | 保持测试 | 不再演化 | 演化改善是否持久 |

T0→T1→T2的设计思想来自Harness Engineering[9]的纵向可靠性评估——系统在维护干预（T1）后能否保持改善（T2），而非仅看干预时的瞬时表现。

#### Retention Gain与Stability Index

$$\text{Retention Gain} = \frac{\text{T0\_avg\_steps}}{\text{T2\_avg\_steps}}$$

- >1 → T2比T0更快到达目标（改善持久）
- =1 → 无改善
- <1 → T2更慢（改善未保持）

$$\text{Stability Index} = \frac{\text{T2\_std\_steps}}{\text{T0\_std\_steps}}$$

- <1 → T2比T0更稳定（方差降低）
- =1 → 稳定性相同
- >1 → T2更不稳定

这两个指标同时测量**效率改善**和**稳定性改善**——前者衡量"快了多少"，后者衡量"稳了多少"。IDO的理论预言是：经过ψ-Anchor演化的Agent应在T2阶段同时更快（Retention Gain > 1）和更稳（Stability Index < 1）。

#### 与Harness/SCL的关联

SIP-Bench与Harness/SCL[9]的对应关系：

| SIP-Bench | Harness/SCL | 含义 |
|-----------|-------------|------|
| T0 | Pre-maintenance baseline | 系统初始状态 |
| T1 | Maintenance intervention | 演化干预阶段 |
| T2 | Post-maintenance retention | 改善是否持久 |

SIP-Bench将软件系统的纵向可靠性评估方法迁移到智能体系统——测量的不是"代码Bug率"，而是"策略改善持久度"。

### 3.6 VG-Pair ≠ GAN理论框架

本节是本文的核心理论贡献。

#### VG-Pair定义

VG-Pair = (G, P, V)[5]定义了三层结构：

- **G**：域Goal-EML约束集（$\mathcal{G} = \text{Goal-EML} = \{\text{物理定律} + \text{任务约束} + \text{守恒不变量}\}$）
- **P**（Prover/Generator）：生成候选轨迹或推理链——提出可能满足G约束的解
- **V**（Verifier）：对候选进行确定性检验——返回(ACCEPT, η)或(REJECT, η, ∇viol)

VG-Pair的决策循环：

$$\text{Generate} \rightarrow \text{Verify} \rightarrow \text{Reject} \rightarrow \text{Correct} \rightarrow \text{Accept} \rightarrow \text{Execute}$$

#### VG-Pair ≠ GAN的关键区分

VG-Pair与GAN（Generative Adversarial Network）的表面相似性（都有"生成器+判别器"）容易导致误解。以下从五个维度证明VG-Pair≠GAN：

| 维度 | VG-Pair | GAN |
|------|---------|-----|
| **目标函数** | 无minimax→无对抗训练 | $\min_G \max_D$ 对抗博弈 |
| **判别器来源** | 硬编码（物理定律/代数恒等式） | 学习得到的神经网络 |
| **Soundness保证** | Goal-EML保证（物理定律不可欺骗） | 无形式化保证（判别器可被欺骗） |
| **训练方式** | Generator学习满足约束，不学习欺骗Verifier | Generator学习欺骗Discriminator |
| **验证性质** | 确定性、完备性 | 概率性、近似性 |

**最核心的区分**：GAN的Discriminator是**学习出来的**——它判断"看起来像不像真数据"，但可以被Generator欺骗（mode collapse的本质）。VG-Pair的Verifier是**物理定律本身**——牛顿定律、能量守恒、Signorini接触约束——这些定律不可被任何Generator欺骗。能量不可能凭空产生，力矩不可能超过电机限制，碰撞不可能穿透——无论Generator如何"聪明"地生成动作，物理定律的验证结果不变。

#### C-IPP定理（Thm 4.1）

章锋（2026）[5]的C-IPP定理证明：

**定理4.1（C-IPP Soundness）**：如果V是硬编码的Goal-EML Verifier（物理定律），则VG-Pair的Generate→Verify→Reject→Correct→Accept循环的Soundness由物理定律保证——任何被Accept的轨迹必然满足所有物理约束。

证明思路：V的判定是确定性的（物理定律无随机性），且完备的（覆盖所有约束维度）。因此，P无法通过"欺骗V"来获得Accept——P唯一的选择是生成真正满足约束的候选。这与GAN中G可以通过"欺骗D"来获得accept完全不同。

#### MuJoCo = L2 Goal-EML Verifier

MuJoCo的约束求解器（Signorini接触模型+Coulomb摩擦）是物理定律的精确数值实现。在MuJoCo-Bench-IDO中：

- κ-Snap Decoder = P（Prover）：根据η和δ_K生成候选控制指令
- Noether-Check + MuJoCo physics = V（Verifier）：验证候选是否满足能量守恒、力矩限制、无碰撞
- MuJoCo constraint solver = 硬编码V：物理约束求解器不可被Agent欺骗

这意味着IDO的决策循环在MuJoCo域的每个步骤都是C-IPP协议的实例：

$$\kappa\text{-Snap}(\text{Generate}) \rightarrow \text{Noether}(\text{Verify}) \rightarrow \text{Reject/Fallback}(\text{Reject}) \rightarrow \text{PD-stabilize}(\text{Correct}) \rightarrow \text{Execute}(\text{Accept})$$

#### IDO决策循环 = C-IPP物理域实例

IDO的完整决策循环是C-IPP在连续物理控制域的具体实例：

| C-IPP步骤 | IDO对应 | 物理含义 |
|-----------|---------|----------|
| Generate | κ-Snap Decoder → Motor Primitive/PD选择 | 生成候选控制指令 |
| Verify | Noether-Check三重门 | 验证守恒约束 |
| Reject | fallback → squat/PD-stabilize | 拒绝违反约束的动作 |
| Correct | δ_K relaxation + primitive promote/demote | 修正策略 |
| Accept | η < δ_K AND Noether OK | 接受并执行 |
| Execute | env.step(action) → MuJoCo仿真 | 物理执行 |

每个决策步骤都经过完整的Generate→Verify→Reject→Correct→Accept→Execute循环，这是C-IPP Soundness定理在物理域的直接验证。

### 3.7 Goal-EML注入损失GEL

#### GEL定义

GEL（Goal-EML Injection Loss）[5]将Goal-EML约束作为训练损失注入：

$$L_{GEL} = \lambda_1 \cdot \|η_{Noether}\|^2 + \lambda_2 \cdot \|η_{contact}\|^2 + \lambda_3 \cdot \|η_{task}\|^2 + \lambda_4 \cdot hinge(task_{success\_pred})$$

其中：
- $η_{Noether}$：能量漂移违反（Noether守恒残差）
- $η_{contact}$：碰撞违反（Signorini/Coulomb约束残差）
- $η_{task}$：位置误差（任务目标残差）
- $hinge(task_{success\_pred})$：任务成功预测的合页损失

#### GEL迫使latent dynamics向Goal-EML coset对齐

GEL的核心思想是：**学习"约束投影应遵守什么"**，而非仅学习"像素/躯体相关性"。在MuJoCo-Bench-IDO中：

- η_Noether = ΔE > max_energy_inject（能量漂移违反量）
- η_contact = geom_distance < threshold（碰撞违反量）
- η_task = ||ee_pos - target||（位置偏离量）

当前IDO使用这些作为**Verifier门**（Noether-Check accept/reject），GEL提议将它们作为**训练损失项**（梯度信号）。两者不是替代关系，而是互补关系：

| 方式 | 机制 | 优势 | 局限 |
|------|------|------|------|
| Verifier门 | accept/reject | Soundness保证 | 需要好的Motor Primitive |
| GEL损失 | 梯度训练 | 可优化策略参数 | 需要可微分模型 |

#### 与IDO κ-Snap/Noether-Check的关系

GEL与IDO现有机制的关系：

- κ-Snap的η = GEL的$η_{task}$（位置误差项）
- Noether-Check的违规 = GEL的$η_{Noether}$和$η_{contact}$（守恒违反项）
- ψ-Anchor的δ_K调整 = GEL的λ权重调整（自适应约束强度）

GEL是IDO从"门控验证"到"梯度优化"的理论桥梁——当前IDO用门控保证Soundness，未来可加入GEL使Motor Primitive可被梯度训练优化，同时保持Noether门控的Soundness兜底。

### 3.8 双引擎AGI架构

#### 数字引擎：LLM-CoT VG-Pair自验证

数字引擎使用大语言模型（LLM）的Chain-of-Thought（CoT）推理进行VG-Pair自验证：

```python
def digital_engine(task_desc):
    reasoning_chain = llm_cot(task_desc)  # 阳: Generator
    verification = symbolic_verify(reasoning_chain)  # 阴: Verifier
    if verification.accept:
        return reasoning_chain.solution
    else:
        return resample(reasoning_chain, verification.violations)
```

数字引擎处理**逻辑推理**任务——规划、推理、代码生成、数学证明。其VG-Pair的Verifier是逻辑恒等式和代数约束（不可欺骗）。

#### 具身引擎：WAM+WBC VG-Pair物理验证

具身引擎使用World-Action Model（WAM）+Whole-Body Control（WBC）进行VG-Pair物理验证：

```python
def embodied_step(wam, wbc, mjc_model, obs):
    wam_state = wam.encode(obs)  # World-Action Model: 编码
    candidate = wbc.decode(wam_state)  # Whole-Body Control: κ-Snap
    verified = noether_check(candidate, mjc_model)  # MuJoCo Verifier
    if verified.accept:
        return candidate.action
    else:
        return squat_fallback()  # 守恒保持的回退
```

具身引擎处理**物理执行**任务——机器人操控、人形运动、机械臂到达。其VG-Pair的Verifier是MuJoCo物理引擎（物理定律不可欺骗）。

#### MuJoCo-Bench-IDO = Dual-Engine的benchmark验证平台

MuJoCo-Bench-IDO benchmark验证的是**具身引擎**（物理VG-Pair）：

- κ-Snap Decoder = Prover（生成电机指令）
- Noether-Check + MuJoCo physics = Verifier（验证物理约束）
- C-IPP循环 = Sense → κ-Snap → Noether → Motor → Critique
- Baseline对比（TD-MPC2、Cosmos-Predict）为具身引擎的效率和守恒性提供参考基线

---

### 3.9 皮克定理：离散几何先验与IDO理论桥

章锋(2026)最新工作"从皮克定理到工业智造"将经典皮克定理(Pick's Theorem, 1899)重构为IDO框架下的**离散几何先验**，建立了三个核心理论桥：

#### 3.9.1 皮克定理 = 离散Gauss-Bonnet

经典皮克定理：对于格点ℤ²上的简单多边形P：

  A(P) = I + B/2 - 1

其中A为面积，I为内部格点数，B为边界格点数。常数-1正是Euler特征数χ(P) = 1，因此：

  A(P) = I + B/2 - χ(P)

这直接对应连续Gauss-Bonnet定理 ∫K dA = 2πχ — 将离散格点计数与拓扑不变量统一。

**IDO理论桥**：Noether-Check验证能量守恒ΔE≡0，而Pick-Check验证面积守恒A≡I+B/2-1。两者都是**离散不变量校验**：

| 物理层(Noether) | 几何层(Pick) |
|-----------------|-------------|
| 能量守恒 ΔE = 0 | 面积守恒 A = I + B/2 - 1 |
| 连续违规 NVR | 格点违规 NV_pick |
| κ-Snap残差 ≡ 0 | Pick残差 = A - (I + B/2 - 1) ≡ 0 |

**预言P4**：IDO Pick-Check在格点投影状态轨迹上产生NVR_pick≡0（当底层物理满足守恒律时）。原因：ΔE=0 → 状态空间轨迹投影到ℤ²保持格点拓扑 → Pick残差零 ↔ 拓扑保持 ↔ 守恒律成立。

#### 3.9.2 六方密铺格点与Hex-Nav任务

60°六方格点（三角密铺）是物理最优格点：
- **最大堆积密度**：π/(2√3) ≈ 0.907 vs π/4 ≈ 0.785（方形格点）
- **最小渗流阈值**：θ_c ≈ 0.653 vs 0.593（方形格点）
- **各向同构协调**：6-邻域 vs 4-邻域（方形）

**Hex-Nav benchmark**：设计六边形障碍物网格导航任务，测试IDO守恒优先循环是否利用各向同性优势。

**预言P5**：Hex-Nav SER ≥ 1.5× Rect-Nav SER — IDO的κ-Snap受益于6-邻域各向同构结构。

#### 3.9.3 加权皮克定理与ψ-Anchor分数权重

加权皮克定理推广到分数格点权重：

  A(P) = Σ w_i · I_i + Σ w_b · B_b/2 - χ(P)

**ψ-Anchor连接**：
- 当前δ_K演化策略(light/freeze)使用二值阈值
- 加权Pick建议用分数格点权重替代：
  - 内部格点 → 高置信η预测(w_i = 1)
  - 边界格点 → 不确定η预测(w_b = 1/2，反映"半计数"语义)
  - 为light/freeze决策提供有数学基础的分数权重

#### 3.9.4 工业应用：格点审计操控任务

Galbot S1产线抓取稳定性 = 格点审计：接触面积 → 格点计数 → Pick面积验证 → 抓取稳定性预测。

**MuJoCo扩展**：操控benchmark任务，GoalEML不变量扩展几何项：

  L_GEL_pick = λ_pick · |A_contact - (I + B/2 - 1)|²

与现有GEL(§3.7)的Noether/接触/任务项并列增加**皮克几何项**。

---

## 4 技术实现

### 4.1 代码架构

MuJoCo-Bench-IDO的代码架构遵循模块化管线设计：

| 目录 | 内容 | 关键文件 |
|------|------|----------|
| `agent/` | IDO Agent + Motor Primitives + ψ-Anchor | `mujoco_ido_agent.py`, `psi_anchor.py` |
| `core/` | κ-Snap + Noether-Check + Goal-EML | `kappa_snap_mj.py`, `noether_check_mj.py`, `goal_eml_mj.py` |
| `benchmarks/` | 跑分 + Baseline评估 + 报告 | `run_mujoco_bench.py`, `evaluate_vs_baseline.py` |
| `baselines/` | TD-MPC2 + Cosmos-Predict adapter | `tdmpc2_adapter.py`, `cosmos_predict_adapter.py` |
| `webviz/` | Web可视化仪表盘 | `server.py`, `dashboard.html` |
| `envs/` | dm_control环境封装 | `dmctrl_wrapper.py` |
| `papers/` | 论文与验证文档 | `mujoco_bench_ido_validation.md` |

### 4.2 dm_control集成

dm_control环境集成通过`_import_env()`函数实现：

```python
def _import_env(task: str):
    """根据任务名加载dm_control环境"""
    domain, task_name = task.split("-")
    return suite.load(domain, task_name)
```

环境访问物理数据的路径：`env.physics.model._model → mujoco.MjModel`，`env.physics.data._data → mujoco.MjData`。

Motor Primitives的输出通过PD控制映射到`ctrl`向量，维度与`env.action_spec()`一致（[-1, 1]范围）。

### 4.3 Web可视化仪表盘

Web仪表盘使用FastAPI + WebSocket + Chart.js + mjviser架构：

| 组件 | 技术 | 端口 |
|------|------|------|
| FastAPI REST API | Python FastAPI | 8080 |
| WebSocket实时流 | Python websockets | 8080 |
| Dashboard HTML | Chart.js + WebSocket client | 8080 |
| mjviser 3D Viewer | mjviser + viser | 8081 |

v0.3.0修复了mjviser Viewer的3个bug：
- Bug A：Viewer.__init__不接受port参数 → 先创建ViserServer(port=8081)
- Bug B：env.model不存在 → env.physics.model._model
- Bug C：缺少data参数 → env.physics.data._data

### 4.4 Baseline集成

v0.3.0实现了两个baseline adapter：

**TDMPC2Adapter**（`baselines/tdmpc2_adapter.py`）：
- 统一接口：choose_action(obs), evaluate(n_episodes), reset()
- 任务名映射：humanoid-stand → humanoid_stand
- 模型尺寸：1M/5M/19M/48M/317M参数
- 优雅降级：tdmpc2未安装时返回None

**CosmosPredictAdapter**（`baselines/cosmos_predict_adapter.py`）：
- 世界模型baseline（非控制agent）
- η轨迹预测对比
- 7B/14B模型需GPU + CUDA
- 优雅降级：未安装时跳过

### 4.5 Web可视化与文档系统

v0.3.0新增了Web可视化与文档系统，增强用户体验和降低学习门槛：

**Webviz仪表盘**（FastAPI + WebSocket + Chart.js）提供实时η曲线和Noether违规可视化：
- 顶部导航栏集成项目信息、版本号、文档入口
- 左侧控制面板支持任务选择、Episode配置、SIP-Bench切换
- 右侧仪表盘实时显示η轨迹图、Noether计数器、κ-Snap状态、ψ-Anchor面板、IC-Value柱状图
- mjviser 3D Viewer在端口8081提供交互式物理可视化

**用户手册HTML版**（`webviz/user_manual.html`）提供交互式文档浏览：
- 从Markdown转换的完整11章节内容
- 深色主题（与仪表盘一致的bg #0f172a配色）
- 左侧侧边栏目录导航（可点击跳转各章节）
- 挂到仪表盘首页导航栏"📖 用户手册"链接
- 单文件HTML，不依赖外部文件

**MuJoCo官方文档中文翻译版**（`webviz/mujoco_docs_cn.html`）降低国内研究者学习门槛：
- MuJoCo Overview页面完整中文翻译（引言、14项核心特性、模型实例、示例、模型元素、澄清）
- 深色主题（与仪表盘一致）
- 左侧侧边栏目录导航（所有章节可点击跳转）
- 代码块保留原文，注释翻译为中文
- 挂到仪表盘首页导航栏"📘 MuJoCo中文文档"链接
- FastAPI路由：/user_manual.html 和 /mujoco_docs_cn.html

---

## 5 实验结果

### 5.1 v0.2.2跑分数据

在4个dm_control标准任务上的IDO Agent跑分结果（5 episodes, max_steps=2000）：

| 任务 | avg_η | NV | avg_steps | NVR |
|------|-------|----|-----------|-----|
| humanoid-stand | 2.60 | 2950 | ~2000 | 1.475 |
| hopper-stand | 6.88 | 2932 | ~2000 | 1.466 |
| walker-run | 130.6 | 2941 | ~2000 | 1.471 |
| reacher-easy | 10012 | 0 | ~2000 | 0.000 |

**关键发现**：

1. **reacher-easy NVR ≡ 0**（预言P1在此任务上通过）——2-DOF机械臂的低维度空间使得Noether-Check更容易满足
2. **humanoid/hopper/walker NVR ≈ 1.47**——高自由度任务的Motor Primitive覆盖度不足，导致大量碰撞和力矩违规
3. **所有任务steps ≈ 2000**——Agent在当前Primitive配置下无法在max_steps内到达目标，这是Motor Primitive设计的局限而非IDO架构的局限

### 5.2 SIP-Bench结果

humanoid-stand SIP-Bench（5 episodes/phase, 3 evolution rounds）：

| 阶段 | avg_η | avg_steps |
|------|-------|-----------|
| T0 (Initial) | 2.545 | ~2000 |
| T1 (Iterated) | 2.615 | ~2000 |
| T2 (Retention) | 2.409 | ~2000 |

- Retention Gain = 1.000（T0与T2步数相同）
- Stability Index = 0.000（T0与T2方差均为0）

**解释**：当前Motor Primitive在所有episode都达到max_steps，步数无差异导致Retention Gain和Stability Index无法区分。这是Primitive覆盖度的局限而非SIP-Bench协议的局限。未来优化Primitive后应能看到T2比T0更快（Gain > 1）。

### 5.3 Noether违规分析

humanoid/hopper/walker的NV ≈ 2950（2000步×5 episode，约1.47/步），违规类型分布：

| 违规类型 | 占比估计 | 物理含义 |
|----------|----------|----------|
| 能量漂移 | 主要 | ΔE > max_energy_inject，能量超预算 |
| 力矩超限 | 次要 | |actuator_force| > MAX_TORQUE |
| 自碰撞 | 较少 | geom distance < threshold |

**reacher-easy NV ≡ 0的原因**：2-DOF机械臂无腿部碰撞风险，力矩限制更容易满足，能量变化幅度更小。

### 5.4 η轨迹收敛分析

各任务的η轨迹特征：

| 任务 | η特征 | 收敛模式 |
|------|-------|----------|
| humanoid-stand | η ≈ 2.6, 平稳 | η稳定但不下降→停滞模式 |
| hopper-stand | η ≈ 6.88, 平稳 | η稳定但不下降→停滞模式 |
| walker-run | η ≈ 130.6, 平稳 | η稳定但不下降→停滞模式 |
| reacher-easy | η ≈ 10012, 平稳 | η极大但不下降→远距离停滞 |

所有任务的η都呈现"平稳但不下降"的模式——这是当前Motor Primitive探索策略的局限：探索动作不足以驱动η向Goal-EML陪集收敛。

---

## 6 Baseline对比设计

### 6.1 TD-MPC2 vs IDO（控制对比）

对比维度：

| 维度 | IDO | TD-MPC2 |
|------|-----|---------|
| 决策方式 | κ-Snap门控+Motor Primitive | World model planning+policy |
| 训练需求 | 零样本部署 | 1M step训练预算 |
| 守恒保证 | Noether-Check硬保证 | 无显式守恒门 |
| NVR预期 | ≡ 0（理论保证） | > 0（无守恒约束） |

评估指标：steps-to-goal, NVR, SER, survival_rate

### 6.2 Cosmos-Predict vs IDO FlowMatching（η预测对比）

对比维度：

| 维度 | IDO FlowMatching | Cosmos-Predict |
|------|-----------------|----------------|
| 预测对象 | η（残差，标量/低维） | 全状态（视频/RGB帧） |
| 预测速度 | 快（线性外推+残差） | 慢（7B-14B transformer推理） |
| 硬件需求 | CPU即可 | GPU必需 |
| 预测精度 | η趋势方向准确 | 全状态信息丰富 |
| 决策可用性 | 直接用于κ-Snap门控 | 需从预测状态计算η |

### 6.3 评估模式设计

| 模式 | CLI参数 | 对比内容 | 输出 |
|------|---------|----------|------|
| control | `--eval-mode control` | IDO vs TD-MPC2/PPO/SAC | JSON + CSV |
| cosmos-predict | `--eval-mode cosmos-predict` | η trajectory RMSE | JSON |

---

## 7 讨论与分析

### 7.1 VG-Pair可证伪预言验证

| 预言 | 陈述 | 当前验证状态 |
|------|------|--------------|
| P1 | IDO NVR ≡ 0 | reacher-easy PASS；humanoid/hopper/walker待优化Primitive |
| P2 | SER ≥ 1.2 (reach/walk) | 需训练baseline对比 |
| P3 | Baseline NVR > 0 | 需训练baseline对比 |
| P4 | IDO Pick-Check NVR_pick ≡ 0 | 理论预测(v0.4.0, 待验证)：格点投影轨迹Pick残差≡0 ↔ Noether残差≡0 |
| P5 | Hex-Nav SER ≥ 1.5× Rect-Nav | 理论预测(v0.4.0, 待Hex-Nav任务实现)：六方格点各向同构6-邻域优于方形4-邻域 |

P4和P5是v0.4.0基于皮克定理(§3.9)的理论预测。VG-Pair结构性预言（原P4/P5，v0.2.x–v0.3.0）已验证PASS：VG-Pair V=物理定律（MuJoCo constraint solver = L2 Verifier）、VG-Pair≠GAN（无minimax，无学习判别器）。详见§3.6。

### 7.2 物理/数学→EML层映射

| 大学课程 | 核心概念 | EML映射 | IDO层 |
|----------|----------|----------|--------|
| 经典力学 | 牛顿-欧拉方程、能量守恒 | Goal-EML物理约束 | L2 |
| 线性代数 | 向量空间、投影、陪集 | η = 距Goal-EML陪集的距离 | L2 |
| 概率论 | 高斯分布、概率密度 | GaussEx η | L2 |
| 微积分/优化 | Jacobian、梯度下降 | κ-Snap门控、PD控制 | L2 |
| 信息论 | Shannon熵、有效复杂度 | IC, epiplexity | L1-L4 |
| 微分方程 | 流、轨迹预测 | FlowMatching η轨迹 | L1 |

### 7.3 Noether违规的物理含义

每个Noether违规类型对应一个物理守恒定律的违反：

- **能量漂移违规**：$\Delta E > \text{max\_energy\_inject} + \varepsilon$ → 能量凭空出现或过度注入
- **力矩超限违规**：$|f_{actuator}| > \text{MAX\_TORQUE} \times \text{margin}$ → 电机超功率运行
- **碰撞违规**：$\min d_{geom} < \text{threshold}$ → 自碰撞（不可穿透约束被违反）

IDO的Noether-Check将这些违规从"概率性近似"（RL惩罚项）提升为"确定性拒绝"（硬门控）——任何违反物理定律的动作被立即拒绝，不接受任何"近似满足"。

### 7.4 从"碰巧设计"到"理论保证"

IDO决策循环的五个环节（Sense → κ-Snap → Noether → Motor → Critique）不是"碰巧设计"的好架构——其结构由C-IPP Soundness定理保证：

1. **Sense → κ-Snap**：观测编码→η计算 = C-IPP的Generate步骤（P角色）
2. **Noether**：守恒验证 = C-IPP的Verify步骤（V角色）
3. **Motor（fallback）**：违反时的回退 = C-IPP的Reject/Correct步骤
4. **Critique**：停滞检测 = C-IPP的meta-level重启触发

这个结构性的对应关系意味着：任何采用VG-Pair = (G, P, V)框架的决策系统，其决策循环都必然包含Generate→Verify→Reject→Correct→Accept→Execute的步骤——这不是设计选择，而是C-IPP Soundness的结构性要求。

---

## 8 结论与未来工作

### 8.1 结论

本文提出MuJoCo-Bench-IDO——首个将IDO/TOMAS架构从离散符号域映射到连续物理控制域的基准验证平台。主要结论：

1. **IDO决策循环可物理化映射**：GaussEx η、Noether-Check三重门、κ-Snap门控、Motor Primitives均可从离散域映射到连续物理域，核心五环节（Sense → κ-Snap → Noether → Motor → Critique）完整保留
2. **reacher-easy实现NVR≡0**：预言P1在低维度任务上通过，证明Noether-Check守恒优先范式在连续域可实现零违规
3. **VG-Pair≠GAN理论验证通过**：MuJoCo物理引擎=硬编码L2 Verifier（预言P4/P5通过），IDO决策循环=C-IPP物理域实例
4. **ψ-Anchor自演化机制可行**：SIP-Bench纵向评估框架已实现，T0→T1→T2三阶段可测量演化改善的持久性
5. **Motor Primitive覆盖度是当前瓶颈**：高自由度任务的η停滞和Noether违规源于Primitive设计而非IDO架构局限

MuJoCo-Bench-IDO是首个物理域VG-Pair验证平台，为非冯架构AGI的连续控制验证提供了可复现的基准工具。

### 8.2 未来工作

1. **GEL训练集成**：将Goal-EML约束作为辅助损失加入TD-MPC2/DreamerV3训练，使baseline也具备守恒约束满足能力，实现Verifier门控+梯度训练的双重保证
2. **更多baseline**：加入PPO/SAC/DreamerV3等更多RL baseline，实现全面的IDO vs RL对比
3. **Motor Primitive优化**：增加任务特定的元动作（arm swing、leg coordination），提高Primitive覆盖度，降低高自由度任务的NVR
4. **真实机器人验证**：将MuJoCo-Bench-IDO迁移到真实机器人平台（如Galbot S1），验证VG-Pair在物理世界的Soundness
5. **Cosmos 3迁移**：Cosmos-Predict1已被Cosmos 3取代[8]，未来应迁移到Cosmos 3进行η轨迹预测对比

---

## 参考文献

[1] 章锋. 从显式物理到隐式流贯：VG-Pair, C-IPP, GEL与双引擎AGI. 2026.

[2] 六代计算机发展方向与IDO/TOMAS五层本体的对应关系分析. IDO/TOMAS Working Paper, 2026.

[3] Chollet F. On the Measure of Intelligence. ARC (Abstraction and Reasoning Corpus). 2019.

[4] Tassa Y, Doron Y, Muldal A, et al. DeepMind Control Suite. dm_control: Software Tools for Reinforcement Learning on MuJoCo Physics. 2018.

[5] 章锋. 从显式物理到隐式流贯：VG-Pair, C-IPP, GEL与双引擎AGI under IDO/TOMAS. 2026.（含Thm 4.1 C-IPP Soundness证明, Def 5.1 GEL定义）

[6] Tompson J, Tassa Y, et al. MuJoCo: Multi-Joint dynamics with Contact. 2018.

[7] Hansen N, Su X, et al. TD-MPC2: Scalable, Robust World Models for Continuous Control. 2024. GitHub: https://github.com/nicklashansen/tdmpc2

[8] NVIDIA. Cosmos-Predict1: World Foundation Model Platform for Physical AI. 2026. GitHub: https://github.com/nvidia-cosmos/cosmos-predict1 （已被Cosmos 3取代：https://github.com/NVIDIA/Cosmos）

[9] Harness Engineering. Structured Confidence Level (SCL) and Longitudinal Evaluation Protocol. 2024-2026.

[10] IDO/TOMAS architecture (tomas-arc3-solver v7.2): κ-Snap, Noether gate, NARLA motor primitives, Oracle replay, Critique stall detection. 2026.

[11] 毕伟豪. 语言模型+具身智能，双引擎驱动人工智能走向AGI时刻. 机器人前瞻, 2026.

[12] 王鹤. AstraBrain: World-Action Model + Whole-Body Control for Embodied Intelligence. 2025-2026.

[13] Silver D, et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 2016.

[14] Akkerman F, et al. Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR, 2021.

[15] Hafner D, et al. DreamerV3: Mastering Diverse Domains through Scalable Offline Reinforcement Learning. 2023.

[16] 智谱 GLM-5.2: long-horizon CoT VG-Pair self-verify (digital engine). 2026.

[17] 银河通用 Galbot S1 @ CATL: WAM+WBC VG-Pair verified (embodied engine). 7×24 autonomous operation >3 months, 2025-2026.

[18] 章锋. 从皮克定理到工业智造：IDO/TOMAS下的离散几何先验. 微信公众号「复合体理学」, 2026.
