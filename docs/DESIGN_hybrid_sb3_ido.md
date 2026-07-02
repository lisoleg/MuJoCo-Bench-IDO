# IDO+SB3 Hybrid Agent Architecture Design — Phase 3

## 核心洞察

**IDO 的认知层（"脑子"）智能，但 Motor 层（"手脚"）残废。**
**SB3 的 Policy（"手脚"）训练后很强，但缺乏 η-aware 适应性。**
**混合 = IDO 脑子 + SB3 身体 = η-aware adaptive behavior。**

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                 IDO+SB3 Hybrid Agent                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  SB3 Policy  │    │  IDO Meta    │    │  Noether     │  │
│  │  (PPO/SAC)   │    │  Layer       │    │  Filter      │  │
│  │              │    │              │    │              │  │
│  │ • Trained    │    │ • EML → κ    │    │ • Energy     │  │
│  │ • Good motor │    │ • η monitor  │    │ • Torque     │  │
│  │ • baseline   │    │ • ψ-Anchor   │    │ • Collision  │  │
│  │              │    │ • Creative-  │    │              │  │
│  │              │    │   Probe      │    │              │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Action Modulation Engine                    │  │
│  │                                                       │  │
│  │  base_action = SB3_policy.predict(obs)               │  │
│  │                                                       │  │
│  │  MODE SELECTION (based on η trajectory):              │  │
│  │                                                       │  │
│  │  EXPLOIT: η improving → deterministic SB3 action      │  │
│  │  EXPLORE: η stagnation → Creative-Probe perturbation  │  │
│  │  SAFE:   Noether violation → reduce action magnitude  │  │
│  │                                                       │  │
│  │  final_action = modulated base_action                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 三种运行模式

### 1. EXPLOIT 模式（η 趋势良好）
- 条件：η 在衰减（η_t < η_{t-1}）或 FlowMatching 预测 η 会继续改善
- 行动：使用 SB3 的 deterministic action（最高性能）
- 这是 "站稳了就别动" 模式 — 让训练好的 policy 发挥

### 2. EXPLORE 模式（η 停滞）
- 条件：κ-Snap 检测到 η 停滞（relative plateau threshold）
- 行动：Creative-Probe — 在 SB3 action 上叠加扰动
  - phase_offset perturbation（偏移 policy 的时序）
  - gain_multiplier perturbation（放大/缩小 action幅度）
  - action_noise perturbation（Gaussian noise叠加）
- ψ-Anchor 动态调整 δ_K 和 evolution policy
- κ-Snap 评判扰动是否被接受（η 是否改善）

### 3. SAFE 模式（Noether 违规预测）
- 条件：Noether 校验预测下一步会违反守恒律（能量/力矩/碰撞）
- 行动：减小 action magnitude（safe clip）
- 如果严重违规 → 切换到 task PD controller 的 safe_action
- 确保 IC-Value 不因守恒律破坏而暴跌

## 关键创新点

1. **η-aware exploration timing**: RL 的 exploration schedule 是固定的（ε-greedy decay），IDO 的 exploration 是 **η-driven** — 只在 η 停滞时才探索
2. **Conservation-law safety filter**: RL 不检查物理守恒律，IDO 的 Noether filter 确保 action 不违反能量/力矩约束
3. **Creative-Probe as smart perturbation**: RL 的 exploration 是 random noise，IDO 的 Creative-Probe 是 **结构化扰动**（phase offset, gain multiplier），保留 policy 的基本形态

## 价值主张（vs 纯 PPO/SAC）

| 特征 | 纯 PPO/SAC | IDO+SB3 Hybrid |
|------|-----------|----------------|
| 标准任务性能 | 高（训练结果） | ≈ 高（exploit 模式 ≈ deterministic） |
| η-aware exploration | ❌ 固定 schedule | ✅ η-driven（只在停滞时探索） |
| Conservation-law safety | ❌ 无 | ✅ Noether filter |
| 可解释性 | ❌ 黑箱 | ✅ η trajectory + κ-Snap + ψ-Anchor |
| 分布外适应 | ❌ 泛化差 | ✅ Creative-Probe + ψ-Anchor 动态调整 |
| 连续适应能力 | ❌ 需重新训练 | ✅ η-driven 在线适应 |

**IDO 的优势不在标准 benchmark 的 raw score**，
**而在 η-aware adaptive behavior、守恒律安全、可解释性、和分布外适应。**

## 实现计划

### 文件结构
```
agent/
  hybrid_sb3_ido_agent.py   ← 新的混合代理类
  mujoco_ido_agent.py       ← 保留（纯 IDO，用于对比）
baselines/
  sb3_adapter.py            ← 保留（提供 SB3 policy）
benchmarks/
  compare_hybrid.py         ← 新的对比脚本（IDO vs PPO vs Hybrid）
```

### HybridSB3IDOAgent 核心代码骨架

```python
class HybridSB3IDOAgent:
    """IDO+SB3 Hybrid Agent: η-aware adaptive behavior on trained policy."""
    
    def __init__(self, env, goal, sb3_adapter, task_name):
        self.sb3_adapter = sb3_adapter        # Trained PPO/SAC policy
        self.goal = goal                      # GoalEML for η computation
        self.psi_anchor = PsiAnchor(...)      # ψ-Anchor meta-layer
        self.flow_predictor = FlowMatchingEtaPredictor(...)
        self.task_controller = get_controller_for_task(task_name, physics)
        self.mode = 'exploit'                 # Current mode
        self.creative_probe_count = 0         # Creative-Probe counter
        
    def choose_action(self, timestep, physics):
        # 1. Get base action from SB3 policy
        base_action = self.sb3_adapter.choose_action(timestep)
        
        # 2. Compute η via κ-Snap (EML → residual)
        eta = gauss_ex_residual(observation, self.goal)
        
        # 3. Monitor η trajectory (ψ-Anchor + FlowMatching)
        self.psi_anchor.update(eta, ...)
        self.flow_predictor.update(eta)
        
        # 4. Mode selection
        if eta_stagnation_detected:
            self.mode = 'explore'  # Creative-Probe mode
        elif noether_violation_predicted:
            self.mode = 'safe'     # Conservation-law safety
        else:
            self.mode = 'exploit'  # Deterministic policy
        
        # 5. Action modulation based on mode
        if self.mode == 'exploit':
            final_action = base_action  # Deterministic, highest performance
            
        elif self.mode == 'explore':
            # Creative-Probe perturbation
            probe_type = random.choice(['noise', 'phase_offset', 'gain_multiplier'])
            final_action = creative_probe(base_action, probe_type, ...)
            
        elif self.mode == 'safe':
            # Noether safety filter
            final_action = np.clip(base_action, -safe_clip, safe_clip)
            # If severe violation → switch to PD safe_action
            
        # 6. Noether post-check
        nvr_result = noether_check_mj(prev_data, physics.data, self.goal)
        if not nvr_result['ok']:
            # Record violation, inform ψ-Anchor
            self.psi_anchor.record_noether_violation(nvr_result)
        
        return final_action
```

## Phase 3 实验设计

### 对照组
1. **纯 IDO**（开环PD控制器）— 当前 v0.5.5 基线
2. **纯 PPO/SAC**（训练好的 RL policy）— Phase 2 基线
3. **IDO+SB3 Hybrid**（η-aware adaptation on trained policy）— Phase 3 新方案

### 关键对比维度
- **标准 benchmark score**: Hybrid ≈ PPO/SAC（exploit 模式 ≈ deterministic）
- **η-aware exploration**: 只在 η 停滞时探索 → 更高效的 exploration
- **Noetter safety**: Hybrid 的 NVR ≈ 0 vs PPO/SAC 的 NVR > 0
- **分布外适应**: 改变任务参数（如 walker 目标速度从 1→3 m/s），Hybrid 应能 η-driven 适应
- **可解释性**: Hybrid 的决策路径可通过 η trajectory 完全解释

### 评估脚本
```python
# compare_hybrid.py
# 1. Train PPO/SAC (Phase 2, already done)
# 2. Run IDO baseline (current)
# 3. Run Hybrid (IDO meta-layer + SB3 policy)
# 4. Compare: avg_return, NVR, η trajectory, success_rate
# 5. Distribution-shift test: change task parameters
```
