# 焊接EML因果蒸馏参考文档

> 章锋2026-07-04论文《面向硅基生命操作系统的焊接机器人多模态数据采集与因果蒸馏框架》技术参考

## 1. 八元数代数基础

### 1.1 Cayley-Dickson 构造

八元数 (Octonion, O) 是Cayley-Dickson构造的第三层超复数:

```
复数 C = R[i]        (i² = -1)
四元数 H = C[j]      (j² = -1, ij = -ji = k)
八元数 O = H[l]      (l² = -1, 虚部乘法非交换非结合)
```

八元数有8个分量:

```
O = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇
```

其中 e₀ = 1 (实部), e₁...e₇ 为虚部基底。

### 1.2 乘法表

| × | e₀ | e₁ | e₂ | e₃ | e₄ | e₅ | e₆ | e₇ |
|---|----|----|----|----|----|----|----|----|
| **e₀** | e₀ | e₁ | e₂ | e₃ | e₄ | e₅ | e₆ | e₇ |
| **e₁** | e₁ | -e₀ | e₃ | -e₂ | e₅ | -e₄ | -e₇ | e₆ |
| **e₂** | e₂ | -e₃ | -e₀ | e₁ | e₆ | e₇ | -e₄ | -e₅ |
| **e₃** | e₃ | e₂ | -e₁ | -e₀ | e₇ | -e₆ | e₅ | -e₄ |
| **e₄** | e₄ | -e₅ | -e₆ | -e₇ | -e₀ | e₁ | e₂ | e₃ |
| **e₅** | e₅ | e₄ | -e₇ | e₆ | -e₁ | -e₀ | -e₃ | e₂ |
| **e₆** | e₆ | e₇ | e₄ | -e₅ | -e₂ | e₃ | -e₀ | -e₁ |
| **e₇** | e₇ | -e₆ | e₅ | e₄ | -e₃ | -e₂ | e₁ | -e₀ |

### 1.3 关键性质

| 性质 | 公式 | 说明 |
|------|------|------|
| 非交换性 | a·b ≠ b·a | 虚部乘法不可交换 |
| 非结合性 | (a·b)·c ≠ a·(b·c) | 乘法不满足结合律 |
| 交错律 | (a·a)·b = a·(a·b) | 比结合律弱, 但仍成立 |
| 共轭 | a* = (a₀, -a₁, ..., -a₇) | 实部不变, 虚部取反 |
| 范数乘性 | \|a·b\| = \|a\|·\|b\| | 范数满足乘法性 |
| Fano平面 | Aut(Fano) = PSL(2,7), \|G\| = 168 | 自同构群阶数168 |

### 1.4 Python 实现

```python
from core.octonion_ops import OctonionOps, OctonionEMLNode

# 八元数乘法
a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
c = OctonionOps.mul(a, b)  # 非交换: OctonionOps.mul(b, a) ≠ c

# 从焊接状态构造EML节点
node = OctonionEMLNode.from_welding_state(
    current=200.0, voltage=24.0, speed=6.0, stickout=15.0,
    heat_input=0.8, penetration=2.0, porosity=0.02, distortion=0.5,
    weld_type="flat",
)
```

## 2. Φ流贯演化算子

### 2.1 定义

```
Φ(q, ω) = (q · ω) · q
```

左结合约定: 先计算 q·ω, 再将结果右乘 q。

### 2.2 物理含义

| 符号 | 含义 |
|------|------|
| q | 演化算子 (焊接工艺参数在八元代数空间的编码) |
| ω | 初始状态 (焊接质量指标的八元数表示) |
| Φ(q, ω) | 演化后的状态 (工艺参数对质量指标的因果影响) |

### 2.3 η残差

```
η = ||Φ(q, ω) - ω||²
```

- η ≈ 0: q 对 ω 的演化接近恒等 (保守操作)
- η 大: 演化偏离大 (激进操作)

## 3. EML因果蒸馏

### 3.1 网络结构

```
输入: x ∈ R^8 (归一化物理读数)
  │
  ├─→ Feature Extractor (8→128→128, ELU)
  │       │
  │       ├─→ to_oct → q ∈ R^8 (八元数演化算子)
  │       │
  │       └─→ omega_net (8→128→8) → ω ∈ R^8 (初始状态)
  │
  └─→ Φ(q, ω) = (q·ω)·q → p ∈ R^8 (演化结果)
```

### 3.2 蒸馏损失

```
L = α·L_recon + β·L_eta + γ·L_pareto + δ·L_sparse
```

| 项 | 公式 | 权重 (默认) | 说明 |
|----|------|-------------|------|
| L_recon | BCE(Σp², y_η) | α=1.0 | η残差重建 |
| L_eta | MSE(q₀, y_d) | β=0.5 | 物理量回归 |
| L_pareto | mean((Σq²-1)²) | γ=0.01 | 单位长度约束 |
| L_sparse | — | δ=0.01 | 稀疏性正则 |

### 3.3 PyTorch / Numpy 双后端

```python
from core.welding_eml_distillation import WeldingEMLDistiller, HAS_TORCH

distiller = WeldingEMLDistiller(hidden_dim=128)
# HAS_TORCH=True → 使用PyTorch (GPU加速, 自动微分)
# HAS_TORCH=False → numpy回退 (仅前向推理, 无梯度)
```

## 4. 焊接物理公式

### 4.1 目标熔深

```
target_pen = k_I × I² / (v × t)
```

| 参数 | 符号 | 默认值 | 单位 |
|------|------|--------|------|
| 电流系数 | k_I | 0.085 | — |
| 焊接电流 | I | — | A |
| 焊接速度 | v | — | mm/s |
| 板厚 | t | — | mm |

### 4.2 名义电压

```
V_nom = 16.0 + 2.0 × (thickness > 3)
```

- 厚度 ≤ 3mm → V_nom = 16V
- 厚度 > 3mm → V_nom = 18V

### 4.3 热输入

```
heat_input = (I × V) / (v × 1000)  [kJ/mm]
```

### 4.4 Python 调用

```python
from core.welding_process_proxy import WeldingProcessProxy

proxy = WeldingProcessProxy()
target, actual, dev = proxy.evaluate_detailed(I=200, V=24, v_mms=6, t_mm=2, stick_out=15)
v_nom = proxy.compute_nominal_voltage(thickness_mm=2.0)  # → 16.0
```

## 5. 异构计算基准

### 5.1 基准维度

| 维度 | 纯GPU | GPU+T-Processor |
|------|-------|------------------|
| 能耗 (J) | GPU 170W × 时间 | GPU 85W × 时间 + T-Proc 3.3mW × 时间 |
| 延迟 (μs/step) | ~650 (η+Ψ+Snap) | ~0.08 |
| Ψ-Check跳过率 | 2.0% | 0.01% |
| 事故成本 ($) | 高 | 极低 |

### 5.2 CLI 使用

```bash
# 表格输出
python -m tools.hetero_benchmark --steps 10000

# JSON输出
python -m tools.hetero_benchmark --steps 10000 --json
```

## 6. CIM 忆阻器交叉阵列

### 6.1 能耗对比

| 方案 | 能耗 (8×8 MVM) | 能效比 |
|------|----------------|--------|
| SRAM + ALU | 335.36 pJ | 1× |
| CIM (忆阻器) | ~0.8 pJ | ~400× |

### 6.2 原理

```
y[i] = Σ_j G[i,j] × x[j] × v_read   (基尔霍夫电流定律)
E = Σ_{i,j} G[i,j] × v_read² × dt   (总能耗)
```

## 7. κ-Snap 审计机制

### 7.1 审计条目结构

```python
snap_entry = {
    "eta": 0.0234,          # η残差
    "psi_passed": True,      # Ψ-Check是否通过
    "violation": "",         # 违规描述 (空=无违规)
    "step": 142,             # 步骤号
    "timestamp": 1.234,      # 时间戳 (s)
}
```

### 7.2 聚合统计

```python
from tools.wps_pqr_generator import WPSGenerator

gen = WPSGenerator()
stats = gen.aggregate_ksnap_stats(snap_entries)
# → {eta_mean, eta_max, eta_min, eta_std, psi_pass_rate, violation_types, ...}
```

## 8. 数据质量QA检查

### 8.1 检查维度

| 检查 | 方法 | 阈值 |
|------|------|------|
| 完整性 | NaN缺失率 | < 5% |
| 一致性 | 物理约束边界 | V∈[14,32], I∈[50,350] |
| 时效性 | 采样率一致性 | ±10% |
| 准确性 | 3σ异常值检测 | < 3% |
| HDF5结构 | 数据集存在性 | 全部存在 |

### 8.2 使用

```python
from tools.qa_data_health import WeldDataQACheck

qa = WeldDataQACheck()
result = qa.run_all(data_dict, sample_rate_hz=100.0)
```

## 9. 模块依赖关系

```
core/octonion_ops.py          ← 八元数代数基础
    ↓
core/welding_eml_distillation.py  ← EML蒸馏 (依赖octonion_ops)
    ↓
tools/hetero_benchmark.py     ← 异构基准 (独立, 可引用octonion)
tools/tproc_cim_simulator.py  ← CIM模拟 (独立)
core/welding_process_proxy.py ← 焊接物理 (独立, 已有模块升级)
tools/wps_pqr_generator.py    ← WPS/PQR生成 (独立, 已有模块升级)
tools/qa_data_health.py       ← 数据QA (独立)
```

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.3.0 | 2026-07-04 | 新增八元数EML、异构基准、CIM、QA模块 |
| v0.2.0 | 2026-07-03 | 焊接环境、控制器、安全模块 |
| v0.1.0 | 2026-07-01 | 初始MuJoCo-Bench-IDO框架 |
