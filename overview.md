# MuJoCo-Bench-IDO — v0.19.0

## TL;DR
焊接系统大规模增强：18种焊缝类型（+125%）、20项物理指标（+43%）、16传感器完整套件、NumPy向量化性能优化。681/681测试零回归。

## 四大增强

### 1. 焊缝类型 8→18种
新增10种焊缝接头类型（AWS A3.0 / AWS D1.1 / ISO 15614 / EN ISO 4063）：

| 类型 | 中文 | 电流 | 电压 | 速度 | eta | 熔深 | t85 | HAZ | HV |
|------|------|------|------|------|-----|------|-----|-----|-----|
| corner | 角接焊缝 | 200A | 24V | 6mm/s | 0.000 | 2.29 | 5.4s | 2.50 | 409 |
| edge | 边缘焊缝 | 180A | 22V | 7mm/s | 0.002 | 1.88 | 7.6s | 2.10 | 376 |
| plug | 塞焊焊缝 | 210A | 23V | 4mm/s | 0.001 | 2.97 | 3.6s | 3.07 | 437 |
| slot | 槽焊焊缝 | 200A | 22V | 4.5mm/s | 0.001 | 2.59 | 4.4s | 2.76 | 424 |
| surfacing | 堆焊 | 230A | 25V | 5mm/s | 0.000 | 2.14 | 3.7s | 3.00 | 434 |
| tack | 定位焊 | 190A | 23V | 8mm/s | 0.001 | 1.68 | 7.9s | 2.07 | 372 |
| butt | 对接焊缝 | 210A | 24V | 5mm/s | 0.000 | 3.14 | 4.3s | 2.81 | 426 |
| tee | T形焊缝 | 215A | 24V | 5.5mm/s | 0.000 | 2.48 | 4.6s | 2.71 | 421 |
| multipass | 多层焊 | 225A | 25V | 5mm/s | 0.001 | 2.57 | 3.8s | 2.96 | 433 |
| repair | 补焊 | 195A | 23V | 5mm/s | 0.001 | 2.91 | 4.8s | 2.65 | 418 |

原有8种类型（flat/horizontal/vertical/overhead/fillet/groove/lap/pipe）零回归。

### 2. 逼真物理仿真（+6个方法）
- **t8/5冷却时间**: 800°C→500°C冷却时间，决定微观组织
- **微观组织相变**: 马氏体/贝氏体/铁素体/珠光体比例
- **热影响区宽度**: HAZ ∝ √(heat_input)
- **最大硬度**: HV = 200 + CE×400 + max(0, 10-t85)×15
- **残余应力**: σ ≈ σ_yield × (0.6 + 0.4×heat_factor)
- **凝固裂纹敏感性**: CSR ∝ I / (v × bead_width)

### 3. 完整传感器套件（16种）
`core/welding_sensors.py` — WeldingSensorSuite 类：

| 传感器 | 单位 | 噪声 |
|--------|------|------|
| arc_voltage | V | ±0.1 |
| arc_current | A | ±2 |
| wire_feed_speed | m/min | ±0.1 |
| gas_flow | L/min | ±0.5 |
| contact_tip_temp | °C | ±5 |
| weld_pool_width | mm | ±0.2 |
| weld_pool_length | mm | ±0.3 |
| arc_sound | dB | ±2 |
| ir_temp_1 (焊缝) | °C | ±10 |
| ir_temp_2 (HAZ) | °C | ±8 |
| ir_temp_3 (母材) | °C | ±5 |
| magnetic_arc_blow | mT | ±0.01 |
| pool_oscillation | Hz | ±1 |
| spatter_count | p/s | ±5 |
| seam_tracking | mm | ±0.1 |
| bead_profile | 3D | - |

### 4. 性能优化
- 预计算参数范围倒数（避免除法）
- NumPy向量化eta计算（替代for循环）
- 缓存sqrt计算（热输入公式复用）
- EMPIRICAL_COEFFS提取为实例属性

## 修改文件清单
| 文件 | 变更 |
|------|------|
| core/welding_process_proxy.py | +483/-73 — 10新类型+6物理方法+性能优化 |
| core/welding_sensors.py | +591/-309 — 16传感器完整套件重写 |
| envs/welding_env.py | +33 — 3字典扩展10新类型 |
| benchmarks/welding_eval.py | +13/-3 — WELD_TYPE_OPTIMAL扩展 |
| baselines/sac_weld_train.py | +4/-2 — choices扩展到18种 |
| tests/test_welding_integration.py | +42/-28 — 适配新WeldingQuality字段 |
| tests/test_welding_proxy.py | +3 — 适配新字段 |
| papers/mujoco_bench_ido_中文论文.md | +90 — Section 13 |

## 测试
- 681/681 pass (100%)，零回归

## Git
- Commit: v0.19.0
- 推送到 origin/main
