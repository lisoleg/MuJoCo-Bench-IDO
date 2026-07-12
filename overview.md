# MuJoCo-Bench-IDO — v0.21.0

## TL;DR
焊接系统材质库从4种扩展到10种+generic兜底，新增MUS多假设材质辨识器（epistemic humility），DIKWP规则库扩展至10条。723/723测试零回归。

## 参考文章
1. **DIKWP-IDO融合** — 跨材质支持 + IntentGuard四级安全
2. **云端驱动具身机器人** — 三层架构(L1云/L2适配/L0本地) + Psi-Anchor硬拦截
3. **从播放器到工匠** — 非标场景因果自适应 + DIKWP K-层规则库 + eta-PID闭环控制
4. **EML超图世界模型** — MUS多假设(epistemic humility) + material params(c_p, k, rho)编码

## 版本演进

### v0.21.0 新增：材质库扩展 + MUS辨识器

#### 10种材料 + generic兜底
| 材料 | 热导率(W/m·K) | 熔点(°C) | 密度(kg/m³) | 比热(J/kg·K) | 膨胀系数(1/K) | 牌号 |
|------|-------------|---------|-----------|------------|-------------|------|
| steel(钢) | 50 | 1500 | 7850 | 470 | 12e-6 | Q235 |
| aluminum(铝) | 167 | 660 | 2700 | 900 | 23e-6 | Al6061 |
| stainless(不锈钢) | 16 | 1450 | 8000 | 500 | 16e-6 | SS304 |
| titanium(钛) | 7 | 1668 | 4500 | 520 | 9e-6 | Ti-6Al-4V |
| copper(铜) | 398 | 1085 | 8960 | 385 | 17e-6 | T2纯铜 |
| nickel(镍) | 90 | 1455 | 8900 | 445 | 13e-6 | N6纯镍 |
| cast_iron(铸铁) | 50 | 1200 | 7200 | 540 | 10e-6 | HT250灰铸铁 |
| inconel(高温合金) | 9.8 | 1350 | 8440 | 410 | 12.8e-6 | Inconel 625 |
| magnesium(镁合金) | 156 | 650 | 1810 | 1020 | 25e-6 | AZ91D |
| bronze(青铜) | 75 | 950 | 8800 | 380 | 18e-6 | QSn6.5-0.1 |
| generic(通用兜底) | 50 | 1500 | 7850 | 470 | 12e-6 | 未知材料降级 |

**关键改进**：未知材料自动回退到generic，不再使用steel作为默认兜底。

#### MUS多假设材质辨识器
参考EML超图文章5.3节 epistemic humility概念：
- 输入可观测量（热导率/密度/熔点），输出最可能材料+置信度
- 置信度 < 0.6 时启用MUS模式（多假设并行eta演化）
- 无输入时返回generic + mus_mode=True
- generic不参与辨识（它是兜底，不是可辨识材料）

#### DIKWP规则库扩展至10条
新增4条规则覆盖新材质：
- R007: copper高热导率 → 电流+15A, 电压+0.5V
- R008: nickel薄板间隙 → 电流-10A, 电压-0.3V, 摆动+0.5mm
- R009: inconel低热导率 → 电流-5A, 电压-0.2V, 摆动+0.3mm
- R010: cast_iron冷裂风险 → 电流+20A, 电压+1.0V, 摆动-0.5mm

### v0.20.x 已有功能
- 25种焊缝类型 + generic兜底
- IntentGuard四级安全分类（SAFE/SUSPICIOUS/DANGEROUS/CRITICAL）
- 非标场景分类器（geometric/semantic/environmental/production）
- eta-PID自适应控制（有限差分梯度 + PID控制律 + 15%限幅）
- 20项物理指标（含t8/5冷却、微观组织、HAZ宽度、硬度、残余应力、裂纹敏感性）
- 16传感器套件

## API端点
| 端点 | 功能 |
|------|------|
| `/api/welding/types` | 25种焊缝类型 + 11种材质 |
| `/api/welding/identify_material` | MUS多假设材质辨识器 |
| `/api/welding/non_standard` | 非标场景分类 |
| `/api/welding/dikwp_rules` | DIKWP规则库参数调整（10条规则） |
| `/api/welding/eta_pid` | eta-PID自适应控制 |
| `/api/welding/intent_safety` | IntentGuard四级安全 |
| `/api/welding/status` | 焊接状态 |
| `/api/welding/quality` | 质量指标 |
| `/api/welding/sensors` | 16传感器数据 |
| `/api/welding/safety` | 安全约束 |
| `/api/welding/start` | 启动焊接 |
| `/api/welding/stop` | 停止焊接 |

## 测试
- 723/723 passed (+14 new tests for v0.21.0)
- 零回归

## Git
- Remote: https://github.com/lisoleg/MuJoCo-Bench-IDO
- Version: v0.21.0

## 服务
- Webviz: http://localhost:8080
- ARM100 Viewer: http://localhost:8091 (按需启动)
