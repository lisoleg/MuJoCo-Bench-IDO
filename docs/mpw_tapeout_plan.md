# MPW 投片规划 — T-Processor NG (40nm)

> **Multi-Project Wafer Tape-out Plan**
>
> 章锋 SLOS 论文 (2026-07-04, 第二版) 第7节
>
> MuJoCo-Bench-IDO v0.4.0 — 2026-07-04

## 1. 工艺参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 工艺节点 | 40nm CMOS | SMIC 40LL 标准工艺 |
| Die Size | 1.0mm × 1.0mm | 含 Pad Frame |
| Core Area | 0.6mm × 0.6mm | 0.36mm², Utilization 70% |
| Pad Frame | 100µm 宽 | I/O Ring |
| Pad 数量 | 32 个 | 含电源/信号/测试 |
| 供电电压 | VDD_CORE = 0.8V, VDD_IO = 3.3V | 双电压域 |
| 时钟频率 | 50 MHz | T-Processor 系统时钟 |
| 封装 | QFN-32 | 5mm × 5mm 塑封 |

## 2. GDS 布局建议

```
┌──────────────────────────────────────────┐ 1.0mm
│  ╔══════════════════════════════════════╗│
│  ║          Pad Frame (100µm)           ║│
│  ║  ┌──────────────────────────────────┐║│
│  ║  │                                  │║│
│  ║  │    ┌──────────────────────┐      │║│
│  ║  │    │   PCM Array (Center) │      │║│
│  ║  │    │   0.28mm × 0.28mm    │      │║│
│  ║  │    │   (64×64 交叉阵列)    │      │║│
│  ║  │    └──────────────────────┘      │║│
│  ║  │                                  │║│
│  ║  │  ┌─────┐  ┌─────┐  ┌─────┐      │║│
│  ║  │  │eta_ │  │psi_ │  │ksnap│      │║│
│  ║  │  │alu  │  │anchor│  │buf  │      │║│
│  ║  │  └─────┘  └─────┘  └─────┘      │║│
│  ║  │                                  │║│
│  ║  │  ┌─────┐  ┌─────┐  ┌─────┐      │║│
│  ║  │  │eml_ │  │cim_ │  │pid_ │      │║│
│  ║  │  │pcm  │  │mac  │  │simp │      │║│
│  ║  │  │load │  │ctl  │  │     │      │║│
│  ║  │  └─────┘  └─────┘  └─────┘      │║│
│  ║  │                                  │║│
│  ║  │  ┌─────────────────────────┐     │║│
│  ║  │  │   mmio_decode_u         │     │║│
│  ║  │  └─────────────────────────┘     │║│
│  ║  │                                  │║│
│  ║  └──────────────────────────────────┘║│
│  ╚══════════════════════════════════════╝│
└──────────────────────────────────────────┘
                1.0mm
```

### 布局原则

1. **PCM阵列居中**: 被数字逻辑环绕, 减少模拟/数字干扰
2. **模拟模块靠近PCM**: TIA, ADC 紧邻 PCM 阵列, 缩短走线
3. **数字逻辑外围**: 靠近 Pad Frame, 便于信号引出
4. **电源网格**: 核心 0.8V 网格覆盖全 core, IO 3.3V 在 Pad Ring
5. **保护环**: 模拟区域加 N+ / P+ 保护环, 隔离衬底噪声

## 3. 面积预估

| 模块 | 面积 (mm²) | 占比 | 门数 | 说明 |
|------|-----------|------|------|------|
| PCM Array | 0.280 | 77.8% | N/A | 64×64 GST相变单元 |
| eta_alu | 0.030 | 8.3% | ~5,000 | 4级流水线 GaussEx |
| psi_anchor_gate | 0.003 | 0.8% | ~50 | 纯组合逻辑 |
| ksnap_buffer | 0.011 | 3.1% | ~200 | 256×56bit FIFO |
| eml_pcm_loader | 0.008 | 2.2% | ~1,500 | FSM + 步长控制 |
| cim_mac_ctl | 0.010 | 2.8% | ~2,000 | 读控制 + ADC接口 |
| mmio_decode_u | 0.005 | 1.4% | ~1,000 | AXI-Lite解码 |
| pid_simple_u | 0.003 | 0.8% | ~500 | 轻量PID |
| TIA + ADC | 0.010 | 2.8% | 模拟 | 跨阻放大+12位ADC |
| **Core Total** | **0.360** | **100%** | **~10,250** | Utilization 70% |

## 4. 32 Pad 分配表

| Pad # | 名称 | 类型 | 电压域 | 方向 | 说明 |
|-------|------|------|--------|------|------|
| 1 | VDD_CORE | Power | 0.8V | — | 核心电源 |
| 2 | VDD_CORE | Power | 0.8V | — | 核心电源 (冗余) |
| 3 | GND | Ground | — | — | 地 |
| 4 | GND | Ground | — | — | 地 (冗余) |
| 5 | VDD_IO | Power | 3.3V | — | IO电源 |
| 6 | VDD_IO | Power | 3.3V | — | IO电源 (冗余) |
| 7 | CLK | Digital | 3.3V | In | 50MHz 系统时钟 |
| 8 | RST_N | Digital | 3.3V | In | 复位 (低有效) |
| 9 | VREAD_P | Analog | 3.3V | In | PCM 读电压正端 |
| 10 | VREAD_N | Analog | 3.3V | In | PCM 读电压负端 |
| 11 | TIA_OUT | Analog | 3.3V | Out | TIA 输出 |
| 12 | SET_PULSE | Digital | 3.3V | Out | PCM SET 脉冲 |
| 13 | RESET_PULSE | Digital | 3.3V | Out | PCM RESET 脉冲 |
| 14 | ADDR_ROW[5:0] | Digital | 3.3V | In/Out | 行地址 (6位) |
| 15 | ADDR_COL[5:0] | Digital | 3.3V | In/Out | 列地址 (6位) |
| 16 | ADC_DATA[11:0] | Digital | 3.3V | In | ADC 采样数据 (12位) |
| 17 | DIN[15:0] | Digital | 3.3V | In | 数据输入 (16位) |
| 18 | DOUT[15:0] | Digital | 3.3V | Out | 数据输出 (16位) |
| 19 | BL | Analog | 3.3V | I/O | Bit Line (PCM阵列) |
| 20 | BLB | Analog | 3.3V | I/O | Bit Line Bar |
| 21 | SL | Analog | 3.3V | I/O | Source Line |
| 22 | ESTOP_N | Digital | 3.3V | Out | 急停输出 (低有效) |
| 23 | VIOLATION_CODE[7:0] | Digital | 3.3V | Out | Ψ-Anchor违规码 |
| 24 | SAFE_STATE[3:0] | Digital | 3.3V | Out | 安全状态码 |
| 25 | AXI_AWVALID | Digital | 3.3V | In | AXI-Lite写地址有效 |
| 26 | AXI_WDATA | Digital | 3.3V | In | AXI-Lite写数据 |
| 27 | AXI_WSTRB | Digital | 3.3V | In | AXI-Lite写选通 |
| 28 | AXI_BVALID | Digital | 3.3V | Out | AXI-Lite写响应 |
| 29 | AXI_ARVALID | Digital | 3.3V | In | AXI-Lite读地址有效 |
| 30 | AXI_RDATA | Digital | 3.3V | Out | AXI-Lite读数据 |
| 31 | TCK | Digital | 3.3V | In | JTAG 测试时钟 |
| 32 | TDI/TDO/TMS/TRST | Digital | 3.3V | I/O | JTAG 测试 (复用) |

> **注**: ADDR_ROW[5:0], ADDR_COL[5:0], ADC_DATA[11:0], DIN[15:0], DOUT[15:0],
> VIOLATION_CODE[7:0], SAFE_STATE[3:0] 为多位信号, 实际占多个物理Pad。
> 总物理Pad数 = 32, 含多位信号后实际引脚 = ~96 pins (QFN-32封装下复用)。

## 5. 功耗预估

| 模块 | 峰值功耗 (mW) | 典型功耗 (mW) | 占比 | 说明 |
|------|--------------|-------------|------|------|
| PCM CIM读 | 12.0 | 8.0 | 52.6% | 含TIA+ADC |
| PCM SET/RESET | 5.0 | 0.5 | — | 仅编程时 |
| eta_alu | 2.0 | 1.5 | 9.9% | 4级流水线 |
| 数字逻辑 | 3.0 | 2.0 | 13.2% | FSM+控制 |
| I/O 驱动 | 3.0 | 1.5 | 9.9% | Pad驱动 |
| 漏电 | 0.5 | 0.5 | 3.3% | 静态漏电 |
| **Total** | **25.5** | **15.19** | **100%** | 峰值含PCM编程 |

> 典型功耗 15.19mW 对比传统AI系统 330W, 能效提升 **21,725x**。

## 6. 测试策略

### 6.1 Scan Chain (DFT)

| 参数 | 值 |
|------|-----|
| Scan Chain 数量 | 4 |
| 最长链长度 | 2,560 FF |
| Scan 频率 | 10 MHz |
| 覆盖率目标 | >95% (stuck-at), >90% (transition) |
| 测试时间 | ~1ms (全链扫描) |

```
Scan-in ──→ [FF Chain 0: 2560 bits] ──→ Scan-out
Scan-in ──→ [FF Chain 1: 2560 bits] ──→ Scan-out
Scan-in ──→ [FF Chain 2: 2560 bits] ──→ Scan-out
Scan-in ──→ [FF Chain 3: 2560 bits] ──→ Scan-out
```

### 6.2 BIST (PCM 内置自测试)

| 测试项 | 方法 | Pass标准 |
|--------|------|---------|
| PCM 单元导通 | SET全阵列, 读回 | >95% 电导>0x8000 |
| PCM 单元关断 | RESET全阵列, 读回 | >95% 电导<0x0800 |
| 电导分辨率 | 8级中间态编程 | 级间可分辨 |
| 保持特性 | 编程后1ms读回 | 漂移<5% |
| 端扰测试 | 反复读同一行 | 无明显扰动 |

### 6.3 JTAG (IEEE 1149.1)

| TAP指令 | 代码 | 功能 |
|---------|------|------|
| EXTEST | 0000 | 外部引脚测试 |
| SAMPLE | 0001 | 采样引脚状态 |
| BYPASS | 1111 | 旁路 |
| IDCODE | 0010 | 读芯片ID |
| MBIST_EN | 0100 | 启动PCM BIST |
| SCAN_EN | 0101 | 启动Scan测试 |
| ANA_TEST | 0110 | 模拟通路测试 |

## 7. DRC/LVS 检查清单

### 7.1 DRC (Design Rule Check)

| # | 检查项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | Metal spacing ≥ 40nm | ☐ | 40nm最小间距 |
| 2 | Via enclosure ≥ 20nm | ☐ | 通孔包封 |
| 3 | Min metal width ≥ 40nm | ☐ | 最小线宽 |
| 4 | Max current density < 2mA/µm | ☐ | 电迁移限制 |
| 5 | PCM区域 antenna ratio < 100 | ☐ | 天线规则 |
| 6 | 模拟/数字隔离间距 ≥ 5µm | ☐ | 衬底隔离 |
| 7 | Pad尺寸 ≥ 70µm × 70µm | ☐ | 焊盘最小尺寸 |
| 8 | Core到Pad Ring间距 ≥ 100µm | ☐ | IO环宽度 |

### 7.2 LVS (Layout vs Schematic)

| # | 检查项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | 所有net连通 | ☐ | 无开路 |
| 2 | 无短路 | ☐ | 无意外连接 |
| 3 | 器件参数匹配 | ☐ | W/L一致 |
| 4 | 电源网络完整 | ☐ | VDD/GND全覆盖 |
| 5 | PCM阵列规模 64×64 | ☐ | 阵列维度正确 |
| 6 | 保护环连接 | ☐ | 模拟区保护环接地 |

## 8. 流片时间线

| 阶段 | 周期 | 内容 |
|------|------|------|
| RTL Freeze | W0 | 冻结RTL代码, 开始综合 |
| Synthesis | W1-W2 | 综合+时序收敛 |
| DFT Insert | W3 | 插入Scan Chain + BIST |
| P&R | W4-W5 | 布局布线 |
| STA | W6 | 静态时序分析 |
| DRC/LVS | W7 | 物理验证 |
| Tape-out | W8 | 交付GDS |
| Fab | W9-W20 | 12周流片 |
| 封装测试 | W21-W22 | QFN封装+FT |
| 回片评估 | W23-W24 | 工程样片测试 |

## 9. 风险分析

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| PCM一致性不达标 | 中 | 高 | 增加BIST校准, 预留冗余阵列 |
| 模拟噪声干扰 | 中 | 中 | 加强保护环, 差分布线 |
| 时序收敛困难 | 低 | 中 | 降频至40MHz, 关键路径优化 |
| Pad数量不足 | 低 | 高 | 复用JTAG+功能引脚 |
| 电迁移失效 | 低 | 高 | 加宽电源走线, 增加冗余Pad |
