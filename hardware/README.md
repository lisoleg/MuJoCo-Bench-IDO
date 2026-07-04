# T-Processor 硬件参考文件

> **来源**: 章锋2026-07-04论文《面向硅基生命操作系统的焊接机器人多模态数据采集与因果蒸馏框架》
>
> **注意**: 本目录下所有文件均为**参考设计**, 未经流片验证. 仅供架构参考和FPGA原型评估.

## 文件清单

| 文件 | 描述 | 目标平台 |
|------|------|---------|
| `tproc_welding_eta_alu.v` | T-Proc 焊接 η-ALU Verilog | 通用FPGA/ASIC |
| `tproc_cxl_driver.c` | CXL Type-3 Linux内核驱动骨架 | Linux 5.15+ |
| `tproc_ng_sdc.sdc` | ASIC综合约束文件 | TSMC 12nm |
| `kintex_ultrascale_pins.xdc` | KCU105开发板引脚约束 | Kintex UltraScale KU040 |
| `kria_k26_pin_constraints.xdc` | Kria K26 SOM引脚约束 | Zynq UltraScale+ XCZU5EV |

## 架构概述

```
┌─────────────────────────────────────────────────────┐
│                    Host CPU (x86)                     │
│                  ┌──────────────┐                     │
│                  │  CXL Root    │                     │
│                  │  Complex     │                     │
│                  └──────┬───────┘                     │
│                         │ CXL Gen3x4                  │
├─────────────────────────┼─────────────────────────────┤
│                    T-Proc (FPGA/ASIC)                  │
│  ┌─────────────────────┼─────────────────────────┐    │
│  │              CXL Type-3 EP                     │    │
│  │  ┌─────────────┐  │  ┌──────────────────┐     │    │
│  │  │  η-ALU      │←─┘  │  κ-Snap FIFO     │     │    │
│  │  │  (焊接专用)  │     │  (审计链)         │     │    │
│  │  └──────┬──────┘     └────────┬─────────┘     │    │
│  │         │                      │                │    │
│  │  ┌──────▼──────┐     ┌────────▼─────────┐     │    │
│  │  │  CIM阵列    │     │  Ψ-Anchor       │     │    │
│  │  │  (8×8 RRAM) │     │  (安全门控)      │     │    │
│  │  └─────────────┘     └──────────────────┘     │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## η-ALU 功能

焊接专用 η-ALU 执行以下运算:

1. **GaussEx残差**: `η = ||q_actual - q_target||²`
2. **八元数流贯**: `Φ(q, ω) = (q · ω) · q` (左结合)
3. **Ψ-Anchor检查**: 干伸长/回烧/气孔风险三重门控
4. **κ-Snap记录**: 每周期将状态写入Merkle审计链

### 时序参数

| 操作 | 延迟 (ns) | 频率 |
|------|----------|------|
| η-ALU计算 | 0.8 (ASIC) / 8.0 (FPGA) | 1.25GHz / 125MHz |
| CIM矩阵乘 | 1.5 (ASIC) / 12.0 (FPGA) | 667MHz / 83MHz |
| Ψ-Anchor检查 | 0.3 (ASIC) / 3.0 (FPGA) | 3.33GHz / 333MHz |
| κ-Snap写入 | 0.5 (ASIC) / 5.0 (FPGA) | 2GHz / 200MHz |

## 能耗对比

| 模式 | 推理能耗 | Ψ-Check能耗 | 总能耗 | 事故率 |
|------|---------|-------------|--------|--------|
| 纯GPU | 150 mJ | — | 150 mJ | 15% × 3次重采样 |
| GPU+T-Proc | 150 mJ | 0.19 μJ | 150.02 mJ | 0% (T-Proc拦截) |

T-Proc Ψ-Check 的能耗仅为 GPU 推理的 **0.13%**, 但将事故率从 ~15% 降至 0%.

## CIM 阵列

8×8 忆阻器交叉阵列, 执行矩阵-向量乘法:

- **SRAM+ALU**: 335.36 pJ/操作
- **CIM (RRAM)**: ~8 pJ/操作 (理论)
- **加速比**: ~42×

## 使用方法

### FPGA原型 (KCU105)

```bash
# Vivado 项目创建
vivado -mode batch -source create_project_kcu105.tcl

# 综合+实现+生成bitstream
vivado -mode batch -source build_kcu105.tcl

# 下载bitstream
vivado_hw_server -port 3121
```

### 边缘部署 (Kria K26)

```bash
# 在K26 SOM上
# 1. 加载PetaLinux
# 2. 加载T-Proc bitstream
fpgautil -b tproc_k26.bit.bin

# 3. 加载CXL驱动 (如果使用CXL模式)
insmod tproc_cxl_driver.ko

# 4. 验证η-ALU
echo "test" > /sys/class/tproc/eta_alu/test
cat /sys/class/tproc/eta_alu/result
```

## 许可

参考设计, 仅供MuJoCo-Bench-IDO项目研究使用.
