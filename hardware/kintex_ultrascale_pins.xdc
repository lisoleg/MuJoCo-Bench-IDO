## Kintex UltraScale KU040 引脚约束 (T-Proc η-ALU 参考设计)
## ============================================================
## 基于 KCU105 开发板 (Kintex UltraScale XCKU040-2FFVA1156E)
## 章锋2026-07-04论文附录S — T-Processor硬件参考
##
## 注意: 这是参考约束文件,实际使用需根据PCB设计调整.

## ── 时钟 ──
set_property PACKAGE_PIN G21 [get_ports clk_200m]        ;# 200MHz差分时钟正端
set_property PACKAGE_PIN G22 [get_ports clk_200m_n]       ;# 200MHz差分时钟负端
set_property IOSTANDARD LVDS [get_ports clk_200m]
set_property IOSTANDARD LVDS [get_ports clk_200m_n]

## ── 复位 ──
set_property PACKAGE_PIN H22 [get_ports rst_n]
set_property IOSTANDARD LVCMOS15 [get_ports rst_n]

## ── CXL接口 (PCIe Gen3x4) ──
## CXL Type-3 设备使用PCIe物理层
set_property PACKAGE_PIN D4  [get_ports {cxl_tx[0]}]     ;# PCIe TX Lane 0
set_property PACKAGE_PIN C5  [get_ports {cxl_tx[1]}]
set_property PACKAGE_PIN D6  [get_ports {cxl_tx[2]}]
set_property PACKAGE_PIN C7  [get_ports {cxl_tx[3]}]
set_property IOSTANDARD LVDS [get_ports {cxl_tx[*]}]

set_property PACKAGE_PIN E1  [get_ports {cxl_rx[0]}]     ;# PCIe RX Lane 0
set_property PACKAGE_PIN F2  [get_ports {cxl_rx[1]}]
set_property PACKAGE_PIN E3  [get_ports {cxl_rx[2]}]
set_property PACKAGE_PIN F4  [get_ports {cxl_rx[3]}]
set_property IOSTANDARD LVDS [get_ports {cxl_rx[*]}]

## ── η-ALU 数据总线 (32位) ──
## 连接T-Proc处理核心与外部SRAM/寄存器组
set_property PACKAGE_PIN J1  [get_ports {eta_data[0]}]
set_property PACKAGE_PIN J2  [get_ports {eta_data[1]}]
set_property PACKAGE_PIN J3  [get_ports {eta_data[2]}]
set_property PACKAGE_PIN J4  [get_ports {eta_data[3]}]
set_property PACKAGE_PIN J5  [get_ports {eta_data[4]}]
set_property PACKAGE_PIN J6  [get_ports {eta_data[5]}]
set_property PACKAGE_PIN J7  [get_ports {eta_data[6]}]
set_property PACKAGE_PIN J8  [get_ports {eta_data[7]}]
set_property IOSTANDARD LVCMOS15 [get_ports {eta_data[*]}]
## (pin 8-31 省略, 按Bank 44顺序分配)

## ── η-ALU 控制信号 ──
set_property PACKAGE_PIN K1  [get_ports eta_clk]          ;# η-ALU 时钟 (125MHz)
set_property PACKAGE_PIN K2  [get_ports eta_valid]        ;# 计算有效
set_property PACKAGE_PIN K3  [get_ports eta_ready]        ;# 计算就绪
set_property PACKAGE_PIN K4  [get_ports eta_opcode]       ;# 操作码: 0=GaussEx, 1=Φ, 2=η
set_property IOSTANDARD LVCMOS15 [get_ports eta_*]

## ── CIM 忆阻器阵列接口 (8x8) ──
## 行选择线 (Word Lines)
set_property PACKAGE_PIN L1  [get_ports {cim_wl[0]}]
set_property PACKAGE_PIN L2  [get_ports {cim_wl[1]}]
set_property PACKAGE_PIN L3  [get_ports {cim_wl[2]}]
set_property PACKAGE_PIN L4  [get_ports {cim_wl[3]}]
set_property PACKAGE_PIN L5  [get_ports {cim_wl[4]}]
set_property PACKAGE_PIN L6  [get_ports {cim_wl[5]}]
set_property PACKAGE_PIN L7  [get_ports {cim_wl[6]}]
set_property PACKAGE_PIN L8  [get_ports {cim_wl[7]}]
set_property IOSTANDARD LVCMOS15 [get_ports {cim_wl[*]}]

## 列读出线 (Bit Lines) — 连接ADC
set_property PACKAGE_PIN M1  [get_ports {cim_bl[0]}]
set_property PACKAGE_PIN M2  [get_ports {cim_bl[1]}]
set_property PACKAGE_PIN M3  [get_ports {cim_bl[2]}]
set_property PACKAGE_PIN M4  [get_ports {cim_bl[3]}]
set_property PACKAGE_PIN M5  [get_ports {cim_bl[4]}]
set_property PACKAGE_PIN M5  [get_ports {cim_bl[5]}]
set_property PACKAGE_PIN M7  [get_ports {cim_bl[6]}]
set_property PACKAGE_PIN M8  [get_ports {cim_bl[7]}]
set_property IOSTANDARD LVCMOS15 [get_ports {cim_bl[*]}]

## ── LED调试 ──
set_property PACKAGE_PIN AP8 [get_ports led_eta_low]      ;# η<阈值 (绿)
set_property PACKAGE_PIN H9  [get_ports led_eta_high]     ;# η>阈值 (红)
set_property IOSTANDARD LVCMOS15 [get_ports led_*]

## ── 时序约束 ──
create_clock -name clk_200m -period 5.000 [get_ports clk_200m]
create_clock -name eta_clk -period 8.000 [get_ports eta_clk]
set_clock_groups -asynchronous -group [get_clocks clk_200m] -group [get_clocks eta_clk]

## η-ALU 路径约束: 8ns内完成八元数Φ运算
set_max_delay 8.000 -from [get_ports eta_valid] -to [get_ports eta_ready]

## CIM 读出路径: 2ns建立时间
set_input_delay -clock eta_clk -max 1.000 [get_ports {cim_bl[*]}]
set_input_delay -clock eta_clk -min 0.500 [get_ports {cim_bl[*]}]
