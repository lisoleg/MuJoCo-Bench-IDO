## Kria K26 SOM 引脚约束 (T-Proc 边缘部署参考)
## ============================================================
## 基于 Kria K26 System-on-Module (XCZU5EV-2SBV484I)
## 章锋2026-07-04论文 — T-Proc边缘侧部署参考
##
## K26 SOM 使用PS-PL接口, 大部分引脚通过PS端配置.
## 本文件仅约束PL端专用引脚.

## ── PL时钟 ──
set_property PACKAGE_PIN A9  [get_ports pl_clk_100m]      ;# PL 100MHz (来自PS)
set_property IOSTANDARD LVCMOS18 [get_ports pl_clk_100m]

set_property PACKAGE_PIN B9  [get_ports pl_clk_300m]      ;# PL 300MHz (PLL输出)
set_property IOSTANDARD LVCMOS18 [get_ports pl_clk_300m]

## ── PS-PL AXI接口 (不用物理引脚, 用内部互联) ──
## AXI接口通过Zynq PS-PL总线, 不需要XDC约束
## 但需要设置AXI时钟域
set_property CONFIG.PCW_USE_S_AXI_HP0 1 [get_bd_cells processing_system7_0]
set_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 [get_bd_cells processing_system7_0]

## ── η-ALU GPIO接口 ──
## 通过EMIO扩展到PL端
set_property PACKAGE_PIN Y12 [get_ports {gpio_eta[0]}]    ;# η状态输出
set_property PACKAGE_PIN Y13 [get_ports {gpio_eta[1]}]    ;# Ψ-Check结果
set_property PACKAGE_PIN Y14 [get_ports {gpio_eta[2]}]    ;# κ-Snap触发
set_property PACKAGE_PIN Y15 [get_ports {gpio_eta[3]}]    ;# 中断请求
set_property IOSTANDARD LVCMOS18 [get_ports {gpio_eta[*]}]

## ── 焊接传感器接口 ──
## ADC接口 (12位, 1MSPS)
set_property PACKAGE_PIN AA12 [get_ports {sensor_adc[0]}]  ;# 焊接电流ADC
set_property PACKAGE_PIN AA13 [get_ports {sensor_adc[1]}]  ;# 焊接电压ADC
set_property PACKAGE_PIN AA14 [get_ports {sensor_adc[2]}]  ;# 温度ADC
set_property PACKAGE_PIN AA15 [get_ports {sensor_adc[3]}]  ;# 干伸长ADC
set_property IOSTANDARD LVCMOS18 [get_ports {sensor_adc[*]}]

set_property PACKAGE_PIN AB12 [get_ports sensor_adc_clk]   ;# ADC采样时钟 (1MHz)
set_property PACKAGE_PIN AB13 [get_ports sensor_adc_cs_n]  ;# ADC片选
set_property IOSTANDARD LVCMOS18 [get_ports sensor_adc_*]

## ── PWM输出 (送丝/焊接电流控制) ──
set_property PACKAGE_PIN AA16 [get_ports pwm_wire_feed]    ;# 送丝PWM (20kHz)
set_property PACKAGE_PIN AA17 [get_ports pwm_torch]        ;# 焊枪PWM (500Hz)
set_property IOSTANDARD LVCMOS18 [get_ports pwm_*]

## ── CAN总线 (机器人通信) ──
set_property PACKAGE_PIN W12 [get_ports can_tx]            ;# CAN TX
set_property PACKAGE_PIN W13 [get_ports can_rx]            ;# CAN RX
set_property IOSTANDARD LVCMOS18 [get_ports can_*]

## ── UART调试 ──
set_property PACKAGE_PIN W14 [get_ports uart_tx]
set_property PACKAGE_PIN W15 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS18 [get_ports uart_*]

## ── LED指示灯 ──
set_property PACKAGE_PIN AB18 [get_ports led_power]        ;# 电源指示
set_property PACKAGE_PIN AB19 [get_ports led_eta_low]      ;# η低 (绿色)
set_property PACKAGE_PIN AB20 [get_ports led_eta_high]     ;# η高 (红色)
set_property PACKAGE_PIN AB21 [get_ports led_ksi_snap]     ;# κ-Snap活跃
set_property IOSTANDARD LVCMOS18 [get_ports led_*]

## ── 时序约束 ──
create_clock -name pl_clk_100m -period 10.000 [get_ports pl_clk_100m]
create_clock -name pl_clk_300m -period 3.333 [get_ports pl_clk_300m]
create_clock -name adc_clk -period 1.000 [get_ports sensor_adc_clk]

## 跨时钟域约束
set_clock_groups -asynchronous \
    -group [get_clocks pl_clk_100m] \
    -group [get_clocks pl_clk_300m] \
    -group [get_clocks adc_clk]

## η-ALU 在300MHz域内完成 (3.333ns)
set_max_delay 3.333 -from [get_cells eta_alu_inst/*] -to [get_cells eta_alu_inst/*]

## ADC数据建立/保持时间
set_input_delay -clock adc_clk -max 0.500 [get_ports {sensor_adc[*]}]
set_input_delay -clock adc_clk -min 0.200 [get_ports {sensor_adc[*]}]

## PWM输出约束
set_output_delay -clock pl_clk_100m -max 2.000 [get_ports pwm_*]
set_output_delay -clock pl_clk_100m -min 1.000 [get_ports pwm_*]
