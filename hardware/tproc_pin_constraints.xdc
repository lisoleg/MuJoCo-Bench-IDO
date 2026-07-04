# tproc_pin_constraints.xdc
# T-Processor 焊接模块引脚约束 (Xilinx Artix-7 / Kintex-7)
# 对应章锋论文附录E (FPGA原型验证)
# MuJoCo-Bench-IDO 硬件参考

# ── 时钟引脚 ──
# 核心时钟 250MHz (外部振荡器)
set_property PACKAGE_PIN K17 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

# Ψ-Check专用时钟 500MHz (独立PLL输出)
set_property PACKAGE_PIN K18 [get_ports clk_psi]
set_property IOSTANDARD LVCMOS33 [get_ports clk_psi]

# ── 复位引脚 ──
set_property PACKAGE_PIN N17 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
set_property PULLUP true [get_ports rst_n]

# ── 焊接参数输入引脚 (Q16.16定点数) ──
# i_current[15:0] — 焊接电流反馈
set_property PACKAGE_PIN J16 [get_ports {i_current[0]}]
set_property PACKAGE_PIN J17 [get_ports {i_current[1]}]
set_property PACKAGE_PIN J18 [get_ports {i_current[2]}]
set_property PACKAGE_PIN K15 [get_ports {i_current[3]}]
set_property PACKAGE_PIN K16 [get_ports {i_current[4]}]
set_property PACKAGE_PIN L15 [get_ports {i_current[5]}]
set_property PACKAGE_PIN L16 [get_ports {i_current[6]}]
set_property PACKAGE_PIN L17 [get_ports {i_current[7]}]
set_property PACKAGE_PIN M15 [get_ports {i_current[8]}]
set_property PACKAGE_PIN M16 [get_ports {i_current[9]}]
set_property PACKAGE_PIN M17 [get_ports {i_current[10]}]
set_property PACKAGE_PIN N15 [get_ports {i_current[11]}]
set_property PACKAGE_PIN N16 [get_ports {i_current[12]}]
set_property PACKAGE_PIN P15 [get_ports {i_current[13]}]
set_property PACKAGE_PIN P16 [get_ports {i_current[14]}]
set_property PACKAGE_PIN R15 [get_ports {i_current[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {i_current[*]}]

# i_voltage[15:0] — 焊接电压反馈
set_property PACKAGE_PIN T15 [get_ports {i_voltage[0]}]
set_property PACKAGE_PIN T16 [get_ports {i_voltage[1]}]
set_property PACKAGE_PIN U15 [get_ports {i_voltage[2]}]
set_property PACKAGE_PIN U16 [get_ports {i_voltage[3]}]
set_property PACKAGE_PIN V15 [get_ports {i_voltage[4]}]
set_property PACKAGE_PIN V16 [get_ports {i_voltage[5]}]
set_property PACKAGE_PIN W15 [get_ports {i_voltage[6]}]
set_property PACKAGE_PIN W16 [get_ports {i_voltage[7]}]
set_property PACKAGE_PIN Y16 [get_ports {i_voltage[8]}]
set_property PACKAGE_PIN Y17 [get_ports {i_voltage[9]}]
set_property PACKAGE_PIN Y18 [get_ports {i_voltage[10]}]
set_property PACKAGE_PIN AA16 [get_ports {i_voltage[11]}]
set_property PACKAGE_PIN AA17 [get_ports {i_voltage[12]}]
set_property PACKAGE_PIN AB16 [get_ports {i_voltage[13]}]
set_property PACKAGE_PIN AB17 [get_ports {i_voltage[14]}]
set_property PACKAGE_PIN AB18 [get_ports {i_voltage[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {i_voltage[*]}]

# i_arc_len_est[15:0] — 弧长估计值
set_property PACKAGE_PIN AC16 [get_ports {i_arc_len_est[0]}]
set_property PACKAGE_PIN AC17 [get_ports {i_arc_len_est[1]}]
set_property PACKAGE_PIN AD16 [get_ports {i_arc_len_est[2]}]
set_property PACKAGE_PIN AD17 [get_ports {i_arc_len_est[3]}]
set_property PACKAGE_PIN AE16 [get_ports {i_arc_len_est[4]}]
set_property PACKAGE_PIN AE17 [get_ports {i_arc_len_est[5]}]
set_property PACKAGE_PIN AF16 [get_ports {i_arc_len_est[6]}]
set_property PACKAGE_PIN AF17 [get_ports {i_arc_len_est[7]}]
set_property PACKAGE_PIN AF18 [get_ports {i_arc_len_est[8]}]
set_property PACKAGE_PIN AG16 [get_ports {i_arc_len_est[9]}]
set_property PACKAGE_PIN AG17 [get_ports {i_arc_len_est[10]}]
set_property PACKAGE_PIN AG18 [get_ports {i_arc_len_est[11]}]
set_property PACKAGE_PIN AH16 [get_ports {i_arc_len_est[12]}]
set_property PACKAGE_PIN AH17 [get_ports {i_arc_len_est[13]}]
set_property PACKAGE_PIN AJ16 [get_ports {i_arc_len_est[14]}]
set_property PACKAGE_PIN AJ17 [get_ports {i_arc_len_est[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {i_arc_len_est[*]}]

# ── 目标参数输入引脚 (Q16.16) ──
# tgt_current / tgt_voltage / tgt_arc_len
set_property PACKAGE_PIN D17 [get_ports {tgt_current[0]}]
set_property PACKAGE_PIN D18 [get_ports {tgt_current[1]}]
set_property PACKAGE_PIN E17 [get_ports {tgt_current[2]}]
set_property PACKAGE_PIN E18 [get_ports {tgt_current[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {tgt_current[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {tgt_voltage[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {tgt_arc_len[*]}]

# ── 输出引脚 ──
# o_delta_current[15:0] — 电流修正量
set_property PACKAGE_PIN F17 [get_ports {o_delta_current[0]}]
set_property PACKAGE_PIN F18 [get_ports {o_delta_current[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {o_delta_current[*]}]

# o_estop — 紧急停机 (关键输出, 使用IOB寄存器)
set_property PACKAGE_PIN G17 [get_ports o_estop]
set_property IOSTANDARD LVCMOS33 [get_ports o_estop]
set_property IOB true [get_ports o_estop]

# o_violation_code[7:0] — 违规码
set_property PACKAGE_PIN G18 [get_ports {o_violation_code[0]}]
set_property PACKAGE_PIN H17 [get_ports {o_violation_code[1]}]
set_property PACKAGE_PIN H18 [get_ports {o_violation_code[2]}]
set_property PACKAGE_PIN J15 [get_ports {o_violation_code[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {o_violation_code[*]}]

# o_weave_trigger — 摆动焊触发
set_property PACKAGE_PIN K14 [get_ports o_weave_trigger]
set_property IOSTANDARD LVCMOS33 [get_ports o_weave_trigger]

# ── 引脚约束说明:
#   目标器件: Xilinx Artix-7 XC7A200T-FBG676 (或 Kintex-7)
#   IO标准: LVCMOS33 (3.3V)
#   o_estop 使用 IOB=true: 强制输出寄存器放入IOB, 最小化时钟到输出延迟
#   rst_n 使用 PULLUP: 确保复位信号在未连接时为高 (非复位状态)
#   引脚分配基于 FMC HPC 连接器布局 (可按实际PCB调整)
