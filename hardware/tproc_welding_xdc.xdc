# tproc_welding_xdc.xdc
# T-Processor Welding Module Vivado XDC 约束文件
# 对应章锋论文附录E (FPGA原型验证)
# MuJoCo-Bench-IDO 硬件参考

# ── 时钟约束 ──
# 主时钟: 250 MHz (4ns周期) — T-Processor FPGA原型目标频率
create_clock -name clk_core -period 4.000 [get_ports clk]
create_clock -name clk_psi -period 2.000 [get_ports clk_psi]

# 时钟域交叉约束 (异步)
set_clock_groups -asynchronous \
    -group [get_clocks clk_core] \
    -group [get_clocks clk_psi]

# ── 输入/输出延迟约束 ──
# η-ALU输入: 电流/电压/弧长 (Q16.16定点)
set_input_delay  -clock clk_core -max 0.8 [get_ports {i_current[*] i_voltage[*] i_arc_len_est[*]}]
set_input_delay  -clock clk_core -min 0.2 [get_ports {i_current[*] i_voltage[*] i_arc_len_est[*]}]
set_output_delay -clock clk_core -max 0.6 [get_ports {o_delta_current[*]}]
set_output_delay -clock clk_core -min 0.1 [get_ports {o_delta_current[*]}]

# Ψ-Check输出: 紧急停机信号 (关键路径, 极低延迟)
set_output_delay -clock clk_psi -max 0.3 [get_ports {o_estop o_violation_code[*]}]
set_output_delay -clock clk_psi -min 0.05 [get_ports {o_estop o_violation_code[*]}]

# 摆动焊触发 (非关键路径)
set_output_delay -clock clk_core -max 1.0 [get_ports {o_weave_trigger}]

# ── 多周期路径 ──
# 八元数乘法需要3个周期完成 (Cayley-Dickson展开)
set_multicycle_path -from [get_pins WeldingEtaALU/err_arc_reg[*]/Q] \
                    -to   [get_pins WeldingEtaALU/eta_raw_reg[*]/D] 3

# ── 假路径 ──
# 复位路径不需要时序分析
set_false_path -from [get_ports rst_n]

# ── 扇出限制 ──
# Ψ-Check紧急停机信号扇出限制 (确保快速传播)
set_property MAX_FANOUT 20 [get_nets o_estop]

# ── IO标准 ──
# 焊接I/O使用LVCMOS33 (3.3V FPGA IO)
set_property IOSTANDARD LVCMOS33 [get_ports {clk clk_psi rst_n}]
set_property IOSTANDARD LVCMOS33 [get_ports {i_current[*] i_voltage[*] i_arc_len_est[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {tgt_current[*] tgt_voltage[*] tgt_arc_len[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {o_delta_current[*] o_estop o_violation_code[*] o_weave_trigger}]

# ── 约束说明:
#   clk_core: 250MHz — T-Processor核心时钟 (η-ALU, PID控制)
#   clk_psi:  500MHz — Ψ-Check独立时钟 (超低延迟安全检查)
#   异步时钟域交叉: 使用set_clock_groups隔离
#   八元数乘法: 3周期多周期路径
#   o_estop扇出限制20: 确保紧急停机信号快速传播
