# tproc_ng_sdc.sdc
# T-Processor NG ASIC 综合约束
# 对应章锋论文附录E
# MuJoCo-Bench-IDO 硬件参考

create_clock -name clk -period 0.5 [get_ports clk]
set_input_delay 0.05 -clock clk [all_inputs]
set_output_delay 0.05 -clock clk [all_outputs]
set_max_area 1500000
set_max_power 0.08
set_max_delay 0.3 -from [get_pins OctonionALU/PhiOperator/*]
set_max_delay 0.15 -from [get_pins PsiCheckUnit/*]
set_min_retention_time 315360000 -cells [get_cells ReRAM_Array/*]

# 形式化断言
# 八元数左乘单位元: OP_MUL_L 且两操作数为0时, 结果应为16'h3C00 (FP16的1.0)
assert_property { @(posedge clk) (opcode == OP_MUL_L && rs1==0 && rs2==0) |-> ##1 result[15:0] == 16'h3C00 }

# 八元数共轭性质: OP_CONJ 且目标寄存器等于源寄存器时, 共轭后与原值相乘应为1
assert_property { @(posedge clk) (opcode == OP_CONJ && rd == rs1) |-> ##1 (oct_mul(result, original) == 1) }

# 约束说明:
#   时钟周期: 0.5ns (2GHz) — T-Processor NG目标频率
#   输入/输出延迟: 0.05ns — 极低延迟
#   最大面积: 1,500,000 μm²
#   最大功耗: 0.08W — 超低功耗
#   Φ算子最大延迟: 0.3ns — 八元数乘法
#   Ψ检查单元最大延迟: 0.15ns — Ψ-Anchor检查
#   ReRAM保持时间: 315,360,000ns (≈0.315s)
