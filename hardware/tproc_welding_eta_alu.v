// tproc_welding_eta_alu.v
// T-Processor 焊接专用 η-ALU（电流环 / 弧长环）
// 对应章锋论文附录H
// MuJoCo-Bench-IDO 硬件参考

`timescale 1ns / 1ps

package WeldPkg;
    parameter DW        = 16;
    parameter LAMBDA_ARC  = 16'h3A00;
    parameter LAMBDA_VOLT = 16'h3400;
    parameter MAX_CURRENT = 16'h56C0;
endpackage
import WeldPkg::*;

module WeldingEtaALU (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [DW-1:0]    i_current,
    input  wire [DW-1:0]    i_voltage,
    input  wire [DW-1:0]    i_arc_len_est,
    input  wire [DW-1:0]    tgt_current,
    input  wire [DW-1:0]    tgt_voltage,
    input  wire [DW-1:0]    tgt_arc_len,
    output reg  [DW-1:0]    o_delta_current,
    output reg              o_estop,
    output reg  [7:0]       o_violation_code,
    output reg              o_weave_trigger
);

    wire [DW-1:0] err_arc, err_volt, eta_raw;
    assign err_arc  = i_arc_len_est - tgt_arc_len;
    assign err_volt = i_voltage - tgt_voltage;
    assign eta_raw = (LAMBDA_ARC * err_arc * err_arc) + (LAMBDA_VOLT * err_volt * err_volt);

    always @(*) begin
        o_estop = 1'b0; o_violation_code = 8'h00; o_weave_trigger = 1'b0;
        if (i_current > MAX_CURRENT && i_voltage < 16'h4200) begin
            o_estop = 1'b1; o_violation_code = 8'h01;
        end
        if (i_voltage > 16'h4F80 && eta_raw > 16'h5000) begin
            o_estop = 1'b1; o_violation_code = 8'h02;
        end
        if (err_arc > 16'h3800) begin
            o_weave_trigger = 1'b1;
        end
    end

    reg [DW-1:0] last_err_arc, integral;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_delta_current <= 0; last_err_arc <= 0; integral <= 0;
        end else begin
            o_delta_current <= err_arc * 16'h3C00 + (err_arc - last_err_arc) * 16'h3400;
            last_err_arc <= err_arc;
        end
    end
endmodule

// η残差计算公式:
// η_raw = λ_arc × (arc_len_est - tgt_arc_len)² + λ_volt × (voltage - tgt_voltage)²
//
// Ψ-锚违规码:
//   0x01: 电流 > MAX 且 电压 < 16'h4200 → 紧急停机 (BURN_BACK)
//   0x02: 电压 > 16'h4F80 且 η_raw > 16'h5000 → 紧急停机
//   —:    弧长误差 > 16'h3800 → 触发摆动焊 (weave)
//
// 控制律:
//   ΔI = Kp × err_arc + Kd × (err_arc - last_err_arc)
//   Kp = 16'h3C00, Kd = 16'h3400 (定点数)
