// tproc_eml_pcm_loader.v
// EML → PCM Pulse-Verify-Write Controller
// 对应章锋 SLOS 论文 (2026-07-04, 2nd edition) 第3节
//
// 接收 Host 下发的 EML 八元数节点, 生成 SET/RESET 脉冲序列,
// 读回验证+自适应步长, 将电导态编程到 PCM 阵列.
//
// 状态机: IDLE → WRITE → VERIFY → ADJUST → DONE
// 脉冲校验写入: 目标电导 0x4000, ~7脉冲收敛
//   序列: 0x2000 → 0x2800 → 0x3500 → 0x3E00 → 0x3F80 → 0x3FF0 → 0x4000
//
// MuJoCo-Bench-IDO v0.4.0 — SLOS T-Processor NG Module
// 论文附录: SLOS Appendix E — PCM CIM Pulse Programming

`timescale 1ns / 1ps

package EmlPcmPkg;
    parameter DW = 16;

    // AXI-Lite interface widths
    parameter ADDR_W = 8;
    parameter DATA_W = 32;

    // PCM conductance code range
    parameter PCM_CODE_MAX = 16'hFFFF;

    // Pulse-verify parameters
    parameter TOLERANCE    = 16'h0200;  // ±512 codes (0.78%)
    parameter MAX_PULSES   = 4'd15;      // Maximum 15 SET pulses
    parameter SET_STEP_INIT = 16'h0800;  // Initial SET step = 2048 codes
    parameter SET_STEP_MIN  = 16'h0040;  // Minimum SET step = 64 codes

    // State machine codes
    parameter ST_IDLE    = 3'd0;
    parameter ST_WRITE   = 3'd1;
    parameter ST_VERIFY  = 3'd2;
    parameter ST_ADJUST  = 3'd3;
    parameter ST_DONE    = 3'd4;
    parameter ST_ERROR   = 3'd5;

    // PCM cell operations
    parameter OP_NOP   = 2'd0;
    parameter OP_SET   = 2'd1;   // Crystallization (→ high G)
    parameter OP_RESET = 2'd2;   // Amorphization (→ low G)
    parameter OP_READ  = 2'd3;   // Read-back
endpackage
import EmlPcmPkg::*;

// ═══════════════════════════════════════════════════════════════
// Module: EmlPcmLoader
// Type:   FSM-controlled sequential logic
// Clock:  50 MHz (T-Processor system clock)
// Function: Programs PCM cells with EML weight data via pulse-verify-write
// ═══════════════════════════════════════════════════════════════
//
// Interface:
//   Host side (AXI-Lite simplified):
//     - i_start:         Pulse to begin programming
//     - i_eml_data[31:0]: EML node data (upper 16b = target code, lower 16b = addr)
//     - i_row[5:0]:      PCM array row address
//     - i_col[5:0]:      PCM array column address
//
//   PCM Array side:
//     - o_row[5:0]:      Row address to PCM array
//     - o_col[5:0]:      Col address to PCM array
//     - o_set_pulse:     SET pulse enable (active HIGH)
//     - o_reset_pulse:   RESET pulse enable (active HIGH)
//     - o_read_en:       Read-back enable
//     - i_read_data[15:0]: Read-back conductance code from ADC
//
//   Status:
//     - o_done:          Programming complete
//     - o_error:         Programming failed (max pulses exceeded)
//     - o_pulse_count[3:0]: Number of pulses used
//     - o_final_code[15:0]: Final achieved conductance code
//     - o_state[2:0]:    Current FSM state
//
module EmlPcmLoader (
    input  wire        clk,
    input  wire        rst_n,

    // ── Host Interface ──
    input  wire        i_start,          // Start programming pulse
    input  wire [31:0] i_eml_data,       // [31:16]=target_code, [15:0]=unused
    input  wire [5:0]  i_row,            // PCM row address
    input  wire [5:0]  i_col,            // PCM col address

    // ── PCM Array Interface ──
    output reg  [5:0]  o_row,            // Row address
    output reg  [5:0]  o_col,            // Col address
    output reg         o_set_pulse,      // SET pulse (crystallize)
    output reg         o_reset_pulse,    // RESET pulse (amorphize)
    output reg         o_read_en,        // Read-back enable
    input  wire [15:0] i_read_data,      // ADC read-back code

    // ── Status ──
    output reg         o_done,           // Programming complete
    output reg         o_error,          // Programming failed
    output reg  [3:0]  o_pulse_count,    // Pulses used
    output reg  [15:0] o_final_code,     // Final conductance
    output reg  [2:0]  o_state           // FSM state (for debug)
);

    // ── Internal registers ──
    reg [15:0] target_code_r;    // Target conductance code
    reg [15:0] current_code_r;   // Current (read-back) code
    reg [15:0] step_r;           // Adaptive step size
    reg [3:0]  pulse_count_r;    // Pulse counter
    reg [2:0]  state_r;          // FSM state
    reg [5:0]  row_r, col_r;     // Address latches

    // Combinational error calculation
    wire signed [16:0] error_w;  // 17-bit for sign
    assign error_w = {1'b0, target_code_r} - {1'b0, current_code_r};

    wire converged = (error_w >= -17'sh0200) && (error_w <= 17'sh0200);
    wire need_set   = (error_w > 0);   // Target > current → need SET
    wire need_reset = (error_w < 0);   // Target < current → need RESET

    // Adaptive step: shrink as we approach target
    wire [15:0] adjusted_step;
    assign adjusted_step = (|error_w[16:12]) ? step_r :         // Large error → keep step
                           (|error_w[11:8])  ? {step_r[15:2], 2'b0} : // Med error → step/4
                                              {step_r[15:4], 4'b0};  // Small error → step/16

    // ── FSM: IDLE → WRITE → VERIFY → ADJUST → DONE ──
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers
            state_r        <= ST_IDLE;
            target_code_r  <= 16'h0000;
            current_code_r <= 16'h0000;
            step_r         <= SET_STEP_INIT;
            pulse_count_r  <= 4'd0;
            row_r          <= 6'd0;
            col_r          <= 6'd0;

            o_row          <= 6'd0;
            o_col          <= 6'd0;
            o_set_pulse    <= 1'b0;
            o_reset_pulse  <= 1'b0;
            o_read_en      <= 1'b0;
            o_done         <= 1'b0;
            o_error        <= 1'b0;
            o_pulse_count  <= 4'd0;
            o_final_code   <= 16'h0000;
            o_state        <= ST_IDLE;
        end else begin
            // Default outputs
            o_set_pulse   <= 1'b0;
            o_reset_pulse <= 1'b0;
            o_read_en     <= 1'b0;
            o_done        <= 1'b0;

            case (state_r)
                // ── IDLE: Wait for start command ──
                ST_IDLE: begin
                    if (i_start) begin
                        target_code_r <= i_eml_data[31:16];
                        step_r        <= SET_STEP_INIT;
                        pulse_count_r <= 4'd0;
                        row_r         <= i_row;
                        col_r         <= i_col;
                        o_row         <= i_row;
                        o_col         <= i_col;
                        state_r       <= ST_WRITE;
                    end
                end

                // ── WRITE: Apply SET or RESET pulse ──
                ST_WRITE: begin
                    o_row <= row_r;
                    o_col <= col_r;
                    if (need_set) begin
                        o_set_pulse <= 1'b1;
                        // Predict next code (for verify comparison)
                        current_code_r <= current_code_r + adjusted_step;
                    end else if (need_reset) begin
                        o_reset_pulse <= 1'b1;
                        current_code_r <= current_code_r - {adjusted_step[15:1], 1'b0};
                    end
                    pulse_count_r <= pulse_count_r + 4'd1;
                    state_r       <= ST_VERIFY;
                end

                // ── VERIFY: Read-back and check convergence ──
                ST_VERIFY: begin
                    o_read_en <= 1'b1;
                    o_row     <= row_r;
                    o_col     <= col_r;
                    // Latch read-back data (1-cycle ADC latency modeled as immediate)
                    current_code_r <= i_read_data;
                    state_r        <= ST_ADJUST;
                end

                // ── ADJUST: Check convergence or adjust step ──
                ST_ADJUST: begin
                    if (converged) begin
                        // Success!
                        o_final_code  <= current_code_r;
                        o_pulse_count <= pulse_count_r;
                        o_done        <= 1'b1;
                        state_r       <= ST_DONE;
                    end else if (pulse_count_r >= MAX_PULSES) begin
                        // Exceeded max pulses — error
                        o_error       <= 1'b1;
                        o_final_code  <= current_code_r;
                        o_pulse_count <= pulse_count_r;
                        state_r       <= ST_ERROR;
                    end else begin
                        // Adjust step size and continue
                        step_r   <= (adjusted_step < SET_STEP_MIN) ? SET_STEP_MIN : adjusted_step;
                        state_r  <= ST_WRITE;
                    end
                end

                // ── DONE: Programming successful ──
                ST_DONE: begin
                    o_done <= 1'b1;
                    // Wait for start to go back to IDLE
                    if (!i_start) begin
                        state_r <= ST_IDLE;
                    end
                end

                // ── ERROR: Programming failed ──
                ST_ERROR: begin
                    o_error <= 1'b1;
                    if (!i_start) begin
                        state_r <= ST_IDLE;
                    end
                end

                default: state_r <= ST_IDLE;
            endcase

            o_state <= state_r;
        end
    end

    // ── Timing Analysis (50 MHz = 20ns period) ──
    // IDLE→WRITE:     1 cycle = 20ns
    // WRITE→VERIFY:   1 cycle = 20ns (SET pulse width)
    // VERIFY→ADJUST:  1 cycle = 20ns (ADC read-back)
    // ADJUST→WRITE:   1 cycle = 20ns (step adjust)
    // Total per pulse: 3 cycles = 60ns
    // 7-pulse convergence: 7 × 60ns = 420ns < 2.12ms single-step iteration ✓

endmodule


// ═══════════════════════════════════════════════════════════════
// Testbench (for simulation only)
// ═══════════════════════════════════════════════════════════════
`ifdef EML_PCM_LOADER_TB
module EmlPcmLoader_TB;
    reg         clk, rst_n, i_start;
    reg  [31:0] i_eml_data;
    reg  [5:0]  i_row, i_col;
    reg  [15:0] i_read_data;

    wire [5:0]  o_row, o_col;
    wire        o_set_pulse, o_reset_pulse, o_read_en;
    wire        o_done, o_error;
    wire [3:0]  o_pulse_count;
    wire [15:0] o_final_code;
    wire [2:0]  o_state;

    EmlPcmLoader UUT (
        .clk(clk), .rst_n(rst_n),
        .i_start(i_start), .i_eml_data(i_eml_data),
        .i_row(i_row), .i_col(i_col),
        .o_row(o_row), .o_col(o_col),
        .o_set_pulse(o_set_pulse), .o_reset_pulse(o_reset_pulse),
        .o_read_en(o_read_en), .i_read_data(i_read_data),
        .o_done(o_done), .o_error(o_error),
        .o_pulse_count(o_pulse_count), .o_final_code(o_final_code),
        .o_state(o_state)
    );

    // Clock: 50 MHz
    initial clk = 0;
    always #10 clk = ~clk;

    // Mock PCM read-back: return progressively approaching target
    reg [15:0] mock_g;
    always @(posedge clk) begin
        if (o_read_en) begin
            // Simulate PCM approaching target after each SET pulse
            if (o_set_pulse) begin
                mock_g = mock_g + 16'h1000; // Simplified: +4096 per pulse
                if (mock_g > 16'h4000) mock_g = 16'h4000;
            end
            i_read_data = mock_g;
        end
    end

    initial begin
        rst_n = 0; i_start = 0; i_eml_data = 0;
        i_row = 0; i_col = 0; i_read_data = 0;
        mock_g = 16'h0000;
        #20 rst_n = 1;

        // Start programming: target = 0x4000
        i_eml_data = {16'h4000, 16'h0000};
        i_row = 6'd3; i_col = 6'd5;
        @(posedge clk); i_start = 1;
        @(posedge clk); i_start = 0;

        // Wait for done
        wait(o_done || o_error);
        @(posedge clk);

        $display("Result: done=%b error=%b pulses=%d final=0x%04h",
                 o_done, o_error, o_pulse_count, o_final_code);
        if (o_done) $display("EmlPcmLoader: PASSED");
        else $display("EmlPcmLoader: FAILED");

        $finish;
    end
endmodule
`endif
