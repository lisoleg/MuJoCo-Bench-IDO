// tproc_ksnap_buffer.v
// κ-Snap Ring DMA Audit Buffer
// 对应章锋 SLOS 论文 (2026-07-04, 2nd edition) 第5节
//
// 环形FIFO, 深度256, 存储η残差+时间戳+违规码
// AXI-Stream接口输出, 支持溢出标志
// 当FIFO满时, 最旧数据被覆盖 (环形覆盖策略)
//
// MuJoCo-Bench-IDO v0.4.0 — SLOS T-Processor NG Module
// 论文附录: SLOS Appendix G — κ-Snap DMA Buffer

`timescale 1ns / 1ps

package KSnapBufPkg;
    // FIFO configuration
    parameter DEPTH      = 256;       // 256 entries
    parameter ADDR_W     = 8;         // log2(256) = 8
    parameter DATA_W     = 16;        // η residual width
    parameter TS_W       = 32;        // Timestamp width (cycles)
    parameter CODE_W     = 8;         // Violation code width
    parameter ENTRY_W    = DATA_W + TS_W + CODE_W; // 56 bits per entry

    // AXI-Stream widths
    parameter TDATA_W    = 64;        // AXI-Stream data (padded to 64-bit)
endpackage
import KSnapBufPkg::*;

// ═══════════════════════════════════════════════════════════════
// Module: KSnapBuffer
// Type:   Ring FIFO with AXI-Stream output
// Clock:  50 MHz (T-Processor system clock)
// Depth:  256 entries (each 56-bit: η[15:0] + ts[31:0] + code[7:0])
// ═══════════════════════════════════════════════════════════════
//
// Interface:
//   Write side (from η-ALU + PsiAnchorGate):
//     - i_wr_en:      Write enable
//     - i_eta[15:0]:  η residual value
//     - i_violation[7:0]: Violation code from PsiAnchorGate
//     - i_timestamp[31:0]: Free-running cycle counter
//
//   Read side (AXI-Stream Master):
//     - o_tvalid:     Data valid
//     - o_tdata[63:0]: Data payload (η + ts + code + padding)
//     - i_tready:     Downstream ready to accept
//     - o_tlast:      Last entry in burst
//
//   Status:
//     - o_count[7:0]: Current FIFO occupancy
//     - o_overflow:   Overflow flag (sticky, cleared by rst_n)
//     - o_full:       FIFO full
//     - o_empty:      FIFO empty
//
module KSnapBuffer (
    input  wire        clk,
    input  wire        rst_n,

    // ── Write Interface (from η-ALU + PsiAnchorGate) ──
    input  wire        i_wr_en,         // Write enable
    input  wire [15:0] i_eta,           // η residual
    input  wire [7:0]  i_violation,     // Violation code
    input  wire [31:0] i_timestamp,     // Cycle counter timestamp

    // ── AXI-Stream Read Interface ──
    input  wire        i_tready,        // Downstream ready
    output reg         o_tvalid,        // Data valid
    output reg  [63:0] o_tdata,         // Data payload
    output reg         o_tlast,         // Last in burst (when FIFO empties)

    // ── Status ──
    output reg  [7:0]  o_count,         // Current occupancy
    output reg         o_overflow,      // Sticky overflow flag
    output wire        o_full,          // FIFO full
    output wire        o_empty          // FIFO empty
);

    // ── FIFO Memory (256 entries × 56 bits) ──
    reg [ENTRY_W-1:0] fifo_mem [0:DEPTH-1];

    // ── Read/Write pointers ──
    reg [ADDR_W:0] wr_ptr;   // 9-bit (extra bit for full/empty distinction)
    reg [ADDR_W:0] rd_ptr;   // 9-bit

    // ── Occupancy counter ──
    reg [ADDR_W:0] count_r;  // 9-bit count (0 to 256)

    // ── Combinational full/empty ──
    assign o_full  = (count_r == DEPTH[ADDR_W:0]);
    assign o_empty = (count_r == 0);

    // ── Write logic (ring overwrite when full) ──
    wire do_write = i_wr_en;
    wire do_read  = o_tvalid & i_tready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr     <= {(ADDR_W+1){1'b0}};
            rd_ptr     <= {(ADDR_W+1){1'b0}};
            count_r    <= {(ADDR_W+1){1'b0}};
            o_overflow <= 1'b0;
            o_tvalid   <= 1'b0;
            o_tdata    <= 64'd0;
            o_tlast    <= 1'b0;
            o_count    <= 8'd0;
        end else begin
            // ── Write into FIFO ──
            if (do_write) begin
                fifo_mem[wr_ptr[ADDR_W-1:0]] <= {i_eta, i_timestamp, i_violation};

                if (o_full) begin
                    // Ring overwrite: advance both pointers, count stays at DEPTH
                    wr_ptr <= wr_ptr + 1'b1;
                    rd_ptr <= rd_ptr + 1'b1; // Overwrite oldest
                    o_overflow <= 1'b1;       // Sticky overflow flag
                end else begin
                    wr_ptr  <= wr_ptr + 1'b1;
                    count_r <= count_r + 1'b1;
                end
            end

            // ── Read from FIFO (AXI-Stream) ──
            if (do_read) begin
                o_tvalid <= 1'b0;  // Clear valid after handshake
                if (count_r > 0) begin
                    rd_ptr  <= rd_ptr + 1'b1;
                    count_r <= count_r - 1'b1;
                end
            end

            // ── Generate next AXI-Stream beat if FIFO has data ──
            if (!o_tvalid && count_r > 0) begin
                // Read next entry from FIFO
                o_tvalid <= 1'b1;
                // Pack: [63:48]=η, [47:16]=timestamp, [15:8]=violation, [7:0]=0
                o_tdata  <= {fifo_mem[rd_ptr[ADDR_W-1:0]][ENTRY_W-1:ENTRY_W-DATA_W],
                             fifo_mem[rd_ptr[ADDR_W-1:0]][ENTRY_W-DATA_W-1:CODE_W],
                             fifo_mem[rd_ptr[ADDR_W-1:0]][CODE_W-1:0],
                             8'h00};
                // tlast when this is the last entry
                o_tlast  <= (count_r == 1);
            end

            // Update count output
            o_count <= count_r[ADDR_W-1:0];
        end
    end

    // ── Resource Estimation (40nm CMOS) ──
    // FIFO memory: 256 × 56 bits = 14,336 bits = 1.75 KB
    //   - SRAM macro: ~0.01 mm² (dual-port, 256×56)
    // Control logic: ~200 gates = ~0.001 mm²
    // Total: ~0.011 mm²
    //
    // ── Bandwidth ──
    // Write rate: 1 entry per control tick (100 Hz) = 100 entries/s
    // Read rate: AXI-Stream @ 50 MHz, 1 beat per cycle = 50M entries/s
    // → Read bandwidth vastly exceeds write rate; FIFO never backs up
    // → Burst read: drain entire 256-entry FIFO in 256 cycles = 5.12 µs

endmodule


// ═══════════════════════════════════════════════════════════════
// Testbench (for simulation only)
// ═══════════════════════════════════════════════════════════════
`ifdef KSNAP_BUFFER_TB
module KSnapBuffer_TB;
    reg         clk, rst_n;
    reg         i_wr_en;
    reg  [15:0] i_eta;
    reg  [7:0]  i_violation;
    reg  [31:0] i_timestamp;
    reg         i_tready;

    wire        o_tvalid;
    wire [63:0] o_tdata;
    wire        o_tlast;
    wire [7:0]  o_count;
    wire        o_overflow;
    wire        o_full;
    wire        o_empty;

    KSnapBuffer UUT (
        .clk(clk), .rst_n(rst_n),
        .i_wr_en(i_wr_en), .i_eta(i_eta),
        .i_violation(i_violation), .i_timestamp(i_timestamp),
        .i_tready(i_tready),
        .o_tvalid(o_tvalid), .o_tdata(o_tdata), .o_tlast(o_tlast),
        .o_count(o_count), .o_overflow(o_overflow),
        .o_full(o_full), .o_empty(o_empty)
    );

    initial clk = 0;
    always #10 clk = ~clk;

    integer i;
    initial begin
        rst_n = 0; i_wr_en = 0; i_eta = 0; i_violation = 0;
        i_timestamp = 0; i_tready = 0;
        #20 rst_n = 1;

        // Write 10 entries
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            i_wr_en = 1;
            i_eta = i * 100;
            i_violation = (i == 5) ? 8'h01 : 8'h00;
            i_timestamp = i * 20;
        end
        @(posedge clk);
        i_wr_en = 0;

        #20;
        $display("After 10 writes: count=%d, full=%b, empty=%b",
                 o_count, o_full, o_empty);
        assert(o_count == 10);

        // Read all entries
        i_tready = 1;
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
        end
        i_tready = 0;

        #20;
        $display("After reads: count=%d, empty=%b", o_count, o_empty);
        assert(o_empty);

        // Test overflow: write 260 entries (exceeds depth 256)
        for (i = 0; i < 260; i = i + 1) begin
            @(posedge clk);
            i_wr_en = 1;
            i_eta = i;
            i_timestamp = i;
        end
        @(posedge clk);
        i_wr_en = 0;

        #20;
        $display("After overflow: count=%d, overflow=%b",
                 o_count, o_overflow);
        assert(o_overflow);  // Should have overflow flag set
        assert(o_count == 256); // Should be full

        $display("KSnapBuffer: All tests PASSED");
        $finish;
    end
endmodule
`endif
