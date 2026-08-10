//! Zig ports of four RSTM v7 microbenchmarks, written against zstm.
//!
//! Each is a line-for-line translation of its C++ counterpart in
//! rstm/bench/. Where the original does something surprising, the surprise is
//! reproduced rather than corrected -- the goal is for both implementations to
//! execute the same workload, not for the workload to be good.

const std = @import("std");
const zstm = @import("zstm");
const harness = @import("harness.zig");

const cfg = &harness.cfg;
const TxWord = zstm.TxWord;
const Word = zstm.Word;

// ---------------------------------------------------------------------------
// CounterBench -- rstm/bench/CounterBench.cpp
//
// One shared word, incremented by every transaction. Maximum contention: every
// pair of concurrent transactions conflicts, so this measures commit-path cost
// and abort behavior rather than throughput of useful work.
// ---------------------------------------------------------------------------
pub const Counter = struct {
    pub fn maxReads() usize {
        return 1;
    }
    pub fn maxWrites() usize {
        return 1;
    }

    var counter: TxWord = .init(0);

    pub fn reparse() void {
        cfg.bmname = "Counter";
    }

    pub fn init(gpa: std.mem.Allocator) !void {
        _ = gpa;
        counter = .init(0);
    }

    fn body(t: *zstm.Tx, addr: *TxWord) zstm.Error!void {
        try t.write(addr, 1 + try t.read(addr));
    }

    pub fn test_(tx: *zstm.Tx, id: usize, seed: *u32, aborts: *u64) void {
        _ = id;
        _ = seed;
        harness.runTx(tx, body, .{&counter}, aborts);
    }

    pub fn verify() bool {
        // Stronger than RSTM's `counter > 0`: with no lost updates the counter
        // must equal the total number of committed transactions, and each
        // writer commit advances the sequence lock by exactly 2.
        const final = counter.unsafeLoad();
        const txns = cfg.txcount.load(.acquire);
        const seq = harness.stm.seq_lock.load(.acquire);
        if (final != txns) {
            std.debug.print("(final value = {d}, expected {d}) ", .{ final, txns });
            return false;
        }
        if (seq != 2 * @as(u64, txns)) {
            std.debug.print("(seq_lock = {d}, expected {d}) ", .{ seq, 2 * @as(u64, txns) });
            return false;
        }
        std.debug.print("(final value = {d}) ", .{final});
        return true;
    }
};

// ---------------------------------------------------------------------------
// ReadNWrite1Bench -- rstm/bench/ReadNWrite1Bench.cpp
//
// `-O` random reads from an `-m`-element array, then a single write to the last
// location read. Read-mostly: the read log grows to O entries while the write
// set holds exactly one.
// ---------------------------------------------------------------------------
pub const ReadNWrite1 = struct {
    pub fn maxReads() usize {
        return cfg.ops;
    }
    pub fn maxWrites() usize {
        return 1;
    }

    var matrix: []TxWord = &.{};

    pub fn reparse() void {
        if (cfg.bmname.len == 0) cfg.bmname = "ReadNWrite1";
    }

    pub fn init(gpa: std.mem.Allocator) !void {
        matrix = try gpa.alloc(TxWord, cfg.elements);
        // match ReadNWrite1Bench.cpp's bench_init exactly
        var s: u32 = 1;
        for (matrix) |*w| w.* = .init(harness.randR32(&s));
    }

    fn body(t: *zstm.Tx, seed: *u32) zstm.Error!void {
        var sum: Word = 0;
        var loc: usize = 0;
        var i: u32 = 0;
        while (i < cfg.ops) : (i += 1) {
            loc = harness.randR32(seed) % cfg.elements;
            sum +%= try t.read(&matrix[loc]);
        }
        try t.write(&matrix[loc], sum);
    }

    pub fn test_(tx: *zstm.Tx, id: usize, seed: *u32, aborts: *u64) void {
        _ = id;
        // The C++ version copies the seed to a local so a longjmp-driven abort
        // restores it. zstm aborts by returning an error, so the seed is
        // naturally restored by re-reading it -- except that our body advances
        // it in place. Snapshot and restore explicitly to match: every attempt
        // of a given transaction must touch the same locations.
        const saved = seed.*;
        var local = saved;
        harness.runTx(tx, body, .{&local}, aborts);
        seed.* = local;
    }

    pub fn verify() bool {
        return true;
    }
};

// ---------------------------------------------------------------------------
// ReadWriteNBench -- rstm/bench/ReadWriteNBench.cpp
//
// `-O` random reads, then `-O` writes back to those same slots. Write-heavy:
// exercises the write-set insert path and the commit-time writeback loop.
// ---------------------------------------------------------------------------
pub const ReadWriteN = struct {
    pub fn maxReads() usize {
        return cfg.ops;
    }
    pub fn maxWrites() usize {
        return cfg.ops;
    }

    var matrix: []TxWord = &.{};

    pub fn reparse() void {
        if (cfg.bmname.len == 0) cfg.bmname = "ReadWriteN";
    }

    pub fn init(gpa: std.mem.Allocator) !void {
        matrix = try gpa.alloc(TxWord, cfg.elements);
        // match ReadWriteNBench.cpp's bench_init
        var s: u32 = 1;
        for (matrix) |*w| w.* = .init(harness.randR32(&s));
    }

    fn body(t: *zstm.Tx, seed: *u32) zstm.Error!void {
        // The C++ original uses fixed 1024-element stack arrays, which caps -O
        // at 1024; we keep the same limit for the same reason.
        var snapshot: [1024]Word = undefined;
        var loc: [1024]usize = undefined;

        var i: u32 = 0;
        while (i < cfg.ops) : (i += 1) {
            loc[i] = harness.randR32(seed) % cfg.elements;
            snapshot[i] = try t.read(&matrix[loc[i]]);
        }
        i = 0;
        while (i < cfg.ops) : (i += 1) {
            try t.write(&matrix[loc[i]], 1 +% snapshot[i]);
        }
    }

    pub fn test_(tx: *zstm.Tx, id: usize, seed: *u32, aborts: *u64) void {
        _ = id;
        const saved = seed.*;
        var local = saved;
        harness.runTx(tx, body, .{&local}, aborts);
        seed.* = local;
    }

    pub fn verify() bool {
        return true;
    }
};

// ---------------------------------------------------------------------------
// DisjointBench -- rstm/bench/DisjointBench.cpp + rstm/bench/Disjoint.hpp
//
// Every thread works in its own cache-line-padded 1009-entry buffer, so
// transactions never conflict. This is the cleanest measurement in the suite:
// with aborts out of the picture, any difference is pure instrumentation
// overhead -- the read barrier, the write barrier, and the commit path.
//
// bmname encodes the parameters as <prefix>-<L>-<R>-<W>:
//   prefix  PrDw = private read buffer, SrDw = shared read buffer
//   L       locations touched per transaction
//   R, W    reads and writes per ten operations
// ---------------------------------------------------------------------------
pub const Disjoint = struct {
    pub fn maxReads() usize {
        return locations_per_transaction;
    }
    pub fn maxWrites() usize {
        return locations_per_transaction;
    }

    const DJBUFFER_SIZE: usize = 1009;
    const BUFFER_COUNT: usize = 256;

    const PaddedBufferEntry = extern struct {
        value: TxWord,
        padding: [64 - @sizeOf(TxWord)]u8,
    };

    const PaddedBuffer = extern struct {
        buffer: [DJBUFFER_SIZE]PaddedBufferEntry,
    };

    var private_buffers: []PaddedBuffer = &.{};
    var public_buffer: *PaddedBuffer = undefined;

    var reads_per_ten: u32 = 0;
    var writes_per_ten: u32 = 0;
    var locations_per_transaction: u32 = 0;
    var use_shared_read_buffer: bool = false;

    pub fn reparse() void {
        if (cfg.bmname.len == 0) cfg.bmname = "DrDw";
    }

    pub fn init(gpa: std.mem.Allocator) !void {
        // Same field order as the C++ `bench_init`: size, read, write.
        var it = std.mem.splitScalar(u8, cfg.bmname, '-');
        _ = it.next(); // prefix
        const size_s = it.next() orelse return error.BadBenchName;
        const read_s = it.next() orelse return error.BadBenchName;
        const write_s = it.next() orelse return error.BadBenchName;

        locations_per_transaction = try std.fmt.parseInt(u32, size_s, 10);
        reads_per_ten = try std.fmt.parseInt(u32, read_s, 10);
        writes_per_ten = try std.fmt.parseInt(u32, write_s, 10);
        use_shared_read_buffer = std.mem.startsWith(u8, cfg.bmname, "SrDw");

        private_buffers = try gpa.alloc(PaddedBuffer, BUFFER_COUNT);
        public_buffer = try gpa.create(PaddedBuffer);

        var s: u32 = writes_per_ten;
        for (private_buffers) |*buf| {
            for (&buf.buffer) |*e| e.value = .init(harness.randR32(&s));
        }
        for (&public_buffer.buffer) |*e| e.value = .init(harness.randR32(&s));
    }

    fn roTransaction(t: *zstm.Tx, id: usize, startpoint: usize) zstm.Error!void {
        const r_buffer = if (use_shared_read_buffer) public_buffer else &private_buffers[id];
        var sum: Word = 0;
        var index = startpoint;
        var i: u32 = 0;
        while (i < locations_per_transaction) : (i += 1) {
            sum +%= try t.read(&r_buffer.buffer[index].value);
            index = (index + 1) % DJBUFFER_SIZE;
        }
        std.mem.doNotOptimizeAway(sum);
    }

    fn rRwTransaction(t: *zstm.Tx, id: usize, startpoint: usize) zstm.Error!void {
        const r_buffer = if (use_shared_read_buffer) public_buffer else &private_buffers[id];
        const w_buffer = &private_buffers[id];

        var index = startpoint;
        var writes: u32 = 0;
        var reads: u32 = 0;
        var buff: Word = 0;

        var i: u32 = 0;
        while (i < locations_per_transaction) : (i += 1) {
            if ((writes + reads) == 10) {
                writes = 0;
                reads = 0;
            }

            const should_write = if (i & 0x1 != 0)
                writes < writes_per_ten
            else
                !(reads < reads_per_ten);

            if (should_write) {
                const oldval = try t.read(&w_buffer.buffer[index].value);
                try t.write(&w_buffer.buffer[index].value, oldval +% 1);
                writes += 1;
            } else {
                buff +%= try t.read(&r_buffer.buffer[index].value);
                reads += 1;
            }

            index = (index + 1) % DJBUFFER_SIZE;
        }
        std.mem.doNotOptimizeAway(buff);
    }

    fn body(t: *zstm.Tx, id: usize, act: u32, start: usize) zstm.Error!void {
        if (act < cfg.lookpct) {
            try roTransaction(t, id, start);
        } else {
            try rRwTransaction(t, id, start);
        }
    }

    pub fn test_(tx: *zstm.Tx, id: usize, seed: *u32, aborts: *u64) void {
        const act = harness.randR32(seed) % 100;
        const start = harness.randR32(seed) % DJBUFFER_SIZE;
        harness.runTx(tx, body, .{ id, act, start }, aborts);
    }

    pub fn verify() bool {
        return true;
    }
};
