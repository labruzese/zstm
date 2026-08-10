//! A port of RSTM v7's `bench/bmharness.cpp` to Zig, driving zstm instead of
//! libstm.

const std = @import("std");
const builtin = @import("builtin");
const zstm = @import("zstm");
const linux = std.os.linux;

/// Mirrors RSTM's `struct Config` / global `CFG` (bench/bmconfig.hpp).
pub const Config = struct {
    bmname: []const u8 = "",
    duration: u32 = 1,
    execute: u32 = 0,
    threads: u32 = 1,
    nops_after_tx: u32 = 0,
    elements: u32 = 256,
    lookpct: u32 = 34,
    inspct: u32 = 66,
    sets: u32 = 1,
    ops: u32 = 1,

    // updated during the run
    time: u64 = 0,
    running: std.atomic.Value(bool) = .init(true),
    txcount: std.atomic.Value(u32) = .init(0),
    aborts: std.atomic.Value(u64) = .init(0),

    /// Selected via `--mode`; controls zstm's publication-safety mode.
    mode: zstm.Tx.Mode = .ala,
};

pub var cfg: Config = .{};

/// RSTM's `getElapsedTime()` (include/common/platform.hpp:405): CLOCK_REALTIME
/// in nanoseconds. Matched exactly so the `time=` field means the same thing on
/// both sides.
pub fn getElapsedTime() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.REALTIME, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Verbatim port of `rand_r_32` (rstm/alt-license/rand_r_32.h).
pub fn randR32(seed: *u32) u32 {
    var next: u32 = seed.*;
    var result: u32 = undefined;

    next = next *% 1103515245 +% 12345;
    result = (next / 65536) % 2048;

    next = next *% 1103515245 +% 12345;
    result <<= 10;
    result ^= (next / 65536) % 1024;

    next = next *% 1103515245 +% 12345;
    result <<= 10;
    result ^= (next / 65536) % 1024;

    seed.* = next;
    return result;
}

/// RSTM's `spin64()` (include/common/locks.hpp:36).
fn spin64() void {
    var i: u32 = 0;
    while (i < 64) : (i += 1) asm volatile ("nop");
}

fn nontxnwork() void {
    if (cfg.nops_after_tx != 0) {
        var i: u32 = 0;
        while (i < cfg.nops_after_tx) : (i += 1) spin64();
    }
}

/// RSTM's lightweight counting barriers (`barrier(uint32_t which)`).
var barriers: [16]std.atomic.Value(u32) = [_]std.atomic.Value(u32){.init(0)} ** 16;

fn barrier(which: usize) void {
    _ = barriers[which].fetchAdd(1, .acq_rel);
    while (barriers[which].load(.acquire) != cfg.threads) {
        std.atomic.spinLoopHint();
    }
}

/// Run one transaction to completion, counting aborts. See note (2) at the top
/// of this file: this is `zstm.Tx.run` with a retry counter threaded through.
pub fn runTx(tx: *zstm.Tx, comptime body: anytype, args: anytype, aborts: *u64) void {
    retry: while (true) {
        tx.txBegin();

        if (@call(.auto, body, .{tx} ++ args)) |_| {
            if (tx.txCommit()) |_| {
                tx.reset();
                return;
            } else |err| {
                tx.reset();
                if (err == error.TxRetry) {
                    aborts.* += 1;
                    continue :retry;
                }
                std.debug.panic("transaction commit failed: {}", .{err});
            }
        } else |err| {
            tx.reset();
            if (err == error.TxRetry) {
                aborts.* += 1;
                continue :retry;
            }
            std.debug.panic("transaction body failed: {}", .{err});
        }
    }
}

fn timerThread(seconds: u32) void {
    const req: linux.timespec = .{ .sec = @intCast(seconds), .nsec = 0 };
    _ = linux.nanosleep(&req, null);
    cfg.running.store(false, .release);
}

/// A workload must expose:
///   pub fn init(gpa: Allocator) !void      -- build the shared data
///   pub fn reparse() void                  -- fill in a default bmname
///   pub fn test_(tx, id, seed, aborts) void -- run exactly one transaction
///   pub fn verify() bool                   -- check invariants after the run
///   pub const max_reads / max_writes: usize -- log capacity to pre-reserve
pub fn Runner(comptime Bench: type) type {
    return struct {
        const Self = @This();

        fn worker(id: usize) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const alloc = arena.allocator();

            var tx: zstm.Tx = .init(alloc, &stm, cfg.mode);
            defer tx.deinit();

            tx.reads.ensureTotalCapacity(alloc, Bench.maxReads()) catch @panic("oom");
            tx.writes.ensureCapacity(alloc, @intCast(Bench.maxWrites())) catch @panic("oom");

            var aborts: u64 = 0;
            var count: u32 = 0;
            var seed: u32 = @intCast(id);

            barrier(0);
            if (id == 0) {
                if (cfg.execute == 0) {
                    timer_thread = std.Thread.spawn(.{}, timerThread, .{cfg.duration}) catch
                        @panic("failed to spawn timer thread");
                }
                cfg.time = getElapsedTime();
            }

            barrier(1);

            if (cfg.execute == 0) {
                while (cfg.running.load(.acquire)) {
                    Bench.test_(&tx, id, &seed, &aborts);
                    count += 1;
                    nontxnwork();
                }
            } else {
                var e: u32 = 0;
                while (e < cfg.execute) : (e += 1) {
                    Bench.test_(&tx, id, &seed, &aborts);
                    count += 1;
                    nontxnwork();
                }
            }

            barrier(2);
            if (id == 0) cfg.time = getElapsedTime() - cfg.time;

            _ = cfg.txcount.fetchAdd(count, .acq_rel);
            _ = cfg.aborts.fetchAdd(aborts, .acq_rel);
        }

        pub fn main(gpa: std.mem.Allocator, out: *std.Io.Writer) !void {
            Bench.reparse();
            try Bench.init(gpa);

            var threads: [256]std.Thread = undefined;
            var j: usize = 1;
            while (j < cfg.threads) : (j += 1) {
                threads[j] = try std.Thread.spawn(.{}, worker, .{j});
            }

            worker(0);

            var k: usize = 1;
            while (k < cfg.threads) : (k += 1) threads[k].join();
            if (timer_thread) |t| t.join();

            const ok = Bench.verify();
            try out.print("Verification: {s}\n", .{if (ok) "Passed" else "Failed"});

            const txns = cfg.txcount.load(.acquire);
            const alg = switch (cfg.mode) {
                .ala => "zstm-ala",
                .sla => "zstm-sla",
            };
            try out.print(
                "csv, ALG={s}, B={s}, R={d}, d={d}, p={d}, X={d}, m={d}, S={d}, O={d}, txns={d}, time={d}, throughput={d}\n",
                .{
                    alg,       cfg.bmname,   cfg.lookpct, cfg.duration,
                    cfg.threads, cfg.execute, cfg.elements, cfg.sets,
                    cfg.ops,   txns,         cfg.time,
                    (1_000_000_000 * @as(u64, txns)) / cfg.time,
                },
            );
            try out.print("aborts, {d}\n", .{cfg.aborts.load(.acquire)});
            try out.flush();

            if (!ok) std.process.exit(1);
        }
    };
}

/// The shared STM instance. One per process, as zstm intends.
pub var stm: zstm.Stm = .init;

var timer_thread: ?std.Thread = null;
