//! A port of RSTM v7's `bench/bmharness.cpp` to Zig, driving zstm instead of
//! libstm.
//!
//! What lives here is the part of the RSTM harness the *workloads* talk to:
//! the shared config, the shared `Stm`, the retry loop, and the RNG. Thread
//! orchestration, timing and statistics live in `main.zig`.
//!
//! A workload must expose:
//!   pub fn init(gpa: Allocator) !void       -- build the shared data
//!   pub fn reparse() void                   -- fill in a default bmname
//!   pub fn test_(tx, id, seed, aborts) void -- run exactly one transaction
//!   pub fn verify() bool                    -- check invariants after the run
//!   pub fn maxReads() usize                 -- read-log capacity to pre-reserve
//!   pub fn maxWrites() usize                -- write-set capacity to pre-reserve

const std = @import("std");
const zstm = @import("zstm");
const linux = std.os.linux;

/// `workloads.zig` carries a vestigial `harness: *harness.Harness` field on each
/// workload. This alias keeps that spelling valid, and makes `Harness.cfg` and
/// `cfg` the same variable.
pub const Harness = @This();

/// Mirrors RSTM's `struct Config` / global `CFG` (bench/bmconfig.hpp).
pub const Config = struct {
    bmname: []const u8 = "",
    duration: u32 = 5,
    execute: u32 = 0,
    threads: u32 = 1,
    nops_after_tx: u32 = 0,
    elements: u32 = 256,
    lookpct: u32 = 34,
    inspct: u32 = 66,
    sets: u32 = 1,
    ops: u32 = 1,

    // RSTM parity only: `main.zig` runs its own clock and its own stop flag.
    time: u64 = 0,
    running: std.atomic.Value(bool) = .init(true),

    // Totals for the run just finished. `main.zig` fills these in before it
    // calls a workload's `verify`, which is the only thing that reads them.
    txcount: std.atomic.Value(u32) = .init(0),
    aborts: std.atomic.Value(u64) = .init(0),
};

pub var cfg: Config = .{};

/// The shared STM instance. One per process, as zstm intends.
pub var stm: zstm.Stm = .init;

/// RSTM's `nontxnwork`: non-transactional filler between transactions.
pub fn nontxnwork() void {
    if (cfg.nops_after_tx != 0) {
        var i: u32 = 0;
        while (i < cfg.nops_after_tx) : (i += 1) spin64();
    }
}

/// Run one transaction to completion, counting aborts. This is `zstm.Tx.run`
/// with a retry counter threaded through.
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

pub fn getElapsedTime() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.REALTIME, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

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

fn spin64() void {
    var i: u32 = 0;
    while (i < 64) : (i += 1) asm volatile ("nop");
}
