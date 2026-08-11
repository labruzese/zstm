//! Benchmark driver for zstm.
//!
//! ## Quick start
//!
//! ```
//! zig build bench                                                 # default sweep, table on stdout
//! zig build bench -- --list                                       # what is registered
//! zig build bench -- --only counter --threads 1,2,4 --trials 9
//! zig build bench -- --duration 300 --trials 3                    # fast smoke pass
//! ```
//!
//! ## A/B-ing a change to the library
//!
//! ```
//! zig build bench -- --save before.csv        # record the baseline
//! $EDITOR src/root.zig                        # ...change something...
//! zig build bench -- --baseline before.csv    # rerun + diff against it
//! ```
//!
//! The baseline diff prints throughput, latency, abort-rate and per-transaction
//! cost side by side, and flags each delta as significant only when it exceeds
//! the run-to-run noise measured across trials. That noise number (`cv%`) is
//! always on screen: if it is large, do not believe small deltas.
//!
//! ## Adding a benchmark
//!
//! 1. Write the workload in `workloads.zig`. It must expose
//!    `init`, `reparse`, `test_`, `verify`, `maxReads`, `maxWrites`
//!    (see `Harness.Runner`).
//! 2. Append one entry to `suite` below. That is the entire registration step.
//!
//! ## Skipping a benchmark
//!
//! Set `.enabled = false` on its entry (it still shows up in `--list`), or pass
//! `--skip <substring>` / `--only <substring>` on the command line.

const std = @import("std");
const builtin = @import("builtin");
const zstm = @import("zstm");
const harness = @import("harness.zig");
const workloads = @import("workloads.zig");

const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const cache_line = std.atomic.cache_line;

// registry

/// knobs the workloads read out of `harness.cfg`. Anything left unset keeps the
/// default shown here
pub const Params = struct {
    /// `Disjoint` parses this as `<PrDw|SrDw>-<locations>-<reads/10>-<writes/10>`
    /// and *requires* all four fields. Other workloads only use it as a label.
    bmname: []const u8 = "",
    /// size of the shared array (`ReadNWrite1`, `ReadWriteN`). smaller array =
    /// more conflicts between threads.
    elements: u32 = 256,
    /// transactional operations per transaction (`ReadNWrite1`, `ReadWriteN`).
    ops: u32 = 1,
    /// percent of `Disjoint` transactions that are read-only.
    lookpct: u32 = 34,
    inspct: u32 = 66,
    sets: u32 = 1,
    /// non-transactional spinning between transactions, in 64-nop blocks. Use
    /// this to model "real work" outside the critical section.
    nops_after_tx: u32 = 0,
};

/// hard limits baked into a workload's implementation. exceeding one is a
/// configuration bug, so the driver refuses to run rather than corrupting memory.
pub const Limits = struct {
    /// `Disjoint` indexes a fixed array of per-thread buffers.
    max_threads: ?u32 = null,
    /// `ReadWriteN` snapshots into fixed 1024-element stack arrays.
    max_ops: ?u32 = null,
};

pub const Bench = struct {
    /// used for `--only`/`--skip` matching and as the row key in saved results.
    name: []const u8,
    /// the workload type from `workloads.zig`.
    workload: type,
    /// one line shown above the table. Say what the benchmark is *for*.
    desc: []const u8 = "",
    /// set to false to keep the entry registered but out of the default run.
    enabled: bool = true,
    /// thread counts to sweep. `null` means "use the default sweep", which is
    /// powers of two up to the CPU count. `--threads` overrides both.
    threads: ?[]const u32 = null,
    params: Params = .{},
    limits: Limits = .{},
};

/// Every benchmark the driver knows about, in report order.
const suite = [_]Bench{
    .{
        .name = "counter",
        .workload = workloads.Counter,
        .desc = "one shared word; every transaction conflicts with every other",
        .params = .{ .ops = 1 },
    },
    .{
        .name = "rn-w1-hot",
        .workload = workloads.ReadNWrite1,
        .desc = "16 reads + 1 write over 256 slots; read-mostly, high conflict",
        .params = .{ .elements = 256, .ops = 16 },
    },
    .{
        .name = "rn-w1-cold",
        .workload = workloads.ReadNWrite1,
        .desc = "16 reads + 1 write over 64K slots; read-mostly, low conflict",
        .params = .{ .elements = 1 << 16, .ops = 16 },
    },
    .{
        // WriteSet.linear_max == cache_line/@sizeOf(Entry) == 4 entries, so a
        // 4-write transaction never builds the hash index...
        .name = "rw-n-linear",
        .workload = workloads.ReadWriteN,
        .desc = "4 reads + 4 writes; write set stays on the linear scan path",
        .params = .{ .elements = 1 << 16, .ops = 4 },
        .limits = .{ .max_ops = 1024 },
    },
    .{
        // ...and an 8-write transaction does, on every single transaction.
        // The pair isolates the cost of the index build.
        .name = "rw-n-indexed",
        .workload = workloads.ReadWriteN,
        .desc = "8 reads + 8 writes; write set crosses into the hashed path",
        .params = .{ .elements = 1 << 16, .ops = 8 },
        .limits = .{ .max_ops = 1024 },
    },
    .{
        .name = "rw-n-large",
        .workload = workloads.ReadWriteN,
        .desc = "256 reads + 256 writes; write-set growth, reindexing, writeback",
        .params = .{ .elements = 1 << 16, .ops = 256 },
        .limits = .{ .max_ops = 1024 },
    },
    .{
        .name = "disjoint-ro",
        .workload = workloads.Disjoint,
        .desc = "64 private reads, zero writes: pure read-barrier overhead",
        .params = .{ .bmname = "PrDw-64-10-0", .lookpct = 100 },
        .limits = .{ .max_threads = 256 },
    },
    .{
        .name = "disjoint-rw",
        .workload = workloads.Disjoint,
        .desc = "64 private locations, half written: only the sequence lock is shared",
        .params = .{ .bmname = "PrDw-64-5-5", .lookpct = 0 },
        .limits = .{ .max_threads = 256 },
    },
    .{
        .name = "disjoint-shared",
        .workload = workloads.Disjoint,
        .desc = "reads one buffer shared by all threads, writes private: shared reads",
        .params = .{ .bmname = "SrDw-64-8-2", .lookpct = 34 },
        .limits = .{ .max_threads = 256 },
    },
    .{
        .name = "counter-spaced",
        .workload = workloads.Counter,
        .desc = "counter plus non-transactional work between transactions",
        .enabled = false,
        .params = .{ .nops_after_tx = 20 },
    },
};

// command line

const Format = enum { table, csv, json };

/// bounds the fixed-size scratch buffers `summarize` sorts in.
const max_trials = 64;

const Options = struct {
    list: bool = false,
    only: []const []const u8 = &.{},
    skip: []const []const u8 = &.{},
    /// empty means "per-benchmark setting, else the default sweep".
    threads: []const u32 = &.{},
    trials: u32 = 5,
    duration_ms: u32 = 1000,
    warmup_ms: u32 = 200,
    /// non-zero switches from timed runs to a fixed transaction count per thread.
    execute: u32 = 0,
    modes: []const zstm.Tx.Mode = &.{.ala},
    latency: bool = true,
    latency_stride: u32 = 64,
    format: Format = .table,
    save: ?[]const u8 = null,
    baseline: ?[]const u8 = null,
    verify: bool = true,
    pin: bool = false,
    quiet: bool = false,
    /// Global overrides for the per-benchmark params; null means "leave alone".
    elements: ?u32 = null,
    ops: ?u32 = null,
    nops_after_tx: ?u32 = null,
};

const usage =
    \\zstm benchmark driver
    \\
    \\Usage: bench [options]
    \\
    \\Selection
    \\  --list                 list registered benchmarks and exit
    \\  --only  <a,b,...>      run only benchmarks whose name contains one of these
    \\  --skip  <a,b,...>      skip benchmarks whose name contains one of these
    \\  --threads <1,2,4,...>  thread counts to sweep (default: powers of 2 up to nproc)
    \\  --mode  <ala|sla|both> publication-safety mode(s) to measure (default: ala)
    \\
    \\Measurement
    \\  --trials <n>           repetitions per point, for noise estimation (default: 5)
    \\  --duration <ms>        length of one timed trial (default: 1000)
    \\  --warmup <ms>          discarded warm-up before each trial (default: 200)
    \\  --execute <n>          fixed transactions per thread instead of a timed run
    \\  --no-latency           skip latency sampling (removes its ~0.5% overhead)
    \\  --latency-stride <n>   sample one transaction in n (default: 64)
    \\  --pin                  pin worker i to CPU i
    \\  --no-verify            skip the workload's post-run invariant check
    \\
    \\Workload overrides (applied to every selected benchmark)
    \\  --elements <n>         array size
    \\  --ops <n>              operations per transaction
    \\  --nops-after-tx <n>    non-transactional spin between transactions
    \\
    \\Output
    \\  --format <table|csv|json>   default: table
    \\  --save <file>          also write the full csv to <file>, for --baseline
    \\  --baseline <file>      diff this run against a previously saved csv
    \\  --quiet                no progress output on stderr
    \\  --help
    \\
;

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("bench: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

fn parseArgs(arena: Allocator, args: std.process.Args) !Options {
    var o: Options = .{};
    var it = args.iterate();
    _ = it.next(); // argv[0]

    while (it.next()) |raw| {
        // Accept both `--key value` and `--key=value`.
        var key: []const u8 = raw;
        var inline_value: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, raw, '=')) |eq| {
            key = raw[0..eq];
            inline_value = raw[eq + 1 ..];
        }

        const next = struct {
            fn get(
                iter: *std.process.Args.Iterator,
                inl: ?[]const u8,
                name: []const u8,
            ) []const u8 {
                if (inl) |v| return v;
                return iter.next() orelse fatal("{s} needs a value", .{name});
            }
        }.get;

        if (eql(key, "--help") or eql(key, "-h")) {
            std.debug.print("{s}", .{usage});
            std.process.exit(0);
        } else if (eql(key, "--list")) {
            o.list = true;
        } else if (eql(key, "--only")) {
            o.only = try splitList(arena, next(&it, inline_value, key));
        } else if (eql(key, "--skip")) {
            o.skip = try splitList(arena, next(&it, inline_value, key));
        } else if (eql(key, "--threads")) {
            o.threads = try splitUints(arena, next(&it, inline_value, key));
        } else if (eql(key, "--trials")) {
            o.trials = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--duration")) {
            o.duration_ms = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--warmup")) {
            o.warmup_ms = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--execute")) {
            o.execute = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--mode")) {
            const v = next(&it, inline_value, key);
            if (eql(v, "ala")) {
                o.modes = &.{.ala};
            } else if (eql(v, "sla")) {
                o.modes = &.{.sla};
            } else if (eql(v, "both") or eql(v, "ala,sla")) {
                o.modes = &.{ .ala, .sla };
            } else fatal("unknown --mode '{s}' (want ala, sla or both)", .{v});
        } else if (eql(key, "--latency")) {
            o.latency = true;
        } else if (eql(key, "--no-latency")) {
            o.latency = false;
        } else if (eql(key, "--latency-stride")) {
            o.latency_stride = parseU32(next(&it, inline_value, key));
            if (o.latency_stride == 0) fatal("--latency-stride must be >= 1", .{});
        } else if (eql(key, "--pin")) {
            o.pin = true;
        } else if (eql(key, "--no-verify")) {
            o.verify = false;
        } else if (eql(key, "--quiet") or eql(key, "-q")) {
            o.quiet = true;
        } else if (eql(key, "--elements")) {
            o.elements = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--ops")) {
            o.ops = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--nops-after-tx")) {
            o.nops_after_tx = parseU32(next(&it, inline_value, key));
        } else if (eql(key, "--format")) {
            const v = next(&it, inline_value, key);
            o.format = std.meta.stringToEnum(Format, v) orelse
                fatal("unknown --format '{s}'", .{v});
        } else if (eql(key, "--save")) {
            o.save = next(&it, inline_value, key);
        } else if (eql(key, "--baseline")) {
            o.baseline = next(&it, inline_value, key);
        } else {
            fatal("unknown option '{s}' (try --help)", .{raw});
        }
    }

    if (o.trials == 0 or o.trials > max_trials)
        fatal("--trials must be between 1 and {d}", .{max_trials});
    if (o.execute == 0 and o.duration_ms == 0) fatal("--duration must be >= 1ms", .{});
    return o;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn parseU32(s: []const u8) u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, s, " "), 10) catch
        fatal("'{s}' is not a number", .{s});
}

fn splitList(arena: Allocator, s: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |part| {
        const t = std.mem.trim(u8, part, " ");
        if (t.len != 0) try out.append(arena, t);
    }
    return out.items;
}

fn splitUints(arena: Allocator, s: []const u8) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |part| {
        const t = std.mem.trim(u8, part, " ");
        if (t.len == 0) continue;
        const n = parseU32(t);
        if (n == 0) fatal("thread count must be >= 1", .{});
        try out.append(arena, n);
    }
    return out.items;
}

// =============================================================================
// 3. One measurement
// =============================================================================

/// Log-scale latency histogram: 8 sub-buckets per octave, so bucket width is
/// under 13% of the value it holds. 4 KiB per thread, and the worker only ever
/// touches its own stack copy.
const Histogram = struct {
    const sub_bits: u6 = 3;
    const sub_count: usize = 1 << sub_bits;
    const bucket_count: usize = 64 * sub_count;

    counts: [bucket_count]u64 = @splat(0),
    total: u64 = 0,
    max: u64 = 0,

    fn index(v: u64) usize {
        if (v < sub_count) return @intCast(v);
        const e: u6 = @intCast(63 - @clz(v));
        const shift: u6 = e - sub_bits;
        const octave: usize = @as(usize, e - sub_bits) + 1;
        const sub: usize = @intCast((v >> shift) - sub_count);
        return octave * sub_count + sub;
    }

    fn lowerBound(i: usize) u64 {
        if (i < sub_count) return @intCast(i);
        const octave = i / sub_count;
        const sub = i % sub_count;
        const shift: u6 = @intCast(octave - 1);
        return @as(u64, sub_count + sub) << shift;
    }

    fn width(i: usize) u64 {
        if (i < sub_count) return 1;
        return @as(u64, 1) << @intCast(i / sub_count - 1);
    }

    fn add(self: *Histogram, v: u64) void {
        self.counts[index(v)] += 1;
        self.total += 1;
        if (v > self.max) self.max = v;
    }

    fn merge(self: *Histogram, other: *const Histogram) void {
        for (&self.counts, other.counts) |*a, b| a.* += b;
        self.total += other.total;
        if (other.max > self.max) self.max = other.max;
    }

    /// Value at quantile `q` (0..1), reported as the midpoint of its bucket.
    fn quantile(self: *const Histogram, q: f64) u64 {
        if (self.total == 0) return 0;
        const want_f = q * @as(f64, @floatFromInt(self.total));
        const want: u64 = @max(1, @as(u64, @intFromFloat(@ceil(want_f))));
        var cum: u64 = 0;
        for (self.counts, 0..) |c, i| {
            if (c == 0) continue;
            cum += c;
            if (cum >= want) return lowerBound(i) + width(i) / 2;
        }
        return self.max;
    }
};

/// Per-worker accumulators. The first field is cache-line aligned so the shared
/// array has one entry per line and workers never false-share on hand-off.
const ThreadStats = struct {
    commits: u64 align(cache_line) = 0,
    aborts: u64 = 0,
    warmup_commits: u64 = 0,
    warmup_aborts: u64 = 0,
    hist: Histogram = .{},
};

const Barrier = struct {
    n: u32,
    count: std.atomic.Value(u32) align(cache_line) = .init(0),
    gen: std.atomic.Value(u32) align(cache_line) = .init(0),

    fn wait(self: *Barrier) void {
        const my_gen = self.gen.load(.acquire);
        if (self.count.fetchAdd(1, .acq_rel) + 1 == self.n) {
            self.count.store(0, .monotonic);
            _ = self.gen.fetchAdd(1, .release);
            return;
        }
        var spins: u32 = 0;
        while (self.gen.load(.acquire) == my_gen) {
            spins += 1;
            if (spins < 4096) {
                std.atomic.spinLoopHint();
            } else {
                // Oversubscribed: stop burning the core the stragglers need.
                std.Thread.yield() catch {};
                spins = 0;
            }
        }
    }
};

const RunCtx = struct {
    barrier: Barrier,
    running: std.atomic.Value(bool) align(cache_line) = .init(true),
    mode: zstm.Tx.Mode,
    /// 0 = timed run; otherwise transactions per thread.
    execute: u32,
    warmup_execute: u32,
    /// 0 = latency sampling off.
    sample_stride: u32,
    /// Cost of reading the clock, subtracted from every sample.
    sample_overhead_ns: u64,
    pin: bool,
    ncpu: u32,
    stats: []ThreadStats,
};

fn Worker(comptime W: type) type {
    return struct {
        fn run(ctx: *RunCtx, id: u32) void {
            if (ctx.pin) pinToCpu(id % ctx.ncpu);

            // A private arena per thread: the read log and write set never
            // contend on a shared allocator, and the whole lot is freed at once.
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const alloc = arena.allocator();

            var tx: zstm.Tx = .init(alloc, &harness.stm, ctx.mode);
            defer tx.deinit();

            // Reserve up front so the measured loop never allocates.
            tx.reads.ensureTotalCapacity(alloc, W.maxReads() + 8) catch @panic("oom");
            tx.writes.ensureCapacity(alloc, W.maxWrites() + 8) catch @panic("oom");

            var st: ThreadStats = .{};
            var seed: u32 = id;
            var aborts: u64 = 0;

            ctx.barrier.wait(); // (1) everybody is set up

            // ---- warm up: fault in the data, fill the caches, ramp the clocks
            if (ctx.execute == 0) {
                while (ctx.running.load(.acquire)) {
                    W.test_(&tx, id, &seed, &aborts);
                    st.commits += 1;
                    harness.nontxnwork();
                }
            } else {
                var i: u32 = 0;
                while (i < ctx.warmup_execute) : (i += 1) {
                    W.test_(&tx, id, &seed, &aborts);
                    st.commits += 1;
                    harness.nontxnwork();
                }
            }

            ctx.barrier.wait(); // (2) warm-up over, nothing is in flight

            st.warmup_commits = st.commits;
            st.warmup_aborts = aborts;
            st.commits = 0;
            aborts = 0;
            st.hist = .{};

            ctx.barrier.wait(); // (3) the clock has started

            // ---- measured phase
            const stride = ctx.sample_stride;
            const overhead = ctx.sample_overhead_ns;
            var countdown: u32 = stride;
            if (ctx.execute == 0) {
                while (ctx.running.load(.acquire))
                    step(&tx, id, &seed, &aborts, &st, stride, overhead, &countdown);
            } else {
                var i: u32 = 0;
                while (i < ctx.execute) : (i += 1)
                    step(&tx, id, &seed, &aborts, &st, stride, overhead, &countdown);
            }

            ctx.barrier.wait(); // (4) the clock has stopped

            st.aborts = aborts;
            ctx.stats[id] = st;
        }

        /// One transaction, plus the non-transactional filler that follows it.
        /// Every `stride`-th call is timed; the sample spans the whole retry
        /// loop, so it is the latency of a *completed* transaction including
        /// everything it had to redo. `overhead` takes the clock read itself
        /// back out, which matters when a transaction costs tens of ns.
        inline fn step(
            tx: *zstm.Tx,
            id: u32,
            seed: *u32,
            aborts: *u64,
            st: *ThreadStats,
            stride: u32,
            overhead: u64,
            countdown: *u32,
        ) void {
            if (stride != 0) {
                countdown.* -= 1;
                if (countdown.* == 0) {
                    countdown.* = stride;
                    const t0 = nowNs();
                    W.test_(tx, id, seed, aborts);
                    st.hist.add((nowNs() -| t0) -| overhead);
                    st.commits += 1;
                    harness.nontxnwork();
                    return;
                }
            }
            W.test_(tx, id, seed, aborts);
            st.commits += 1;
            harness.nontxnwork();
        }
    };
}

/// Everything one trial produced. `hist` is kept so the representative trial
/// can be picked *after* all trials are in.
const Trial = struct {
    wall_ns: u64,
    commits: u64,
    aborts: u64,
    /// Transactions that took the global commit lock, from the sequence lock.
    writer_commits: u64,
    total_commits: u64,
    total_aborts: u64,
    min_thread_commits: u64,
    max_thread_commits: u64,
    hist: Histogram,
    ru: RuDelta,

    fn tps(self: Trial) f64 {
        if (self.wall_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.commits)) * 1e9 /
            @as(f64, @floatFromInt(self.wall_ns));
    }

    /// `harness.cfg.txcount` is a u32, so past 4.3G transactions a workload
    /// that checks an exact count (Counter) cannot be given the truth.
    fn verifiable(self: Trial) bool {
        return self.total_commits <= std.math.maxInt(u32);
    }
};

const RuDelta = struct {
    user_ns: u64 = 0,
    sys_ns: u64 = 0,
    vol_cs: u64 = 0,
    invol_cs: u64 = 0,
    minflt: u64 = 0,
    majflt: u64 = 0,
    maxrss_kb: u64 = 0,

    fn sample() RuDelta {
        const ru = std.posix.getrusage(std.posix.rusage.SELF);
        return .{
            .user_ns = tvNs(ru.utime),
            .sys_ns = tvNs(ru.stime),
            .vol_cs = @intCast(@max(0, ru.nvcsw)),
            .invol_cs = @intCast(@max(0, ru.nivcsw)),
            .minflt = @intCast(@max(0, ru.minflt)),
            .majflt = @intCast(@max(0, ru.majflt)),
            .maxrss_kb = @intCast(@max(0, ru.maxrss)),
        };
    }

    fn tvNs(tv: anytype) u64 {
        const sec: i64 = @intCast(tv.sec);
        const usec: i64 = @intCast(tv.usec);
        return @intCast(@max(0, sec * std.time.ns_per_s + usec * std.time.ns_per_us));
    }

    fn since(now: RuDelta, before: RuDelta) RuDelta {
        return .{
            .user_ns = now.user_ns -| before.user_ns,
            .sys_ns = now.sys_ns -| before.sys_ns,
            .vol_cs = now.vol_cs -| before.vol_cs,
            .invol_cs = now.invol_cs -| before.invol_cs,
            .minflt = now.minflt -| before.minflt,
            .majflt = now.majflt -| before.majflt,
            .maxrss_kb = now.maxrss_kb,
        };
    }
};

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Median cost of one `nowNs`, so latency samples can be corrected for the
/// instrument. On a vDSO clock this lands around 20ns -- the same order as a
/// whole uncontended transaction, which is why it is worth subtracting.
fn calibrateTimer() u64 {
    var samples: [129]u64 = undefined;
    for (&samples) |*s| {
        const a = nowNs();
        const b = nowNs();
        s.* = b -| a;
    }
    std.mem.sort(u64, &samples, {}, struct {
        fn lt(_: void, x: u64, y: u64) bool {
            return x < y;
        }
    }.lt);
    return samples[samples.len / 2];
}

fn pinToCpu(cpu: u32) void {
    var set: linux.cpu_set_t = @splat(0);
    const bits = @bitSizeOf(usize);
    if (cpu / bits >= set.len) return;
    set[cpu / bits] = @as(usize, 1) << @intCast(cpu % bits);
    linux.sched_setaffinity(0, &set) catch {};
}

/// Run one trial of `W` at `threads` threads and return the raw numbers.
fn runTrial(
    comptime W: type,
    gpa: Allocator,
    io: std.Io,
    o: Options,
    p: Params,
    threads: u32,
    mode: zstm.Tx.Mode,
) !Trial {
    // Fresh library state so the sequence lock counts only this trial, and so
    // workloads that assert on it (Counter) see a clean slate.
    harness.stm = .init;
    harness.cfg = .{
        .bmname = p.bmname,
        .duration = @max(1, o.duration_ms / 1000),
        .execute = o.execute,
        .threads = threads,
        .nops_after_tx = p.nops_after_tx,
        .elements = p.elements,
        .lookpct = p.lookpct,
        .inspct = p.inspct,
        .sets = p.sets,
        .ops = p.ops,
        .mode = mode,
    };

    // Workload data is rebuilt per trial; the warm-up phase absorbs the
    // first-touch page faults that come with it.
    var data_arena = std.heap.ArenaAllocator.init(gpa);
    defer data_arena.deinit();

    W.reparse();
    W.init(data_arena.allocator()) catch |err| {
        if (err == error.BadBenchName) fatal(
            "'{s}' is not a valid bmname for this workload" ++
                " (Disjoint wants e.g. PrDw-64-5-5)",
            .{harness.cfg.bmname},
        );
        return err;
    };

    const stats = try gpa.alloc(ThreadStats, threads);
    defer gpa.free(stats);
    @memset(stats, .{});

    var ctx: RunCtx = .{
        .barrier = .{ .n = threads + 1 }, // workers + this thread
        .mode = mode,
        .execute = o.execute,
        .warmup_execute = if (o.warmup_ms == 0) 0 else @max(1, o.execute / 8),
        .sample_stride = if (o.latency) o.latency_stride else 0,
        .sample_overhead_ns = timer_overhead_ns,
        .pin = o.pin,
        .ncpu = @max(1, cpuCount()),
        .stats = stats,
    };

    const workers = try gpa.alloc(std.Thread, threads);
    defer gpa.free(workers);

    // A partial spawn would leave the survivors wedged on a barrier that can
    // never complete, so there is nothing sensible to unwind to.
    for (workers, 0..) |*t, id| {
        t.* = std.Thread.spawn(.{}, Worker(W).run, .{ &ctx, @as(u32, @intCast(id)) }) catch |err|
            fatal("cannot spawn worker {d} of {d}: {t}", .{ id, threads, err });
    }

    ctx.barrier.wait(); // (1)

    if (o.execute == 0) {
        if (o.warmup_ms != 0) try sleepMs(io, o.warmup_ms);
        ctx.running.store(false, .release);
    }

    ctx.barrier.wait(); // (2) all workers are parked; nothing is in flight

    ctx.running.store(true, .release);
    const seq0 = harness.stm.seq_lock.load(.acquire);
    const ru0 = RuDelta.sample();
    const t0 = nowNs();

    ctx.barrier.wait(); // (3) release the workers into the measured phase

    if (o.execute == 0) {
        try sleepMs(io, o.duration_ms);
        ctx.running.store(false, .release);
    }

    ctx.barrier.wait(); // (4) everyone has stopped

    const t1 = nowNs();
    const ru1 = RuDelta.sample();
    const seq1 = harness.stm.seq_lock.load(.acquire);

    for (workers) |t| t.join();

    var trial: Trial = .{
        .wall_ns = t1 -| t0,
        .commits = 0,
        .aborts = 0,
        .writer_commits = (seq1 -| seq0) / 2,
        .total_commits = 0,
        .total_aborts = 0,
        .min_thread_commits = std.math.maxInt(u64),
        .max_thread_commits = 0,
        .hist = .{},
        .ru = RuDelta.since(ru1, ru0),
    };
    for (stats) |*s| {
        trial.commits += s.commits;
        trial.aborts += s.aborts;
        trial.total_commits += s.commits + s.warmup_commits;
        trial.total_aborts += s.aborts + s.warmup_aborts;
        trial.min_thread_commits = @min(trial.min_thread_commits, s.commits);
        trial.max_thread_commits = @max(trial.max_thread_commits, s.commits);
        trial.hist.merge(&s.hist);
    }

    // The workload's own invariant check reads these out of the config, and it
    // reasons about the whole life of the sequence lock -- warm-up included.
    // RSTM's txcount is a u32, so a long enough run cannot be checked at all;
    // `verifiable` says whether the number that lands here is the real one.
    harness.cfg.txcount = .init(@intCast(@min(trial.total_commits, std.math.maxInt(u32))));
    harness.cfg.aborts = .init(trial.total_aborts);

    return trial;
}

fn sleepMs(io: std.Io, ms: u32) !void {
    try std.Io.sleep(io, .{ .nanoseconds = @as(i96, ms) * std.time.ns_per_ms }, .awake);
}

fn cpuCount() u32 {
    return @intCast(std.Thread.getCpuCount() catch 1);
}

// =============================================================================
// 4. Aggregation
// =============================================================================

/// One row of output. Field names are the CSV header *and* the JSON keys *and*
/// what `--baseline` looks for, so adding a metric here plumbs it everywhere.
const Record = struct {
    bench: []const u8,
    mode: []const u8,
    threads: u32,
    trials: u32,
    duration_ms: u32,
    execute: u32,
    elements: u32,
    ops: u32,
    bmname: []const u8,

    /// Median across trials. Everything below comes from the median trial, so
    /// the whole row describes one single self-consistent run.
    tx_per_s: f64,
    tx_per_s_mean: f64,
    tx_per_s_sd: f64,
    tx_per_s_min: f64,
    tx_per_s_max: f64,
    cv_pct: f64,

    ns_per_tx: f64,
    commits: u64,
    aborts: u64,
    writer_commits: u64,
    abort_pct: f64,
    aborts_per_commit: f64,
    writer_pct: f64,

    p50_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    max_ns: u64,

    imbalance_pct: f64,
    user_ns: u64,
    sys_ns: u64,
    sys_pct: f64,
    vol_cs: u64,
    invol_cs: u64,
    invol_cs_per_ktx: f64,
    minflt: u64,
    majflt: u64,
    maxrss_kb: u64,
    verified: []const u8,

    /// Filled in only for the on-screen table; not part of the saved schema.
    const Extra = struct {
        desc: []const u8 = "",
        speedup: f64 = 0,
    };
};

fn mean(xs: []const f64) f64 {
    if (xs.len == 0) return 0;
    var s: f64 = 0;
    for (xs) |x| s += x;
    return s / @as(f64, @floatFromInt(xs.len));
}

fn stddev(xs: []const f64) f64 {
    if (xs.len < 2) return 0;
    const m = mean(xs);
    var s: f64 = 0;
    for (xs) |x| s += (x - m) * (x - m);
    return @sqrt(s / @as(f64, @floatFromInt(xs.len - 1)));
}

fn summarize(
    name: []const u8,
    mode: zstm.Tx.Mode,
    threads: u32,
    o: Options,
    p: Params,
    trials: []Trial,
    scratch: []f64,
    verified: []const u8,
) Record {
    std.debug.assert(trials.len <= max_trials);
    for (trials, 0..) |t, i| scratch[i] = t.tps();
    const rates = scratch[0..trials.len];
    const n = trials.len;

    // Order the trials by throughput, then report the details of the middle
    // one, so latency, aborts and rusage all describe a single real run rather
    // than an average over runs that may have behaved differently.
    var order: [max_trials]u32 = undefined;
    for (0..n) |i| order[i] = @intCast(i);
    std.mem.sort(u32, order[0..n], rates, struct {
        fn lt(r: []const f64, a: u32, b: u32) bool {
            return r[a] < r[b];
        }
    }.lt);
    const rep = trials[order[n / 2]];
    const median = rates[order[n / 2]];

    const m = mean(rates);
    const sd = stddev(rates);
    const attempts = rep.commits + rep.aborts;
    const commits_f = @as(f64, @floatFromInt(@max(1, rep.commits)));
    const cpu_ns = rep.ru.user_ns + rep.ru.sys_ns;

    return .{
        .bench = name,
        .mode = @tagName(mode),
        .threads = threads,
        .trials = @intCast(trials.len),
        .duration_ms = if (o.execute == 0) o.duration_ms else 0,
        .execute = o.execute,
        .elements = p.elements,
        .ops = p.ops,
        .bmname = harness.cfg.bmname,

        .tx_per_s = median,
        .tx_per_s_mean = m,
        .tx_per_s_sd = sd,
        .tx_per_s_min = rates[order[0]],
        .tx_per_s_max = rates[order[n - 1]],
        .cv_pct = if (m > 0) sd / m * 100 else 0,

        .ns_per_tx = @as(f64, @floatFromInt(rep.wall_ns)) *
            @as(f64, @floatFromInt(threads)) / commits_f,
        .commits = rep.commits,
        .aborts = rep.aborts,
        .writer_commits = rep.writer_commits,
        .abort_pct = if (attempts > 0)
            @as(f64, @floatFromInt(rep.aborts)) * 100 / @as(f64, @floatFromInt(attempts))
        else
            0,
        .aborts_per_commit = @as(f64, @floatFromInt(rep.aborts)) / commits_f,
        .writer_pct = @as(f64, @floatFromInt(rep.writer_commits)) * 100 / commits_f,

        .p50_ns = rep.hist.quantile(0.50),
        .p99_ns = rep.hist.quantile(0.99),
        .p999_ns = rep.hist.quantile(0.999),
        .max_ns = rep.hist.max,

        .imbalance_pct = if (rep.max_thread_commits > 0)
            @as(f64, @floatFromInt(rep.max_thread_commits - rep.min_thread_commits)) *
                100 / @as(f64, @floatFromInt(rep.max_thread_commits))
        else
            0,
        .user_ns = rep.ru.user_ns,
        .sys_ns = rep.ru.sys_ns,
        .sys_pct = if (cpu_ns > 0)
            @as(f64, @floatFromInt(rep.ru.sys_ns)) * 100 / @as(f64, @floatFromInt(cpu_ns))
        else
            0,
        .vol_cs = rep.ru.vol_cs,
        .invol_cs = rep.ru.invol_cs,
        .invol_cs_per_ktx = @as(f64, @floatFromInt(rep.ru.invol_cs)) * 1000 / commits_f,
        .minflt = rep.ru.minflt,
        .majflt = rep.ru.majflt,
        .maxrss_kb = rep.ru.maxrss_kb,
        .verified = verified,
    };
}

// =============================================================================
// 5. Reporting
// =============================================================================

fn cell(w: *Writer, width: usize, comptime fmt: []const u8, args: anytype) !void {
    var buf: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..0];
    if (s.len < width) try w.splatByteAll(' ', width - s.len);
    try w.writeAll(s);
    try w.writeByte(' ');
}

fn groupDigits(out: []u8, v: u64) []const u8 {
    var tmp: [24]u8 = undefined;
    const s = std.fmt.bufPrint(&tmp, "{d}", .{v}) catch return "?";
    var n: usize = 0;
    for (s, 0..) |c, k| {
        if (k != 0 and (s.len - k) % 3 == 0) {
            out[n] = ',';
            n += 1;
        }
        out[n] = c;
        n += 1;
    }
    return out[0..n];
}

fn fmtDur(out: []u8, ns: u64) []const u8 {
    const f: f64 = @floatFromInt(ns);
    const s = if (ns < 1_000)
        std.fmt.bufPrint(out, "{d}ns", .{ns})
    else if (ns < 1_000_000)
        std.fmt.bufPrint(out, "{d:.2}us", .{f / 1e3})
    else if (ns < 1_000_000_000)
        std.fmt.bufPrint(out, "{d:.2}ms", .{f / 1e6})
    else
        std.fmt.bufPrint(out, "{d:.2}s", .{f / 1e9});
    return s catch "?";
}

const table_header = [_]struct { w: usize, h: []const u8 }{
    .{ .w = 4, .h = "thr" },
    .{ .w = 13, .h = "tx/s" },
    .{ .w = 6, .h = "cv%" },
    .{ .w = 6, .h = "scale" },
    .{ .w = 5, .h = "eff%" },
    .{ .w = 10, .h = "ns/tx" },
    .{ .w = 7, .h = "ab/tx" },
    .{ .w = 7, .h = "abort%" },
    .{ .w = 6, .h = "wr%" },
    .{ .w = 8, .h = "p50" },
    .{ .w = 8, .h = "p99" },
    .{ .w = 8, .h = "p99.9" },
    .{ .w = 6, .h = "imb%" },
    .{ .w = 6, .h = "sys%" },
    .{ .w = 8, .h = "ics/ktx" },
};

fn printTable(w: *Writer, records: []const Record, extras: []const Record.Extra) !void {
    var i: usize = 0;
    while (i < records.len) {
        // One block per (bench, mode).
        var j = i;
        while (j < records.len and
            eql(records[j].bench, records[i].bench) and
            eql(records[j].mode, records[i].mode)) : (j += 1)
        {}
        const block = records[i..j];
        const head = block[0];

        try w.print("\n{s}  [mode={s}", .{ head.bench, head.mode });
        if (head.elements != 0) try w.print(" elements={d}", .{head.elements});
        try w.print(" ops={d}", .{head.ops});
        if (head.bmname.len != 0) try w.print(" bmname={s}", .{head.bmname});
        try w.print("]\n", .{});
        if (extras[i].desc.len != 0) try w.print("  {s}\n", .{extras[i].desc});

        for (table_header) |c| try cell(w, c.w, "{s}", .{c.h});
        try w.writeByte('\n');

        for (block, extras[i..j]) |r, e| {
            var b1: [32]u8 = undefined;
            var b2: [32]u8 = undefined;
            var b3: [32]u8 = undefined;
            var b4: [32]u8 = undefined;
            try cell(w, table_header[0].w, "{d}", .{r.threads});
            const tps_txt = groupDigits(&b1, @intFromFloat(@round(r.tx_per_s)));
            try cell(w, table_header[1].w, "{s}", .{tps_txt});
            try cell(w, table_header[2].w, "{d:.1}", .{r.cv_pct});
            if (e.speedup > 0) {
                try cell(w, table_header[3].w, "{d:.2}x", .{e.speedup});
                const eff = e.speedup / @as(f64, @floatFromInt(r.threads)) * 100;
                try cell(w, table_header[4].w, "{d:.0}", .{eff});
            } else {
                try cell(w, table_header[3].w, "{s}", .{"-"});
                try cell(w, table_header[4].w, "{s}", .{"-"});
            }
            try cell(w, table_header[5].w, "{d:.1}", .{r.ns_per_tx});
            try cell(w, table_header[6].w, "{d:.3}", .{r.aborts_per_commit});
            try cell(w, table_header[7].w, "{d:.2}", .{r.abort_pct});
            try cell(w, table_header[8].w, "{d:.1}", .{r.writer_pct});
            if (r.p50_ns != 0 or r.p99_ns != 0) {
                try cell(w, table_header[9].w, "{s}", .{fmtDur(&b2, r.p50_ns)});
                try cell(w, table_header[10].w, "{s}", .{fmtDur(&b3, r.p99_ns)});
                try cell(w, table_header[11].w, "{s}", .{fmtDur(&b4, r.p999_ns)});
            } else {
                try cell(w, table_header[9].w, "{s}", .{"-"});
                try cell(w, table_header[10].w, "{s}", .{"-"});
                try cell(w, table_header[11].w, "{s}", .{"-"});
            }
            try cell(w, table_header[12].w, "{d:.1}", .{r.imbalance_pct});
            try cell(w, table_header[13].w, "{d:.1}", .{r.sys_pct});
            try cell(w, table_header[14].w, "{d:.2}", .{r.invol_cs_per_ktx});
            try w.writeByte('\n');
        }

        try printHints(w, block);
        i = j;
    }
}

const Hints = struct {
    w: *Writer,
    any: bool = false,

    fn say(self: *Hints, comptime fmt: []const u8, args: anytype) !void {
        try self.w.writeAll(if (self.any) "       " else "  why: ");
        self.any = true;
        try self.w.print(fmt ++ "\n", args);
    }
};

/// Cheap rules of thumb. They are not conclusions -- they point at the column
/// most likely to explain a number you did not expect. All of them read the
/// widest-thread row, except the noise one, which takes the worst of the block.
fn printHints(w: *Writer, block: []const Record) !void {
    const r = block[block.len - 1];
    var worst_cv: f64 = 0;
    for (block) |x| worst_cv = @max(worst_cv, x.cv_pct);

    var h: Hints = .{ .w = w };

    if (worst_cv > 5) try h.say(
        "run-to-run spread reaches {d:.1}%: smaller deltas are noise",
        .{worst_cv},
    );
    if (r.abort_pct > 25) try h.say(
        "{d:.0}% of attempts abort: contention-bound, not barrier-bound",
        .{r.abort_pct},
    );
    if (r.writer_pct > 50 and block.len > 1 and r.threads > 1) try h.say(
        "{d:.0}% of commits are writers: the sequence lock serializes them",
        .{r.writer_pct},
    );
    if (r.p99_ns > 20 * @max(1, r.p50_ns)) try h.say(
        "p99 is {d}x p50: a few transactions retry a lot",
        .{r.p99_ns / @max(1, r.p50_ns)},
    );
    if (r.sys_pct > 10) try h.say(
        "{d:.0}% of CPU time is kernel: suspect the scheduler, not the STM",
        .{r.sys_pct},
    );
    if (r.invol_cs_per_ktx > 1) try h.say(
        "{d:.1} involuntary switches per 1k tx: oversubscribed, try --pin",
        .{r.invol_cs_per_ktx},
    );
    if (r.imbalance_pct > 25) try h.say(
        "threads differ by {d:.0}% in work done: starvation or uneven pinning",
        .{r.imbalance_pct},
    );
}

const legend =
    \\
    \\Columns
    \\  tx/s     committed transactions per second, median of all trials
    \\  cv%      coefficient of variation of tx/s across trials -- the noise floor
    \\  scale    tx/s relative to this benchmark's 1-thread run; eff% is scale/threads
    \\  ns/tx    wall time x threads / commits: mean CPU-time cost of one commit
    \\  ab/tx    aborted attempts per committed transaction
    \\  abort%   aborts / (aborts + commits)
    \\  wr%      commits that acquired the global sequence lock (writers)
    \\  p50..    sampled latency of one completed transaction, retries included
    \\  imb%     spread between the busiest and idlest thread
    \\  sys%     kernel share of CPU time; ics/ktx = involuntary context switches per 1k tx
    \\
    \\All rows but tx/s come from the median-throughput trial, so a row describes
    \\one real run. Latency sampling costs ~0.5% throughput; --no-latency removes it.
    \\
;

fn printCsv(w: *Writer, records: []const Record) !void {
    inline for (@typeInfo(Record).@"struct".fields, 0..) |f, i| {
        if (i != 0) try w.writeByte(',');
        try w.writeAll(f.name);
    }
    try w.writeByte('\n');
    for (records) |r| {
        inline for (@typeInfo(Record).@"struct".fields, 0..) |f, i| {
            if (i != 0) try w.writeByte(',');
            const v = @field(r, f.name);
            switch (@typeInfo(f.type)) {
                .pointer => try w.print("{s}", .{v}),
                .float => try w.print("{d:.4}", .{finite(v)}),
                else => try w.print("{d}", .{v}),
            }
        }
        try w.writeByte('\n');
    }
}

fn printJson(w: *Writer, records: []const Record) !void {
    try w.writeAll("[\n");
    for (records, 0..) |r, ri| {
        try w.writeAll("  {");
        inline for (@typeInfo(Record).@"struct".fields, 0..) |f, i| {
            if (i != 0) try w.writeAll(", ");
            const v = @field(r, f.name);
            switch (@typeInfo(f.type)) {
                .pointer => try w.print("\"{s}\": \"{s}\"", .{ f.name, v }),
                .float => try w.print("\"{s}\": {d:.4}", .{ f.name, finite(v) }),
                else => try w.print("\"{s}\": {d}", .{ f.name, v }),
            }
        }
        try w.writeAll(if (ri + 1 == records.len) "}\n" else "},\n");
    }
    try w.writeAll("]\n");
}

fn finite(v: f64) f64 {
    return if (std.math.isFinite(v)) v else 0;
}

// =============================================================================
// 6. Baseline comparison
// =============================================================================

/// Parse a csv written by `--save`. Columns are matched by header name, so a
/// baseline recorded before a new metric existed still loads.
fn parseCsv(arena: Allocator, text: []const u8) ![]Record {
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    const header = lines.next() orelse return error.EmptyBaseline;

    const fields = @typeInfo(Record).@"struct".fields;
    var col: [fields.len]?usize = @splat(null);

    var hi: usize = 0;
    var hit = std.mem.splitScalar(u8, header, ',');
    while (hit.next()) |h| : (hi += 1) {
        const name = std.mem.trim(u8, h, " ");
        inline for (fields, 0..) |f, fi| {
            if (eql(name, f.name)) col[fi] = hi;
        }
    }

    var out: std.ArrayList(Record) = .empty;
    var cells: [fields.len * 2][]const u8 = undefined;
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " ").len == 0) continue;
        var n: usize = 0;
        var cit = std.mem.splitScalar(u8, line, ',');
        while (cit.next()) |c| : (n += 1) {
            if (n == cells.len) break;
            cells[n] = std.mem.trim(u8, c, " ");
        }

        var r: Record = undefined;
        inline for (fields, 0..) |f, fi| {
            const raw: ?[]const u8 = if (col[fi]) |ci| (if (ci < n) cells[ci] else null) else null;
            @field(r, f.name) = switch (@typeInfo(f.type)) {
                .pointer => raw orelse "",
                .float => if (raw) |s| (std.fmt.parseFloat(f64, s) catch 0) else 0,
                else => if (raw) |s| (std.fmt.parseInt(f.type, s, 10) catch 0) else 0,
            };
        }
        try out.append(arena, r);
    }
    return out.items;
}

fn findBaseline(base: []const Record, r: Record) ?Record {
    for (base) |b| {
        if (eql(b.bench, r.bench) and eql(b.mode, r.mode) and b.threads == r.threads)
            return b;
    }
    return null;
}

fn pctDelta(now: f64, before: f64) f64 {
    if (before == 0) return 0;
    return (now - before) / before * 100;
}

/// Trial-to-trial spread understates the spread between two *separate* runs of
/// the binary, so the bar is three times the worse cv and never below 2%.
fn significant(delta_pct: f64, cv_a: f64, cv_b: f64) bool {
    return @abs(delta_pct) > @max(2.0, 3 * @max(cv_a, cv_b));
}

fn printComparison(w: *Writer, records: []const Record, base: []const Record) !void {
    try w.writeAll(
        \\
        \\vs baseline  ('*' marks a delta larger than max(2%, 3x the worse cv);
        \\              d-abort is in percentage points, everything else is relative)
        \\
    );
    try cell(w, 20, "{s}", .{"benchmark"});
    try cell(w, 4, "{s}", .{"thr"});
    try cell(w, 11, "{s}", .{"now tx/s"});
    try cell(w, 11, "{s}", .{"base tx/s"});
    try cell(w, 9, "{s}", .{"d-tx/s"});
    try cell(w, 11, "{s}", .{"cv now/base"});
    try cell(w, 8, "{s}", .{"d-ns/tx"});
    try cell(w, 9, "{s}", .{"d-abort"});
    try cell(w, 8, "{s}", .{"d-p99"});
    try w.writeByte('\n');

    var matched: usize = 0;
    for (records) |r| {
        const b = findBaseline(base, r) orelse continue;
        matched += 1;
        const d_tps = pctDelta(r.tx_per_s, b.tx_per_s);
        const d_ns = pctDelta(r.ns_per_tx, b.ns_per_tx);
        const d_p99 = pctDelta(@floatFromInt(r.p99_ns), @floatFromInt(b.p99_ns));

        var b1: [32]u8 = undefined;
        var b2: [32]u8 = undefined;
        try cell(w, 20, "{s}/{s}", .{ r.bench, r.mode });
        try cell(w, 4, "{d}", .{r.threads});
        try cell(w, 11, "{s}", .{groupDigits(&b1, @intFromFloat(@round(r.tx_per_s)))});
        try cell(w, 11, "{s}", .{groupDigits(&b2, @intFromFloat(@round(b.tx_per_s)))});
        try cell(w, 9, "{s}{d:.1}%{s}", .{
            if (d_tps >= 0) "+" else "",
            d_tps,
            if (significant(d_tps, r.cv_pct, b.cv_pct)) "*" else " ",
        });
        try cell(w, 11, "{d:.1}/{d:.1}%", .{ r.cv_pct, b.cv_pct });
        try cell(w, 8, "{s}{d:.1}%", .{ if (d_ns >= 0) "+" else "", d_ns });
        try cell(w, 9, "{s}{d:.2}pp", .{
            if (r.abort_pct - b.abort_pct >= 0) "+" else "",
            r.abort_pct - b.abort_pct,
        });
        try cell(w, 8, "{s}{d:.1}%", .{ if (d_p99 >= 0) "+" else "", d_p99 });
        try w.writeByte('\n');
    }
    if (matched == 0) {
        try w.writeAll("  no baseline row matched this run's bench/mode/threads\n");
    } else if (matched < records.len) {
        try w.print("  {d} of {d} rows had no counterpart in the baseline\n", .{
            records.len - matched, records.len,
        });
    }
}

// =============================================================================
// 7. Driver
// =============================================================================

/// Cost of one clock read, measured once at startup.
var timer_overhead_ns: u64 = 0;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    const o = try parseArgs(arena, init.minimal.args);
    timer_overhead_ns = calibrateTimer();

    var out_buf: [64 * 1024]u8 = undefined;
    var out_file = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_file.interface;
    defer out.flush() catch {};

    var err_buf: [4096]u8 = undefined;
    var err_file = std.Io.File.stderr().writer(io, &err_buf);
    const log = &err_file.interface;

    if (o.list) {
        try out.writeAll("registered benchmarks:\n");
        inline for (suite) |b| {
            try out.print("  {s}{s:<18} {s}\n", .{
                if (b.enabled) "  " else "# ",
                b.name,
                b.desc,
            });
        }
        try out.writeAll("\n('#' = .enabled = false in src/bench/main.zig)\n");
        return;
    }

    const default_threads = try defaultThreadSweep(arena);

    var records: std.ArrayList(Record) = .empty;
    var extras: std.ArrayList(Record.Extra) = .empty;

    // Total point count up front, so the progress line can say how far along we are.
    var total_points: u32 = 0;
    inline for (suite) |b| {
        if (selected(b, o)) {
            const ts = threadSweep(b, o, default_threads);
            total_points += @intCast(ts.len * o.modes.len);
        }
    }
    if (total_points == 0) fatal("no benchmarks selected", .{});

    if (!o.quiet) {
        try log.print("zstm bench: {s}, {s}, {d} cpus, zig {f}\n", .{
            @tagName(builtin.mode),
            @tagName(builtin.target.cpu.arch),
            cpuCount(),
            builtin.zig_version,
        });
        if (o.execute == 0) {
            try log.print(
                "{d} points x {d} trials x ({d}ms warmup + {d}ms) = {d}s under the clock," ++
                    " plus per-trial setup\n",
                .{
                    total_points,
                    o.trials,
                    o.warmup_ms,
                    o.duration_ms,
                    @as(u64, total_points) * o.trials * (o.warmup_ms + o.duration_ms) / 1000,
                },
            );
        } else {
            try log.print("{d} points x {d} trials x {d} tx/thread\n", .{
                total_points, o.trials, o.execute,
            });
        }
        if (builtin.mode != .ReleaseFast and builtin.mode != .ReleaseSafe)
            try log.writeAll("warning: not an optimized build; numbers are meaningless\n");
        try log.flush();
    }

    const trials = try gpa.alloc(Trial, o.trials);
    defer gpa.free(trials);
    const scratch = try gpa.alloc(f64, o.trials);
    defer gpa.free(scratch);

    var done: u32 = 0;
    inline for (suite) |b| {
        if (selected(b, o)) {
            const params = applyOverrides(b.params, o);
            const sweep = threadSweep(b, o, default_threads);

            for (o.modes) |mode| {
                var single_thread_tps: f64 = 0;
                for (sweep) |threads| {
                    checkLimits(b, params, threads);
                    done += 1;

                    if (!o.quiet) {
                        try log.print("[{d}/{d}] {s} mode={s} threads={d} ", .{
                            done, total_points, b.name, @tagName(mode), threads,
                        });
                        try log.flush();
                    }

                    var verified: []const u8 = "-";
                    for (trials, 0..) |*t, i| {
                        t.* = try runTrial(b.workload, gpa, io, o, params, threads, mode);
                        if (o.verify and i + 1 == trials.len) {
                            if (!o.quiet) try log.flush(); // verify() writes to stderr
                            verified = if (!t.verifiable())
                                "skipped"
                            else if (b.workload.verify()) "ok" else "FAIL";
                        }
                    }

                    const rec = summarize(
                        b.name,
                        mode,
                        threads,
                        o,
                        params,
                        trials,
                        scratch,
                        verified,
                    );
                    if (threads == 1) single_thread_tps = rec.tx_per_s;
                    try records.append(arena, rec);
                    try extras.append(arena, .{
                        .desc = b.desc,
                        .speedup = if (single_thread_tps > 0)
                            rec.tx_per_s / single_thread_tps
                        else
                            0,
                    });

                    if (!o.quiet) {
                        var buf: [32]u8 = undefined;
                        try log.print("-> {s} tx/s  ({d:.1}% cv, {d:.1}% abort){s}\n", .{
                            groupDigits(&buf, @intFromFloat(@round(rec.tx_per_s))),
                            rec.cv_pct,
                            rec.abort_pct,
                            if (eql(verified, "FAIL"))
                                "  *** VERIFY FAILED ***"
                            else if (eql(verified, "ok")) "  verified" else "",
                        });
                        try log.flush();
                    }
                }
            }
        }
    }

    switch (o.format) {
        .table => {
            try printTable(out, records.items, extras.items);
            if (o.baseline == null) try out.writeAll(legend);
        },
        .csv => try printCsv(out, records.items),
        .json => try printJson(out, records.items),
    }

    if (o.baseline) |path| {
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(16 << 20)) catch |err|
            fatal("cannot read baseline '{s}': {t}", .{ path, err });
        const base = try parseCsv(arena, text);
        try printComparison(out, records.items, base);
        if (o.format == .table) try out.writeAll(legend);
    }

    if (o.save) |path| {
        try out.flush();
        var save_buf: [64 * 1024]u8 = undefined;
        var file = std.Io.Dir.cwd().createFile(io, path, .{}) catch |err|
            fatal("cannot write '{s}': {t}", .{ path, err });
        defer file.close(io);
        var fw = file.writer(io, &save_buf);
        try printCsv(&fw.interface, records.items);
        try fw.interface.flush();
        if (!o.quiet) {
            try log.print("wrote {s} ({d} rows)\n", .{ path, records.items.len });
            try log.flush();
        }
    }

    try out.flush();
}

/// `--only` matches on substring, so `--only disjoint` takes all three of them.
/// Naming an entry exactly also overrides `.enabled = false`; a substring match
/// does not, so `--only counter` will not drag in a disabled `counter-spaced`.
fn selected(comptime b: Bench, o: Options) bool {
    if (o.only.len != 0) {
        var hit = false;
        for (o.only) |pat| {
            if (eql(b.name, pat)) return !skipped(b, o);
            if (std.mem.indexOf(u8, b.name, pat) != null) hit = true;
        }
        if (!hit or !b.enabled) return false;
    } else if (!b.enabled) return false;

    return !skipped(b, o);
}

fn skipped(comptime b: Bench, o: Options) bool {
    for (o.skip) |pat| {
        if (std.mem.indexOf(u8, b.name, pat) != null) return true;
    }
    return false;
}

fn threadSweep(comptime b: Bench, o: Options, default: []const u32) []const u32 {
    if (o.threads.len != 0) return o.threads;
    if (b.threads) |t| return t;
    return default;
}

/// Powers of two up to the CPU count, plus the CPU count itself.
fn defaultThreadSweep(arena: Allocator) ![]const u32 {
    const ncpu = cpuCount();
    var out: std.ArrayList(u32) = .empty;
    var t: u32 = 1;
    while (t < ncpu) : (t *= 2) try out.append(arena, t);
    try out.append(arena, ncpu);
    return out.items;
}

fn applyOverrides(p: Params, o: Options) Params {
    var out = p;
    if (o.elements) |v| out.elements = v;
    if (o.ops) |v| out.ops = v;
    if (o.nops_after_tx) |v| out.nops_after_tx = v;
    return out;
}

// =============================================================================
// 8. Tests for the bits that are easy to get quietly wrong
// =============================================================================

test "histogram buckets cover their values without gaps" {
    const H = Histogram;
    var v: u64 = 0;
    while (v < 4096) : (v += 1) {
        const i = H.index(v);
        try std.testing.expect(H.lowerBound(i) <= v);
        try std.testing.expect(v < H.lowerBound(i) + H.width(i));
    }
    // Spot-check the top of the range, where the octaves are widest.
    for ([_]u64{ 1 << 20, (1 << 20) + 7, 1 << 40, std.math.maxInt(u32) }) |x| {
        const i = H.index(x);
        try std.testing.expect(i < H.bucket_count);
        try std.testing.expect(H.lowerBound(i) <= x);
        try std.testing.expect(x < H.lowerBound(i) + H.width(i));
    }
}

test "histogram quantiles land in the right bucket" {
    var h: Histogram = .{};
    // 99 samples at 100ns, one at 50us.
    for (0..99) |_| h.add(100);
    h.add(50_000);

    const p50 = h.quantile(0.50);
    try std.testing.expect(p50 >= 96 and p50 <= 104);
    const p99 = h.quantile(0.99);
    try std.testing.expect(p99 >= 96 and p99 <= 104);
    try std.testing.expect(h.quantile(0.999) > 40_000);
    try std.testing.expectEqual(@as(u64, 50_000), h.max);
}

test "records survive a csv round trip" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var before: Record = undefined;
    inline for (@typeInfo(Record).@"struct".fields) |f| {
        @field(before, f.name) = switch (@typeInfo(f.type)) {
            .pointer => "x",
            .float => 1.5,
            else => 7,
        };
    }
    before.bench = "counter";
    before.mode = "sla";
    before.threads = 4;
    before.tx_per_s = 1234567.25;
    before.cv_pct = 0.75;
    before.p99_ns = 4242;

    var buf: [8192]u8 = undefined;
    var w = Writer.fixed(&buf);
    try printCsv(&w, &.{before});

    const parsed = try parseCsv(arena, w.buffered());
    try std.testing.expectEqual(@as(usize, 1), parsed.len);
    const after = parsed[0];

    try std.testing.expectEqualStrings(before.bench, after.bench);
    try std.testing.expectEqualStrings(before.mode, after.mode);
    try std.testing.expectEqual(before.threads, after.threads);
    try std.testing.expectEqual(before.tx_per_s, after.tx_per_s);
    try std.testing.expectEqual(before.cv_pct, after.cv_pct);
    try std.testing.expectEqual(before.p99_ns, after.p99_ns);
    try std.testing.expect(findBaseline(parsed, before) != null);
}

test "a baseline missing a column still loads the rest" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const old =
        \\bench,mode,threads,tx_per_s,cv_pct
        \\counter,ala,2,555.5,1.25
        \\
    ;
    const parsed = try parseCsv(arena_state.allocator(), old);
    try std.testing.expectEqual(@as(usize, 1), parsed.len);
    try std.testing.expectEqualStrings("counter", parsed[0].bench);
    try std.testing.expectEqual(@as(u32, 2), parsed[0].threads);
    try std.testing.expectEqual(@as(f64, 555.5), parsed[0].tx_per_s);
    try std.testing.expectEqual(@as(u64, 0), parsed[0].p99_ns); // absent -> zero
}

test "significance needs a delta bigger than the noise" {
    try std.testing.expect(!significant(1.0, 0.2, 0.2)); // under the 2% floor
    try std.testing.expect(significant(9.0, 0.5, 0.5));
    try std.testing.expect(!significant(9.0, 4.0, 0.5)); // 3 x 4% = 12%
    try std.testing.expect(significant(-30.0, 4.0, 0.5));
}

test "mean and stddev" {
    const xs = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    try std.testing.expectEqual(@as(f64, 5), mean(&xs));
    // Sample stddev of this set is sqrt(32/7).
    try std.testing.expectApproxEqAbs(@sqrt(32.0 / 7.0), stddev(&xs), 1e-12);
}

fn checkLimits(comptime b: Bench, p: Params, threads: u32) void {
    if (b.limits.max_threads) |max| {
        if (threads > max) fatal(
            "{s}: workload supports at most {d} threads (asked for {d})",
            .{ b.name, max, threads },
        );
    }
    if (b.limits.max_ops) |max| {
        if (p.ops > max) fatal(
            "{s}: workload supports at most {d} ops per transaction (asked for {d})",
            .{ b.name, max, p.ops },
        );
    }
}

