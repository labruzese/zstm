//! zstm — NOrec Software Transactional Memory.
//!
//! This module implements the NOrec algorithm as described by Dalessandro, 
//! Spear, and Scott
//! (PPoPP'10, "NOrec: Streamlining STM by Abolishing Ownership Records").
//!
//! NOrec is a low-overhead STM that combines the following ideas:
//!
//!   1. A single global sequence lock 
//!   2. A redo log 
//!   3. Value-based validation 
//!
//! Properties this implementation provides:
//!
//!   - **Livelock freedom.**
//!   - **Privatization safety.**
//!   - **Publication safety (ALA by default, optional SLA).** SLA requires 
//!     one extra validation at commit time and is selectable via comptime `Tx.PubSafety`.
//!   - **Opacity.** A doomed transaction never observes inconsistent state.
//!
//! Properties this implementation does NOT provide:
//!
//!   - Transaction nesting
//!   - Hardware integration
//!   - Memory reclamation safety for transactionally freed pointers.
//!
//! ## Usage
//!
//! ```zig
//! var stm: zstm.Stm = .init;
//! var counter: zstm.TxWord = .init(0);
//!
//! var tx: zstm.Tx = .init(allocator, &stm, .ala);
//! defer tx.deinit();
//!
//! const old = try tx.run(struct {
//!     fn body(t: *zstm.Tx, c: *zstm.TxVar) zstm.Error!zstm.Word {
//!         const v = try t.read(c);
//!         try t.write(c, v + 1);
//!         return v;
//!     }
//! }.body, .{&counter});
//! ```
//!
//! Each thread should own its own `Tx`. The `Stm` is shared across threads.

const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
pub const build_options = @import("build_options"); 

pub const Word = usize;

/// A transactional variable. This is just a `Word` cell; the wrapper exists to
/// make the type system distinguish "shared, transactional state" from ordinary
/// memory and to give us a place to hang the helper accessors.
///
/// Direct field access is NOT thread-safe and bypasses transactional
/// guarantees. Use `Tx.read` / `Tx.write` for transactional access. The
/// `unsafeLoad` / `unsafeStore` helpers are for one-off, lock-free reads or
/// initialization while the variable is provably private.
pub const TxWord = extern struct {
    raw: Word,

    pub fn init(value: Word) TxWord {
        return .{ .raw = value };
    }

    /// Single relaxed atomic load. Useful for non-transactional inspection of
    /// a published variable from outside a transaction (e.g. when checking a
    /// final result). Does NOT participate in the STM and offers no
    /// consistency with concurrent transactions; only use this when no
    /// transactions are active or when you understand the racy semantics.
    pub fn unsafeLoad(self: *const TxWord) Word {
        return @atomicLoad(Word, &self.raw, .monotonic);
    }

    /// Single relaxed atomic store. Same caveats as `unsafeLoad`.
    pub fn unsafeStore(self: *TxWord, value: Word) void {
        @atomicStore(Word, &self.raw, value, .monotonic);
    }
};

/// A transactional cell that wraps any arbitrary type `T` into a sequence of
/// `Word`-sized `TxVar`s. Provides type-safe access through `Tx.readCell`
/// and `Tx.writeCell`.
pub fn TxCell(comptime T: type) type {
    const word_count = (@sizeOf(T) + @sizeOf(Word) - 1) / @sizeOf(Word);

    return extern struct {
        pub const Payload = T;
        pub const NumWords = word_count;

        words: [word_count]TxWord,

        pub fn init(value: T) @This() {
            var self: @This() = undefined;
            var words_copy = @as([]Word, @splat(word_count));
            
            const value_bytes = std.mem.asBytes(&value);
            const words_bytes = std.mem.sliceAsBytes(&words_copy);
            @memcpy(words_bytes[0..value_bytes.len], value_bytes);

            for (&self.words, 0..) |*w, i| {
                w.* = TxWord.init(words_copy[i]);
            }
            return self;
        }

        /// Non-transactional load. 
        /// **Warning for multi-word types:** This is NOT atomic across the 
        /// entire struct. It reads word-by-word, so concurrent writers could 
        /// result in a torn read (half old state, half new state).
        /// See `TxWord.unsafeLoad`
        pub fn unsafeLoad(self: *const @This()) T {
            var words_copy: [word_count]Word = undefined;
            for (&self.words, 0..) |*w, i| {
                words_copy[i] = w.unsafeLoad();
            }
            var result: T = undefined;
            const result_bytes = std.mem.asBytes(&result);
            const words_bytes = std.mem.sliceAsBytes(&words_copy);
            @memcpy(result_bytes, words_bytes[0..result_bytes.len]);
            return result;
        }

        /// Non-transactional store. Same tearing caveats as `unsafeLoad`.
        pub fn unsafeStore(self: *@This(), value: T) void {
            var words_copy = @as([]Word, @splat(0)) ;
            const value_bytes = std.mem.asBytes(&value);
            const words_bytes = std.mem.sliceAsBytes(&words_copy);
            @memcpy(words_bytes[0..value_bytes.len], value_bytes);

            for (&self.words, 0..) |*w, i| {
                w.unsafeStore(words_copy[i]);
            }
        }
    };
}

/// Errors that may flow out of a transactional operation.
///
/// `TxRetry` is an internal control signal: a `read` or `commit` returns it
/// when the transaction has observed inconsistent state. User code should
/// almost always propagate it with `try`; the surrounding `Tx.run` loop
/// catches it and restarts the transaction. Returning `error.TxRetry`
/// manually from a transaction body forces a retry.
///
/// `OutOfMemory` comes from growing the read log or write set. Unlike
/// `TxRetry`, it is propagated to the caller of `run`.
pub const Error = Allocator.Error || error{TxRetry};

/// Globally shared STM state. Typically a program has exactly one of these.
/// All `Tx` instances created against this `Stm` coordinate through it.
pub const Stm = struct {
    /// Even values mean unheld; odd values mean a writer is mid-commit. Each
    /// successful writer commit moves the counter forward by exactly 2, so an 
    /// even snapshot at one point in time uniquely identifies the state of 
    /// all transactionally-protected memory.
    seq_lock: std.atomic.Value(u64) align(std.atomic.cache_line),

    /// Default `Stm`.
    pub const init: Stm = .{ .seq_lock = .init(0) };
};

const ReadLogEntry = struct {
    addr: *const TxWord,
    val: Word,
};

/// Buffered write set: a dense entry list plus a version-stamped open-addressed
/// index into it.
///
/// Occupancy in the index is encoded as `slot.version == self.version` rather
/// than in separate metadata. Clearing is therefore a single increment of
/// `version` -- every slot becomes logically empty at once -- so `reset` is
/// O(1) no matter how much capacity is reserved.
///
/// Entries live in `list`, densely packed in insertion order, so commit-time
/// writeback is a linear scan over exactly the live entries. The index stores
/// *indices* into `list`, never pointers, so growing `list` cannot invalidate it.
///
/// Small write sets skip the index entirely (`linear_max`): probing a
/// handful of entries linearly beats hashing.
///
/// Private to one `Tx`, so nothing here needs atomics.
pub const WriteSet = struct {
    pub const Entry = struct {
        addr: *TxWord,
        val: Word,
    };

    const Slot = struct {
        addr: usize, 
        /// Occupied iff equal to the owning WriteSet's `version`.
        version: u32,
        idx: u32 
    };

    /// At or below this many entries we linear-scan and never touch the index.
    const linear_max = std.atomic.cache_line / @sizeOf(Entry);
    const min_index_bits: u6 = 5;

    const list_alignment = std.atomic.cache_line;
    list: []align(list_alignment) Entry = &.{},
    len: usize = 0,

    index: []Slot = &.{},
    index_bits: u6 = 0,
    /// True once the index is populated for the current transaction.
    indexed: bool = false,

    /// Bumped by `reset` and by `reindex`; never 0 while in use.
    version: u32 = 1,

    pub const empty: WriteSet = .{};

    pub fn count(self: *const WriteSet) usize {
        return self.len;
    }

    /// Live entries in insertion order. Commit iterates this.
    pub fn items(self: *const WriteSet) []const Entry {
        return self.list[0..self.len];
    }

    pub fn deinit(self: *WriteSet, gpa: Allocator) void {
        if (self.list.len != 0) gpa.free(self.list);
        if (self.index.len != 0) gpa.free(self.index);
        self.* = undefined;
    }

    pub fn reset(self: *WriteSet) void {
        self.len = 0;
        self.indexed = false;
        self.bumpEpoch();
    }

    fn bumpEpoch(self: *WriteSet) void {
        self.version +%= 1;
        // This is technically required for soundness but 2^64 is absurd
        // if (self.version == 0) {
        //     // Wrapped after 2^64 epochs; stale stamps could now alias the live
        //     // version, so clear for real and restart.
        //     @memset(self.index, Slot{});
        //     self.version = 1;
        // }
    }

    fn mask(self: *const WriteSet) usize {
        return (@as(usize, 1) << self.index_bits) - 1;
    }

    /// Fibonacci hashing over the significant address bits.
    ///
    /// TxWord is word-aligned, so the low 3 bits are always zero. 
    /// We take the *high* bits of the product, where multiplicative
    /// hashing concentrates entropy.
    fn hash(self: *const WriteSet, key: usize) usize {
        const mixed: u64 = @as(u64, key >> 3) *% 0x9E3779B97F4A7C15;
        const shift: u6 = @intCast(64 - @as(u7, self.index_bits));
        return @intCast(mixed >> shift);
    }

    pub fn find(self: *const WriteSet, addr: *const TxWord) ?Word {
        const key = @intFromPtr(addr);

        if (!self.indexed) {
            for (self.list[0..self.len]) |e| {
                if (@intFromPtr(e.addr) == key) return e.val;
            }
            return null;
        }

        var h = self.hash(key);
        while (self.index[h].version == self.version) {
            const slot = self.index[h];
            if (slot.addr == key) return self.list[slot.idx].val;
            h = (h + 1) & self.mask();
        }
        return null;
    }

    pub fn insert(
        self: *WriteSet,
        gpa: Allocator,
        addr: *TxWord,
        val: Word,
    ) Allocator.Error!void {
        const key = @intFromPtr(addr);

        if (!self.indexed) {
            for (self.list[0..self.len]) |*e| {
                if (@intFromPtr(e.addr) == key) {
                    e.val = val; // coalesce: last write wins
                    return;
                }
            }
            try self.append(gpa, addr, val);
            if (self.len > linear_max) try self.reindex(gpa);
            return;
        }

        var h = self.hash(key);
        while (self.index[h].version == self.version) {
            if (self.index[h].addr == key) {
                self.list[self.index[h].idx].val = val;
                return;
            }
            h = (h + 1) & self.mask();
        }

        try self.append(gpa, addr, val);
        // `append` may have reallocated `list`; the index holds indices, not
        // pointers, so existing stamps remain valid.
        self.index[h] = .{
            .addr = key,
            .version = self.version,
            .idx = @intCast(self.len - 1),
        };

        // Keep load factor under 1/4 so probe chains stay short.
        if (self.len * 4 >= (@as(usize, 1) << self.index_bits)) try self.reindex(gpa);
    }

    fn append(
        self: *WriteSet,
        gpa: Allocator,
        addr: *TxWord,
        val: Word,
    ) Allocator.Error!void {
        if (self.len == self.list.len) {
            try self.growList(gpa, if (self.list.len == 0) 16 else self.list.len * 2);
        }
        self.list[self.len] = .{ .addr = addr, .val = val };
        self.len += 1;
    }

    fn growList(self: *WriteSet, gpa: Allocator, new_cap: usize) Allocator.Error!void {
        const new = try gpa.alignedAlloc(Entry, .fromByteUnits(list_alignment), new_cap);
        @memcpy(new[0..self.len], self.list[0..self.len]);
        if (self.list.len != 0) gpa.free(self.list);
        self.list = new;
    }

    /// Size the index for the current entry count and rebuild it from `list`.
    /// Runs when the linear fast path overflows and when load factor is hit.
    fn reindex(self: *WriteSet, gpa: Allocator) Allocator.Error!void {
        var bits: u6 = min_index_bits;
        while ((@as(usize, 1) << bits) < self.len * 4) bits += 1;
        const needed = @as(usize, 1) << bits;

        if (self.index.len < needed) {
            if (self.index.len != 0) gpa.free(self.index);
            self.index = try gpa.alloc(Slot, needed);
            @memset(self.index, .{
                .addr = undefined, 
                .version = 0, // make sure stamp stale
                .idx = undefined,
            });
            self.index_bits = bits;
        } else {
            // Reuse the existing (possibly larger) table at its full size.
            self.index_bits = @intCast(std.math.log2_int(usize, self.index.len));
        }

        // Slots stamped earlier in *this* transaction must be invalidated before
        // re-stamping, so take a fresh epoch. `version` is just an epoch tag, 
        // there is no invariant that it advances exactly once per transaction.
        self.bumpEpoch();

        for (self.list[0..self.len], 0..) |e, i| {
            var h = self.hash(@intFromPtr(e.addr));
            while (self.index[h].version == self.version) h = (h + 1) & self.mask();
            self.index[h] = .{
                .version = self.version,
                .addr = @intFromPtr(e.addr),
                .idx = @intCast(i),
            };
        }
        self.indexed = true;
    }

    /// Pre-reserve so a timed path never allocates. Unlike a hash map there is
    /// no penalty for reserving more than you use
    pub fn ensureCapacity(self: *WriteSet, gpa: Allocator, n: usize) Allocator.Error!void {
        if (self.list.len < n) try self.growList(gpa, n);
        if (n > linear_max) {
            var bits: u6 = min_index_bits;
            while ((@as(usize, 1) << bits) < n * 4) bits += 1;
            const needed = @as(usize, 1) << bits;
            if (self.index.len < needed) {
                if (self.index.len != 0) gpa.free(self.index);
                self.index = try gpa.alloc(Slot, needed);
                @memset(self.index, Slot{
                    .addr = undefined,
                    .version = 0, // make sure this is a stale version
                    .idx = undefined,
                });
                self.index_bits = bits;
            }
        }
    }
};

/// A transaction context. One per thread.
///
/// `Tx` carries the thread-local state of an in-flight (or about-to-start)
/// transaction.
///
/// ```zig
/// var tx: Tx = .init(gpa, &stm, .ala);
/// defer tx.deinit();
/// _ = try tx.run(body, args);
/// _ = try tx.run(other_body, other_args);
/// // ... reused for as many transactions as the thread runs.
///  ```
pub const Tx = struct {
    stm: *Stm,
    allocator: Allocator,

    /// Snapshot of `stm.seq_lock` at the most recent point at which the
    /// transaction was known to be consistent.
    snapshot: u64 = 0,

    /// Append-only read log. Validation re-reads in order and checks for the
    /// value-equality the read barrier originally observed.
    reads: std.ArrayListUnmanaged(ReadLogEntry) = .empty,

    /// Indexed write set: maps `@intFromPtr(addr)` to the most recent value
    /// the transaction has decided to write. 
    writes: WriteSet = .empty,

    /// Selects the ordering semantics the transaction should provide.
    ///
    /// Use `.ala` (the default) unless you specifically need the stronger
    /// guarantee. SLA costs an extra read-set walk on every commit, including
    /// read-only commits, but is required for full single-lock-atomicity
    /// equivalence in programs with racy publication.
    comptime mode: PubSafety = build_options.zstm_pub_safety,
    pub const PubSafety = @TypeOf(build_options.zstm_pub_safety);

    /// Construct a transaction context. The allocator is used for the read
    /// log and write set; pass a long-lived per-thread allocator for best
    /// performance.
    ///
    /// mode: override the publication mode for this transaction. Default is set by -Dzstm_pub_safety which defaults to ALA
    pub fn init(allocator: Allocator, stm: *Stm, comptime mode: ?PubSafety) Tx {
        return .{
            .stm = stm,
            .allocator = allocator,
            .mode = mode orelse build_options.zstm_pub_safety,
        };
    }

    pub fn deinit(self: *Tx) void {
        self.reads.deinit(self.allocator);
        self.writes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Run `body(self, args...)` as a transaction, retrying as needed. Returns
    /// whatever `body` returns on a successful commit.
    ///
    /// `body` must be a function whose first argument is `*Tx`, that returns
    /// some error union `E!T` where `error.TxRetry` is in `E` (this is true
    /// automatically for inferred error sets that include any call to
    /// `tx.read`/`tx.write`).
    ///
    /// If `body` returns any error other than `error.TxRetry`, that error is
    /// reported to the caller and the transaction is rolled back (as if it
    /// had never run, modulo any non-transactional side effects in `body`).
    ///
    /// Side-effecty work (I/O, syscalls, allocations of state visible
    /// outside the transaction) inside `body` is the caller's responsibility:
    /// because the body may run multiple times, such effects will repeat.
    pub fn run(self: *Tx, comptime body: anytype, args: anytype) RunReturn(@TypeOf(body)) {
        const Ret = RunReturn(@TypeOf(body));
        comptime checkBody(@TypeOf(body));

        retry: while (true) {
            self.txBegin();

            const body_result = @call(.auto, body, .{self} ++ args);

            const value = body_result catch |err| {
                self.reset();
                if (err == error.TxRetry) continue :retry;
                return @as(Ret, err);
            };

            self.txCommit() catch |err| {
                self.reset();
                if (err == error.TxRetry) continue :retry;
                return @as(Ret, err);
            };

            self.reset();
            return value;
        }
    }

    /// Begin a transaction explicitly. Most users should call `run` instead;
    /// this is exposed for users who want full control over the retry loop.
    /// After `txBegin`, call `read`/`write` and finally either `txCommit` or
    /// (on a `TxRetry` error) `reset`.
    pub fn txBegin(self: *Tx) void {
        // start our snapshot at the last valid timestamp, we'll fail when we try and validate
        self.snapshot  = self.stm.seq_lock.load(.acquire) & ~@as(u64, 1);
    }

    /// Attempt to commit the transaction. Returns `error.TxRetry` if the
    /// transaction was invalidated by a concurrent writer.
    pub fn txCommit(self: *Tx) Error!void {
        comptime if (self.mode == .ala) {
            // in ala read-only doesn't serve as a publication point we can skip revalidation
            if (self.writes.count() == 0) {
                return;
            }
        };

        // Acquire the commit lock, moving seq_lock to odd indicating inconsistant state. 
        while (self.stm.seq_lock.cmpxchgStrong(self.snapshot, self.snapshot + 1, .acq_rel, .monotonic)) |new_time| {
            if (new_time & 1 == 0) {
                // time is consistant: validate and try again with new snapshot
                for (self.reads.items) |entry| {
                    if (@atomicLoad(Word, &entry.addr.raw, .monotonic) != entry.val)
                        return error.TxRetry;
                }
                self.snapshot = new_time;
            } else {
                // another writer is committing, wait for them to finish
                while(self.stm.seq_lock.load(.monotonic) == new_time) {
                    std.atomic.spinLoopHint();
                }
            }
        }

        for (self.writes.items()) |entry| {
            @atomicStore(Word, &entry.addr.raw, entry.val, .monotonic);
        }

        // Release the commit lock by moving to even time again
        self.stm.seq_lock.store(self.snapshot + 2, .release);
        self.snapshot += 2;
    }

    /// Reset transaction-local state. Call after a `TxRetry` (or any other)
    /// error, before either retrying with `txBegin` or abandoning the
    /// transaction. Capacity is retained so that repeated transactions in a
    /// loop do not re-allocate.
    pub fn reset(self: *Tx) void {
        self.reads.clearRetainingCapacity();
        self.writes.reset();
    }

    /// Transactional read. Returns `error.TxRetry` if validation fails.
    ///
    pub fn read(self: *Tx, addr: *const TxWord) Error!Word {
        //   1. Check the write set first to satisfy read-after-write.
        //   2. Read the location.
        //   3. If our snapshot is stale (a writer committed since we began),
        //      validate; on success update the snapshot and re-read.
        //   4. Log the (addr, value) pair for future validation.

        // preserve the illusion that the transaction's writes have already 
        // happened.
        if (self.writes.find(addr)) |buffered| return buffered;

        var val = @atomicLoad(Word, &addr.raw, .monotonic);

        // If the global lock has advanced, we may have read a value that is 
        // inconsistent with our earlier reads; fast forward until we get a consistent value or abort.
        while (self.snapshot != self.stm.seq_lock.load(.acquire)) {
            for (self.reads.items) |entry| {
                if (@atomicLoad(Word, &entry.addr.raw, .monotonic) != entry.val)
                    return error.TxRetry;
            }
            val = @atomicLoad(Word, &addr.raw, .monotonic);
            self.snapshot = 
        }

        try self.reads.append(self.allocator, .{ .addr = addr, .val = val });
        return val;
    }

    /// Transactional write. the underlying memory is not touched until commit.
    pub fn write(self: *Tx, addr: *TxWord, val: Word) Error!void {
        try self.writes.insert(self.allocator, addr, val);
    }

    /// Type-safe read of a multi-word `TxCell`. 
    /// Returns `error.TxRetry` if validation fails mid-read.
    pub fn readCell(self: *Tx, cell: anytype) Error!std.meta.Child(@TypeOf(cell)).Payload {
        const CellType = std.meta.Child(@TypeOf(cell));
        const T = CellType.Payload;
        var words_copy: [CellType.NumWords]Word = undefined;

        // if a writer commits while we are partway through reading these words,
        // `self.read` will detect the global lock advancement, re-validate, 
        // and throw TxRetry.
        for (&cell.words, 0..) |*w, i| {
            words_copy[i] = try self.read(w);
        }

        var result: T = undefined;
        const result_bytes = std.mem.asBytes(&result);
        const words_bytes = std.mem.sliceAsBytes(&words_copy);
        @memcpy(result_bytes, words_bytes[0..result_bytes.len]);
        return result;
    }

    /// Type-safe write to a multi-word `TxCell`.
    /// Buffers the new struct state in the redo log.
    pub fn writeCell(self: *Tx, cell: anytype, value: anytype) Error!void {
        const CellType = std.meta.Child(@TypeOf(cell));
        const T = CellType.Payload;
        if (@TypeOf(value) != T) @compileError("type mismatch in writeCell");

        var words_copy = @as([]Word, @splat(CellType.NumWords));
        const value_bytes = std.mem.asBytes(&value);
        const words_bytes = std.mem.sliceAsBytes(&words_copy);
        @memcpy(words_bytes[0..value_bytes.len], value_bytes);

        for (&cell.words, 0..) |*w, i| {
            try self.write(w, words_copy[i]);
        }
    }

    /// Run a consistent-snapshot validation of the read log. 
    /// This errors with TxRetry if one of our reads has been written to
    inline fn validate(self: *Tx) Error!bool {
        // Walk the read log, checking each location still holds the value
        // we originally saw. Any mismatch means the transaction is doomed.
        
    }
};

/// Extract the return type of `body` (which is `Error!T` for some `T`) from a
/// runtime point of view, so that `run` can be declared returning the same
/// type.
fn RunReturn(comptime BodyFn: type) type {
    const fn_info = @typeInfo(BodyFn).@"fn";
    return fn_info.return_type orelse @compileError("transaction body must be a non-generic function");
}

/// Compile-time check on the shape of a transaction body.
fn checkBody(comptime BodyFn: type) void {
    const fn_info = @typeInfo(BodyFn).@"fn";
    if (fn_info.params.len < 1)
        @compileError("transaction body must take *Tx as its first argument");
    const first = fn_info.params[0].type orelse return; 
    if (first != *Tx)
        @compileError("transaction body's first argument must be *Tx, got " ++ @typeName(first));
    const ret = fn_info.return_type orelse return;
    const ret_info = @typeInfo(ret);
    if (ret_info != .error_union)
        @compileError("transaction body must return an error union because transcations may fail, got " ++ @typeName(ret));
}

// =============================================================================
// Tests
// =============================================================================

test "single-threaded read/write produces correct value" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(0);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) Error!Word {
            const v = try t.read(addr);
            try t.write(addr, v + 1);
            return v;
        }
    };

    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const old = try tx.run(Body.run, .{&x});
        try std.testing.expectEqual(@as(Word, i), old);
    }
    try std.testing.expectEqual(@as(Word, 10), x.unsafeLoad());
}

test "read-after-write returns buffered value" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(7);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) Error!Word {
            const before = try t.read(addr);
            try t.write(addr, before + 100);
            // Within the same transaction, the read should see the buffered
            // value, not the underlying memory (which is still 7 at this
            // point because writeback hasn't run).
            const after = try t.read(addr);
            return after;
        }
    };

    const observed = try tx.run(Body.run, .{&x});
    try std.testing.expectEqual(@as(Word, 107), observed);
    try std.testing.expectEqual(@as(Word, 107), x.unsafeLoad());
}

test "read-only transaction commits without acquiring lock (ALA)" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(42);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    const initial_lock = stm.seq_lock.load(.acquire);

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) Error!Word {
            return try t.read(addr);
        }
    };

    const v = try tx.run(Body.run, .{&x});
    try std.testing.expectEqual(@as(Word, 42), v);
    // Read-only ALA commit must not advance the global lock.
    try std.testing.expectEqual(initial_lock, stm.seq_lock.load(.acquire));
}

test "writer commit advances seq_lock by exactly 2" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(0);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    const before = stm.seq_lock.load(.acquire);

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) Error!void {
            try t.write(addr, 1);
        }
    };

    try tx.run(Body.run, .{&x});

    const after = stm.seq_lock.load(.acquire);
    try std.testing.expectEqual(before + 2, after);
    try std.testing.expectEqual(@as(u64, 0), after & 1); // lock is unheld
}

test "manually-thrown TxRetry causes retry then succeeds" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(0);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    var attempts: u32 = 0;

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord, attempts_ptr: *u32) Error!Word {
            attempts_ptr.* += 1;
            const v = try t.read(addr);
            if (attempts_ptr.* < 3) return error.TxRetry;
            try t.write(addr, v + 1);
            return v;
        }
    };

    const old = try tx.run(Body.run, .{ &x, &attempts });
    try std.testing.expectEqual(@as(Word, 0), old);
    try std.testing.expectEqual(@as(Word, 1), x.unsafeLoad());
    try std.testing.expectEqual(@as(u32, 3), attempts);
}

test "non-retry error from body propagates" {
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(0);

    var tx: Tx = .init(allocator, &stm, .ala);
    defer tx.deinit();

    const MyErr = error{Boom} || Error;

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) MyErr!void {
            _ = try t.read(addr);
            return error.Boom;
        }
    };

    try std.testing.expectError(error.Boom, tx.run(Body.run, .{&x}));
    // Aborted transaction must not have changed shared state.
    try std.testing.expectEqual(@as(Word, 0), x.unsafeLoad());
}

test "concurrent increments converge to the correct count" {
    if (builtin.single_threaded) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var counter: TxWord = .init(0);

    const num_threads = 4;
    const per_thread = 2_000;

    const Worker = struct {
        fn run(stm_ptr: *Stm, c: *TxWord, n: usize, gpa: Allocator) void {
            var tx: Tx = .init(gpa, stm_ptr, .ala);
            defer tx.deinit();

            const Body = struct {
                fn run(t: *Tx, addr: *TxWord) Error!void {
                    const v = try t.read(addr);
                    try t.write(addr, v + 1);
                }
            };

            var i: usize = 0;
            while (i < n) : (i += 1) {
                tx.run(Body.run, .{c}) catch |err| {
                    std.debug.panic("transaction failed: {}", .{err});
                };
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Worker.run, .{ &stm, &counter, per_thread, allocator });
    }
    for (threads) |t| t.join();

    try std.testing.expectEqual(@as(Word, num_threads * per_thread), counter.unsafeLoad());
    // Every successful writer commit advances the lock by 2; final value
    // equals 2 * total commits, which equals 2 * num_threads * per_thread.
    try std.testing.expectEqual(@as(u64, 2 * num_threads * per_thread), stm.seq_lock.load(.acquire));
}

test "concurrent two-variable swap maintains sum invariant" {
    if (builtin.single_threaded) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    // Invariant: a + b == 1000 always.
    var a: TxWord = .init(500);
    var b: TxWord = .init(500);

    const num_threads = 4;
    const per_thread = 1_000;

    const Worker = struct {
        fn run(
            stm_ptr: *Stm,
            ax: *TxWord,
            bx: *TxWord,
            n: usize,
            seed: u64,
            gpa: Allocator,
        ) void {
            var tx: Tx = .init(gpa, stm_ptr, .ala);
            defer tx.deinit();

            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();

            const Body = struct {
                fn run(t: *Tx, av: *TxWord, bv: *TxWord, amount: Word) Error!void {
                    const av_old = try t.read(av);
                    const bv_old = try t.read(bv);
                    try t.write(av, av_old -% amount);
                    try t.write(bv, bv_old +% amount);
                }
            };

            var i: usize = 0;
            while (i < n) : (i += 1) {
                const amt = rand.uintLessThan(Word, 100);
                tx.run(Body.run, .{ ax, bx, amt }) catch unreachable;
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (&threads, 0..) |*t, idx| {
        t.* = try std.Thread.spawn(.{}, Worker.run, .{
            &stm, &a, &b, per_thread, @as(u64, 0xC0FFEE) +% idx, allocator,
        });
    }
    for (threads) |t| t.join();

    // After everyone is done, read the final pair atomically inside one
    // transaction and verify the invariant. Using a transaction here matters:
    // it guarantees the two reads are mutually consistent.
    var checker: Tx = .init(allocator, &stm, .ala);
    defer checker.deinit();

    const Read = struct {
        fn run(t: *Tx, av: *TxWord, bv: *TxWord) Error![2]Word {
            return .{ try t.read(av), try t.read(bv) };
        }
    };

    const pair = try checker.run(Read.run, .{ &a, &b });
    try std.testing.expectEqual(@as(Word, 1000), pair[0] +% pair[1]);
}

test "SLA mode also commits read-only with sequence lock unchanged" {
    // Even in SLA mode, a read-only commit only validates; it must NOT
    // increment the global lock.
    const allocator = std.testing.allocator;

    var stm: Stm = .init;
    var x: TxWord = .init(99);

    var tx: Tx = .init(allocator, &stm, .sla);
    defer tx.deinit();

    const before = stm.seq_lock.load(.acquire);

    const Body = struct {
        fn run(t: *Tx, addr: *TxWord) Error!Word {
            return try t.read(addr);
        }
    };

    const v = try tx.run(Body.run, .{&x});
    try std.testing.expectEqual(@as(Word, 99), v);
    try std.testing.expectEqual(before, stm.seq_lock.load(.acquire));
}
