const std = @import("std");
const zstm = @import("zstm");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    var stm: zstm.Stm = .init;
    var counter: zstm.TxWord = .init(0);

    const num_threads = 8;
    const per_thread: usize = 50_000;

    const Worker = struct {
        fn run(stm_ptr: *zstm.Stm, c: *zstm.TxWord, n: usize, allocator: std.mem.Allocator) void {
            var tx: zstm.Tx = .init(allocator, stm_ptr, .ala);
            defer tx.deinit();

            const Body = struct {
                fn run(t: *zstm.Tx, addr: *zstm.TxWord) zstm.Error!void {
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
        t.* = try std.Thread.spawn(.{}, Worker.run, .{ &stm, &counter, per_thread, gpa });
    }
    for (threads) |t| t.join();

    const final = counter.unsafeLoad();
    const expected: usize = num_threads * per_thread;

    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;

    try out.print(
        "zstm demo: {d} threads x {d} txns -> counter = {d} (expected {d})\n",
        .{ num_threads, per_thread, final, expected },
    );
    try out.print("seq_lock = {d} (expected {d})\n", .{
        stm.seq_lock.load(.acquire),
        2 * expected,
    });
    try out.flush();

    if (final != expected) std.process.exit(1);
}
