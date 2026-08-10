const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const opts = .{ .target = target, .optimize = optimize };

    const zstm = b.addModule("zstm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zbench = b.dependency("zbench", opts).module("zbench");
    const bench = b.createModule(.{
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench.addImport("zstm", zstm);
    bench.addImport("zbench", zbench);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench,
    });

    b.installArtifact(bench_exe);

    const bench_step = b.step("bench", "run the benchmarks");
    const bench_run = b.addRunArtifact(bench_exe);
    bench_step.dependOn(&bench_run.step);

    if (b.args) |args| {
        bench_run.addArgs(args);
    }

    const tests = b.addTest(.{
        .root_module = zstm,
    });
    const tests_run = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&tests_run.step);
}
