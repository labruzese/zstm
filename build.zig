const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zstm = b.addModule("zstm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const zstm_check = b.addExecutable(.{
        .name = "zstm",
        .root_module = zstm,
    });
    const check = b.step("check", "Check if zstm compiles");
    check.dependOn(&zstm_check.step);

    const bench_optimize = b.option(
        std.builtin.OptimizeMode,
        "bench-optimize",
        "optimize mode for the benchmarks and the copy of zstm they measure (default: ReleaseFast)",
    ) orelse .ReleaseFast;

    const zstm_measured = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = bench_optimize,
    });

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench/main.zig"),
        .target = target,
        .optimize = bench_optimize,
    });
    bench_mod.addImport("zstm", zstm_measured);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });

    // `zig build` drops it in zig-out/bin/bench, so it can also be run
    // directly, under perf, or against a saved baseline later.
    b.installArtifact(bench_exe);

    const bench_run = b.addRunArtifact(bench_exe);
    // Benchmarks are the side effect; never let the build system decide the
    // result is cached and skip the run.
    bench_run.has_side_effects = true;
    bench_run.stdio = .inherit;
    if (b.args) |args| bench_run.addArgs(args);

    const bench_step = b.step("bench", "run the benchmarks (pass options after --)");
    bench_step.dependOn(&bench_run.step);

    const run_step = b.step("run", "alias for `bench`");
    run_step.dependOn(&bench_run.step);

    // Tests: the library's, plus the benchmark driver's own

    const test_step = b.step("test", "run the library and benchmark-driver tests");

    const lib_tests = b.addTest(.{ .root_module = zstm });
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);

    // built separately from `bench_mod`: these check the driver's statistics
    // and csv handling, which want safety checks on rather than ReleaseFast.
    const bench_test_mod = b.createModule(.{
        .root_source_file = b.path("src/bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_test_mod.addImport("zstm", zstm);

    const bench_tests = b.addTest(.{ .root_module = bench_test_mod });
    test_step.dependOn(&b.addRunArtifact(bench_tests).step);
}
