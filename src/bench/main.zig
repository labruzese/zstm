//! zbench driver for the zstm microbenchmarks.

const std = @import("std");
const zstm = @import("zstm");
const harness = @import("harness.zig");
const workloads = @import("workloads.zig");
