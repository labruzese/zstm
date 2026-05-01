const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

const zstm = @import("zstm");

test "TxVar init stores value" {
    const var_int = zstm.TxVar(i32).init(42);
    try expectEqual(var_int.raw, 42);
}
