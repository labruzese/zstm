## zstm: NOrec Software Transactional Memory.

This module implements the NOrec algorithm as described by Dalessandro, 
Spear, and Scott
(PPoPP'10, "NOrec: Streamlining STM by Abolishing Ownership Records").

NOrec combines the following ideas:

  1. A single global sequence lock 
  2. A redo log 
  3. Value-based validation 

Properties this implementation provides:

  - **Livelock freedom.**
  - **Privatization safety.**
  - **Publication safety (ALA by default, optional SLA).** SLA requires 
    one extra validation at commit time and is selectable via comptime `Tx.PubSafety`.
  - **Opacity.** A doomed transaction never observes inconsistent state.

Properties this implementation does NOT provide:

  - Transaction nesting
  - Hardware integration
  - Memory reclamation safety for transactionally freed pointers.

## Usage

```zig
var stm: zstm.Stm = .init;
var counter: zstm.TxWord = .init(0);

var tx: zstm.Tx = .init(allocator, &stm, .ala);
defer tx.deinit();

const old = try tx.run(struct {
    fn body(t: *zstm.Tx, c: *zstm.TxVar) zstm.Error!zstm.Word {
        const v = try t.read(c);
        try t.write(c, v + 1);
        return v;
    }
}.body, .{&counter});
```

Each thread should own its own `Tx`. The `Stm` is shared across threads.
