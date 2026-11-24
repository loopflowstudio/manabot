# Questions

- Repairing `managym/examples/bench_observation.rs` could isolate encode cost.
  The base `4127aa973f` fixture panics at line 43 when it assumes a
  PassPriority action is present. Keep this separate from the transport slices.
- Use v11 with the small TLA frontier setup for the paired probes. This
  assumes the v9 recipe remains a compatibility workload.
- Use native-endian packed records in logical field order so each table keeps
  one immutable values buffer and its existing CSR offsets.
- Start slice 4 with native spans as the sole owner of v11 legal action fields.
  Store behavior probabilities by their original action coordinates and keep
  the compact dense action path for legacy and external rows.
