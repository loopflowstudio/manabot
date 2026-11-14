---
requires: working implementation and representative consumer workloads
produces: measured optimization plan in the existing scratch design
---
Measure the engine's real bottlenecks and plan worthwhile behavior-preserving improvements.

1. Select bounded ordinary training/play/search workloads and the established
   engine budget harness. Ordinary play rarely builds a big board, so also name
   the game states where the changed code's cost could grow badly (large
   duplicative boards, large diverse boards) and run them through
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)); add a state when
   none has the shape. Fix hardware, build, workload identity and measurement
   limits before comparing. Never create artificial resource pressure or a stress
   loop; a large constructed board run once is a workload, not induced load.
   Use controlled fixtures for scheduling/resource-monitoring behavior.
2. Retain baseline time, throughput and relevant memory/allocation evidence with
   behavior/decision digests. Profile within the declared limits; select measured
   bottlenecks rather than optimizing by inspection alone.
3. Write the active phase's implementation, deletion targets and focused checks
   into the existing design. Preserve complete choices, observation capacity
   semantics, native skipping and cache/debug invariants. Same-runner comparisons
   own speed claims; unrelated historical wall times do not.
4. State what useful improvement is feasible and which tradeoffs need review.
   An already efficient implementation can justify no change with evidence.
   Merely passing the 10% regression allowance is insufficient. Leave unavailable
   hardware measurements to gate/CI with the exact command; make no speed claim.
