---
requires: a working implementation and the workloads that exercise it
produces: a measured optimization plan in `scratch/plan.md`
---
Measure where the engine actually spends time on this change's workloads, and plan the improvements worth making.

1. Pick bounded ordinary training, play and search workloads and the engine
   budget harness. Ordinary play rarely builds a big board, so also name the
   states where the changed code's cost could grow badly (large duplicative
   boards, large diverse ones) and run them through
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)); add a state
   when none has the shape. A large constructed board run once is a workload;
   never create resource pressure or a stress loop.
2. Fix hardware, build, workload identity and limits before comparing. Record
   baseline time, throughput, memory or allocation counts and behavior
   digests. Profile, and pick bottlenecks from the profile, not from reading
   code.
3. Write each improvement into the plan as a slice: its hypothesis, what it
   deletes and its focused check. Complete choices, observation capacity
   semantics, native skipping and cache and debug invariants hold.
   Same-runner comparisons own speed claims.
4. When the implementation is already efficient, say so with the numbers and
   plan no slices; passing the 10% regression allowance alone doesn't show
   that. Leave measurements needing unavailable hardware to gate or CI with the
   exact command, and claim no speed.
