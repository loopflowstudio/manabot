---
requires: a working engine change and the workloads that exercise it
produces: a measured optimization plan in `scratch/plan.md`
action_style: procedural
---
Measure where the engine spends time on this change's workloads, and plan only improvements backed by a profile.

1. Run ordinary training, play and search workloads and the engine budget
   harness. Ordinary play rarely builds a big board, so name the states where
   the changed code's cost could grow badly (large duplicative boards, large
   diverse ones) and run them with
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)); add a state
   when none has the shape. A large constructed board run once is a workload;
   never induce resource pressure or run a stress loop.
2. Fix hardware, build, workload identity and limits before comparing. Record
   baseline time, throughput, memory or allocation counts and behavior
   digests in `scratch/plan.md`. Profile before planning; pick bottlenecks
   from the profile, not from reading code.
3. Write each improvement into `scratch/plan.md` as a slice: its hypothesis,
   what it deletes and its focused check. Complete choices, observation
   capacity semantics, native skipping and cache and debug invariants must
   hold. Speed claims require same-machine numbers.
4. If the implementation is already efficient, record the numbers and plan no
   slices; passing the 10% regression allowance alone doesn't prove that.
   Leave measurements needing unavailable hardware as an exact deferred
   command and make no speed claim.
