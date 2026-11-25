---
requires: a working engine change and the workloads that exercise it
produces: a measured optimization plan in `scratch/plan.md`
action_style: procedural
---
Measure where the engine spends time on this change's workloads, and plan only improvements backed by a profile.

1. Compare candidate and base on the same build CI times: uninstrumented
   release `bench_steps` and `bench_search` (without `--features benchmark`),
   recording behavior digests and wall time. On a contended host, wall time
   is noise—measure load-invariant instructions retired (`/usr/bin/time -l`
   or `perf stat`) and take a CPU sampling profile (`sample` or `perf`)
   instead of deferring speed to CI.
2. Run `cargo run --release --features benchmark --example bench_states
   --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)) on the
   large-board shapes the change touches, adding one when none fits. Never
   induce resource pressure or run a stress loop. Matching `bench_states`
   counters alone never clears per-step speed.
3. Profile before planning; pick bottlenecks from the profile, not from
   reading code. Write each improvement into `scratch/plan.md` as a slice:
   its hypothesis, what it deletes and its focused check, preserving complete
   choices, observation capacities, native skipping and cache/debug invariants.
4. If the profile leaves no bottleneck worth attacking, record the numbers
   and plan no slices; passing the 10% budget floor alone doesn't prove that.
   Leave unavailable hardware as an exact deferred command with no speed claim.
