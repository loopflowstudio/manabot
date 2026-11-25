---
requires: a reviewed card batch and the new-card stress shapes named in `scratch/plan.md`
produces: new-card performance check numbers in `scratch/plan.md` and routing JSON selecting `clean` or `optimize`
action_style: procedural
---
Run a fast performance check scoped to this branch's new cards and diff, then route to `clean` or `optimize`.

Leave CPU sampling and fix planning to `engine-measure` on the `optimize` path;
this step only decides whether optimization is needed and bounds its scope.

1. **Check new-card stress states.** Run
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)) on the
   large-board shapes the new cards touch (adding a compact recipe from the new
   cards when none fits). Check growth across sizes (e.g. `4,8,16`) for
   superlinear `calls`, `states` or retired instructions (`/usr/bin/time -l`).
   Never induce artificial load or run a stress loop.
2. **No-regression check vs. branch base.** Build uninstrumented release
   `bench_steps` and `bench_search` (without `--features benchmark`) for base
   and candidate on the same machine, confirming identical behavior digests.
   Use load-invariant retired instructions (`/usr/bin/time -l` or `perf stat`)
   rather than noisy wall time on a contended host.
3. **Record numbers and scope in `scratch/plan.md`.** Write the measured
   numbers, digests and—if routing `optimize`—a short scope note naming only
   the regressed benchmark or superlinear new-card shape for `engine-measure`
   to profile and fix (directing pre-existing global hotspots to
   `engine-performance` Tasks instead of rewriting shared hot paths on a card
   branch). Don't edit engine code here (adding a missing `bench_states` recipe
   is fine if committed cleanly).

Return `{"path":"optimize"}` when this branch regressed `bench_steps` or
`bench_search` retired instructions (>~2% vs. base) or a new card's
`bench_states` shape scales superlinearly. Otherwise return `{"path":"clean"}`.
