---
requires: a reviewed card batch and the new-card stress shapes named in `scratch/plan.md`
produces: new-card performance measurements in `scratch/plan.md` and routing JSON selecting `clean` or `optimize`
action_style: procedural
---
Check whether the new cards and this branch's diff introduced a performance regression or superlinear scaling, and route to `clean` or `optimize`.

Stay strictly scoped to **this branch's new cards and new/modified code paths**.
Do not start a general engine optimization campaign on a card branch.

1. **Measure new-card stress states.** Run
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml`
   ([stress states](../../docs/development.md#stress-states)) on the
   large-board shapes the new cards touch (adding a compact recipe from the new
   cards when none fits). Inspect growth across sizes (e.g. `4,8,16`) for
   superlinear `calls`, `states` or retired instructions (`/usr/bin/time -l`)
   and sample any blowup (`/usr/bin/sample` or `perf`). Never induce artificial
   load or run a stress loop.
2. **No-regression check vs. branch base.** Build uninstrumented release
   `bench_steps` and `bench_search` (without `--features benchmark`) for base
   and candidate on the same machine, confirming identical behavior digests.
   Use load-invariant retired instructions (`/usr/bin/time -l` or `perf stat`)
   rather than noisy wall time on a contended host. If retired instructions
   regressed by more than ~2% on `bench_steps` or `bench_search`, take a CPU
   profile to confirm whether this branch's diff caused it.
3. **Separate branch fixes from global engine work.**
   - If the profile shows **overhead introduced by this branch** or
     **superlinear scaling in the new cards' paths**, write one concrete,
     profile-backed fix slice into `scratch/plan.md` (hypothesis, exact paths,
     focused debug invariant checks and target `bench_states`/`bench_steps`
     checks) and route `optimize`.
   - If you spot a **pre-existing global engine hotspot** (e.g. shared
     `layers::Context` evaluation or general `PaymentSearch` hashing already
     present on base) that this branch did not regress, record it as a
     follow-up candidate for the `engine-performance` Wave instead of rewriting
     shared hot paths on a parallel card branch.
4. Record all measured numbers, digests and any deferred same-runner CI check
   in `scratch/plan.md` without editing engine code here (adding a missing
   `bench_states` recipe is fine if committed cleanly).

Return `{"path":"optimize"}` when this branch regressed `bench_steps` or
`bench_search` retired instructions (>~2% attributable to this diff) or a new
card's `bench_states` shape scales superlinearly with a profile-backed fix.
Otherwise return `{"path":"clean"}`.
