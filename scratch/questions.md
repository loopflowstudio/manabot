# Questions

- For slice 1, retained external dictionaries can contain numeric features in
  masked object slots. Preserve those values by selecting nonzero rows on the
  dense path until native ownership can certify zero padding or provide spans.

- `managym/examples/bench_observation` panics on base `4127aa973f`
  (`bench_observation.rs:43`, `unwrap` on a missing PriorityPassPriority
  action). Encode-only attribution needs it repaired; that is outside this
  plan. A separate small fix?
- Workload uses v11 on the TLA frontier's capacities, but the frontier
  recipe itself runs v9 (the deprecated fork). Assumed: v11 is the workload
  to optimize, and v9 keeps its dense path unchanged (it is never extended).
- Every timing was taken with machine load at 140–200 on 16 cores (other
  checkouts compiling). Only CPU-time shares, counts, bytes and exactness
  are trusted. End-to-end wall and RSS numbers are deferred (plan.md).

- Slice 2 uses live engine frontiers and emitted vector steps as the producers
  of certified CSR. A writable vector buffer or a Python observation restored
  from JSON cannot establish that provenance. Keep the public extent scanner
  for external dictionaries; the next retention slice owns the collector switch.
