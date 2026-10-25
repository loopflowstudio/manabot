# Shared evaluation follow-up — 2026-10-24

Accepted by Jack Heart, 2026-10-24:

- After interruption at 40/100 complete valid deals, explicit continuation runs
  the missing 60 under one logical evaluation. Keep all attempts and count each
  block once. Fresh full repeats remain distinct. No automatic retries, native
  state recovery or mixing CPU/CUDA/numerical contracts.
- Local small-checkpoint versus heuristic checks share evaluation semantics
  without training, scheduling, remote credentials or callback transport.
- Keep compute lifetime and accounting internal; no public allocation creation
  is required. Checkpoint pairs at growing model sizes are the primary workload.
- Aim for at least two useful hours of eligible queued work before starting a
  cloud eval server, estimated for its measured host/model mix. Release idle
  compute when work finishes. This is no minimum billed lifetime. No urgent
  override or maximum queue wait has been selected.

Implementation assumptions, reversible and scoped to the supplied checkout:

- Preserve the initial attempt in the existing panel directory; subsequent
  attempts live beneath `attempts/<sequence>/`. `evaluation_status` derives
  progress from immutable block receipts and retained attempts. A UUID separates
  fresh evaluations. No second queue or progress database.
- A filesystem lock fences the logical writer. Managed RPC execution holds an
  additional attempt claim; continuation seals old attempts against delayed RPCs.
  These claims are for a shared local filesystem, not cross-host object storage.
  Moving evidence requires moving the entire evaluation directory; checkpoint
  path changes require the same digest and byte length.
- The same complete-block validator owns final statistics and continuation.
  Incomplete blocks retry whole; conflicting duplicates fail closed. Deadline
  and instrumentation may change; compute settings and scientific limits may not.
  Frozen pre-continuation results stay readable but are not silently migrated.
- Per-block replay and immutable shards are published before another block can
  erase progress. The aggregate trace remains diagnostic/calibration evidence;
  reports use row-referenced shards to avoid counting both copies.

Upstream gap rechecked on installed `etude-sdk==0.2.4`: job/cohort rental ceilings
still are not measured useful-work estimates. Etude's launch scheduler owns the
missing >=7200 useful-second admission gate for new rentals; Manabot needs
host/model workload measurements for integration. Existing idle-session release
remains correct. The design records the inspected symbols and exact gap. This is
still-required compute-policy work, separate from the continuation deliverable;
no alternate durable queue or provider helper is selected.

Unselected: actual model mix, measured current evaluation throughput, memory,
host, budget, maximum wait, latency and numerical/performance thresholds. A few
experiments per week suggests on-demand as a hypothesis, not a purchase.
Historical hardware rates, 3–5 minute startup and probe012's roughly 28 games/min
are illustrations, never runtime estimates for this policy. Whole-job cost,
CUDA parity and an end-to-end gain remain unmeasured. Broad suites belong to gate;
this implementation uses focused headless fixtures only.

The published runtime/report foundation is PR #370 at
`79506de9cae0182653e251767d35ffba5427c066`. Earlier uncommitted questions are
preserved in Git blob `d8c8916accba563add0badbfd2cf30970958ca91`, design in
`1dcb52e5bbdf3747342bc89b95d0cb17e519cd40`, and evidence in
`7bae0e735e83f08273f093dbea10422ed6f416db`. Inherited MTG-174/176/180/182
assumptions remain at `07a091bca:scratch/questions.md`; upstream MTG-183/card
history remains at `884249024:scratch/questions.md`. None selects work here.

Concurrent report demo edits and `scratch/eval-perf-demo.md` belong to the
ongoing conversation's report work and are preserved, not reimplemented here.
