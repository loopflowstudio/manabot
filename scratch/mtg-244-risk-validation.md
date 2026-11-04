# MTG-244 performance and learning risk validation

Jack Heart directed this Task on 2026-11-04: "ok. lets make suer we measuer
these risks and attempt to ameliorate them". This adds measurement and mitigation
to the current delivery, not a separate Task. Keep the live Graph and existing
review boundary. No merge or training experiment launch is authorized.

## Deliverable before presenting the change as ready for review

Produce a compact evidence table: risk, baseline, changed result, mitigation,
remeasured result and remaining limitation. Name exact revisions, configurations,
hardware, workloads, counts and commands. Distinguish measured results from
proposed work. Correctness/sanity checks alone do not establish performance or
learning quality. Extend the implementation if findings require repairs; do not
defer all risk evidence to Gate after the interactive demo. Gate retains its
configured full checks and same-runner CI authority.

Use bounded representative workloads and controlled fixtures under TESTING.md;
never induce resource pressure, run stress tools or invent artificial load.
Avoid competing benchmarks on a busy host; use the existing CI measurement path
where appropriate. If a measurement is unavailable, name precisely what evidence
is missing and retain the limitation for review rather than claiming success.

## Engine and search cost

- Preserve and report original TLA workloads separately from the added layer
  workload. Compare base and change on the same runner with identical workload
  definitions. Include a small representative dependency-bearing position, not
  only an anthem board covered by the power/toughness fast path. New semantics
  may make base correctness incomparable; label such timing comparisons and use
  counters plus before/after mitigation results for the new workload.
- Record ordinary step throughput and search total and slow-decision timings
  (distribution/tail with sample count, no unsupported percentile claims), full
  action/score digests and simulation counts. Fewer legal choices or shorter
  trajectories are not speedups. Explain rule-driven digest changes.
- Attribute cache rebuilds, context clones, dependency comparisons and allocation
  cost on representative decisions. Inspect whole-board invalidation on mana
  payment and tapping. Reduce unnecessary rebuilds and repeated dependency work
  where measurements support it, preserving all real dependency reads.
- Keep the debug recomputation oracle and directed event/clone witnesses after
  optimization. Observe the existing 10% Engine Budget; do not adopt a new
  baseline merely to erase a measured regression. Retain the old-vs-new numbers
  when changed digests require the normal reference update procedure.

## Observation and model cost

- Measure native/Python encoding, observation and rollout-buffer bytes, policy
  inference latency/throughput, bounded forward/backward cost and peak memory.
  Use representative configured batch/model sizes and disclose dtype, device,
  thread count and accelerator timing synchronization as applicable.
- On shared representable states, compare old input and v8 under otherwise
  matched settings. Separately measure v8 ordinary and layer-program states;
  do not conflate a larger world, new model parameters and engine cost.
- Inspect the 4096-row effect-detail capacity, scans of padding, per-record
  Python work/device synchronization and recurrent reduction. Prefer processing
  live tokens, bounded program-local reductions and reuse when semantics allow.
  Capacity reductions must follow measured occupancy and preserve complete
  observations with explicit overflow behavior; no truncation for speed.

## Representation and learning validity

- Beyond finite forward/backward, test paired states differing in strategic
  meaning: dependency/timestamp order, source and target joins, grant program
  order, numeric values, duplicate grants and pending programs after source
  removal. Verify representation separation and the intended gradient path.
- Verify invariance to irrelevant padding and independent record/object order
  with references remapped; preserve sensitivity where order is meaningful.
  Check program boundaries and ownership: a GRU across unrelated fact rows must
  not accidentally make one program depend on another program or its padding.
  Exercise every supported v8 model path with new local facts, not just shared
  effect tokens. Include meaningful magnitude/normalization cases.
- Keep native/Python/text/token parity, actual legal actions and resolved effects
  in agreement. Preserve valid old-version checkpoint shapes/omissions and
  explicit early rejection of unrepresentable states. Exercise real training
  and evaluation configuration selection of v8 so v6 defaults cannot cause an
  unexplained mid-run failure; do not silently upgrade saved configurations.
- These checks measure representation correctness and computational cost, not
  sample efficiency, win rate or generalization. Report those learning outcomes
  as unmeasured. A training experiment remains a separate authorization; propose
  a small controlled comparison only if needed, without launching one.

Mitigate concrete findings, rerun the affected checks and measurements, and show
the residual tradeoffs in the demo/PR. Preserve frozen world and experiment
evidence; changed rules do not make old and new learning curves interchangeable.

## Measured local-fact mitigation (2026-11-04)

A concrete representation defect was reproduced: the first program influenced
another holder's program through a GRU spanning all 512 fact rows. The new
isolation assertion fails on the pre-mitigation code (5/16 output elements,
max absolute difference 0.3391394). FactEncoder now validates ordinal boundaries
and owner/provenance consistency, batches each live program as an independent
packed sequence, and bypasses recurrence when no program is present. Independent
program order, remapped holders, duplicate multiplicity and inserted NaN padding
are invariant; token order and signed numeric values remain distinguishable.
Tests exercise gradients and rows/typed_pool/typed_gru inference with captured
programs after source departure. This is representation evidence, not learning
efficacy.

Bounded component measurement, same Apple M4 Max, macOS 26.0.1, Torch 2.10.0,
CPU float32, one thread, batch 16, width 64, 32 objects, **512 facts in both
versions** (explicitly matching capacity, not v7's default). Two warmups then ten
samples per row; processes run sequentially with no concurrent task-owned build
or benchmark. Ordinary fixture has one counter fact; program fixture has two
six-integer programs per observation. No optimizer step, game or training run.
Numbers are median milliseconds; raw receipts also retain total, maximum and
observations/second. These small samples do not establish percentiles.

| Risk / workload | Retained v7 comparator | Original v8 | Mitigation | Remeasured v8 | Remaining limitation |
| --- | ---: | ---: | --- | ---: | --- |
| Ordinary inference | 3.94 | 15.36 | Skip absent programs | 3.94 | Full policy/encoding still unmeasured |
| Ordinary forward/backward | 6.42 | 52.04 | Skip absent programs | 6.42 | No update or learning claim |
| Two-program inference | N/A: unrepresentable | 15.61 | Pack live independent programs | 4.76 | CPU component fixture only |
| Two-program forward/backward | N/A | 47.84 | Same | 10.29 | Ten-sample max was 18.32 ms |
| Process peak RSS, whole harness | Included in each process | 308,150,272 B | Same | 273,973,248 B | Includes imports and all cases; not per-operation allocator peak |

The unchanged v7 control remeasured at 3.62 / 6.06 ms (inference / fwd+bwd),
illustrating ordinary host variation. Matched input tensors use 524,288 bytes;
v8 fact parameters remain 3,268,096 bytes versus v7's 3,166,720. Packing does not
reduce tensor capacity. Full-model latency/memory, shared-detail occupancy and engine/search attribution remain
required before demo/PR; Gate retains full suites and same-runner Engine Budget.
Default allocated observation arrays total 133,616 B (v6), 146,928 B
(v7/128 facts), and 208,368 B (v8/512 facts): v8 adds 61,440 B over v7,
43,008 B from shared details and 18,432 B from local facts. This is measured by
summing `array.nbytes` from `ObservationSpace(ObservationConfig(policy_input_version=v)).encoder.allocate()`;
a rollout storing N such observations uses N times this payload, before other
rollout fields/allocator overhead. No capacity was reduced or truncated.
In particular the existing anthem-only added benchmark does **not** yet satisfy
the requested dependency-bearing workload, and mana-pool invalidation remains
unoptimized. Do not treat this component improvement as satisfying those checks.

Reproduce: `PYTHONPATH=. uv run --no-sync python .cache/mtg-244-risk/measure_facts.py before`
and the same command with `after`; the harness and raw receipts are retained
locally and as exact git blobs (`git show <blob>` restores their bytes):

- Before FactEncoder: `dadc7a116893b56ae1f5b1d138356581cc0135df`.
- After FactEncoder: `8d85444cc2779d9f5866d529c80e5f35a2bea1ec`.
- Harness: `abaf2ee834bbf95c79dc0fe6878ceccbd3faf01e`.
- Before/after receipts: `674acb711334adeea9dc3d308f42d4e1304d66cb` /
  `5dfed2c9a18d82b981d5d87b8bec1cbb856ad54f`.
- Failing pre-mitigation assertion: `d7693829ae671e5f6ba4f570a5de83a6a7a2faff`.

These sources are uncommitted work above HEAD `9486ff3b4be15002c6423ef6cf633ba536752033`;
blobs, rather than that HEAD, identify the measured implementations. Program
recurrence still uses Python packing/validation and scans the fact buffer;
accelerator synchronization and representative end-to-end measurements remain
open. All full learning outcomes remain unmeasured.
