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

## Evidence and mitigations — 2026-11-04

The complete preceding note is preserved as git blob
`1aa5fd5c96f4c5ba822dedb223207ea7639da8f2` (`git show <blob>`), including original
command lines, tables and contrary evidence. Exact executable evidence bundles:

| Evidence | Exact source/receipt reference |
| --- | --- |
| Local-fact isolation and CPU component measurements | Implemented at `043705783efe6aefb843a520681e841effef0e09`; FactEncoder before/after blobs `dadc7a116893b56ae1f5b1d138356581cc0135df` / `8d85444cc2779d9f5866d529c80e5f35a2bea1ec`; harness `abaf2ee834bbf95c79dc0fe6878ceccbd3faf01e`; receipts `674acb711334adeea9dc3d308f42d4e1304d66cb` / `5dfed2c9a18d82b981d5d87b8bec1cbb856ad54f`; failing assertion `d7693829ae671e5f6ba4f570a5de83a6a7a2faff`. |
| Dependency benchmark, cache/model fixes, numeric and profiler probes | Bundle `b273985619373fe6c86996eae2e27b5fb440fd14` retains raw JSON/text, exact harnesses, binary hashes and patch over `fb3555ffb`. Committed mitigation: `b2fb06cbadb8978d1f2373c9b2490551c358c907`. |
| Complete legal payments and paired inventories | Bundle `88d6abd34d8b56a9606c3ec58696063955d08226` (245,225 B): harnesses, raw receipts, engine/model/binary/wheel hashes, instrumentation-only historical patch and command/result scope. Before: `fb3555ffbd36bb797908110fc549999fe8e1a381`; after: `85cb1ceab87e75712265e6a4c9f26351c68d4add` plus measurement edits. Production cache/model match `b2fb06cba`; subsequent main sync changed Wave organization only. |
| Consolidated program fixture parity | `81fa8070f9ff772ff01b2eab6c210e2ac72e362b` retains the matching observation SHA256/configuration, 208,368 input bytes and 21/11 live detail/fact rows. |

### Representation and component cost

A whole-buffer GRU let one holder's program influence another: the new isolation
assertion failed on the predecessor (5/16 elements, maximum difference 0.3391394).
FactEncoder now validates ordinals and owner/provenance, packs independent live
programs, preserves duplicate occurrences and skips recurrence when no program
exists. Independent program order, remapped holders and inserted NaN padding are
invariant; token order and signed numeric values remain distinguishable. Captured
programs reach rows/typed_pool/typed_gru after their source departs.

Component measurements: Apple M4 Max, macOS 26.0.1, Torch 2.10.0, CPU float32,
one thread, B16/D64, 32 objects, **512 facts for both versions**, two warmups/ten
samples, sequential processes without task-owned competing builds. Ordinary
fixture: one counter fact; program fixture: two six-integer programs. Median ms:

| Workload | Retained v7 | Original v8 | Independent-program v8 | Limitation |
| --- | ---: | ---: | ---: | --- |
| Ordinary inference | 3.94 | 15.36 | 3.94 | Component only |
| Ordinary forward/backward | 6.42 | 52.04 | 6.42 | No optimizer or learning claim |
| Two-program inference | Unrepresentable | 15.61 | 4.76 | CPU fixture only |
| Two-program forward/backward | Unrepresentable | 47.84 | 10.29 | Ten-sample max 18.32 ms; no population percentile |
| Whole-harness process peak RSS | Included in each process | 308,150,272 B | 273,973,248 B | Imports/all cases included; not an operation allocator peak |

The unchanged v7 control remeasured at 3.62/6.06 ms, showing ordinary variation.
Matched tensors use 524,288 B; fact parameters remain 3,268,096 B (v8) versus
3,166,720 B (v7). Packing does not reduce capacity. These measurements precede
combined-head acceptance; they establish neither full-policy cost nor learning
quality. The original sources were uncommitted above `9486ff3b4be15002c6423ef6cf633ba536752033`;
the exact blobs above identify them.

Further deterministic findings:

| Risk | Before | Mitigation / result | Limitation |
| --- | --- | --- | --- |
| Numeric meaning | Integers 4096/8192 collide in the embedding bucket and saturated raw projection; max output difference 0 | V8-only signed `log1p(abs(x))/10` in both reducers; difference 0.0576074, gradients/separation pass | Transport unchanged; learned compression is not injective |
| Shared dispatch, B16/D64, 21 live details each | 16 GRU/pack/index-add calls, 416 linear calls | Batch independent programs: 1 GRU/pack/index-add, 71 linear calls | CPU profiler counts; contended timings discarded |
| Warmed pool write, each of 8 clones | 471 allocations / 133,918 B; 1 board / 2 clones / 2 comparisons / 1 edge | Retain mana-independent raw board: 10 allocations / 2,054 B; zero boards/clones/comparisons | Finalized activation outputs and trigger indexes refresh; conditional layers still rebuild |
| Tap-dependent layers, each of 8 clones | 471 allocations / 133,918 B; 1 board / 2 comparisons | Required invalidation retained: 471 allocations / 133,926 B; same board/comparisons | 8 B extra metadata; no tap speedup claimed |

Mana dependency discovery includes nested conditions and granted statics. Debug
recomputation covers dynamic outputs, mana-dependent statics and clone isolation.
Shared/local tests cover source/target joins, meaningful timestamp/token order,
duplicate multiplicity, irrelevant record order/padding, object remapping and
gradients. Twenty-two focused Python cases pass (two optimizer smoke cases
excluded), including all v8 model arms, checkpoint reload, one production
collector transition and arena checkpoint admission. No optimizer/experiment ran.

### Complete legal-payment attribution

On the same Apple M4 Max/macOS host, sequential release runs pay Cadets' real {2}
activation on Moon/Urborg boards: eight exact forks each from floating RR and
from two land sources. Four sources keep another activation payable, so native
trivial skipping remains enabled without resolving the measured stacked ability.
Each payment uses two external actions. Setup, forks and receipt serialization
are unmeasured; native step-return observation is included. Roots stay unchanged.

All complete legal-action sets, selected actions, mana/life/tap resources,
captured programs and semantic digests agree before/after. Counts repeat exactly
within each revision:

| Payment | Before | Mitigated | Limitation |
| --- | --- | --- | --- |
| Floating RR allocations / requested bytes | 1,176 / 295,613 B | 510 / 101,175 B | Allocation traffic, not peak live heap |
| Floating RR boards / clones / comparisons / edges | 1 / 2 / 2 / 1 | 0 / 0 / 0 / 0 | One mana finalization remains |
| Land-funded allocations / requested bytes | 2,882 / 830,700 B | 1,668 / 458,929 B | Includes native step observation |
| Land-funded boards / clones / comparisons / edges | 4 / 8 / 8 / 4 | 2 / 4 / 4 / 2 | Both necessary tap rebuilds and two mana invalidations remain |

This closes payment-specific attribution for the existing mitigation. No further
runtime optimization or reduced legal action space was needed. The comparator
now requires every funding/sample identity, not merely sixteen equal rows.

### Full input storage and remaining timing limits

Default arrays: v6 133,616 B; v7/128 facts 146,928 B; v8/512 facts 208,368 B.
V8 adds 61,440 B over v7: 43,008 B shared details plus 18,432 B local facts.
A rollout storing N observations uses N times the payload before other fields
and allocator overhead. No capacity was reduced or silently truncated.

The full-policy harness uses the baseline recipe's width-64/depth-2/four-head
FiLM model with world catalog, CPU float32, one thread, B16, two warmups/eight
samples. Compared capacities: **512 facts in v6/v7/v8**, 512 actions, 60 cards/
40 permanents per seat. Before/after inventory wheels matched all seven cases
(native/Python arrays, observation hashes/configurations, storage, occupancy).
Historical imports resolved inside the archive; inventory does no policy calls.

| Input/state | Observation B | B16 observation B | Parameter B | Live details/facts |
| --- | ---: | ---: | ---: | ---: |
| v6 ordinary | 133,616 | 2,137,856 | 4,357,128 | 0/0 |
| v7 ordinary | 165,360 | 2,645,760 | 7,555,592 | 0/0 |
| v8 ordinary | 208,368 | 3,333,888 | 7,761,416 | 0/0 |
| v8 grant + captured trigger | 208,368 | 3,333,888 | 7,761,416 | 21/11 |

The representable shared fixture resolves one Drill activation: v6 has 20 local
facts; v7/v8 have two details, ten explicit memberships and no local facts.
Storage equals each version's ordinary row. The 21-detail program fixture is
not a universal occupancy bound. Retain 4,096 capacity and explicit overflow.
Shared reduction visits header-owned slices; per-record Python validation,
fact-buffer scans and accelerator synchronization remain costs/risks.
Accelerators, sample efficiency, win rate and generalization are unmeasured.

Contrary timing evidence must remain visible: the shared host reported load
522.29/461.04/351.76 during earlier unrelated compilation and 614.75 during later
inventory work. Its original Search-64 run was stopped without a complete seed;
no background benchmark remains. Completed measurements under contention:

- Original TLA: 512 games, 101,650 steps, zero caps, digest
  15843333459759757527; pre-mitigation elapsed 35.470791 s.
- Dependency: 16 games/689 steps, eight search decisions/1,600 simulations,
  no search caps; pre-mitigation step attribution 2,284 boards, 4,536 clones/
  comparisons and 2,268 edges.
- Pre-mitigation full v8 program: inference 208.46 ms and forward/backward
  428.76 ms medians (eight samples); process peak RSS 1,048,084,480 B.

These timings establish neither regression nor speedup and cannot be compared
with CI times. Before whole-workload rows incorrectly repeat the last probe's
allocation counters when `counting_allocations=false`; ignore those fields.
Allocation-enabled individual probes are valid; the final harness emits zero
for non-counting runs. No clean final step/search/full-policy comparison exists
yet. Earlier anthem-only benchmarks did not exercise dependency ordering.

## Isolated producer and remaining acceptance

Reproduction commands: `docs/development.md#bounded-layerinput-attribution`.
`layer-risk.yml` runs all builds/measurements sequentially on one GitHub-hosted
VM. Its first PR publication triggers it; manual dispatch becomes available
once it exists on the default branch. Publication belongs to the later Graph
step. The workflow supplies an execution path, not a performance result.

`scripts/measure_layer_risk.py` archives the Engine Budget reference and pinned
pre-mitigation source, applies instrumentation/fixture access only, and uses
identical fixtures. It retains revision/harness/binary identities, environment,
full-score digests, counts and samples, even if a later phase fails. It runs:

- Original TLA step/search at baseline, before and after. Reference-world
  differences are explicit, never equivalent-workload speedup claims.
- Dependency step/search timing and separate allocation runs before/after;
  complete-payment attribution; search samples and first 128 step durations.
  Prefix maxima are not population percentiles; overlapping timers cannot sum.
- Seven matched v6/v7/v8 model cases per revision, rebuilding each wheel and
  using fresh case processes. Native/Python encoding, inference/backward
  throughput and process peak RSS (imports/warmups included) are retained.

One exact policy-case list drives production and comparison. The comparator
rejects incomplete seeds, changed payment/actions/digests/counts, missing or
mislabeled policy cases, inventory masquerading as timing, nonfinite/incomplete
samples, missing memory and unmatched warmups/configuration/storage. Seventeen injected-receipt
checks pass without running games or training; slower but comparable timings
remain in the output for inspection, never automatically accepted.

Still required: collect the first isolated artifact, inspect matched results,
mitigate demonstrated findings and remeasure before review readiness. Gate owns
full suites and the unchanged 10% same-runner Engine Budget; a green budget alone
does not close dependency/model risk evidence. No baseline adoption, merge,
training experiment or learning-quality claim is authorized by these checks.
