# MTG-241 — measure and reduce performance and learning risks

Jack Heart directed this in the Task conversation on 2026-11-04: “Let's make
sure to both measure those risks and attempt to ameliorate them.” This extends
the existing acceptance work. The live code Graph retains implementation
ownership; review remains the boundary. No training experiments or merge are
authorized by this direction.

## Performance evidence and mitigation

Use the existing bounded, same-runner Engine Budget procedure and directed
fixtures, never artificial load. Record exact revisions, configuration, counts
and digests. Separate unchanged-work comparisons from rules-driven trace changes.
Include a main/reference-to-candidate comparison on shared supported workloads;
successive already-modified cuts alone cannot establish net cost. After ordinary
integration, disclose inherited layer/payment costs separately from MTG-241's
incremental costs and remeasure the combined head against the standing budget.

- Measure ordinary step time and search total/tail time; include the common
  no-replacement path and representative nested-choice/replacement positions.
- Attribute demonstrated overhead to fork/clone work, retained frame/batch
  storage, replacement candidate discovery/rechecks, trigger snapshots and
  observation projection. Use bounded profiling or operation/allocation counts
  where supported; state measurement limits rather than inventing precision.
- Report legal branching and decisions per fixed scenario alongside search
  nodes/work at a fixed budget. Separate extra legitimate decisions from slower
  execution of equivalent work. Do not truncate choices to improve a benchmark.
- Attempt targeted mitigations where measurements identify costs: ownership
  transfer instead of redundant clones, eliminate repeated collection/snapshots,
  and keep no-replacement execution cheap. Cache only demonstrated repeated work
  with correct revisions and debug oracles. Preserve complete legal choices,
  simultaneous events and exact clone semantics. Remeasure affected cases after
  mitigation; retain before/after and unresolved costs. The configured 10%
  Engine Budget remains in force; moving a reference does not itself mitigate.

## Learning-facing evidence and mitigation

- Construct paired pending states with identical present offers but different
  consequential iteration progress, selected/result bindings or completion
  events. Check that the successor policy input distinguishes viewer-permitted
  differences, with native/Python parity and actual model-input coverage.
  Complete missing projections and reconcile schema ownership with MTG-244.
- Test the inverse information boundary: changes only to unavailable hidden
  information must not reveal it in observations or offers. Exercise retained
  selected subsets and caller frames through hidden-world materialization;
  preserve known exact memberships and coherently refresh unknown commitments.
- Exercise real Commands, divergent forks, replay and reconnect around nested
  selection/replacement/payment pauses. Assert one cost payment, one logical
  commit/finalization and correct trigger/SBA/priority timing, including the
  actual linked-return entry-replacement payment witness.
- Check capacities/overflow and reject unsupported input versions explicitly
  before inference; never silently erase pending state. Preserve frozen widths,
  saved omissions, evidence and world identities.
- Use bounded deterministic tests and configured model checks to establish
  representation and execution properties. These do not establish convergence,
  win rate or sample efficiency; no new training campaign is implied.

At demo/review, summarize each risk with its witness/metric, original result,
mitigation, final result and remaining limitation. Identify exact tested heads;
earlier focused passes do not validate later edits. Keep residual risks visible
and scoped rather than treating the full suite as proof of all interactions.

## Implementation measurements — 2026-11-04

Raw evidence: `scratch/mtg-241-risk-evidence.tar.gz` (committed gzipped tar). The archive contains
all four timing rounds, allocation/capacity/model samples, harness, exact optimized
source delta/new files, base revisions and binary hashes. The pre-optimization
binary hash and measured tracked delta are retained; its earlier new-file snapshot
predates the final projection join/summary additions, so only the optimized source
snapshot is a complete reconstruction. Do not mistake either diagnostic for CI.

The common harness is `managym/examples/bench_execution.rs`: 32 seeded ordinary
games capped at 500 decisions; two search seeds, six decisions each, two worlds,
one rollout, 256-step cap. Every leg runs the same 7,300 decisions (four capped
games), with 36 simulations and ten cap hits per search seed. Legal action counts,
winning choices and full search-score digests match across all three revisions.
Three sequential samples rotate the execution order; no parallel measured work.
Reference is `ea316ff14bc6376dcc302b65d7c9c9330cd512e9`; published layers is
`beaacaa198acddb4162e5081cd570b1ec133ba05`; combined source is based on
`457a779facf27b24eb74afec4e8a2edaea22f234` plus the archived delta.

Final-source medians (milliseconds on one runner):

| Revision | Ordinary steps | Search total | Worst sampled reply |
| --- | ---: | ---: | ---: |
| Reference | 238.799 | 255.758 | 36.954 |
| Published layers | 304.753 | 332.923 | 49.815 |
| Combined | 276.501 | 290.905 | 43.599 |

Thus inherited layers costs +27.6% steps / +30.2% search over reference; combined
is +15.8% / +13.7%, while execution's incremental comparison is −9.3% / −12.6%
against layers. Three short samples are noisy and not tail quantiles. The net
regression is unresolved against the standing 10% budget. No baseline adoption,
accepted cost or claim of readiness follows. Full same-runner Engine Budget is
gate's acceptance measure; this work does not supersede the sibling speed repair.

At 32 fixed fork/projection boundaries (two with retained execution), reference /
layers / combined projection allocations are 3,092 / 3,384 / 3,762, allocating
805,028 / 886,388 / 926,275 bytes. Forks allocate 2,451 / 2,452 / 2,452 times and
971,706 / 1,035,338 / 1,036,386 bytes. This isolates the small retained-storage
increment from the larger inherited clone cost; the corpus has limited nested
coverage. Reusing the rules-token writer, borrowing static serializer names and
hashing closing names without a temporary String reduces combined projection
allocations from 3,858 to 3,762. Before/after combined step/search medians are
278.044/293.418 → 274.980/287.684 ms, but reference also moved 246.155/263.759 →
238.757/258.571, so allocation reduction is established; timing improvement is
not statistically established. The LKI round left sampled allocations unchanged but produced 257.569/274.550
(reference), 400.459/385.816 (layers) and 314.513/336.055 ms (combined), a large
shift across every revision. The final admission repair produces the table above.
All rounds are retained, not selected for a favorable result. The learning
catalog's exact hash is unchanged.

## Learning and retained-state evidence

Six same-offer native pairs distinguish selected object, iteration progress,
completion event, counter cause, logical token count and departed-source power. Native/Python arrays and
LLM text agree, inverse-viewer frames match, forks preserve the capture, old inputs
reject unrepresentable state, and low capacity produces typed Capacity. Fixed
nonzero fact readout weights exercise real rows/typed_pool/typed_gru forwards and
gradients without fitting weights. The completion-event pair exposed a missing
controller-holder join: the critic separated states while action logits did not.
The explicit execution-holder edge repairs the original five pairs for all three
encoders. The final audit found SourcePower reads departed LKI beyond a definition
join; exact referenced LKI now reaches v8 and the sixth model pair separates it.
Fresh fact outputs remain zero by design; these checks establish representability,
not learned use, convergence or strength.

The six-copy nested entry fixture originally used 181/475/512/549/586/623/660 facts
across seven unique pauses (4/7 overflow at default 512). Ordered replacement
records, omission of empty batch fields and default entry terms reduce that to
157/344/373/402/431/460/489 (0/7 overflow), retaining all choices and payloads.
Relation peak stays 12; no width/capacity change. Under #426 recovery, overflow
discards games and biases sampling; this small constructed corpus is not an
estimate of general workload frequency or proof arbitrary nesting fits.

A separate fixed twelve-state forward probe uses one CPU thread, hidden width 16,
one warm-up and seven alternating full/ablated samples per model. It removes only
execution facts/reference edges in the ablation, with identical v8 shapes and
weights: 2,500,416 input bytes, 52–106 occupied facts and 5–7 relation rows per
state. Full versus ablated batch medians are 6.870/5.861 ms (rows), 9.201/7.642 ms
(typed_pool), 9.356/7.807 ms (typed_gru). The extra 1.0–1.6 ms buys the distinguishing
information; it is not a frozen-model or training-throughput comparison. Compact
projection mitigates active token growth but padded input memory is unchanged.
No optimizer or training campaign ran.

Twelve hidden-selection witnesses cover both samplers/viewers, six capture
lifetimes, aliases, moved captures, exact incarnations and inspected-library
partitions. Real Leaves/Caradora/Doubling Season prove 3-versus-4 ordering and
cause-sensitive applicability; real doubled token attachment/attack/sacrifice
and copied-entry choices preserve one suffix/finalization. Actual linked-entry
mana-cost Commands/forks prove once-only payment. Academy pending-resolution
reconnect retains the same offers and summaries, rejects a duplicate commit,
and refreshed authored-match parity covers live/headless/replay. This is retained
session recovery, not process-crash durability.

Check: focused debug rules/projection/hidden-selection and Python model/text/
reconnect/trigger-order checks pass; final-head affected suites, admission and
full budget remain gate-owned. Residuals are net reference cost, general nested
capacity frequency and whether training learns to use the represented differences.

Frozen-input mitigation: a real Fabled Passage witness found that preserving the
legacy code alone did not preserve admission: the added LibrarySearched completion
was still rejected. Matching search/Scry/Surveil obligations now imply their
completion event on frozen inputs; novel same-offer saved events remain v8-only.
The native check covers v1–v8 and the real fetch path checks both encoders.

## Current-head baseline and simplification direction — 2026-11-05

The 15.8%/13.7% figures above predate merging sibling `e4e527621`. Local Apple
Silicon, Engine Budget benches with this head's benchmark sources, one serial
sample per revision. Steps 101,650 and all step/search digests match across all
three. Wall time was contended (search 173.55 s real, 59.54 s user at head), so
it is not evidence here. This is a diagnostic, not Engine Budget acceptance.

| Revision | Steps instructions | Search instructions | Search user s |
| --- | ---: | ---: | ---: |
| Reference `ea316ff14` | 65.53 B | 1,495.05 B | 74.62 |
| Layers `e4e527621` | 53.37 B | 1,082.18 B | 59.53 |
| Head `bf7bf53ad` | 53.99 B | 1,060.54 B | 59.54 |

Head is −17.6% steps / −29.1% search instructions against reference, and
+1.2% / −2.0% against layers. These curated workloads rarely reach nested
replacement paths, so this says nothing about their cost.

Jack Heart directed (2026-11-05, this conversation) a bigger swing using Fable
subagents: make the new system radically simpler and more performant so it is a
win on both. Layers stay MTG-244's. MTG-244 does not depend on this Task; this
Task contains it, so #427 is expected to land first.
