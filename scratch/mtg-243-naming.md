# MTG-243 — complete catalog naming

## Accepted direction and boundary

Jack Heart authorized complete, learnable catalog-wide naming beyond 512 names
as one reviewable PR on 2026-11-04. Subsequent accepted direction covers measured
risk reduction, the combined native/vector/Academy proof, Linux numerical and
Engine Budget diagnosis (comment d4db22c5), and the speed pass below (comment
86384701). PR #424 remains at interactive review: no merge, auto-merge, paid
execution or strength experiment. Gate owns broad validation; publication and
demo own the review artifacts and interactive handoff.

The current speed baseline is `b31998cc38bb3b86abedce397e2531c4e0d91379`.
Earlier implementation, repair, review and publication history is preserved in
`b31998cc3:scratch/mtg-243-naming.md`. The accepted speed direction
and pre-curation notes are retained in `.lf/logs/naming-speed/plan-before-curation.md`.
Original implementation: `9db184b9`; risk reduction: `5a57096f2`; combined-proof
repair: `a643a344b`; upstream integration: `bca261295`; numerical/test
simplification: `bca3ff438`; configuration sync: `c52b70a53`; pilot assertion
repair: `f2b54710f`. Frozen evidence has not been rewritten.

## Design and ownership

`ContentPack::name_candidates` owns immutable sorted public names and a digest.
`Decision::NameCard` shares that vocabulary; accepting one choice constructs
one `ExileNamed` and resumes its suspended frame. Commands retain modal order
and labels. `SearchNamed` retains private inspection, remaining selections and
fail-to-find semantics. Filtering commits nothing. Reconnect restores the actual
pending decision; committed exiles survive. This is retained-session recovery,
not process-crash durability. Search, replay, forks, LLMs and Academy consume the
same atomic choices without browser-only semantics.

Explicit policy input v8 has `name_candidates[A,2]`: categorical public name and
optional one-based definition join. A is max(configured actions, vocabulary).
Names never consume card slots or imply supported definitions. Fresh defaults
remain v6; v1–v7 shapes and omitted checkpoint defaults remain frozen. Default
overflow rejects; explicit first-legal on legacy inputs is not complete naming
support. V8 reserves its whole vocabulary even under that explicit policy.
Additional production names and fresh-default adoption remain separate decisions.

Flat and compound policies gather legal rows, share object/program computation,
and scatter scores to original coordinates, including holes and reordered
choices. One action-embedding owner serves both. Continuation joins select
columns before row gathers. Prepared inference caches programs only within its
unchanged policy scope; training retains gradients through every referenced
program.

`PackedActions` owns offsets, original slots, legal action fields and complete
behavior probabilities/log probabilities. Pending transitions and `RolloutBatch`
retain those rows; object fields stay on CPU. Both learners validate behavior in
bounded CPU chunks and materialize only selected minibatches. Returns, references,
chosen indices and numerical invariants remain unchanged. Numerical incidents
(schema 2) retain compact evidence plus failing minibatch data. Recovery snapshots
still require empty pending/transition storage.

V8 joins shared name identity into public definition references before
card/known-hand/decklist, continuation and history pooling. Candidate scoring
receives raw executable program embeddings without double-adding names. Closed
native decklists remain zero; hidden membership is never inferred. Names without
definitions create no objects. V8 checkpoint binding includes
`naming_representation=public-identity-v1`; changed names/order/joins and earlier
unreleased previews fail before weights load, without warm-start migration.

Live schema 8 / capture 5 / state hash 16 are the original Task transition.
Historical captures 3/4 use schema 7 for read-only validation and preserve their
identity/digest. Schema 7 rejects new name references. Retained schema-7 replay
fixtures compare all current scalar/vector decision fields exactly, excluding
only the explicitly asserted schema/fingerprint transition. Frozen bytes stay intact.

## Delete — do not maintain

Completed in the naming change: per-name owned effect branches and resolution
sorting; padded action networks; dense retained action/distribution padding;
whole-rollout device expansion; `RolloutBatch.from_dense`; duplicate packing
copies; fabricated large-vocabulary model rows; duplicate Rust vocabulary
builders; stale schema-7 live-frame assertions; platform-default least-squares
solver and bitwise float assertions on reordered GEMMs; duplicate digest checks;
candidate/build-reference/reference timing. Their required behavior and evidence
now live on the surviving native, packed-action and balanced-sampling paths.

This speed pass replaces embedding padded programs and one unbounded packed GRU
call. There is one production execution path with the same GRU, weights, tokens,
checkpoint buffers and complete gradients. The old packed execution survives
only as a numerical test reference at `tests/model/naming_reference.py`, alongside
the frozen dense action reference. The rejected dense-bucket prototype remains
in `.lf/logs/naming-speed/dense-buckets.py`, not production.

Catalog selection/scattering stays in `definitions`; `_encode_programs` owns
packing, recurrent chunks and restoration of supplied row order. The reordered
oracle selects a descending-length subset directly, including the longest
program. One `tests/retained_decision.py` assertion replaces three copies of the
schema-transition comparison: version and fingerprint must change as declared,
and every remaining decision/offer field must match. No further deletion target
remains; frozen references and evidence stay intact.

## Evidence carried into the speed pass

The checked-in 928-name Scryfall scenario vocabulary is one native authority
through Rust, scalar/vector encoding, the model and Academy. A policy-selected
row above 511 executes through scalar Commands, vector buffers and the WebSocket;
post-execution state/encoding agree. Separate choices 0/511/512, Gray Ogre and
Fire Lord Sozin retain fork/replay/stale checks. Gray Ogre exercises private
search, remaining selections, reconnect and irreversible committed exiles.
Names without matches retain trivial-decision skipping. Scenario injection
changes only the pending decision, not catalog installation or deck admission.

An equal-rule public-decklist pair exposed identical policy logits before the
identity join. A checked two-example SVD fit of the existing final head now
expresses opposite preferences with identity/program gradients and reordering
invariance; closing decklists removes the distinction. A preceding 16-step Adam
probe failed. This proves representability, not SGD convergence, exploration,
sample efficiency, transfer or strength. Action-type-uniform references offer
no relevance prior within one naming action type.

The fit matrix had singular values 2.657710831 and 0.002939572 (condition ~904),
rank 2, and six zero columns. Pinned Linux x86 Torch 2.10 CPU `gelsy` reported
rank 0; `gelsd` retained rank 2 and residual <3e-17. macOS 2.10 and ARM Linux
2.15 did not reproduce that failure. SVD rank/residual checks repair the fixture.
A later Linux closed-input assertion differed by at most 1.49e-7 in logits;
inputs remain bitwise equal and outputs use 2e-6 absolute / 1e-6 relative bounds.
Preference thresholds and complete support remain strict. These were solver and
floating-point failures, not CPU-load symptoms or lost learnability. Raw evidence
is retained in `.lf/logs/naming-fit.pt`, `naming-linux-x86-fit.log`,
`naming-closed-head.pt` and `naming-linux-closed-head-internal.log`.

Four controlled observations reduced retained action/distribution bytes from
237,568 to 43,308 (81.8%); including unchanged objects, 620,480 to 426,220 (31.3%).
Live buffers and output logits remain dense. Earlier timings versus `9db184b9`
were variable and did not establish throughput gains; full program encoding and
backprop dominated. The 928-name fixture mostly adds definition-free names, not
928 programs. New speed measurements below use 344 referenced executable
programs / 41,159 real tokens for naming and four / 190 for ordinary decisions.

## Native budget and contrary evidence

Tolerance remains 10%, with reference `ea316ff14` and exact digests/counts.
`engine_budget.py sample` builds neither side: both must already exist. Its fixed
candidate/reference/reference/candidate order retains every sample, CPU/wall time,
context switches and load/pressure counters, then compares median durations.
Failed reference builds/runs retain candidate evidence and still fail the job.
Controlled executable fixtures verify these semantics without artificial load.

Old CI `41925981431` failed steps (23,489.9 → 21,126.6/s) while search passed
and digests matched. No utilization telemetry exists: contention is unproven.
The pre-sync local paired run retained a worst-reply failure, 4.290 s versus
2.399 s limit, though steps and search total passed. Candidate search wall was
78.852/95.557 s versus CPU 77.841/79.652 s, with increased involuntary switches.
Scheduling contributed to variation; no per-reply CPU evidence attributes the
whole 6.498-second outlier. Reference launch delay did not enter its timed step
loop. No reruns, discarded samples or baseline movement cleared that failure.
Evidence: `.lf/logs/naming-budget-paired/`.

Linux `41942374422` at `bca261295` passed with exact digests: worst reply +5.1%,
search total +4.8%, time/step +8.0%. Both orders nearly exhausted wall time on
CPU. Linux `41946199054` at `55cde4796` likewise passed (+5.5%, +5.0%, +8.3%).
These repeatable within-budget costs are real aggregate engine work, not
explained away by load/order noise, but include intervening changes since
`ea316ff14` and do not isolate naming. Different runners' wall times must not be
compared. Raw artifacts: `.lf/logs/naming-budget-final-linux/` and
`.lf/logs/naming-publish-budget-linux/`.

The latter run passed all 1,099 main integration tests, debug Rust, Academy,
browser, protocol and clean-machine checks. Its later evaluation step failed two
schema-transition assertions (324 others passed). Their test-only repair keeps
all decision fields and frozen bytes; all 48 pilot tests passed locally.
Publication validation also passed 24 naming/packed/recovery tests, 91 admission/
recovery checks, Clippy/format, current receipt verification, headless Perf report
and bounded CLI smoke. Receipt refresh changed source fingerprints only, with
135 Commands and all state/viewer/event/stale witnesses unchanged. Those are
pre-speed-pass results, not new-source acceptance; broad checks remain with gate.

## Coordination and integration

Main `ee85d6fc8`'s opt-in isolated capacity recovery is incorporated. Naming writers
convert `ObservationWriteError` through `AgentError::from`; default rejection and
fatal malformed buffers stay intact. Main `51b4bd274` was a configuration-only
sync. The separate MTG-213 pilot v10 supplies no execution authority here.

Main's accepted full Standard roadmap supersedes the former Allies/Lessons-only
Wave scope; it adds no card-authoring, prepared-spell or training work to MTG-243.
MTG-241 owns continuation execution and has
reported `FrameCompletion`, obligation codes 27–29, retained events/payments and
postambles; this checkout still has `FrameFinalize`, maximum code 26, naming code
10. Ordinary sync must reconcile them; no collision is known.

The speed-pass ownership reads found MTG-241 active at realign and MTG-242/244
active at implement. Native profiles point into their stabilization, payment and
layer domains. This pass does not duplicate their changes or introduce a new
cross-action invalidation scheme. Further native structural changes require
coordination after integration; native unchanged measurements are controls, not
speedup claims.

## Accepted speed pass — 2026-11-04

After reviewing the repaired performance evidence, Jack Heart directed:
“lets try to ake it stupidly fast”. Continue this Task and PR with a measured
optimization pass; the design and complete naming/learning semantics remain
accepted. Passing the existing 10% budget is the starting point, not the desired
speedup. Publish the results for interactive review, without merge/Auto-Ship or
paid execution/strength experiments.

Baseline for attributable improvements is current published/runtime head
`b31998cc38bb3b86abedce397e2531c4e0d91379`, alongside the unchanged CI budget
reference `ea316ff14`. Baseline-head Engine Budget run `41951255376`, job
`126995532291`, passes: worst reply 3.638 -> 3.830 s (+5.3%), total search
123.901 -> 128.930 s (+4.1%), throughput 17511.2 -> 16527.4/s (-5.6%, +6.0%
time/step). Both candidate orders nearly exhaust wall time on CPU: searches
128.841/129.086 elapsed vs 128.797/129.070 CPU seconds. These comparisons still
include intervening engine changes and do not isolate naming. Preserve earlier
failed samples; no budget reference/tolerance changes are authorized. That run
completed successfully, including main/long integration, Linux learning
assertions and Engine Budget (rechecked during reconciliation on 2026-11-04).
It certifies the pre-speed baseline, not the optimized source.

The profiling and implementation phase is complete at `5bb773ebf`: the measured
highest-payoff change is packed program backward, with native profiling retained
as an ownership-bound finding. Acceptance still requires complete choices,
original action coordinates, viewer-safe identity, exact replay/Command digests,
frozen inputs and full learning gradients. No detached trainable cache or new
native invalidation scheme was introduced. Fixed balanced measurements retain
all samples and CPU/scheduling telemetry; no artificial load, adaptive retry,
cross-runner timing comparison or budget/reference movement is authorized.

Native structural optimization remains outside this diff because its measured
hotspots overlap MTG-241/242/244. No native gain is claimed. The achieved learner
speedup and the decision to leave those native changes with their owners remain
part of interactive review, not an implicit reduction of the accepted outcome.

## Speed implementation — 2026-11-04

The actual naming/mixed-model CPU profile found packed GRU backward dominated by
slice-gradient zeroing and accumulation: `aten::add_`, `fill_` and `add` accounted
for 68.6% of the first mixed sample's self CPU time. Naming slice backward
allocated about 5.25 GiB cumulatively per pass at width 16. This is allocation
traffic, not peak RSS. Ordinary decisions referenced only four programs and did
not have that long-sequence cost.

`SemanticCardEncoder.definitions` now packs integer IDs before embedding and
runs the same packed GRU in blocks of at most 64 time steps. Hidden state carries
full autograd history across blocks; completed suffixes restore original
definition coordinates. Every real token is processed once. There is no detached
cache, new parameter, persisted buffer, truncation, optimizer change or input
version. The old unbounded packed execution remains only in the test reference.

A first dense length-bucket implementation improved naming forward/backward
249.2 → 80.1 ms but slowed forward 18.6 → 30.5 ms. It was rejected in favor of
chunking, which retains packed arithmetic. Its source and all balanced samples
remain in `.lf/logs/naming-speed/dense-buckets*`; no failed approach is hidden.

The final bounded measurements use the same native four-observation mixture,
fixed weights, one CPU thread, widths 16 and 64, and two complete
candidate/reference/reference/candidate cycles. Each position warms once; every
sample is retained, including CPU/wall differences and scheduling counters.
Reference execution loads the exact `b31998cc3` method into an otherwise identical
model. Profiling is separate from timing. Both learner measurements include
packed minibatch materialization, the real PPO or damped policy loss primitives,
critic loss and full backward; no optimizer step or training campaign runs.
These are batch costs, not end-to-end collection/update throughput.

| Workload, width 64 | Baseline wall / CPU ms | Chunked wall / CPU ms |
| --- | ---: | ---: |
| Mixed forward + backward | 914.5 / 910.5 | 154.1 / 155.1 |
| Mixed PPO minibatch | 921.9 / 919.0 | 156.6 / 156.4 |
| Mixed damped minibatch | 915.8 / 913.4 | 159.2 / 158.5 |
| Naming forward + backward | 1120.0 / 1074.0 | 168.0 / 164.0 |
| Ordinary forward + backward | 7.79 / 7.77 | 6.69 / 6.69 |
| Mixed forward only | 39.0 / 39.7 | 36.7 / 37.2 |

These medians show roughly 5.8–6.6x lower CPU cost for width-64 mixed/naming
learning batches. Width 16 mixed/naming forward-backward CPU fell 287.5 → 72.0
and 261.5 → 64.4 ms. Width-16 ordinary results were noisy: forward-backward CPU
5.41 → 5.47 ms, wall 5.41 → 23.20 ms; PPO CPU 5.60 → 8.04 ms and damped CPU
5.76 → 7.08 ms. That smaller workload does not establish a universal improvement.
Width-16 mixed forward wall also rose 22.57 → 31.97 ms while CPU fell slightly
20.87 → 20.10 ms. Host contention is evidenced by individual CPU/wall divergence
and switches, not inferred from load averages alone. No sample was dropped or
rerun to obtain a preferred result.

The native sampling profile attributes 1,243 of 1,255 application samples to
random playout under `flat_mc_scores`; branch cloning/catalog setup is not the
dominant search cost. The largest observed subtree was stabilization (426),
including state-based actions (348); action generation contributed 193 samples.
These nested counts are not additive or full-run percentages. Their layer,
payment and continuation owners are active in MTG-241/242/244. No speculative
native cache/invalidation change accompanies the learner optimization. The
balanced native timing is an unchanged-binary control, not a native speedup.

Reproduction: `uv run --extra dev python scratch/naming-speed.py measure --width 64 --label <fresh-label>`;
the width-16 default is identical. Raw timings, profiles, prototype source and
baseline binaries are retained under `.lf/logs/naming-speed/`. Native sampling
uses `scratch/native-speed-profile.py`; balanced timing uses the existing budget
sampler. The native source, benchmark digests and committed budget stay unchanged.
CUDA is unavailable; GPU execution/performance and larger learner batch sizes
remain unmeasured. Definition-free vocabulary growth does not simulate a larger
executable catalog.

### Final evidence and remaining acceptance

The final profile reports naming `slice_backward` cumulative allocations of
5,637,168,000 → 508,046,656 bytes (91.0% less). Counts are allocator traffic,
not peak memory. Packing IDs also embeds 41,159 real naming tokens instead of
245,272 padded positions; the serialized token table remains unchanged.

Native baseline/candidate executables are byte-identical. The fixed control
retains 101,650 steps in 512 games, step digest `15843333459759757527` and search
digests `13733780288109658817`, `5337676601429031145`, `10035331835025021780` in
every sample. Reference/candidate medians: 28,444.6 → 26,277.9 steps/s,
85.078 → 88.376 s total search, 2.977 → 3.3495 s worst reply. The last metric
**fails** the unchanged 10% threshold despite identical executable bytes.
Candidate search process wall/CPU was 93.030/82.104 and 84.045/78.116 s;
reference was 87.430/78.633 and 82.778/77.962 s. Scheduling accounts for some
elapsed variation; CPU duration also varies. Code identity rules out a code
regression in this control, without explaining old different-source failures or
clearing the production budget. No reference/tolerance change or retry followed.

The unnormalized width-64 whole-catalog float32 gradient oracle initially failed
one input-weight entry: 0.012508392 versus 0.012538776, with gradient relative L2
error 1.64e-6 and exactly equal outputs. Full double-precision recomputation
reduced maximum gradient error to 2.88e-13 (relative L2 4.17e-15). The algebraic
oracle now uses float64 with 1e-11 absolute / 1e-9 relative gradient tolerance;
actual float32 model-loss tests retain 2e-6 / 3e-4 and cover both measured widths,
attention/FiLM combinations, sparse reordered actions and prepared inference.
This is changed reduction order, not lost cross-chunk gradients. The failed
float32 diagnostic is retained in `gradient-roundoff.json`.

Retained post-simplification checks cover 14 program-value/gradient and mixed
policy cases plus 14 retained-capacity/shared-effect/pilot parity cases; Ruff
check/format and diff checks passed. Earlier six consumer cases passed. The
whole-catalog float32 failure and its diagnosis remain above. No Rust changed
in the speed pass, so no rebuild or new debug-suite result is claimed here.

### Remaining acceptance — reconciled 2026-11-04

- Gate: affected model/inference and active-training suites, plus current-source
  Linux CI with the unchanged 10% Engine Budget. At reconciliation, PR #424
  points to `5bb773ebf`; its Linux checks are in progress. Baseline run
  `41951255376` is green, but supplies no optimized-head result.
- Performance review: width-64 mixed/naming batch gains are measured, while
  width-16 ordinary PPO/damped CPU costs increased and native worst-reply control
  failed. These observations remain limitations, not load-only explanations or
  cleared failures. GPU, larger minibatches, optimizer/update throughput and
  end-to-end collection/learning remain unmeasured.
- Publication/demo: refresh PR copy and review artifacts around the achieved
  learner gain, failed native control and scope limits, then obtain interactive
  review. The concept page now covers the speed pass; other review artifacts
  still need refresh. Merge/auto-merge,
  paid execution and strength experiments remain unauthorized.

No new product decision is required for reconciliation. Additional production
names, fresh v8 defaults and a broader native redesign remain separate decisions.

[`naming-speed-evidence.json.gz`](naming-speed-evidence.json.gz) retains both
widths' raw samples, CPU telemetry, numerical diagnosis, allocation profiles,
native controls, binary hashes, the rejected prototype and retained probe/code
sources. Original raw files remain in `.lf/logs/naming-speed/`. Reproduction
requires a fresh `--label`; the probe refuses to overwrite prior measurements.

The archive's `source_note` predates simplification: the measured recurrent loop
was subsequently extracted into `_encode_programs`, leaving catalog selection
and scattering in `definitions`. The arithmetic and chunk boundaries are
unchanged, but the current helper layout was not separately timed. Retained
post-extraction equivalence tests support correctness; archived sources and raw
measurements remain intact rather than being relabeled as exact-head timings.

Check: `git diff --check` and `lf context` — passed; memory and scratch remain within both limits. Behavioral checks above are reused, not rerun.

## Concept review — 2026-11-04, head `5bb773ebf`

Outcome: no design change and no open product choice for this scope. Jack Heart's
accepted atomic naming interaction still fits the implementation: filter without
commitment, submit one bound name choice, then inspect/select through the native
private search. Reconnect restores the actual pending choice and remaining search;
stale submissions cannot repeat committed exiles. Public vocabulary, executable
definitions and hidden membership remain distinct. Explicit v8 and frozen/default
input boundaries stay as documented in `docs/training.md`.

The speed pass adds no user-facing concept. `definitions` owns catalog selection
and original-row scattering; `_encode_programs` owns transient packing and GRU
execution. A 64-step chunk is neither a game decision nor a training truncation,
checkpoint boundary or resumable job. Full hidden-state gradients cross chunks;
weights, serialized buffers, policy-scoped inference caching and empty-rollout
recovery requirements remain unchanged. No new configuration, workflow skill,
usage-doc change or implementation direction is needed.

The consequential review mismatch is evidential: faster complete learning batches
do not imply faster native play or end-to-end training. Keep the measured gain,
ordinary-workload regressions and failed identical-binary native control together
at interactive review, without treating noise as budget acceptance. Leaving native
structural work with its active owners does not silently reduce the accepted
speed request. No further product approval is inferred here.

Counterexamples remain a missing score above 511, preferences tied to action
ordering, hidden-membership leakage, stale/reconnect loss of commitment, dropped
behavior probabilities, detached cross-chunk gradients or remapped checkpoint
identity. Existing targeted witnesses cover these contracts; this review inspected
source and retained evidence without rerunning behavioral tests or measurements.
Next proof remains the reconciled gate and publication/demo obligations above;
fresh current-source acceptance is not supplied by the baseline's green run.
Broader production names, fresh v8 defaults and native redesign remain outside
this review's accepted scope. `scratch/concept-review.html` presents this result;
loop-or-next owns navigation.

## Publication validation — 2026-11-05 UTC

`lf sync` reconciled main through `653b014b1` into `ed3c914f2` and resolved
only `scratch/questions.md`; no runtime or test source changed from `5bb773ebf`.
Incoming pilot report/artifact-manifest changes remain upstream-owned. The
scratch conflict keeps MTG-243 scope and a pinned reference to main's pilot notes.
The sync's artifact-tracking check passed and the working tree was clean.

Publication inspected the actual `origin/main...HEAD` diff, native naming and
scalar/vector bindings, model identity/scoring, full-gradient packed execution,
rollout/learner ownership, retained schema transition and combined Academy proof.
`uv run --extra dev --extra artifacts pytest tests/model/ tests/training/test_active_training.py -q`
passed **249 tests, 1 skipped** in 84.71 seconds. `git diff --check` passed.

Linux CI `41956530425` at `5bb773ebf` passed debug Rust, Python unit, long
integration, Academy integration/UI, protocol/kernel conformance and clean-machine
Play. Its main integration was cancelled after sync pushed `ed3c914f2`; Engine
Budget was still running at this check. Fresh CI `41957382497` is queued for the
synced head. These partial results are not a fully green optimized-head result.
Publication may expose this checked change for interactive review, but full Linux
integration and the unchanged 10% Engine Budget remain acceptance obligations.
No merge, auto-merge, Task completion or new performance run is selected.

PR copy must include the packed-program speed pass: measured width-64 mixed/naming
CPU batch costs improve about 5.8–6.6x against `b31998cc3`; allocation traffic in
naming slice backward falls 91%. These are retained measurements before helper
extraction, not new timings of this merge head, peak RSS, GPU results or complete
training-update throughput. Preserve the width-16 ordinary PPO/damped regressions
and failed identical-binary native worst-reply control beside those gains.
Native runtime source is unchanged in the speed pass, its structural hotspots
remain with MTG-241/242/244, and no native speedup or cleared budget is claimed.
The numerical oracle retains complete gradients; double precision diagnoses the
whole-catalog float32 reduction-order discrepancy, while real float32 policy-loss
checks retain their own bounds. Frozen evidence/reference/tolerance stay intact.
