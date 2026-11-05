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
`b31998cc3:scratch/mtg-243-naming.md`. The uncommitted accepted speed direction
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
The coverage Wave remains read-only. MTG-241 owns continuation execution and has
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
reference `ea316ff14`. Current-head Engine Budget run `41951255376`, job
`126995532291`, passes: worst reply 3.638 -> 3.830 s (+5.3%), total search
123.901 -> 128.930 s (+4.1%), throughput 17511.2 -> 16527.4/s (-5.6%, +6.0%
time/step). Both candidate orders nearly exhaust wall time on CPU: searches
128.841/129.086 elapsed vs 128.797/129.070 CPU seconds. These comparisons still
include intervening engine changes and do not isolate naming. Preserve earlier
failed samples; no budget reference/tolerance changes are authorized. Final main
integration in that run was still running when this speed pass was requested.

Execution direction:

- First profile bounded actual step/search workloads and the existing real
  naming/model mixture. Separate setup, native clone/decision/projection,
  observation encoding, semantic program encoding, action scoring, and learner
  materialization/backward where relevant. Inspect existing profiling hooks and
  representative native workloads. Identify dominant cost and allocation owners
  before selecting structural changes. Produce ranked measured hotspots rather
  than speculative optimization or another diagnostic-only conclusion.
- Optimize the highest-payoff paths revealed by those profiles. Existing evidence
  points to executable-program encoding/backprop and ordinary native runtime as
  candidates; verify rather than assuming. Prefer removing repeated work,
  redundant allocations/copies, or oversized representations with clear owners.
  Preserve complete choices, original action coordinates, viewer-safe identity,
  replay/Command semantics, frozen versions, and full behavior evidence. Learning
  must retain gradients: no detached trainable-program cache across updates or
  inference-only shortcut described as a learner speedup. Native cache changes
  must preserve revision invalidation and debug recomputation invariants.
- Measure before/after against b31998cc3 on the same machine, building both before
  measurement and using a fixed balanced run order with all samples retained.
  Use CPU/wall time and scheduling telemetry, exact memory/allocation counts
  where available, and matching digests. Keep profiling separate from timing.
  No other workers started by this pass during timing; no artificial load,
  contention injection, adaptive reruns until green or cross-runner wall-time
  comparisons. Use controlled fixtures and bounded real workloads only.
- Seek a material, repeatable gain beyond noise, reporting native step throughput,
  total/worst search latency and actual model/learner costs separately. Do not
  infer end-to-end gains from a microbenchmark or claim general scaling from
  definition-free names. Preserve mixed and ordinary workload coverage so a
  naming optimization cannot silently slow every other turn. Do not promise an
  arbitrary multiplier before profiling. If a proposed large engine redesign is
  separate work, retain the findings and identify that boundary rather than
  silently expanding this PR.
- Check active ownership before changing shared hot paths; do not duplicate
  ongoing engine/continuation work. Keep one implementation and remove obsolete
  paths. Validate meaningful semantic/numerical equivalence, the combined 928-name
  execution proof, debug Rust/native rebuild as applicable, and required affected
  checks. Current-source CI must retain the unchanged budget. Update the PR and
  review artifacts with the achieved speedup, baseline hashes, raw evidence and
  remaining limits; keep any failed approaches in concise evidence notes.

The Graph is restarted at implement for profiling plus implementation, returning
to demo for review. This direction supersedes the earlier statement that no
further runtime optimization was requested; it does not approve shipping.

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

Reproduction: `uv run --extra dev python scratch/naming-speed.py measure --width 64`;
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

Checks after simplification: `uv run --extra dev pytest tests/model/test_policy_catalog.py::test_program_execution_preserves_values_and_parameter_gradients tests/model/test_catalog_naming.py::test_legal_row_scoring_preserves_mixed_logits_gradients_and_preparation -q` — 14 passed; `uv run --extra dev --extra artifacts --extra evaluation pytest tests/verify/test_native_buffer_capacities.py::test_retained_airship_growth_preserves_every_permanent tests/verify/test_native_buffer_capacities.py::test_retained_airship_overflow_isolates_only_its_stream tests/verify/test_native_buffer_capacities.py::test_repaired_retained_prefix_has_scalar_vector_python_parity tests/verify/test_shared_effect_transport.py::test_retained_drill_prefix_admits_shared_effects tests/eval/test_tla_pilot.py::test_pilot_retains_generated_cards_and_sideboard_in_all_encoders -q` — 14 passed. Ruff check/format and diff checks passed. Earlier six consumer cases passed; the first whole-catalog gradient oracle failure and its diagnosis remain above.
No Rust changed, so no rebuild or new debug-suite result is claimed. Gate owns
the affected model/inference and active-training suites and fresh Linux CI;
publication/demo own PR copy, review-artifact refresh and interactive review.

[`naming-speed-evidence.json.gz`](naming-speed-evidence.json.gz) retains both
widths' raw samples, CPU telemetry, numerical diagnosis, allocation profiles,
native controls, binary hashes, the rejected prototype and retained probe/code
sources. Original raw files remain in `.lf/logs/naming-speed/`. Reproduction
requires a fresh `--label`; the probe refuses to overwrite prior measurements.
