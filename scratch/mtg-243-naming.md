# MTG-243 — complete catalog naming

## Accepted direction — 2026-11-04

Jack Heart authorized complete, learnable catalog-wide naming beyond 512 names
as one reviewable PR, with no merge or training experiments. After publication
of `9db184b9aa3df9286b07abdafa2d41db0d93e651` in PR #424, Jack Heart requested:
“Let's make sure to both measure those risks and attempt to ameliorate them.”
That accepted pass covers legal-row scoring, compact pending/rollout storage
through both learners and incidents, bounded same-host costs, paired public-state
competency, actual executable-program attribution, and the five CI compatibility
failures. The risk reduction is committed at `5a57096f2`, the accepted proof and
diagnostic repairs at `a643a344b`, and the upstream integration repair at
`bca261295`. The rounding repair and model, test and documentation
simplifications were committed at `bca3ff438`; PR #424 now publishes the
subsequent configuration-only sync `c52b70a53`. Its description and review
artifacts still describe earlier proof. Current-source
acceptance and interactive review remain outstanding. No auto-merge, paid
execution or strength experiment is authorized.

The original implementation and publication evidence are retained at
`9db184b9:scratch/mtg-243-naming.md`; the uncommitted research/accepted-pass text
was preserved at `.lf/logs/naming-risk/plan-before-reduction.md` before curation.
Earlier T7 direction is historical at
`4463e57f5:scratch/questions.md`.

## Reconciled upstream context — 2026-11-04

The latest main sync, `51b4bd274` at `c52b70a53`, only restores existing
Linear Wave bindings in supported configuration; it changes no runtime or tests
relative to `bca3ff438`. Current-head CI `41945242063` is running as of this
reconciliation, not passing evidence. Completed run `41942374422` belongs to
`bca261295`, before the rounding repair and simplifications.

Main `1546aae3f` added the published Standard roadmap and
`managym/PRINCIPLES.md`, renamed the Wave directory from engine to coverage,
and introduced separate engine-performance and infrastructure scopes. That sync
changed no runtime or tests relative to `5a57096f2`. Coverage still has no
immediate child directories; the two sibling memories contain only headings.
Both memory and scratch fit their configured context targets.

The subsequent sync of main `ee85d6fc8` at `f9f97d0d1` does change runtime:
isolated training streams can explicitly recover observation-capacity failures,
retaining their input contract and distinguishing resource limits from engine
errors. Default capacity rejection and fatal buffer/malformed-payload errors
remain intact. The new typed `ObservationWriteError` required the naming scenario
writer's `AgentError::from` conversion, repaired at `bca261295`; the release
extension was rebuilt and atomically installed. Retained permanent growth and
sharded recovery fixtures coexist with v8 naming capacity coverage. The recorded
post-sync focused run passed 25 cases covering those paths, v1–v8 nondefault
axes, action-prefix parity and the combined native/model/vector proof. Earlier
Academy and broad-suite results predate this sync.

Main also advances the separate MTG-213 pilot to protocol v10. Its decisions
remain at `ee85d6fc8:scratch/questions.md`; this Task has no pilot execution
authorization. Historical schema-7 frame comparisons preserve all decision/offer
fields, excluding only the existing schema/fingerprint transition and never
rewriting frozen receipts.

Jack Heart's accepted full post-rotation Standard direction supersedes the old
Wave's restriction to Allies/Lessons. The roadmap treats MTG-243 as a foundation
and anticipates naming integration for prepared spells; it does not add prepared
spell execution or broad card authoring to this Task. MTG-245 owns training
methodology, and the roadmap's later training campaigns supply no execution
authorization here. Its API sketches remain proposals.

Main `8e8d702f8` capacity binding and explicit first-legal policy remain
incorporated. MTG-241's latest coordination reports `FrameCompletion`, top
ordering code 27, `SelectCards` code 28 and `OrderReplacement` code 29, with
retained event batches, paid announcements and postambles. These changes are
not synced here: this checkout still has `FrameFinalize` and maximum obligation
26; naming retains modal obligation 10. There is no reported ordinal collision,
but combined behavior and observation compatibility await ordinary sync.

## Design and implemented ownership

`ContentPack::name_candidates` owns immutable sorted public names and a digest.
`Decision::NameCard` shares that vocabulary; only the accepted choice constructs
one `ExileNamed` and resumes the existing suspended frame. Commands retain the
same modal order and labels. `SearchNamed` retains private inspection, remaining
selections and fail-to-find semantics. Search, replay, forks, LLMs and Academy
all retain the complete atomic choice; filtering never commits a choice.

Explicit input v8 carries `name_candidates[A,2]`: a categorical public name and
optional one-based definition join. A is max(configured actions, vocabulary).
Names never consume card slots. Fresh defaults remain v6; v1–v7 shapes and
omitted checkpoint defaults remain frozen. Default overflow rejects; explicit
first-legal may truncate legacy inputs and is not complete naming support.
V8 still reserves the installed naming vocabulary under that policy.

The action networks now gather only legal rows, share object/program computation,
and scatter logits to original coordinates, including holes and reorderings.
Flat and compound policies use one action-embedding owner. Continuation checks
select join columns before gathering rows, avoiding repeated full payload copies.
Prepared inference retains its policy-scoped program cache.

`PackedActions` owns offsets, original slots, complete legal action fields and
full behavior probabilities/log probabilities. `_Pending` retains those rows;
`RolloutBatch` concatenates them in time/trajectory order and keeps object fields
on CPU. Both learners validate all behavior in bounded CPU chunks and materialize
only selected minibatches. Returns, references, selected indices and numerical
invariants remain unchanged. Numerical incidents (schema 2) keep compact offsets,
slots, capacity and distributions plus failing minibatch evidence. Recovery
snapshots still occur only with empty pending/transition storage.

The equal-rule fixture reproduced identical logits for two differently named
public decklists before identity reached the object encoder. V8 now adds the
same name vectors to public definition references before card/known-hand/decklist,
continuation and history pooling. Raw program embeddings remain the candidate
scorer's semantic input, avoiding a duplicate name addition. No hidden membership
is inferred; native closed decklists remain zero. Definition-free names add no
object identities. The finalized unreleased v8 checkpoint binding includes
`naming_representation=public-identity-v1`; earlier v8 previews and changed
names/order/definition joins fail before weights load. No warm-start migration.

Schema 8 / capture 5 / state hash 16 remain the original Task transition.
Historical capture 3/4 explicitly select schema 7 for read-only validation and
preserve their original identity/digest. Schema 7 rejects new name references;
live readers still require schema 8. Retained schema-7 replay fixtures now assert
exact current scalar/vector frames and unchanged offers/decision fields, with
only the declared schema/fingerprint transition excluded. Frozen files are intact.

## Delete — do not maintain

Completed: owned per-name effect branches and per-resolution sorting; batch-wide
padded action networks; dense retained action/distribution padding; whole-rollout
learning device expansion; stale tests demanding schema-7 bytes from schema 8.
Removed `RolloutBatch.from_dense`: its two test fixtures now use the production
`PackedActions.pack` entry point directly. Boolean/integer indexing already owns
its NumPy output, so packing and pending object slices need no second copy.
Inspection tests reuse one materialization or read retained object fields directly;
the history fixture materializes only its selected row. No further production
deletion target remains in this design.

Removed the one-forward timer/print from `test_catalog_naming_programs_affect_scores`;
the bounded probe below owns cost measurement. The equal-rule preference test
now builds only its naming observation, without the unused search/ordinary roots
from `_mixed_naming_batch`. Academy's recovery setup no longer includes the
unreachable naming branch in its pre-naming action preferences. No acceptance
assertion was removed. `EmbeddedObservation.programs` now names the raw executable
table explicitly, separate from the public name-enriched object embeddings; this
changes no checkpoint parameter, tensor operation or gradient path. Benchmark
documentation now matches build-before-timing and retained-sample diagnosis.

The only dense scoring reference is `tests/model/naming_reference.py`, frozen
from the published action computation for numerical checks. No production fallback
or alternate learner was added. The scratch packing prototype is retained only
in `.lf/logs/naming-risk/prototype-probe.py`.

## Evidence and limitations

Directed checks cover mixed naming/search/ordinary observations, sparse action
coordinates, attention/FiLM on and off, values and parameter gradients, prepared
inference, malformed joins, full compound naming support, exact storage/copy
ownership, both loss gradients, bounded minibatch materialization, numerical
incidents and native collector recovery. CI now selects the packed-action and
hand-calculated credit tests explicitly. The controlled 928-name vocabulary now enters native authority through
`scenario_naming_vocabulary`, sharing the checked-in Scryfall fixture across
Rust, scalar/vector encoding, models and Academy. The model test no longer
fabricates observation rows. A policy-selected row beyond 511 executes through
scalar Commands, vector buffers and the Academy WebSocket; states/encodings
agree after execution. Separate choices at 0/511/512, Gray Ogre and the real
Fire Lord Sozin back face retain fork/replay/stale checks. Gray Ogre's private
search preserves remaining selections, reconnect, duplicate/stale rejection and
committed exiles. Names without matches preserve native trivial-decision
skipping. The fixture changes only a pending scenario decision; the installed
catalog, deck admission and subsequent normal game setup stay unchanged.

The equal-rule competency check changes only two installed encoder token rows,
not native definitions or deck admission. Without the public identity join the
paired logits are identical. With it, a two-example least-squares fit of the
existing final linear head yields opposite preferences; gradients reach names
and program tokens, reordering preserves preferences, and closing the differing
decklists makes logits identical. A preceding 16-step Adam probe did not converge.
This demonstrates representability, not SGD convergence, useful exploration,
transfer, sample efficiency or game strength. The action-type-uniform reference
still offers no relevance prior within a single naming action type.

### Bounded cost observations

`uv run --extra dev python scratch/naming-risk-probe.py` compares the published
Python model against this implementation on the same four real observations,
one CPU thread, width 16, attention + FiLM, one warmup and three measured passes.
No optimizer, stress workload, oversized allocation or training experiment.
Results are in `scratch/naming-risk-results.json`. Native paths affected only by
the historical reader change are reported as current attribution, not a speedup.

Legal supports are [344, 21, 4, 4]: action networks process 373 rows rather than
1,376 (2,048 with interior padding). Action/distribution storage is 237,568 →
43,308 bytes (81.8% less); including unchanged object fields, 620,480 → 426,220
bytes (31.3% less). Live observation buffers still occupy 604,096 bytes for these
four rows. Dense output logits and minibatch axes still scale with vocabulary.

Elapsed milliseconds, median [min, max] of three observations:

| Phase | Published 9db184b9 | Retained |
| --- | ---: | ---: |
| Full forward | 96.754 [24.060, 133.317] | 28.311 [20.172, 41.945] |
| Backward only | 793.138 [690.672, 886.133] | 554.426 [417.337, 656.149] |
| Forward + backward | 843.185 [539.592, 983.388] | 810.091 [771.600, 887.931] |
| Action networks | 0.796 [0.639, 1.210] | 0.389 [0.370, 0.390] |

These timings are variable and establish no end-to-end throughput improvement.
Before the relation-copy refinement, the first pass observed forward 23.368 →
21.835 ms and forward+backward 305.510 → 307.333 ms; action networks were then
0.561 → 0.661 ms. Preserve that contrary evidence rather than treating the later
samples as an established speedup. Full program encoding/backprop remains the
principal measured cost. Model setup medians are 34.081 → 31.682 ms after warmup.

The installed catalog has 356 executable definitions, 41,892 expanded tokens
and a 713-token maximum program. Ordinary/naming subsets request 4 / 344
programs (190 / 41,159 tokens). The final program-encoding medians are
0.999 / 39.977 ms before and 1.079 / 70.980 ms after, with wide sample variation
and unchanged program computation. The 928-name fixture adds mostly names
without programs; it is not a 928-program measurement. CPU is the configured
probe target; CUDA is unavailable and available MPS was not exercised. GPU
correctness, peak memory and intended-size learner throughput remain unmeasured.
Current native projection, encoding, fork and decision-projection samples, and
packing/materialization times, are recorded separately in the JSON.

Published CI run `41914195610` passed Engine Budget with unchanged digests:
search 116.194 → 121.779 s (+4.8%), worst reply 3.417 → 3.621 s (+6.0%),
steps 18,749.7 → 17,511.8/s (−6.6%). Its reference is `ea316ff14`, not this PR's
immediate base `8e8d702f8`; its whole delta cannot be attributed to naming. Later
Linux measurements appear below. Prior native allocation attribution remains at the
published design reference above; no naming-engine algorithm changed this pass.

## Review and earlier validation

The concept review of `d11ca430d` retained Jack Heart's accepted design with no
open product choice. Filtering commits nothing; one Command chooses the name,
then the existing private search commits exiles or stops and shuffles. Reconnect
restores the pending decision and inspection permissions; it does not promise
process-crash durability. Learner recovery still requires empty pending storage.
Production admission of additional names and default adoption of v8 remain
separate decisions. Detailed review and earlier check output are retained at
`2acc50a37:scratch/mtg-243-naming.md`.

At that revision, full debug Rust passed (970 tests, two ignored), as did
Clippy/formatting. Broad Python checks exposed packed-rollout comparison and
float32 equality assumptions, repaired at `4e7e4ea94`. Two recovery failures
came from committing while their parent cached the old source identity; both
passed in a fresh process without relaxing admission. CI-selection/release
checks passed (58), and Academy labels passed (four). The authored-match receipt
refresh changed only source fingerprints; Commands, viewer projections, events
and state witnesses remained identical, and the three receipt tests passed.
Frozen evidence was unchanged. These are earlier-source results, not acceptance
of the current repair.

The publication check at `2acc50a37` stopped on a real gap: the large-vocabulary
model test fabricated observation rows while vector and Academy still used the
344-name installed catalog. The accepted repair below replaces that split proof.
PR #424's head is now `c52b70a53` (verified 2026-11-04), with auto-merge disabled.
Its description still calls the >512 proofs separate and omits the compact
storage/scoring, public identity repair and diagnostic limitations. Its older
suggestion that the hash transition requires a budget-reference adoption is
unsupported by the matching benchmark digests; the reference remains unchanged.
CI run `41942374422` has passed Engine Budget, Rust, Academy and browser
checks. Main Linux integration completed with one float32 equality failure
(1,098 passed, one skipped), repaired below; subsequent job steps were skipped.
The retained measurements are analyzed below; this is not full acceptance. The existing
`scratch/naming-risks.html` assessment and prior `scratch/pr-review.html`
walkthrough remain review artifacts without new rendering proof.

## Accepted repair pass — 2026-11-04

Jack Heart accepted the design and requested repairs before shipping, explicitly
asking whether CPU load explained the failures (comment d4db22c5). Implementation
continues within this Task; the selected Graph owns subsequent gate, publication
and interactive review. No merge, auto-merge or strength experiment is authorized.
The pre-repair uncommitted direction is preserved at
`.lf/logs/naming-plan-before-repair.md`; failing CI run `41925981431`, head
`96b055105`, is retained as historical failure evidence.

### Completed repair simplifications

Completed this pass: remove the model's fabricated large-vocabulary action rows
and the duplicate Rust fixture builder in favor of one native scenario fixture;
replace the platform-default least-squares fit with checked SVD; remove bitwise
float assertions for reordered legal-row GEMMs while keeping the >=95% learned
preference tests; replace candidate/build-reference/reference timing with a
fixed two-order measurement. Complete legal choices, original Commands,
behavior/gradient/privacy assertions, failed measurements, the 10% budget,
exact digests and reference-build-failure candidate recording are preserved.
The reduction also removes the duplicate sampler digest/count predicate: both comparisons and
sample aggregation use `digest_checks`. Native fixture injection retains the
shared vocabulary through `Env`; only the Python list boundary copies entries,
so vector injection discards no copied name list. Rust tests call the fixture
owner directly, without a forwarding wrapper or unused configuration argument.
The SVD fit supplies its own rank check; the separate rank decomposition is
removed. No remaining deletion target is known. Production naming, model weights,
frozen schema/evidence and the budget reference are unchanged.

### Repaired proof and numerical diagnosis

The combined native/model/vector/Academy path passed locally before the latest
sync against the actual 928-name scenario decision. Academy uses the same model-selected index above
511 and compares its resulting native state and observation with scalar replay
and vector execution. The separate matching-name path exercises committed exile
and reconnect; back-face and definition-free choices remain distinct.

The equal-rule feature matrix has singular values 2.657710831 and
0.002939572 (condition ~904), rank 2, and six exact zero columns. The pinned
Linux x86 `torch==2.10.0+cpu` reproduces default `gelsy` rank 0 and a 0.001
margin residual on this exact exported matrix, including a zero-first column
permutation. `gelsd` returns rank 2 and residual <3e-17. macOS 2.10 and newer
ARM Linux 2.15 do not reproduce that QR failure; platform/version matters.
The test now explicitly selects SVD and checks rank, conditioning and residual
before installing the fitted head. It retains identity/program gradients,
permutation invariance, positive complete support and closed-information parity.
This is a solver-sensitive test failure, not evidence of CPU load or lost model
capacity. The failed earlier Adam probe and limits on SGD/strength claims remain.
Raw numerical evidence: `.lf/logs/naming-fit.pt`, `naming-linux-x86-fit.log`,
`naming-linux-fit.log`. Two ordinary-choice erased-channel ties now admit
float32 rounding (2e-6 absolute, 1e-6 relative); their 0.95 preference thresholds
and native execution assertions remain strict. Both preference directions pass
locally after the repair.

### Engine Budget diagnosis

The old CI failed only step throughput: 23489.9 -> 21126.6/s, below 21354.5;
search total 93.734 -> 101.409 s, worst 2.733 -> 2.975 s. Digests matched.
There is no retained utilization telemetry, so historical CPU contention cannot
be claimed. The reference is `ea316ff14`, not this PR's immediate base.

Both binaries are now built before timing. `engine_budget.py sample` executes a
fixed candidate/reference/reference/candidate plan, retains every raw sample,
CPU user/system time, context switches and host load / Linux pressure counters,
and compares median durations (inverse median time/step for throughput). All
sample digests and workload counts must agree; tolerance remains 0.10 and the
reference is untouched. No adaptive retries, artificial load or discarded
outliers. Failed reference builds/runs still retain the completed candidate's
recording, then fail the job. Injected executable fixtures cover aggregation,
changed digests, unchanged slowdown rejection and failure propagation.

The fixed local paired run completed before the `ee85d6fc8` sync, with no other
test/compiler/diagnostic workers started by that pass during timing. It is not a
measurement of `bca261295`. Source reference was extracted with
`git archive` into `.cache/naming-budget-reference`, not a second Task/worktree.
Raw evidence and executable hashes: `.lf/logs/naming-budget-paired/`.

| Local metric | Reference | Candidate | Existing budget |
| --- | ---: | ---: | --- |
| Step throughput | 29,983.0/s | 28,338.2/s | pass |
| Search total | 81.658 s | 87.025 s | pass |
| Worst search reply | 2.181 s | 4.290 s | **fail** (limit 2.399 s) |

All digests and workload counts agree. Candidate search totals were
78.541 / 95.511 s; worst replies 2.309 / 6.498 s. Whole-process CPU totals
were 77.841 / 79.652 s, while process wall times were 78.852 / 95.557 s;
involuntary switches rose 27,186 -> 52,177. This shows scheduling contributes
to the local elapsed-time variation, but lacks per-reply CPU accounting to
attribute the entire 6.498-second spike. Reference CPU totals were
75.381 / 74.763 s; the candidate's ~4.9% higher aggregate CPU time is not an
isolated naming-only cost or a demonstrated >10% code regression.

An important distinction: reference step process wall times were 4.780 / 3.464 s
but the benchmark's timed step loop was 3.379 / 3.401 s, with CPU 3.371 / 3.381 s.
The large process-launch delay did not enter its reported throughput. Thus
observing host contention alone cannot explain the old CI step failure.
The failed worst-reply result is retained without retries, baseline movement or
budget relaxation. Final-source Linux CI remains required; this local run does
not clear Engine Budget or justify accepting a persistent cost.

The pre-sync focused repair suite passed 24 cases across budget fixtures, large
naming support, equal-rule preferences and Academy recovery; the debug native
naming tests passed two cases. The release extension, Ruff/format and diff
checks also passed then. Details remain in
`.lf/logs/naming-plan-before-compress.md`; these are reused evidence, not new
current-source suite results.

The sample collector added typed telemetry fields and a fresh-artifact guard after
the actual timing run; the reduction shares existing digest/count validation.
No workload, aggregation or tolerance changed, and no timing was repeated.
The local budget failure above remains visible. `lf context` reports memory and
scratch below both token and byte targets after condensing the earlier review.

### Published `bca261295` Linux measurement — 2026-11-04 (run dated November 5 UTC)

Run `41942374422` completed Engine Budget successfully for `bca261295` on
one AMD EPYC 9V74 runner. The downloaded artifact is retained at
`.lf/logs/naming-budget-final-linux/`; comparison of those existing bytes passes
all digests, counts and the unchanged 10% timing limits. No benchmark was rerun,
no sample was discarded, and no reference, tolerance or frozen evidence changed.

| Linux metric at `bca261295` | Reference `ea316ff14` | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Worst search reply | 3.652 s | 3.839 s | +5.1% |
| Search total | 123.708 s | 129.587 s | +4.8% |
| Step throughput | 17,712.8/s | 16,404.5/s | -7.4% (+8.0% time/step) |

The two candidate step loops took 6.196 / 6.197 s and used 6.196 / 6.197 CPU
seconds; reference loops took 5.721 / 5.757 s and used 5.722 / 5.758 CPU
seconds. Candidate search used 129.093 / 130.082 CPU seconds versus
129.129 / 130.139 process wall seconds; reference used 123.448 / 123.976 CPU
seconds versus 123.459 / 123.998 wall seconds. Candidate timings agree across
both orders (step throughput within 0.003%, search totals within 0.8%).
The host-wide CPU pressure counter is not this process's scheduling delay.

Thus CPU load and ordering noise do **not** explain away the repeatable
candidate/reference cost in this run. It is within budget and includes the
intervening engine changes since `ea316ff14`; it is not isolated to naming.
The earlier local 6.498-second spike and historical CI throughput failure remain
retained contrary evidence with incomplete attribution. This Linux
run does not reproduce either budget violation, but cannot retrospectively
prove their causes. No speculative runtime optimization is justified by these
samples alone. Native diff inspection found no new catalog enumeration in
ordinary search rollouts; naming remains a separate suspended-decision path.

The same run passed debug Rust, unit, long Linux integration, Academy,
browser, clean-machine and conformance jobs. Main Linux integration completed
with one failure, 1,098 passes and one skip; the later evaluation/report/smoke
steps were skipped. Logs are retained under
`.lf/logs/naming-final-{rust,budget,integration}.log`.

The remaining failure was the closed-information assertion in
`test_equal_rule_names_learn_opposite_public_preferences`: 234/512 logits
were not bitwise equal, with maximum absolute difference 1.4901161e-7. The SVD
rank/residual checks, opposite preferences, gradients, complete support and
permutation check had already passed. This is a numerical equality failure,
not a timing/load symptom. The test now first requires every closed input row
to be bitwise identical, and compares the two privacy/control logit pairs with
2e-6 absolute / 1e-6 relative tolerance, consistent with the existing reordered
row checks. The 0.001 fitted margins and all learning/support assertions stay
unchanged. No production model, native engine, budget or frozen evidence changed.

The focused test passed locally; Ruff check/format and diff checks passed.
A bounded probe exported the real closed fitted-head inputs/weights and executed
only that linear operation on pinned x86 Linux `torch==2.10.0+cpu`. Its inputs
and outputs were exactly equal; it does not isolate the full model's CI rounding
difference to the last linear layer. Evidence is retained in
`.lf/logs/naming-closed-head.pt` and `naming-linux-closed-head-internal.log`.
An earlier container attempt could not unpack Torch through the host-mounted
cache (too many open files); the internal-cache attempt completed and its
container exited. No host limits were changed or artificial load introduced.
The complete repaired test still requires Linux gate/CI validation. The repair
and subsequent simplifications are committed at `bca3ff438`; the current
`c52b70a53` sync adds configuration changes only. Existing local validation
passed all 17 catalog-naming cases and all three Academy named-search recovery
cases after those simplifications. Those results are reused here, not rerun.

### Concept review at `c52b70a53` — 2026-11-04

Outcome: nothing to change in the accepted design and no open product choice
within this Task. Jack Heart's acceptance in comment `d4db22c5` remains the
authority. This review inspected the current implementation, consumer tests and
usage guidance; it ran no gameplay, model, benchmark or browser checks.

The experience remains one deliberate name choice followed by the existing
private search. Filtering changes visible offers without submitting a Command.
The immutable vocabulary identifies public names; optional definition joins
provide learning semantics without admitting cards. A bound Command selects one
candidate and resumes the suspended frame with one `ExileNamed`. A stale Command
must leave that frame unchanged. Retained-session reconnect restores either the
uncommitted naming decision or the search's remaining choices and inspection;
committed exiles survive and stopping search cannot undo them. This does not
promise process-crash recovery.

The consequential model distinction is still public name identity versus hidden
membership: equal-rule cards need distinguishable visible identities, while a
closed opponent's deck supplies none. V8's shared name vectors and strict
vocabulary binding express that distinction. Complete legal-row scoring and
packed behavior storage remove padding work without changing choices, original
slots or learner evidence. The repaired native scenario now lets all consumers
exercise the same large vocabulary; it adds no production admission lifecycle.

Counterexamples remain a missing score beyond 511, preferences changing with
offer order, hidden membership affecting closed inputs, a reconnect remapping a
Command, or packing losing behavior mass. Existing directed checks target these
contracts; their current-source acceptance remains below. Solver rank failures,
float32 output differences and scheduling variation are separate evidence
questions, not reasons to weaken the contract or the 10% Engine Budget.

`docs/training.md` and the package READMEs already describe explicit v8 and the
installed-catalog boundary accurately. No usage or skill redesign is proposed.
Broader production names, fresh-default adoption and training strength remain
outside this acceptance. The next proof is the existing gate, not a new product
decision. The refreshed `scratch/concept-review.html` records this disposition;
it is an illustration without rendering proof. No Graph navigation, publication
or merge was performed by this step.

### Remaining acceptance and review

Gate owns affected suites, full debug Rust/Clippy, current-source receipts,
browser coverage and Linux validation of the test-only repair, including the
evaluation/report/smoke steps skipped after the failed integration step. Current
CI `41945242063` is running on `c52b70a53`; no result from it is claimed.
Engine Budget passed on `bca261295`, whose native engine and sampler are
unchanged at the current head. The retained local worst-reply failure still has
incomplete attribution, and no out-of-budget persistent cost has been accepted.
Publication preparation still owes PR #424's description and
review artifacts an account of the repaired tree and remaining limits.
Interactive review is the accepted endpoint; prior green Graph steps do not
establish current acceptance. GPU behavior, intended-size throughput and strength
remain unmeasured. No new product decision is needed for installed-catalog
naming; additional production names and default v8 adoption remain separate.

Check: `uv run --extra dev python scripts/engine_budget.py compare --artifact .lf/logs/naming-budget-final-linux --reference .lf/logs/naming-budget-final-linux/reference` — PASS; same comparison on `.lf/logs/naming-budget-paired` — retained FAIL (worst reply 4.290 s > 2.399 s); no benchmarks rerun. Current-source Linux/full-suite acceptance remains with gate.
