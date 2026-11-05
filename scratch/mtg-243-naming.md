# MTG-243 — complete catalog naming

## Accepted direction — 2026-11-04

Jack Heart authorized complete, learnable catalog-wide naming beyond 512 names
as one reviewable PR, with no merge or training experiments. After publication
of `9db184b9aa3df9286b07abdafa2d41db0d93e651` in PR #424, Jack Heart requested:
“Let's make sure to both measure those risks and attempt to ameliorate them.”
That accepted pass covers legal-row scoring, compact pending/rollout storage
through both learners and incidents, bounded same-host costs, paired public-state
competency, actual executable-program attribution, and the five CI compatibility
failures. The risk-reduction implementation is committed at `5a57096f2` and
main `1546aae3f` is merged at `d11ca430d`; publication of these changes and
interactive review remain outstanding. No auto-merge, paid execution or strength
experiment is authorized.

The original implementation and publication evidence are retained at
`9db184b9:scratch/mtg-243-naming.md`; the uncommitted research/accepted-pass text
was preserved at `.lf/logs/naming-risk/plan-before-reduction.md` before curation.
The 2026-11-04 reconciliation follows the accepted Task directive and records
durable lessons in the selected coverage Wave through the code Graph.
Earlier T7 direction is historical at
`4463e57f5:scratch/questions.md`.

## Reconciled upstream context — 2026-11-04

Main `1546aae3f` adds the published Standard roadmap and
`managym/PRINCIPLES.md`, renames the Wave directory from engine to coverage,
and introduces separate engine-performance and infrastructure scopes. The sync
changes no runtime or tests relative to `5a57096f2`; existing focused results
remain applicable, with full acceptance still outstanding. Coverage has no
immediate child memory directories; the two sibling memories contain no lessons
yet. Both memory and scratch fit their configured context targets.

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

## Completed simplifications

Completed: owned per-name effect branches and per-resolution sorting; batch-wide
padded action networks; dense retained action/distribution padding; whole-rollout
learning device expansion; stale tests demanding schema-7 bytes from schema 8.
Removed `RolloutBatch.from_dense`: its two test fixtures now use the production
`PackedActions.pack` entry point directly. Boolean/integer indexing already owns
its NumPy output, so packing and pending object slices need no second copy.
Inspection tests reuse one materialization or read retained object fields directly;
the history fixture materializes only its selected row. No further production
deletion target remains in this design.

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
immediate base `8e8d702f8`; do not attribute its whole delta to naming. The new
head still requires CI. Prior native allocation attribution remains at the
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
PR #424's published head remains `96b055105`; its copy needs the repaired proof,
compact storage/scoring, identity fix and measured limitations. The existing
`scratch/naming-risks.html` assessment and prior `scratch/pr-review.html`
walkthrough remain review artifacts without new rendering proof.

## Accepted repair pass — 2026-11-04

Jack Heart accepted the design and requested repairs before shipping, explicitly
asking whether CPU load explained the failures (comment d4db22c5). Implementation
continues within this Task; the selected Graph owns subsequent gate, publication
and interactive review. No merge, auto-merge or strength experiment is authorized.
The pre-repair uncommitted direction is preserved at
`.lf/logs/naming-plan-before-repair.md`; published failing CI remains
run `41925981431`, head `96b055105`.

### Delete — do not maintain

Completed this pass: remove the model's fabricated large-vocabulary action rows
and the duplicate Rust fixture builder in favor of one native scenario fixture;
replace the platform-default least-squares fit with checked SVD; remove bitwise
float assertions for reordered legal-row GEMMs while keeping the >=95% learned
preference tests; replace candidate/build-reference/reference timing with a
fixed two-order measurement. Preserve complete legal choices, original Commands,
all behavior/gradient/privacy assertions, failed measurements, the 10% budget,
exact digests and reference-build-failure candidate recording. The reduction
also removes the duplicate sampler digest/count predicate: both comparisons and
sample aggregation use `digest_checks`. Native fixture injection retains the
shared vocabulary through `Env`; only the Python list boundary copies entries,
so vector injection discards no copied name list. Rust tests call the fixture
owner directly, without a forwarding wrapper or unused configuration argument.
The SVD fit supplies its own rank check; the separate rank decomposition is
removed. No remaining deletion target is known. Production naming, model weights,
frozen schema/evidence and the budget reference are unchanged.

### Repaired proof and numerical diagnosis

The combined native/model/vector/Academy path passes locally against the actual
928-name scenario decision. Academy uses the same model-selected index above
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

The fixed local paired run completed, with no other test/compiler/diagnostic
workers started by this pass during timing. Source reference was extracted with
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

Check: `uv run --extra dev --extra play pytest tests/infra/test_engine_budget.py tests/model/test_catalog_naming.py::test_catalog_naming_large_support_learns_and_rejects_wrong_binding tests/model/test_catalog_naming.py::test_equal_rule_names_learn_opposite_public_preferences tests/academy/test_named_search_recovery.py -q` — 24 passed; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib flow::naming_tests` — two passed; release extension rebuilt and atomically installed, Ruff/format/diff checks passed. Prior repair check details are retained in `.lf/logs/naming-plan-before-compress.md`; gate owns broad suites and final-source CI.

The sample collector added typed telemetry fields and a fresh-artifact guard after
the actual timing run; the reduction shares existing digest/count validation.
No workload, aggregation or tolerance changed, and no timing was repeated.
The local budget failure above remains visible. `lf context` reports memory and
scratch below both token and byte targets after condensing the earlier review.

### Remaining boundary

Gate owns affected suites, full debug Rust/Clippy, current-source receipts,
browser coverage and final-source Linux CI / Engine Budget once. Implementation
runs only focused changed-behavior checks and the specifically requested bounded
diagnosis. Final publication must use the complete repaired tree, update PR #424
and stop at interactive review; previous green Graph steps or old CI do not
establish that outcome. GPU behavior, intended-size throughput and strength
remain unmeasured. The local review page must distinguish the repairs from the
still-published failing head; it supplies no new browser rendering proof.
