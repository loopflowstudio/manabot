# Step 3 continuation — reconciled 2026-11-08

Jack Heart replied "keep going" after the evidence-review recommendation:
"I recommend repairing BF16, lengthening the packed cells, and completing the missing measurements."
Comment 079b3f46 covers this bounded continuation; comment 8cb4c5af reviews C1
against that authority, not a new approval. The prospective design is in
`experiments/protocols/training-throughput-b015d5-v1.md`, **Step 3 continuation C1**.
It owns the schedule, identities, admission, readings and spending limits.

## Outcome and retained limits

Measure the missing packing crossover and a384l8, and repair/validate BF16 on
real CUDA. These are speed and gross numeric-health results, not strength or
learning-equivalence claims. The longer-term planner remains outside C1 software.
Jack Heart's direction to use effective hashed records remains in force; exact
quotes and the separate c5fba4 diagnostic are in `scratch/vocabulary-performance.md`.

The original w17 / cb722dc8c evidence stays unchanged: 11 valid solo cells,
seven insufficient-overlap packed cells, six BF16 startup failures and no a384l8
measurements. TF32 clip fraction moved 3.45 times the float32 repeat difference.
Step-3 spend $1.741 and experiment spend $3.781 are estimates, not final invoices.
Original reports, failures, run identities and cleanup receipts remain published
under `experiments/data/training-throughput-b015d5-v1/step-3/` and its parent report.

C1 retains the Elemental learning recipe, 256 streams, both seats, complete-game
closure, one producer, input v6 / 512 facts, effective hashed records, all other
capacities and precision defaults. New source/world controls are mandatory;
original receipts cannot serve as matched controls after integration.

- Fresh seeds 26401–26404; 26400 for CUDA admission, 26410 for diagnostics.
- Eight-update speed cells; at least three complete overlapping post-warmup
  updates per seed. One separate twelve-update whole-cell overlap retry, with
  a matching solo endpoint where necessary. First valid scheduled attempt wins;
  attempts are never pooled or selected by speed.
- One rental first: real CUDA regression and a two-update w160l8 BF16 worker
  plus normal checkpoint reload. Each additional GPU needs the CUDA regression
  before BF16. A384l8 solo runs early. Memory admission stays at 85%; K=2
  crossover at 1.10 and K=4 admission at 1.30 remain conditional execution rules.
- A40's w160l8 float32/repeat/TF32/BF16 block remains four twelve-update legs.
  W64 supplies a missing lower bracket only after higher-priority cells.
- Caps include setup/failures/diagnostics: $10 step 3, $50 experiment; new
  liability at most min($10 minus prior step 3, $50 minus prior total, $8),
  15 GPU hours and 1,600 updates. Proposed lines: A40 $3.30/5h, A4000 $1.60/6h,
  L4 $2.40/4h, $.70 contingency within those windows. No fourth GPU.
- Zero evaluation games, draws or checkpoint selection. Training exposure,
  failed attempts/replacement debt and whole-rental bills remain separate from
  completed speed-cell clocks. All rentals require confirmed provider deletion.

## Implemented, awaiting delivery and CUDA admission

The ed2300 repair is in this checkout: semantic definitions cast recurrent
outputs to the embedding table's dtype, including the empty-row contract,
without detaching gradients. Float32 output/gradient parity and injected Half/
BF16 joins have CPU tests. A mandatory CUDA test covers full, selected, empty,
cached, backward, optimizer and reload paths and prints the actual GRU dtype.
The CPU test no longer claims to certify CUDA kernels. Actual CUDA dtype and
end-to-end BF16 success remain unmeasured for C1.

The launch owner preserves legacy four-update jobs and adds C1 identities,
eight/twelve-update endpoints, fresh seeds and a two-update admission job.
The admission job runs the CUDA test, ordinary worker and hash-verified raw
checkpoint loader. Discovery rents nothing. Cost admission is arithmetic;
execution still owns quote, memory, crossover, deadline and cleanup decisions.

The maintained `experiments/training_throughput_readings.py` reads only the C1
cohort, enforces recipe/source/host identity and complete ordinals, matches solo
and packed ordinals, selects the first valid scheduled attempt, and requires a
disposition for every cell. It reports per-seed/window sensitivity and whole-rental
accounting separately. Reconciliation repaired the missing numeric-health games
and decisions/game: both derive from cumulative exposure differences, not learner
rows. Precision differences include repeat-relative magnitudes, null at a zero
repeat difference; near-zero denominators still do not support equivalence.

The two optional diagnostics operate on one retained 256-row observation fixture,
one reaped child and 120 seconds each. Recording separates allocation/first touch/
resident copying; transfers separate selection, host gather and H2D wait. They
preserve copied values and do not reproduce a full update or explain the original
recording stall. No speculative production speed repair was added.

Removed: the unsafe mixed-dtype definition join and CPU-to-CUDA coverage claim.
No production path, schema, historical reader or evidence was deleted. No deletion
targets remain.

## Integration and remaining work

Main e0832f67f was merged as fec7fdd28, bringing 5da5f4's v7 shared-effect batching
and GPU report. Its one-matchup pool/GRU study establishes neither an architecture
advantage nor equivalence, and is not an Elemental hashed-v6 speed control.
Its retained traces caution against assuming synchronization dominates host routing.
The separate c5fba4 catalog repair retains its owner; this integration is not proof
of that repair or its CUDA speed. C1's source comparison must use fresh controls.

The implementation-time ed2300 read on 2026-11-08 found it unstarted with no PR,
Sessions or Graph processes; no competing repair was launched. Linking ed2300 to
the next b015d5 software PR and recording validation there remains delivery work.

Remaining acceptance:
1. Publish the verified software, then complete its delivery review. Local
   verification is recorded below; no final freeze is recorded.
2. Freeze the integrated source, native world/definitions, roster, resolved recipe
   bytes and capacities, dependency lock and bundle digest. A roster, layout or
   learning-recipe change requires redesign; no old identity substitutes for it.
3. Reconcile prior invoices or reserve defensible outstanding liability, refresh
   quotes under the existing isolated client/home, then perform CUDA admission
   on one bounded rental. Admission failure requires repair and a new freeze.
4. Execute admitted missing cells within all caps, retaining skips, failures,
   exact ordinals, exposure, cleanup and bills; publish C1 evidence separately and
   return its measured recommendation and unresolved questions for evidence review.

No new design decision is identified. Material scope/budget departures remain
separate decisions; existing "keep going" authority covers routine C1 work.
No rental, quote, bill refresh or scientific run occurred during reconciliation.

Check: `uv run --extra dev pytest tests/training/test_throughput_c1_readings.py tests/training/test_training_throughput_sizes.py::test_c1_discovery_preserves_controls_and_declares_endpoints tests/model/test_semantic_card_precision.py -k 'not cuda_semantic_policy_admission' -q` — 13 passed, 2 CUDA cases deselected; real CUDA remains execution admission.

## Concept review — 2026-11-08

No design change or unresolved product choice. Jack Heart's "keep going"
continues the bounded measurement outcome above; it does not select a precision
default or turn throughput into a strength recommendation. The existing C1
software-interface instructions already describe the intended operator experience.

The consequential distinction is **job completion versus usable measurement**:
the first pass completed seven packed cells without sufficient overlap. Preserve
three concepts: a declared cell with its attempts, a derived reading, and the
rental carrying all liability. `Leg`/`runs` describe execution;
`Disposition` retains its outcome; `read_cell`/`compare` derive evidence from
receipts; `Rental` and `admit_cost` account for shared machine time. Combining
these into one success flag would lose required behavior. No additional planner
or campaign lifecycle is needed for C1.

Concrete walkthrough: after freeze and CUDA admission, run the declared solo
and K=2 cells on one host, retain their receipts, and compare matched ordinals.
If K=2 completes without three overlapping updates per seed, retain completion
and `insufficient_overlap` together. Admit only the declared whole-cell u12
retry, with a matching fresh solo endpoint if needed; keep both attempts and
use the first valid scheduled one. A second insufficient reading leaves the
crossover unknown. This counterexample forbids pooling attempts or picking the
fastest result. A CUDA admission failure instead stops expansion and requires
repair/new freeze; an unconfirmed deletion keeps rental liability outstanding.

Ownership stays with manabot for training and readings, Etude for launching,
and the execution operator for conditional scheduling, bills and confirmed
cleanup. Original evidence and failed-attempt exposure survive every recovery.
The next proof remains the existing software gate followed by frozen-source
CUDA admission, not another product decision. This review launched no tests or
rentals and changed no implementation. `scratch/concept-review.html` presents
the same contract for inspection.

## Publication verification — 2026-11-08

`lf sync` completed cleanly against main e0832f67f. The full affected run
(model suite, active training, precision, size jobs, C1 readings, CI selection,
release and untracked-artifact checks) returned 379 passed, 9 skipped, one
failure in `test_token_preserves_action_focus_indexes[1]`. The identical
2.0954757928848267e-09 mismatch reproduced with main's semantic-card module
loaded in memory: dense and compact float32 scoring round differently near
zero. The test's absolute floor is now 1e-8 (relative tolerance unchanged).
Value-aggregation and semantic precision rerun: 22 passed, two CUDA skips.
An in-memory one-row shift of focus embeddings made both focus tests fail;
the adjusted tolerance still detects incorrect indexing. No production scoring
change was made.

The exact core CI command collected 1,235 tests. Ruff passed on all changed
Python files; `git diff --check` passed. All 326 changed artifact files match
their manifest byte counts and SHA256 hashes; no manifest entries were removed.
The two real-worker size smoke tests passed on CPU. CUDA admission was not run;
its two local skips do not admit BF16 execution. No rental or paid run occurred.

The ed2300 status read still reports unstarted with no PR. Its repair is delivered
only here; publication will link the PR and these validation limits there.

## Gate — 2026-11-09

Comparison: PR #451 against `main`, including uncommitted gate fixes. GitHub
integration (core/evaluation/long) and Linux/macOS package/install checks passed
at `f88a2aef6`; those unchanged results remain evidence. The repository's
`.github/ci/select-tests.sh` selects integration only for this diff; it has no
local content-cache runner. No Rust, frontend, world or observation shape changed.

Review found a deterministic post-worker admission failure: `StageRecord.artifacts`
contains dictionaries, while the new script used attribute access. Admission now
validates the existing `ArtifactReference` type and uses its mapping fields.
Missing optimizer counts and missing stages reject clearly. A synthetic receipt
with a real small checkpoint covers reload and corrupt-byte rejection on CPU;
it does not certify CUDA training.

Focused validation plan (before running):
- `manabot/model/semantic_cards.py` and changed model tests: semantic-card,
  semantic precision, inference and value-aggregation tests cover dtype joins,
  gradients, cached callers and the float32 tolerance.
- Size launch, admission, readings and diagnostic scripts: size-launch tests
  (including two CPU worker smokes), C1 reading fixtures and precision tests cover
  discovery, real receipt/checkpoint loading, overlap, accounting and context wiring.
- Active-training and regime tests cover the required model-cost/discovery checks;
  CI-selection/release and untracked-artifact tests cover the workflow and manifest.
- Ruff on changed Python files and `git diff --check`; CUDA-only tests remain
  deferred to the protocol's required frozen-source rental admission. Existing
  CI owns the broader integration matrix. No paid measurement or artificial load.

Results:
- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/model/test_semantic_cards.py tests/model/test_semantic_card_precision.py tests/model/test_inference.py tests/model/test_value_aggregation.py tests/training/test_precision.py tests/training/test_training_throughput_sizes.py tests/training/test_throughput_c1_readings.py tests/training/test_active_training.py tests/training/test_regimes.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/infra/test_experiment_data_untracked.py -q` — 154 passed, 2 CUDA skips, one new-fixture failure in 187.96s; both CPU worker smokes passed.
- `uv run --extra dev --extra artifacts pytest tests/training/test_training_throughput_sizes.py::test_admission_reloads_receipt_artifact_and_rejects_unhealthy_or_changed_bytes -q` — 1 passed in 2.29s after correcting the fixture's explicit match binding and required identities field. These were fixture-construction errors, not relaxed admission. Other suites' inputs stayed unchanged; their results were retained.
- `git diff --name-only origin/main...HEAD -- '*.py' | xargs uv run --extra dev ruff check --fix` — passed, no fixes; `git diff --check` — passed.

Final focused coverage: 155 passed, two CUDA-only cases deferred. No check timed
out, no performance improvement is claimed, and no rental was started. The reader
usage example now includes required `--source`. Delivery must publish these gate
fixes and obtain CI for the new revision; the earlier green PR checks cover
`f88a2aef6`, not the uncommitted gate changes. Continue the software Graph only;
do not complete the parent experiment. CUDA admission, freeze and billing remain
the existing execution checks after software review.
