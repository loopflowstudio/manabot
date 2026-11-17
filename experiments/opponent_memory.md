# Whole-game decision memory

## Question

**Exploratory.** Can viewer-safe memory of the opponent's past decisions become
an affordable training basis without materially hurting throughput or learning?
The true incumbent has history disabled. The control retains and validates the
log and reserves its vocabulary but has no memory encoder. The active comparison
is control versus GRU-K1, with the incumbent retained as the throughput reference.
Opponent-only is primary; own-seat inclusion remains built and default off.

## Answer so far

The software works, but the required matched-learning comparison has not run.
The frozen supervised probe selected K1 by its validation/CPU tie-break rule:
57 completed heuristic games, 1,812 decision rows, three capacity exclusions,
94 hand targets, whole-game splits and separate open/closed readouts. K4−K1
validation loss was +0.002763 [−0.000620, +0.006803]. This is a scripted proxy
with a limited state baseline, not evidence of playing strength or adoption.
The probe protocol, outputs and competence limits are in the
[dated result](#supervised-probe-result--2026-11-16).

Gated summary tokens preserve exact control initialization; finite gate gradients
precede encoder gradients after the gate opens. Common-trunk transfer, published
producer binding, ordinary checkpoint persistence and blank-memory evaluation
intervention are implemented. Native snapshots remain the sole viewer-safe
fact authority; hidden probe labels never enter policy inputs or caches.

The expanded-reference CUDA run completed two updates and finite raw/EMA reload
for all six frozen workers. Its matched-host incumbent median was 350.15 learner
decisions/s, control 187.96 (0.537×) and GRU 124.57 (0.356×). References had
observed headroom at 128 of 256 slots; all memory-worker maxima fit
512/16,384/256/8,192 record/detail/reference/member capacities. These are short
worker/resource admissions, not matched learning or strength results.
The [CUDA closeout](#expanded-reference-cuda-closeout--2026-11-17) retains full
host, source, trajectory and resource limits. Its source predates CPU host
optimizations, so those worker rates do not measure the chunk cache.

The [chunk-cache candidate](#record-aligned-chunk-cache-candidate--2026-11-17)
reduces CPU fixture preparation and yields 1.14–1.20× GRU fixture speedups with
exact feature/output/gradient/state comparisons. GRU remains 0.30–0.46× the
matched incumbent fixture rate. Warm cache arrays add 61.6 MB for 256 prefixes;
raw transport and cached features are now budgeted separately and the cache is
kept. The [native-epoch suffix capture](#native-epoch-suffix-capture--2026-11-17)
cut then reduces whole-game CPU capture time 2.5–3.1×, with exact behavioral and
gradient hashes. Control's total rate is 0.963–0.980× the incumbent on the
two-stream fixture, while GRU remains 0.429–0.490×. No optimized CUDA-worker or
learning result follows.

## Current method and limits

Jack Heart's 2026-11-17 direction is throughput first: "the major concern is
throufhput w decision memory" and "we cannstart w gru only". No further paid
calibration or pilot launches until host cost is reduced. The already-running
refs256 run finished unchanged, including its frozen transformer worker; all
rentals ended. Transformer implementation and frozen evidence remain available,
but no transformer rerun or paid arm is planned.

CPU work uses the frozen 256-prefix fixture drawn from replayed Elemental census
Commands. Compare incumbent, control and GRU at the same 32/128/256 execution-row
limits. Record preparation separately from inference and learning; retention
contains capture and must not be added to capture as disjoint time. The first
vectorized program expansion preserved exact arrays, outputs, gradients and
weights and improved this fixture. It did not establish acceptable worker cost.
The current cut admits record-aligned immutable chunks by content and predecessor
cursor, caches vocabulary-bound features, then assembles batch offsets and
prefix-relative recency. Dense imports still receive complete validation.

The approximately 10% throughput guide informs review. After the host chunk cut,
collection now uses native epochs/counts for suffix capture. Recency-independent
record embeddings and collection/learner folds remain separate measured steps.
The Task session's engineering direction permits one admission at or below USD1
inside the calibration allowance only after CPU GRU is within about 1.3× of the
incumbent; current CPU results do not meet that gate. A pilot still needs a person
to see the numbers. Coalescing/multiplicity, capacities and world binding stay fixed.

If paid work resumes after reviewing these costs, use three independent seed
labels with a common EMA trunk per label and control/GRU branches: three trunks
plus six branches. Freeze trunk length, gate scale and learner-decision horizon
from affordable execution measurements first. Strict capacity failure ends a run;
never truncate history or discard failed games while continuing scored training.
Train mixed per-game disclosure through the landed native API. Compare branches
at matched actual learner decisions, retaining overshoot and explicit checkpoint
interpolation rather than relabeling update counts.

Evaluate paired deals with seats swapped, separately for open and closed lists;
retain descriptive matchup intervals, one heterogeneity test and endpoint intact
versus blank-memory reliance. Whole training seeds are the evidence unit. There
are no completed pilot seed comparisons, paired evaluation deals, learning curves
or strength intervals. Disclosure-only training arms (B1) and full-layout input
ablations (B2) remain deferred inside the same budget. Earlier five/three-arm
plans and evaluation game counts in dated entries are historical declarations,
not observed samples or current launch authority.

## Cost

Launcher closeout records **USD1.94475 across five calibration runs**, including
failed attempts. The first three statuses are final; the last two remain launcher
estimates. All five rentals ended. Costs are not reconstructed from rates or
wall times. The frozen probe and CPU work used no rental. The inclusive authorized
ceiling remains USD50; its allocation is a budget, not measured spending.
No further paid work is launched by this report.

## Recommendation

Keep the history-disabled incumbent as the default. The software and short CUDA
admissions support continuing host optimization, but current worker evidence
shows a substantial throughput regression and supplies no learning comparison.
Do not adopt memory or remove recent events on this evidence. Review the matched
CPU chunk-cache costs before repricing a control/GRU pilot. A default change needs
affordable whole-training cost and the missing matched-learning and paired
strength evidence, with failures and uncertainty retained.

The dated appendix retains original observations, including results superseded
by later measurements. The earlier undated protocol and storage counterexample
can be recovered from `7d680aac170f12ca60e60dfb812e0c00d7cda7fa:experiments/opponent_memory.md`.

## Appendix: execution evidence and dated changes

2026-11-15 execution check at `1ecbde8b4348c8ce1871cf8deb5272e71b3e8537`:
the locked Etude client (`d60236d015a4b2a01055b2c861856f06bb1852a2`),
run through an isolated `uv run --no-project --with` environment, returned no
rows for `etude runs --experiment opponent_memory`. Its
`etude status --experiment opponent_memory` returned `last loop: never`;
this does not establish live provider rental state. No launch or rental was
requested in this execution step, and there is no run ID or launcher cost to
record. Execution is blocked: this committed revision still explicitly raises
in both `runs()` and `eval()` because the shared decision log, native/Python
inputs, transformer/GRU encoders and concrete training/evaluation jobs are
missing. Deliver that software before the replay census and calibration; the
existing inclusive USD 50 approval remains sufficient, with no new approval
needed. No experiment result or adoption decision follows from this check.

No paid launcher run ID or cost exists. Census and controlled storage evidence
are retained below. The memory-software PR, calibrated horizon, concrete launch
wiring and evaluation jobs are outstanding; no original Command corpus was found. No paid run is authorized by this design file
beyond the task's existing inclusive USD 50 authority.

Design validation (2026-11-15): Ruff check/format and syntax checks passed;
Etude discovery matched the exact inventory using the locked SDK commit
`d60236d015a4b2a01055b2c861856f06bb1852a2`, and the explicit pre-software launch
stop was exercised. Full `tests/eval/test_experiment_files.py` and report
construction await the development environment/native extension. An initial
isolated import failed at missing `managym._managym`; a discovery attempt using
the older PyPI SDK missed existing experiment files. Neither is experiment data.

2026-11-15 reporting revision: reorganized the design into question, planned
method, results, support, recommendation and caveats. No new run or result was
added. The missing software and measurements remain outstanding; no result
files were produced for publication, so the artifact manifest is unchanged.

2026-11-16 kickoff reconciliation: five arms now replace the earlier six-arm
proposal, following the recorded transformer/GRU × 1/4 summary direction. The
planned count is 15 training runs and 19,200 evaluation games; the inclusive
USD 50 allocation is unchanged. No census, training or evaluation was run.
The shared implementation and gate plan is `scratch/opponent-memory-design.md`.


2026-11-16 census source substitution: PR #549 retains aggregate displacement
statistics and a reproducible seed harness, but no Command corpus was located in
its body, comments or changed-file inventory. Source
`95b8d4d0bbf16b3eef30a127561b7d6b46c1b10f:managym/examples/bench_steps.rs`
specifies seeds 0–511, per-game ChaCha8 uniform legal-index choices, a 2,000-step
cap and the curated UR Lessons/GW Allies matchup. The no-rental census will
regenerate that workload and compare its step digest with the retained reference.
This is a dated corpus substitution, not replay of retained original Commands.
The census reports capacities without policy tensor allocation; it supplies no
learning or throughput-adoption claim.


2026-11-16 implementation checkpoint: native memory tests (6), structured
Command tests (10), focused Python memory/transport/update checks (29), text and
package-boundary checks (101), artifact tracking and native observability checks
passed. Both encoders completed the bounded 16-row exact-quota CPU update;
optional-layout/legacy and malformed-word admission checks also passed. Strict
library Clippy, Ruff and formatting passed; the CPython 3.13 extension was rebuilt
and replaced atomically. Full gate/CI/CUDA admission and the demo remain pending.
That checkpoint sent the storage revision back to review. The subsequent shared
storage cut and its resource limit are recorded below; software delivery and
the experiment remain incomplete.


### Shared-prefix storage proof, 2026-11-16

The transport now owns immutable record/detail chunks and persistent tails.
Coalescing versions a tail; forks, resets and subsequent updates cannot mutate
already sampled rows. Only selected prefixes are padded. Collector recovery
serializes reachable chunks once and validates them against native journal
replay. Incidents preflight complete selected prefixes against the 32 MiB
budget; missing evidence is explicit. Configured observation capacities and the
independent dense encoders remain unchanged.

Command: `uv run --no-sync python scripts/verify_opponent_memory_storage.py --source experiments/data/opponent-memory/20261116-census/summary.json --out experiments/data/opponent-memory/20261116-storage/receipt.json`.
The manifest-backed receipt is published in the artifact store and names source
byte hashes. It uses a **controlled valid synthetic prefix** with 595 records
and 665,431 distinct detail facts, then retains 16,384 decision references over
its immutable tails. It does not reconstruct the original corpus or create a
full padded rollout. The retained census counts imply 1,807,045,512 terminal
payload bytes before interning across its 1,024 viewer logs; counts alone cannot
establish their actual cross-record deduplication.

| Component | Bytes | Meaning |
| --- | ---: | --- |
| Unique controlled chunks | 21,336,632 | Measured ownership of the one synthetic prefix and all its tails. |
| Prefix/tail/array metadata | 4,117,503 | CPython ownership accounting for the controlled decision references. |
| Intern indexes | 196,168 | Retained weak lookup indexes, separately counted. |
| One selected raw gather peak | 21,344,084 | Host allocation measurement at the maxima. |
| One host preparation peak | 89,325,995 | Host allocation measurement, excluding the already retained raw gather/model. |
| One prepared memory input | 37,306,992 | Sum of its array byte sizes. |
| 256 native padded memory buffers | 8,608,808,960 | Configured buffers remain dense. |
| 256 selected raw prefixes | 5,462,177,792 | Arithmetic envelope if every selected prefix has both maxima. |
| Two host batches, raw + prepared | 30,025,535,488 | Includes one-ahead preparation. |
| One device memory input | 9,550,589,952 | Tensor byte envelope, not measured CUDA residency. |
| 16 duplicate incident prefixes | 21,565,792 | Shared archive fits 32 MiB; exact tail identities round-trip. |

**Dependent wiring remains stopped.** The original accounting summed a
48,210,584,703-byte (44.9 GiB) envelope and compared it with the declared
42,949,672,960 bytes (40 GiB). The concept review correction below supersedes
that comparison: the envelope combines host and device allocations, while the
execution limit checks host RSS. The retained receipt bytes are unchanged.
It excludes ordinary observations, native game logs, preparation temporaries,
transfer staging, model/optimizer and attention/gradient allocations. This is a
conservative synthetic envelope, not a measured CUDA peak or evidence that
ordinary census minibatches require this amount. It establishes neither a safe
batch size nor throughput. The draft still scans/hashes the dense native prefix
on capture; sharing its ownership does not remove that work.

Focused storage/model/native parity checks pass, including both encoders and
summary counts, both viewers/disclosures, coalesced versions, duplicate/shuffled
gathers, bootstrap, frozen inference, later-update lifetime, serialized supported
collector replay, corrupt/missing chunks and incident preflight. Existing recovery,
closure and numerical-health checks also pass. Software gate/CI, headless demo,
CUDA calibration, calibrated Etude declarations, paid training/evaluation and
software delivery remain outstanding. No rental or scored run occurred.

### Concept-review correction to resource accounting, 2026-11-16

The retained 44.9 GiB total combines host and device bytes. The recipe's
`Execution.memory_bytes` limits process-tree host RSS, so comparing that sum
with 40 GiB does not establish a host-cap failure. The receipt remains unchanged;
its accounting also excludes the padded bootstrap arrays retained through the
update, preparation temporaries and detail activations. It establishes no fit.

Keep dependent wiring stopped. Next implement direction: retain bootstrap
prefixes and gather them only for admission/value evaluation; account for host
and device peaks separately, including collection, frozen inference, bootstrap,
learner prefetch and backward. A smaller learner minibatch alone cannot bound
the other forwards. The controlled fixture's shared tails do not estimate
independent-game deduplication. Record explicit execution row sizes before
calibration; preserve full histories, learner clocks, five arms and USD 50 cap.
No runtime change, new test run or paid experiment occurred in this review.

### Bootstrap lifetime and separate execution proof, 2026-11-16

Bootstrap now retains ordinary next-state fields and immutable prefixes, with
no padded memory arrays in the returned batch. Admission and preparation gather
temporary arrays; weak-reference checks confirm those arrays die while Prepared
bootstrap input remains live. Learner prefetch releases its raw memory arrays too,
keeping ordinary action/reference fields and immutable prefixes alongside Prepared.
Both encoders and K values pass exact Prepared
parity for mixed empty/duplicate rows, value parity, native-buffer reuse,
collector journal restore and bounded incident capture. Collection/frozen/
bootstrap fixtures use one stream; the ordinary update uses 16 learner decisions
and four-row learner minibatches. These are explicit proof sizes, not pilot sizes.
No optimizer schedule, learner clock, history capacity or arm changed.

Command: `uv run --no-sync python scripts/verify_opponent_memory_storage.py --source experiments/data/opponent-memory/20261116-census/summary.json --out experiments/data/opponent-memory/20261116-bootstrap/receipt.json`.
The [new receipt](data/opponent-memory/20261116-bootstrap/receipt.json) is published
through the artifact manifest. It preserves the original storage receipt and
its source identities unchanged. One controlled prefix contains all 595 records
and 665,431 details, under the same 1,024/1,048,576 bounds. It shares tails across
16,384 references; unique chunks are not an independent-game estimate.

All four memory arms completed a full policy/board forward, backward and Adam
step with finite outputs/gradients/weights and nonzero encoder gradients at
**one execution row** with **two host Prepared batches**. Width is 64, float32,
CPU with one Torch thread. This is an operational numerical proof; it does not
measure playing strength or throughput. Per-phase receipt samples cover model,
preparation, prefetch, device input, forward, backward and optimizer.

| Encoder | Summaries | Cumulative process peak RSS (bytes) |
| --- | ---: | ---: |
| transformer | 1 | 1,821,507,584 |
| transformer | 4 | 1,832,337,408 |
| gru | 1 | 1,844,281,344 |
| gru | 4 | 1,844,576,256 |

These are cumulative process high-water marks, not independent arm peaks or
phase deltas. Allocator retention carries across arms. Phase-end process-tree
RSS is also recorded; without sampling, it cannot establish transient tree
peaks. On this CPU run input tensors share host arrays. CUDA allocated/reserved
current and peak bytes, and device capacity, are null: no CUDA hardware was
available and no GPU fit follows. The same proof command supports `--device cuda`
for a later admitted device; it does not replace real-worker admission.

For the **unexecuted** 256-stream/256-selected-row worst case, host memory
components total **33,197,816,959 bytes**: 8,608,808,960 native memory-buffer
bytes, 24,563,357,696 for a temporary raw gather plus two retained host Prepared
memory batches, and controlled prefix ownership/indexes. Bootstrap raw arrays
exist only during admission/preparation and precede learner prefetch; they are
not an additional update-long owner. Prepared bootstrap tensors still live
through their value forward, and every bootstrap call still uses all trajectory
rows. Native complete-game logs, ordinary fields, preparation temporaries and
model/optimizer/activation memory are additional. This partial ownership total
is **not** a host-limit admission, even though it is below the 40 GiB RSS guard.

Device input memory separately totals **9,550,589,952 bytes** at 256 selected
maxima. Just one width-64 float32 detail activation would add **43,609,686,016
bytes**, before other detail intermediates, encoder/board activations, gradients,
optimizer and allocator reserve. Do not sum these device numbers with the host
limit. The census terminal payload context remains 1,807,045,512 bytes without
interning; the small shared fixture cannot predict 256 independent games.

Focused checks: `uv run --no-sync pytest tests/training/test_memory_prefix_storage.py tests/model/test_opponent_memory.py::test_opponent_memory_collects_bootstraps_and_learns_from_real_prefixes tests/training/test_update_backend_protocols.py::test_prefetched_minibatches_preserve_order_and_policy_input -q` — 14 passed; the final bootstrap/learner lifetime, mixed-row recovery and real-update checks passed 6 cases.

Keep experiment wiring stopped. Explicit device limits and concurrent lifetimes
for actual collection, frozen inference, bootstrap, admission and learner sizes
still need CUDA/worker calibration before choosing job sizes. No execution
chunking or smaller pilot workload was silently substituted. The five arms,
matched learner decisions, full histories and inclusive USD 50 cap remain.

### Explicit execution limits, 2026-11-16

Implemented separate `Execution.collection_rows`, `bootstrap_rows` and
`learner_rows` limits, required for training with opponent-memory buffers,
including the no-history control. Limits select prefix references before raw
materialization, admission, preparation and forward. Frozen opponents use the
collection limit. Bootstrap concatenates only small predictions. Sampling keeps
one backend RNG call per original group; learner chunks retain selection/order,
separate logical actor/critic denominators and one clip/Adam step per logical
minibatch. PPO's actual prefetch releases its previous completed Future before
submitting another chunk. No history capacity or logical workload changed.

Command: `uv run --no-sync python scripts/verify_opponent_memory_storage.py --source experiments/data/opponent-memory/20261116-census/summary.json --out experiments/data/opponent-memory/20261116-execution-final/receipt.json --collection-rows 2 --bootstrap-rows 2 --learner-rows 2`.
The [execution receipt](data/opponent-memory/20261116-execution-final/receipt.json) is
manifest-backed; all 16 recorded source hashes matched the measured source at
that checkpoint.
The earlier census, storage and bootstrap receipts remain unchanged. The
preliminary `20261116-execution/receipt.json` is retained too; the final receipt
repeats execution after correcting source docstrings for the chunked boundary.

This controlled CPU proof runs the actual collection, frozen-inference,
bootstrap, admission, PPO preparation and update paths at **2/2/2 execution
rows**, with **five logical optimizer rows**, a final one-row chunk, **three
simultaneous policies** (learner, bootstrap/EMA and frozen opponent), Adam and
one-ahead host preparation. Every selected prefix contains all 595 records and
665,431 details. All four arms finish with finite outputs, gradients and weights,
nonzero encoder gradients, five optimizer exposures and exactly one Adam step.
Width is 64, float32, one Torch thread. This is a numerical/resource proof, not
a throughput, learning or playing-strength measurement.

| Encoder | Summaries | Sampled concurrent process-tree RSS peak (bytes) | Cumulative self high-water RSS (bytes) |
| --- | ---: | ---: | ---: |
| transformer | 1 | 3,205,169,152 | 3,208,380,416 |
| transformer | 4 | 3,221,864,448 | 3,221,864,448 |
| GRU | 1 | 3,235,430,400 | 3,235,463,168 |
| GRU | 4 | 3,204,202,496 | 3,235,463,168 |

Tree samples occur every 20 ms across the complete model/update lifetime;
phase samples and the self high-water mark complement them. Sampling can miss
short transients, and allocator retention carries across arms. On CPU tensors
share host arrays. CUDA allocated/reserved current and peak bytes and device
capacity are null because CUDA is unavailable here. No host/device sum or GPU
fit claim follows.

For the unexecuted 256-stream native-buffer envelope, the bounded execution
memory components total **8,826,360,495 host bytes**, including native buffers,
the controlled shared-prefix payload/indexes and the largest raw/Prepared
execution overlap. This partial arithmetic excludes native game logs, ordinary
fields, preparation temporaries and model/activation/optimizer memory. It is
not a host-limit admission. Shared tails still cannot estimate independent-game
storage; the full 256-stream workload was not executed. The old 256-selected-row
33,197,816,959-byte host envelope remains explicitly unexecuted in the receipt.

Focused parity covers both encoders/K values, duplicate/empty/shuffled prefixes,
unequal chunks, scalar/categorical bootstrap, both filter scopes, clipped PPO
and damped updates, frozen RNG partition and three-stream supported collector
restore. Post-Adam parameters and logical diagnostics agree within numerical
tolerances; model batching is not promised bitwise identical. Near-constant PPO
advantage normalization can amplify ordinary floating-point forward differences;
this is not evidence of learning equivalence.

CUDA worker/update/checkpoint reload admission and independent-game resource
accounting remain prerequisites for calibrated Etude wiring. The five arms,
matched learner decisions, full histories and inclusive USD 50 cap are unchanged.
No training/evaluation rental or pilot ran in this iteration.

Validation: `uv run --extra dev pytest tests/training/test_memory_execution_rows.py -q` — 18 passed. Simulation/collection, precision and CI-selection checks — 76 passed. The required backend checks passed after moving malformed-bootstrap rejection coverage from construction to the bootstrap read; MLX cases were skipped because MLX is unavailable. Retained prefix/recovery and ordinary memory-update checks passed. Training CI includes both memory test files; its exact training invocation collected 423 tests. Artifact tracking, Ruff and `git diff --check` passed. Full gate and CUDA admission remain outstanding.

### Independent-game host measurement, 2026-11-16

Select **2/2/2 collection/bootstrap/learner execution rows** for the pilot's
initial CUDA admission. Keep 256 streams, the 64-decision floor, complete-game
closure, full 1,024/1,048,576 record/detail capacities and all five arms. These
are concrete resource limits awaiting device admission, not a changed learner
clock or optimizer schedule.

The first host attempt reached complete-game closure, then failed before model
execution: Torch incorrectly required memory heads whenever observations
retained logs. Its [failure](data/opponent-memory/20261116-independent-host-attempt1/failure.json)
and measured source are retained, without a host-peak claim. The existing
admission path now derives retained input from observation capacities and
learned memory from summary count. The no-history control retains and validates
logs but creates no memory parameters or prepared memory features. Its outputs
match the incumbent exactly, and ordinary checkpoint reload preserves its input
space. Focused model/preparation checks: 54 passed.

Command: `uv run --no-sync python scripts/verify_opponent_memory_storage.py --independent-games --source experiments/data/opponent-memory/20261116-census/summary.json --out experiments/data/opponent-memory/20261116-independent-host/receipt.json --collection-rows 2 --bootstrap-rows 2 --learner-rows 2`.
This mode uses independently seeded native games over the uniform Elemental
roster, four production native producers and uniform legal actions. It retains
every sampled viewer prefix through the floor and complete-game closure. The
production ordinary-row recorder starts with its 25% spare capacity; this
measurement also retains dense legal-action fields for resampling, rather than
claiming an exact production packed-action footprint.

All native buffers and recorded rows stay live while each arm samples, runs
frozen inference, bootstraps and executes PPO with three concurrent policy
copies, Adam and actual one-ahead preparation. One 256-row logical minibatch
selects each stream's largest observed prefix. Update targets are a fixed
numerical fixture, not game outcomes. This measures concurrent host
ownership with independent games; it does not execute the whole cohort's
on-policy worker update, change the pilot's logical minibatches, or measure
learning or throughput. Learned-game tails can differ. CUDA peak measurement,
ordinary worker/update and checkpoint reload remain prerequisites before Etude
wiring, within the existing USD 3 calibration allowance and inclusive USD 50 cap.

The [successful receipt](data/opponent-memory/20261116-independent-host/receipt.json)
retains **52,304 decision prefixes** from **259 completed games**, with 64–597
decisions per stream. Selected complete prefixes contain 9,214–579,591 detail
rows. Native memory buffers occupy **8,608,808,960 bytes** (all native fields:
8,666,700,288 bytes). Retained unique chunks occupy **867,925,248 bytes**, plus
79,950,282 bytes of Python metadata and interning indexes. Ordinary recorder
allocation is 18,524,405,760 bytes, including unused spare pages and the dense
action fields; allocation is not the same as resident memory. The store
receipt's `selected_padding_bytes` is arithmetic for gathering every retained
prefix together; that padding was never allocated.

The measured **concurrent process-tree peak is 29,347,774,464 bytes (27.33 GiB)**,
below the declared 40 GiB host limit for this cohort. The process high-water RSS
agrees; its peak occurs during cohort recording/growth. After complete-game
closure, phase-end RSS is 22,421,291,008 bytes. Each arm below retains that cohort
and the full native buffers through its three-policy execution.

| Arm | Sampled concurrent host RSS peak (bytes) |
| --- | ---: |
| No history | 23,021,436,928 |
| Transformer, one summary | 26,183,073,792 |
| Transformer, four summaries | 26,068,205,568 |
| GRU, one summary | 25,943,195,648 |
| GRU, four summaries | 26,023,559,168 |

All five arms complete with finite outputs, gradients and weights, 256 optimizer
exposures and one Adam step; all four memory arms have nonzero encoder gradients.
The control's serialized summary setting is ignored: it produces zero memory
tokens. RSS is sampled every 20 ms; short transients can be missed and allocator
retention carries across arms. High-water RSS is cumulative, not an independent
arm peak. This is float32 CPU execution with one Torch thread. All CUDA allocated,
reserved and peak fields are null. The 670.59-second measurement is not a training
throughput result, and the observed host fit does not guarantee learned-game fit.

Both attempts are manifest-backed. Each source patch reconstructs its measured
checkout from `b859eb68dd41f09f16f790a887b6e1a70c05f56e`; companion source metadata
records patch, lock and native-extension hashes. All seven source hashes in the
successful receipt matched the checkout at that measurement checkpoint. Earlier
receipts remain unchanged. No
rental, paid calibration, learning pilot or evaluation ran. Etude job construction
stays stopped pending CUDA peaks, the ordinary worker/update, checkpoint reload
and matched-decision horizon calibration.


Validation: `uv run --no-sync pytest tests/model/test_opponent_memory.py tests/model/test_prepare_rows.py -q` — 54 passed; artifact tracking guard — one passed. Ruff, patch reconstruction check and `git diff --check` passed. Full gate remains outstanding.

### Prompt, headless proof and collection attribution, 2026-11-16

The headless demo now runs with
`uv run --extra dev python -m manabot.verify.opponent_memory --out .runs/opponent-memory-demo`
(use a fresh directory). Its [retained summary](data/opponent-memory/20261116-headless-demo/summary.json)
records both disclosures and all four memory arms. Constructed priority roots
use trivial skipping; subsequent choices are ordinary replayed Commands. Both
viewers' native tensors, independent Python tensors and JSON round trips agree.
The proof includes eligible passes, private selection of a Mountain followed by
its public play without changing earlier records, a declined block, and an
isolated cast branch. Prompts, captures, viewer logs, Commands and checkpoints
are retained beside the summary; source snapshots and native-extension hashes
are in `source.json` and `source/`.

Every arm has distinct legal probabilities under intact versus blank memory,
finite nonzero temporal-encoder gradients from actor and value losses tested
separately, and matching checkpoint reload. Probability maximum deltas range
from 1.06e-5 to 1.09e-4. These use a declared nonzero FiLM weight intervention to
exercise connectivity; they are not evidence of learned memory use. A separate
ordinary GRU local-smoke update starts from the recipe's normal initialization,
collects 193 learner decisions with complete-game closure and census capacities
at 2/2/2 execution rows, completes one update, and reloads finite weights. It
establishes neither CUDA admission nor playing strength.

`manabot-text-8` shares identical historical reference renderings and rules
snapshots within the prompt. Every chronological record, reference group,
multiplicity, span, life/hand value and board count remains listed; phase is now
explicit beside turn and step. Repeated-snapshot fixtures at the census maximum
595 records and configured 1,024 records fit the existing 262,144-byte budget.
The census has counts rather than original record bytes, so these are controlled
size fixtures, not retained long-game evidence. Arbitrarily distinct snapshots
can still exceed a byte/token budget and fail with `RepresentationOverflow`;
no history is hidden, truncated or silently retried. Tokenizer and chat-template
admission remain the harness's responsibility.

Real-game text size, 2026-11-16: uniform legal play (`random.Random(seed)`,
not the census schedule) in the curated UR Lessons versus GW Allies matchup,
rendering the acting view every 100 steps over the seeds reached in 200 seconds,
sampled 616 decisions with no overflow. The largest was 484 records at 209,614
bytes; a least-squares fit gives about 40 KB plus 382 bytes per record. That
projects the budget being crossed near 580–600 records, at the census maximum
(595; p99 442), and about 430 KB at 1,024 records. The repeated-snapshot
fixtures fit because they cycle 87 records' references; they understate
distinct long games. This was an unretained laptop check, not a receipt.

The earlier QA result (control collection 13.9 s versus incumbent 2.3 s) remains
an uncontrolled CPU observation, not a CUDA throughput estimate. A bounded
[attribution probe](data/opponent-memory/20261116-control-cost/attribute_cost.py)
replays 160 uniform choices in the same seeded game, encoding the acting view
into native dense buffers and retaining every control prefix. Both configurations
finish at state digest
`dddebeb9072740e1b61eb46509b3f3b1b5744c5324a31f12d285e8993195ec7c`,
with viewer record counts 67/87. It profiles no policy inference, vector worker,
bootstrap or optimizer. Census capacities are 1,024/1,048,576.

The [before receipt](data/opponent-memory/20261116-control-cost/before.json)
measures control native observation/encoding at 0.415 s and prefix capture at
0.904 s, versus incumbent encoding at 0.026 s. Python profiling attributes
0.839 s cumulatively to `admit_memory`: it repeatedly checks finiteness and
integrality over the full padded detail capacity, allocating full-size floor
results. Admission now checks unused rows once for zero padding (also rejecting
NaN, infinity and fractional garbage), then checks numeric values on the owned
prefix. The rejection contract is unchanged; directed tests cover live and
padding corruption in both companion buffers.

The [after receipt](data/opponent-memory/20261116-control-cost/receipt.json)
measures control encoding at 0.395 s and prefix capture at 0.436 s; incumbent
encoding is 0.027 s. Profiled admission falls to 0.378 s, with 0.319 s still in
array `any` reductions, chiefly the mandatory zero-padding check. Native encoding
still rebuilds every historical detail row and clears the configured buffers.
The capture reduction is local attribution with profiling overhead and host
variation; it does not erase the earlier sixfold collection result or establish
a whole-update speedup. Measure the actual worker against the incumbent during
calibration, including these remaining costs.

Focused validation: 82 model/native/demo/boundary tests and 78 text/prefix-storage
tests passed; Ruff and `git diff --check` passed. No Rust changed. The artifacts
are manifest-backed. CUDA resource/worker/reload admission, matched-decision
calibration, Etude wiring, gate and the capped five-arm pilot remain outstanding.
No paid allocation was made.

### Lossless long-game context rendering, 2026-11-16

Text representation `manabot-text-9` prints each first context fully, then only
changed turn/phase/step/life/hand/board/untapped-source fields. Its prompt explains
inheritance; choices/responses never inherit and empty groups are explicit.
Every chronological record, selector, count/span and ordered reference remains.
Historical aliases continue to resolve exact rendered snapshots. The v8 size
projection above remains historical evidence, not a current cost estimate.

The cycling-prefix tests are replaced with native games: seed 7, fixed 200-card
decks (60 Mountain, 20 Island, 120 Gray Ogre), trivial skipping enabled, build
until eight permanents then pass priority and decline attacks; other choices use
the first legal action. At 602 and 1,029 records, the full prompts measured
137,123/224,948 bytes open and 137,001/224,827 closed before a final 69-byte
legend clarification. These logs have hundreds of distinct contexts; they are
controlled long-game fixtures, not census replay or typical-game measurements.
Uniform legal curated play, seed 166 at step 600, reproduces the earlier
484-record maximum with over 250 distinct contexts and 50 distinct historical
references. Its complete new prompt fits below 180,000 bytes, versus the earlier
v8 observation of 209,614 bytes. No prefix is repeated to create either proof.

An independent test reader reconstructs all rendered context values, reference
groups (including order and multiplicity), historical descriptions, choice
selectors and count/spans against native records. Directed fixtures also check
zero/negative values, false selectors, disappearing sources and empty choices.
All 70 tests in `tests/semantic/test_text_observation.py` pass (13.60 s), including
ordinary Command binding and explicit byte/token overflow. Ruff and patch
whitespace checks pass. The 262,144-byte default is unchanged; arbitrarily
larger distinct snapshots and model token budgets can still reject a complete
prompt. This is text fidelity/size evidence, not a CUDA, throughput or strength
result. No paid run or new retained artifact was created; the fixtures above
and checked-in test source reproduce the checks. CUDA admission, calibration,
wiring, gate and the capped pilot remain outstanding.

### CPU construction baseline, 2026-11-16

**Worse:** same-host collection with retained tensor logs took **3.432 s
[3.029, 4.547]**, versus incumbent **0.986 s [0.683, 1.076]**, for identical
250-decision/two-game batches, five alternating runs per side and zero errors.
The ratio of medians is 3.48×; paired ratios have median 4.11× [2.99, 5.31].
The [baseline report](data/opponent-memory/20261116-cpu-baseline/report.md),
[phase table](data/opponent-memory/20261116-cpu-baseline/measure.html) and
[raw receipt](data/opponent-memory/20261116-cpu-baseline/instrumented/receipt.json)
retain exact commands, source/binary identities, all repeats and semantic checks.
No production optimization was made; the code Graph remains at demo.

At clean `194f20120f0a`, the bounded production collector uses two streams,
one producer, both learning seats, seed 1, complete-game closure, float32 CPU,
one Torch thread, local-smoke's normal width-64 attention/FiLM policy and 2-row
execution on both sides. Control capacities remain 1,024/1,048,576; all 250
prefixes are retained (largest 82 records/76,103 details). All ten runs have
identical action/value/logprob/reward/done and ordinary-observation digests.
Native stepping includes unconditional viewer-log capture on both sides, so
this is a tensor-retention comparison, not capture cost against a pre-feature
engine. Native build flags are unrecorded; both sides use the same extension
hash as the earlier headless proof. Background load averages rose from 27 to 39;
no load was induced. This large separation survives observed timing ranges,
but small component differences remain inconclusive.

Control phase medians are native stepping/capture 0.046 s, native observation
encoding 0.666 s, host prefix transport 1.384 s, host preparation 0.567 s
(including nested validation 0.301 s), Torch forward 0.577 s and residual
0.143 s. Native encoding and host prefix transport account for median per-run
shares of 20.7% and 42.7%. Component medians need not sum. Residual is computed
per run without assigning sampling, recording or orchestration speculatively.
Native stepping and no-head forward ranges overlap those of the incumbent.
Construction, bootstrap, optimizer and checkpoint IO are outside these timers.

The existing 160-choice seed-0 probe reproduces its original state digest and
67/87 viewer counts. Native curated seed 166 supplies retained real prefixes
at 0/34/52/484 records and 0/9,460/17,901/546,812 details. Each agrees with the
independent Python encoder and exact prefix gather. Five warmed repetitions
per operation and all six policy configurations separate native scalar encoding,
dense/compact admission, transport, production-cached preparation and forward.
Long-prefix learned preparation/forward costs are material; overlapping ranges
do not rank GRU/transformer or one/four summaries. Profiled timings are excluded
from these summaries. Separate cProfile attribution puts 1.907 of 2.209 seconds
inside prefix capture into admission across its mixed probe workload; native
sample stacks expose repeated per-detail field-name formatting/BLAKE3 hashing
inside full-prefix encoding. Dense zero-prefix admission still costs about
2.3 ms versus 0.07 ms compact admission. These are existing-path diagnostics,
not a measured production optimization.

For optimize-plan, investigate a typed compact native transport/admission
boundary that preserves the dense reference ABI and every complete immutable
prefix, plus static field-token reuse. Removing all prefix transport alone
would still leave roughly twice the incumbent collection time; forward tuning
alone cannot recover the common log overhead. Full capacities, privacy,
coalesced-tail versions, typed overflow and fork/replay identity stay required.
The unchanged-contract checks passed 16 tests. This is CPU attribution, not
256-stream admission, a whole training update, CUDA throughput, learning or
strength evidence. No rental, paid calibration or pilot ran.

### Shared snapshots and table-wide admission, 2026-11-16

Each native viewer log interns exact entitled reference snapshots once; records
own ordered mana/response/choice members and fixed context columns. At seed 166's
484-record prefix, detail rows fall from 546,812 to 4,863 and live table bytes
from 17,532,832 to 333,320. Canonical expanded histories, independent Python
encoding, prefix gather and every ordinary frozen buffer match the earlier layout.
The featurizer changed (nonlinear field/value binding before pooling, separate
member groups), so learned outputs are not comparable with the earlier layout.

Admission then replaced per-record, per-group and per-reference loops with
table-wide checks. Against the shared-snapshot commit `f5688cf686`, five
alternating CPU processes per source reduced log-control collection from
**7.353 s [6.202, 7.845] to 2.203 s [1.747, 5.803]** per 250 decisions/two games,
and 484-record dense admission from 78.301 ms to 2.596 ms. A 4,298-case
differential corpus found no acceptance or rejection mismatch; it is finite and
does not compare first-error precedence. Collector action, value, log-probability
and ordinary-buffer digests matched across sources. Learned long-prefix cost ranges
overlap and remain inconclusive. Host load ranged 37.71–192.42 without induced
load, so these are bounded CPU readings, not CUDA, learner-update, host-fit,
learnability or strength results. Two later behavior-preserving simplifications
(`Prefix.tables`, a hoisted `definition_programs()` lookup) passed the same
corpus; the final measurement re-freezes their source hashes. Receipts remain
local in `.runs/decision-memory-admission/` until evidence packaging.

## Final post-sync admission measurement — 2026-11-16

**Better for CPU log-control collection.** On the same post-sync release engine, five
alternating processes per source give **1.709 [1.672, 1.924] → 0.771 [0.690, 0.898] s
per 250 decisions / two completed games**. The only substituted production behavior is
the old versus final admission implementation. Ranges separate; the narrow admission
software remains justified. This measurement adds no production optimization and makes
no policy-adoption claim.

The final true history-disabled incumbent is **0.446 [0.420, 0.835] s**, so log control
is **1.730×** its median. The approximately 10% throughput guide is not established by
these CPU timings. Both arms still share native decision capture; the history-disabled
input does not measure a pre-feature engine. No CUDA, learner-update/backward, peak
RSS/host-fit, probe, learnability or strength result exists. Own-seat inclusion stays
default false and is uncosted.

Collection shares (median per-run fractions): native step/capture 1.9%, encoding 1.6%,
host transport 16.1%, preparation 37.2%, nested validation 23.0%, Torch forward 35.6%,
residual 7.9%. Validation is nested inside preparation and capture admission is inside
transport; do not add nested validation to disjoint totals. Removing encoding entirely
saves at most 1.6% of this path. Baseline transport+prepare occupies 79.2%; removing all
of it bounds total speedup at 4.80×. Separate profiles map the admission hot spot to
policy/decision_memory.py, MemoryStore.capture and Admission.check; production
preparation and forward remain costs. Profile wall times are excluded from all tables.

The engine was rebuilt from synced revision `7adcb0943a6d443f75c8131bafd742b34afb51b4`
in release for CPython 3.13 and atomically installed, SHA256
`8d4e83e9e0850e0c07d2837acab8203cd29847fc47de0b5aabddce9523c0722d`. Baseline uses only
admit_memory/validate_memory_vocabulary from exact commit f5688cf686 before consumer
imports; all other final Python code, current collector/journal/statistics paths and the
native binary are shared. Source/harness hashes match before/after. Frozen pre-sync
receipts are historical and untouched. No world/layout change was made in this
measurement, so no new world-shape sweep was added.

Five alternating source processes per scalar/collector arm; prefixes use five process
medians of five warmed operations each (25 samples per arm/prefix). Scalar: 160 choices.
Collection: two streams, execution_rows=2, complete-game closure,250 learner/environment
decisions, two games,146 inference calls and 42 idle slots. Prefixes: 0/34/52/484 real
records at seed 166 steps 0/80/160/600. Empty and long prefixes distinguish easier and
harder work. Float32 CPU, seeded 1 weights, cached production PolicyInference,
Torch/inter-op threads=1. All ordinary work, current recording-wrapper bookkeeping,
native capture/encoding, host capture/admission, preparation, forward and assembly
execute within collection timing; scalar wall includes ordinary digest work. Ordinary
collector digests are checked outside timing. Durable journals, native statistics and
recovery remain at their actual disabled defaults in this harness. Enabled
journal/statistics correctness is covered separately; no enabled workload was silently
timed or omitted.

Zero measurement errors, rejected games or truncations. Scalar state/ordinary digests
and collector ordinary/action/logprob/value/reward/done digests agree across both sides
and repeats. All four learned prefix outputs match exactly across validators and
repetitions; incumbent/control no-head outputs match. Untimed contracts verify 320
scalar viewer histories and 251 collector/bootstrap views per side, independent
native/Python encoding, canonical frozen histories and ordinary buffers. Every prefix
process rejects NaN padding in all four companions. The rebuilt-extension differential
proof has 4,298 cases and zero mismatches; this finite corpus does not prove exhaustive
equivalence or first-error precedence. Focused checks and exact commands are in
checks.json and check-*.log.

Shared host: Apple M4 Max / 128 GiB, macOS, CPython 3.13.15, Torch 2.15.0. One-minute
load endpoints 11.87–23.96; no induced load, discarded outliers or tuning changes. Cold-
worker/warm-prefix procedures match, but OS cache identity and scheduler contention are
uncontrolled. Overlapping metric ranges are no measurable difference, regardless of
median movement. A quieter matched-host repeat would settle those costs. Common-trunk
parameter copy/control-equivalent initialization and CUDA/update/resource admission
remain planned, unimplemented prerequisites to the learning pilot.

Commands: uv run --no-sync python .runs/decision-memory-final/checks.py; then
measure.py, which freezes identities, runs separate history/prefix contracts and
profiles, performs five alternating pairs and checks digests/source identities. All
workers are sequential and reaped. Exact source bytes, old/final profiles, raw receipts,
commands, replay/census proofs and the verified lossless 1,024-file census archive are
packaged under experiments/data/opponent-memory/20261116-final. Publication is recorded
separately in the experiment report and artifact manifest; raw local files are retained.

 ### Additional final readouts

Scalar log-path wall time is better: **0.490 [0.482, 0.594] → 0.130 [0.129, 0.151] s per
160 choices**, n=5 per side. At 484 records dense admission is **13.545 [13.269, 14.124]
→ 0.710 [0.670, 0.765] ms** (five process medians, five inner samples each). All four
484-record learned prepare+forward ranges separate: GRU-K1 **25.795 [25.602, 26.366] →
13.100 [12.454, 13.268] ms**; GRU-K4 **25.940 [25.759, 26.124] → 13.005 [12.827, 13.509]
ms**; transformer-K1 **32.919 [31.914, 33.150] → 19.616 [19.228, 20.483] ms**;
transformer-K4 **32.875 [32.095, 33.106] → 20.084 [19.392, 21.009] ms**. These are
admission-cost comparisons within unchanged learned models, with exactly matching
outputs; they do not rank encoder quality or establish learning equivalence. Prefix-only
unchanged native encoding and forward measurements cannot be attributed to validator
changes; scheduler variation remains a limitation.

Separate profiles retain 397 structural admissions on each side (251 captures +146
prepared batches), while `np.isin` calls fall 83,856→7,575 and `array_equal`
calls84,724→4,287. The final path still performs 587 parameter-version enumerations
(`policy.py:651`), as does baseline. Source mapping is `decision_memory.py:80/81`,
`memory_store.py:233`, `admission.py:84`; counts and full profiles are retained in
profile-counts.json and *-attribution.prof. Unchanged forward/cache work and shared-host
contention mean the whole observed 2.215× collection multiplier cannot be assigned to
admission alone, although its reduction is both explained and arithmetically within the
4.80× host-work bound.

Focused post-sync validation: 23 native debug tests passed; 81 journal/recording/replay
tests and 188 storage/execution/model/admission/package tests passed (one demo case
deliberately deselected); 4,298 differential comparisons, zero mismatches. No test
exceeded its watchdog. No new production code, model/layout change, probe, rental, paid
call, child agent or Graph movement occurred.

Evidence publication completed through Doppler-injected artifact CLI: 19 manifest
entries under `experiments/data/opponent-memory/20261116-final` (seven verified bundles,
per-file manifests, census proof and publication receipt). The original 18 files total
83,368,982 bytes; all match the published content-addressed manifest. Every archived
member was verified locally, including all 1,024 census logs; no fresh remote-fetch
round trip was performed. Raw logs and historical receipts remain local and untracked.
The artifact guard passed; no servers, watchers or background workers remain. This is
software/cost evidence only: the 10% guide, common-trunk/control-equivalent
initialization, CUDA/update/resource admission, probe, learnability and strength remain
unresolved.


## Supervised probe result — 2026-11-16

**Use K=1 for both pilot encoders.** The validation difference K4−K1, averaged
across both encoders and three fitting seeds, was +0.002763 log loss with a
95% game-bootstrap interval [−0.000620, +0.006803]. K4 did not meet the
predeclared improvement criterion. Aggregate fit wall time across the six
memory fits per K was 1917.7 s for K1 and 2047.2 s for K4, selecting
K1 under the CPU-cost tie-break. These are supervised fit times, not policy
collection throughput or CUDA admission.

The corrected command exited 0:

```bash
uv run python -m experiments.opponent_memory probe --max-games 60 --out .runs/decision-memory-probe-roster-20261116
```

Source `383f33c11a` integrates main `5af8490a1c9b`, including #547 and #550.
World is `w17` with the current v11 disclosure/library/memory layout;
the release CP313 extension SHA256 is
`58a9f152d8ca0bd09bbc61fb978124112593e04a2aa60926a1194c015e6c3875`.
Runtime: ordinary CPython 3.13.15, Torch 2.15.0, macOS ARM64, CPU float32,
one Torch thread. The 20-game smoke took 41.85 s to collect and 9.39 s for
one transformer epoch, admitting the predeclared 60-game ceiling. The completed
run took 4218.49 s (70.31 min), within its 7,200 s cap.
Launcher cost: USD 0; no rental or paid arm was launched.

Of 60 attempted games, 57 completed, three exceeded capacity and none hit the
2,000-decision cap. Whole failing games were excluded, with their errors and
committed action prefixes retained. The corpus has 1,812 sampled decisions and
94 target definitions from roster main decks **and sideboards**. Fit games:
19 open/20 closed; validation: five per disclosure; test: four per disclosure.
Each native seat owns a separate stateful heuristic. Labels were taken before
the corresponding action and remain outside observations, Prepared and journals.
The test split was scored only after `selection.json` recorded K.

Below: game-weighted per-card Bernoulli log loss and paired difference from
state-only; lower is better. Intervals resample whole games, conditional on the
three fitting seeds 39101–39103. There are only four test games per disclosure.

| Candidate | Open loss [95% CI] | Open Δ state [95% CI] | Closed loss [95% CI] | Closed Δ state [95% CI] |
| --- | --- | --- | --- | --- |
| analytic | 0.076012 [0.038787, 0.114522] | — | 0.170632 [0.134304, 0.206960] | — |
| state-only | 0.082362 [0.055015, 0.119659] | — | 0.137563 [0.098612, 0.176514] | — |
| transformer-k1 | 0.082100 [0.059532, 0.117065] | -0.000262 [-0.003547, +0.003640] | 0.121899 [0.085031, 0.158767] | -0.015664 [-0.020007, -0.011310] |
| transformer-k4 | 0.082033 [0.057248, 0.120137] | -0.000329 [-0.007129, +0.005664] | 0.122076 [0.083831, 0.160321] | -0.015487 [-0.023381, -0.006887] |
| gru-k1 | 0.076052 [0.052790, 0.110854] | -0.006311 [-0.013658, +0.001037] | 0.121987 [0.087174, 0.156800] | -0.015576 [-0.021861, -0.010578] |
| gru-k4 | 0.078884 [0.053209, 0.116479] | -0.003478 [-0.012013, +0.004759] | 0.126261 [0.091968, 0.160554] | -0.011302 [-0.015960, -0.002949] |

On these held-out closed games, every memory row has a negative paired
interval relative to the coarse state baseline. Every open-list paired-difference interval crosses
zero. This readout is conditional on scripted play, a small corpus, one fitting
recipe and capacity admission. The baseline omits current known-hand/library,
stack and sideboard facts; the result cannot establish information beyond the
full native observation. It supplies the planned K choice, with no RL,
playing-strength or policy-adoption result. Own-seat inclusion remains off.

Three interrupted attempts preceded this result: the first predated integration
of the upstream knowledge/disclosure world; the second shared one stateful
heuristic across seats; the third omitted four sideboard-only target definitions.
All stopped before K selection. Their logs and source receipts are retained;
the corrected corpus and fits use none of their rows or weights. A directed
replay at seed 48103, decision 101 catches the selector bug, and target coverage now
rejects any physical hand definition absent from the roster vocabulary.

Evidence is manifest-backed under
`experiments/data/opponent-memory/20261116-supervised-probe`: the complete
archive, per-member integrity record, summary, selection, runtime, stdout and
interrupted-attempt receipts (eight manifest entries). An isolated empty-cache
fetch downloaded the 122-byte selection record from the bucket and matched its
original bytes. The full archive was verified locally; it was not fetched again.
All 1,855 archive members were verified locally
against their original bytes (108,150,319 bytes before compression).

Validation: 10 native debug tests passed; focused Python check runs passed
43, 65, 151 and 5 cases, followed by the corrected recorder/target regressions.
Final closeout: `uv run --extra dev pytest tests/verify/test_decision_memory_probe.py tests/infra/test_experiment_data_untracked.py -q` — 7 passed in 75.51 s.
The ordinary world-admission/monitoring checks passed. These are software and
probe checks; gate still owns full CI. Gated J1/trunk copy, current-world CUDA
cost/resource/update/reload admission and the three-arm pilot remain outstanding.
B1/B2 stay deferred until calibration shows what fits the same USD 50 total.

## Gated join and Elemental capacity census — 2026-11-16

The three pilot arms are `control`, transformer-K1 and GRU-K1. All memory
policies use the zero-initialized per-layer J1 gate.
All three branches start through `TrainSelfPlay.pretrained` from a compatible
control checkpoint. Admission checks the producer and non-memory configuration;
the executor copies shared tensors strictly, initializing only memory and gates.
Adam, EMA, streams and schedule start fresh. The run record retains
`pretrained/source_run`, `pretrained/checkpoint` and `producer_cost`; checkpoint
source continues to name the run that wrote it. Update metrics report
`decision_memory/gate_<layer>`. The remote `TrainingJob` adapter binds the
producer record and selected checkpoint through digest-checked inputs (see the
pretrained job validation below); CUDA branch admission is still outstanding.

One prerequisite changed after inspection: memory adds catalog embedding rows,
so the old log-enabled control could not supply identical shared tensors. The
common trunk now reserves the memory vocabulary whenever its observation space
enables decision records. The history-disabled throughput incumbent still omits
that vocabulary. This supersedes the earlier same-parameter-incumbent claim for
the log-enabled control; previous cost receipts keep their measured identities.
Live attention also keeps the trunk's board row count through projection and
feedforward operations: processing extra summary rows in the same matrix
multiplication otherwise produced a 4.66e-10 logit difference on CPU.

A free heuristic census covered every ordered Elemental 10v10 pair once in each
disclosure mode: **200 completed games, zero capacity failures or decision caps**.
It retained both viewers' complete terminal tables and exact action journals,
with one heuristic instance per native seat and own-seat inclusion false.
Raised census bounds were 4,096 records, 131,072 reference-detail rows, 4,096
references and 131,072 members. The measured maxima and next powers of two are:

| Table | Maximum | Capacity for CUDA admission/calibration |
|---|---:|---:|
| decision_records | 338 | 512 |
| decision_reference_details | 14,787 | 16,384 |
| decision_references | 98 | 128 |
| decision_members | 5,127 | 8,192 |

The detail maximum came from open Elemental UG versus UG. Every matchup and
viewer count is retained under the manifest-backed prefix
`experiments/data/opponent-memory/20261116-elemental-capacity` (four artifacts:
source, action/count journals, summary and provenance). Runtime was 87.43 seconds;
launcher cost **USD0**. This is capacity sizing on heuristic terminal prefixes,
not learned-game-tail or CUDA/resource admission. The published supervised probe
keeps its original 8,192 detail bound and exclusions; it was not rerun.

Reproduce the census with:

```sh
uv run python -m experiments.opponent_memory census \
  --out .runs/decision-memory-capacity
```

The release extension was rebuilt and installed atomically for CPython 3.13:
`466492569adec10fe609c24240edddd97123bed1877d6d07a5276d31fe855d15`.
CUDA admission, horizon/gate-scale calibration, Etude job wiring and paid pilot
remain outstanding. Paid-run overflow handling remains an open pre-launch
choice; no rental or paid run was started. Opponent-only is still the default,
own-seat inclusion remains optional/default false, and the inclusive cap is USD50.

Validation: focused gate/model, backend protocol, update, Prepared, storage,
replay/demo, package-boundary and artifact checks: **288 passed, 7 skipped**
(230.31 s). The gate matrix covers both encoders, K1/K4, padded/live and compact
layouts, mixed empty/128-record batches, with gradients enabled and inference
mode, at zero output tolerance. Real-prefix damped updates prove zero encoder
displacement after the gate-opening update and encoder learning on the next
update. Native debug decision-memory checks: **10 passed**. Ruff and
`git diff --check` passed. No CUDA, MLX-memory or playing-strength claim follows.
The model CI invocation collected **1,215 tests**, including the new gate file.
The first collection lacked the local LLM extra; installing the declared locked
extra resolved those import errors. This was collection only, not a full CI run.


### Pretrained executor validation — 2026-11-16

Control, transformer-K1 and GRU-K1 now run through published `pretrained`
admission and ordinary training execution. Bounded CPU fixtures use two streams
and exact-quota collection; each memory branch's exported initialization matches
its trained control's shared tensors and real-prefix outputs with zero tolerance.
Different encoder producers and changed non-memory settings are rejected; missing
shared keys, unexpected keys and incompatible shapes fail restoration.

Focused validation: **265 distinct Python cases passed, 7 skipped** across the
gate, pretrained executor, decision model, backend protocols/updates, container,
Prepared, package boundaries and experiment discovery. The broad selection first
reported 211 passes and one failure: gate diagnostics accessed an absent attention
layer in a memory-disabled policy. Returning no gates for disabled memory fixed
it; the failing integration and staged-gradient/scale selection then passed all
9 cases. Ruff check/format and whitespace checks passed. These checks establish
software wiring, not CUDA fit, throughput or learnability. No Rust changed, no
probe/census was repeated and no rental or paid run was started.


### Pretrained job binding — 2026-11-16

`TrainingJob.pretrained` now carries a `PretrainedInitialization`: the
producer's immutable TrainingRun record and selected raw/EMA checkpoint as
`StagedExport` inputs, plus the source stage and weight role. Job construction
validates the bound regime at `/inputs`; the worker binds delivered paths and
uses ordinary digest, producer-metadata, world, observation and strict model
admission. Producer cost uses the existing admission calculation. Pretrained,
initial weights and learner continuation are exclusive, and the frozen regime
cannot supply another starting state.

The experiment's pure `branch_jobs` builder freezes all three arms from one
published EMA export per seed label, taking the shared branch recipe, horizon,
gate scale and caps from its caller. It makes ordinary Etude RunRequests with
identical producer inputs within a seed. This supplies the software connection
for calibration; `training_jobs` and `eval_jobs` still stop until CUDA admission
and calibration fix their declarations. No calibrated recipe or paid launch is
claimed here. EMA remains the recorded agent assumption for the branch role.

Real local worker proofs start the control and GRU from an executor-published
trunk record and EMA. Both exported initializations equal the trunk's shared
weights and real nonempty-history action distributions/values at zero tolerance.
The branches start at iteration 1 with fresh Adam, preserve `producer_cost` and
retain the selected checkpoint digest. Corrupt record/checkpoint bytes, a wrong
weight role, a changed non-memory model and competing starting states are refused
before updates. The transformer job is frozen from the same inputs; its executor
and exact initialization proofs remain those recorded above.

Validation: `uv run --extra dev --extra artifacts pytest tests/training/test_pretrained_jobs.py -q`
— **7 passed** (96.30 s). Existing worker, experiment discovery and workflow-path
checks — **25 passed** (94.80 s). After simplifying job input selection,
**6 focused rechecks passed** (17.51 s), covering pretrained/control branches,
starting-state exclusivity, initial weights, continuation and job freezing.
Ruff and whitespace checks passed. The new worker proof is included in the experiment CI group. This is USD0 software
validation: no CUDA/resource, throughput, calibration, learnability or strength
result; no Rust change, probe/census rerun, rental or background service.


### Published-trunk reader and fixture identity — 2026-11-16

`PretrainedInitialization.from_storage` connects published producer records to
job inputs. It requires a completed run and named stage, selects raw or EMA,
checks the stored checkpoint's SHA256 and size, and includes the S3 prefix in
bucket-relative input keys. It returns the producer's seed; the branch builder
refuses mismatched seed labels. The worker still owns ordinary policy, world
and cost admission. Existing frozen experiment helpers are unchanged.

The local worker proof now declares `TrainingJob(experiment="fixture", ...)`
directly at fixture seed 197 for control and GRU. Only a pure declaration check
uses the pilot builder; no training runs under the pilot identity. Initial
weights and live-history outputs still equal the producer at zero tolerance.

Validation: 18 final focused cases passed across the job-file and targeted
reader/builder invocations (the first declaration check exposed missing fixture
capacity/execution bounds, then passed after those were explicit). Coverage
includes completion, absent stage/role, digest and size rejection, raw/EMA and
S3 prefixes, worker initialization and pre-update refusals. Ruff, format and
whitespace passed. USD0 software evidence only; CUDA admission and calibration
remain next, with paid-pilot overflow handling still a separate decision.

### Bounded CUDA calibration procedure — 2026-11-16

The `--calibration` declaration runs one exploratory job, separate from pilot
training. It first trains a two-update control producer (seed 39100), then runs
incumbent/control/transformer-K1/GRU-K1 in forward and reverse order, each in a
fresh process. All branches use that producer's EMA via ordinary staged-input
worker admission. The true history-disabled incumbent starts fresh and omits
the memory vocabulary. These are short operational trajectories, not matched
learning or strength evidence.

The workload is mixed-disclosure Elemental independent ten-deck self-play,
256 streams, four producers, complete-game closure, two updates, float32,
512/16,384/128/8,192 memory capacities and 2/2/2 execution rows. The provisional
100-update damping schedule is identical across arms; calibration must still
choose the pilot horizon and gate scale. The explicit `UniformDecks` source
owns disclosure and sampling; it replaces the older curriculum flag, which
cannot be combined with a source in the current recipe API.

Each worker retains its ordinary run, tracking, checkpoints, native/host/learner
phase timings, realized maxima for all four memory tables, process-tree RSS and
CUDA allocator peaks. Raw and EMA checkpoints must pass normal loading and
finite-parameter checks on CUDA. Capacity errors fail calibration, with no
truncation or recovery. Paid-pilot overflow handling remains undecided.

The job has one attempt, a USD1.20 ceiling and 0.9-hour lifetime; the measurement
itself stops within 2,400 seconds, each worker within 690 seconds including
finalization. Quote the whole rental including startup before launching with
`--max-usd 1.20`; do not infer actual cost from these ceilings. Incomplete
repeats or a worker failure leave training/evaluation declarations stopped.
The CPU smoke uses eight exact-quota rows on two streams and is plumbing only.

Preparation checks at source `76e9c49cd0`: the final CPU smoke completed all
five workers in 69.96 seconds (producer, incumbent and three EMA branches),
including normal raw/EMA reload. An earlier smoke exposed a zombie-child race
in process-tree RSS sampling; the sampler now ignores exited children and
propagates other sampling failures. The clean rerun had no sampling errors.
Focused probe-rule and experiment-discovery checks passed two tests; Ruff,
format and whitespace checks passed. Frozen probe artifacts were not rerun.

Etude's prospective quote on 2026-11-16 was one A40, at least six vCPUs and
48 GB host memory, USD0.59/hour and **USD0.68 planned rental cost**, including
startup, within the USD1.20 job/launch cap. This is a quote, not incurred cost.
The launcher had no existing `opponent_memory` runs. Scoped discovery used
`r2://manabot/jobs/opponent_memory` and `--job cuda-calibration`, so calibration
does not invoke the still-stopped evaluation declaration. The unscoped bucket
scan was interrupted before submission; no rental or run resulted from it.

### First CUDA admission stopped in collection — 2026-11-16

Source `76e9c49cd0`, run `opponent_memory/cuda-calibration@39100`, launch
`8ea6cceb615c`, attempt `53bf706d0206`: **failed admission**. The common control
producer hit `TimeoutError: training resource wall deadline exceeded` in the
ordinary collector. It completed zero updates; no subsequent incumbent or
memory arm ran, and no final raw/EMA reload was available. The outer launcher
records failure and the inner training record records interruption. This is
not a throughput, CUDA-memory-arm fit, learning or strength result.

Host: Xeon Gold 6342, 7.65-CPU cgroup quota, 49,999,998,976-byte host limit.
A40: 47,699,722,240 device bytes, capability 8.6; Torch 2.15.0+cu132, CUDA 13.2,
world w17, CP313 native SHA256
`c2c7f44161500b00abcf0e5537b6e968a2eec9f1dace0569574c18088595af26`.
Worker receipt wall time was 600.946 s. Sampled process-tree peak RSS was
8,908,734,464 bytes; CUDA allocated/reserved peaks were 109,712,896 /
121,634,816 bytes. These describe the incomplete control trajectory only.

| Table | Realized maximum before timeout | Configured capacity |
|---|---:|---:|
| decision_records | 116 | 512 |
| decision_reference_details | 8,700 | 16,384 |
| decision_references | 79 | 128 |
| decision_members | 1,209 | 8,192 |

There was no capacity error. These partial prefixes cannot admit learned-game
tails or justify shrinking capacities. Collection occupied 594.348 s:
preparation 170.987 s and inference/action selection 343.907 s, each across
19,717 execution calls. Native step/encoding was 11.103 s (nested native
stepping 4.147 s, encoding 3.107 s); recording 5.672 s and bookkeeping 0.342 s.
Nested native timings must not be added again. No completed learner-decision
count exists, so no decisions/s is inferred from an incomplete batch.

The rental `mtg252-a40-0ae24e1a885c` was terminated and the launcher confirms
`rental_ended=true`. Its retained launcher cost is **USD0.12234, estimated**;
no settled bill is available. All **22** output files were fetched through the
SDK with digest checks, including four private native failure journals, the
interrupted record, initial checkpoint and phase/resource receipt. The R2
timeouts during log reads resolved sufficiently to retrieve the final evidence.

**Prospective bounded retry:** use 32/32/32 execution rows, retaining 256 streams,
four producers, the same capacities, float32, two-update target and provisional
schedule. The control's small CUDA peak and large number of small inference
calls motivate this resource choice; they do not prove its speedup or admit
memory-arm peaks. The job name records `rows32`, with one attempt and the same
USD1.20 cap. Both job ceilings total USD2.40, below the USD3 calibration allowance.
No table is truncated, game recovered or prior failed attempt relabelled.
The script now reuses training's existing process-tree RSS reader; its four
focused exit/zombie/permission/real-process tests passed. Pilot declarations
remain stopped pending complete CUDA admission and calibration.

### CUDA short-chunk backward repair — 2026-11-16

Source `99b559e123`, run `opponent_memory/cuda-calibration-rows32@39100`, attempt
`cf12074a0eef`: **failed admission**, with `RuntimeError: Invalid max_seqlen_k:234`
in the first control update's backward pass. Collection completed in 212.159 s;
learning ran for 26.052 s before failure. No update completed, no subsequent arm
ran and no final-policy reload was available. These are partial phase timings,
not an admitted training throughput. Both A40 rentals reported Xeon Gold 6342,
but this rental exposed no CPU quota and used cgroup-v1; it is not a same-host
causal comparison with the first attempt.

Self-play prefix maxima were **432/13,623/113/5,196**, beside capacities
**512/16,384/128/8,192**. No capacity overflow occurred in this completed
collection. Sampled process-tree peak RSS was 15,856,558,080 bytes; CUDA
allocated/reserved peaks were 186,126,336 / 257,949,696 bytes. These remain
control measurements, not memory-arm admission. Preparation/inference occupied
79.822/31.384 s over 2,281 calls; native step/encoding 21.301 s over 876 calls.
Worker wall time was 243.051 s. All 18 outputs were fetched and hash-verified.
The launcher confirms the rental `mtg252-a40-3b95659b1c93` ended; its retained
cost is **USD0.05991, estimated**, without a settled bill.

The error exposes an existing live-token attention bound: padded slots can
exceed the entire packed sequence in a sparse final execution chunk. PyTorch's
[memory-efficient backward check](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/attention_backward.cu)
requires each maximum sequence bound to fit that packed total. `LiveTokens`
now supplies `min(slots, packed_length)`, still a valid upper bound on every
row, with no device-scalar read or attention/model change. The directed
regression compares sparse one-/two-row chunks against padded attention through
backward on CPU and CUDA when available.

One repaired admission is bounded by **USD0.60 / 0.75 hours**, keeping the three
job ceilings at exactly **USD3.00**. It retains 32-row execution and all other
measurement settings; its name `cuda-calibration-bound32` identifies the
repaired source. No pilot run is authorized by this failed admission. The
repaired rental runs the directed CUDA backward regression before its workers.
Local checking also found two fixture references to the deleted dense-row
reader; both now use the surviving retained-row path, without restoring an API.
`uv run --no-sync pytest tests/model/test_live_tokens.py -q`: **6 passed,
2 CUDA-only cases skipped** (19.57 s); Ruff and whitespace passed. The initial
file run exposed four stale-fixture failures before that consumer cut; the two
new CPU gradient cases already passed. CUDA proof remains the rental's next
admission step.

### Repaired backward passed; worker paused — 2026-11-16

Source `ffc3f773bf`, run `opponent_memory/cuda-calibration-bound32@39100`, attempt
`fc861e77929a`: the directed CPU/CUDA sparse-chunk backward regression passed
**all four cases** in 7.94 s on the A40. The control then completed **one update,
54,467 learner decisions and 213 optimizer steps** in 332.227 s, with
244.733 s collection and 85.424 s learning. The ordinary worker paused because
its 600-second allowance could not fund another update plus its stop reserve;
the calibration correctly rejected the incomplete two-update target. Its outer
outcome remains failed, its inner outcome paused. Final raw/EMA/Adam were
exported, but the calibration's reload check was not reached.

Prefix maxima remained 432/13,623/113/5,196. Sampled process-tree RSS peaked at
15,844,237,312 bytes; CUDA allocated/reserved peaks were 186,126,336 /
257,949,696 bytes. The 25 outputs were fetched and hash-verified. Rental
`mtg252-a40-58f3e2b06d73` is ended; launcher cost is **USD0.07871, estimated**.
The completed control update is operational evidence only, without memory-arm
admission, a comparative throughput result or a learning/strength conclusion.

**Calibrate the time limit from this result.** Keep the exact workload,
two-update targets, 32-row execution, capacities and 100-update provisional
schedule. Allow 1,800 seconds per worker, with 90 seconds for worker finalization;
the complete forward/reverse sequence has a 9,000-second guard inside a
three-hour job. This funds two measured 332-second updates with room for the
unadmitted memory arms, instead of repeatedly pausing the control. Progress
logs now name the preflight and each worker start/completion.

The ended runs release their unused reservations: Etude retains approximately
USD0.26 in estimated commitments, not their original summed job ceilings.
Quote the complete host window, including startup, for a job capped at USD2.00
and launch under the unchanged **USD3.00 calibration cap**. No settled bill is
claimed and no paid pilot is launched by this resource-bound correction.

The initial three attempts are manifest-backed at
`experiments/data/opponent-memory/20261116-cuda-admission` (17 entries).
All 65 archived output files, totaling 50,808,576 original bytes, were fetched
with SDK digest checks and reverified member-by-member, including exact archive
membership. Cleanup snapshots confirm all three rentals ended. The artifact
tracking guard passed. The full run at `469339f508` remains separate evidence.

### Planned paid-run overflow handling — 2026-11-16

Headless implementation choice: every pilot trunk and branch will use
`game_errors=None`, as calibration does. A memory-table capacity error ends the
run with retained diagnostics; the comparison will not discard an overflowing
game and continue learning. This supersedes the earlier open overflow choice.
No pilot has launched. An overflow requires a prospective capacity revision
across the comparison, with the failed run retained as incomplete evidence.

### Reference-repeat completion plan — 2026-11-16

The full calibration completed its producer, first incumbent, and both control,
transformer and GRU branch repeats. Its conservative 1,890-second admission
reserve then refused the last incumbent despite a completed preceding worker.
The attempt is incomplete (eight of nine), not a nine-worker success. No live
source or bound is changed. Measurements and cost are recorded after delivery.

Complete reference variability prospectively with a separately labelled
`cuda-calibration-reference` job: fresh incumbent/control/incumbent processes,
the same seed, workload, precision, capacities, execution rows and two-update
targets, with ordinary raw/EMA reload. This runs after the first rental ends,
with a 0.65-hour window and USD0.60 cap under the unchanged cumulative USD3
calibration limit. It does not rerun memory arms. The control is fresh, like the
original producer; it is not represented as another pretrained branch. Keep
host identities and rates separate: a new rental cannot supply the missing
same-host incumbent repeat or make the original sequence complete.

The first reference quote refused the USD0.50 cap because it did not cover
the whole window plus startup. Increase only that ceiling to USD0.60 before
launch; the quote must still fit the unchanged cumulative USD3 cap. No machine
was acquired by the refused quote.

### Two-update CUDA measurements and capacity revision — 2026-11-16

Source `469339f50885`, run `opponent_memory/cuda-calibration-full@39100`,
attempt `eb98cd76fb06`, A40 `mtg252-a40-d351c44bf214`: eight workers completed
two updates and admitted finite raw/EMA reloads; the final incumbent repeat
was not started because of the conservative reserve. The run therefore failed
its nine-worker completion target. The retained launcher estimate is
**USD1.21337**, not settled billing; deletion is confirmed. Its delivered
manifest names 230 outputs, now being fetched with SDK digest checks.

Xeon Gold 6342, 7.65 CPU quota, 49,999,998,976-byte cgroup-v2 host limit;
A40 has 47,699,722,240 bytes. Same-source mixed-disclosure workload, 32-row
execution, float32. Rates below use committed learner rows / stage seconds
(including collection, learning and export); reload adds about one second.
These are differing actual self-play trajectories, not isolated encoder costs.

| Worker | Rows | Stage s | Collection s | Learning s | Decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Producer control | 105,216 | 764.22 | 581.10 | 180.14 | 137.68 |
| Incumbent | 106,298 | 335.51 | 200.88 | 132.39 | 316.82 |
| Control 1 | 105,189 | 745.29 | 561.15 | 181.46 | 141.14 |
| Transformer 1 | 105,599 | 1,154.09 | 861.48 | 289.75 | 91.50 |
| GRU 1 | 105,770 | 1,162.84 | 853.46 | 306.55 | 90.96 |
| GRU 2 | 105,331 | 1,124.34 | 836.42 | 284.96 | 93.68 |
| Transformer 2 | 106,167 | 1,159.87 | 868.95 | 287.94 | 91.53 |
| Control 2 | 104,194 | 751.79 | 560.34 | 188.29 | 138.60 |

Branch medians are 139.87/91.52/92.32 decisions/s for control/transformer/GRU:
44.1%/28.9%/29.1% of the one measured incumbent process. The roughly 10% cost
guide is clearly unmet on this workload. There is only one same-host incumbent
process, so no repeated-baseline interval follows; the report makes no policy
adoption or strength claim. Preparation alone took 218–221 s in the control
branches and 439–457 s in memory branches; native step/encoding took 34–41 s.
Collection and whole-training costs stay distinct.

| Arm (process peak across repeats) | Host tree GB | CUDA allocated GB | CUDA reserved GB | Records/details/references/members maximum |
| --- | ---: | ---: | ---: | --- |
| Incumbent | 15.60 | 0.180 | 0.256 | 0/0/0/0 |
| Log control | 15.93 | 0.187 | 0.258 | 447/13,623/123/6,493 |
| Transformer | 15.95 | 1.112 | 3.483 | 447/13,623/123/6,493 |
| GRU | 15.97 | 1.106 | 3.473 | 447/13,623/**128**/6,493 |

GB are decimal bytes, process peaks include ordinary reload, and GPU peaks
are allocator peaks rather than whole-device usage. No capacity error occurred,
but one GRU repeat filled **128 of 128 reference rows**. Freezing that bound
for longer learned games would leave no observed headroom. Revise only reference
capacity to **256**, keeping records/details/members at 512/16,384/8,192 and
strict run failure on overflow. Existing evidence and checkpoints stay bound to
their original layout; do not load/rebind them into the expanded layout.

This finding supersedes the quoted reference-only completion plan: that job
was never launched. Delete its unused selector and instead re-admit the expanded
layout with fresh producer/incumbent/control/transformer/GRU/incumbent workers
(two updates each), all memory branches from the new producer EMA. A 1.75-hour
job with USD1.20 ceiling includes startup and must be admitted by Etude under
the unchanged USD3 cumulative calibration cap before launch. This measures every
pilot arm on the selected capacity and two incumbent processes on one new host.
The original host remains a separate stratum; its missing repeat stays missing.

Operational note: R2 confirmed the attempt delivered before local launcher
state updated. The global wait was stopped; the launcher's existing per-supply
collection and termination helpers were invoked under its normal lock for
`mtg252-a40` only. They retained the failed outcome, 230-output manifest and
confirmed rental deletion. No unrelated supply or SDK source was changed.

The expanded-layout quote was admitted at USD1.18 planned, USD1.20 job cap,
with USD1.47 already committed under the cumulative USD3 limit. Source
`857fa4517996` freezes the six-worker workload before launch. The original
230 SDK-verified outputs (342,991,544 bytes) were packaged and every archived
member hash rechecked. All eight live receipts match their sealed SDK bytes.
Fifteen added manifest entries publish that bundle, individual receipts, source
and attempt records and confirmed deletion. The artifact guard passed; no
fresh archive-fetch round trip from the artifact store is claimed.

### CPU host baseline after the throughput steer — 2026-11-17

No further rental is launched. The already-running refs256 job stays on source
`857fa4517996`; live code now stops paid declarations and selects control/GRU-K1.
Observation retention and its nested Memory capture have profiler tracks. A
log-enabled timer-on/off fixture and the ordinary fixture retain identical
collected actions, log probabilities, rewards, dones and values; the branch
builder now declares only control and GRU (3 focused checks passed).

Replayed all 200 frozen heuristic census games, 56,682 original action indices,
with native trivial skipping and the original disclosure/seed/decks. Every game
finished and both terminal table-count records matched. Retained a fixed 256-row
fixture of real nonterminal viewer prefixes, selected evenly across the largest-
detail prefix per viewer. Selected maxima: 338/14,787/94/5,127; declared bounds
512/16,384/256/8,192. The replay took 109.03 s: native encode 29.47 s, MemoryStore
capture 67.40 s, stepping 6.76 s. This isolates the capture cost on heuristic
trajectories; it is not the CUDA worker's missing-time measurement.

The CPU matrix uses fresh processes for incumbent/control/GRU at each common
execution bound 32, 128 and 256, one Torch thread, 256 logical rows, three
collection + actual damped-update samples per process. GRU gates are fixed
open initially (0.5 parameters), so encoder gradients are exercised. Controlled
rewards are numerical-update fixtures, not outcomes or learning evidence. Native
inputs, harness/native/source hashes, output/gradient/final-state hashes and raw
phase times are frozen before the host cut. First-sample warmup and a contended
host make these ranges descriptive; no independent-process confidence interval.

| Rows | Incumbent collection s | Control collection s | GRU collection s | Incumbent update s | Control update s | GRU update s |
| ---: | --- | --- | --- | --- | --- | --- |
| 32 | 0.427–1.946 | 0.713–1.182 | 2.075–3.069 | 4.213–9.357 | 5.604–6.942 | 10.478–13.857 |
| 128 | 0.344–0.480 | 1.011–1.119 | 1.817–2.740 | 2.161–3.605 | 3.195–4.859 | 8.577–9.081 |
| 256 | 0.304–0.414 | 0.778–0.882 | 1.046–2.307 | 1.930–2.945 | 2.383–3.635 | 4.105–8.332 |

At matched row bounds, median control/incumbent decision-rate ratios are
0.725/0.435/0.467; GRU/incumbent ratios are 0.238/0.180/0.191 for
32/128/256 respectively. These exclude corpus capture and are **policy fixture**
rates, not end-to-end worker decisions/s. The true incumbent omits all four
memory inputs and their vocabulary. Existing CPU/code/native evidence remains
separate from the A40 measurements.

First host change removes per-reference Python program expansion in
`prepare_memory`: immutable native program rows are cached once, and offset/
count routes combine catalog and explicit details in original reference order.
Unused definitions receive no policy-vocabulary lookup. Prepared arrays match
the frozen implementation exactly at 1/32/128/256 rows; four focused tests cover
shortcut/full equivalence, unknown programs, group routes and GRU gradients.
A separate matrix measures this change before chunk admission/preparation work.

The first before/after matrices had a host confound: incumbent/control also
became much faster despite never calling the changed featurizer. Do not
attribute those across-window speedups to vectorization. A follow-up alternates
only `manabot.policy.prepare.prepare_memory` between its frozen original function
and the vectorized function in fresh processes, forward then reverse order.
Admission, capture, model, update and input bytes stay current and identical;
source/selector identities are retained. Two processes per variant/arm/bound,
three samples each; the ranges below are six samples, not confidence intervals.

| Rows | Old GRU preparation s | Vectorized preparation s | Old collection median s | Vectorized median s |
| ---: | --- | --- | ---: | ---: |
| 128 | 0.384–0.476 | 0.280–0.375 | 1.008 | 0.809 |
| 256 | 0.463–1.163 | 0.318–0.368 | 1.323 | 0.962 |

GRU preparation ranges separate. Whole-collection ranges overlap at 128 rows;
medians correspond to 1.246×/1.376× fixture throughput, not a whole-worker
claim. Incumbent median old/new factors are 1.020/0.992 and control factors
1.007/0.938 at 128/256 rows. All 36 additional output/gradient/final-state hash
comparisons are identical (63 total with the first matrix); prepared fields
also match exactly at all four tested row sizes. Keep vectorization, while
reporting the remaining large cost against the true incumbent. Chunk admission
and feature caching remain the next cut; no rental is authorized by this result.

### Expanded-reference CUDA closeout — 2026-11-17

Source `857fa4517996`, run `opponent_memory/cuda-calibration-refs256@39100`,
attempt `cff5863d5ebc`, machine `mtg252-a40-010318be3ea1`: **all six workers
completed two updates and finite raw/EMA reload on CUDA**. This was already
running when the throughput-first steer arrived. Its frozen transformer worker
finished; live declarations now select GRU and stop further paid work.

The A40 host again reports Xeon Gold 6342 and a 49,999,998,976-byte host limit,
this time through cgroup-v1 with CPU quota unreported (not claimed unlimited).
GPU memory is 47,699,722,240 bytes. World w17, input v11, float32, 256 streams,
four producers and 32-row execution are unchanged. References are 256; the
new producer and branches use that layout, with no old-checkpoint rebinding.

| Worker | Actual learner rows | Stage s | Collection s | Learning s | Decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-control | 105,269 | 595.85 | 429.82 | 163.96 | 176.67 |
| 01-incumbent | 107,267 | 304.40 | 168.30 | 134.04 | 352.39 |
| 02-control | 104,906 | 558.11 | 398.36 | 157.45 | 187.96 |
| 03-transformer-k1 | 104,396 | 832.99 | 608.99 | 221.97 | 125.33 |
| 04-gru-k1 | 104,059 | 835.34 | 620.16 | 212.79 | 124.57 |
| 05-incumbent | 105,558 | 303.40 | 156.22 | 145.40 | 347.92 |

The two incumbent processes give **347.92–352.39 decisions/s**
(median 350.15). The pretrained control is 0.537× that median;
GRU is 0.356×. The frozen transformer is 0.358×. These are
whole-training stage rates, not isolated encoder comparisons or confidence
intervals. Faster absolute times than the first host are **not** attributed
to the capacity change: quota reporting and actual trajectories differ.
This source also predates the local profiler/vectorization changes, so these
CUDA numbers are not a measurement of that optimization.

Realized maxima across memory workers are **447/13,623/128/6,493** against
**512/16,384/256/8,192**, so references now have observed headroom. No overflow
occurred. Host process-tree peak across workers is 15.90 GB; CUDA allocator
peaks are 1.112 GB allocated and 3.483 GB reserved. This admits these short
trajectories, not arbitrary learned-game tails. Strict overflow failure remains.

The launcher recorded this run's estimate at **USD0.58608**; deletion is
confirmed. All **164 outputs / 231,354,814 original bytes** were downloaded
through SDK digest checks, then every archive member hash was verified. Raw
receipts, the complete calibration record, source/attempt records, summary and
cleanup are published beside the earlier CUDA evidence. No fresh artifact-store
archive-fetch round trip is claimed.

At close, the launcher retains **USD1.94475 across all five calibration runs**,
including the failures. The first three statuses are no longer estimated; the
last two remain estimates. Earlier dated estimates and their bytes stay frozen;
`refs256-all-calibration-cleanup.json` records the closeout observation. No cost
is reconstructed from wall time or the price quote. All five rentals confirm
ended, inside the original USD3 calibration allowance.

Stop here under the new direction: no additional rental, pilot training or
evaluation launch. No pilot horizon/gate scale or matched learning/strength
result is frozen. The implemented first CPU step exposes retention/capture and
keeps exact vectorized preparation; content-admitted chunk feature caching is
still unfinished and remains the next implementation cut.

### Record-aligned chunk cache candidate — 2026-11-17

The working implementation replaces four independent table chains with a
32-record chunk owning its members, newly introduced snapshots and their details.
An immutable node's exact predecessor binds its five-integer admission cursor.
Dense/import validation folds the same checks over chunks after shapes, integers
and padding are checked. Retained collection, frozen inference, bootstrap and
learning assemble vocabulary-bound host feature fragments without dense history.
Recency remains prefix-relative; coalescing gets a new tail and certificate.
No learned embedding or GRU state is cached. Explicit dense exports remain
mutable inputs and selected exports preserve their supplied bytes.

**Correctness is supported; the original footprint acceptance is not met.**
The final 4,298-case differential proof has zero mismatches and outcome digest
`d2fed54e780521122ab1a06dcb63fad4f4eafed0306121f81be3cefe2fd13a92`.
All 2,189 prepared-array comparisons over the frozen 256-prefix fixture and
all 81 sampled-output/gradient/post-update-state hash comparisons match exactly.
Directed tests cover cross-chunk backward joins, changed predecessor spans,
changed detail companions, coalesced versions, vocabulary rejection, mutable
dense exports, archives, replay recovery and weak cache lifetimes.

The final CPU matrix uses the unchanged frozen census-derived fixture and
baseline `7d680aac170f12ca60e60dfb812e0c00d7cda7fa`. Each arm/row-limit/source runs
in a fresh process with three successive 256-row sampling + learner updates,
Torch intra/inter-op threads both one. The first sample includes cold feature
preparation; later samples reuse cached features and have updated weights.
These are medians of three successive samples, **not three independent runs**.
Source order is before/after at 32/256 and after/before at 128. The initial
matrix is retained separately; it predates the explicit-dense-selection fix and
is not pooled with this final source. No artificial contention was induced.

| Execution rows | Arm | Before sampling + learning s | After s | Speedup | After / matched incumbent rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | Incumbent | 1.986 | 1.902 | 1.044× | 1.000× |
| 32 | Control | 2.377 | 1.957 | 1.214× | 0.971× |
| 32 | GRU-K1 | 5.194 | 4.544 | 1.143× | 0.419× |
| 128 | Incumbent | 1.308 | 1.718 | 0.761× | 1.000× |
| 128 | Control | 1.829 | 1.295 | 1.413× | 1.326× |
| 128 | GRU-K1 | 4.529 | 3.764 | 1.203× | 0.456× |
| 256 | Incumbent | 1.188 | 1.169 | 1.016× | 1.000× |
| 256 | Control | 1.779 | 1.251 | 1.422× | 0.934× |
| 256 | GRU-K1 | 4.650 | 3.859 | 1.205× | 0.303× |

GRU collection preparation medians are 0.324/0.304/0.307 seconds before and
0.054/0.039/0.037 after at 32/128/256 rows. Its total sample ranges at 128
rows overlap: [4.412, 4.849] before, [3.703, 4.552] after; at 32 and 256
they separate in this finite fixture. Incumbent variation, especially at 128,
prevents attributing every whole-fixture timing difference to caching. The
control's 1.326× incumbent rate there is not evidence that logging accelerates
training. GRU still costs substantially more than the true incumbent.
No CUDA, whole-game collection or on-policy worker speedup is established.

Cold import/capture of these 256 mostly independent prefixes takes
0.195–0.279 seconds before and 0.273–0.311 after across memory-enabled cases;
this phase is outside the table. The cold capture cost did not improve. Full
transport hashing, shape/padding reads and new-node admission remain; native
suffix transport is not implemented. These cold imports do not measure reuse
across all decisions of a game.

| Accounted ownership, 256 prefixes | Before | Candidate cold | Candidate warm |
| --- | ---: | ---: | ---: |
| Unique raw chunk bytes | 18,596,416 | 28,701,632 | 28,701,632 |
| Python metadata estimate | 14,209,319 | 1,788,266 | 1,788,266 |
| Weak index estimate | 5,813,544 | 242,672 | 242,672 |
| Cached feature/schedule array bytes | 0 | 0 | 61,558,036 |
| Accounted sum | 38,619,279 | 30,732,570 | 92,290,606 |
| Explicit selected dense gather bound | 128,311,296 | 128,311,296 | 128,311,296 |

The sum is component accounting, not process RSS, and the metadata estimate
does not include every warmed fragment's Python object header. Record alignment
loses some independent table sharing but reduces chain metadata: 31,407 nodes
become 1,083. Warm cached features add 61.6 MB, so the original unconditional
retained-byte no-growth criterion fails. This candidate is returned for design
review; that criterion is not silently reinterpreted as cold-only acceptance.
A proposed revision would budget raw transport and cached features separately;
it is not treated as an accepted design change. All weak nodes/chunks expire
once the fixture's prefixes and prepared arrays are released (observed 0/0).
No global game or full-prefix feature cache remains.

Final focused regression command: memory chunk/prefix/execution tests, shared
preparation/admission tests, active-training tests and the three affected
package-boundary selections: **84 passed**. Earlier full memory model checks
passed during the cut; Ruff and whitespace pass. No Rust was changed or rebuilt.
The candidate remains uncommitted for the Graph's review; software publication,
landing and scientific Task completion are separate. No additional paid work
was launched. Matched training and evaluation remain absent.

Evidence: `experiments/data/opponent-memory/20261117-chunk-cache/` records the
frozen before/candidate sources, fixture identity, harnesses, both timing
matrices, exact hashes, differential proof and ownership accounting. Its corpus
is the existing `20261117-host-cost` fixture; the evidence bundle does not
regenerate it or change frozen probe/CUDA bytes.

2026-11-17: kept and committed at `574112a0aa`. The retained-byte no-growth
condition was a design-review bar, not a person's; the cache is accepted with
accounted transport ownership (30.7 MB cold, 28.7 MB of it raw chunk bytes) and cached features (61.6 MB warm per 256
prefixes) reported as separate budgets. The measured numbers above are
unchanged. Capture and per-decision whole-prefix encoding remain the next
host-cost targets.


## Native-epoch suffix capture — 2026-11-17

**Keep the capture cut; the GRU is still too slow for paid admission.**
Native log counts now remain current at commit/deserialization, and existing
process-local game generations authorize reuse across synchronous reads.
Collection extends each stream/viewer's previous prefix, reusing every node
before the old last chunk. It always re-reads the last chunk, including when
counts do not change, so coalescing retains its span and multiplicity. Repeated
seeds cannot authorize reuse across resets, forks or replay restoration.
Dense imports/archives retain complete admission. Native padded encoding is
unchanged: this removes host scans/hashing/admission of the closed prefix,
not native suffix encoding or neural full-prefix work.

The pre-cut capture method is frozen from `35e3fe823055` (chunk-cache code
`574112a0aa`). Baseline/candidate use the same rebuilt release engine and all
other current Python code, isolating host capture. This does not isolate the
small native cost of maintaining counts from the preceding engine. Evidence:
`experiments/data/opponent-memory/20261117-suffix-capture` (source hashes,
exact source files/patch, harness, every case, profiles, ranges and proofs).
The extension is `94bb1e63d33a54d0582750cb52bb38f7f2eb0cf4372015a8e10120238cbfe8e7`.

45 fresh sequential processes: three repeats per matched 32/128/256-row limit,
incumbent plus baseline/candidate control and GRU-K1. Each uses the production
collector, two streams, one complete game per stream, both learning seats,
independently sampled Elemental decks, mixed disclosure, one actual damped
policy update and a fixed randomly initialized width64/two-layer policy.
GRU gates are opened to 0.5 to exercise its path. Every case collected 556
learner decisions; observed maxima records/details/references/members were
272/5,688/82/3,592. Collection sampling still uses the original two-row groups;
32/128/256 bound learner execution. This is a CPU software-cost workload, not
competence, learning equivalence, strength or 256-stream host-fit evidence.

Medians in seconds (capture is nested in retention, itself in collection;
never sum those three). Total is collection + update:

| Rows | Arm | Capture before → after (s) | Retention before → after (s) | Collection before → after (s) | Update before → after (s) | Total before → after (s) | After / incumbent throughput |
|---|---|---|---|---|---|---|---|
| 32 | control | 0.136 → 0.054 | 0.153 → 0.073 | 1.568 → 1.642 | 2.681 → 2.640 | 4.261 → 4.160 | 0.963× |
| 32 | gru-k1 | 0.151 → 0.057 | 0.171 → 0.076 | 3.023 → 2.885 | 5.365 → 5.339 | 8.295 → 8.176 | 0.490× |
| 128 | control | 0.151 → 0.052 | 0.171 → 0.071 | 1.768 → 1.579 | 1.968 → 2.001 | 3.736 → 3.579 | 0.966× |
| 128 | gru-k1 | 0.158 → 0.055 | 0.178 → 0.073 | 3.132 → 2.812 | 4.761 → 4.952 | 7.893 → 7.763 | 0.446× |
| 256 | control | 0.148 → 0.048 | 0.167 → 0.065 | 1.711 → 1.500 | 1.953 → 1.973 | 3.664 → 3.474 | 0.980× |
| 256 | gru-k1 | 0.157 → 0.055 | 0.178 → 0.074 | 3.128 → 2.825 | 5.003 → 5.119 | 8.131 → 7.943 | 0.429× |

Capture ranges separate in all six pairs: before 0.131–0.240 s, after
0.046–0.093 s across the entire matrix. Median speedups are 2.5–3.1×;
retention medians improve 2.1–2.6×. Whole collection/update ranges overlap,
so their median changes do not establish an end-to-end speedup. The
history-disabled incumbent's total medians at 32/128/256 rows are
4.005/3.459/3.404 s. Control's median total rate is within 4% on this fixture;
GRU remains at 0.429–0.490×, below the session's conditional CPU gate for a
single CUDA admission. No rental, pilot, training comparison or evaluation
was launched. The roughly 10% adoption guide is not established generally.

Raw unique chunks remain 1,744,008 bytes and the selected dense-gather bound
139,195,712 bytes; GRU feature arrays remain 4,444,452 bytes. Live stream-owner
index accounting is 640 bytes versus the empty baseline's 64 bytes, with
110 more accounted prefix metadata bytes. These bounded active owners are
included by `MemoryStore.measure`, alongside retained rollout nodes.
The host has 16 logical CPUs, one Torch intra/inter-op thread and one native
producer; CPU model, quota and schedulable CPU counts were unreported.
Observed one-minute system load ranged 7.40–20.32, without induced load.

Correctness: 72 exact output/gradient/updated-state/prefix hash comparisons;
4,298 admission differential cases, zero mismatches, unchanged outcome digest
`d2fed54e…`; 10 native debug tests; 15 focused Python checks including native
viewer/count parity, chunk boundaries, same-count coalescing, repeated-seed
epoch changes, sharded forwarding, collection/bootstrap/learning and existing
archive/recovery behavior. Release build, Ruff and whitespace checks passed.

Next is the separate prefix-independent record embedding and parameter-versioned
collection fold, tested against stateless encoding and measured on this same
whole-game harness before learner-side sharing. No model arithmetic changed
in this capture slice; frozen probe and CUDA evidence keep their meaning.

Evidence publication: five content-addressed manifest entries; all 131 source
and measurement archive members verified locally. No fresh remote-fetch check.
The experiment-data tracking guard passed. No server, watcher or measurement
worker remains running.


## Learner sharing and prefix-independent records — 2026-11-17

**Sharing supports trying the learner fold; recency removal alone does not
admit paid work.** This exploratory software slice measures the prerequisite
and removes prefix-relative recency from the shared record embedding. GRU
recurrence still starts from zero for every prefix. Neither learner sharing
nor a learned collection cache is implemented in this slice.

On the unchanged whole-game harness from the suffix-capture report, observe
every `AdmittedRollout.rows` call consumed by `Learner.loss`. Count each row's
chain-node visits and the distinct content-bound nodes in that actual execution
chunk. Sum distinct counts across chunks: a node used in two chunks is charged
twice because their backward graphs cannot share it. Each of three fresh
processes collected the same 556 learner rows, with one actual damped-policy
update, 256 logical minibatch rows, two native streams and both learning seats.
The measurement changes neither row order nor the objective.

| Execution rows | Actual chunks | Row × node visits | Distinct nodes, summed per chunk | Visits / distinct | Record visits / distinct records |
|---|---:|---:|---:|---:|---:|
| 32 | 18 | 1,355 | 199 | 6.809× | 34,969 / 4,520 = 7.737× |
| 128 | 5 | 1,355 | 126 | 10.754× | 34,969 / 2,499 = 13.993× |
| 256 | 3 | 1,355 | 117 | 11.581× | 34,969 / 2,242 = 15.597× |

These are exact counts for this fixture, not measured fold speedups. They
clear the design's near-1 stop condition at the planned non-smoke 32-row
limit. Subdividing those same selected rows into two-row chunks gives only
1.374× node sharing; this is a counterfactual count, not another update run.
Complete-game collection packs seat trajectories consecutively
(`SeatRoutedCollector._build_complete_batch`), so increasing stream count does
not itself interleave every game's rows. Wider complete-game rollouts may
retain sharing within a trajectory, but their history lengths, game boundaries
and chunk alignment need measurement. Rectangular time-major rollouts can
place different games in every execution row and approach 1× sharing. No
256-stream sharing, host-fit or general training-throughput claim follows.

The record input now has 38 numeric coordinates rather than 39. Ordinal and
span remain absolute; prefix-relative recency is removed from the GRU's input.
The transformer adds a bias-free learned 1→64 projection of recency
`(prefix_length - 1 - record_slot) / 128` to record embeddings before its
sequence attention. Cached `MemoryFeatures` contains only the thirteen feature
arrays assembly reads; per-batch routes, packing schedules, lengths and width
belong to `MemoryRows`. Assembly no longer copies numeric features to rewrite
recency. Dense and retained full-prefix encoders remain the stateless reference
for the next fold's behavior/gradient tests.

This deliberately changes unmerged model arithmetic and parameter shapes.
Frozen probe, calibration and CUDA checkpoints describe the earlier featurizer;
their bytes and findings remain unchanged. No learning-equivalence claim is
made. Ordinary current-code checkpoint round trips pass for both encoders;
there is no compatibility loading path for the previous unmerged architecture.

The same whole-game CPU harness was then run in 18 fresh sequential processes:
one before/after case per arm and matched execution limit. Before uses the
three exact production source files frozen at `e733d9fdfb`; after uses this
slice. Both use the same release extension (`94bb1e63…`), fixed seeds, decks,
full admission and native suffix capture. Every case still collected 556
rows and identical retained prefixes; maxima remain 272/5,688/82/3,592.
No compiler or test runner ran concurrently with these measurements.
Single samples, seconds; total includes collection + actual update:

| Rows | Arm | Collection before → after | Update before → after | Total before → after | After / matched incumbent rate |
|---|---|---|---|---|---|
| 32 | incumbent | 1.515 → 1.447 | 2.517 → 2.493 | 4.032 → 3.941 | 1.000× |
| 32 | control | 1.614 → 1.639 | 2.499 → 2.508 | 4.113 → 4.147 | 0.950× |
| 32 | gru-k1 | 2.799 → 2.874 | 5.298 → 5.291 | 8.097 → 8.166 | 0.483× |
| 128 | incumbent | 1.295 → 1.319 | 2.052 → 1.995 | 3.346 → 3.314 | 1.000× |
| 128 | control | 1.492 → 1.580 | 1.944 → 1.930 | 3.436 → 3.510 | 0.944× |
| 128 | gru-k1 | 2.824 → 2.865 | 4.768 → 4.774 | 7.592 → 7.639 | 0.434× |
| 256 | incumbent | 1.344 → 1.430 | 1.927 → 1.887 | 3.271 → 3.316 | 1.000× |
| 256 | control | 1.659 → 1.621 | 1.885 → 1.900 | 3.544 → 3.521 | 0.942× |
| 256 | gru-k1 | 2.854 → 2.817 | 5.013 → 4.916 | 7.867 → 7.734 | 0.429× |

No statistical speedup is established by one sample per cell. GRU is still
2.07–2.33× incumbent total time, outside the session's 1.3× CPU gate.
Warm cached feature arrays decline 4,444,452 → 4,379,888 bytes; raw transport
is unchanged. Incumbent/control output, gradient and updated-state hashes all
match their before cases exactly. GRU hashes change as intended, while all
prefix hashes match. Load, host facts, source hashes and complete profiles are
retained in the evidence. This is a two-stream randomly initialized CPU policy,
not learnability, strength, CUDA or wide-stream cost evidence.

Validation: 27 focused Python checks passed (exact dense/retained features,
outputs and gradients, immutable earlier-record inputs after appends, both
encoders' mixed empty/live/compact forms, actor/value gradients, checkpoint
round trips and real collection/bootstrap/update). Ruff and whitespace checks
passed. Native code and admission rules were unchanged; the earlier 4,298-case
proof was not rerun for this arithmetic-only slice.

Evidence: `experiments/data/opponent-memory/20261117-sharing-recency` contains
five published manifest entries: measurement/source bundles, per-file
identities and verification. All 60 archive members were verified locally; the
experiment-data tracking guard passed. No fresh remote-fetch verification is
claimed. Next: the
execution-chunk learner fold and its stateless-reference output/gradient proof,
then the separately measured parameter-versioned collection fold. No paid
admission, pilot or matched training/evaluation was launched.
