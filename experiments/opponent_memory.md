# Whole-game opponent memory

## Question

**Exploratory.** Can a learned memory of the opponent's past decisions become
an affordable training basis without materially hurting throughput or early
learning? Census and controlled storage measurements are retained below;
GPU calibration, training and paired evaluation have not run. The USD 50 allocation is a
budget, not a measured cost. No playing-strength finding follows from this report.

The incumbent is the current no-history policy. The best guess is a small
transformer over the opponent's whole-game decision records, with four learned
summary tokens joining the board transformer's token stream. Compare throughput
and learning at matched learner decisions. Do not require a strength gain to
recommend retaining memory as the next basis, but do not interpret a noisy short
run as proof of non-inferiority. Report the size and uncertainty of regressions;
the roughly 10% throughput cost mentioned in the task is context for that
judgment, not a precomputed verdict.

## Method (planned, not executed)

Use five arms with three independent training seeds per arm, sharing
seed labels across arms for blocking. The seed, not an evaluation game, is the
unit of learning evidence. No factorial cross and no dedicated control-deck
training arm in this pilot; stratify the existing Elemental matchups instead.

| Arm | Changes from best guess |
| --- | --- |
| No history | Incumbent; no memory tokens or recent events |
| Transformer, four summaries | Best guess: two record-transformer layers, four summary tokens |
| GRU, four summaries | Single-layer GRU, same record width and four summary tokens |
| Transformer, one summary | One summary token instead of four |
| GRU, one summary | Single-layer GRU, one summary token |

Jack Heart selected transformer and GRU, extra board tokens (J1), and summary
counts 1 and 4; see `scratch/questions.md` for the exact direction. This replaces
the earlier six-arm proposal. The experiment module declares these five arms
but deliberately stops before job construction until resource sizing, CUDA admission
and calibrated job wiring are ready. The implementation plan is
`scratch/opponent-memory-design.md`.

Torch only; MLX support is outside this task. Assumption: memory width 64,
four attention heads, feedforward width 128;
keep the current small Elemental board model, optimizer, precision and learning
rule fixed across arms. Use the baseline's 64-wide two-layer board attention,
masked-mean value and FiLM in every arm, so J1 also reaches objectless actions.
Override sampled_self_play's per-wall fitted schedule with one calibrated
learner-decision schedule across arms and disable capacity recovery; long-game
overflow must remain a visible failure. Initialize fresh models; checkpoints from before the
layout change cannot be rebound. Keep parameter counts visible: these are design
alternatives with different costs, not parameter-matched architecture claims.
Run all arms on one new input layout and retain all log records even for the
control; measure the common log/encoding overhead against pre-change replay
receipts separately, so a shared input cost is not hidden by the control.

### Shared software before a paid run

Build on integrated main, inspect PRs #549 and #550 and reconcile their edits to
policy history, observation encoders and search. Do not rewrite their frozen
evidence or add a checkpoint compatibility path.

Native records each opponent decision for each viewer before applying the
choice, then appends only its viewer-safe public outcome. Include public decision
kind, turn/step, untapped mana-source characteristics, hand size, life totals,
board summary, the attack/spell/source it responds to, and the public choice.
Never encode hidden alternatives or private legal-action counts. Card identities
use native viewer entitlement at the time; future reveals cannot retrospectively
label earlier unknown cards. Own decisions are excluded. Record eligible passes
only with priority, an untapped mana source and a card in hand. Coalesce adjacent
passes only if their entire viewer-safe context and response are identical;
retain their count and chronological span. Nontrivial opponent choices are
included; trivial skipping remains enabled, and no model sees forced decisions.

Use immutable historical snapshots, not joins that accidentally read a changed
live object. Preserve public object identity across references without turning
raw storage IDs into learning tokens. Keep records at native authority so forks,
replay, resume and Academy reconnect reconstruct the same input without carried
neural state. New state must reach native and Python tensors in parity, learning
tokens, LLM text, and exhaustive observability/privacy guards for both seats.

Encode record chronology with relative order and turn/step; padding is masked.
The transformer summarizes the records into learned queries. The GRU reads the
same records in order and projects its final state to K summary tokens. Board
tokens can attend to these summaries; actions cannot attend to individual
records. Recompute from the full game for the pilot. Defer caching unless real
measurements demand it; any future cache is derived and invalidated by game,
viewer, prefix and model weights. Gradients must reach the record encoder from
the ordinary policy/value losses. Empty history has defined masked behavior.

The native log is whole-game. Tensor capacity is sized after the census; overflow
is a typed Capacity error, never truncation, dropping old decisions or silently
recovering away the longest games. Variable-length preparation or padding buckets
must preserve every record and companion shape. Publish ordinary software tests
for privacy, coalescing, ordering, scalar/vector parity, replay/fork identity,
empty histories, overflow and encoder gradients. Do not add schedule tests for
this one experiment.

### Smallest execution sequence

1. **No rental: replay census.** Locate PR #549's retained 512 random games and
   replay their original Commands under a compatible engine; record exact source
   identities. Count per-viewer qualifying records from pre-choice public state,
   not presentation-event rows. Report decision-prefix and terminal p50/p90/p95/
   p99/max lengths, eligible passes before/after coalescing, no-pass lengths,
   records by kind, game length and matchup, and projected buffer bytes. If those
   bytes or replay compatibility are unavailable, say so; do not invent counts,
   rebind old worlds or substitute freshly played games without dating the change.
   Random games size the first capacity; the paid pilot must also report its own
   lengths, failures and peak memory because learned play can be longer.
2. **Calibration, at most USD 3.** On one authorized GPU host, interleave real
   no-history repeats with the four memory arms using retained replay inputs and
   short production-path updates. Include initialization/checkpoint reload and
   native encoding, preparation, collection inference, learner and peak memory.
   No artificial load. Separate replayed phase timings from whole-update rates.
   Record host CPU/quota, GPU, source/world/input and precision. Use these readings
   to choose one common learner-decision horizon and milestone spacing that fit
   the training allowance; date that choice here before training. If five arms
   cannot fit meaningful learning, report that rather than covertly narrowing the
   comparison or making a learning claim from smoke updates.
3. **Learning pilot, at most USD 27.** Fifteen fresh runs, seeds 39101–39103,
   five arms, at most USD 1.80 each including startup and finalization. Inspect
   PR #547 / Task 4b0591 before freezing: if mixed per-game disclosure training
   has landed, use a balanced open/closed mix identically across arms; otherwise
   train open lists and label closed-list evaluation as a disclosure shift.
   Build the
   current-world Elemental uniform 10-deck self-play recipe with
   `manabot.recipes.builders`, both learner seats and ordinary complete-game
   closure. Fix the damping schedule across arms. Record actual learner counts;
   equal updates or equal wall limits do not establish a matched-decision
   comparison. Use shared decision milestones with retained absolute counts and
   compare the common supported range. If complete batches overshoot, show the
   actual neighboring counts and interpolation explicitly, never relabel them as
   exact matches or extrapolate a stopped run. Retain initialization, midpoint
   and endpoint checkpoints for early learning curves.
4. **Paired evaluation, at most USD 12.** At each milestone, each history-arm
   checkpoint plays its same-seed no-history checkpoint at matched learner
   decisions on the same held-out Elemental 10x10 grid, separately under open
   and closed decklists, with
   native `MatchConfig.decklists` as authority. Disclosure adds evaluation
   conditions, not training arms; neither direction of its interaction with
   memory is assumed. Assumption: one deal per unordered deck pair including mirrors,
   seeds `3_910_000_000 + 2*pair_index + deal_index`, lexicographic pair order.
   (`deal_index=0`). Cross-deck pairs play all four deck/seat legs; mirrors play
   two seats. This gives 200 games per disclosure, 400 per checkpoint pair, and covers
   every ordered matchup in both conditions. Keep deals
   disjoint from training/monitoring, retain exact seats, Commands and replay
   validation, and report failures beside scores. If this exceeds the allowance,
   report incomplete coverage rather than silently changing deals. Reuse the
   arena's replay-checked game path; do not duplicate a game runner here.
   At the endpoint also play each history checkpoint against a copy of itself
   with history blanked before preparation, on the same paired deals and both
   disclosures. Preserve shapes and all current board/legality information;
   blank that arm's records, including length/count/presence cues. Tag
   these `reliance`, separately from primary `control` games. This is a test of
   reliance under an intervention, not evidence that memory improves learning.
   There are 14,400 primary games and 4,800 endpoint reliance games at three
   milestones and three seeds. Quote/calibrate their cost before allocating;
   retain partial coverage explicitly if the allowance cannot finish them.
   A heuristic curve is optional context only, after the primary comparisons.

If affordable from the same allocation, fit a linear hand-content probe on frozen
summary tokens and the identical probe on the control's pooled state. Collect
labels from a privileged offline recorder only, never from policy inputs. Split
by whole held-out games/deals and training seed, fit on separate probe-training
games, and show hand-size/deck-frequency baselines and uncertainty. No encoder
fine-tuning, training changes or additional rentals solely for the probe. Its
labels cannot enter native viewer observations, caches or replay projections.
Mark this diagnostic deferred if data collection/fitting does not fit the cap.

USD 8 remains for failures/retries and unresolved startup cost, inside the same
USD 50 total, not an additional budget. The task already authorizes this amount.
Etude owns launch/admission/recovery; inspect existing runs before allocating,
quote full startup plus `Job.max_hours`, launch with `--max-usd` bounded by the
remaining total, record each launcher's cost once, and verify rentals ended.
Never start an unsupported arm or substitute a no-history policy for it.

### Planned analysis

Report learner decisions/second and the candidate/incumbent ratio with repeated
same-host baseline variability; include common native-log overhead, model size,
memory and phase breakdown. Treat differing trajectories and game lengths as
part of whole-training cost; retained-input timings isolate encoder cost.

Plot overall head-to-head score against the no-history control (neutral score
0.5) against actual learner decisions and elapsed training time,
show all three seeds and paired differences, and give the 10x10 matchup table
plus per-deck and play/draw breakdowns, separately for open and closed lists.
Carry a `decklists` column into the evaluation `plays` table; compare each
history arm against the incumbent within disclosure, not just pooled scores.
The matchup table is descriptive: show intervals, including the question of
whether an arm wins from both decks of a pair. Test heterogeneity once across
all matchups with paired deal blocks and seed strata; do not select the largest
cell or claim effects this small pilot cannot resolve. Use seed-level uncertainty for learning
and paired-deal uncertainty within each seed; hundreds of games do not supply
hundreds of independent training replicates. Report numerical health, gradient
flow and loss/entropy/value diagnostics separately from durable strength.
The report's Etude clock is `learner_transitions`, the existing learner-decision
coordinate; attach it to checkpoint exports when wiring the run.

Separate learned skill from deck strength by retaining every deck/seat leg of a
whole deal. Keep rules, world, exposure and disclosure matched across arms;
a rule change or disclosure shift is not a memory effect. Use policy-only
inference for the primary comparison; any search-assisted result needs separate
compute accounting and cannot stand in for it. Report paired whole-deal score
differences and uncertainty within seeds, then use training seeds as the unit of
evidence for the method.

## Results

The effect on playing strength is unmeasured: there is no paired score difference
in percentage points and no uncertainty interval. There are no trained seeds or
completed evaluation deals from which to estimate either. The planned 19,200
games are a design count, not an observed sample; no score denominator has been
populated. In a partial evaluation, missing games must remain identified as
missing in the planned denominator, never scored as zero or dropped.

The 2026-11-16 no-rental census regenerated the source seed workload: 512 games,
102,209 decisions, zero capped games and step digest `17459788495858550277`,
matching the retained reference. This is the original two-deck matchup, not an
Elemental 10-deck or learned-game capacity certificate. It is not retained-Command
replay. There are 102,209 acting-decision prefixes and 1,024 terminal viewer logs.

| Rows | p50 | p90 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix records | 46 | 121 | 164 | 313.92 | 582 |
| Terminal records | 68 | 192.70 | 302.70 | 442.08 | 595 |
| Prefix details | 14,504 | 56,546 | 83,419 | 255,418.88 | 616,295 |
| Terminal details | 26,571.50 | 107,218.30 | 250,918.15 | 420,561.22 | 665,431 |

Eligible passes numbered 17,748 before and after coalescing; none in this workload
had identical consecutive contexts. Power-of-two candidate capacities are 1,024
records and 1,048,576 details, or 33,628,160 bytes per padded observation. At the
planned 256-stream × 64-decision minimum, the existing dense `_ObservationRows`
store requires **513.125 GiB for memory alone**, before its 25% spare allocation,
native buffers, other observations and complete-game overshoot. The recipe's
declared limit is 40 GiB. This is a storage counterexample, not a safe capacity
configuration to launch.

The [manifest-backed census receipt](data/opponent-memory/20261116-census/native.jsonl),
[summary](data/opponent-memory/20261116-census/summary.json) and
[source identity](data/opponent-memory/20261116-census/source.json) retain the
measurements; the engine patch is stored beside them. Reproduce the analysis with
`uv run python -m experiments.opponent_memory census experiments/data/opponent-memory/20261116-census/native.jsonl --out .runs/opponent-memory-census/summary.json`.

Whole-training throughput, GPU memory, CUDA admission and baseline timing noise
remain unmeasured. The census run includes diagnostic reads and is not an ordinary
throughput comparison. Focused CPU checks exercise both encoder choices, both
summary counts, independent actor/value gradients, checkpoint reload, padded/live/
compact/bucketed paths and bounded ordinary updates. Those fixtures establish
implementation health, not learnability or readiness at the intended load.

Training/evaluation declarations still stop before constructing jobs. Dependent
wiring is stopped pending full resource admission; there was no paid run
or rental, and there is no launcher run ID or cost to record.

## What the results support

The census establishes that dense whole-rollout history storage does not fit the
planned load. The shared-prefix implementation below removes dense rollout
ownership, while selected-batch resource sizing remains unresolved.
There is no basis
to compare any candidate with today's no-history incumbent, choose a transformer
over a GRU, choose the number of summary tokens, or judge the value of passes.
Open versus closed decklists, matchup-specific advantages, the learning noise
floor and memory reliance all remain unresolved. Absence of runs supports
neither equivalence nor a regression.

## Recommendation

Keep the no-history incumbent. Shared immutable game/viewer prefixes now back
collection, selected gathers, bootstrap and recovery. Bootstrap retains prefixes
without keeping padded memory through the update. The original controlled proof
combined host and device bytes; the new proof separates them and establishes
finite one-row CPU execution, not pilot fit.
Collection, frozen-opponent and bootstrap inference still materialize every
selected prefix at once; explicit execution row limits must bound them, with
output and gradient parity, before sizing. Then calibrate those row sizes with
separate host RSS and CUDA peaks before concrete job wiring; record any
workload choice here. Preserve every record and typed overflow.
Do not adopt memory or remove recent events. This report launches no new run.

Evidence that would change this recommendation is a measured comparison at
matched learner decisions across independent training seeds, with whole-deal
paired uncertainty, affordable whole-training cost and disclosed failures.
Comparable early learning at an acceptable cost could justify keeping memory
available even without a resolved strength gain; noisy curves alone would not
establish non-inferiority. The roughly 10% throughput guide informs that judgment
rather than deciding it automatically.

## Caveats

The strongest confound is that short self-play horizons can hide small effects
inside training variance. Open lists could make past choices more interpretable or reduce
their usefulness; report the interaction without assuming either. Learning
failure on all arms is an uninformative pilot. Similar-looking noisy curves do
not establish equivalence. Broad regressions favor the incumbent; comparable
learning with affordable throughput supports keeping memory available and the
best supported design as the next basis. A localized benefit or cost calls for
its matchup-specific explanation. Inconclusive remains a legitimate outcome.

Adopting the basis has a **separate follow-up**: replace the recent-event window
and its option and reconcile PR #549's event kinds with decision records, keeping
any independent public knowledge needed by observations, text, replay and Academy.
Do not remove those surfaces in this task or treat replacement as already done.

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
manifest-backed; all 16 recorded source hashes match the measured source.
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
successful receipt match the checkout. Earlier receipts remain unchanged. No
rental, paid calibration, learning pilot or evaluation ran. Etude job construction
stays stopped pending CUDA peaks, the ordinary worker/update, checkpoint reload
and matched-decision horizon calibration.


Validation: `uv run --no-sync pytest tests/model/test_opponent_memory.py tests/model/test_prepare_rows.py -q` — 54 passed; artifact tracking guard — one passed. Ruff, patch reconstruction check and `git diff --check` passed. Full gate remains outstanding.
