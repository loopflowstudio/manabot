# Whole-game decision memory

## Question

**Exploratory.** Does a learned memory of the opponent's past decisions change
what a policy learns from the same number of learner decisions? The memory is
the viewer-safe log of the opponent's whole game, read by a GRU with one summary
that every board layer adds through a gate. The comparison is `control` (the log
is kept and validated, no encoder reads it) against `gru-k1`, on mixed-disclosure
Elemental 10v10. The history-disabled incumbent is the throughput reference only.

## Answer

**No, at this scale. The work is paused with memory off by default.**

- **The policy did not use its memory.** Across three seeds the gates ended
  at 0.06 to 0.08 of full strength, and a final GRU checkpoint scored 0.48 to
  0.53 against the same checkpoint with its memory blanked.
- **The GRU arm still ended somewhat stronger.** Final GRU against control
  scored 0.54 to 0.62, and against the heuristic 0.42 to 0.44 where control
  scored 0.31 to 0.42. Its games held fewer decisions, so the same number of
  decisions gave it more updates (136 to 150 against 86 to 114). The cause of
  the shorter games is open; shared gradient clipping is ruled out as the
  main one.
- **Neither arm had finished learning.** Both rose from about 0.10 against
  the heuristic at the branch point and were still below it at 7 to 8M
  decisions.
- **The memory holds little hand information here.** Fitted to predict the
  opponent's hand on 7,634 heuristic-game decisions, it improved closed-list
  log loss by 0.002 to 0.005 over the public state and did not help with
  open lists. Scripted and weak play may be why: history is informative when
  choices depend on the hand in ways the board does not show.
- **Cost.** With history off this source costs the same as main: 0.998× on
  an A40 and 1.0044× on CPU. With memory on, the A40 measured 1.105× for
  control and 1.201× for the GRU arm.

The wide model was not run. Total rental was about USD 24.

**To pick up later:** rerun when policies beat the heuristic, since stronger
play is where history should carry information; compare arms at matched
decisions and matched updates; and decide whether hidden-hand labels may
enter the training loop before building an auxiliary loss.

Each result has a dated entry in the appendix.

## Method

The pilot compares memory within two model sizes. Small has 64 hidden units
and two attention layers. Wide has 128 hidden units and three attention layers.
Both use four heads. Each size trains three control trunks (seeds 39101–39103)
to 3.3M learner decisions. Within each size and seed, control and GRU-K1 start
from the same trunk EMA with fresh Adam and a fresh 8.8M-decision schedule.
Zero gates give GRU the control policy at initialization. The gate scale stays
at 2.0. The panels record each layer’s gate during training.

Both sizes use 256 streams, four producers, complete-game collection, float32,
128 execution rows and record/detail/reference/member capacities of
1,024/65,536/512/32,768. Each game draws open or closed decklists with equal
probability. Overflow fails the run. Complete-game collection can overshoot the
decision target by one update. The branch update ceiling is 240. Compare raw
checkpoints at equal interior updates and each arm’s own final update, retaining
actual learner clocks.

Each pairing plays four games over both seats and decks. Each size uses the
same held-out deals, separately for each seed and disclosure.

| Readout | Checkpoints | Pairings per seed and disclosure | Games per size |
|---|---|---:|---:|
| GRU against control | final | 200 | 4,800 |
| GRU against control | updates 40, 80 and 120 | 50 per update | 3,600 |
| Intact against blanked GRU | final | 50 | 1,200 |
| Total | | | 9,600 |

The full design plays 19,200 games. Within each size the final head-to-head
has a nominal 95% margin of ±1.4 percentage points near a 50% score, ±2.0 per
disclosure or ±3.5 per seed and disclosure. Those margins assume independent
games. Paired legs share deals, and generalization across training runs rests
on three seeds. The curves show each seed separately.

A blanked player is the same checkpoint with its four memory tables zeroed
before every decision; the board, legal actions and native game are untouched.
It measures whether the trained policy uses the log.

The report gives each training seed its own head-to-head panel, split by
decklist disclosure. The play/draw ranking counts head-to-head games against
control only.

### Running it

Use the committed revision and the same store that holds the producer receipts.
Each training selection below is independent. Run the small branches first
once their three trunks finish. The wide commands describe the full design for
the launching session after the budget decision.

```sh
pilot=experiments/opponent_memory.py
store=r2://manabot/jobs/opponent_memory
revision=$(git rev-parse HEAD)
train() {
  uv run etude launch "$pilot" --source "$store" --revision "$revision" \
    --rent mtg252-a40 --max-usd 29.95 "$@"
}

# Small branches
for arm in control gru-k1; do
  train --size small --trunks "$store" --job "decision-memory-$arm"
done

# Wide trunks
train --size wide --job decision-memory-wide-trunk

# Wide branches, once all three wide trunks finish
for arm in control gru-k1; do
  train --size wide --trunks "$store" --job "decision-memory-wide-$arm"
done

# Remote evaluation after the three small trunk addresses already exist.
# Set eval_ceiling to current committed USD plus the remaining eval allowance.
uv run etude launch "$pilot" --source "$store" --revision "$revision" \
  --rent mtg252-a40 --max-usd "${eval_ceiling:?set cumulative evaluation ceiling}"

uv run etude report "$pilot" --source "$store" --open
```

Etude counts prior experiment commitments and may admit only part of a cohort.
USD29.95 covers calibration plus the USD27 training allocation, including failed
trunks. The session should inspect admission for each arm. Training commands
filter by job so they skip evaluation. For a remote evaluation pass, the default
declaration resolves the three known small trunks and evaluates available
checkpoint pairs from both sizes. Limit all passes together to USD12 and the
experiment to USD50. Finished matches are retained between passes.

## Cost

| Stage | Runs | Planned USD | Sum of job caps USD |
|---|---:|---:|---:|
| Calibration, prior spend | | 2.95 | |
| Failed first trunks, prior spend | | 0.60 | |
| Small trunks | 3 | 2.52 | 7.20 |
| Small control branches | 3 | 5.99 | 7.65 |
| Small GRU branches | 3 | 7.38 | 9.45 |
| Wide trunks | 3 | 4.60 | 13.05 |
| Wide control branches | 3 | 11.54 | 14.85 |
| Wide GRU branches | 3 | 14.31 | 18.45 |
| Pilot training subtotal, successful runs | 18 | 46.34 | 70.65 |
| Evaluation | up to 3 passes | up to 12.00 | 4.00 per pass |
| Inclusive total with full evaluation allowance | | 61.89 | |

The projection uses USD0.59/h, 0.25 h startup per successful job and 780 small
control decisions/s. It assumes GRU takes 1.25× control time and wide takes 2×
small time. A wide trunk is projected at 2.35 compute hours. Its 7 h wall cap
doubles the existing small trunk cap. Control/GRU branch wall caps are 4/5 h
small and 8/10 h wide, including setup and publishing time. The hourly rate,
GRU cost and wide cost need confirmation by the launching session. Evaluation
throughput is unmeasured. Each pass has a two-hour wall cap.

The full schedule exceeds USD27 training and USD50 inclusive. Preserve the
three small-seed comparisons first and defer the wide cohort if those budgets
stand. Small projects to USD16.49 training including the failed trunks. The
declarations retain the full wide design pending a funding or scope decision.

### Trunk quote — 2026-11-17

Source `2144305471` declares the three trunks and the branch/evaluation wiring.
The original trunk quote, with that revision, selected all three seeds
39101–39103 and refused:

> cannot launch on mtg252-a40: provider reports no available machines; etude queue would wait

At 2026-11-17 16:20:38 UTC the provider reported USD0.59/hour (ceiling USD0.60),
none available, zero of nine in use and not rentable now. There is no admitted
whole-rental quote; the hourly offer does not guarantee capacity or price the
startup window. The planned-dollars table remains an estimate.

The launcher reported 10 retained training runs, zero eval runs and USD2.95
committed. Its run listing contained calibration runs only, with no trunk or
branch. The initial quote used USD7.20 as the experiment ceiling and selected
only one trunk because prior calibration spending also counts; the corrected
USD10.20 ceiling selected all three. Both were quote-only refusals. Nothing was
queued, launched or rented, and no pilot learning comparison has run.

## Limits

- Horizons come from 32-row A40 rates. The 128-row setting and the two
  record-cache commits after that calibration have no CUDA measurement; the
  first trunk is their admission for the control, and the first GRU branch for
  the GRU.
- Evaluation plays raw weights; EMA weights are used only to start branches.
- The transformer encoder, own-seat records, disclosure-only training arms and a
  hand-content probe on the summaries are built or designed and not in this
  pilot.

## Recommendation

Keep the history-disabled incumbent as the default until the pilot reports.
With history off the CPU fixture matches main, so the memory can stay available
at its measured cost whatever the pilot shows; a default change needs the
matched-learning and paired-strength result, with failures and uncertainty kept.

The dated appendix retains original observations, including results superseded
by later measurements and the throughput-first plans this method replaces. The
earlier undated protocol and storage counterexample can be recovered from
`7d680aac170f12ca60e60dfb812e0c00d7cda7fa:experiments/opponent_memory.md`.

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

## Execution-chunk learner fold — 2026-11-17

**Keep the learner fold; the CPU gate is still unmet.** Attribution did not
trigger the seventh review's stop condition. The GRU update now prepares and
embeds each distinct selected chain node once, and advances the recurrent state
from its exact predecessor. Policy rows read their own tails, preserving order,
duplicates, actor/critic denominators and one clip/Adam step per logical
minibatch. There is no neural state beyond the execution chunk's autograd graph.
Collection/bootstrap and transformer encoding remain stateless; a collection
fold is still a separate, unimplemented step.

First, run the existing suffix-capture whole-game harness at 32 execution rows
on source `a275e27ff086`, with injected CPU wall timers inside DecisionMemory
and its GRU only. Tensor hooks on hidden output and packed input gradients bound
recurrent backward; semantic-card GRUs are excluded. The same two streams,
556 learner decisions and one actual update take 5.315 s update time. Memory
forward is 0.585 s, including 0.044 s recurrence; recurrent backward is 0.183 s.
Thus recurrence forward + backward is **0.227 s**, below the design's 0.98 s
stop condition. This is the measured batch-width recurrence, not a prediction
of folded recurrence speed. Instrumentation preserves exact collection output,
prefix, final gradient and updated-state hashes against the later stateless
32-row case.

A separate Torch autograd profiler run on that same workload attributes CPU
self time by each forward operation's autograd sequence number and its matching
backward evaluate-function range. Sum self time once, avoiding nested ranges:

| Memory-local work | Forward (s) | Backward (s) |
|---|---:|---:|
| Detail/reference/member/record embedding and summary projection/normalization | 0.556 | 0.897 |
| GRU recurrence | 0.056 | 0.182 |

The profiler's update takes 5.789 s; its overhead is not a throughput result.
Local embedding/summary work dominates recurrence (1.453 versus 0.238 s).
The earlier 2.78 s GRU-over-control update excess is not wholly attributed by
these scopes: shared definition encoding/gradients, extra gated board attention,
preparation, optimizer and other work remain outside. Do not subtract profiler
self times from an unprofiled wall-time difference and present the residual as
another measured phase. The wall timers and profiler agree that recurrence
alone does not trigger the stop, permitting this fold.

Host preparation now builds a temporary node tree from the selected immutable
prefixes, routes both newly introduced and predecessor-owned references into
one feature table, and prepares packed GRU calls by tree depth. Each node's
predecessor state is an ordinary differentiable tensor; repeated descendants'
gradients add through it. Dense inputs and stateless retained preparation remain
the behavioral reference and the inference path. The fold owns only prepared
arrays, not native prefixes or a learned cache. There is no sampling change,
row reordering, truncated history, new model parameter or checkpoint migration.

Measure the implementation with 36 fresh sequential CPU processes: three
repeats at each of 32/128/256 execution rows, incumbent, log-enabled control,
stateless GRU and folded GRU. Alternate case order on the middle repeat. Only
stateless GRU substitutes `_prepare_retained` for `_prepare_fold`; every other
production path and the harness are identical. No build, tests or induced load
ran concurrently. Medians [min, max], seconds:

| Rows | Arm | Collection | Update | Total |
|---|---|---|---|---|
| 32 | incumbent | 1.460 [1.419, 1.474] | 2.596 [2.583, 2.938] | 4.057 [4.015, 4.398] |
| 32 | control | 1.620 [1.542, 1.685] | 2.617 [2.587, 3.111] | 4.207 [4.158, 4.796] |
| 32 | stateless GRU | 2.886 [2.691, 2.910] | 5.330 [5.310, 5.350] | 8.196 [8.021, 8.261] |
| 32 | folded GRU | 2.879 [2.832, 2.969] | 3.827 [3.813, 3.886] | 6.765 [6.645, 6.796] |
| 128 | incumbent | 1.353 [1.301, 1.481] | 2.031 [1.938, 2.060] | 3.413 [3.332, 3.419] |
| 128 | control | 1.464 [1.449, 1.644] | 2.028 [1.963, 2.062] | 3.526 [3.478, 3.607] |
| 128 | stateless GRU | 2.933 [2.734, 2.973] | 4.851 [4.826, 4.959] | 7.759 [7.693, 7.825] |
| 128 | folded GRU | 2.897 [2.855, 2.910] | 3.203 [3.178, 3.237] | 6.100 [6.033, 6.147] |
| 256 | incumbent | 1.365 [1.315, 1.506] | 1.992 [1.924, 2.004] | 3.357 [3.319, 3.429] |
| 256 | control | 1.561 [1.431, 1.647] | 2.006 [1.952, 2.009] | 3.570 [3.437, 3.599] |
| 256 | stateless GRU | 2.977 [2.969, 3.933] | 5.038 [4.954, 8.324] | 8.015 [7.923, 12.257] |
| 256 | folded GRU | 2.788 [2.749, 3.999] | 3.255 [3.171, 4.636] | 6.043 [5.920, 8.635] |

Update medians improve 1.393/1.515/1.548×, with separated ranges at every row
limit. Total medians improve 1.212/1.272/1.326×; total ranges overlap at 256
rows, and collection is unchanged by this implementation. Folded total rates
are **0.600/0.559/0.556× incumbent**, outside the session's 1.3× time gate.
All cases retain 556 learner decisions. Stateless/folded GRU collection outputs
and prefix hashes match exactly at each limit; prefixes are not assumed
byte-identical across different arms. Counts remain 272/5,688/82/3,592.
The release extension stays `94bb1e63…`; complete identities, host/load records
and per-phase profiles are in the receipts. These are two-stream random-policy
software measurements, not wider-stream host fit, CUDA, learning or strength.

Validation: focused tests compare action/value outputs and every parameter
gradient against dense stateless encoding, at tolerances 2e-6/2e-5 for outputs
and 2e-5/2e-4 for gradients (absolute/relative). They cover uneven depths,
repeated/shuffled rows with unequal loss weights, a reference introduced after
the first chunk, backwards reference joins, changed roots, fork/coalesced tails,
K1/K4, live/padded attention and a parameter update. Prepared arrays retain no
native nodes; empty history is exactly zero and transformer stays stateless.
The selected execution checks preserve sampling, bootstrap and Adam. Final
focused run: 17 passed, plus the earlier real-prefix/checkpoint/cache run
(9 passed). Ruff, whitespace and the experiment-data tracking guard pass.
Native code/admission were unchanged; no native rebuild or full suite rerun.

Evidence: five manifest entries published under
`experiments/data/opponent-memory/20261117-learner-fold`, with all 93 archive
members verified locally. No fresh remote fetch is claimed. Raw measurements
and source are retained in `.runs/decision-memory-attribution/` and
`.runs/decision-memory-learner-fold/`; they contain the exact pre-fold source.
No paid work, pilot or matched training/paired evaluation was launched. The
Task's scientific comparison remains unrun. Next is the separately measured
collection fold; paid admission and decision 1 remain unresolved.

## Collection-fold attribution and stop — 2026-11-17

**Stop before building the collection cache; return to decision 1.** On source
`4a01b1507e31`, the measured earlier-node preparation plus arithmetic medians
are **0.577/0.581/0.586 s** at 32/128/256 execution rows. They are below the
eighth review's half-excess lines, **0.630/0.715/0.615 s**. This applies that
agent-defined rule to medians, a headless engineering choice rather than a new
human throughput requirement. No production code, model, cache or import
structure changed in this slice. The proposed parameter-versioned collection
fold remains unbuilt.

Use the same two-stream complete-game CPU harness, fixed initialization, gates,
seeds, mixed disclosure and real learner update as the learner-fold entry.
Nine fresh sequential processes (three repeats per row limit) each retain 556
learner decisions. No other measurement or check was launched concurrently.
Inject wall timers into the existing `prepare_memory` call and retained
preparation's node loop: earlier nodes have a successor in that supplied
prefix; the last node is recomputed. Time canonical empty-shape construction
separately; all remaining host work is shared assembly. The classification is
per supplied prefix, preserving repeated rows.

For forward attribution, first run the original `DecisionMemory.forward` and
return its result to collection. Then recompute that same supplied input in a
diagnostic partition: references introduced in earlier nodes versus the last
node, followed by earlier versus last records with unchanged backwards
reference joins. Advance the GRU over earlier records, then the last records
from that hidden state. Time arithmetic separately from diagnostic slicing,
route construction and packed-schedule construction; summaries are separate.
Every diagnostic summary matches the original at atol=2e-6, rtol=2e-5,
including empty histories. Reference embeddings remain shared across that
partition; a last-chunk member can still use an earlier reference.

This partition changes kernel batching and recomputes the input; its timers
are **attribution measurements, not a cache speedup or unmodified collection
throughput**. The raw collection wall times include that diagnostic work and
must not be used for throughput ratios. Original full-forward timers below
exclude it. Frozen source and per-call samples accompany the receipts.
Medians [min, max], seconds per complete collection:

| Rows | Preparation: earlier nodes | Forward arithmetic: earlier nodes | Their paired sum | Half-excess stop line |
|---|---:|---:|---:|---:|
| 32 | 0.012 [0.011, 0.012] | 0.565 [0.540, 0.577] | 0.577 [0.551, 0.589] | 0.630 |
| 128 | 0.012 [0.011, 0.013] | 0.569 [0.555, 0.644] | 0.581 [0.567, 0.657] | 0.715 |
| 256 | 0.012 [0.011, 0.015] | 0.574 [0.551, 0.695] | 0.586 [0.562, 0.710] | 0.615 |

| Rows | Preparation: last node | Preparation: empty shapes | Preparation: shared assembly | Forward: last records/references | Forward: summary | Original full forward |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.095 | 0.072 | 0.078 | 0.226 | 0.008 | 0.772 |
| 128 | 0.098 | 0.074 | 0.079 | 0.230 | 0.008 | 0.777 |
| 256 | 0.097 | 0.075 | 0.079 | 0.235 | 0.009 | 0.784 |

Limits matter near this line. The 256-row earlier-node range crosses its stop
line. The 0.078–0.079 s median shared assembly is not divided into avoidable
and fixed work: charging all of it to the earlier-node sum would put 32 and
256 rows above their lines. Partitioned arithmetic also has different kernel
shapes from the original; its phase medians need not sum to the original
forward median. Thus this is an operational stop using the measured node-work
medians, **not proof that every possible collection optimization saves less
than half the excess**. It does not establish that the remaining excess is
all last-chunk or board-join arithmetic. No further cache implementation or
measurement is inferred from this uncertainty in a headless run.

All nine receipts match the corresponding existing learner-fold case exactly
for collection output, prefix, final gradient and updated-state hashes, with
556 decisions and the same release extension `94bb1e63…`. There are 395
collection preparation/forward calls per process. The diagnostic cannot change
sampling: only the original forward result is used and it consumes no RNG.
Exploratory attempts before the final partition kept slice-construction costs
inside its timing; those are superseded, retained separately and excluded from
the tables. Final runs are repeats 8/9/10; the frozen runner and partition
source identify that method. Host load and complete identities remain in each
receipt. No learning, strength, CUDA, wider-stream host-fit or new end-to-end
speedup result follows.

The kept implementation's last unmodified total rates remain
**0.600/0.559/0.556× incumbent** (previous entry). Even replacing collection
with control's time leaves **1.343/1.367/1.435× incumbent time**, above the
session's 1.3× admission gate. Decision 1 remains how much host cost must come
off before paid work resumes, and whether a different model cost is acceptable.
No paid admission, pilot or evaluation was launched; this Task is incomplete.

Evidence: `experiments/data/opponent-memory/20261117-collection-attribution`
contains source and measurement archives with per-member hash manifests and
local archive verification. Publication and validation are recorded below.

Publication/validation: five artifact manifest entries published; all 22 archive
members verified locally (9 source, 13 measurement), no fresh remote fetch.
`uv run --no-sync python .runs/decision-memory-collection-attribution/summarize.py`
passed all nine exact original-result comparisons; 3,555 diagnostic forward
comparisons passed inside the measurements. Only evidence/docs changed; native
rebuild and model test reruns are unnecessary. `uv run --no-sync --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`
passed (1 test); `git diff --check` passed. No background work remains.

### Optimized-source CUDA admission: supply unavailable — 2026-11-17

The Task session `session_c8789f79209f4b03bb0c28923fa63e81` chose open
decision 1 option b and waived the session's 1.3× CPU gate for one admission.
This is the session choice, not a new quotation from a person. The allowance
is at most USD1.00 for this job and USD3 cumulative calibration; pilot training,
evaluation, model shape, coalescing and capacities remain unchanged.

Source **`6cfc8cf530`** commits the optimized implementation and its bounded
`--calibration` declaration. It retains refs256's launcher setup, source binding,
CUDA preflight and ordinary fresh-worker / producer-EMA admission. The existing
sequence is fresh control producer, incumbent, pretrained control, folded
GRU-K1, incumbent, two updates each: mixed-disclosure Elemental, float32,
256 streams, four producers, 32-row collection/bootstrap/learning execution,
512/16,384/256/8,192 memory layout. No transformer worker is declared.
The job permits one attempt, 1.25 hours and USD1.00. Optional separately labelled
256-row workers were not added after the supply stop.

**Stop: no A40 supply.** At 2026-11-17 09:52:56 UTC the existing `mtg252-a40`
on-demand returned USD0.59/hour (USD0.60/hour ceiling), none available, zero of
nine in use, not rentable now. The formal whole-rental quote, including the
launcher's startup allowance, then refused:

> cannot launch on mtg252-a40: provider reports no available machines; etude queue would wait

Consequently there is **no admitted whole-rental dollar quote**; the hourly
offer is not a rental quote or a capacity guarantee. Nothing was queued or
launched, and there was no provider retry or alternative supply. The quote-only
command used the existing scoped source `r2://manabot/jobs/opponent_memory`,
`--job cuda-calibration-optimized --calibration --revision 6cfc8cf530`,
`--rent mtg252-a40 --max-usd 3.00 --quote` to retain the cumulative check.
No launch command was issued; the requested launch ceiling of `--max-usd 1.00`
was never exercised. The job itself is capped at USD1.00.

The launcher's retained prior total is **USD1.944746355308370234114244420**,
so the full USD1 job allowance would keep the cumulative commitment below USD3.
All five prior calibration statuses still confirm `rental_ended=true`, and
there is no `cuda-calibration-optimized` run or new rental to terminate.
No wait/fetch/termination call is appropriate without a run or rental; no
worker outputs exist for SDK digest checks. No cost for a new run is reported.

Rows, stage/collection/learning seconds, decisions/s, incumbent-median ratios,
phase split and CUDA/host peaks are **unmeasured on the optimized CUDA source**.
In particular, this attempt cannot identify which CUDA phase holds the remaining
GRU cost. The CPU attribution and old refs256 results remain separate evidence;
they do not substitute for this missing measurement. Admission is incomplete,
and no pilot training, evaluation or PR publication follows.

Evidence: `experiments/data/opponent-memory/20261117-cuda-optimized-unavailable`
retains the exact formal refusal, observed availability, source-bound declaration,
three recipes and launcher cleanup/status observation. Declaration assertions
checked one attempt, dollar/time bounds, two updates, 256 streams, 32-row
execution, float32 and strict game errors. No production/native arithmetic changed.

Publication/validation: all seven evidence files published to the artifact store
and their manifest entries retained; no fresh artifact-store fetch is claimed.
Experiment discovery passed (5 tests), the shared-input branch declaration check
passed (1 test), and the artifact-tracking guard passed (1 test). Ruff, format
and `git diff --check` passed. No background work remains.

### Optimized CUDA calibration and next host cuts — 2026-11-17

Run `opponent_memory/cuda-calibration-optimized-retry@39100`, source `4641c7686b`, NVIDIA A40 (`mtg252-a40-e33ad664e01d`). Five sequential fresh workers, unchanged refs256 workload, two updates each; raw and EMA reloads admitted. Times below are verified from the fetched receipts. Incumbent reference is the median of its two workers, 286.126 s. This is exploratory cost evidence, not learning or strength evidence.

| Worker | Decisions | Stage s | Collection s | Learning s | Time / incumbent | Decisions/s | Host tree peak GB | CUDA allocated peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0: control | 106321 | 321.890 | 197.750 | 122.282 | 1.125 | 330.30 | 15.874 | 0.181 |
| 1: incumbent | 105912 | 276.680 | 153.279 | 121.437 | 0.967 | 382.80 | 15.543 | 0.177 |
| 2: control | 105872 | 327.051 | 202.736 | 122.167 | 1.143 | 323.72 | 15.889 | 0.182 |
| 3: gru-k1 | 105915 | 442.850 | 275.416 | 165.169 | 1.548 | 239.17 | 16.324 | 0.253 |
| 4: incumbent | 106113 | 295.571 | 163.783 | 129.722 | 1.033 | 359.01 | 15.568 | 0.177 |

| Phase s | Incumbent median | Pretrained control | GRU-K1 |
|---|---:|---:|---:|
| Retention (includes capture) | 3.538 | 27.814 | 26.824 |
| Memory capture (nested) | 0.000 | 23.974 | 22.981 |
| Native observation encoding | 1.780 | 9.109 | 9.613 |
| Preparation | 48.808 | 59.657 | 96.391 |
| Inference/action selection | 62.531 | 63.934 | 99.629 |
| Learner forward/loss (includes backward/step) | 115.939 | 112.121 | 155.279 |

Launcher estimate USD0.2958674063638888861128685272; rental ended. The earlier `cuda-calibration-optimized` attempt ended `capacity_refused` at USD0, consuming its single attempt; the successful job therefore uses a new name. The Task session launched with `--max-usd 3.10` because the lost attempt left USD2.06 committed, drawing about USD0.10 from the Task reserve, within USD50. These allowance figures describe session accounting, not reconstructed billing.

Control is 1.143× incumbent time and GRU-K1 1.548× (rates 0.875× and 0.646× using stage seconds); earlier refs256 rates were 0.537× and 0.356×. Worker decision counts differ, so time ratios and decisions/s are both shown. Retention plus native observation encoding remains about 37 s in the GRU; GRU preparation and inference each add about 37 s over control. The CUDA split supersedes the earlier CPU collection stop for the two authorized host cuts. Model shape, capacities, coalescing and transformer evidence stay fixed.

Evidence: `experiments/data/opponent-memory/20261117-cuda-optimized-retry`: all 134 outputs fetched with SDK digest checks (182,529,894 bytes), archived and each archive member verified against the retained SHA256/size manifest; launcher status, attempt and record plus lost-attempt status retained. No new rental, pilot, evaluation or PR. Optimization continues with native suffix transport and a parameter-versioned collection fold.

### Native cursor and suffix transport — 2026-11-17

Per-stream native cursors expose four table counts and a tail revision derived from the last committed decision position, including coalesced passes. Unchanged captures return the issued prefix directly; changed captures re-admit the open chunk. Registered native vector buffers retain a caller-owned snapshot index and write only new table rows plus the potentially coalesced last record. A viewer switch starts a full projection because that row now belongs to another log; reset/import/recovery start a fresh lineage. Scalar encodes, explicit viewer reads and archives retain full-table encoding/admission. No model, capacity or coalescing change.

Existing whole-game harness, two streams, one real update, 556 decisions in every case; 27 sequential fresh processes, three per arm/row limit. All collection, prefix, final-gradient and updated-state hashes match the retained learner-fold receipts exactly. Times are medians [min, max], seconds. Ratios use the fresh true incumbent at each row limit.

| Rows | Arm | Collection s | Update s | Total s | Total / incumbent |
|---|---|---:|---:|---:|---:|
| 32 | incumbent | 1.412 [1.400, 1.421] | 2.589 [2.584, 2.602] | 4.001 [3.985, 4.022] | 1.000 |
| 32 | control | 1.502 [1.480, 1.512] | 2.581 [2.578, 2.588] | 4.080 [4.062, 4.099] | 1.020 |
| 32 | gru-k1 | 2.794 [2.769, 2.822] | 3.821 [3.817, 3.822] | 6.615 [6.590, 6.639] | 1.653 |
| 128 | incumbent | 1.418 [1.409, 1.421] | 2.007 [2.005, 2.046] | 3.425 [3.414, 3.467] | 1.000 |
| 128 | control | 1.535 [1.494, 1.538] | 2.017 [2.003, 2.026] | 3.552 [3.497, 3.564] | 1.037 |
| 128 | gru-k1 | 2.774 [2.758, 2.813] | 3.235 [3.215, 3.281] | 6.009 [5.973, 6.095] | 1.755 |
| 256 | incumbent | 1.419 [1.417, 1.435] | 1.994 [1.975, 1.994] | 3.413 [3.392, 3.429] | 1.000 |
| 256 | control | 1.474 [1.467, 1.508] | 2.003 [1.987, 2.013] | 3.470 [3.461, 3.521] | 1.017 |
| 256 | gru-k1 | 2.781 [2.758, 2.787] | 3.197 [3.196, 3.205] | 5.978 [5.956, 5.992] | 1.751 |

Control CPU time ratios are 1.020/1.037/1.017×; GRU 1.653/1.755/1.752×. The prior learner-fold medians were 1.04/1.03/1.06× control and 1.67/1.79/1.80× GRU. These small differences have overlapping ranges and do not establish a general speedup. The native projection must still be fully rewritten when the acting viewer changes; suffix transport alone cannot remove that cost with one shared output row. No new CUDA result follows.

Validation: touched debug nextest tests passed (2); CPython 3.13 release extension rebuilt from the root and replaced atomically; prefix-storage tests passed (14), including recovery, with the strengthened 140-step native-buffer-versus-dense test passing afterward. Evidence archive retains source, harness, all 27 receipts/logs and member hashes. No paid launch or PR. Collection folding follows as a separate cut.

### Parameter-bound collection fold — 2026-11-17

The GRU now uses one causal fold for learning and collection. Each unchanged PolicyInference scope holds weak complete-node results: detached reference/record embeddings and the node end state. Host preparation omits cached nodes; device forward embeds only new nodes and advances from supplied roots. Every selected open tail recomputes, including old tails selected alongside descendants. Native generation plus prefix identity bounds reuse across reset, fork and resume; parameter changes invalidate the scope and clear its cache. Exit clears the cache. Node-local clones avoid retaining an entire execution allocation, and Prepared result sinks retain only weak node references. No neural state enters observations, replay, checkpoints or search forks. Learner folds have no cached roots; dense stateless encoding and the transformer retain their reference paths. Model shape, capacities, coalescing and transformer evidence are unchanged.

The existing whole-game CPU harness ran 27 sequential fresh processes (three per arm and row limit), each retaining 556 decisions, two streams, mixed disclosure and one real update. Gates are open. Same harness, seeds and workload as the native-cursor cut; original sampling groups and optimizer/loss denominators are unchanged. Times are medians [min, max], seconds; ratios use the fresh true incumbent.

| Rows | Arm | Collection s | Update s | Total s | Total / incumbent |
|---|---|---:|---:|---:|---:|
| 32 | incumbent | 1.420 [1.410, 1.428] | 2.559 [2.553, 2.583] | 3.981 [3.969, 4.004] | 1.000 |
| 32 | control | 1.534 [1.505, 1.544] | 2.571 [2.563, 2.585] | 4.097 [4.076, 4.129] | 1.029 |
| 32 | gru-k1 | 2.239 [2.239, 2.289] | 3.825 [3.798, 3.856] | 6.087 [6.064, 6.095] | 1.529 |
| 128 | incumbent | 1.419 [1.396, 1.441] | 2.021 [1.995, 2.031] | 3.426 [3.414, 3.461] | 1.000 |
| 128 | control | 1.507 [1.497, 1.526] | 2.016 [1.989, 2.031] | 3.528 [3.496, 3.542] | 1.030 |
| 128 | gru-k1 | 2.238 [2.194, 2.272] | 3.198 [3.196, 3.213] | 5.435 [5.407, 5.467] | 1.586 |
| 256 | incumbent | 1.417 [1.375, 1.447] | 1.977 [1.963, 1.979] | 3.397 [3.352, 3.410] | 1.000 |
| 256 | control | 1.533 [1.493, 1.543] | 1.987 [1.952, 2.008] | 3.486 [3.480, 3.550] | 1.026 |
| 256 | gru-k1 | 2.263 [2.247, 2.266] | 3.209 [3.152, 3.212] | 5.456 [5.418, 5.476] | 1.606 |

Control CPU time ratios are **1.029/1.030/1.026×**; GRU **1.529/1.586/1.606×**. After the transport cut, GRU totals were 6.615/6.009/5.978 s; now 6.087/5.435/5.456 s, with collection falling from 2.794/2.774/2.781 s to 2.239/2.238/2.263 s. Update medians remain about 3.82/3.20/3.21 s. This improves the measured CPU path but remains well above a 10% time regression. It is not a CUDA measurement, capacity/fit proof, learning result or adoption recommendation. No paid launch, pilot, paired evaluation or PR follows.

Numerical proof: six additional whole-game runs pair the unchanged stateless retained collection preparation with the collection fold at all three row limits; proof instrumentation only saves arrays, outside the timed matrix. Actions, rewards, dones, 556 decisions and prefix identities match exactly. Maximum absolute differences across every compared element: logprobs/values **1.4901e-7**, updated parameters/buffers **5.9605e-8**, final gradients **1.4901e-8**. All pass the existing fold tolerances (outputs atol=2e-6/rtol=2e-5; gradients/state atol=2e-5/rtol=2e-4). No tolerance was enlarged. Floating-point batching changes mean GRU output/gradient/state hashes differ from the stateless receipts; byte equality is not claimed. Within the new matrix, each arm/row combination repeats the same hashes exactly. Incumbent/control hashes still match the transport receipts exactly.

Validation: 30 tests passed in the four affected storage/fold files, including open-gate cold/warm, late references, K1/K4, live/padded, coalesced tails, reset/fork/resume lifetimes, parameter invalidation, shuffled/repeated/empty policy rows, all learner gradients and weak ownership even with a warm Prepared still alive. The shared model decision-memory tests passed (108). Ruff and whitespace checks passed. The native-cursor commit already supplied debug nextest and the rebuilt extension; this cut changes no Rust.

Evidence: `experiments/data/opponent-memory/20261117-collection-fold` retains the implemented source, unchanged harness, runners, all 27 receipts/logs, six full-array proof runs and the current plan/design plus pre-cut design chronology. Every archive member is verified against its SHA256/size manifest. The current design closes decision 1 as the Task session taking option b; the two authorized optimization cuts are complete. Matched training/paired evaluation remain unrun pending the Task session’s next decision.

### Collection-fold CUDA calibration — 2026-11-17

Run `opponent_memory/cuda-calibration-fold-2@39100`, source `fc5bc66985`, A40 `mtg252-a40-126c56e020a3`. Five sequential fresh workers on the unchanged refs256 workload, two updates each. Raw and EMA reloads admitted. Incumbent reference is the two-worker median, 312.553 s. Exploratory cost evidence; matched training and paired strength evaluation remain unrun.

| Worker | Decisions | Stage s | Collection s | Learning s | Time / incumbent | Decisions/s | Host tree peak GB | CUDA allocated peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0: control | 105938 | 360.619 | 217.098 | 141.053 | 1.154 | 293.77 | 15.909 | 0.181 |
| 1: incumbent | 105609 | 311.666 | 175.281 | 134.551 | 0.997 | 338.85 | 15.610 | 0.177 |
| 2: control | 106020 | 351.715 | 210.396 | 138.920 | 1.125 | 301.44 | 15.949 | 0.182 |
| 3: gru-k1 | 105784 | 450.861 | 277.395 | 171.109 | 1.443 | 234.63 | 16.320 | 0.252 |
| 4: incumbent | 106559 | 313.439 | 175.739 | 135.628 | 1.003 | 339.97 | 15.609 | 0.177 |

| Phase s | Incumbent median | Pretrained control | GRU-K1 |
|---|---:|---:|---:|
| Retention (includes capture) | 3.792 | 24.630 | 22.546 |
| Memory capture (nested) | 0.000 | 20.263 | 18.383 |
| Native observation encoding | 1.856 | 3.493 | 3.492 |
| Preparation | 55.130 | 65.886 | 102.784 |
| Inference/action selection | 69.365 | 69.727 | 103.863 |
| Learner forward/loss (includes backward/step) | 124.247 | 127.738 | 160.135 |

Launcher estimate USD0.3188911458250000161266749881; rental terminated and the Task session confirmed the pod gone. The original `cuda-calibration-fold` launch was refused for no A40 stock (Task session observation; retained request `512c2c6e3fcd` is `failed`). Request `cdc1c3370199` remains stuck `admitting` after an interrupted `etude queue`; it rented nothing according to the Task session. The stuck request is not a successful run or a new charge. No retry, cancellation, paid launch or job rename was performed here.

Verified ratios: control 1.125×, GRU-K1 1.443× incumbent time, versus 1.143× / 1.548× in the prior optimized CUDA run. Suffix transport reduced native encoding to about 3.5 s, but retention remains about 20 s over incumbent. GRU preparation and inference still each add about 35 s over control. The existing gated J1 layer runs joint and control attention separately; the next cut replaces that unmerged arithmetic with a single-pass gated additive summary. Frozen probe and CUDA bytes keep their original identities.

Evidence: `experiments/data/opponent-memory/20261117-cuda-fold-2`. All 134 outputs (182,529,881 bytes) fetched through SDK digest checks; archive members verified against the retained SHA256/size manifest. Launcher status, attempt, record and both original launch requests retained. The launcher has no run record for the original fold job, so its cost is not independently reported here. No learning, strength or adoption claim.

### Single-pass gated additive join — 2026-11-17

The Task session authorized replacing unmerged J1 token attention after the CUDA
split showed persistent GRU preparation/inference cost. Each trunk layer now adds
`tanh(scale * gate) * projection(mean(summaries))` to each real board input, then
runs self-attention once and one unsplit feedforward block. Both owners receive
the same addition. Padding and the optional value token receive none; the value
token still attends the conditioned board. Projections are bias-free and
initialized nonzero; gates start at zero. K1 remains the pilot, K4 works through
mean pooling. MemoryJoin, host partitions, appended summary slots and second
attention calls are deleted. This changes unmerged model arithmetic; frozen
probe/CUDA bytes and their source identities are unchanged.

Existing whole-game CPU harness: two streams, mixed disclosure, open gates,
one real update, 556 decisions per process. Twenty-seven sequential fresh
processes, three per arm/row limit. The initial repeat overlapped tests and was
replaced in full before these results. Times are medians [min, max], seconds;
ratios use fresh true incumbents at the matched row limit.

| Rows | Arm | Decisions | Collection s | Update s | Total s | Total / incumbent |
|---|---|---:|---:|---:|---:|---:|
| 32 | incumbent | 556 | 1.398 [1.382, 1.437] | 2.573 [2.556, 2.583] | 3.971 [3.965, 3.993] | 1.000 |
| 32 | control | 556 | 1.533 [1.475, 1.536] | 2.579 [2.546, 2.591] | 4.080 [4.054, 4.127] | 1.027 |
| 32 | gru-k1 | 556 | 2.111 [2.073, 2.121] | 2.790 [2.778, 2.810] | 4.899 [4.863, 4.921] | 1.234 |
| 128 | incumbent | 556 | 1.437 [1.412, 1.448] | 2.011 [1.993, 2.027] | 3.439 [3.430, 3.459] | 1.000 |
| 128 | control | 556 | 1.525 [1.497, 1.560] | 2.031 [2.010, 2.056] | 3.556 [3.507, 3.617] | 1.034 |
| 128 | gru-k1 | 556 | 2.125 [2.076, 2.129] | 2.163 [2.124, 2.187] | 4.263 [4.249, 4.291] | 1.240 |
| 256 | incumbent | 556 | 1.395 [1.382, 1.398] | 2.090 [2.051, 2.108] | 3.485 [3.448, 3.490] | 1.000 |
| 256 | control | 556 | 1.511 [1.489, 1.561] | 2.028 [2.015, 2.031] | 3.526 [3.516, 3.592] | 1.012 |
| 256 | gru-k1 | 556 | 2.105 [2.088, 2.155] | 2.159 [2.105, 2.177] | 4.263 [4.260, 4.264] | 1.223 |

Control ratios are **1.027/1.034/1.012×** at 32/128/256 rows; GRU-K1 is
**1.234/1.240/1.223×**, versus the collection-fold cut's 1.529/1.586/1.606×.
GRU update medians fall from 3.825/3.198/3.209 s to 2.790/2.163/2.159 s;
collection falls from 2.239/2.238/2.263 s to 2.111/2.125/2.105 s. Different
arithmetic can change behavior, so this is a cost comparison, not equivalence
with J1 or evidence of equal learnability. The fixture's 556 decisions happen
to remain equal. Incumbent/control rollout, prefix, gradient and updated-state
hashes match the previous cut exactly; each new arm/row combination repeats
its hashes exactly. No CUDA measurement, learning result or adoption follows.

The gate tests retain exact initialized control identity for padded/live,
compact/bucketed, empty/mixed/long rows and masked-mean/value-token critics.
They also check empty history with an open gate, finite first gate gradients,
zero initial projection/encoder gradients, later projection/encoder gradients,
strict shared-trunk restoration and checkpoint output round trips using matched
execution options. A directed attention-input test checks owner-neutral addition,
unchanged padding/value inputs and exactly one attention call. The existing
memory/fold checks passed; the first expanded checkpoint assertion compared
unmatched execution options and was corrected to preserve those options.

Evidence: `experiments/data/opponent-memory/20261117-additive`, exact source,
harness, sequential runner, all 27 receipts/logs and member SHA256/size manifest.
Every archive member verified locally. No paid launch, pilot, job rename or PR.
Retention attribution is the next separate cut.

Validation: memory-model and collection/learner-fold files passed (119 cases);
after matching reload execution options, gate and Prepared-row tests passed
(72 cases). Ruff and whitespace checks passed. Gate/CI retain the broader
backend/recovery and active-training checks; no Rust changed in this cut.

### Retention attribution and integer admission cut — 2026-11-17

A cProfile wrapper around `_capture_memory` in the unchanged whole-game CPU
harness isolates retention without adding game work. The control game has
556 decisions, 396 capture calls and 557 selected rows (including bootstrap).
Only 91 row captures enter chunk partitioning; 100 new Tail nodes are admitted.
Unchanged tail revisions already skip partitioning, hashing and admission.
Before the cut, changed-chunk admission consumes 39.3 ms of 63.9 ms capture;
800 small `np.isin` calls consume 15.1 ms cumulatively. About 6 ms is a cold
NumPy import nested in that membership path, not a separate recurring cost.
The four native seed/seat/generation/cursor methods together consume 0.47 ms;
immutable interning consumes 5.4 ms. Identity/object churn and unchanged-tail
hashing do not justify a new cache in this two-stream profile.

Replaced exactly the contiguous integer-category membership checks with bounds
checks **after** finite/integer admission. Phase, step, chooser, selectors,
reference flags and boolean detail checks retain identical accepted sets and
errors. Noncontiguous catalog membership, words, partitions, temporal continuity,
first-occurrence/reachability, program/mana routes and all other checks remain.
Dense/import/archive paths retain their admission. No capture is skipped and
no coalescing, capacity, identity, sampling or game persistence changes.

Afterward capture is 50.7 ms, admission 25.4 ms, and the 800 equivalent bounds
checks 2.0 ms. The counts and every profile receipt's rollout, prefix, gradient
and updated-state hash match exactly. Profiling includes instrumentation and
cold imports; its timings are attribution, not the unprofiled cost estimate.
Remaining capture includes full changed-chunk admission, immutable interning,
per-row capacity/generation checks and profiler/identity construction. The
256-stream CUDA retention excess is not fully apportioned by this two-stream
CPU profile; new CUDA evidence is needed to measure its reduction.

Unprofiled matrix: 27 sequential fresh processes, three per arm/row limit,
556 decisions each, two whole-game streams, mixed disclosure, open gates and
one real update. All four hashes match the additive-join receipts exactly for
every arm and row limit, with unchanged extension/world identities. Times are
medians [min, max], seconds, against fresh matched-row true incumbents.

| Rows | Arm | Decisions | Collection s | Update s | Total s | Total / incumbent |
|---|---|---:|---:|---:|---:|---:|
| 32 | incumbent | 556 | 1.422 [1.368, 1.425] | 2.559 [2.555, 2.583] | 3.980 [3.927, 4.005] | 1.000 |
| 32 | control | 556 | 1.524 [1.471, 1.525] | 2.569 [2.554, 2.589] | 4.078 [4.040, 4.113] | 1.025 |
| 32 | gru-k1 | 556 | 2.068 [2.056, 2.079] | 2.774 [2.768, 2.810] | 4.853 [4.824, 4.878] | 1.219 |
| 128 | incumbent | 556 | 1.419 [1.359, 1.426] | 2.044 [2.008, 2.046] | 3.435 [3.405, 3.463] | 1.000 |
| 128 | control | 556 | 1.501 [1.463, 1.525] | 2.014 [1.998, 2.029] | 3.516 [3.492, 3.522] | 1.024 |
| 128 | gru-k1 | 556 | 2.099 [2.055, 2.113] | 2.148 [2.133, 2.185] | 4.240 [4.232, 4.261] | 1.234 |
| 256 | incumbent | 556 | 1.414 [1.396, 1.418] | 2.064 [2.051, 2.099] | 3.477 [3.468, 3.496] | 1.000 |
| 256 | control | 556 | 1.506 [1.500, 1.510] | 2.030 [2.010, 2.070] | 3.530 [3.515, 3.579] | 1.015 |
| 256 | gru-k1 | 556 | 2.080 [2.064, 2.116] | 2.106 [2.094, 2.130] | 4.194 [4.187, 4.210] | 1.206 |

Control CPU time ratios are **1.025/1.024/1.015×** at 32/128/256 rows;
GRU-K1 **1.219/1.234/1.206×**, versus 1.234/1.240/1.223× before this cut.
Total ranges overlap at 32/128 rows, and host/update variation is larger than
this small retention saving. No general end-to-end speedup is claimed from
these near-line total ratios. Retention/capture ranges do separate:

| Rows | Arm | Retention before s | Retention after s | Nested capture before s | Nested capture after s |
|---|---|---:|---:|---:|---:|
| 32 | control | 0.0560 [0.0547, 0.0566] | 0.0473 [0.0464, 0.0477] | 0.0381 [0.0377, 0.0388] | 0.0299 [0.0297, 0.0300] |
| 32 | gru-k1 | 0.0576 [0.0568, 0.0580] | 0.0480 [0.0467, 0.0483] | 0.0395 [0.0391, 0.0398] | 0.0305 [0.0292, 0.0306] |
| 128 | control | 0.0562 [0.0557, 0.0580] | 0.0474 [0.0445, 0.0479] | 0.0386 [0.0384, 0.0398] | 0.0298 [0.0278, 0.0304] |
| 128 | gru-k1 | 0.0580 [0.0557, 0.0587] | 0.0490 [0.0475, 0.0492] | 0.0398 [0.0379, 0.0404] | 0.0310 [0.0301, 0.0311] |
| 256 | control | 0.0559 [0.0549, 0.0571] | 0.0469 [0.0459, 0.0472] | 0.0382 [0.0379, 0.0392] | 0.0297 [0.0289, 0.0298] |
| 256 | gru-k1 | 0.0569 [0.0555, 0.0595] | 0.0487 [0.0475, 0.0494] | 0.0388 [0.0377, 0.0409] | 0.0309 [0.0301, 0.0314] |

Capture is nested inside retention; these are not disjoint costs to sum.
Unprofiled capture falls by about 8–9 ms per 556 decisions in both arms;
retention falls by the same amount. The remaining roughly 30 ms capture is
retained work, not proof that no further safe optimization exists.

Validation: prefix/chunk-storage checks passed (34 cases), including 75 corrupt
changed-tail mutations across 15 category fields (below/above bounds, fractions,
NaN and infinity). Existing dense/admission rejection cases passed (72); the
explicit finite-first-gate-gradient check passed (4). Ruff and whitespace checks
passed. No Rust, observation layout or native rebuild changed. Broader gate/CI
checks remain with the caller.

Evidence: `experiments/data/opponent-memory/20261117-retention`: before/after
capture profiles and receipts, original admission source, unchanged whole-game
harness, implemented source, runner, all 27 receipts/logs, derived phase tables
and per-member SHA256/size manifest. Archive members verified locally and
published. No paid launch, pilot, PR, job rename or Task completion. Matched
training and paired strength evaluation remain unrun; this new arithmetic has
no CUDA measurement yet. All three bounded parts are complete.

### Additive-join CUDA calibration — 2026-11-17

Run `opponent_memory/cuda-calibration-additive@39100`, source `cdc0368534`, A40 `mtg252-a40-985cdbcb45ff`. Five sequential fresh workers, unchanged refs256 workload, two updates each; raw and EMA reload admitted. Launcher estimate USD0.3054048066777777718369988078; rental ended. Exploratory cost evidence; matched training and paired strength evaluation remain unrun.

| Worker | Decisions | Stage s | Collection s | Learning s | Time / incumbent | Host tree peak GB | CUDA peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0: control | 104997 | 340.294 | 211.050 | 126.770 | 1.090 | 15.862 | 0.181 |
| 1: incumbent | 106081 | 299.910 | 179.154 | 118.673 | 0.961 | 15.515 | 0.177 |
| 2: control | 104497 | 327.456 | 205.253 | 119.842 | 1.049 | 15.889 | 0.182 |
| 3: gru-k1 | 104769 | 394.889 | 261.324 | 130.985 | 1.265 | 16.356 | 0.243 |
| 4: incumbent | 106698 | 324.253 | 195.985 | 126.080 | 1.039 | 15.570 | 0.177 |

Incumbent median **312.082 s**; its two workers differ by **8.1%**, so ratios carry about ±4% reference spread. Excluding the producer control, control is **1.049×**, GRU **1.265×** incumbent time (previous fold-2: 1.125× / 1.443×). These are stage-time comparisons with different completed-game decision totals, not matched learning or strength evidence.

| Phase seconds | Incumbent 1 | Control branch | GRU-K1 | Incumbent 2 |
|---|---:|---:|---:|---:|
| Retention | 3.926 | 20.755 | 20.628 | 4.342 |
| Preparation | 59.281 | 69.329 | 111.950 | 64.284 |
| Inference/action selection | 73.264 | 73.129 | 86.655 | 81.427 |
| Learner forward/loss | 107.597 | 108.328 | 119.297 | 114.119 |
| Native observation encoding | 1.546 | 2.816 | 2.703 | 1.690 |

The additive join removes most of the earlier learner/inference excess. GRU preparation exceeds control by 42.621 s over 4,025 calls (10.589 ms/call); both log arms retain about 17 s more than incumbents. Memory capture is nested within retention, and backward/optimizer tracks are nested within forward/loss; do not sum nested tracks. No new paid launch or pilot follows in this slice.

All **134 outputs / 182,750,320 bytes** fetched through public SDK digest checks and archive members verified by size/SHA256. Evidence: `experiments/data/opponent-memory/20261117-cuda-additive`, including output archive, member manifest and launcher/attempt/run records. Frozen output bytes and source identities remain unchanged.

### Many-stream preparation cut — 2026-11-17

Whole-game CPU harness: 32 streams, one native worker, one Torch thread, one whole game per stream, mixed disclosure, open gates and one real damped-policy update. One sequential fresh process per arm/execution-row limit before and after the cut. These are bounded exploratory measurements, with no timing ranges or statistical speedup claim. Gates, sampling groups, capacities and model shapes are unchanged.

| Rows | Arm | Decisions | Preparation before/after s | Retention before/after s | Nested capture before/after s | Collection s | Update s | Total s | Total / incumbent | Decisions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | incumbent | 6858 | 0.9090 / 0.9137 | 0.0548 / 0.0540 | 0.0000 / 0.0000 | 5.560 | 27.249 | 32.808 | 1.000 | 209.03 |
| 32 | control | 6503 | 0.9869 / 1.0040 | 0.3854 / 0.3955 | 0.3279 / 0.3365 | 5.631 | 25.986 | 31.617 | 0.964 | 205.68 |
| 32 | gru-k1 | 6514 | 2.0480 / 1.8330 | 0.3958 / 0.4067 | 0.3317 / 0.3416 | 8.354 | 28.507 | 36.861 | 1.124 | 176.72 |
| 128 | incumbent | 6858 | 0.8997 / 0.9055 | 0.0529 / 0.0549 | 0.0000 / 0.0000 | 5.435 | 22.081 | 27.516 | 1.000 | 249.23 |
| 128 | control | 6503 | 0.9862 / 0.9994 | 0.3871 / 0.3954 | 0.3292 / 0.3365 | 5.610 | 20.923 | 26.533 | 0.964 | 245.09 |
| 128 | gru-k1 | 6514 | 2.0883 / 1.8192 | 0.4018 / 0.4032 | 0.3370 / 0.3390 | 8.322 | 22.442 | 30.763 | 1.118 | 211.74 |
| 256 | incumbent | 6858 | 0.9299 / 0.9059 | 0.0544 / 0.0538 | 0.0000 / 0.0000 | 5.427 | 21.692 | 27.118 | 1.000 | 252.89 |
| 256 | control | 6503 | 0.9825 / 1.0024 | 0.3907 / 0.3905 | 0.3333 / 0.3317 | 5.557 | 20.550 | 26.107 | 0.963 | 249.09 |
| 256 | gru-k1 | 6514 | 2.0943 / 1.8253 | 0.4017 / 0.3998 | 0.3372 / 0.3350 | 8.302 | 22.091 | 30.393 | 1.121 | 214.33 |

Capture is nested in retention. Each arm retains its original completed-game decision total (incumbent 6,858; control 6,503; GRU 6,514), so stage-time ratios and decisions/s are both shown. All nine cases preserve **exact rollout, prefix, final-gradient and updated-state hashes** against the prior implementation. Existing stateless/fold tolerances remain unchanged. CPU timings do not establish CUDA cost, matched learning or strength. No paid launch, pilot, PR, calibration-job rename or Task completion follows.

Attribution: the 32-stream cProfile comparison has 445 GRU preparation calls
versus 309 control calls (their acting games differ). Before the cut, GRU
preparation is 3.183 s profiled; fold assembly is 1.046 s, including 0.715 s
in chunk-feature lookup/build and 0.776 s in 1,844 dense-featurization calls
(including canonical empties). These nested costs must not be summed. Cache
binding/root lookup are small; model cache updates occur in forward, outside
this track. Already cached complete nodes were **not** having raw features
rebuilt or transferred by the old code; the avoidable work was repeated
ancestry/routing plus many small new-fragment featurization calls. Cumulative
root reference joins now cut off cached ancestry entirely, unused cached record
embeddings are deleted, and new fragments share one host arithmetic batch.
No cached complete-node feature is admitted, assembled, pinned or transferred.

Afterward fold assembly is 0.686 s; dense-featurization calls fall 1,844 → 723
and their cumulative time 0.776 → 0.429 s. Profiled preparation is 2.811 s.
Unprofiled GRU preparation at 32/128/256 rows falls
2.048/2.088/2.094 → 1.833/1.819/1.825 s. Required open-tail features and ordinary
policy preparation remain; this does not eliminate the whole GRU preparation
premium. CPU transfer bookkeeping remains about 0.35 s, and CPU cannot attribute
CUDA pin/copy or synchronization costs. The A40's +10.589 ms/call is an observed
aggregate, **not** a directly profiled Python attribution; no claim that this
CPU saving removes all 42.621 CUDA seconds is supported without the next rental.

The new collection root equals stateless encoding at the existing tolerances,
including mixed lifetimes, late references, coalescing, reset, fork and parameter
updates. The narrow collection/learner/chunk checks passed (32 cases); Ruff and
whitespace checks passed. Evidence archive:
`experiments/data/opponent-memory/20261117-many-stream-preparation`, with source,
runner, before/after profiles, all 18 unprofiled receipts/logs and member digests.

### Many-stream retention cut — 2026-11-17

Whole-game CPU harness: 32 streams, one native worker, one Torch thread, one whole game per stream, mixed disclosure, open gates and one real damped-policy update. One sequential fresh process per arm/execution-row limit before and after the cut. These are bounded exploratory measurements, with no timing ranges or statistical speedup claim. Gates, sampling groups, capacities and model shapes are unchanged.

| Rows | Arm | Decisions | Preparation before/after s | Retention before/after s | Nested capture before/after s | Collection s | Update s | Total s | Total / incumbent | Decisions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | incumbent | 6858 | 0.9137 / 0.9121 | 0.0540 / 0.0544 | 0.0000 / 0.0000 | 5.500 | 26.989 | 32.490 | 1.000 | 211.08 |
| 32 | control | 6503 | 1.0040 / 0.8041 | 0.3955 / 0.3498 | 0.3365 / 0.2952 | 4.948 | 25.454 | 30.402 | 0.936 | 213.90 |
| 32 | gru-k1 | 6514 | 1.8330 / 1.6045 | 0.4067 / 0.3716 | 0.3416 / 0.3102 | 7.753 | 27.814 | 35.567 | 1.095 | 183.15 |
| 128 | incumbent | 6858 | 0.9055 / 0.8930 | 0.0549 / 0.0534 | 0.0000 / 0.0000 | 5.336 | 21.928 | 27.263 | 1.000 | 251.55 |
| 128 | control | 6503 | 0.9994 / 0.8501 | 0.3954 / 0.3658 | 0.3365 / 0.3083 | 5.265 | 20.322 | 25.587 | 0.938 | 254.16 |
| 128 | gru-k1 | 6514 | 1.8192 / 1.6233 | 0.4032 / 0.3808 | 0.3390 / 0.3179 | 7.965 | 22.054 | 30.019 | 1.101 | 217.00 |
| 256 | incumbent | 6858 | 0.9059 / 0.9074 | 0.0538 / 0.0533 | 0.0000 / 0.0000 | 5.402 | 21.347 | 26.749 | 1.000 | 256.38 |
| 256 | control | 6503 | 1.0024 / 0.8264 | 0.3905 / 0.3557 | 0.3317 / 0.2996 | 5.139 | 20.241 | 25.379 | 0.949 | 256.23 |
| 256 | gru-k1 | 6514 | 1.8253 / 1.5961 | 0.3998 / 0.3728 | 0.3350 / 0.3112 | 7.741 | 21.322 | 29.063 | 1.087 | 224.13 |

Capture is nested in retention. Each arm retains its original completed-game decision total (incumbent 6,858; control 6,503; GRU 6,514), so stage-time ratios and decisions/s are both shown. All nine cases preserve **exact rollout, prefix, final-gradient and updated-state hashes** against the prior implementation. Existing stateless/fold tolerances remain unchanged. CPU timings do not establish CUDA cost, matched learning or strength. No paid launch, pilot, PR, calibration-job rename or Task completion follows.

256-stream collection-only profiles (one worker/thread, execution rows 32):

| Implementation | Arm | Decisions | Preparation s | Retention s | Nested capture s | Collection s | Scalar capture calls |
|---|---|---:|---:|---:|---:|---:|---:|
| before | control | 53658 | 9.516 | 4.075 | 3.601 | 49.117 | 53659 |
| before | gru-k1 | 54650 | 19.919 | 4.251 | 3.743 | 72.389 | 54651 |
| preparation | control | 53658 | 9.511 | 4.138 | 3.664 | 48.594 | 53659 |
| preparation | gru-k1 | 54650 | 16.133 | 4.199 | 3.722 | 68.179 | 54651 |
| retention | control | 53658 | 9.036 | 3.629 | 3.166 | 46.607 | 10135 |
| retention | gru-k1 | 54650 | 15.470 | 3.665 | 3.201 | 66.144 | 10418 |

Both tracks are profiled, so these are attribution timings, not unprofiled throughput. All six runs keep exact rollout/prefix hashes and initial state hashes; **there is no learner update in these profiles**. The separate 32-stream matrix proves final gradients and updated state. These 256-stream profile variants freeze only the fold/root implementation and capture entry method; the imported batched vocabulary admission is common to all six. They isolate fold/capture cost rather than replaying every historical source line.

Receipt description erratum: the 32-stream matrix inherited an old "Two-stream" limitation string. Executed source sets 32 streams. Both 256-stream before receipts inherited it too; executed source sets 256 streams and omits learning. Values and hashes were not rewritten; later profile descriptions are corrected.

The final retention path compares a numeric validity mask plus native generation,
seed identity, counts and tail revision for all selected streams before scalar
capture. An unchanged stream contributes its previous Prefix object. Changed
streams reuse their game/viewer identity, batch transport shape/capacity checks,
and retain full immutable-chunk admission. Vocabulary, definition and own-seat
admission batch the new chunks across rows; temporal, partition, reachability,
integer and range checks stay on each exact chunk/predecessor. Numeric presence
bits avoid object-array equality calling Python for every unchanged row; the
regression test rejects any such Prefix comparison. Metadata bytes are counted.

A final cache review also found and repaired a lifetime edge case: rejecting a
newer root could expose a predecessor cached for a different generation through
the shared root map. Final root selection checks each reachable boundary against
its selected prefix's identity/epoch. A mixed-cache regression proves rejection;
normal root walks still stop at the cached boundary. The final GRU update cases
and 256-stream profile were rerun with this guard and the numeric presence mask.

At 256 streams, calls average 26.0 control / 27.2 GRU rows, close to the CUDA
calibration's roughly 26 rows/call. Before the fold cut, GRU's host fold assembly
is 7.836 s, including 5.593 s in chunk-feature lookup/build; dense featurization
runs 12,978 times (5.717 s). Root lookup/binding is only 0.296/0.315 s, and CPU
conversion/bookkeeping is 2.044 s. Model cache updates run in forward, outside
preparation. Final fold assembly is 4.483 s, dense calls 3,815 (2.613 s), root
lookup/binding 0.155/0.195 s, and CPU conversion 1.709 s. These are nested
attributions. Cached complete nodes provide a state/reference handle without
raw features; required open tails and ordinary policy inputs still need work.
The profiled CPU preparation premium per call over control falls about
5.28 → 3.30 ms. This attributes the avoidable host work, not every millisecond
of the A40's observed +10.589 ms/call; CUDA pin/copy/synchronization remain
unmeasured for this source.

Final GRU capture still performs **10,966 immutable-node admissions**, exactly
as before, while scalar row entries fall 54,651 → 10,418. Its 3.201 s capture
includes 2.204 s in full chunk admission. Control likewise keeps its 10,649
node admissions while scalar entries fall 53,659 → 10,135. The avoided work is
unchanged-row Python and repeated identities, not fewer admitted changed facts.
This is a modest capture saving, not removal of the entire CUDA retention excess.

The unprofiled 32-stream matrix reports GRU stage-time ratios
1.095/1.101/1.087× incumbent at 32/128/256 execution rows. Because completed-game
counts differ by arm, decision-rate ratios are **0.868/0.863/0.874×**, so the
roughly 10% throughput-cost target is not established. Single cases and changing
host timing do not support a general speedup claim. Collection in that matrix
has at most 32 rows; the wider limits exercise learning. The 256-stream profiles
supply the larger collection scale, with no learner update.

Validation: storage (15), cache/collection (27) checks passed; the strengthened
mixed-lifetime and no-Python-comparison regressions passed, with collection
checks repeated after the lifetime guard. Ruff and whitespace checks passed.
All nine final update cases retain exact rollout, prefix, gradient and updated
state hashes. All six large collection cases retain exact rollout/prefix hashes.
No Rust, model shape, capacity, coalescing, sample grouping, logical loss
normalizer, prefetch lifetime or optimizer-step count changed.

Evidence: `experiments/data/opponent-memory/20261117-many-stream-retention`.
The size/SHA256-verified archive includes before/after receipts, all six large
profiles and receipts, source, runners, current notes and the description
erratum. Frozen CUDA bytes/source identities and the calibration job name remain
unchanged. No paid launch, pilot, PR or Task completion. The Task session owns
CUDA measurement of these host cuts; matched training and paired strength remain
unrun.

### CUDA suffix calibration — 2026-11-17

Run `opponent_memory/cuda-calibration-suffix@39100`, source `5c644ef19e`, A40 `mtg252-a40-36729973e845`. All 134 outputs fetched through SDK digest checks and archived with size/SHA256 verification. One preceding launch was refused for no A40 stock. Launcher estimate USD0.2933; rental terminated and pod gone (Task session observation). The launch used `--max-usd 3.80`, with USD2.68 committed beforehand. No launch in this slice.

| Worker | Seconds | / incumbent | Retention s | Preparation s | Inference s | Forward/loss s | Native encoding s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00-control | 332.295 | 1.089 | 18.369 | 53.672 | 63.669 | 130.172 | 3.770 |
| 01-incumbent | 305.775 | 1.002 | 3.628 | 48.564 | 63.530 | 128.694 | 1.985 |
| 02-control | 337.066 | 1.105 | 20.182 | 58.725 | 66.567 | 130.481 | 3.778 |
| 03-gru-k1 | 366.312 | 1.201 | 18.478 | 75.165 | 72.080 | 139.669 | 3.730 |
| 04-incumbent | 304.331 | 0.998 | 4.221 | 50.922 | 64.886 | 125.529 | 2.023 |

The incumbent median is 305.053 s; its two measurements agree within 0.5%. Control is 1.105× and GRU 1.201× incumbent time. The previous additive run was 1.049×/1.265× with about 8% incumbent spread. These are exploratory cost receipts, not learning or strength evidence. Capture is nested under retention, never added as another cost. Repeated open-tail work remains a candidate; unchanged-stream skipping alone cannot remove it in opponent-only self-play, where nearly every viewer log changes at nearly every step.

Evidence: `experiments/data/opponent-memory/20261117-cuda-optimized-retry/suffix-output.tar.gz` and adjacent member/launcher records. Matched training and paired strength comparison remain unrun. No paid launch, pilot, PR, model-shape change or calibration-job rename follows.

### History off against main — 2026-11-17

Native decision capture is now opt-in. Training composition roots enable it only for positive observation record capacity or include-own. A disabled game keeps empty logs and returns before capture work; Game search clones never capture, and transactional rollback preserves the setting. Native scalar/vector resets and vector replacement/recovery retain it. New disabled diagnostic inputs serialize `decision_memory: false`; older omitted settings retain their historical enabled replay meaning. The census/probe/demo explicitly opt in, and the calibration job name is unchanged.

CPU comparison uses actual main source `3afd92d8f3bf48b87d873693580f0d5f8ccf3612`, separately built for CPython 3.13. Both sides use history off, 32 complete-game streams, one worker/Torch thread, identical seeded width-64 model and mixed deck disclosure, one real update, and logical minibatches of 256. Execution-row limits are unset on both: main predates that API, and bounded history execution is unnecessary with history disabled. Cases run branch/main/main/branch in fresh processes. Imported source paths and extension hashes are in every receipt.

| Source | Order | Decisions | Collection s | Update s | Total s | Decisions/s |
|---|---:|---:|---:|---:|---:|---:|
| branch | 0 | 6858 | 4.586 | 20.122 | 24.708 | 277.56 |
| branch | 3 | 6858 | 4.676 | 20.065 | 24.741 | 277.19 |
| main | 1 | 6858 | 4.574 | 20.051 | 24.625 | 278.50 |
| main | 2 | 6858 | 4.499 | 20.108 | 24.607 | 278.70 |

Median total branch/main is **1.0044×** (24.725/24.616 s). All 6,858 decisions, rollout outputs, final gradients and updated model/buffer hashes are exact across all four cases. This bounded CPU fixture supports approximately unchanged history-off cost; two cases per side are not a confidence bound or a CUDA claim.


Release engine benchmarks use the committed `managym/perf/baseline.json` reference `1ad670d8fdbb763ec28f1392f8dbdb623a5ccaa3`, built and measured on this same local host. The standard sample runs branch/reference/reference/branch; actual main is also built and run twice. No comparison uses the historical CI wall time.

| Source | Steps/s | Search total s | Search worst s |
|---|---:|---:|---:|
| branch | 36724.3 | 52.715 | 1.449 |
| baseline reference | 35583.0 | 55.082 | 1.510 |
| actual main median | 35649.8 | 54.045 | 1.489 |

Branch/main step-duration ratio is 0.9707×; search-total ratio is 0.9754×. Engine Budget passes against the committed same-host reference with unchanged step/search/score digests; the baseline file is not moved. Raw per-seed rows and scheduling samples remain in the archive.


Validation: 12 targeted native debug checks; the 63 memory/storage/collection/learner/execution cases (the demo was corrected for search forks retaining history without appending); 30 recording/collection/probe checks; five interruption and game-error recovery cases. Final cache snapshot/lifetime checks and artifact tracking validation are retained with the evidence. No CUDA launch, pilot, PR or Task completion. Matched learning and paired strength remain unrun.

Evidence: `experiments/data/opponent-memory/20261117-history-off-main`. The comparison preserves main and benchmark-reference source revisions, build/measurement commands, source diffs, raw receipts, arrays, benchmark binaries/digests and test logs.

### Per-record collection cache — 2026-11-17

Bounded exploratory CPU harness: 32 whole-game streams, one worker and Torch thread, execution rows 32, mixed disclosure, open gates and one real damped-policy update. One fresh process per arm and implementation. The 256-stream runs use the same collection-row limit, profile preparation, and omit learning. Both implementations use the same rebuilt native engine; the baseline restores the prior host/collection fold. The shared store keeps full changed-chunk admission. No paid launch, pilot, PR, model-shape change or calibration-job rename.


32-stream collection and update:

| Implementation | Arm | Decisions | Retention s | Preparation s | Inference s | Collection s | Update s | Total / incumbent | Decisions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | incumbent | 6858 | 0.051 | 0.827 | 3.130 | 4.818 | 25.929 | 1.000 | 223.04 |
| before | control | 6503 | 0.362 | 0.825 | 2.906 | 5.029 | 24.879 | 0.973 | 217.44 |
| before | gru-k1 | 6514 | 0.378 | 1.610 | 4.666 | 7.710 | 27.174 | 1.135 | 186.73 |
| after | incumbent | 6858 | 0.052 | 0.835 | 3.139 | 4.840 | 25.843 | 1.000 | 223.51 |
| after | control | 6503 | 0.354 | 0.806 | 2.832 | 4.889 | 24.511 | 0.958 | 221.19 |
| after | gru-k1 | 6514 | 0.355 | 1.620 | 3.443 | 6.371 | 27.289 | 1.097 | 193.52 |

256-stream collection-only preparation profiles:

| Implementation | Arm | Decisions | Retention s | Preparation s | Inference s | Collection s | Update s | Total / incumbent | Decisions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | incumbent | 55010 | 0.353 | 7.738 | 23.714 | 37.713 | 0.000 | 1.000 | 1458.65 |
| before | control | 53658 | 2.592 | 8.671 | 23.593 | 42.259 | 0.000 | 1.121 | 1269.73 |
| before | gru-k1 | 54650 | 2.640 | 14.691 | 35.805 | 60.575 | 0.000 | 1.606 | 902.18 |
| after | incumbent | 55010 | 0.358 | 7.809 | 23.974 | 38.151 | 0.000 | 1.000 | 1441.89 |
| after | control | 53658 | 2.661 | 8.775 | 23.816 | 42.718 | 0.000 | 1.120 | 1256.10 |
| after | gru-k1 | 54650 | 2.757 | 18.584 | 27.268 | 56.253 | 0.000 | 1.474 | 971.50 |

Raw prefix digests and sampled actions/rewards/dones remain exact. On the 32-stream GRU array proof, maximum absolute differences are rollout 2.38e-07, final gradients 1.79e-05, updated state 5.28e-06; every array passes the existing fold tolerances (rollout 2e-6/2e-5, gradient/state 2e-5/2e-4 absolute/relative). GRU floating hashes are not relabelled as identical. Control/incumbent retain all four hashes.

This is the permitted store fallback: raw chunk hashing, changed-chunk admission and archive/recovery identities stay unchanged. Append-only serving segments cache host features per record/version; collection caches states through the penultimate record and recomputes the last record to preserve coalescing. The learner retains its chunk fold and logical loss semantics. Cumulative reference joins share storage when no new snapshot appears. Interned raw chunks can be shared across streams, so growing segments belong to the native stream, not the chunk. A whole-game counterexample exposed that ownership error; the regression covers identical starts followed by divergent continuations.

An initial per-record recurrence used one packed call per record and raised profiled preparation from 14.970 to 21.367 s while lowering inference 36.421 to 28.023 s at 256 streams. The final implementation packs each unbranched suffix once and retains its intermediate outputs. Those initial receipts are retained as diagnostic evidence, not substituted for the final table.

Retention includes capture; these are nested costs. This fallback does not claim to reduce repeated raw admission or hashing. Arms finish different decision totals, so stage-time ratios alone are not throughput comparisons. Each case is a single observation, with no confidence range. Profiled CPU results do not measure CUDA pin/copy/synchronization, matched learning or playing strength. Receipt limitation strings inherited a prior two-stream description; executed source and decision counts establish 32/256 streams, and the 256-stream runs contain no learner update.

Evidence: `experiments/data/opponent-memory/20261117-record-cache`. Source, runners, raw receipts, profiles, arrays and member hashes are retained. The Task session owns any further CUDA measurement and the still-unrun matched training/paired strength comparison.

The separate 256-stream numeric proof retains all 54,650 sampled actions, rewards and dones exactly. Maximum absolute logprob/value differences are 2.38e-7/3.58e-7, within the existing 2e-6 absolute / 2e-5 relative fold tolerances. Its additional baseline replay is a numeric proof only; the original baseline timing/profile receipts were restored and remain the table inputs.

At 256 streams, retained raw bytes stay 193.44 MB and feature caches fall 484.23 → 212.87 MB, while Python metadata rises 20.26 → 84.64 MB for 84,866 record views. At 32 streams after the update, feature caches rise 61.58 → 87.08 MB because learner chunk fragments and collection record fragments coexist. These measured owners include raw view backings and host feature caches; detached neural cache state is scope-bound and is not included in these store counts. The existing incident/state limits are unchanged.

### Shared record preparation through learning — 2026-11-17

The final review found a remaining duplicate: collection used cached record fragments, but learner chunk preparation still independently featurized each partial raw snapshot. The surviving featurizer now accepts records only. Learner chunks assemble those same vocabulary-bound fragments, with bulk offsets for detail/reference/member joins; the neural learner fold, sampling, denominators and optimizer boundary remain unchanged. Unprepared bootstrap prefixes first resolve their stream segment, so only appended records require new host features. Immutable raw content keys can share record views, but never a growing stream segment. Raw admission, hashing and recovery stay on the permitted store fallback.

A regression warms collection, requires no nonempty featurization during learner preparation, then appends an unprepared bootstrap record and requires exactly one new featurized row. Fifty-five focused chunk/learner/collection/execution checks pass, including gradient, coalescing, mixed lifetimes and collector recovery.

CPU harness is unchanged: 32 complete-game streams, execution rows 32, one worker/Torch thread, mixed disclosure, open gates and one real update. Baseline is `394496a4e8` host preparation on the shared rebuilt engine. One fresh process per case; no confidence range. The final GRU case follows the bulk join-offset cut; its initial scalar-assembly draft is retained separately as diagnostic evidence.

| Implementation | Arm | Retention s | Preparation s | Inference s | Collection s | Update s | Total s | Decisions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| before | incumbent | 0.051 | 0.821 | 3.143 | 4.803 | 25.909 | 30.712 | 223.30 |
| before | control | 0.354 | 0.813 | 2.849 | 4.929 | 24.822 | 29.751 | 218.58 |
| before | gru-k1 | 0.362 | 1.608 | 3.533 | 6.492 | 27.680 | 34.172 | 190.63 |
| after | incumbent | 0.051 | 0.826 | 3.109 | 4.792 | 25.895 | 30.687 | 223.48 |
| after | control | 0.344 | 0.799 | 2.815 | 4.850 | 24.512 | 29.361 | 221.48 |
| after | gru-k1 | 0.346 | 1.597 | 3.416 | 6.351 | 26.795 | 33.146 | 196.52 |

All four hashes are exact in every whole-game pair. Completed decisions remain incumbent 6,858, control 6,503 and GRU 6,514. The final GRU total is 1.080× incumbent time and its decision rate is 0.879× incumbent; the unequal totals and single cases do not establish a general throughput claim.

Final 256-stream collection-only profiles (no learning, one case per arm):

| Arm | Decisions | Retention s | Preparation s | Inference s | Collection s |
|---|---:|---:|---:|---:|---:|
| incumbent | 55010 | 0.373 | 8.109 | 24.815 | 39.746 |
| control | 53658 | 2.744 | 8.927 | 24.077 | 43.629 |
| gru-k1 | 54650 | 2.777 | 18.439 | 27.492 | 56.520 |

Against the prior record-cache collection, all 54,650 GRU sampled actions/rewards/dones and raw prefix digests remain exact. Floating hashes differ: maximum logprob/value differences are 2.38e-07/2.68e-07, within the existing 2e-6 absolute / 2e-5 relative fold tolerances. The extra numeric-only replay is not substituted for timing receipts. Capture remains nested in retention. Receipt limitation strings inherit the previously documented two-stream wording; executed source uses 32/256 streams and the latter omits learning.

This closes per-record host preparation across collection and learning. Changed raw chunks still undergo whole-chunk admission and hashing. Rust, model shape, capacities, coalescing and the calibration-job name are unchanged. No paid launch, pilot, PR or Task completion follows. History-off/main and native benchmark evidence above remain applicable; CUDA cost and peak neural-cache memory for these cuts, matched training and paired strength remain unmeasured.

Evidence: `experiments/data/opponent-memory/20261117-record-assembly`; raw receipts/profiles, numeric proof, runner, final source/diff, baseline host source and test logs are size/SHA256-verified and published. Earlier evidence is unchanged.

### Trunk overflow and capacity revision — 2026-11-17

Launch of `decision-memory-trunk` at `93a2ebf3ec` (quote USD2.21 planned per
trunk, job budget USD7.20): seeds 39101 and 39103 **failed on a memory-table
capacity error** under the strict rule, at 791,535 and 928,014 learner
decisions (`decision_reference_details 16626 > 16384` and `17880 > 16384`);
seed 39102 was lost to a provider capacity refusal at USD0.00. Launcher
estimates USD0.32 and USD0.28; no pods remain. Learned games outgrew the
heuristic census (maximum 14,787 details) within a quarter of the trunk.

Prospective revision for every arm, the Task session's choice: capacities
**1,024 / 65,536 / 512 / 32,768** (records / details / references / members),
four times the detail bound and twice the others. Strict failure stays. The
failed runs keep the label `decision-memory-trunk` as incomplete evidence; the
relaunch is `decision-memory-trunk-2`. Throughput at the larger layout is
unmeasured; the first trunk serves as its admission.

### Evaluation admission and wall budget — 2026-11-18

The first evaluation pass (`decision-memory-elemental-v1-f824428377ca`,
revision `ca4d688099`) played no game: the world-bound pilot checkpoints
refused authored decks that were not submitted to the world. The evaluation
matchup now submits them (`89041ebb47`). The next pass
(`decision-memory-elemental-v1-f8ff34d95eee`) played 112 of 1400 matches on
an A40 with 28 single-threaded workers, then failed when one game reached the
600 s wall budget (median game 86 s, p90 238 s, max 521 s across 240 retained
games). Workers outnumbered the machine's cores about 3.5 to 1. Evaluation
now runs 30 workers on a 32-vCPU CPU machine (`410fe064d9`, run
`decision-memory-elemental-v1-ac7e7ee21101`).

### First evaluation pass, gates and the update clock — 2026-11-18

Run `decision-memory-elemental-v1-ac7e7ee21101` (revision `410fe064d9`)
played 500 matches on final checkpoints. Scores are the GRU arm's, with a
binomial standard error that treats the four legs of a deal as independent.

| Seating | Decklists | Games | Score |
| --- | --- | ---: | ---: |
| GRU v control, seed 39103 | open | 800 | 0.547 ± 0.018 |
| GRU v control, seed 39103 | closed | 800 | 0.542 ± 0.018 |
| GRU v control, seed 39101 | open | 32 | 0.656 ± 0.084 |
| GRU v control, seed 39101 | closed | 20 | 0.550 ± 0.111 |
| Intact v blanked memory, seed 39101 | open | 200 | 0.480 ± 0.035 |
| Intact v blanked memory, seed 39101 | closed | 196 | 0.490 ± 0.036 |
| Intact v blanked memory, seed 39103 | open | 200 | 0.490 ± 0.035 |
| Intact v blanked memory, seed 39103 | closed | 200 | 0.530 ± 0.035 |

Memory gates `tanh(2a)` of the two attention layers, absolute value, read
from the stored GRU checkpoints (all start at 0):

| Seed | Update 40 | Update 80 | Update 120 | Final |
| --- | --- | --- | --- | --- |
| 39101 | 0.053 / 0.062 | 0.067 / 0.068 | 0.079 / 0.074 | 0.084 / 0.078 |
| 39102 | 0.042 / 0.054 | 0.052 / 0.065 | 0.065 / 0.072 | 0.069 / 0.074 |
| 39103 | 0.035 / 0.048 | 0.040 / 0.059 | 0.051 / 0.067 | 0.059 / 0.069 |

**The arms are not matched in learner decisions at equal updates.** An update
is not a fixed number of learner decisions in this recipe, and the cause of
the difference between arms is not yet identified.

| Seed | Arm | Update 40 | Update 80 | Update 120 | Final (updates) |
| --- | --- | ---: | ---: | ---: | ---: |
| 39101 | control | 2.64M | 6.72M | — | 7.58M (89) |
| 39101 | GRU | 2.27M | 4.54M | 6.66M | 8.17M (150) |
| 39102 | control | | | — | 7.09M (86) |
| 39102 | GRU | 2.24M | 4.64M | 7.07M | 7.97M (136) |
| 39103 | control | | | — | 8.04M (114) |
| 39103 | GRU | 1.86M | 4.27M | 6.63M | 8.00M (143) |

No control branch reached update 120, and every branch stopped on its wall
cap before 8.8M decisions. The update-40 and update-80 head-to-heads compare
unequal training, and seed 39103's final head-to-head compares equal
decisions with 143 against 114 updates. The gate and blanked-memory readouts
are within one checkpoint and do not depend on this.

**Update size follows game length.** The recipe closes a batch on complete
games: each of 256 streams collects at least 64 learner rows and then
finishes its game. Final decisions per update were 85k, 82k and 71k for the
control branches against 54k, 59k and 56k for the GRU branches (seeds 39101
to 39103), and 67k for the trunk. Control games therefore ran longer in
learner decisions in all three seeds. Whether that is learned behaviour or
an accounting difference between the arms is not established.

**Blanking barely moves decisions.** On 16 local self-play games per seed
and disclosure with each final GRU checkpoint (about 3,000 to 3,900
decisions per cell), blanking the memory changed the action distribution by
a mean total variation of 0.019 to 0.025 and the most likely action at 1.4%
to 2.3% of decisions. The same checkpoint differed from its previous stored
checkpoint by 0.036 to 0.050.

**The arms learn different policies; decision accounting is the same.**
Learner decisions equal environment decisions in every run. Decisions per
game from the training metrics, by update range:

| Seed | Arm | 0–20 | 20–40 | 40–60 | 60–80 |
| --- | --- | ---: | ---: | ---: | ---: |
| 39101 | control | 200 | 309 | 398 | 385 |
| 39101 | GRU | 194 | 238 | 224 | 198 |
| 39102 | control | 209 | 287 | 373 | 377 |
| 39102 | GRU | 208 | 217 | 228 | 216 |
| 39103 | control | 182 | 217 | 308 | 331 |
| 39103 | GRU | 174 | 173 | 232 | 214 |

Policy entropy over updates 40–80 was 0.22, 0.21 and 0.24 for control
against 0.34, 0.40 and 0.32 for GRU. Thirty local self-play games per final
checkpoint, on identical decks and seeds, reproduce the difference and place
it within turns:

| Seed | Arm | Decisions per game | Turns per game | Decisions per turn |
| --- | --- | ---: | ---: | ---: |
| 39101 | control | 302 ± 38 | 21.7 | 13.9 |
| 39101 | GRU | 186 ± 20 | 20.8 | 8.9 |
| 39102 | control | 405 ± 47 | 24.7 | 16.4 |
| 39102 | GRU | 194 ± 17 | 21.7 | 8.9 |
| 39103 | control | 278 ± 24 | 24.1 | 11.5 |
| 39103 | GRU | 226 ± 22 | 20.7 | 10.9 |

Both arms clip the whole model's gradient to norm 0.267 and the clip is
active in nearly every update. Over updates 20–40 the GRU arm's pre-clip
norm was 2.10 against control's 0.81 (seed 39101) and 1.06 against 0.41
(39102), but 1.22 against 1.54 (39103). Shared clipping is a candidate
cause of the slower sharpening and is not established.

### Head-to-head and reliance, all three seeds — 2026-11-18

Runs `decision-memory-elemental-v1-ac7e7ee21101` and `-a5d1642838e8`
(revision `410fe064d9`) completed every declared head-to-head and reliance
match. Scores are the GRU arm's; each deal's four legs are averaged first and
the standard error is across deals. This supersedes the first-pass table.

| Seed | Decklists | Update 40 (50 deals) | Update 80 (50 deals) | Final (200 deals) | Intact v blanked, final (50 deals) |
| --- | --- | ---: | ---: | ---: | ---: |
| 39101 | open | 0.560 ± 0.030 | 0.540 ± 0.034 | 0.619 ± 0.016 | 0.480 ± 0.025 |
| 39101 | closed | 0.560 ± 0.032 | 0.560 ± 0.030 | 0.588 ± 0.014 | 0.490 ± 0.032 |
| 39102 | open | 0.490 ± 0.029 | 0.560 ± 0.035 | 0.562 ± 0.015 | 0.495 ± 0.033 |
| 39102 | closed | 0.540 ± 0.034 | 0.490 ± 0.030 | 0.554 ± 0.015 | 0.485 ± 0.031 |
| 39103 | open | 0.555 ± 0.026 | 0.505 ± 0.035 | 0.547 ± 0.015 | 0.490 ± 0.036 |
| 39103 | closed | 0.535 ± 0.026 | 0.530 ± 0.028 | 0.542 ± 0.014 | 0.530 ± 0.023 |

The GRU arm's final checkpoint beat control's in all three seeds and both
disclosures, while blanking its memory cost nothing. The final checkpoints
differ in learner decisions for seeds 39101 (8.17M against 7.58M) and 39102
(7.97M against 7.09M) and are equal for 39103 (8.00M against 8.04M); update
counts differ in all three. The advantage is therefore not attributable to
memory use at play time.

### Final checkpoints against the heuristic — 2026-11-18

Run `decision-memory-elemental-v1-784fb252dbbe` (revision `5197b1c9e7`)
played each arm's final checkpoint against the native heuristic, 50 paired
deals per cell. Scores are the checkpoint's, with the standard error across
deals.

| Seed | Decklists | Control | GRU |
| --- | --- | ---: | ---: |
| 39101 | open | 0.295 ± 0.031 | 0.400 ± 0.029 |
| 39101 | closed | 0.325 ± 0.027 | 0.480 ± 0.028 |
| 39102 | open | 0.385 ± 0.036 | 0.455 ± 0.033 |
| 39102 | closed | 0.365 ± 0.030 | 0.380 ± 0.032 |
| 39103 | open | 0.420 ± 0.031 | 0.460 ± 0.036 |
| 39103 | closed | 0.410 ± 0.036 | 0.410 ± 0.036 |

Every checkpoint loses to the heuristic. The GRU arm scores at least as
well as control in all six cells (means 0.431 against 0.367); at seed 39103,
where the final checkpoints have equal learner decisions, the difference is
within one standard error.

**Shared gradient clipping does not explain the divergence.** For each GRU
checkpoint, six local self-play games gave 970 to 2,830 decisions, and a
stand-in loss (final result as the return, policy-gradient plus value
terms) was backpropagated in batches of 256. The memory pathway's gradient
norm was 0.31 to 2.68 against 1.18 to 2.62 for the rest of the model, so
the rest holds 70% to 97% of the total norm (median about 84%). A shared
clip therefore shrinks the rest of the model's gradient by 3% to 30%, and
Adam largely normalizes a uniform rescale. The cause of the GRU arm's slower
sharpening is open; the candidates left are the memory term added to every
board token and its interaction with the damped update.

### Heuristic curve along training — 2026-11-18

Run `decision-memory-elemental-v1-ead9d86f4363` (revision `3da9fb0686`)
played stored checkpoints against the heuristic, 25 paired deals per
disclosure. Cells pool open and closed (50 deals; finals 100 deals) with the
standard error across deals. Update 0 is the trunk export both arms start
from.

| Seed | Arm | Update 0 | Update 40 | Update 80 | Update 120 | Final |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 39101 | control | 0.105 ± 0.023 | 0.225 ± 0.030 | 0.325 ± 0.038 | — | 0.310 ± 0.020 |
| 39101 | GRU | | 0.240 ± 0.027 | 0.390 ± 0.034 | 0.410 ± 0.032 | 0.440 ± 0.020 |
| 39102 | control | 0.095 ± 0.023 | 0.280 ± 0.027 | 0.365 ± 0.032 | — | 0.375 ± 0.024 |
| 39102 | GRU | | 0.315 ± 0.031 | 0.435 ± 0.037 | 0.405 ± 0.030 | 0.417 ± 0.023 |
| 39103 | control | 0.095 ± 0.023 | 0.265 ± 0.028 | 0.330 ± 0.033 | — | 0.415 ± 0.024 |
| 39103 | GRU | | 0.260 ± 0.027 | 0.390 ± 0.035 | 0.410 ± 0.037 | 0.435 ± 0.025 |

Both arms improve throughout; control does not decline as its policy
sharpens. The trunk export scores about 0.10, so the branches did most of
the learning. At equal updates the arms are close, with the GRU arm ahead
at update 80 in all three seeds; control reached update 80 after 6.7M
decisions (seed 39101) where the GRU arm needed 4.5M, because control's
games hold more decisions.

### History off against main on the A40 — 2026-11-18

Run `opponent_memory/cuda-history-off@39100` (revision `e4aca24cbd`,
USD 0.32) built main's base commit `3afd92d8f3` beside this branch on one
A40 and ran `scripts/measure_history_off.py compare`: history off, 256
whole-game streams, four collection workers, bf16, the seeded width-64
model, three collection-and-update iterations per fresh process with the
first discarded. Both sources played the same 107,048 timed decisions.

| Order | Source | Collection s | Update s | Decisions/s |
|---:|---|---:|---:|---:|
| 0 | branch | 198.8 | 118.4 | 337.5 |
| 1 | main | 191.2 | 122.0 | 341.8 |
| 2 | main | 166.0 | 118.6 | 376.2 |
| 3 | branch | 157.5 | 122.0 | 383.1 |

Branch over main in seconds per decision is **0.998×** (medians of two
cases each); update time alone is 0.9995×. Collection time drifted by about
20% over the rental on both sources, which the branch/main/main/branch order
balances; two cases per side give no confidence bound. With history off the
branch costs the same as main on the training GPU, to within this
measurement.

### Hand information in the memory, larger corpus — 2026-11-18

Before training an arm from a hand-fitted memory encoder, the fit itself was
checked. 240 heuristic games at the pilot's memory capacities gave 7,634
labelled decisions (5,338 fit, 1,144 validation, 1,152 test; no capacity
losses). For each seed, the GRU branch's start checkpoint kept its trunk
frozen while the memory encoder and a linear head were fitted to
opponent-hand presence for four epochs; the baseline is the same head on the
public state vector alone. Test log loss per card, game-weighted, at the
best validation epoch:

| Readout | Open | Closed |
| --- | ---: | ---: |
| public state only | 0.0765 | 0.0937 |
| + memory, seed 39101 | 0.0820 | 0.0913 |
| + memory, seed 39102 | 0.0779 | 0.0899 |
| + memory, seed 39103 | 0.0805 | 0.0888 |

The memory adds 0.002 to 0.005 with closed decklists and nothing with open
ones. The 2026-11-16 probe's 0.016 closed gain came from 1,812 decisions,
where the state-only head scored 0.138; with three times the data that
baseline reaches 0.094 and the gain mostly disappears. On heuristic games
the opponent's decision history carries almost no hand information beyond
the public state, so a hand-prediction auxiliary loss has little to teach
the encoder. Limits: scripted opponents, one fitting recipe, no interval on
the differences. No arm was trained from these encoders.

The same fit on 240 games that seed 39101's final GRU checkpoint played
against itself (7,651 decisions, sampled actions) gave test log loss 0.0824
open and 0.1031 closed from the public state, and 0.0862 and 0.0970 with
memory: a 0.006 closed gain, none open. The result is not specific to the
scripted heuristic, but this checkpoint still loses to that heuristic, so it
says nothing about what a strong policy's history would carry.
