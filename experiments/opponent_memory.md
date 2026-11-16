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

A supervised probe comes first, then three paid arms across three independent
seed labels, shared across arms for blocking: nine branch runs from three
no-history trunks, not nine independent seeds. The seed, not
an evaluation game, is the unit of learning evidence. No factorial cross and no
dedicated control-deck training arm in this pilot; stratify the existing
Elemental matchups instead.

### Supervised probe (planned, not executed)

Do summaries of the opponent's past decisions say anything about the opponent's
hand that the current public state does not? The probe answers that on a
laptop, at no cost, and fixes the summary count K for the paid arms:

```bash
uv run python -m experiments.opponent_memory probe --out .runs/decision-memory-probe
```

It prints one table of held-out per-card log loss, each memory row also as its
difference from the state-only row with a game-bootstrap interval, separately
for open and closed decklists (values appear when it runs):

```text
candidate          open                     closed
analytic           <loss>                   <loss>
state only         <loss>                   <loss>
transformer K=1    <loss>  <diff [lo, hi]>  <loss>  <diff [lo, hi]>
transformer K=4    ...
gru K=1            ...
gru K=4            ...
selected K: <1|4> (<reason>)
games <n>  rows <n>  capacity overflows <n>
```

and writes the same content to `summary.json` in the output directory. Use a
fresh directory.

- **Games.** The native heuristic plays both seats over the Elemental 10-deck
  roster, uniform ordered pairs, half with open and half with closed decklists,
  forced decisions skipped. No trained policy can read this layout before the
  pilot's own trunks exist, and none may be rebound to it. Games split 70/15/15
  into fit, validation and test by game seed, stratified by disclosure in fixed
  40-game seed blocks (14/3/3 per mode); no game appears in two splits and
  appending games never moves one to another split.
- **Rows.** Each acting decision of each viewer with a non-empty log, at most
  16 per viewer per game. Capacities are the census bounds; a game exceeding
  them is a typed capacity error and is counted, never truncated.
- **Label.** Which card definitions the opponent holds, from the privileged
  native hand snapshot. The recorder stores labels beside the observation,
  never in an observation key, a prepared policy input, a cache or a replay.
- **Candidates.** The production memory encoder and the embeddings it reads,
  fitted from scratch for transformer and GRU at K=1 and K=4, each with a linear
  head over its summaries joined to a public state vector (opponent hand and
  library size, turn, per-definition public-zone counts, decklist counts when
  open). Baselines on the same rows: that state vector alone through the same
  head, and the analytic hand size × unseen-deck frequency. Three fitting seeds.
- **Fitting.** Adam at 0.003, batches of 16 decisions, eight epochs, retaining
  the epoch with the lowest game-weighted validation loss. Training averages
  per-card Bernoulli log loss over decisions; readouts give each game equal
  weight and average the three fitting seeds. The test split is scored after
  K selection. `--epochs` and `--max-games` allow predeclared smaller runs;
  defaults are eight and 120, with the 20-game smoke sizing the actual count.
- **Choosing K.** K=4 only if its validation loss beats K=1, averaged over both
  encoders, with an interval excluding zero; otherwise the cheaper on CPU, then
  K=1. The test split is read once, for the table.
- **Cost.** Local CPU, no rental. The game count is sized from a 20-game timing
  smoke so that all fits finish within two hours.

Limits: these are scripted games, not learned play, so the probe can mislead K;
hand presence is one use of memory among several; a weak probe does not show
that reinforcement learning cannot use memory. Repeating the probe on the
trunks' own self-play games is an optional check inside the budget and does not
change the chosen K.

| Arm | Changes from the incumbent |
| --- | --- |
| No history | Incumbent; no memory tokens or recent events |
| Transformer, K summaries | Two pre-norm bidirectional record-transformer layers |
| GRU, K summaries | Single-layer GRU over the full prefix, same record width |

Jack Heart selected transformer and GRU with extra board tokens (J1), then
"Probe first; choose K, then run three arms", which replaces the five-arm pilot;
see `scratch/questions.md` for the exact direction. The experiment module lists
the four probe candidates and the control but stops before job construction
until resource sizing, CUDA admission and calibrated job wiring are ready.
Opponent-only records are the measured workload; the default-false
`decision_memory_include_own` option adds no arm. The implementation plan is
`scratch/opponent-memory-design.md`.

Torch only; MLX support is outside this task. Assumption: memory width 64,
four attention heads, feedforward width 128;
keep the current small Elemental board model, optimizer, precision and learning
rule fixed across arms. Use the baseline's 64-wide two-layer board attention,
masked-mean value and FiLM in every arm, so J1 also reaches objectless actions.
Override sampled_self_play's per-wall fitted schedule with one calibrated
learner-decision schedule across arms and disable capacity recovery; long-game
overflow must remain a visible failure. Keep parameter counts visible: these are
design alternatives with different costs, not parameter-matched architecture claims.
Use the true history-disabled incumbent as the throughput reference. A control
that retains the log isolates encoder/join cost, but cannot establish the total
cost of adding memory. Report native capture, encoding and host transport
separately; the shared engine still captures decisions with history input disabled.

### Common-trunk initialization (planned, unimplemented)

Each seed label trains one no-history trunk on the memory-enabled observation
space, then branches three arms from its checkpoint: no history, transformer-K
and GRU-K. The arms do not start as unrelated randomly initialized agents. The
three seed labels 39101–39103 provide independent learning units; the three
branches within each label form a blocked comparison. Older-world checkpoints
cannot be rebound or admitted by bypassing world/shape checks.

A memory arm starts as its trunk exactly: the
[gated join](../docs/observation-and-model.md#starting-from-a-no-history-trunk)
blends each board layer's board-only and joint attention outputs by
`tanh(scale * gate)`, with the gates at zero. Zero-valued summaries alone are
not inert attention tokens, and a mask would be exact but could never learn.
Before the pilot, tests show that

- an arm built from a trunk gives the trunk's logits and value with zero
  tolerance on padded, live, compact and empty-history inputs, for both
  encoders and K=1 and 4;
- at initialization the gates have non-zero gradient and every memory-encoder
  gradient is exactly zero, and after one real damped update the encoder
  gradients are non-zero;
- a checkpoint round trip keeps the gates.

The gate scale is fixed once for all arms from the calibrated horizon, so that
a gate pushed consistently in one direction can reach 0.5 within the first 10%
of a run. Each layer's gate value is a logged training metric. If the gates stay
near zero, the pilot compared three copies of the control and says nothing
about memory; report that rather than the curves.

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
2. **Probe, then calibration at most USD 3.** Run the probe locally and date the
   chosen K here. Then, on one authorized GPU host, interleave real
   no-history repeats with the selected memory arms using retained replay inputs and
   short production-path updates. Include initialization/checkpoint reload and
   native encoding, preparation, collection inference, learner and peak memory.
   No artificial load. Separate replayed phase timings from whole-update rates.
   Record host CPU/quota, GPU, source/world/input and precision. Use these readings
   to choose one common learner-decision horizon, trunk length, gate scale and
   milestone spacing that fit the training allowance; date that choice here
   before training. If three arms
   cannot fit meaningful learning, report that rather than covertly narrowing the
   comparison or making a learning claim from smoke updates.
3. **Learning pilot, at most USD 27.** Twelve runs: one no-history trunk for
   each independent seed label 39101–39103, then three branches from each. The
   per-run cap, including startup and finalization, is set at calibration so
   that all twelve fit the allowance. Inspect
   PR #547 / Task 4b0591 before freezing: it merged on 2026-11-16 as
   `c035d5566a4a6546f212deabb67524c47bc8a7d6`. Sync its game-level
   `decklists_open` input and mixed per-game disclosure, plus #550's native
   library/hand knowledge, before training. Use a balanced open/closed mix
   identically across arms (`UniformDecks(decklists="mixed")`).
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

If affordable from the same allocation, repeat the supervised probe on the
trunks' own self-play games, with the same recorder, split, candidates and
baselines. No additional rental solely for it, and no change to the chosen K.
Mark this check deferred if collection and fitting do not fit the cap.

USD 8 remains for failures/retries and unresolved startup cost, inside the same
USD 50 total, not an additional budget. The task already authorizes this amount.

### Disclosure and input comparisons to consider

Jack Heart's 2026-11-16 direction was to consider the remaining #547/#550
experiments in this series after both landed. The history layout must settle
first, and these share the same USD50 total. The core probe, K selection and
three-arm pilot retain priority.

- **B1:** compare the mixed-training control with open-only and closed-only
  training, evaluating each in both disclosure modes. This would inform whether
  one player should serve both modes and the default training disclosure mix.
- **B2:** mask one admitted viewer-safe input at a time at Prepared on the full
  layout: initial main/sideboard counts, decklist rows, `known_library` facts,
  and `decklists_open`. Preserve native observations, legality and layout
  identity. A null result can reflect redundant information retained in history.

Both allocations are deferred until CUDA calibration prices the primary pilot;
no extra arms or ablation implementation are added speculatively. Report which
fit the remaining allowance and which remain deferred. Native knowledge stays
sound single-card facts; inference beyond those facts belongs to the model.
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

Shared viewer snapshots now replace repeated reference programs: the padded
memory payload falls from 33,628,160 to 645,120 bytes per viewer (98.1%) at
capacities 1,024 records, 8,192 details, 256 snapshots and 16,384 members. With
table-wide admission, CPU log-control collection is **1.44×** its history-disabled
incumbent median (2.203 s versus 1.528 s per 250 decisions), so the roughly 10%
guide is not met on CPU; see the
[dated entry](#shared-snapshots-and-table-wide-admission-2026-11-16).

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
Collection, frozen-opponent, bootstrap and learner execution now use explicit
2/2/2 row limits with output and gradient parity. The independent CPU cohort's
27.33 GiB host peak covers its recorded histories and one logical minibatch;
it does not admit an on-policy CUDA worker. Measure separate host RSS and CUDA
peaks, worker/update and checkpoint reload before job wiring and horizon
calibration. Preserve every record and typed overflow.
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
