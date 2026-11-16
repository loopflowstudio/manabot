Exploratory: determine whether a best-guess whole-game opponent-memory design is practical to train without materially hurting throughput or early learning, within USD 50 total.

No runs have launched. This is a pilot for choosing the next software basis,
not a claim about playing strength or long-run learning equivalence. A result
to be cited as a finding needs its own prospective run. The experiment module
currently stops launch and evaluation explicitly: the shared log, encoders and
matched-decision evaluation integration must ship first as an ordinary PR from
this checkout. This report and module are the design, not delivered machinery.

## Question and design

The incumbent is the current no-history policy. The best guess is a small
transformer over the opponent's whole-game decision records, with four learned
summary tokens joining the board transformer's token stream. Compare throughput
and learning at matched learner decisions. Do not require a strength gain to
recommend retaining memory as the next basis, but do not interpret a noisy short
run as proof of non-inferiority. Report the size and uncertainty of regressions;
the roughly 10% throughput cost mentioned in the task is context for that
judgment, not a precomputed verdict.

Assumption: use six arms with three independent training seeds per arm, sharing
seed labels across arms for blocking. The seed, not an evaluation game, is the
unit of learning evidence. No factorial cross and no dedicated control-deck
training arm in this pilot; stratify the existing Elemental matchups instead.

| Arm | Changes from best guess |
| --- | --- |
| no-history | Incumbent; no memory tokens or recent events |
| recent-events | Existing pooled 32-row recent-event encoder, no decision memory |
| transformer-k4 | Best guess: two record-transformer layers, four summary tokens |
| gru-k4 | Single-layer GRU, same record width and four summary tokens |
| transformer-k1 | One summary token instead of four |
| transformer-k4-no-passes | Filter eligible passes out of the same native log |

Torch only; MLX support is outside this task. Assumption: memory width 64,
four attention heads, feedforward width 128;
keep the current small Elemental board model, optimizer, precision and learning
rule fixed across arms. Initialize fresh models; checkpoints from before the
layout change cannot be rebound. Keep parameter counts visible: these are design
alternatives with different costs, not parameter-matched architecture claims.
Run all arms on one new input layout and retain all log records even for the
control; measure the common log/encoding overhead against pre-change replay
receipts separately, so a shared input cost is not hidden by the control.

## Shared software before a paid run

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

## Smallest execution sequence

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
   no-history repeats with the five other arms using retained replay inputs and
   short production-path updates. Include initialization/checkpoint reload and
   native encoding, preparation, collection inference, learner and peak memory.
   No artificial load. Separate replayed phase timings from whole-update rates.
   Record host CPU/quota, GPU, source/world/input and precision. Use these readings
   to choose one common learner-decision horizon and milestone spacing that fit
   the training allowance; date that choice here before training. If six arms
   cannot fit meaningful learning, report that rather than covertly narrowing the
   comparison or making a learning claim from smoke updates.
3. **Learning pilot, at most USD 27.** Eighteen fresh runs, seeds 39101–39103,
   six arms, at most USD 1.50 each including startup and finalization. Inspect
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
   The recent-event control is included as a history arm in these comparisons.
   At the endpoint also play each history checkpoint against a copy of itself
   with history blanked before preparation, on the same paired deals and both
   disclosures. Preserve shapes and all current board/legality information;
   blank only that arm's memory input (recent events for the pooled arm). Tag
   these `reliance`, separately from primary `control` games. This is a test of
   reliance under an intervention, not evidence that memory improves learning.
   There are 18,000 primary games and 6,000 endpoint reliance games at three
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

## Readout and adoption

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

## Run ledger

No commit/run ID, launcher cost or measured result yet. The memory-software PR,
replay corpus identity, census, calibrated horizon, concrete launch wiring and
evaluation jobs are outstanding. No paid run is authorized by this design file
beyond the task's existing inclusive USD 50 authority.

Design validation (2026-11-15): Ruff check/format and syntax checks passed;
Etude discovery matched the exact inventory using the locked SDK commit
`d60236d015a4b2a01055b2c861856f06bb1852a2`, and the explicit pre-software launch
stop was exercised. Full `tests/eval/test_experiment_files.py` and report
construction await the development environment/native extension. An initial
isolated import failed at missing `managym._managym`; a discovery attempt using
the older PyPI SDK missed existing experiment files. Neither is experiment data.
