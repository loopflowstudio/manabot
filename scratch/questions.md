# c5fba4 open questions and assumptions — 2026-11-05

The previous contents described MTG-243 (naming), inherited from the parent
commit; they remain at `8e5d152ab:scratch/questions.md`.

Material assumptions made headless by the first implement step (reversible
until launch):

- **Larger size = width 256, 4 layers, 8 heads (15× compute parameters).** The
  brief's 290k base count does not match the current v9 count (648k excluding
  the five 4096-bucket tables); the ratio is computed on the current code. A
  larger choice (e.g. width 384, ~34×) would cut the large size's updates on
  USD38; reconsider after a first per-update cost reading.
- **Input v9 with 512 facts** rather than v10's v7/128: v1–v7 now reject layer
  payloads that TLA cards can produce, and v9 carries naming identity.
- **Same learning rate for both sizes.** Larger transformers often want lower
  rates; 1e-4 max is already conservative, and a change would confound scale.
- **Equal wall-clock budget, not equal updates.** Equal-update comparisons are
  the shared milestones; equal compute is the endpoint head-to-head.
- **16 pairs (64 games) per comparison**, twice v10's, for ±~0.09 intervals.
- **cpu32 for evaluation passes**, as in MTG-186 and the learning-signal study;
  1.75 h keeps a pass inside USD2 at USD0.96/hour.
- **600 s per game**, five times MTG-213's limit, sized from one-observation
  latency on this Mac rather than a rented-CPU measurement.
- **No execution in this step.** Launching paid training needs the protocol
  reviewed and frozen at a merged revision.
- **Calibration length: five updates in a 3,600 s wall.** Direction said "a few
  updates". A size slower than 2,400 s/update records no update; its failure or
  empty reading is still the answer. Timing comes from a fresh policy, so late-run
  games may cost more.
- **The USD12 evaluation cap stays an operator check,** as directed; Etude has no
  cross-pass ceiling here.
- **Launch gate: 1,500 s/update for w256** (second concept review). A rough line,
  chosen to leave three head-to-head milestones in 54 hours. Reversible until
  launch. The opposite reading stays with the launch decision: if calibration
  shows w256 costs little more than w64, because collection dominates, a larger
  size inside 10–100× is worth a new protocol version before scoring.
- **Failed legs bound a reading rather than void it** (second concept review).
  The analysis rule changes before any scoring; the strict `complete` flag stays.
- **Export every 25 updates; milestones 75 and 150 added** (third concept review).
  Both are reversible until launch. A cadence of 50 would halve exports and double
  the loss bound; 25 is the choice because a gate-speed w256 loses 10.4 h per gap.
- **First-64 cutoffs are reported, not an issue** (fourth concept review). A nonzero
  evaluation count leaves `complete` unchanged, because overflow is accepted
  behavior. Reversible until launch.
- **Checkpoint tag rather than an Etude release for naming a size** (fifth concept
  review). `TrainingRun.job_name` is the better owner, but a release would hold
  this PR. Reversible: `size_run` is the only reader.
- **Pre-scoring host screen: 1.3×, updates 2–6, one replacement per size** (fifth
  concept review). Rough lines from Elemental's bands; TLA has no anchor yet.
  Reversible until launch. Without the screen, host class is reported only.
- **SIGTERM ends every `TrainingJob` as `interrupted`** (2026-11-06, found implementing
  direction 17). Wider than this experiment, but the handler sits at the worker's entry
  and its only effect is recording an end. Reversible.
- **A pod past update 25 is kept, not replaced** (2026-11-06). The simpler choice over
  teaching `size_run` a replacement order, which `TrainingRun` has no start time to
  support. Reversible until launch.
- **Evaluation rehearsal in calibration, gated at 300 s** (sixth concept review). Up to USD2
  of the reserve; the 300 s line is half the limit and rough. Reversible until launch.
- **`startup_seconds` 900 on every rented on-demand** (sixth concept review). Taken from
  `l4` and `07c490-cpu32`. The launcher setting stays an operator step at launch.
- **Rehearsal gate wider than "no killed game"** (2026-11-06). All 128 games stored and valid, so a
  truncated or missing game also fails it. Reversible until launch.
- **The 600 s limit's 0.4 s rationale may be high** (2026-11-06). A real fresh w256 match here took
  about 0.05 s a decision. The limit stays; the rehearsal's rented reading decides.
- **Launcher status closes a lost run, in manabot** (seventh concept review). Etude's `observe`
  is the better owner, but a release would move the control pin and hold this PR. Reversible:
  one helper. Its unknown-run error means analysis and `etude report` need launcher access.
- **`C5FBA4_PASS_ATTEMPT` names a relaunched pass** (seventh concept review). The same trade,
  against a fresh id from `advance`. Reversible until launch; absent, names are unchanged.
- **Equal-hours head-to-head after a loss, declared now and built only if needed** (eighth concept
  review). Implied by "matched updates/compute" once a loss is about as likely as not; the survivor's
  last export at or before the lost run's hours. Reversible until launch.

## Design revision (2026-11-06, headless, after design review)

- **Four producers, no one-producer calibration arm.** "Two arms only" read as
  covering calibration too, so producer value is not measured on a rental; the
  software's acceptance check proves equal draws, and calibration reads the step
  phase. Reversible before freeze: add a w64 one-producer pod (about USD0.30).
- **Horizon S = min(10,000, ⌊194,400 / s⌋)**, s = calibration's median s/update
  over updates 2–5, no slowdown allowance. The end of the curve is flat, so a run
  reaching 55% of S ends at a rate near 3.8e-5 instead of 2.0e-5.
- **Update cap raised to 10,000** so the wall binds and the scale reading is not
  lost to the cap at four producers; milestones extend to 6400 (1000 and 2000
  dropped).
- **Equal-hours pairs: w64's last checkpoint at or before w160l8's hours**,
  scheduled in `eval` from an Etude `Checkpoint.wall_ms` (an Etude PR, per
  etude-is-ours) rather than a manabot `hours` clock on checkpoints.
- **Evaluation USD14, reserve USD10** to cover about 45 comparisons with the
  equal-hours series; host-screen cancellations no longer draw on the reserve.
- **Defaults land in recipe builders, not Pydantic field defaults**, so saved
  configurations omitting `workers` or `iteration_schedule` keep their meaning.
- **Scale stays the endpoint head-to-head when both finish at the wall**; the
  equal-hours curve replaces direction 25's post-loss pair.
- **Protocol prose tests** now expect two `etude launch` blocks and five rented
  commands, since the `--size` relaunch block is cut; the code paths they
  describe are deleted in the software change.

## Design revision for steer 82d2041b (2026-11-06, headless)

- **"Compute parameters" after consolidation excludes embedding lookup tables**,
  so the recount stays comparable with the brief's definition. If shared token
  embeddings make that line unclear, report both counts. A recount outside
  10–100× returns the size choice to review. Reversible before calibration.
- **Policy input stays v9.** Consolidation is read as a model-architecture change,
  not a new observation version; if 30df0c's merge moves the version, the
  protocol takes what lands and the freeze pins it.
- **The 25% rolling game-error stop stays** beside `recover_capacity`. Aborting
  one game does not remove the run-level stop; a run that hits it ends at its
  last export and is reported like any ended run.
- **No calibration gate on overflow counts.** Calibration reports them per label;
  fresh-policy counts say little about trained games.

## Implement pass (2026-11-06, headless)

- **Calibration ceilings USD0.45 / 1.10 / 0.45** (w64, w160l8, w64 one producer),
  keeping USD2 total with the accepted third pod. Each small pod buys setup plus
  five 300 s updates at USD0.67/hour; the large pod's 1.2 h binds first.
  Reversible before calibration.
- **Provisional horizons 1,296 (w64) and 129 (w160l8)** in the regime and protocol
  JSON until calibration writes the measured ones. Five calibration updates do
  not feel the difference.
- **The large arm's horizon lives in the protocol's `large.expected_updates`**,
  beside its width, depth and heads; w64's lives in its regime file. One owner each.
- **The Etude `Checkpoint.wall_ms` PR is not opened from this step**: it is a
  separate repository's PR and release. Equal-hours scheduling, its analysis rows
  and an hours curve wait for it; nothing was built around its absence.
- **`sampled_self_play` sets the game-error threshold to .25**, v10's, not the
  field default .1. Reversible; a recipe choice, not a field default.
