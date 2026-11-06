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
