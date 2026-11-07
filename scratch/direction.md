# c5fba4 direction

Task comment `694f4190`, 2026-11-05 (Jack Heart). It supplements the brief and
was posted after the Task's launch, so the first `implement` step's context
did not include it.

- One seed per model size.
- Prioritize regular evaluations during training, not only at the end. At each
  milestone, play the two sizes against each other (at matched updates/compute)
  and each against the heuristic, so progress is visible as it goes.
- Evaluation stays within the USD100 and a fraction of training cost, on rented
  CPU.

## Graph (Jack Heart, 2026-11-06)

This Task belongs on the `experiment` graph, the Wave default, not `pursue`.
Auto-Ship was stopped and the `pursue` run ended after its ninth concept
review. Protocol v1 and the software on this branch enter
`experiment-design-review` as the proposal: they were drafted headless and have
not been reviewed. Their "closed" decisions stay open for that review. Any
paid launch waits for `experiment-execution-review`.

## Design review decisions (Jack Heart, 2026-11-06)

- **Head-to-heads at equal training hours.** Pair each w256 milestone with the
  w64 export at the same training hours (`TrainingPoint.hours`), throughout
  training, alongside the equal-update games while both have them.
- **Multiple workers.** One producer is wrong for this run. It is forced by code,
  not tuned: deck-source (per-game roster) sampling and `elemental-uniform`
  accept only `execution.workers` 1 (`collection.py`, `regime.py`). Make
  roster sampling work with 2–4 producers and use them here.
- **Defaults track the frontier.** Each new frontier training run on a new card
  set updates the default training recipe so it is as close as possible to
  optimal for USD10–100 runs. This run's software change should leave the
  defaults it establishes (workers, damping horizon) as defaults, without
  rebinding frozen regimes or experiment identities.
- **Ataraxos-style damping, fitted to the expected run length.** Keep the damped
  policy rule, but map its schedule onto the updates each size is expected to
  reach (`DampedPolicyLearning.iteration_schedule`, `IterationSchedule.steps`),
  rather than the absolute iteration schedule, which a ~100–1,300-update run
  barely advances. Expected updates come from calibration. **Required change**
  (Jack Heart: "make sure to change"): both frontier regimes declare
  `iteration_schedule.steps` fitted to their expected updates, and the default
  recipe for new frontier runs fits its horizon to the run's expected length
  instead of the fixed 10,000-update reference. The protocol must not freeze
  without it.
- **Size stays 15×** (w256: width 256, 4 layers, 8 heads).
- **Simplify the protocol.** Cut the host-speed screen and the lost-rental
  paths the protocol describes but does not build. A lost run stops at its last
  export and is reported. Calibration and the evaluation rehearsal stay.
- **Merge b015d5's PR #435 first.** Jack Heart asked for it to merge; it was red
  on Integration (core) at 2026-11-06 17:49 UTC and b015d5 is fixing it (steer
  `a0fb32a0`). Build this Task's software on main after #435 lands, and freeze
  only after it.
- **Order of steps.** Software merges → calibration (own experiment, at most
  USD2 of the reserve) → each size's damping horizon set from it → freeze →
  execution review → training launch.
- **Large arm is deeper, not just wider: w160, 8 layers, 8 heads.** Supersedes
  "Size stays 15× (w256)". On the current code (synced main) compute parameters,
  excluding the 4096-row hashed tables, are w64 L2 386k, w256 L4 8.66M (22.5×,
  not the protocol's stale 15×) and w160 L8 h8 5.54M (14.4×). Recount in the
  protocol from current code. Calibration reads the deep arm's per-update cost; if
  it misses the speed gate, fall back to w256 L4 in a new version before any
  scoring. Two arms only. Size labels follow the shape (e.g. `w160l8`).
- **Abort overflowing games, keep the labels (Jack Heart, MTG-252, 2026-11-06).**
  An engine error and a capacity overflow (e.g. `continuation relations 65 > 64`,
  which killed MTG-252's `value_token` seed 25202 on Elemental) both abort the
  one game and training continues; they stay labelled separately
  (`engine_error` vs `resource_limit`). This regime keeps
  `game_errors.recover_capacity: true`. Under "defaults track the frontier",
  fresh training recipes should default to recovering capacity overflows too,
  without changing frozen configurations. Report the per-label counts.
- **Input consolidation is its own PR: Task 30df0c (Jack Heart, 2026-11-06).**
  Fact and effect ids embed through the card encoder's exact token vocabulary,
  and sources join the card table, replacing the six hashed 4096-row tables.
  Like PR #435, it lands before this protocol freezes, and both arms use the
  consolidated architecture. Recount parameters on main after it merges.
- **Permanents per seat raised to 96 (Jack Heart, 2026-11-06).** From 64 in
  both arms (`max_permanents_per_player`), derived from the selected
  `ObservationSpace` like every capacity. Slots add no parameters; calibration
  reads their per-update cost. Overflows past 96 still abort the game under
  `recover_capacity` and are counted.

## Design accepted (Jack Heart, 2026-11-06)

The revised design is accepted with its four recorded assumptions and the
one-producer calibration pod (about USD0.30), so calibration measures whether
four producers help.

- **Validate catalog fact embeddings before long runs.** PR #437 (Task 30df0c)
  makes catalog ids the default for fresh v8/v9 models but measures wiring and
  parameter counts only. Before either 54-hour run launches, a short matched
  check compares catalog against hashed ids: same size (w64), recipe, seed,
  decks and updates, differing only in `ModelConfig.record_vocabulary`. It must
  show catalog learning at least as well as hashed by a criterion fixed before
  it runs (learning curves and a small vs-initialization / head-to-head read).
  A failure returns the embedding choice to review before any long run. It runs
  inside this Task's reserve on rented hardware, coordinated with 30df0c, which
  owns the software and runs no paid compute.
- **PR #437 readiness for the gate (30df0c; read at head bab2e80d6, since moved).** Fresh regimes can
  select `record_vocabulary: hashed` (pinned by
  `test_fresh_regimes_can_select_hashed_ids`). Rerun the coverage sweep on the
  frozen revision before calibration:
  `uv run python -m manabot.verify.record_coverage --games 200 --seed 0`
  (seed 0: 36,202 decisions, 333/333 cards, every id mapped). Its seed 1 ran
  about 25× slower per decision on very large boards: a game-time and
  per-update cost risk calibration and the rehearsal should read, not a
  vocabulary issue.
- **No manabot workaround for checkpoint wall time.** Equal-hours pairing needs
  Etude's `Checkpoint.wall_ms`. Gaps in Etude are fixed with Etude PRs (we
  write Etude), not manabot workarounds such as reconstructing hours on the
  manabot side. Leave equal-hours pairing marked as waiting on that Etude
  change; filed as Etude Task 3d4f40 (not started; running it is Jack Heart's call).
- **Fact-contribution reading (Jack Heart, 2026-11-07).** Neither the hashed
  nor the catalog fact embeddings has learning evidence: no experiment ever
  measured them, and their output layers start at zero. Add a descriptive,
  no-training reading over retained checkpoints (vocabulary-check arms and each
  long run's milestones): on a fixed sample of positions from the frozen pairs,
  the norm of the fact and of the shared-effect contribution added to each
  object, relative to that object's pre-fact embedding, by fact kind. Report it
  beside learning; it gates nothing. Declare it in the protocol before freeze and
  implement it in this PR (#439). A reading near zero after training is a
  finding that neither design is being used.
