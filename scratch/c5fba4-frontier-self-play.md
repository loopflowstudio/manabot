# c5fba4 — frontier self-play on TLA trophy decks at two sizes

## Accepted design (2026-11-06) — current

The design review (scratch/direction.md, steers d52dc416, a3318c8a, 82d2041b)
returned protocol v1; Jack Heart accepted the revised design the same day (steer
ff3c999d) with the one-producer calibration pod and a catalog-vs-hashed gate. It
is `experiments/protocols/tla-frontier-c5fba4-v1.md`, marked "accepted design, not
frozen". The JSON, regime and runner follow it except where "Software" says
otherwise.

- Arms: w64 (64/2/4) and w160l8 (160/8/8). Recounted with #437 merged at
  563ae37dd, embedding lookup tables excluded: 307,010 and 5,341,922 compute
  params, 17.4× (totals 412,482 / 5,605,602, 13.6×). The pre-consolidation
  counts (385,538 / 5,538,242, 14.4×) are superseded; the test pins the new ones.
- 96 permanents per seat (from 64), both arms; calibration reads the cost.
- `recover_capacity` stays on: engine errors and overflows abort one game,
  labelled `engine_error` / `resource_limit`; analysis and calibration report
  per-label counts from `TrainingRun.game_errors.by_cell`. Fresh recipes
  default to recovering capacity (builders, not the Pydantic default). w256 L4
  (8,663,554; 22.5×) is the fallback if w160l8 misses the 1,500 s/update gate.
- Four producers for both; roster sampling must work with 2–4 producers.
- Damping horizon per size: `iteration_schedule.steps` = min(10,000,
  ⌊194,400 / calibrated s/update⌋), reference 10,000. Update cap 10,000.
- Equal-hours head-to-heads at every w160l8 milestone and its endpoint (w64's last
  checkpoint at or before the same hours), plus equal-update and endpoint games.
- Host screen, `--size` relaunch, continuation and the direction-25 post-loss
  pair are cut. A lost run stops at its last export and is reported.
- Budget: 2 × USD38 training + USD14 evaluation; USD10 reserve (calibration ≤ 2,
  rehearsal ≤ 2).
- Order: #435 and #437 (merged) → parameter recount (done) → software
  (protocol "Before freeze" 1, 2, 4–6) merges as PR #439 → record coverage sweep
  (seed 0) → calibration (three pods, `--parallel 3`, ≤ USD2) + rehearsal (≤ USD2) →
  catalog-vs-hashed check (≤ USD6) → horizons written → item 3 (Etude `wall_ms`) lands →
  freeze → execution review → launch.

Software (numbered as protocol "Before freeze" 1–6). #435 (2026-11-06), b015d5's throughput
protocol (70d79ccb8) and #437 (30df0c, 2026-11-07, merge 43150799f) are merged in.

1. Done: `ShardedVectorEnv` forwards rosters/decks and concatenates
   `current_player_configs`/`current_matchup_indices`; deck-source,
   `elemental-uniform` and collector one-producer validators removed. Draws come
   from each game's native seed, so no Rust change.
2. Done: `manabot.recipes.builders.sampled_self_play` (four producers, fitted
   `IterationSchedule`, `recover_capacity`, threshold .25, cap 10,000) and
   `fitted_horizon`; field defaults unchanged; docs/training.md, docs/learning.md.
3. Waiting: Etude PR #86 (Task 3d4f40, open 2026-11-07, approved; no patch release).
   When it merges, `tla_frontier.checkpoint_hours` (now returns None) rebases `wall_ms`
   per run, and the freeze change moves the job project and control client to
   `etude-sdk` from main together.
4. Done except the recount: w160l8 label and `LargeSize`, regime from the builder
   (96 permanents, workers 4, cap 10,000, provisional horizons 1,296 / 129), new
   milestones, USD14, per-label game errors, calibration's `expected_updates` and
   `w64-p1` pod; host screen and `--size` deleted. Equal-hours slots are built:
   `comparison_schedule(state, protocol, hours)` over a typed `CheckpointHours`,
   `_partner` (w64's last at or before; pending until w64 passes the hour;
   unreached if w64 ended earlier), scheduled last so `eval`'s `_distinct` plays a
   pair shared with an equal-update/endpoint slot once under that name; analysis
   `Comparison.hours`; a report curve at w160l8's updates. Fixture-hours tests
   cover them. Recount done (see Arms).
5. Built: `experiments/tla_frontier_vocabulary.py` and its tests; the criterion
   is in the protocol (head-to-head `best.upper` < .5; hashed learned and catalog
   rejected; late value loss > 1.10× hashed). With #437 in, the arm test ran and
   showed `EXPORTS` passing update 100 as a milestone, which `TrainingJob` rejects
   (milestones are interior); now `MILESTONES = (25, 50, 75)` and the stop export
   gives 100. The pre-#437 guard and its skip test are deleted.

6. Built (demo review, steer 6472bb87): the fact-contribution reading,
   protocol "Reading › Fact contribution" and "Before freeze" 6.
   `experiments/tla_frontier_facts.py`: `sample_positions` (one seeded
   random-action game per frozen pair, every 4th decision, ≤32 per pair; 495
   positions, 0.3 s), `read_model` (forward hooks on `FactEncoder`, its output
   layer and `SharedEffectEncoder`; ratio to the pre-fact object norm, total, per
   kind and shared effects), `read_checkpoints`/`read` and a CLI over either
   experiment. Fresh w64: facts exactly 0; shared effects median about .21 (no
   zero-init), so update 0 is the shared-effect baseline.

Beyond the protocol's list (direction 26): the set-aside rule is gone. `size_run(state,
label)` admits the one tagged run of a size; a run lost below update 25 is its size's run,
endpoint at its last export, its milestone and endpoint slots unreached (no matchup), its
endpoint readings None, scale absent. `FrontierAnalysis.runs` (`Attempt`) lists every stored
run, size None without a checkpoint.

Check: `uv run --extra dev --extra evaluation --extra artifacts pytest
tests/eval/test_experiment_files.py tests/eval/test_tla_frontier.py
tests/eval/test_tla_frontier_vocabulary.py tests/eval/test_tla_frontier_facts.py -q` 92
passed on 0f9736f21 (2026-11-07). CI run 42347918751 on that head passed all three
Integration suites, Python Unit Tests and Tests Result.

### Concept review 13 (2026-11-06, headless): nothing to change

- Software items 1, 2, 4 and 5 match the accepted design; item 3 waits on Etude Task
  3d4f40. No implement direction follows from the design.
- Settled by the accepted design as written, so no longer open: the freeze waits
  for item 3. Jack Heart accepted a protocol that lists item 3 under "Before freeze",
  and equal-hours slots read `checkpoint_hours`, which is runner code and cannot change
  after the freeze. Calibration and the coverage sweep do not need item 3. Freezing
  without it, and playing equal-hours slots later in a new version, would be a revision
  of the accepted design for Jack Heart to decide, not a choice this step can make.
  Evidence: `training_checkpoints.wall_ms` is already in the store and the analysis's
  `_training_points` reads it for `TrainingPoint.hours`, so the hours exist from the
  first export; item 3 only lets `eval` see them.
- Operator risk, not a design question: 3d4f40 has not started, and starting it is
  Jack Heart's call. If it is still unstarted when calibration's horizons are written,
  the freeze waits there.
- Wording (fixed at realign, 2026-11-07): the protocol's status names both horizon
  files, and "Before freeze" says calibration runs from items 1, 2, 4–6 while the
  freeze waits for item 3.

Nothing is launched. Concept review 15 states the authority for each paid step.

### Concept review 14 (2026-11-07, headless): nothing to change

- Item 6 (steer 6472bb87) matches the demo review: declared under protocol
  "Reading › Fact contribution" and "Before freeze" 6, built in
  `experiments/tla_frontier_facts.py`, descriptive and gating nothing. It stays a
  module beside `tla_frontier_analysis` rather than inside it, because it loads
  checkpoints and the analysis reads only matches; the analysis docstring names
  it. That is a placement under the direction, not a product choice.
- Main has not moved past the base; PR #439 is mergeable with Integration
  pending. Item 3 still waits on Etude Task 3d4f40, which holds the freeze, not
  calibration. No open product choice.

### Concept review 15 (2026-11-07, headless): nothing to implement; one authority gap for the freeze

- The realigned Launch sequence, "Before freeze" and this plan agree, and the Launch
  sequence's order is the one Jack Heart accepted. No design or code change follows.
  Main has not moved; PR #439 is mergeable, its three Integration jobs pending.
- **Paid steps before execution review have no Graph owner.** The Launch sequence
  spends before step 7: calibration and the rehearsal (step 4) and the fact-embedding
  check (step 6). `experiment-freeze` inspects "without executing games or training",
  `experiment-software-review` launches nothing, and only `pool-validation` runs paid
  work, after execution review. The Graph therefore cannot produce the horizons
  `experiment-freeze` must bind. Left alone, the freeze stops there.
- **What authority exists.** The Graph direction says "Any paid launch waits for
  `experiment-execution-review`". The later order and accepted design place calibration
  before the freeze, with its USD2 ceiling and one-producer pod. That settles the order
  for calibration; the rehearsal is in the protocol Jack Heart accepted. Neither
  statement says who launches them, and no step may assume it. The fact-embedding check
  is required only "before either 54-hour run", so its place before execution review is
  the protocol's, not a direction. This corrects review 13's "spend waits for execution
  review", which contradicted the order.
- **Question for Jack Heart at the freeze boundary:** does the design acceptance
  authorize a person or operator conversation to launch calibration, the rehearsal and
  the fact-embedding check within their ceilings before execution review, or does each
  need its own go-ahead? This is execution authority, not a product choice, and the
  Graph's skills keep it at that boundary. It touches no software in PR #439, so
  delivery continues. Until answered, no step launches any of the three.
- Operator note: the fact-embedding check waits on the freeze only because its commands
  name the frozen revision, so it also waits on Etude Task 3d4f40. It needs the horizons
  but not equal-hours code. Running it first would also spare a new protocol version if
  catalog fails. That reorders an accepted protocol, so it is Jack Heart's to decide.

### Concept review 16 (2026-11-07, headless): nothing to change

- Reviewed at 0f9736f21. Main has not moved, PR #439 is mergeable and clean, and every
  check that ran passed. Since review 15 only the discovery test's expected set changed.
- `sampled_self_play` has no production caller, so the default could have drifted from
  the run. Rebuilt from the committed w64 regime's own inputs, its dump matches with no
  differences. Do not pin that in a test: the next frontier run moves the builder while
  this regime stays frozen, so such a test would fail on the intended change.
- Etude Task 3d4f40 is still unstarted, unowned and has no PR; it holds the freeze only.
  Review 15's authority question stays open for Jack Heart at the freeze boundary. It is
  execution authority, so it does not block delivery of PR #439.

### Realign after demo (2026-11-07, headless)

Main has not moved past this branch's base (43150799f). PR #439 is open at 546b496d5:
mergeable, Python Unit Tests, packages and the Linux install passed. CI run 42345721634 then
failed Integration (ubuntu-latest, evaluation), 1 failed, 481 passed, 1 skipped, and Tests
Result with it: `test_etude_finds_each_current_experiment_and_nothing_else` did not list
`tla_frontier`, `tla_frontier_calibration` or `tla_frontier_vocabulary`. The expected set now
names them at 0f9736f21, where CI run 42347918751 is green and the PR is mergeable and clean;
`tla_frontier_facts` and `tla_frontier_analysis` are helpers Etude rightly does
not discover. The protocol's Launch sequence now
matches "Before freeze": step 1 needs items 1, 2 and 4–6, and the freeze in step 5 also needs
item 3. Plan items now use the protocol's numbers.

## History (full text at `739f95b8d:scratch/c5fba4-frontier-self-play.md`)

Protocol v1 was drafted 2026-11-05 (headless) and concept-reviewed eleven times; the
pre-review design (w256, one producer, host screen) is at `8e4ed4a9c`, reviews 1–6 at
`1ef23d800`. Directions 1–27 are implemented and stated in the protocol. Live facts:

- **Identity.** `EXPERIMENT` is the file stem; a launched run's stored name is its launcher
  id, so `size_run` reads the regime tag `RunContext` records. `TrainingRun.job_name` would
  be the better owner (Etude proposal).
- **Lost runs** (directions 22, 26). `with_launcher_ends` closes a `running` run whose launcher
  status is done and unsuccessful; only a worker writes `state.json`. A lost run is its size's
  only run; an update-0 endpoint is unreached. Reopens if a launcher stores two runs under one tag.
- **Passes** (direction 23). A pass id digests its pending matches and the launcher refuses a known
  id, so `C5FBA4_PASS_ATTEMPT` renames a relaunched pass. `etude eval` leaves failed matches alone
  without `--retry-failed`; a failed launched pass blocks later passes from that computer
  (`launched_pass_failed`), untested on a real rental. `etude report` calls `eval`, so it needs
  launcher access.
- **Order** (direction 27). #437 merged before this software did, and the merge was clean
  against items 1–2. A stale local extension (`learning_record_names` missing) fails four
  tests after merging #437: rebuild it.
- **Clients.** Locked 0.2.12 `etude ondemands` fails on a `vast` row; the 0.2.14 overlay with
  nested Doppler works and its evaluation keys match 0.2.12's. The GPU model lives in the
  launcher's session record (`etude status`), so same-GPU is an operator check.
- **Disks.** Arena journals go to `/dev/shm` (MTG-251's fsync cost); capacity with 28 workers is
  inherited, not measured.

Counterexamples still live: `best`/`worst` straddling every threshold (rules or capacity fix,
new version); exports denting large-arm throughput or a pass costing well above USD0.45 (widen
cadence or drop milestones 75/150 by operator stop); calibration peak RSS above about 27 GB;
dropped actions at a meaningful rate (64 slots limit strength; the reading stands); tags failing
on a real checkpoint (Etude `job_name` first); a rental unable to read calibration's checkpoints;
trained games far longer than fresh ones; a late delivery moving an endpoint after its comparisons
played (the endpoint head-to-head name carries no updates, so Etude raises `matchup_conflict`).

## Delete — do not maintain

Deleted 2026-11-06: `read_screen`, `ScreenReading`, `SLOW_HOST_RATIO`, `--screen`,
`runs(size=…)`/`--size` and their tests; the one-producer validators for sampled
rosters; `is_set_aside`, `SetAside`, `FrontierAnalysis.set_aside` and their two tests
(direction 26); `ModelContribution.positions` (the reading's top-level count holds it).
Deleted 2026-11-07 (compress): the `with_launcher_ends` status parameter, the separate
environment and `Benchmark` builders, `read_comparison`'s seven-argument surface and
`sampled_self_play`'s caller-less `updates`/`streams`/`transitions`/`device`/`memory_bytes`.
Shared now: `shortened`, `one_run`/`tagged_runs`, `evaluation_jobs`, `read_comparisons`,
`deck_matchup`, `pod_reading` (one receipt read; `Calibration.pod`). Kept: host speed
(reported, not a gate). `experiments/tla_pilot_v10.py` is executed evidence and stays
unchanged.

## Remaining work

- PR #439 merges (items 1, 2, 4–6). Etude PR #86 (item 3) must merge before the freeze;
  it does not hold calibration, the rehearsal or the fact-embedding check.
- Then the protocol's "Launch sequence": on-demand checks, coverage sweep, calibration,
  rehearsal, the catalog-vs-hashed check, both gates, horizons, freeze, execution review,
  training (paid; needs an execution decision), evaluation driver. The operator conversation
  launches the three pre-freeze paid steps from the merged revision inside the USD10 reserve
  (direction, 2026-11-07), which answers concept review 15. No Graph step does.
  `frozen()` rejects a drifted world, so a card-catalog change before launch moves the pin in
  a new protocol version. Open rules PRs such as #433 can do that.
  `etude launch` refuses several runs without
  `--cohort`, and a cohort's default `--parallel 1` would train the large arm only after
  w64's 54 hours, so both commands name a cohort and two machines.
- Unproven until the first real rental: delivery to the training store, setup cost
  per rental, `/dev/shm` with 28 workers, and calibration's stock worker on CUDA. The
  rehearsal reads all but the last ahead of scoring.
- Before reading a score, confirm from `etude status` that both runs had the
  same GPU model. Host speed is reported in calibration and the analysis.
- Any retry of a size is a new protocol version with new regime ids, since
  `size_run` admits one run per regime.
- A merged PR completes neither the Task nor the spend decision; freeze, execution
  review, execution, analysis and evidence review follow on the `experiment` graph.
- The final report is written after the runs. `tla_frontier_analysis` gives its
  numbers: whole-pair intervals with failed-leg bounds, per-size learning, the
  scale score, hours per checkpoint and every failed, missing or unreached slot. `report()` stays
  descriptive curves. A repeated match key fails analysis loudly, inherited from
  `pair_result`.

## Proposed wave memory (not written)

This Graph's steps leave `.lf/wave/` read-only, so these wait for a step that curates it.

- Inherited per-game time limits need rechecking when model size changes.
- `evaluation()` jobs need explicit setup and the `evaluation` extra to rent.
- A failed rented pass blocks later `etude eval` passes on that computer.
- A protocol's launch commands need checking against the real CLI, not only their job
  file: a locked client can fail on provider rows it cannot parse, and `--max-usd` without
  `--cohort` is refused, while tests of `runs()` stay green. `ondemand startup` takes
  `--seconds`, and its default 0 disables the limit.
- Export cadence and evaluated milestones are separate choices, because exports set what a
  lost rental costs.
- Etude's rental ceiling follows storage (R2 a week, AWS 36 h), superseding MTG-183's
  43,000 s.
- `complete-games` closure makes update size, host memory and update cost follow game
  length.
- Evaluation rows carry first-64 cutoff counts from c5fba4 on; earlier pilots' rows have
  none.
- A launched run's stored name is its launcher id, not `<job>-seed<seed>`, and
  `ExperimentState` carries no job name, so experiments distinguish runs by checkpoint
  tags. Fixtures must not write identities a worker does not write.
- Etude ends an attempt with SIGTERM within 0.1 s of cancellation, so an end state depends
  on the worker's handler, not its stop-file poll.
- A protocol whose failed games are never replayed needs its evaluation path proved on a
  rental before scoring; a killed game is stored as delivered, so no retry reaches it.
- Only the worker writes `state.json`. A lost pod or a guardian kill reads `running` in the
  store for good, so anything keyed on `TrainingRun.ended` needs the launcher's status.
- A launched pass's id is a digest of its pending matches, and the launcher refuses a known
  id, so a pass with nothing new to add cannot be relaunched as it stands. Tournament names
  sit outside match keys, which gives a fresh id with the same matches.
- `etude launch` with a cohort is all-or-nothing and can be rerun. A booted pod ends itself
  through its guardian. Only an unbooted pod needs the controller, and a supply with no
  `startup_seconds` then holds its whole window.
