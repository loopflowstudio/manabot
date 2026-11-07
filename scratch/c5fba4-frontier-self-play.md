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
  (protocol "Before freeze" 1, 2, 4, 5) merges → record coverage sweep (seed 0) →
  calibration (three pods, `--parallel 3`, ≤ USD2) + rehearsal → horizons
  written, freeze → catalog-vs-hashed check → execution review → launch.

Software (protocol "Before freeze" 1–5). #435 (2026-11-06), b015d5's throughput
protocol (70d79ccb8) and #437 (30df0c, 2026-11-07, merge 43150799f) are merged in.

1. Done: `ShardedVectorEnv` forwards rosters/decks and concatenates
   `current_player_configs`/`current_matchup_indices`; deck-source,
   `elemental-uniform` and collector one-producer validators removed. Draws come
   from each game's native seed, so no Rust change.
2. Done: `manabot.recipes.builders.sampled_self_play` (four producers, fitted
   `IterationSchedule`, `recover_capacity`, threshold .25, cap 10,000) and
   `fitted_horizon`; field defaults unchanged; docs/training.md, docs/learning.md.
3. Waiting: Etude `Checkpoint.wall_ms` PR and release in `loopflowstudio/etude`.
   Filing belongs to the operator conversation (steer a6c38666), not this step.
   When it lands, `tla_frontier.checkpoint_hours` (now returns None) rebases
   `wall_ms` per run; the control pin moves to that release.
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

6. Built (direction 26): the set-aside rule is gone. `size_run(state, label)` admits the one
   tagged run of a size; a run lost below update 25 is its size's run, endpoint at its last
   export, its milestone and endpoint slots unreached (no matchup), its endpoint readings None,
   scale absent. `FrontierAnalysis.runs` (`Attempt`) lists every stored run, size None without
   a checkpoint.

Check: `uv run --extra dev --extra evaluation --extra artifacts pytest
tests/eval/test_tla_frontier.py tests/eval/test_tla_frontier_vocabulary.py -q`
83 passed (Question ratio fix, with #437 merged, 2026-11-06). Gate owns wider suites.

### Concept review 12 (2026-11-06, headless): next implement direction

- Done: the protocol's Question now says about 17× in compute parameters (17.4×;
  13.6× in total), matching the counts section; no other copy of the old ratio remains.
- Not a change, kept for the freeze: Etude's store already holds `wall_ms` in
  `training_checkpoints`, so the hours are on record from the first export. Item 3
  only exposes them, so it gates when equal-hours slots can be played, not training.
  It does not gate calibration either, which runs from 1, 2 and 4.
- Unresolved (Jack Heart's call, not yet blocking): if Etude Task 3d4f40 has not
  released by the time calibration's horizons are written, does the freeze wait for
  it (the protocol as written, item 3 before freeze), or does it freeze and play
  equal-hours slots retroactively once a new version moves the client pin?
  Evidence: `training_checkpoints.wall_ms` is already in the store
  (`etude/eval/query.py`), and `checkpoint_hours` returns None until the Etude
  change lands.

Not changed: execution authority. Nothing is launched; spend waits for
`experiment-execution-review`.

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
(direction 26). Kept: `host_speed` (reported, not a gate). `experiments/tla_pilot_v10.py` is executed
evidence and stays unchanged.

## Remaining work

- Review of the protocol before freezing. `frozen()` rejects a drifted world, so a
  card-catalog change before launch moves the pin in a new protocol version. Open
  rules PRs such as #433 can do that; main matched the checkout base on 2026-11-06.
- Freeze at a merged revision, then follow the protocol's "Launch sequence":
  on-demand checks, calibration and rehearsal (paid, USD2 ceiling each), both gates,
  horizons, the catalog-vs-hashed check, training (paid; needs an execution
  decision), evaluation driver.
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
