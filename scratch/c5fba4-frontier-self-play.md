# c5fba4 — frontier self-play on TLA trophy decks at two sizes

## Accepted design (2026-11-06) — current

The design review (scratch/direction.md, steers d52dc416, a3318c8a, 82d2041b)
returned protocol v1; Jack Heart accepted the revised design the same day (steer
ff3c999d) with the one-producer calibration pod and a catalog-vs-hashed gate. It
is `experiments/protocols/tla-frontier-c5fba4-v1.md`, marked "accepted design, not
frozen". The JSON, regime and runner follow it except where "Software" says
otherwise.

- Arms: w64 (64/2/4, 385,538 compute params) and w160l8 (160/8/8, 5,538,242;
  14.4×), counted on f6f171d34 with six 4096-row tables excluded. Provisional:
  Task 30df0c's input consolidation (steer 82d2041b) removes those tables and
  merges before freeze; both arms use it and counts are retaken on main after it.
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
- Order (revised, eleventh review): #435 (merged) → software (protocol "Before
  freeze" 1, 2, 4, 5) merges without waiting for #437 → #437 (30df0c) merges →
  parameter recount → record coverage sweep (seed 0) →
  calibration (three pods, `--parallel 3`, ≤ USD2) + rehearsal → horizons
  written, freeze → catalog-vs-hashed check → execution review → launch.

Software (protocol "Before freeze" 1–5). #435 merged 2026-11-06 and is synced in;
#437 (30df0c) is still open, head now 40ecda6be (was bab2e80d6). origin/main has
one commit beyond the last sync, 70d79ccb8 (b015d5's throughput probe protocol); it
conflicts only in `scratch/questions.md`.

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
   cover them. Remaining, blocked on #437: recount parameters on main.
5. Built: `experiments/tla_frontier_vocabulary.py` and its tests; the criterion
   is in the protocol (head-to-head `best.upper` < .5; hashed learned and catalog
   rejected; late value loss > 1.10× hashed). Merge dependency: `vocabulary_regime`
   needs #437's `ModelConfig.record_vocabulary` and raises without it; the arm
   regime test is skipped until #437 is merged in, so rerun
   `tests/eval/test_tla_frontier_vocabulary.py` after syncing #437.

6. Built (direction 26): the set-aside rule is gone. `size_run(state, label)` admits the one
   tagged run of a size; a run lost below update 25 is its size's run, endpoint at its last
   export, its milestone and endpoint slots unreached (no matchup), its endpoint readings None,
   scale absent. `FrontierAnalysis.runs` (`Attempt`) lists every stored run, size None without
   a checkpoint.

Check: `uv run --extra dev --extra evaluation --extra artifacts pytest
tests/eval/test_tla_frontier.py tests/eval/test_tla_frontier_vocabulary.py -q`
83 passed, 1 skipped (needs #437) (realign after direction 26, 2026-11-06). Gate owns wider suites.

Not changed: execution authority. Nothing is launched; spend waits for
`experiment-execution-review`.

## Eleventh concept review (2026-11-06, headless): software lands ahead of #437

Main moved to 70d79ccb8, which adds b015d5's draft throughput protocol. That protocol leaves
multi-producer collection to this Task ("read from its result, not repeated here") and builds
its two-seat exact-quota step "after c5fba4's `collection.py` change lands". This branch's
order holds every software merge behind #437 (protocol "Before freeze" opening line and Launch
sequence step 1). The accepted direction does not ask for that: software builds on main after
#435, and 30df0c lands "before this protocol freezes". The stricter order now delays a second
Task with no gain, because nothing in items 1, 2 or 4 reads #437, and item 5 already runs on
both sides of it (`HAS_RECORD_VOCABULARY` picks the refusing test before #437 and the arm test
after it).

### Direction, for implement

27. Software merges once #435 is in, not after #437. Edit the protocol's "Before freeze" opening
    line and Launch sequence step 1 so they agree: software merges after #435; #437 merges before
    freeze; the parameter recount follows #437 and comes before calibration. The status paragraph
    already puts freeze after #437 and the recount, and stays. Item 4's "recounted parameters" is
    a freeze step, not a merge blocker. Item 5's citation of #437 head `bab2e80d6` gives way to
    "rerun on the frozen revision", because the head has moved (40ecda6be). Sync origin/main; the
    only conflict is `scratch/questions.md`, which main rewrote for b015d5. Keep this Task's
    entries and main's.

What would reopen this: #437 changes `ModelConfig`, the regime builders or `collection.py` in a
way that conflicts with items 1 and 2. In that case the later of the two PRs rebases; the order
of freeze steps does not change. No product choice is open.

## Tenth concept review (2026-11-06): lost runs are not replaced

The accepted design cut replacement ("a lost run stops at its last export and is reported";
protocol §Lost run), but `is_set_aside` still admitted a second run when the first ended below
update 25, so a w160l8 pod lost early left its size with no run and the analysis could never be
complete. Direction 26 (item 6 above) removed it. An update-0 endpoint is unreached rather than
played, because a policy played against itself would read as learning rejected. What would reopen
this: any launcher path that stores a second run under one regime tag (Etude `max_attempts` is
unused). The admission rule would then need an owner, preferably Etude's run identity.

## Draft as reviewed

Protocol v1 drafted 2026-11-05 (headless) and concept-reviewed nine times; its
design (w256, one producer, absolute schedule, host screen) and checks are at
`8e4ed4a9c:scratch/c5fba4-frontier-self-play.md`, earlier reviews at
`1ef23d800:scratch/c5fba4-frontier-self-play.md`. Nothing has launched or spent money.

## Reconciliation findings (2026-11-05)

Observed:

- `game_seconds` is a hard kill that counts as a failed game. One real
  observation took 130–170 ms (w64) and 380–420 ms (w256) per single-thread
  forward on this Mac; v10 games averaged 180 decisions. Raised to 600 s. A real
  match later ran about 0.05 s a decision (direction 18's notes), and the protocol
  now says the limit may be generous.
- `eval()` had no machine setup, so a stock rented machine could not build the
  engine or write results. Now follows `experiments/learning_signal.py`.
- Update 0 and `clocks["updates"]` are real: execution exports the initial raw
  checkpoint and v10 read the same clock.
- Checkpoint clocks carry updates and transitions, no wall time, so `eval` cannot
  pick equal-compute pairs before the endpoints. Etude does retain each
  checkpoint's `wall_ms`, in `training_checkpoints`; only `ExperimentState`
  drops it. The analysis reads it as hours per checkpoint.
- Installed Etude source, read 2026-11-05: `etude eval` counts a failed match and
  leaves it alone unless given `--retry-failed`, so a default pass never trips
  analysis's repeated-match error. Its launch ledger (one file per computer)
  skips matches a rented pass still has in flight, so overlapping passes from
  one computer do not duplicate games.
- The same ledger raises `launched_pass_failed` on every later rented pass once
  any launched pass ends unsuccessfully, and nothing clears the entry, so a
  genuine pass failure stops evaluation until someone intervenes. This is
  untested against a real rental; direction 6 removes the routine trigger.
- The GPU model is held in the launcher's session record, shown by `etude
  status`, not in the training store, so analysis cannot check the same-model
  rule. It stays an operator check.
- `uv run etude ondemands` on locked 0.2.12 fails today on a `vast` provider row;
  the overlay with nested Doppler lists offers, so the protocol's bare commands
  could not have launched. Under 0.2.14 the 22 tests pass, and an evaluation job,
  its environment identity and match keys are byte-identical to 0.2.12's, so a
  0.2.14 planner, 0.2.12 worker and either analysis agree on keys.
- Frontier `play` wrote its fsync-per-record arena journal to the rented disk,
  inside the 600 s limit. MTG-251 found that at about 0.1 s per record and
  `learning_signal` works around it. Now mirrored; `/dev/shm` capacity with 28
  workers is inherited from that study, not measured here.

Estimates, not measurements: w64 about 1,300 updates at v10's 150 s/update (v9
inputs may cost more); w256 per-update GPU cost and CUDA memory are unknown. About 36 comparisons
would cost about USD8.50 with per-rental setup (see directions 4 and 11), within
the USD12 cap. At
USD0.59/hour the 54-hour wall spends about USD33 per size, so USD38 is a ceiling,
not an expected cost.


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
  `--cohort`, and a cohort's default `--parallel 1` would train w256 only after
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

## Concept reviews one to six (digest)

Full text: `1ef23d800:scratch/c5fba4-frontier-self-play.md`. Directions 1–21 are implemented and
stated in the protocol. Live facts and choices:

- **Identity.** `EXPERIMENT` is the file stem. A launched run's stored name is its launcher id, so
  `size_run` reads the regime tag `RunContext.checkpoint` records; tags stay out of match keys.
  `TrainingRun.job_name` would be the better owner, an Etude proposal.
- **Reach.** `endpoint` is the last stored checkpoint of any ended run. `scale` needs both finished
  below the cap. `execute_regime` closes a wall stop `finished`. SIGTERM ends a `TrainingJob` as
  `interrupted`/`failed`, so a cancelled pod records its end.
- **Failure.** Failed legs bound a reading (`worst`/`best`); nothing is replayed, and a killed game is
  a delivered match. `etude eval` leaves failed matches alone without `--retry-failed`. Cutoff counts
  ride on `games` rows (`PolicyActions.games` says how many returned one).
- **Gates.** Calibration: 5 updates, 3,600 s, USD1 per size, w256 at 1,500 s/update or less. Rehearsal:
  all 128 games valid and the slowest under 300 s. A real fresh w256 match here ran about 0.05 s a
  decision, against the protocol's 0.4 s single-observation reading.
- **Host screen.** Step speed over updates 2–6, 1.3× line, one replacement per size, below update 25;
  a pod past 25 is kept. `--size` relaunches alone with no `--max-usd`. A never-started pod is not that
  replacement. Past the screen no run moves because of a score.
- **Rentals.** `l4`, `mtg252-a40` if L4 is empty, `cpu32` for passes; `ondemand startup --seconds 900`
  where absent. A booted pod's guardian ends it; only an unbooted pod needs the controller.
- **Estimates.** Expected spend about USD77 of USD100. Host memory follows rows (fit 11.1 GB + 131 KB
  a row at v7; v9's row 207 KB). About 15 rentals at USD0.45. Rental ceiling follows storage (R2 a
  week). One loss over 108 pod-hours is about as likely as not (one event, crude). Launch disks hold.
- **Left alone.** Launch timing against b015d5 (PR #435, open) is a launch decision: it projects about
  25% off an update, touches no protocol file, and conflicts only in `scratch/questions.md`. Ledger
  scoping, heuristic saturation, `workers: 1`, a hand-assembled final report, Academy loading of the
  deliverable checkpoints, and clip rates at w256 belong to the final read or later proposals.

Counterexamples still live:

- `best` and `worst` straddle every threshold: a rules or capacity fix and a new version.
- A retried pass replays a stored match (`repeated match`): passes go back to one attempt.
- Monitoring exports dent w256 throughput, or a pass costs well above USD0.45: widen the cadence, or
  drop milestones 75 and 150 by operator stop.
- Calibration's peak RSS exceeds about 27 GB: review host and limit in a new version.
- Dropped actions at a meaningful rate: 64 slots limit strength; the reading stands.
- Tags fail on a really delivered checkpoint: Etude's `job_name` first, and launch waits.
- `step_spread` above 1.3× on one pod: no screen for that launch, reporting only.
- A corrected one-size launch refused at a real rental: the screen needs another shape.
- A rental cannot read calibration's checkpoints: rehearse from a small file of its own.
- Trained games run far longer than fresh ones: failed-leg bounds report it; the next limit moves.

## Seventh concept review (2026-11-06, headless)

Read the protocol's recovery sentences against what Etude does when something is lost. Only a
worker writes `state.json`, so a lost pod read `running` for good and its size lost its endpoint,
learning and strength readings. A pass's id is a digest of its pending matches and the launcher
refuses a known id, refused creates included, which stranded the endpoint pass and the rehearsal.
`etude eval` returns at launch, and a refused first create ends `--watch` silently.

Evidence, from Etude 0.2.14 source:

- Nothing under `etude/launch` writes `state.json` or names `RunState`. `Status.done` with `lost`,
  `out_of_time` or `over_budget` already says the attempt is over, and a guardian deletion leaves the
  same store as any death that gives the worker no time.
- On the real scheduler in a scratch `ETUDE_HOME`: a failed run, then `run '...' already admitted`,
  while a fresh id is accepted. A refused create pins its id too (`reconciling existing request`, seen
  live in MTG-251 and 07c490). Stock refusals are ordinary: an `available` offer is not stock.
- `played` counts what any eval run of the experiment delivered in its environment, "whichever
  tournament it was held for", and analysis reads by key and environment.
- `max_attempts` covers a machine lost after admission; `advance` calls `client.launch`, which raises,
  and the CLI catches only `KeyboardInterrupt`.
- Checked and closed: a pass's stored run is its launcher id (`ETUDE_RUN` wins, `tournament.py:155`);
  `observe` skips `kind == "evaluation"`; main matched this checkout's base, PR #435 was still open,
  and Etude main had no fix for either finding.

### Directions, implemented

22. `tla_frontier.with_launcher_ends` joins `Status` to the store: a `running` run `done` with an
    outcome other than `succeeded` becomes `failed`. Both `eval`s and analysis's `main` call it;
    `comparison_schedule` and `analyze` stay pure.
23. `C5FBA4_PASS_ATTEMPT`, read only in `pass_name`, suffixes both benchmark names.
24. `etude wait <run>` before `--rehearsal`; the driver's stop and restart rules; "retries" narrowed
    to machines lost or refused after admission.

Implementation notes (2026-11-06):

- The default status callable is a wrapper, `_launcher_status`, not `client.status` itself. It is the
  behavior the direction names, resolved at call time, so a test's replacement reaches an `eval` that
  Etude loaded through `runpy`. `read_rehearsal` calls `client.status` directly for a pass's cost.
- A run with a stored end is never asked about, so "unknown raises" applies to `running` runs only.
  Finished history reads with no launcher; the real 0.2.14 and 0.2.12 clients raise `no run`.
- **`etude report` calls `eval` too** (`page._wanted`), so it now needs launcher access. Its protocol
  command already runs under `control`, and the text says so.
- **Report tag curves do not split.** `played` counts delivery by any run, so over a store with both
  passes each tournament shows identical tag totals, with or without the variable. The only effect is
  a second name in the selector, which is cosmetic. A test holds it.
- A rehearsal match stored as failed stays failed, so the attempt recovers a pass that delivered
  nothing or part, never a failed match; that reading fails the gate. The protocol says so.
- `etude ondemand max cpu32 2` and `etude wait <run>` match the real 0.2.14 `--help`.
- `read_rehearsal` finds its runs from match receipts, so `pass_usd` omits a pass that stored nothing,
  the case the attempt recovers. The gate reads no cost, so the protocol names the gap and sends
  that pass's cost to `etude status`.
- The endpoint head-to-head's name, `w256 final vs w64 final`, carries no updates. A late delivery
  moving an endpoint would define that name two ways, so Etude would raise `matchup_conflict` and end
  the driver, not merely leave games unread (read from source, not run). A dead pod delivers nothing
  later, so this is left alone. Recorded under the counterexample below.

### Left alone

- **Etude owners.** `observe` joining launcher ends and `advance` taking a fresh id belong in Etude
  ([[etude-is-ours]]) and should be proposed, beside ledger scoping and `job_name`. Each would move
  the control client's pin and hold this PR. Each manabot side has one reader and goes when Etude lands.
- **Continuation after a loss** stays a recorded decision under a new version.
- **A supervised driver** would also restart past a failed pass someone should read.

### Counterexamples that would reopen this

- A real lost rental's status reads anything but `done` once its attempts end: the helper's condition
  moves.
- `played` or `statsdb` refuses two tournaments over one environment on a real store (it does not on
  a local one): 23 needs the Etude change first.
- A late delivery moves an endpoint after its comparisons played: report the unread games, and give
  the endpoint head-to-head's name its updates so the driver keeps running.

## Eighth concept review (2026-11-06, headless)

Walked the normal path, a lost pod and recovery against the brief and direction 694f4190, looking
past recovery mechanics to what the experiment can still say. The schedule, gates and recovery hold.
The checks were not rerun for this review beyond the frontier tests: 68 passed at this worktree.
Main has not moved, and PR #435 is still open.

One mismatch. The first review kept equal-compute head-to-head at the endpoint only, with equal-hour
pairs left "if someone later wants" them, and that reasoning assumed both runs finish. The third review
then estimated a loss over 108 pod-hours at about 47% (1 − exp(−108/170.5), one event, crude). So the
primary scale reading is absent about as often as present, while the evidence to read equal compute
after a loss is already stored: every 25-update export, and each checkpoint's wall time, which
`TrainingPoint.hours` reads. Nothing needs building now, since the comparison can be played after the
loss. What cannot wait is the rule: a pair chosen after seeing curves is the choice this protocol exists
to prevent, so it has to be written before scoring. This narrows the first review's decision and does
not reverse it; `scale` and the runner stay as they are.

### Direction, implemented

25. Add to the protocol's "Lost rental" section, with no code change. If a run ends unfinished, whatever
    the scores show, one equal-hours head-to-head is played: the lost run's endpoint against the other
    size's last export at or before the same `TrainingPoint.hours`, on the 16 frozen pairs and four legs,
    w256 first, read with the same bounds. It is a descriptive secondary reading: `scale` stays absent,
    and both hours are reported. The pairing never gives the survivor more compute, so the gap favors the
    lost run, by up to 10.4 hours when w256 is the survivor at the gate speed; say so. If both runs are
    lost, the earlier-ending run is the one paired. The survivor's export need only be stored, so the
    game can wait for the survivor's run to pass that hour. It costs one comparison, inside the USD12.
    Like continuation, this version does not build the path: a small follow-up file plays it through
    `tla_frontier.play`'s conditions, and its environment is its own, as the rehearsal's is.

### Left alone

- Mid-run equal-hour pairs on the normal path: still the first review's decision. Both heuristic curves
  and `TrainingPoint.hours` show strength per hour; a reading of more than that wants a `wall_ms` on
  `ExperimentState`, an Etude proposal.
- A `hours` clock on checkpoints would let `eval` schedule this pair itself, at the cost of a change to
  shared training code for a conditional reading.
- `<store>` is a placeholder through the protocol. Launch names it, like `<frozen>`.

### Counterexample that would reopen this

- A real loss whose survivor export sits too far from the lost run's hours to mean anything: report the
  gap and leave the reading out rather than move the rule.

## Ninth concept review (2026-11-06, headless)

Nothing to change and no open product choice. Main has not moved from the checkout base and PR #435
is still open. Direction 25 is in the protocol's "Lost rental" section as recorded. Its `hours` is
`TrainingPoint.hours`, `wall_ms` since the run's first stored checkpoint (`tla_frontier_analysis.py`).
That checkpoint is update 0 for both sizes, so pre-training setup counts for neither, and "at or
before" never gives the survivor more compute. The brief's deliverables, the direction's progress
readings at milestones (each size against the heuristic, plus equal-update head-to-heads) and the
cost bounds each still have one owner in the protocol. The remaining decisions belong to the
`experiment` graph's tail, not here: freeze revision, launch timing against #435, and spend.

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
