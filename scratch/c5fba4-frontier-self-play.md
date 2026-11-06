# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 (headless), concept-reviewed eight times through
2026-11-06. Nothing has launched or spent money. Directions 1–25 are implemented. `frozen()`
passes, so `runtime_world` needed no re-freeze. Brief: Task c5fba4; direction comment 694f4190
in [direction.md](direction.md). Contract: `experiments/protocols/tla-frontier-c5fba4-v1.{md,json}`;
runner `experiments/tla_frontier.py`; analysis `experiments/tla_frontier_analysis.py`; calibration
`experiments/tla_frontier_calibration.py`; tests `tests/eval/test_tla_frontier.py`. Full text of the
earlier reviews: `1ef23d800:scratch/c5fba4-frontier-self-play.md`.

Check: `uv run --extra dev --extra evaluation --extra artifacts pytest tests/eval/test_tla_frontier.py -q`: 68 passed on locked 0.2.12, `frozen()` passes and ruff clean (2026-11-06, rerun after direction 25); 68 passed under the 0.2.14 control client at `eeae53b0c`. Gate owns wider suites.

## Design (as implemented)

- Two regimes, identical to MTG-213 v10 except: input v9 / 512 facts, wall
  54 h, update cap 3,000. One regime file (w64) is hash-pinned; w256 is derived
  in `frozen()` from the protocol's `large` size (width 256, 4 layers, 8 heads;
  15.0× compute parameters of w64 = 648k → 9.71M, hashed tables excluded).
  Static declarations of those bytes live in tests, not `frozen()`.
- Damped learning rule kept unchanged (absolute iteration schedule, so the
  update cap does not alter learning rates). Closure-lambda v1 (PR #431) gives no
  reason to move it: closure showed no difference and no arm beat .5 / .8, which
  this regime uses with `complete-games` (read from the resolved stage). That study
  is Allies/Lessons and early training, so it is no TLA evidence; its Elemental
  repeat, 07c490, has not reported.
- Exports (`export_every: 25`, joined with the milestones by `export_updates`)
  are separate from evaluated milestones; `eval` plays only milestones and endpoints.
- Wall-bounded runs → equal compute per size; shared update milestones give
  equal-update head-to-heads while both are running; endpoint head-to-head is
  the equal-compute comparison. Endpoint vs own initialization shows learning.
- 16 fixed pairs × 4 legs per comparison; training USD38 / 56 h per size;
  cumulative evaluation USD12; USD12 reserve under the USD100 cap.
- Both sizes share one GPU model, chosen at launch and recorded before scoring.
  USD38 / 56 h admits at most USD0.67/hour: L4, or A40 if L4 stock is empty, not
  RTX 5090.
- Evaluation passes are cpu32 rentals (28 workers, USD2, 1.75 h plus setup, three
  attempts for lost machines), prepared with the Rust toolchain and `evaluation` extra. Game limit
  600 s; the arena journals in `/dev/shm` and copies out, as `learning_signal` does.
- Control commands use the 0.2.14 client with nested Doppler and the `artifacts`
  and `evaluation` extras, through the protocol's one `control` array (an array so
  `caffeinate` can run it). `--screen` and `ondemand startup` use it too, and the
  protocol owns calibration's commands. Workers stay on locked 0.2.12.

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

Nothing: new capability. `experiments/tla_pilot_v10.py` is executed evidence
and stays unchanged (its callback is mirrored, not shared, because its source
is hashed into retained environments).

## Remaining work

- Review of the protocol before freezing. `frozen()` rejects a drifted world, so a
  card-catalog change before launch moves the pin in a new protocol version. Open
  rules PRs such as #433 can do that; main matched the checkout base on 2026-11-06.
- Freeze at a merged revision, then follow the protocol's "Launch sequence":
  on-demand checks, calibration and rehearsal (paid, USD2 ceiling each), both gates,
  training (paid; needs an execution decision), host screen, evaluation driver.
  `etude launch` refuses several runs without
  `--cohort`, and a cohort's default `--parallel 1` would train w256 only after
  w64's 54 hours, so both commands name a cohort and two machines.
- Sequence the freeze against PR #435 (b015d5), a launch decision; see the fifth
  review's "Left alone".
- Unproven until the first real rental: delivery to the training store, setup cost
  per rental, `/dev/shm` with 28 workers, and calibration's stock worker on CUDA. The
  rehearsal reads all but the last ahead of scoring.
- Before reading a score, confirm from `etude status` that both runs had the
  same GPU model. Host speed is read by `--screen` and in the analysis.
- A retry of a size past the screen (direction 17) is a new protocol version with
  its recorded decision and new regime ids, since `size_run` admits one run per
  regime. The protocol's "Lost rental" section states the default and both limits.
- This Task runs builtin `pursue`, which ends at publication. Freeze, execution review,
  execution, analysis and evidence review are the `experiment` graph's tail
  (`docs/workflows.md`). A merged PR therefore completes neither the Task nor the spend
  decision; that tail is still to be taken up.
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
