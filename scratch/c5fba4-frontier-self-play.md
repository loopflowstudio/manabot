# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 by the first implement step, reconciled and
concept-reviewed five times the same day (headless); directions 15–17 implemented and
reconciled 2026-11-06, then reviewed a sixth time. Nothing has launched or spent money.
Directions 1–21 are implemented. `frozen()` passes, so `runtime_world` needed no re-freeze. Brief: Task c5fba4; direction
comment 694f4190 in [direction.md](direction.md). Contract:
`experiments/protocols/tla-frontier-c5fba4-v1.{md,json}`; runner
`experiments/tla_frontier.py`; analysis `experiments/tla_frontier_analysis.py`;
calibration `experiments/tla_frontier_calibration.py`; tests
`tests/eval/test_tla_frontier.py`.

Check: `uv run --extra dev --extra evaluation --extra artifacts pytest tests/eval/test_tla_frontier.py -q`: 56 passed on locked 0.2.12 and under the 0.2.14 control client (2026-11-06, after compress; `tests/eval/test_tla_pilot.py` too on 0.2.12). Gate owns wider suites.

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
  forward on this Mac; v10 games averaged 180 decisions. W256 head-to-heads
  would average about 50 s before rented-CPU slowdown, so MTG-213's 120 s would
  fail long games by size. Raised to 600 s.
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
- The final report is written after the runs. `tla_frontier_analysis` gives its
  numbers: whole-pair intervals with failed-leg bounds, per-size learning, the
  scale score, hours per checkpoint and every failed, missing or unreached slot. `report()` stays
  descriptive curves. A repeated match key fails analysis loudly, inherited from
  `pair_result`.

## Concept review (2026-11-05, headless)

Reviewed against the brief, direction 694f4190 and the change. It found three
paths the tests skipped, all now closed, without changing sizes, recipe, schedule
or budget split:

1. **One experiment name.** `etude eval` and `etude launch` name an experiment for
   its file, so the old `tla_frontier_c5fba4` would have played nothing while
   training ran, and failed each worker at `track`
   (`experiment_differs_from_launch`). `EXPERIMENT` is now the stem, `eval` rejects
   a foreign state, and a store-backed test crosses that seam.
2. **Reach survives failure.** `endpoint` is the last stored checkpoint of any ended
   run. Analysis keeps learning and strength beside `state`; `scale` is None with
   an issue unless both runs finished.
3. **Calibration is its own job.** Five updates, 3,600 s wall, at most 1.2 h and
   USD1 per size, numerics gate on, under its own experiment and regime ids.
   `execute_regime` logs `memory/peak_{allocated,reserved}_cuda_bytes` on each CUDA
   update, so it needs only the stock worker. If w256 cannot fit or leaves too few
   updates, the size changes in a new protocol version, before any scoring.
4. **The evaluation driver is named in the protocol.** One `etude eval --watch
   14400` from a checkout of the frozen revision on an always-on host. Estimate:
   about 15 rentals at USD0.45 (setup included, unmeasured), about USD7; one rental
   per comparison would approach USD10–13. The USD12 cap is an operator check.

`launched_pass_failed` ends the watch loop, and the ledger has no way to clear an
entry and is shared by every experiment on the computer (today it holds
`elemental-lambda-v1-6860cc47ef30`; Etude main matches). The protocol records the
check and recovery. The real fix belongs in Etude (scope the ledger per experiment,
add an acknowledge step) as a separate change, not required for this launch.

Checked and left unchanged:

- Damped updates ignore `progress`, so `schedule_clock: iteration_fraction` does
  not make the update cap a schedule input; the protocol's claim holds.
- Match keys exclude source versions, so a later commit cannot replay games.
  It would change `seats.source`, which analysis flags as a mixed cohort, so
  every pass runs from the frozen revision.
- Wall stops record the final raw checkpoint through `experiment.checkpoint`, and
  the run counts as a launcher success, so `run.final` is the real endpoint.

### Settled by accepted direction

- **Head-to-head past w256's reach.** Direction 694f4190 says "matched
  updates/compute". Both sizes train concurrently, so equal-update games continue
  as w256 reaches each milestone, nearly to the end of the wall budget, and the
  endpoint pair is equal compute. That meets the direction. Each size's curve
  against the heuristic covers w64's extra updates. No Etude `wall_ms` change. The
  wall times are retained if someone later wants equal-hour pairs.
- **Launch timing against b015d5.** This is sequencing, not a question of what the
  experiment is: the brief says to coordinate, and the protocol is unchanged
  unless the world moves. It belongs to the launch decision, after this PR, with
  calibration's per-update reading as its evidence. It does not hold the PR.

### Counterexamples that would reopen this

- Calibration puts w256 past roughly 1,500 s/update (about 130 updates in 54 h, a
  rough line): too few for a useful curve, so the size choice needs review.
- A pass cannot deliver to the store the runs use: the driver needs a different
  shape than `--watch`.
- `observe` does not report a dead pod's run as ended: reach stays unknown until
  the launcher records it, and direction 2 needs an operator read of the stored
  checkpoints.

## Second and third concept reviews (2026-11-05, headless)

Full text: `af1d4a1a2:scratch/c5fba4-frontier-self-play.md`. Directions 5–12 are
implemented and stated in the protocol: failed legs bound a reading (`worst`/`best`), three
evaluation attempts, the named `etude report` read, the 1,500 s/update gate, `scale` needing
both finished and neither at the cap, `export_every: 25`, milestones 75 and 150, and the
lost-pod path. Still live:

- `execute_regime` closes a wall stop `finished`, so `scale` survives the normal path.
- Etude bounds a rental by its storage's grant (R2 a week), so 56 h is admitted. No rented
  attempt has run past 23.5 h. The launcher's 1 machine loss in 170.5 GPU-hours suggests a
  loss over 108 pod-hours is about as likely as not (one event, a crude rate).
- A dud pod's cost stays unmeasured; a shared match in two slots plays once; ledger scoping
  stays an Etude change for the launch decision; `report()`'s head-to-head mirror is cosmetic.

Counterexamples that would reopen them:

- `best` and `worst` straddle every threshold: a rules or capacity fix and a new version.
- A retried pass replays a stored match (`repeated match`): passes go back to one attempt.
- Monitoring exports dent w256 throughput (`monitoring_export_seconds`): widen the cadence.
- A pass costs well above USD0.45: drop 75 and 150 first, by operator stop.

## Fourth concept review (2026-11-05, headless)

Walked what the first three left out: what a run 80 times longer than v10 does
that 16 updates could not show, and what the final read needs for the brief's
"what limits it next". Sizes, recipe, schedule, budget split and reading rules
stay. One accepted diagnostic is missing from evaluation.

### Checked and closed

Measured on v10's retained receipts, extrapolated to about 1,300 updates:

- **Host memory does not leak, and it follows rows.** RSS stays 16.6–18.3 GB
  across all 16 updates. `complete-games` closure makes `transitions` a floor, so
  an update is about one game per stream (263 games, 47k rows) and its size follows
  game length. The fit over 14 updates is 11.1 GB + 131 KB per row (r 0.99); v7's
  row is 145 KB. V9's row is 207 KB, which predicts about 22 GB at v10's peak rows.
  The regime's 40 GiB stop (`execution.py:834`) then needs about 153k rows, 2.8×
  that peak. This is a fit to 14 points at v7, so direction 14 reads the real value.
- **CUDA memory does not follow rows.** Learning moves minibatches to the device
  (`damped_policy.py:317`), so calibration's peak holds as rows change.
- **Long-run files stay small.** The game ledger is 4.5 KB a game, about 1.5 GB
  over 340,000 games. `by_cell` saturates at 35,912 cells of 153 bytes, about 5.5 MB
  of `run.json`. Failed games alone keep actions. With at most about 7 GB of
  exports, the rented disk needs roughly 9 GB. The template's disk size was not
  read here; that is a launch check, and a full disk ends at the last export.
- **Nothing kills a slow update.** The only watchdog is the whole-run budget. The
  stop rule waits for an update boundary, and 56 h leaves 2 h past the wall.

### Directions, implemented

13. **Evaluation retains the first-64 cutoff counts.** The overflow exception
    Jack Heart accepted on 2026-11-04 comes with aggregate replay-linked
    diagnostics, and the protocol gives every policy that cutoff. Training keeps
    its receipt (`failures/action-cutoff-*.json`; v10 counted 755,995 decisions and
    no overflow). Evaluation's report stayed on the game (`native_game.py:366`), so
    it reached the retained traces but no row: all 230 v10 evidence files hold no
    count, and the final read could not say from the store whether a trained policy
    lost choices in a scored game. The brief names overflow past 64 slots as a limit
    left open, and bigger boards from stronger play are where it would first appear.

    Carry the decisions and dropped actions per game into `play`'s `games` rows,
    sum them per `Comparison`, and show the totals in the analysis. Frozen rows
    must keep their digests, so any `MatchRow` field is omitted when absent. An
    overflow is accepted behavior, so a nonzero count is reported and adds no
    issue: `complete` stays about missing and failed games. Totals span both
    players, as in training's receipt; per-policy rates are not claimed. First
    prove with a real game that the key is populated on the evaluation path.

14. **Calibration reads host memory and rows.** It reports the peak of
    `coordinates/rss_bytes` and median rows per update beside its CUDA peaks. V9
    has never run on a GPU, and the 40 GiB stop is a real failure that the
    reading's "cannot fit" should cover. These are retained rows, so this adds two
    fields and no gate.

Implementation notes (2026-11-05):

- Real proof: one `tla_frontier.play` on a fresh w64 checkpoint against the
  heuristic played four valid games. Each game's report summed to its decision
  count (266, 73, 156 and 285) with 0 dropped at 64 slots. A test repeats this at 4 slots,
  where real games do overflow.
- V10's own traces were not inspected for the report; rows carry it from here on,
  so analysis reads it from the store.
- `MatchRow.policy_actions` uses `exclude_if`. `EvaluationRow` reads the key from its
  extras, because a declared field would dump as None and change the digest
  `_verify_block` recomputes.
- A killed game returns no count, and limits fall on long games. `PolicyActions`
  therefore adds `games`, the games counted, beside the two totals. Fewer than the
  played count means the totals are a lower bound.
- Analysis reads cutoffs once, grouped by match. The whole-analysis total sums
  comparisons, as `counts` does, so a match shared by a milestone and the endpoint
  head-to-head counts twice in both and `games` stays comparable with played games.
- `coordinates/rss_bytes` is sampled once per update, so `peak_rss_bytes` can miss a
  peak inside an update. `sampled_peak_rss_bytes` stays in `run.json`.
- Etude omits an all-null column from a block and reads blocks by name, so a match
  of four killed games stores no count column. The real-store test now holds one
  such match beside counted ones, and it counts zero games there.

### Left alone

- **Rows per update are not fixed, so an "update" is about one game per stream.**
  Equal-update comparisons stay fair because both sizes have that shape. Each
  checkpoint's transitions are retained if the report needs them.
- **The final report stays hand-assembled**, as v10's was. The numbers for "what
  limits it next" are retained: `time/*` phases, `game_errors`, attempts and cost.
  Building it now would guess at what the run shows. A dead pod loses its
  `run.json` and cutoff receipt; the protocol already names that limit.
- **Expected spend is about USD77 of USD100** (about 2 × USD33, USD8.50 of
  evaluation, USD2 of calibration). That is headroom for direction 12's
  continuation. Extending the wall would need a price known only at launch.

No open product choice. Direction 13 implements a diagnostic the overflow exception
already requires, and 14 widens an accepted reading.

### Counterexamples that would reopen this

- Calibration's peak RSS exceeds about 27 GB: a 1.55× rise in rows, MTG-183's
  slowdown, would then pass the 40 GiB stop. Review the host and limit in a new
  protocol version before launch.
- Scored games show dropped actions at a meaningful rate: 64 slots limit strength,
  and the next protocol's capacity needs review. This does not void the reading.


## Fifth concept review (2026-11-05, headless)

Walked what the first four left out: the store as a real launch writes it, and the
pods a launch actually gets. Sizes, recipe, schedule, budget split and reading rules
stay. One defect stops all evaluation. One new fact from b015d5 changes what the
hardware check records.

### Observed

- **A launched run's stored name is a random id, so `size_run` finds nothing.**
  `Scheduler._work` hands `run=run.id` (`uuid4().hex[:12]`), the worker exports it
  as `ETUDE_RUN`, and `manabot/infra/runtime.py:78` tracks under it. V10's store
  confirms this: its folder is `ae76c17951c3`, though its job was named for the
  regime. `RunRecord.name`, `<job name>-seed<seed>`, exists only in the launcher.
  `size_run` matches that name by prefix, and the test fixture `_tracked` writes it
  by hand, so every store-backed test passes on a name no worker writes. A probe
  wrote both runs under hex names with checkpoints at 0, 25 and 50: `size_run` gave
  None for both sizes, the schedule held 0 playable matchups, and `eval` returned
  `[]`. It raises nothing, so the watch loop would sit idle for all 54 hours and
  direction 694f4190's evaluations would never play.
- **`ExperimentState` cannot name a size.** `TrainingRun` holds name, experiment,
  state and checkpoints in both 0.2.12 and 0.2.14. `observe` drops the declaration's
  `job_name`. `Checkpoint.tags` does reach it, and `Checkpoints.tagged` reads them.
- **Calibration has the same blind spot.** `Calibration.run` would be a hex id, so
  its reader could not say which reading is w256, the one the gate needs.
  `RunData.file.declaration.job_name` is available there, so this needs no Etude
  change.
- **Host class sets the pace, not vCPUs (b015d5, `experiments/training-update-time.md`
  in its PR #435, open and unmerged on 2026-11-06).** One producer means one CPU thread, and this
  regime has `workers: 1`. At a fixed GPU label, update time moved 1.6–1.8×. Retained
  Elemental runs split 128–146 µs per decision (fast) from 205–239 (slow). Of the
  retained L4 runs, 5 were fast and 2 slow; every A40 was slow, at the same price. The
  protocol's check records vCPUs, which that work refuted as a cause, and offers A40
  as equal to L4.
- **TLA has no host anchor.** V10 spent 2,388 µs of collection per decision against
  Elemental's roughly 580, unattributed. Its store is not on this machine, so its
  step series and `rows` key were not read. Those bands are Elemental's.
- **A run has one host.** Replacing a slow pod is `etude cancel` and a new launch from
  update 0, under a new run id. `size_run` then raises on a second run, and
  `submit` refuses a repeated cohort name.

### Directions, implemented

15. **Name each size on its stored checkpoints (blocks launch).** `RunContext.checkpoint`,
    the one path for all three `experiment.checkpoint` callers, records a tag naming
    the regime, such as `regime:tla-frontier-c5fba4-w64`. `size_run` selects by
    `run.checkpoints.tagged(...)` instead of by name. A tag makes the stored record
    self-describing, uses a mechanism both Etude versions already carry, and needs no
    release. Check that tags stay out of match keys.

    `_tracked` must write hex run names and take its tags from the real path, and one
    test must drive a real `RunContext` under `ETUDE_RUN=<hex>` and read the size back
    through `observe`, so no fixture states the seam again. Calibration shows the job
    name beside the run id. Correct `size_run`'s docstring.

    The gap itself belongs to Etude: `TrainingRun.job_name`, from the declaration, is
    the better owner and [[etude-is-ours]] applies. It would need a release and a
    control-client bump, so it is proposed separately, like ledger scoping. If it
    lands first, `size_run` can read it and the tag can go.

16. **Record host speed, not vCPUs.** The protocol's Hardware section replaces the
    vCPU check. Each pod's speed is its median
    `time/collection/environment_step_and_encoding` per row over updates 2–6. That
    phase holds no network work, so it compares w64's pod with w256's directly. Add it
    to `Calibration` and to `SizeReading`, and name `host.cpu_model` from
    `training/run.json` as the second fact when the revision has it (b015d5). A
    difference is reported and does not void the reading, as before. The protocol rents
    L4, and A40 only if L4 stock is empty. Say that a gate reading depends on its host:
    a slow pod reads up to 1.8× above a fast one, so the calibration reading shows
    its step speed.

    Calibration's rows are hand-written in tests, so prove on one short real
    `execute_regime` update that every key it reads exists, `rows` included.

17. **Declare a pre-scoring host screen.** Over 54 hours a slow pod costs a size up to
    about 40% of its updates and buys it nothing, and on the retained L4 split a
    launch of two pods meets at least one slow host about half the time. Replacing one
    costs at most about USD1.50. Per-update cost is the brief's binding constraint, and
    the brief says to coordinate with b015d5, whose recommendation this is.

    - The evaluation driver starts after the screen, so no game exists when it acts.
    - A pod reads slow at more than 1.3× the lowest step speed among this launch's
      pods and calibration's. The bands imply about that ratio; it is a rough line,
      like the gate. Both pods reading slow against calibration counts.
    - The slower pod is cancelled, not deleted, and relaunched once per size. A size
      that reads slow twice keeps its second run. Costs come from the reserve.
    - `runs` takes a size parameter (a string, which `etude launch` can read), so one
      size relaunches alone with its own USD38 ceiling and no cohort.
    - `size_run` sets aside an ended run whose last update is below the first
      milestone, and analysis lists those runs, since every attempt is reported.
    - The protocol's "No automatic retry" stays. This is an operator step it declares
      before scoring. Any later retry stays a new version.

Implementation notes (2026-11-06):

- **A cancelled run never read as ended, so the screen's replacement could not work.**
  Etude's worker sees the stop file within 0.1 s and sends SIGTERM; training polls
  that file once a second and handled no signal, and `atexit` does not run on signal
  death. No `state.json` was written, so the run read `running` for good and the
  replacement made `size_run` raise, ending the watch loop. A real worker process
  reproduced it. `manabot.experiment.job`'s entry now turns SIGTERM into
  `KeyboardInterrupt` (game-error isolation catches `Exception`), so the existing
  `except BaseException` and close callback record `interrupted` / `failed`. This
  reaches every `TrainingJob`, and a hard deadline's SIGTERM too. A machine lost
  outright still records no end.
- **The screen reads each pod, below update 25.** An ended run past the first milestone
  stays its size's run, and w64 passes 25 before w256 passes 6 once w256 is slower than
  about 625 s/update at w64's 150 s. The protocol's "both pods pass update 6" would have
  cancelled a run that blocks evaluation. A test pins the refusal. Cost: a w64 pod slow
  only against a later w256 reading is kept and reported.
- Live metrics do reach the store (Etude seals and ships every few seconds), so the
  screen reads running pods.
- **The protocol's relaunch command was refused** (found reconciling, 2026-11-06). Etude
  rejects `--max-usd` or `--parallel` without `--cohort`, before any submission; the real
  0.2.14 CLI printed `several runs or cohort limits need --cohort NAME`. The job already
  carries USD38, so the command drops the flag, and without it the CLI goes on to resolve
  the rental. A test holds both launch blocks to that rule. Run ids are allocated per
  launch, so a relaunch repeats no id. A real relaunch stays unproven.

### Left alone

- **Launch timing against b015d5** stays a launch decision. It gets stronger: PR #435
  projects about 25% off an update and adds `TrainingRun.host`, so calibrating on a
  revision without it would measure a slower trainer. It is bit-identical on CPU only;
  CUDA is unchecked. Read against this change on 2026-11-06: it touches no protocol file,
  keeps `environment_step_and_encoding` as the parent series `host_speed` reads, and
  writes `host.cpu_model` where `_receipt_cpu_model` looks. A trial merge conflicts only
  in `scratch/questions.md`. Nothing here waits on it.
- **Heuristic saturation.** If a size nears 1.0 against the heuristic, that curve and
  the endpoint vs initialization stop resolving progress, and only head-to-heads
  remain. V10 stood at 9/32, so no evidence says this happens. A next protocol can add
  endpoint vs own earlier milestone; these checkpoints will exist.
- **`workers: 1`** is the next lever on the same constraint, and b015d5 calls lifting
  it a separate decision. It belongs in the final read's "what limits it next".

No open product choice. 15 repairs what direction 694f4190 requires. 16 corrects a
recorded fact. 17 stays inside the USD100, keeps every reading rule and precedes all
scoring, and it is reversible until launch.

### Counterexamples that would reopen this

- Tags do not survive a real delivered checkpoint: then 15 needs the Etude change
  first, and launch waits for it.
- Calibration shows the step phase varying by more than 1.3× between updates on one
  pod: the screen cannot read host class on TLA, so drop 17 and keep 16's reporting.
- A corrected one-size launch is still refused at a real rental: the screen needs a
  different shape, and 16 still stands.

## Sixth concept review (2026-11-06, headless)

Walked what the first five left out: the first rented evaluation pass, and the launch as
the controller sees it. Sizes, recipe, schedule, budget split and reading rules stay.
Calibration gains an evaluation rehearsal, and the protocol's launch steps get more exact.
Etude 0.2.14 source and the launcher database were read, and nothing was changed.

### Observed

- **The first real evaluation pass is already scored.** Calibration plays no game, so
  nothing runs `tla_frontier.play` on a rental before update 0 and 25 count. This is the
  one protocol step where a mistake cannot be repaired:
  - A game past its limit is stored as a delivered match with an invalid game, so even
    Etude's `--retry-failed` would not replay it, and the protocol forbids that anyway.
  - A match that raises stops its run (`etude/eval/match.py`) and leaves the ledger entry
    that ends the watch loop.
- **The 600 s limit has no rented reading.** It comes from one observation's latency on
  this Mac (`questions.md`). `initial-w256` puts w256 in both seats, so at 0.4 s a decision
  600 s allows 1,500 decisions on this Mac, and fewer on a loaded 28-worker rental, by an
  unmeasured factor. If rented games run too long, whole comparisons become bounds of
  roughly 0 to 1 at USD0.45 a pass. The second review's "counterexamples" name that
  outcome, but only a scored run would reveal it.
- **Other unmeasured items wait on the same pass:** `/dev/shm` with 28 workers, setup cost
  per rental, delivery to the store, and a tag on a really delivered checkpoint (the fifth
  review's first counterexample).
- **Mid-run reads have no stopping rule.** Direction 694f4190 makes scores visible for 54
  hours. The protocol fixes when training stops, but not what a read may change. Cancelling
  a size past the screen for its curve is optional stopping, and analysis would show only
  a failed run.

### Checked and closed

- **A cohort launch cannot split the sizes.** `_admit` boots every first-wave machine
  empty and commits runs and cohort together, and a refusal cleans up and raises, so no
  cohort row remains. Rerun the same command. `_life` gives each rental USD76/2 and 56 h
  plus setup.
- **A booted pod ends itself.** The bootstrap starts an independent guardian that deletes
  the machine at its deadline or budget, with no controller. "Rentals end only while a
  controller ticks" holds only for a pod that never reached its bootstrap. A laptop
  sleeping through a 54-hour run can delay records and eval handoffs but cannot overspend
  a booted pod.
- **Launch disk:** `l4` is 50 GB and `mtg252-a40` 80 GB, against the fourth review's
  roughly 9 GB. That launch check is closed.
- **Never-started pods have billed almost nothing.** 41 rented sessions have a receipt and
  no start. Most ended within a minute. `l4`'s five averaged 7 minutes and USD0.19 in total,
  and `mtg188-a6000`'s one took 15 minutes and USD0.10. The two on `fc6ffb-cpu32`, which has
  no startup limit, held 40–60 minutes with USD0 billed. Their cost so far is time, not
  dollars. Causes were not traced per session.
- **Main matches this checkout's base** (0 commits ahead, 2026-11-06). Task comments hold no
  new direction.

### Directions, implemented

18. **Rehearse evaluation inside calibration.** Give `tla_frontier_calibration` an
    `eval(state)` that plays calibration's own checkpoints through the same job shape as a
    frontier pass: `tla_frontier.play`, cpu rental, 28 workers, journal in `/dev/shm`,
    600 s limit, one pass ceiling of USD2. Share the job construction rather than copy it.
    - Two comparisons, 32 matches, so all 28 workers are busy: w256's last checkpoint vs
      its u0, the heaviest shape, and w256's last vs w64's last. Use the frozen pairs.
      These are untrained calibration checkpoints under calibration's experiment and an
      environment of their own, so no frontier slot, match key or score is touched. A size
      that stored only u0 cannot give a distinct pair; decide in implement, and it is
      probably a gate failure anyway.
    - Select runs by their regime tags (`tla-frontier-c5fba4-calibration-*`), which also
      proves a tag on a delivered checkpoint.
    - Extend the reader to show games, invalid games, median and slowest `seconds`, and the
      pass's cost.
    - Second launch gate, a rough line like the first: no killed game and the slowest
      under 300 s, half the limit. Otherwise `game_seconds` or the size changes in a new
      protocol version, before scoring. Say the reading is of a nearly fresh policy.
    - Protocol changes: calibration no longer "plays no comparison"; the sequence is
      calibration, rehearsal, gates, launch, screen, driver; the budget is USD2 calibration,
      USD2 rehearsal (about USD0.50 expected), USD3 cancelled pods, the rest setup and
      failures. A failed rehearsal pass leaves a ledger entry, and clearing it rehearses that
      recovery too. `_source()` omits the calibration file, so no frozen hash moves.
19. **Name the rentals.** A dud on a supply with no startup limit holds its whole window, and
    for training that means a size silently not training for up to 56 hours while the screen
    and the driver wait on it. `l4` has `startup_seconds` 900. `mtg252-a40`, the protocol's
    fallback, and `cpu32` have none. `07c490-cpu32` ran its three sessions under 900.
    The protocol names `l4` and its fallback, and adds this operator check before launch:
    every rented on-demand has a startup limit, set with `etude ondemand startup <name> 900`
    where absent. `cpu32` allows one machine, which fits one pass at a time. Run the driver
    under `caffeinate -is` if the always-on host is a laptop, and check the controller is
    ticking. A test holds the placeholders out of the launch blocks.
20. **Say what a pod that never started is.** It stores no run, trains nothing and is not
    the screen's one replacement, so relaunch that size with the `--size` command. The
    launcher's record of it is reported with the attempts. A refused cohort launch reruns
    unchanged. The Host screen's "lost outright" sentence stays for a pod that trained.
21. **Fix the stopping rule.** Reading progress changes nothing: past the host screen, no run
    is cancelled, relaunched or extended because of a score. A stop for cost, a dead run or
    any outside reason is a recorded decision that keeps its evidence and has no scale
    reading. This restates "fixed before scoring" and limits no decision Jack Heart makes
    about the money.

Implementation notes (2026-11-06):

- `etude eval` reads the calibration file: a test drives Etude's real `advance` over a
  stored calibration up to the rental, 32 new matches seating both runs. Adding `eval`
  makes the file an experiment file, which `etude launch` accepts, so the counterexample's
  separate file was not needed.
- `tla_frontier.evaluation_pass` is the one job constructor, and `paired_matchup` and
  `checkpoint_inference` are shared. `_source()`'s list is unchanged and the protocol JSON
  and regime hashes did not move. `tla_frontier.py` itself is in that list, so its file
  hash moved; nothing has played, so no retained evidence names the old one.
- Eval passes are stored runs of the experiment, so `read_calibration` and `read_screen`
  now skip `kind == "evaluation"`; otherwise each pass read as an empty training run.
- Only-u0 decision: `eval` raises for a w256 with nothing past update 0 (the first gate has
  failed), and returns `[]` while any calibration run is live. A w64 with only u0 still
  plays, since its pair stays distinct. A relaunched size uses its furthest run.
- The gate is stricter than the direction's "no killed game": all 128 games stored and valid
  and no failed match. `play` marks kills invalid, and a raising match leaves games missing.
- Pass cost is the launcher's `client.status(run).usd`, so `--rehearsal` needs the control
  client. A pass that delivered nothing has no stored run; `etude status` holds it.
- **Direction 19's command was refused.** Real 0.2.14 CLI: `etude ondemand startup X 900`
  fails with `Got unexpected extra argument(s) (900)`, and the option defaults to 0, which
  disables the limit. The protocol uses `--seconds 900`.
- Real proof: one rehearsal match at frozen sizes through `tla_frontier.play`, fresh w256 in
  both seats, on this Mac: 4 valid games, 10–17 s, 178–320 decisions, 0 dropped. That is
  about 0.05 s a decision, not the protocol's 0.4 s, which came from a one-observation
  reading. A fresh policy on this Mac says nothing of a loaded rental; the rehearsal does.

### Left alone

- **Deliverable checkpoints in Academy.** The Task asks for trained checkpoints, and they
  stay in the store. Loading through the real demo is a KR of the chapter, not of this
  brief. It is one check after the runs, and the final read should state whether it was done.
- **Inherited absolute settings at w256.** Gradient clip 0.267 and the learning rate are
  recorded as unchanged (`questions.md`). Adam mostly absorbs a constant clip scale. EMA does
  not act: `behavior` is `current-self` and selection is raw. The rate stays at its 1e-4 clamp
  until about update 2,300, so neither size sees the schedule decay. The final read should
  report clip rates if w256 underperforms.
- **Etude owners.** A `TrainingRun.job_name` and ledger scoping stay separate proposals.
- **`<store>` and `<frozen>`** stay placeholders in the commands; direction 19 names rentals.

No open product choice. 18 protects the accepted rule that failed games are never replayed,
precedes all scoring and fits the USD12 reserve inside the USD100. 19 and 20 make existing
steps exact, and 21 restates the brief's fixed-before-scoring rule for visible progress.

### Counterexamples that would reopen this

- A real rental cannot read calibration's checkpoints, though the local `advance` path can:
  run the rehearsal from a small experiment file of its own, keeping the same job shape.
- The rehearsal's slowest game passes 300 s: this is the design working. The size or limit
  changes in a new protocol version.
- A trained policy's games run far longer than a fresh one's, so scored games fail despite
  a clean rehearsal: the failed-leg bounds report it, and the next protocol's limit moves.

## Proposed wave memory (not written)

This Graph's steps leave `.lf/wave/` read-only, so these wait for a step that curates it.

- Inherited per-game time limits need rechecking when model size changes.
- `evaluation()` jobs need explicit setup and the `evaluation` extra to rent.
- A failed rented pass blocks later `etude eval` passes on that computer.
- A protocol's launch commands need checking against the real CLI, not only their job
  file: a locked client can fail on provider rows it cannot parse, and `--max-usd` without
  `--cohort` is refused, while tests of `runs()` stay green.
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
- `etude launch` with a cohort is all-or-nothing and can be rerun. A booted pod ends itself
  through its guardian. Only an unbooted pod needs the controller, and a supply with no
  `startup_seconds` then holds its whole window.
