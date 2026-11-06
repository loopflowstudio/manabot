# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 by the first implement step,
reconciled twice and concept-reviewed the same day (headless). Nothing has launched
or spent money. Concept-review directions 1–4 are implemented. Brief: Task c5fba4; direction comment 694f4190 in
[direction.md](direction.md). Contract: `experiments/protocols/tla-frontier-c5fba4-v1.{md,json}`;
runner `experiments/tla_frontier.py`; analysis
`experiments/tla_frontier_analysis.py`; calibration
`experiments/tla_frontier_calibration.py`; tests `tests/eval/test_tla_frontier.py`.

## Design (as implemented)

- Two regimes, identical to MTG-213 v10 except: input v9 / 512 facts, wall
  54 h, update cap 3,000. One regime file (w64) is hash-pinned; w256 is derived
  in `frozen()` from the protocol's `large` size (width 256, 4 layers, 8 heads;
  15.0× compute parameters of w64 = 648k → 9.71M, hashed tables excluded).
  Static declarations of those bytes live in tests, not `frozen()`.
- Damped learning rule kept unchanged (absolute iteration schedule, so the
  update cap does not alter learning rates).
- Wall-bounded runs → equal compute per size; shared update milestones give
  equal-update head-to-heads while both are running; endpoint head-to-head is
  the equal-compute comparison. Endpoint vs own initialization shows learning.
- 16 fixed pairs × 4 legs per comparison; training USD38 / 56 h per size;
  cumulative evaluation USD12; USD12 reserve under the USD100 cap.
- Both sizes share one GPU model, chosen at launch and recorded before scoring.
  USD38 / 56 h admits at most USD0.67/hour: L4 or A40 at USD0.59, not RTX 5090.
- Evaluation passes are cpu32 rentals (28 workers, USD2, 1.75 h plus setup, one
  attempt), prepared with the Rust toolchain and `evaluation` extra. Game limit
  600 s.

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
  any launched pass ends unsuccessfully, and nothing clears the entry. With
  `max_attempts=1` and a 1.75 h deadline, one failed rental stops evaluation
  until someone intervenes. This is untested against a real rental.
- The GPU model is held in the launcher's session record, shown by `etude
  status`, not in the training store, so analysis cannot check the same-model
  rule. It stays an operator check.

Estimates, not measurements: w64 about 1,300 updates at v10's 150 s/update (v9
inputs may cost more); w256 per-update GPU cost and CUDA memory are unknown, so
the earlier "several hundred updates" claim was removed. About 30 comparisons
(1,900 games, 17 CPU-hours) would cost about USD7 with per-rental setup (see
direction 4), within the USD12 cap. At
USD0.59/hour the 54-hour wall spends about USD33 per size, so USD38 is a ceiling,
not an expected cost.

Proposed wave memory (not written; this step leaves `.lf/wave/` read-only):
inherited per-game time limits must be rechecked when model size changes;
`evaluation()` jobs need explicit setup and the `evaluation` extra to rent; a
failed rented pass blocks later `etude eval` passes.

## Delete — do not maintain

Nothing: new capability. `experiments/tla_pilot_v10.py` is executed evidence
and stays unchanged (its callback is mirrored, not shared, because its source
is hashed into retained environments).

## Remaining work

- Review of the protocol before freezing; re-freeze `runtime_world` if the card
  catalog moves before launch (`frozen()` rejects a drifted world).
- Freeze at a merged revision, run calibration (paid, USD2 ceiling), read it with
  `uv run python -m experiments.tla_frontier_calibration --source <store>`, then
  launch the protocol's training command (paid; needs an execution decision) and
  start its evaluation driver. `etude launch` refuses several runs without
  `--cohort`, and a cohort's default `--parallel 1` would train w256 only after
  w64's 54 hours, so both commands name a cohort and two machines.
- Before reading a score, confirm from `etude status` that both runs had the
  same GPU model.
- A retry of a size is a new protocol version with its recorded decision and
  new regime ids, since `size_run` matches the old run's prefix.
- The final report is written after the runs. `tla_frontier_analysis` gives its
  numbers: whole-pair intervals, per-size learning, the scale score, hours per
  checkpoint and every failed, missing or unreached slot. `report()` stays
  descriptive curves. A repeated match key fails analysis loudly, inherited from
  `pair_result`.

## Concept review (2026-11-05, headless)

Reviewed against the brief, direction 694f4190 and the implemented change.
Outcome: the design changed; nothing is open for a person. Directions 1–4 are
implemented. Nothing in them changes the sizes, recipe, schedule or budget split.

The experience the brief asks for: launch two sizes, then watch each milestone
appear as curves against the heuristic and each other, without further action
unless something fails; finally read where strength stands. Walking that path
through the code:

**Normal path.** Breaks at the first pass. `etude eval FILE` names the experiment
for the file: `read_experiment` gives `tla_frontier`, and `advance` calls
`observe(source, "tla_frontier")`. Runs track under `EXPERIMENT =
"tla_frontier_c5fba4"` (confirmed by running both). So `eval` sees no runs,
returns `[]`, and every pass reports nothing to play and exits cleanly while
USD76 of training proceeds. Tests call `eval` with a hand-built state, so they
pass. `tla_pilot_v10.py` avoided this: its stem equals its name and its `eval`
rejects a foreign state.

**Failure path.** A run that fails at hour 30 is `failed`, so `run.final` is None.
Its endpoint stays None for good: the endpoint comparisons never play and
analysis reports no learning or strength reading, although its checkpoints are
stored. The size's spend then buys nothing readable. The brief asks for an honest
read of where strength stands, so reach survives failure; equal compute does not.

**Recovery path.** The preflight in the previous plan has no way to run. Launching
w256 for real and cancelling leaves a run that `size_run` refuses to share a
prefix with, and a fix means new protocol bytes anyway. Reading cost and memory
is a separate job, not a first attempt at the frozen run.

Checked and left unchanged:

- Damped updates ignore `progress`, so `schedule_clock: iteration_fraction` does
  not make the update cap a schedule input; the protocol's claim holds.
- Match keys exclude source versions, so a later commit cannot replay games.
  It would change `seats.source`, which analysis flags as a mixed cohort, so
  every pass runs from the frozen revision.
- Wall stops record the final raw checkpoint through `experiment.checkpoint`, and
  the run counts as a launcher success, so `run.final` is the real endpoint.

### Implemented direction

All four are in the change, as written below, with these additions found while
implementing them:

- `etude launch` names an experiment file's runs for its stem and `track` raises
  `experiment_differs_from_launch` on another name, so the old `EXPERIMENT` would
  also have failed each worker at start. The seam test reads the real job file.
- No run record held a CUDA peak, so `execute_regime` now logs
  `memory/peak_{allocated,reserved}_cuda_bytes` on each CUDA update. Calibration
  then uses the stock worker and reads everything from the store; the frontier
  runs carry the same rows.
- Calibration is five updates, 3,600 s wall, at most 1.2 h and USD1 per size,
  with the numerics gate on as in the real runs.

1. **One experiment name.** `EXPERIMENT = Path(__file__).stem`, the convention of
   `elemental.py` and `seat_comparison.py` and the contract in
   `manabot/experiment/job.py`. Reject a foreign `state.experiment` in `eval`, as
   v10 does. Add a test that `read_experiment(...).name == EXPERIMENT` and that a
   store-backed `observe` under that name finds a `TrainingJob`-named run, so the
   proof crosses the seam the hand-built state skipped.
2. **Reach survives failure.** `endpoint` is the last stored checkpoint of any
   ended run, not only a finished one, so the endpoint comparisons play. Analysis
   keeps `state` beside the endpoint, keeps the learning and strength readings,
   and sets `scale` to None with an issue unless both runs finished: the
   comparison is still reported, but not as equal compute. Update the protocol's
   Reading section to match, and test a failed size.
3. **Calibration is its own job.** Add a short calibration entry point, under its
   own experiment and regime ids so `size_run` never sees it: each size for a few
   updates on the chosen GPU, a USD2 total ceiling inside the reserve, reading
   seconds per update and CUDA peak. Run it after merge, before `runs()`. If w256
   cannot fit or leaves too few updates, the size changes in a new protocol
   version, which is still before any scoring.
4. **Name the evaluation driver in the protocol.** Regular evaluation is one
   command, not per-milestone operator work: `etude eval
   experiments/tla_frontier.py --source <store> --rent <cpu32> --watch 14400`,
   from a checkout of the frozen revision, on an always-on host beside the
   controller. A four-hour period batches both sizes' new milestones. Every
   milestone still gets played, within a period of arriving. Estimate: about 15
   rentals at USD0.45 (setup included, unmeasured), about USD7. One rental per
   comparison would approach USD10–13, because each pays setup; the earlier
   USD1–4 estimate left setup out. The USD12 cap stays an operator check: read
   spend after each pass and stop the loop when it reaches the cap.

The protocol also records recovery from a failed pass. `launched_pass_failed`
ends the watch loop by design ("rerun it after looking"), but the ledger has no
way to clear an entry, and it is one file for every experiment on the computer
(today it holds `elemental-lambda-v1-6860cc47ef30`; Etude main matches). So
another experiment's failure can stop this one. Before starting the loop, check
that entry's status. After a failed pass, look, then remove its entry from
`~/.etude/experiments/launched.json` and restart the loop. The real fix belongs in
Etude (scope the ledger per experiment, add an acknowledge step) and is worth
a separate Etude change; it is not required for this launch.

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

Check: `uv run --extra dev --extra evaluation pytest tests/eval/test_tla_frontier.py -q` — 21 passed; `frozen()` passes, so `runtime_world` needed no re-freeze.
