# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 by the first implement step,
reconciled three times and concept-reviewed twice the same day (headless). Nothing
has launched or spent money. Directions 1–9 from both reviews are implemented. Brief: Task c5fba4; direction comment 694f4190 in
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
- Evaluation passes are cpu32 rentals (28 workers, USD2, 1.75 h plus setup, three
  attempts for lost machines), prepared with the Rust toolchain and `evaluation` extra. Game limit
  600 s; the arena journals in `/dev/shm` and copies out, as `learning_signal` does.
- Control commands use the 0.2.14 client with nested Doppler and the `artifacts`
  and `evaluation` extras. Workers stay on locked 0.2.12.

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
- `etude launch` loads `runs` through `read_job_file`, so the calibration file
  needs no `eval`; only `read_experiment` does, and nothing reads it that way.

Estimates, not measurements: w64 about 1,300 updates at v10's 150 s/update (v9
inputs may cost more); w256 per-update GPU cost and CUDA memory are unknown. About 30 comparisons
(1,900 games, 17 CPU-hours) would cost about USD7 with per-rental setup (see
direction 4), within the USD12 cap. At
USD0.59/hour the 54-hour wall spends about USD33 per size, so USD38 is a ceiling,
not an expected cost.

Proposed wave memory (not written; this step leaves `.lf/wave/` read-only):
inherited per-game time limits must be rechecked when model size changes;
`evaluation()` jobs need explicit setup and the `evaluation` extra to rent; a
failed rented pass blocks later `etude eval` passes; a protocol's bare `uv run
etude` commands need checking against the launcher, since a locked client can
fail on provider rows it cannot parse while tests stay green.

## Delete — do not maintain

Nothing: new capability. `experiments/tla_pilot_v10.py` is executed evidence
and stays unchanged (its callback is mirrored, not shared, because its source
is hashed into retained environments).

## Remaining work

- Review of the protocol before freezing; re-freeze `runtime_world` if the card
  catalog moves before launch (`frozen()` rejects a drifted world).
- Freeze at a merged revision, run calibration (paid, USD2 ceiling), read it with
  the command in `experiments/tla_frontier_calibration.py`, then launch the
  protocol's training command (paid; needs an execution decision) and start its
  evaluation driver. `etude launch` refuses several runs without
  `--cohort`, and a cohort's default `--parallel 1` would train w256 only after
  w64's 54 hours, so both commands name a cohort and two machines.
- Unproven until the first real rental: the evaluation driver delivering to the
  training store, setup cost per rental, and calibration's stock worker on CUDA.
- Before reading a score, confirm from `etude status` that both runs had the
  same GPU model, and record both pods' vCPUs.
- A retry of a size is a new protocol version with its recorded decision and
  new regime ids, since `size_run` matches the old run's prefix.
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

## Second concept review (2026-11-05, headless)

Walked three interactions the first review left out: the final read when one
game failed, the 56-hour watch loop when a rental goes wrong, and what someone
does at hour 20 to see progress. Sizes, recipe, schedule and budget split stay.

### Directions, implemented

5. **A failed game narrows a reading, it does not erase it.** One unscored leg
   left its comparison's score None, permanently, since no retry is allowed. A
   probe on the `_outcomes` fixture gave `scale=None` and `learned=None` from 2
   failed games of 1,856. At v10's 0.095% failure rate, at least one of 30
   comparisons loses its score 84% of the time, so this is the normal path.
   `Comparison.result` holds `score`, `worst` and `best` once every pair has a
   receipt (each unscored leg as 0 and 1, same bootstrap indices); `learned`,
   strength and scale read it. `complete` stays False with an issue naming the count. Nothing is
   replayed or dropped: limits and kills fall on long games, so exclusion would
   not be neutral. A real Etude store confirms that a raised match keeps its seats
   and receipt, so it bounds rather than reading as missing.
6. **Evaluation passes survive machine loss.** Three attempts, Etude's default;
   training stays at one. Etude retries only `capacity_refused`, `setup_failed`
   and `machine_lost*`, resuming the same run and skipping stored matches. A
   raising match ends the run as `failed` and is not retried.
7. **The mid-run read is named.** The protocol carries the `etude report` command.
   A test plays a real store and renders `report()`.
8. **Calibration's launch gate is in the protocol:** 1,500 s/update or less.
9. **"Equal compute" stays true.** `scale` needs both finished with neither
   endpoint at the cap, otherwise an issue. The same-GPU operator check records
   both pods' vCPUs; a difference is reported, not voiding.

### Left alone

- The evaluation estimate of USD7 assumes no dud pods. A pod that never boots
  held its window in MTG-251, up to USD1.92 here if billed, which is unmeasured.
  The USD12 operator stop and the USD12 reserve cover it, as the protocol says.
- `report()`'s head-to-head selection holds both subjects, each the other's
  mirror (the render shows w256 at 0.75 and w64 at 0.25). Cosmetic; left as is.
- Both runs ending on one shared milestone make the endpoint head-to-head that
  milestone's match. Etude plays a shared match once and analysis reads it by key.
- Ledger scoping stays an Etude change for the launch decision. Direction 6
  removes its routine trigger, so it stays out of this PR.

No open product choice. Direction 5 keeps every accepted rule (failures stay
failed, no replacement, incomplete stays incomplete) and adds only a reading the
brief's "honest read" asks for.

### Counterexamples that would reopen this

- Failures are numerous enough that `best` and `worst` straddle every threshold:
  the answer is then a rules or capacity fix and a new protocol version, not a
  looser reading.
- A real retried pass replays a stored match (`repeated match` in analysis):
  direction 6's reading of Etude was wrong, and passes go back to one attempt.

Check: `uv run --extra dev --extra evaluation pytest tests/eval/test_tla_frontier.py -q` — 28 passed on 0.2.12 and the 0.2.14 overlay after the compress pass; `frozen()` passes, so `runtime_world` needed no re-freeze. Broader suites stay with gate.
