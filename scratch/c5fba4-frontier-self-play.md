# c5fba4 — frontier self-play on TLA trophy decks at two sizes

Status: protocol v1 drafted 2026-11-05 by the first implement step and
reconciled the same day (headless, no design review yet). Nothing has launched
or spent money. Brief: Task c5fba4; direction comment 694f4190 in
[direction.md](direction.md). Contract: `experiments/protocols/tla-frontier-c5fba4-v1.{md,json}`;
runner `experiments/tla_frontier.py`; analysis
`experiments/tla_frontier_analysis.py`; tests `tests/eval/test_tla_frontier.py`.

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

Estimates, not measurements: w64 about 1,300 updates at v10's 150 s/update (v9
inputs may cost more); w256 per-update GPU cost and CUDA memory are unknown, so
the earlier "several hundred updates" claim was removed. About 30 comparisons
(1,900 games, 17 CPU-hours) would cost USD1–4, within the USD12 cap.

Proposed wave memory (not written; this step leaves `.lf/wave/` read-only):
inherited per-game time limits must be rechecked when model size changes;
`evaluation()` jobs need explicit setup and the `evaluation` extra to rent.

## Delete — do not maintain

Nothing: new capability. `experiments/tla_pilot_v10.py` is executed evidence
and stays unchanged (its callback is mirrored, not shared, because its source
is hashed into retained environments).

## Remaining work

- Review of the protocol before freezing; re-freeze `runtime_world` if the card
  catalog moves before launch (`frozen()` rejects a drifted world).
- Freeze at a merged revision, then launch `runs(revision)` (paid; needs an
  execution decision) and drive `eval` passes during training.
- Launch preflight: read w256's first per-update cost and CUDA memory on the
  chosen GPU before committing its full allocation. Neither is measured.
- `size_run` refuses a second run per size, so a retry also needs a runner
  amendment alongside its recorded decision.
- The final report is written after the runs. `tla_frontier_analysis` gives its
  numbers: whole-pair intervals, per-size learning, the scale score, hours per
  checkpoint and every failed, missing or unreached slot. `report()` stays
  descriptive curves. A repeated match key fails analysis loudly, inherited from
  `pair_result`; a failed match must not be replayed by a later `eval` pass, and
  whether Etude replays one is unverified.

## Decisions needed

- **Launch timing against b015d5.** That Task is still implementing (uncommitted
  changes in learner and rollout code). Choose: wait for its merge, so both
  sizes get more updates per dollar, or freeze now at the current cost.
  Unresolved; freezing after it merges needs no protocol change unless the world
  moves.
- **Head-to-head past w256's reach.** Direction 694f4190 asks for head-to-heads
  at each milestone at matched updates/compute. Equal-update games stop at
  w256's last milestone, and equal compute appears only at the endpoints.
  Choose: accept that, or expose the retained `wall_ms` on Etude's `Checkpoint`
  (an Etude change, no manabot training change) for equal-compute pairs during
  training. Unresolved; the wall times are retained, so pairs can be added after
  launch without new training.

Check: `uv run --extra dev --extra evaluation pytest tests/eval/test_tla_frontier.py tests/eval/test_tla_pilot.py -q` — 13 and 50 passed (run separately).
