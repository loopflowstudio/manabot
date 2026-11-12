# MTG-324 plan

Changed docs: `docs/training.md` § Reduced precision (ladder, threshold, measurement links).

## Built
- `manabot.training.regime`: `Execution.precision` accepts `auto` (default);
  `BF16_MIN_HIDDEN_DIM = 384`, `ladder_precision(device, hidden_dim)`.
  `TrainingRegime.resolve_precision_ladder` replaces `auto` per stage (copies, never
  mutates) and marks it with the private `Execution._from_ladder`.
  `TrainingRegime.authored_dump()` restores `auto` for those stages; builders
  (`with_agent`/`with_capacity`, `move_learning` baseline, `sampled_self_play`) and the
  round trips in `experiments/batched_baseline.py`, `elemental_variants.py`,
  `training_throughput_sizes.py` use it so a width change re-resolves. `model_dump` stays
  the record (resolved value only). `Execution.__eq__` ignores the private flag so a
  regime equals its saved form (`test_builders` relies on it).
- Saved stage dicts whose execution omits `precision` (pre-#489) read as `float32`.
- Runtime reads `Execution.resolved_precision` (raises on unresolved `auto`).
- No named regime in `experiments/` changes precision (scanned all `regime_names()`).
- Tests: `tests/training/test_precision.py` ladder both sides, off-CUDA, serialization,
  explicit/saved kept, pre-field saved stage.
- Check: `pytest test_precision test_regimes test_builders test_training_throughput_sizes
  test_experiment_files` → 54 passed.

## Remaining (blocks landing)
Steers (Jack Heart, 2026-11-12): "Run the w256l8 step now"; then "Budget raised: Jack approved
about USD 30 for the learning A/B ("sure thats fine"). Run the full-length version, about 300
updates per arm ... Task cap is now USD 35 total."
Supplies: on-demands `mtg324-l4-step` (max 1, USD 0.70/h) and `mtg324-l4-ab` (max 4, USD 0.70/h);
`mtg324-l4` (0.60/h) is unused: L4s were listed at USD 0.69/h all afternoon of 2026-11-12.
Driving: `/tmp/mtg324-etude.sh wait RUN|COHORT` (0.2.15 overlay + nested doppler + certifi;
recreate it from the etude-launch-client memory if /tmp is gone). A queued or unbooted pod
needs a live controller loop; a booted one is kept to its window by its guardian.
1. Done: w256l8 step run `f30d26f9bc7a` ($0.29; two pods lost before start, ~$0.04):
   bucketed BF16 1.66x learner, 1.68x large collection -> `BF16_MIN_HIDDEN_DIM = 256` (10c5c8fe3);
   dated section in `experiments/cuda_precision.md`, evidence published. Machine terminated.
   Same host's w160 capture update took 352 s (264 s collection), so the w384 float32 pace is
   probably above the assumed 150 s, and *M* will likely be below 300.
2. Learning A/B, scored: cohort `mtg324-precision-ab-300c` at 783ccdba1, 300-update horizon,
   `--hours 10.9 --usd-per-run 7.85 --max-usd 32`, runs 07f84834e617 / aa9518f6fc4b (float32),
   2be568a65518 / 27416907ad6c (bf16). `etude launch` was refused (no 4 L4s at once), so it is
   queued; check `status mtg324-precision-ab-300c` and re-run `wait` if any are still queued.
   After all end: `etude terminate` each machine, check `runpodctl pod list`, then
   `etude eval experiments/precision_ladder.py --rent cpu32 --source r2://manabot/jobs` (USD 2),
   and fill the report's Answer/Cost. Spent so far ~USD 0.35 of the USD 35 Task cap.
3. Apply the A/B decision. Accepted: keep the docs sentence and fill the report. Rejected or
   unresolved: `ladder_precision` returns float32 everywhere and the docs paragraph says what
   separated. The docs' learning-comparison sentence (`docs/training.md` § Reduced precision)
   states the accepted outcome and is unverified until (2) runs.

## Compress (2026-11-12)
Kept `_from_ladder` + `Execution.__eq__` + `authored_dump`: deriving "from ladder" from
the value would re-resolve explicit/saved float32 on widening, which the tests forbid.
Cut: the docs table's `auto` row now points at the ladder paragraph.
Second pass: `precision_ladder.py` dropped its copies of `training_runs` and the heuristic
matchup builder (schedule 16/32/48/64 → learning_signal's 20/40 + end; report .md updated,
prediction unchanged). `pytest test_experiments test_precision` → 35 passed.

## Concept review (2026-11-12)
Concepts hold: one authoring value (`auto`) that the regime resolves at build time, so
every record names a real precision and resume never meets a moved threshold. Storing
`auto` and resolving at run time was the alternative; it would let a later threshold
change silently retrain saved regimes, so it stays rejected.
Direction for remaining step 2 (no code now): the 64-update length rests on an assumed
~150 s/update. The w256 step run measures float32 update time on the L4; before launching
the A/B, set its length to the most updates (up to 300) that keep both arms × both seeds
plus eval under the USD 10 cap, and edit the pre-registered paragraph in
`experiments/precision_ladder.md` to match before any scored run.
