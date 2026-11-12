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
Supplies: on-demands `mtg324-l4` (max 4, USD 0.60/h ceiling, the A/B) and `mtg324-l4-step`
(max 1, USD 0.70/h, the step run only: L4 listed at USD 0.69/h on 2026-11-12 18:52 UTC).
Driving both: `/tmp/mtg324-etude.sh wait RUN COHORT` (0.2.15 overlay + nested doppler + certifi);
an unbooted pod needs a live controller loop, or it is lost (first step run d9976d77ffb1 was).
1. w256l8 step run: `cuda_precision.py --sizes w256l8` at 35764ee13, run 5fcf338bac39.
   If BF16 learner and large collection calls are both >= 1.1x float32 at w256 (bucketed), set
   `BF16_MIN_HIDDEN_DIM = 256` and edit the threshold paragraph + regime comment; record the run
   as a dated line in `experiments/cuda_precision.md`. Then `etude terminate` its machine.
2. Learning A/B, scored: 300 updates, cohort `mtg324-precision-ab-300` at 3efb3bdf1, runs
   5713f9936db8 / 51a6ea46563e (float32) and a504bd8e523f / a9a39029b2dd (bf16), USD 7.85 each,
   `--max-usd 32`; eval pass USD 2 on cpu32 after. The 64-update cohort was cancelled unrented.
   Float32 pace at w384l8 on the L4 is assumed 150 s; slower runs stop at the watchdog and the
   comparison uses *M* (pre-registered in `experiments/precision_ladder.md`).
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
