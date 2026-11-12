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
1. w256l8 step run with `experiments/cuda_precision.py` (bucketed on/off), ~USD 0.20.
   If BF16 learner and large collection calls are both >= 1.1x float32 at w256, set
   `BF16_MIN_HIDDEN_DIM = 256` and edit the threshold paragraph + regime comment.
2. Learning A/B: written, not launched. `experiments/precision_ladder.py` (arms
   float32/bf16 explicit, w384l8 bucketed, `sampled_self_play` Elemental, seeds 32401/32402,
   64 updates, horizon 64; `runs` and the heuristic matchups reuse `learning_signal`'s
   `training_runs` / `scheduled_matchups` (scored at 20, 40 and each run's end), plus
   bf16-vs-float32 head-to-heads at the last shared checkpoint)
   and `experiments/precision_ladder.md` (pre-registered paragraph, USD 10 cap). Launch:
   `etude launch experiments/precision_ladder.py --rent l4 --parallel 4 --max-usd 10
   --revision COMMIT`, then `etude eval ... --rent cpu32`. Check: `read_experiment` loads it,
   `runs('HEAD')` gives 4 runs at 3.0 h, `evaluation_matchups` on a fake state (one run
   stopped at 57, one unended) gives 11 heuristic + 2 head-to-heads (u64, u40); ruff clean.
3. Apply the A/B decision. Accepted: keep the docs sentence and fill the report. Rejected or
   unresolved: `ladder_precision` returns float32 everywhere and the docs paragraph says what
   separated. The docs' learning-comparison sentence (`docs/training.md` § Reduced precision,
   "In a learning comparison at w384l8 ... against the heuristic") states the accepted outcome
   and is unverified until (2) runs.

Both runs (step run and A/B) are deferred: they wait for the L4: the Task says not to start until MTG-318 and MTG-319 have had
their L4 time. Rechecked 2026-11-12 (second realign): MTG-318 at pr-publish with PR #493 open; MTG-319 at
implement (loop 1 pass 2). origin/main has nothing new since the last merge. PR #489 has landed
and the branch contains current main, so no upstream change affects the plan.

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
