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
2. Learning A/B `experiments/precision_ladder.py` / `.md` at w384l8 bucketed, frontier
   regime (`sampled_self_play`), float32 vs bf16 written explicitly, >= 2 seeds each, a few
   hundred updates, strength vs frozen anchor at matched updates + loss/clip/KL
   trajectories. Pre-register in the report before any scored run. USD 8 budget, cap 10.
   Not written yet.
3. If BF16 is rejected: `ladder_precision` returns float32 everywhere and the docs
   paragraph says what separated. The docs' learning-comparison sentence currently states
   the accepted outcome and is unverified until (2) runs.

Both runs wait for the L4: on 2026-11-12 MTG-318 (qa) and MTG-319 (implement) were still
running, and the Task says not to start until they have had their L4 time.

## Compress (2026-11-12)
Kept `_from_ladder` + `Execution.__eq__` + `authored_dump`: deriving "from ladder" from
the value would re-resolve explicit/saved float32 on widening, which the tests forbid.
Only cut: the docs table's `auto` row now points at the ladder paragraph. Docs-only, no rerun.
