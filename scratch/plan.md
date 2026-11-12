# MTG-312 bucketed shapes as `Execution.bucketed_shapes`

Concept (concept review, 2026-11-12): one `Execution` choice per stage, next to `device`, recorded in the regime and absent from the model. It covers every policy forward the stage's collector runs; evaluation, search and monitoring stay on the default path.

Built:
- `Execution.bucketed_shapes` (omitted when False); rejected on non-self-play stages, with `token_stream="pooled"`, and when a live continuation changes it.
- `ActorCritic(..., *, bucketed_shapes=False)` and `load_checkpoint(..., *, bucketed_shapes=False)` (`manabot/model/checkpoint.py`). `execute_regime` passes the stage's value to the learner (collection and the EMA `deepcopy` use that policy) and to the frozen-opponent load (`manabot/training/execution.py` near the `stage.opponent` branch). A pooled opponent checkpoint fails with ActorCritic's existing pooled error. Pretrained/initialization loads only copy weights into the bucketed agent.
- `manabot train --recipe local|baseline --bucketed-shapes`; rejected without a recipe.
- Docs: docs/observation-and-model.md "Bucketed padding (profiling)" (names the frozen opponent), docs/training.md.
- Tests: `tests/training/test_bucketed_execution.py` (4, including `test_frozen_opponent_runs_the_stage_setting`, which asserts both learner and opponent flags), `tests/model/test_bucketed_shapes.py`, `tests/training/test_baseline_recipe.py`.

Deleted: `MANABOT_BUCKETED_SHAPES`, `BUCKETED_SHAPES_ENV`, `bucketed_shapes_enabled` and its env test.

Receipt: `run.json` dumps the regime (`atomic_json(... run.model_dump)`), so an enabled stage records `execution.bucketed_shapes: true`; unset stages keep their bytes and hashes.

Last check: focused set (`tests/model/test_bucketed_shapes.py tests/training/test_bucketed_execution.py tests/training/test_baseline_recipe.py tests/training/test_precision.py tests/training/test_regimes.py tests/training/test_frozen_opponent.py`, notebook/artifacts/reports extras) 81 passed before the learner-only test was folded into the frozen-opponent test; `test_bucketed_execution.py` 4 passed after; ruff clean.

Remaining: publish. The default stays unbucketed; changing it waits on a separate learning check that weighs the 1.38x slower single-row collection calls.
