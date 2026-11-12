# MTG-312 bucketed shapes as `Execution.bucketed_shapes`

Built:
- `Execution.bucketed_shapes` (omitted when False); rejected on non-self-play stages, with `token_stream="pooled"`, and when a live continuation changes it.
- `ActorCritic(..., *, bucketed_shapes=False)`; `execute_regime` passes the stage's value to the learner (collection uses that same policy).
- `manabot train --recipe local|baseline --bucketed-shapes`; rejected without a recipe.
- Docs: docs/observation-and-model.md "Bucketed padding (profiling)", docs/training.md.

Deleted: `MANABOT_BUCKETED_SHAPES`, `BUCKETED_SHAPES_ENV`, `bucketed_shapes_enabled` and its env test.

Check: `pytest tests/model/test_bucketed_shapes.py tests/training/test_bucketed_execution.py tests/training/test_baseline_recipe.py tests/training/test_precision.py tests/training/test_regimes.py` (with notebook/artifacts/reports extras) passed; ruff clean.

Compress: `Stage.supported_workers` returns early for self-play; the execution test patches `ActorCritic` with a recording function; docs drop the retired env-var note. Same pytest set: 68 passed.
