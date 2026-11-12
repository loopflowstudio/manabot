# MTG-312 bucketed shapes as `Execution.bucketed_shapes`

Built:
- `Execution.bucketed_shapes` (omitted when False); rejected on non-self-play stages, with `token_stream="pooled"`, and when a live continuation changes it.
- `ActorCritic(..., *, bucketed_shapes=False)` and `load_checkpoint(..., *, bucketed_shapes=False)`; `execute_regime` passes the stage's value to the learner (collection and EMA use that policy) and to the frozen opponent load.
- `manabot train --recipe local|baseline --bucketed-shapes`; rejected without a recipe.
- Docs: docs/observation-and-model.md "Bucketed padding (profiling)", docs/training.md.

Deleted: `MANABOT_BUCKETED_SHAPES`, `BUCKETED_SHAPES_ENV`, `bucketed_shapes_enabled` and its env test.

Check: `pytest tests/model/test_bucketed_shapes.py tests/training/test_bucketed_execution.py tests/training/test_baseline_recipe.py tests/training/test_precision.py tests/training/test_regimes.py` (with notebook/artifacts/reports extras) passed; ruff clean.

Compress: `Stage.supported_workers` returns early for self-play; the execution test patches `ActorCritic` with a recording function; docs drop the retired env-var note. Same pytest set: 68 passed.

Receipt: `run.json` dumps the regime (`manabot/training/execution.py` `atomic_json(... run.model_dump)`), so an enabled stage's `execution.bucketed_shapes: true` is recorded there; unset stages keep their bytes and hashes.

Remaining: publish. The default stays unbucketed; changing it waits on a separate learning check that weighs the 1.38x slower single-row collection calls.

## Concept review (2026-11-12)

The concept holds: one `Execution` choice per stage, next to `device`, recorded in the regime and absent from the model. The name, CLI scope, validation and receipt need no change.

Next implement direction: `bucketed_shapes` covers every policy forward the stage's collector runs. Today the frozen opponent (`execution.py` `load_checkpoint(str(opponent_path))`) stays unbucketed while the learner and EMA (a `deepcopy`) are bucketed. As a result, "run the same regime once with each setting" compares a mixed path whenever `stage.opponent` is set, and the docs' "collection policies" claim is false for that case. Give `load_checkpoint` a keyword-only `bucketed_shapes: bool = False` passed to `ActorCritic`, pass `stage.execution.bucketed_shapes` for the frozen opponent, and add one test with a frozen opponent asserting both policies' flags. A pooled opponent checkpoint then fails with ActorCritic's existing pooled error, which is acceptable. Pretrained and initialization loads only copy weights into the bucketed agent and need nothing. Evaluation, search and monitoring stay on the default path as documented.

Frozen opponent done: `load_checkpoint` takes `bucketed_shapes`; the frozen-opponent load passes the stage's value; `test_frozen_opponent_runs_the_stage_setting` asserts both flags. Focused set plus `tests/training/test_frozen_opponent.py`: 81 passed; ruff clean.

Compress 2: dropped the learner-only execution test, which the frozen-opponent test covers (it asserts the learner's flag too); `test_bucketed_execution.py`: 4 passed.
