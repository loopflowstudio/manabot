# MTG-328 plan

Docs written for the intended behavior (read these before implementing):
- docs/observation-and-model.md — "Bucketed padding" and "Live-token attention"
- docs/training.md — Run training (`--padded-shapes`), Stage operations (`auto`)
- docs/evaluation.md — Shared inference, "Forward layout"

Built (implement pass 1):
- `Execution.bucketed_shapes`/`live_tokens`: `bool | "auto"` (default auto),
  off omitted; `live_tokens=True` off CUDA fails; `resolved_*` properties.
- `TrainingRegime`: saved stages omitting them read False; `resolve_forward_layout`
  (on for CUDA train_self_play) before `references`;
  `authored_dump` restores auto; builders/experiments/study rebuild with it.
- CLI/recipes: `--padded-shapes`; `--bucketed-shapes` warns (removed 0.3.0).
- `EvalAllocation.bucketed_shapes`/`live_tokens` resolve auto from device,
  outside `EvalCompute`; serving, policy_search, value players and range
  likelihood load CUDA policies with the layout. Merged main after #495 and
  #498 (pooled stream removed; no pooled handling remains here).

Remaining:
- Measure one evaluation pass at w384l8 on an L4 before/after (<USD3), report
  under experiments/ in the science shape; state measured CPU live-token cost
  in the PR.

Check: focused tests (test_bucketed_execution, test_live_tokens, test_batched_eval,
baseline-recipe CLI, tracked self-play run) pass after merging main at 974ad9861.

## Delete — do not maintain

Removed `Execution._from_auto` and its custom `__eq__`: Pydantic's
`model_fields_set` distinguishes resolved defaults from explicit writes,
including edits after resolution. `authored_dump` restores only those defaults.
Folded `_read_saved_layout` and `_read_saved_self_play` into `_read_saved_stage`;
one reader preserves saved layout, seat and closure defaults. No pending cuts.

Check: `uv run --extra dev --extra artifacts pytest tests/training/test_bucketed_execution.py tests/training/test_builders.py -q` — 23 passed.
Check: `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_regimes.py -q` — 18 passed; focused Ruff lint/format and `git diff --check` pass.
