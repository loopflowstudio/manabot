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

Width rule: `auto` is on for CUDA `train_self_play` at every width, with no
threshold. The brief asked to measure below the paying width first; the
MTG-319 table (experiments/live_token_attention.md) already covers w160l8:
both on is 2.3 s learner + 2.9 s collection per pass, against 2.9 + 2.4 s for
live tokens over full tables and 7.8 + 9.2 s padded, so both-on is fastest in
total at both measured widths. The 0.5 s collection cost at w160l8 is
documented in docs/observation-and-model.md as the known small-width cost.

Remaining:
- Measure one evaluation pass at w384l8 on an L4, padded vs resolved layout
  (<USD3), report under experiments/ in the science shape. L4 is shared with
  MTG-324's pending runs; check it is free first.
- Measure CPU live-token cost (the per-observation fallback) at w160l8 on a
  small fixture and state it in the PR; outcome 1 requires the number.
- MTG-324 has not landed (still iterating on its learning A/B; main has no
  `precision="auto"`). When it lands, merge main and combine its ladder with
  `resolve_forward_layout`/`authored_dump`/`_read_saved_stage` (questions.md 5).
- MTG-325 (moving) is moving the torch forward to `manabot/policy` +
  `manabot/backends/torch` and ActorCritic over `prepare.Prepared`. Its merge
  will touch where `bucketed_shapes`/`live_tokens` enter the forward; keep one
  path (constraint: "switches reach the backend forward through one path").

## Delete — do not maintain

Removed `Execution._from_auto` and its custom `__eq__`: Pydantic's
`model_fields_set` distinguishes resolved defaults from explicit writes,
including edits after resolution. `authored_dump` restores only those defaults.
Folded `_read_saved_layout` and `_read_saved_self_play` into `_read_saved_stage`;
one reader preserves saved layout, seat and closure defaults. No pending cuts.

Check: `uv run --extra dev --extra artifacts pytest tests/training/test_bucketed_execution.py tests/training/test_builders.py -q` — 23 passed.
Check: `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_regimes.py -q` — 18 passed; focused Ruff lint/format and `git diff --check` pass.
