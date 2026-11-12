# MTG-328 plan

Docs written for the intended behavior (read these before implementing):
- docs/observation-and-model.md — "Bucketed padding" and "Live-token attention"
- docs/training.md — Run training (`--padded-shapes`), Stage operations (`auto`)
- docs/evaluation.md — Shared inference, "Forward layout"

Built (implement pass 1):
- `Execution.bucketed_shapes`/`live_tokens`: `bool | "auto"` (default auto),
  off omitted; `live_tokens=True` off CUDA fails; `resolved_*` properties.
- `TrainingRegime`: saved stages omitting them read False; `resolve_forward_layout`
  (on for CUDA train_self_play outside pooled) before `references`;
  `authored_dump` restores auto; builders/experiments/study rebuild with it.
- CLI/recipes: `--padded-shapes`; `--bucketed-shapes` warns (removed 0.3.0).
- `EvalAllocation.bucketed_shapes`/`live_tokens` resolve auto from device,
  outside `EvalCompute`; serving, policy_search, value players and range
  likelihood load CUDA policies with the layout; `load_checkpoint` keeps pooled
  checkpoints padded.

Remaining:
- Measure one evaluation pass at w384l8 on an L4 before/after (<USD3), report
  under experiments/ in the science shape; state measured CPU live-token cost
  in the PR.
