# Assumptions taken (reversible) and open questions

- `experiments/` keep their deprecated import paths (each runs at its own
  commit; aliases resolve until 0.3.0). Exception: `experiments/cuda_precision.py`
  now imports `manabot.backends.torch.learner`, because `training.precision`
  was never released and has no alias.
- MTG-321's names are final (its steer f1d09a50): container metadata keys are
  `hypers`, `world_binding`, `architecture`, `source`, `converted_from`.
  The write-docs drafts of `policy/README.md` and `backends/README.md` used
  other names; both were replaced by MTG-321's files.
- `AdmittedRollout` stays in `manabot.training.updates.learner`;
  `legal_mask`/`policy_logs` stay in `manabot.model.policy_distribution`.
- Torch-importing modules outside the backend are listed by path in
  `tests/test_package_boundaries.py::_TORCH_NOT_YET_MOVED`, checked both ways
  so the list can only shrink.
- Import sorting: CI does not run ruff, and this checkout's ruff classifies
  `managym` as first-party, so a few touched files moved a managym import.

## Open

- Who extends `manabot/policy/prepare.py` and `checkpoint.py` to every torch
  configuration (FiLM, belief, history, pooled token stream, v10, categorical
  value, historical_mean, attention off) and the `.pt` extras? Assumed: this
  Task, after PR #495 lands, editing those files in place. Until then the
  torch forward keeps its inline lookups.
- Does `save_checkpoint` switch to writing `.safetensors` in this PR (docs say
  so) or keep `.pt` until Etude staging and recipes are checked against the
  container? Docs assume the switch; code still writes `.pt`.
