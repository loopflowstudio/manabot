# MTG-329 evidence

## Evaluation errors — 2026-11-16

- `uv run --extra dev --extra artifacts pytest tests/eval/test_elemental_play.py tests/eval/test_etude_agents.py::test_native_failures_remain_rows_across_etudes_game_process tests/eval/test_etude_agents.py::test_replay_only_retains_native_failures -q`: 24 passed (37.31s). Injected native AgentError/PyO3-shaped panic retain score-null games and continue the group through Etude's real game process; Python RuntimeError/ValueError/KeyError propagate from play and replay. All-invalid Elemental callbacks fail with retained rows/journals, and successful retries preserve earlier evidence, for local and launched-style output paths.
- Removed the redundant RetryRequested override: transport retry now follows
  the same propagation path as other non-native errors. No real Rust panic,
  paid run or CLI smoke was injected in this slice. Nightly remains for gate.

Prior commands and evidence: bf18f7c3e9:scratch/implementation.md, including
retired-loop bitwise parity, stop/resume, native-error retry journals, installed
3.12/3.13 wheels, isolated integration checks, world/monitor checks and the
choice-corpus fix. No paid/provider/service run or landing.

## Concept follow-up — 2026-11-16

- `uv run --extra dev --extra artifacts pytest
  tests/test_package_boundaries.py::test_live_training_and_evaluation_without_saved_regime_readers
  tests/training/test_outcome_tables.py tests/training/test_phase_timing.py -q`:
  11 passed (6.06s). Fresh imports of the live loop, Torch collection/runtime
  and duel/monitor reject training.regime and training.records.
- `uv run --extra dev --extra artifacts pytest
  tests/training/test_self_play_loop.py::test_resume_keeps_adam_ema_counts_and_immutable_policies
  tests/training/test_self_play.py::test_windows_pool_counts_and_old_reports_remain_unavailable
  tests/training/test_monitoring.py tests/training/test_recipe_etude.py
  tests/training/test_regimes.py -q`: 14 passed (18.55s), one expected v9
  deprecation warning. Initial collection found a missed multiline report
  import of Updates/read_updates/stage_rows; moved it to reports/update_metrics.
- With `ETUDE_HOME=/tmp/manabot-local-policy.XlGkFY`,
  `uv run etude train manabot/recipes/local.py --json` returned
  `{"run":"local","experiment":"local","outcome":"finished"}`.
  The actual final EMA policy is
  `track/local/out/policies/00000002-ema.safetensors` under that home;
  raw/EMA policies at updates 0, 1 and 2 and checkpoint records all exist.
  The guide now uses the corresponding default-home path. Fresh importing
  recipes.local also loaded neither saved-regime module.
- Ruff check/format pass for all changed Python files; `git diff --check` passes.
  `lf context` reports default 16000-token/131072-byte targets, no usage entries;
  protected notes retained, older evidence condensed by git reference.

- Compress, 2026-11-16: `docs/training.md` names
  `manabot.reports.update_metrics.read_updates` (was the training path), and
  `Collector.operation` carries its own comment. `uv run --extra dev --extra
  artifacts pytest tests/training/test_phase_timing.py -q`: 5 passed (2.30s).

Not shown: CUDA, real Rust panic or native hang injection, launched/provider
runs, browser proof, full suite, automatic policy→sampler scheduling. The full
nightly baseline remains a gate obligation; publication stays with the caller.

- QA fourth pass, 2026-11-16 (scratch/qa.html): the evaluation, training and
  workflows guides run as written on a fresh ETUDE_HOME. Fixed snapshot
  stranding an empty output on a wrong source (opens the source first);
  `pytest tests/training/test_learning_dashboard.py -q -k snapshot`: 1 passed.
