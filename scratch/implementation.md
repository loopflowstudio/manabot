# MTG-329 evidence

Full transcripts and commands: 4de3a76ba0:scratch/implementation.md,
382c653ec8:scratch/implementation.md. No paid/provider/service run or landing.

2026-11-14 (earlier pins/native): seed 197 two-update PPO and damped-policy
raw/EMA, 87 tensors each, bitwise equal to retired 6ce618b8ad. Real
SIGINT/SIGTERM saved at update 1; resume finished at 3, rows [1,2,3]. Old
state.pt parity does not prove the revised save API.

2026-11-16 before main sync, Etude d9a6bc8, CPU only:
- uv lock and locked sync pass; existing cp313 native extension used.
- test_etude_agents plus native card resume/partial-failure retry: 9 passed
  (54s). Across real Etude game processes, public AgentError and a PyO3-shaped
  PanicException each leave game one invalid, score null, a failure journal and
  seven following games clean. Injected clock expiry: eight timeouts; one-Command
  cap: eight command_cap rows, scores null. RetryRequested escapes to Etude with
  its journal. Loader batches Prepared rows in order with masked logits (float
  tolerance) and repeats a seeded sample. Etude omits an all-null scalar column:
  absence means unavailable score, never zero.
- Card/world/monitor/head-to-head: 87 passed; the SDK filters completed matches
  before scheduling (skipped=0), so resume tests assert no replayed match.
- Isolated artifacts/SDK/source/recipe command: 12 passed. Self-play, minimal,
  recipe and belief provenance incl. save/resume and EMA pass.
- Installed wheels outside the checkout, CPython 3.12.12 and 3.13.15, Torch
  2.15.0, Etude 0.4.0 at d9: 2-update training, policy reload, CLI, Academy.
- A non-pickleable agent closure once stranded fixture cleanup; the bridge now
  sends only the synthetic match and survives EOF. Resume configs use JSON
  model dumps. No frozen evidence rewritten.
- Compress pass: `pytest tests/eval/test_etude_agents.py
  test_evaluation_policy_inputs.py test_retained_compute.py
  test_eval_performance.py test_experiment_files.py tests/test_package_boundaries.py
  tests/training/test_head_to_head.py test_monitor_evaluation.py`: 89 passed
  (73s); ruff clean on changed files.

2026-11-16 after main sync, merged 3cb2aacbb5, Etude d9a6bc8:
- Initial focused run: 69 failed / 32 passed (8.73s); stale native binary lacked
  `known_library`. Rebuilt with `uv run --extra play maturin build --release -i
  .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` (81s), then
  atomically installed the matching cp313 extension.
- `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest
  tests/training/test_collection.py tests/training/test_training_game_journals.py
  tests/env/test_decklist_disclosure.py tests/training/test_recipe_etude.py -q
  --tb=short`: 100 passed / 1 failed (46.98s). Root reassignment read native
  seeds before journaling intent; proposed setups now use synchronized cached
  seeds before that read. Reran `test_training_game_journals.py` with the same
  flags: 65 passed (28.37s), including interruption before reassignment reads.
- Same command flags with `tests/training/test_training_decks.py
  tests/training/test_self_play_loop.py tests/training/test_game_errors.py
  tests/training/test_batch_closure.py tests/model/test_layer_input.py
  tests/semantic/test_elemental_world.py tests/env/test_engine_recording.py`:
  129 passed / 3 failed (27.27s). Two standalone test workers retained the
  removed inference argument. Removed it; reran `test_game_errors.py` with the
  same flags: 32 passed (23.90s). Includes bounded lost-worker/timeout fixtures,
  not a native hang or load probe. Total: 233 distinct focused checks pass.
- Updated stalled-training docs to Etude config/path/phase semantics.
  `uv run --extra dev ruff check` and `ruff format --check` on recording and
  game-error fixtures, plus `git diff --check`: pass.

- Merged state, rebuilt cp313 extension: the eleven files of the two commands
  above in one run, same flags: 233 passed at 15d69f186f (39.73s) and again at
  be4c9397b8 (48.18s; only docs changed between). Those docs (README,
  engine-python.md, .lf/skills/qa.md) now give the Etude
  `--config '{"training_game_logs": true}'` form instead of the removed
  `manabot train --training-game-logs`. `scripts/check_loopflow_workflows.py`: valid.

Not shown: CUDA batching, real Rust panic injection, native hang/load injection,
launched or provider runs, browser proof, a full-suite run.

2026-11-16 concept-review naming/ownership cleanup:
- `uv run --extra dev --extra artifacts --extra reports --extra notebook pytest tests/training/test_builders.py tests/training/test_regimes.py tests/training/test_monitor_evaluation.py tests/training/test_monitoring.py tests/training/test_experiment_report.py tests/training/test_policy_import.py tests/training/test_experiments.py tests/training/test_precision.py tests/model/test_recent_events.py tests/training/test_recipe_etude.py tests/training/test_self_play_loop.py tests/belief/test_learning.py::test_real_population_learning_is_held_out_and_beats_p0 -q --tb=short`: 129 passed, 1 skipped, 1 failed (162.83s). The failure was an assembled fixture bypassing validation: retained automatic execution fields resolved during round-trip. Validate the assembled fixture before its equality/rejection checks; production behavior stays unchanged.
- `uv run --extra dev --extra artifacts --extra reports --extra notebook pytest tests/training/test_policy_import.py tests/training/test_learning_dashboard.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/test_package_boundaries.py -q --tb=short`: 139 passed (11.86s), including the corrected fixture and retained SQLite snapshot reader.
- Fresh-process imports of eval.duel and eval.monitor load neither training.regime nor training.records. Torch collection still reaches SelectionGameSpec through selection_data; moving that separate schema was outside the six directed fixes.
- Ruff check/format on changed and moved Python files, and `git diff --check`: pass. No runtime keys, stored bytes or scheduling behavior changed. Same-seed scheduling remains Etude's question; gate/publication are caller steps.
