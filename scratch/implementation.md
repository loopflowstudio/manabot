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
- First compress pass: 89 focused eval/boundary checks passed (73s).

2026-11-16 after main sync (3cb2aacbb5), Etude d9a6bc8; full commands in
f6a2660745:scratch/implementation.md:
- First focused run 69 failed / 32 passed: the stale native binary lacked
  `known_library`. Rebuilt the cp313 extension (81s) and installed it atomically.
- Root reassignment read native seeds before journaling intent; proposed setups
  now use synchronized cached seeds first. Two standalone test workers kept the
  removed inference argument. After both fixes the eleven collection, journal,
  disclosure, recipe, deck, loop, game-error, closure, layer-input, world and
  recording files: 233 passed at 15d69f186f and be4c9397b8. Lost-worker/timeout
  fixtures are bounded, not a native hang or load probe.
- Docs give the Etude `--config '{"training_game_logs": true}'` form.
- Concept-review cleanup (9336e7aa08): 129 passed, 1 skipped, 1 failed (163s);
  the failure was an assembled policy-import fixture bypassing validation, fixed
  in the fixture. Follow-up five files: 139 passed. Fresh-process imports of
  eval.duel and eval.monitor load neither training.regime nor training.records.

Not shown: CUDA batching, real Rust panic injection, native hang/load injection,
launched or provider runs, browser proof, a full-suite run.

2026-11-16 second compress pass (dead recovery surface, see plan.md):
- `uv run --extra dev --extra artifacts --extra reports --extra notebook pytest
  tests/training/test_update_backend_protocols.py test_experiments.py
  test_learning_dashboard.py test_self_play_loop.py test_numerical_health.py
  test_recipe_etude.py test_artifact_storage.py test_collection.py
  test_game_errors.py tests/test_package_boundaries.py tests/sim/test_recording.py
  tests/infra/test_ci_selection.py test_release.py
  tests/verify/test_learning_regression.py`: 269 passed, 2 skipped, 1 failed
  (101s). The failure predates the pass (same on 9336e7aa08); see questions.md.
- `etude train` of the baseline recipe, 1 update, temp ETUDE_HOME: run blocks sit
  in `track/baseline/baseline@197/`, `out/` beside them; `read_all` finds the
  run from the experiment or track directory, not from the run directory or
  `out`. report_recipe read `out/etude`, which never exists; it now reads the
  run's parent directory by run name.
