# MTG-277 gate — 2026-11-13

Reviewed the branch against origin/main: native heuristic strategy/knowledge,
per-seat Env/VectorEnv settlement, Python bindings and stubs, receipt/replay
mapping, registration identities, packaging and public guides. No strategy,
world, input-layout or information-boundary change is introduced by gate.

The repository CI selector selects all jobs for native/package changes; it
has no file-level local runner. Local checks target the modified code and
its direct consumers. Full Rust/workflow coverage is reused from CI run
[43540649606](https://github.com/loopflowstudio/manabot/actions/runs/43540649606),
HEAD fc22c9eb5a. No whole package/workspace test suite was launched locally.

Commands and results (one run per content, except changed inputs):

- `git diff --name-only -z origin/main...HEAD | bash .github/ci/select-tests.sh` — all nine CI job classes selected.
- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci -p managym --lib -E 'test(agent::heuristic) | test(agent::env::) | test(agent::vector_env::)' --test-threads 2` — initial 43 passed, debug; rerun after Clippy edits recorded below.
- `uv run --no-sync pytest tests/env/test_native_player_policies.py tests/sim/test_heuristic.py tests/env/test_captured_vector_bookkeeping.py tests/env/test_opponents.py tests/semantic/test_player_control.py tests/env/test_env.py tests/env/test_potential_shaping.py tests/env/test_observation.py tests/model/test_agent.py tests/sim/test_rollout.py tests/training/test_collection.py tests/training/test_automatic_game_capture.py tests/training/test_elemental_uniform.py tests/training/test_action_cutoff.py tests/verify/test_native_buffer_capacities.py tests/verify/test_input_v10.py tests/verify/test_observation_parity.py tests/test_package_boundaries.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/eval/test_mirror_admission.py tests/training/test_game_errors.py -q --durations=10` — 450 passed, four stale-test failures repaired; 198.20 s total, slowest test 53.09 s, no timeout.
- `uv run --no-sync pytest tests/training/test_game_errors.py::test_resource_limit_replay_rejects_old_schema_and_reproduces_current_boundary tests/agent/test_managym.py::TestManagym::test_invalid_action_handling tests/verify/test_input_v10.py -q` — 7 passed in 5.48 s.
- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci -p managym --test engine_tests -E 'test(env_reports_negative_action_index)' --test-threads 2` — 1 passed, debug.
- `git diff --name-only -z origin/main -- '*.py' '*.pyi' | xargs -0 uv run --no-sync ruff check --fix` — passed; corresponding `ruff format` — 26 files unchanged.
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` and `git diff --check` — passed before final lint edits; final result below.
- `ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build` — passed; built site included in staged wheels. No browser or server started.
- `uv run --no-sync python scripts/release/prepare.py .runs/mtg277-gate/staging --site academy/ui/build` — passed; staged paired wheels built with `uv build --wheel --no-sources --python <3.12|.venv/bin/python> ... --out-dir .runs/mtg277-gate/dist` (engine builds set `CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR="$PWD/managym/target"`).
- Engine-only copy-installed wheels, `env -u PYTHONPATH uv run --no-project --python /tmp/mtg277-gate-d0f233-<312|313>/bin/python engine_probe.py`, outside checkout — both passed all nine scalar/vector seat pairs and terminal rewards; manabot/Torch/Gymnasium/Etude absent.
- Copy-installed paired wheels, outside checkout, `env -u PYTHONPATH PATH="<venv>/bin:$PATH" uv run --no-project --python <venv>/bin/python smoke.py` — full release training smoke passed on CPython 3.12.12 and 3.13.15, Torch 2.15.0, Etude SDK 0.3.2, NumPy 2.5.4, Pydantic 2.14.3; final identity-file rebuild results below. Copy mode protects uv's cache during source-drift mutations.
- Both initial engine wheels contain all 96 Rust source files and Cargo.lock byte-for-byte, and the Academy site. Installed smokes verify registration, all four source-drift rejections/restorations, gradients, one CPU update, checkpoint reload and Academy bot discovery.
- `CARGO_BUILD_JOBS=2 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — initially found explicit-counter-loop and PyO3 generated-code false positive; repaired, final rerun below.
- `gh run view 43540649606 --json headSha,url` and jobs — exact runtime HEAD; Engine Budget, semantic kernel, protocol, Academy UI/integration, clean-machine play, Python unit, training, recovery, experiment and both evaluation groups passed. Rust/model failed only stale negative-index assertions (1254/1255 Rust; 860 model passed, 13 skipped per steer). Final repaired revision still needs CI.
- `gh run download 43540649606 -n engine-budget -D .runs/mtg277-gate/engine-budget` then `uv run --no-project python scripts/engine_budget.py compare --artifact .runs/mtg277-gate/engine-budget --reference .runs/mtg277-gate/engine-budget/reference` — PASS, unchanged search and 512-game/102209-step digests. Same-runner reference d53220d36: worst search reply 2.604 → 2.806 s; total search 90.922 → 97.917 s; steps 19334.2 → 18536.4/s, within 10%. No baseline moved. Final Clippy edits preserve strategy and unused-policy paths; CI owns final-revision budget confirmation.

Linux, minimum dependencies and the complete release inventory remain with
release CI. No paid run, artificial load or UI source changes.

Final tree results:

- Same focused native nextest command after loop-counter edit — 43 passed in 31.777 s.
- Same all-target/all-feature Clippy command — passed in 9.87 s; workspace fmt passed.
- Repeated `prepare.py` into `.runs/mtg277-gate-final/staging`, then the same paired wheel commands into `.runs/mtg277-gate-final/dist` — pure Python, cp313 and cp312 wheels built; all 96 Rust files, lock and site verified.
- Final paired wheels copy-reinstalled in both standalone consumers; same full release smoke — both passed, same dependency versions. Installed/checkout heuristic digest: `733039059bf9669c672336cc1f4f0ec55cd35623aad5ea37313ca9e83c24e3af`.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 extension replaced atomically, 96 packaged sources verified.
- `uv run --no-sync pytest tests/sim/test_heuristic.py -q` after final rebuild — 15 passed in 5.55 s, including retained decision parity.
- Temporary consumer directories removed after validation; no run-owned processes remain. Final-revision CI remains required before landing.
