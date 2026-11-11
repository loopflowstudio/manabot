# MTG-271 gate — 2026-11-11

Scope: CPython 3.13 development, 3.12/3.13 consumers, public Torch/Etude
minimums, and version-generic release completion. Version remains 0.2.0;
no tag, publication, backfill or merge.

## Validation plan

The CI selector selects every job because shared dependency metadata changed.
It offers job selection, not focused local execution or result reuse. Leave the
full matrix to CI and run the affected contracts locally:

- Release and CI-selection tests: metadata identity, four-wheel inventory,
  publication retries, notes, permissions and workflow selection.
- Launcher and clean-machine verifier tests: 3.13/ABI defaults, orchestration
  and stale native-extension rejection.
- Isolated artifact/Etude integration tests: dependencies and generated machine
  setup cannot rely on incidental installed extras.
- Package-boundary and regime-discovery tests with CI extras: dependency
  ownership and imports after removing direct pandas.
- Loopflow workflow validation for the repository QA skill; lock validation;
  changed-file Ruff formatting/lint, actionlint and shell syntax checks.

Reuse QA's current-metadata macOS cp312/cp313 wheel smoke, minimum/latest
consumer installs and cold startup proof; release-only changes since that
proof do not alter package code, dependency metadata or launcher behavior. Linux wheels/runtime,
Ubuntu cold startup, the complete release inventory and full integration
matrix remain CI checks. CUDA execution was not tested locally or by the CPU
release matrix.

## Results

- `git diff --name-only -z origin/main...HEAD | bash .github/ci/select-tests.sh` — all nine CI job groups selected by shared dependencies.
- `uv run --no-project --python 3.13 --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py tests/infra/test_ci_selection.py -q` — 76 passed in 2.94 s.
- `uv run --extra dev --extra notebook --extra artifacts --extra reports --extra llm pytest tests/academy/test_play_launcher.py tests/academy/test_clean_machine_verifier.py tests/test_package_boundaries.py tests/training/test_regimes.py -q` — 68 passed in 28.66 s.
- `uv run --isolated --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_etude_job.py -q` — 31 passed in 54.98 s.
- `uv lock --check` — passed; 170 packages resolved, no lock changes.
- `uv run --no-sync python scripts/check_loopflow_workflows.py` — passed; seven graphs compiled and delivery/experiment boundaries valid.
- `uv run --no-sync ruff format manabot/experiment/job.py scripts/play.py scripts/release/check.py scripts/release/complete.py scripts/release/prepare.py scripts/release/smoke.py scripts/release/validate.py tests/academy/test_clean_machine_verifier.py tests/academy/test_play_launcher.py tests/infra/test_release.py tests/training/test_etude_job.py` — 11 files unchanged.
- `uv run --no-sync ruff check --fix manabot/experiment/job.py scripts/play.py scripts/release/check.py scripts/release/complete.py scripts/release/prepare.py scripts/release/smoke.py scripts/release/validate.py tests/academy/test_clean_machine_verifier.py tests/academy/test_play_launcher.py tests/infra/test_release.py tests/training/test_etude_job.py` — passed, no changes.
- `actionlint .github/workflows/academy-integration.yml .github/workflows/baseline.yml .github/workflows/card-evaluation.yml .github/workflows/ci.yml .github/workflows/engine-budget.yml .github/workflows/layer-risk.yml .github/workflows/manabot-integration.yml .github/workflows/release.yml` — passed.
- `sh -n scripts/play scripts/play-elemental scripts/verify-clean-machine scripts/verify-public-commitment-parity` — passed.
- `git diff --check` — passed.

## Outcome

175 focused tests passed. No code change was needed. Updated the development
guide to name the current 3.13 Ubuntu reference profile while preserving the
historical 3.12 evidence. Removed stale pending-validation claims from the plan
and corrected the QA report's suggestion that CPU release CI validates CUDA.
The branch is ready for CI/review with the deferrals above; no local blocker.
