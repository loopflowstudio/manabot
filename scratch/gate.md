# MTG-287 gate — 2026-11-11

Scope: paired `managym` and `manabot` distributions, exact shared versions,
engine-only wheels, compatibility imports, source archives and release recovery.
Reviewed the complete diff from `4c041d6` and the removed design/QA notes at
`ab40fe19^`. No tag, upload, publication or merge is authorized here.

## Plan and findings

The repository selector enables all nine CI groups for these package moves.
Locally, focus on release metadata/staging/retry tests, import boundaries,
source binding, recipe discovery, deck/decision/protocol APIs, then the parity
readers whose source manifests moved. Reuse unchanged native and consumer
evidence; keep the full platform/browser/integration matrix in CI.

Ruff fixed four import groups. The compatibility stub now marks its intentional
wildcard export; development guidance includes `.pyi` in Python lint checks.

A fresh authored-match replay differed from its frozen receipt only at
`identity.relevant_source`. Verification now follows the existing
public-commitment comparison policy: allow source provenance to differ, compare
all other data exactly, and return the untouched historical receipt. Two
regression cases verify successful relocation and rejection of changed command
counts, including unchanged receipt bytes. The public-commitment tests now
declare the retained measurement they require through `experiment_data`.

## Commands and results

- `git diff --name-only -z 4c041d6...HEAD | bash .github/ci/select-tests.sh` — all nine groups selected; broad runtime/browser coverage deferred to CI.
- `uv run --locked --extra dev --extra artifacts pytest tests/infra/test_release.py tests/infra/test_ci_selection.py tests/test_package_boundaries.py tests/training/test_etude_source.py tests/training/test_regimes.py tests/semantic/test_decklist_import.py tests/semantic/test_decision_contract.py tests/academy/test_experience_protocol.py -q` — 162 passed in 146.99 s.
- `uv run --locked --extra dev --extra artifacts --extra reports pytest tests/test_package_boundaries.py tests/academy/test_authored_match_parity.py tests/academy/test_public_commitment_parity.py tests/semantic/test_text_observation.py -q` — initially 73 passed, 3 failed: one historical source comparison and two missing measurement files; no timeout.
- `uv run --locked --extra dev --extra artifacts --extra reports python` (bounded `build_receipt` comparison) — all authored replay data matched after substituting only historical source provenance.
- `doppler run -p manabot -c prd --only-secrets R2_ACCOUNT_ID --only-secrets R2_ACCESS_KEY_ID --only-secrets R2_SECRET_ACCESS_KEY -- uv run --no-sync python -m manabot.data fetch experiments/data/rul-9-played-workloads-v1.measurement.json` — verified one file from the local content-addressed cache; no frozen bytes changed.
- `uv run --locked --extra dev --extra artifacts --extra reports pytest tests/academy/test_authored_match_parity.py tests/academy/test_public_commitment_parity.py -q` — 9 passed in 42.22 s after the comparison fix and artifact restoration.
- `uv run --no-sync ruff check --fix <changed Python and stub paths>` and `uv run --no-sync ruff format <same paths>` — passed after import fixes and explicit compatibility-export suppression; final edits checked separately.
- `actionlint .github/workflows/ci.yml .github/workflows/release.yml` — passed.
- `uv run --no-sync python scripts/check_loopflow_workflows.py` — all eight graphs compiled, all skills resolved, boundaries valid.
- `uv run --no-project python scripts/release/check.py` — shared version 0.2.1 valid.
- `uv run --no-project python scripts/release/validate.py .runs/qa/dist` and `uv run --no-project --with twine twine check --strict .runs/qa/dist/*` — retained three wheels and two source archives passed.
- `uv run --no-project python scripts/release/prepare.py .runs/gate/staging` — staged current tracked source bytes.
- `CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=/tmp/mtg287-qa-target uv build --no-sources --python 3.13 .runs/gate/staging/managym --out-dir .runs/gate/dist` — engine archive and macOS arm64 CPython 3.13 wheel built; wheel built from the archive, log `.runs/gate/engine-build.log`.
- `uv build --no-sources .runs/gate/staging --out-dir .runs/gate/dist` — training archive and pure-Python wheel built; log `.runs/gate/bot-build.log`.
- `uv run --no-project python scripts/release/validate.py .runs/gate/dist` and `uv run --no-project --with twine twine check --strict .runs/gate/dist/*` — all four fresh artifacts passed.
- `uv venv --python 3.13 /tmp/managym-gate.bBmXJT/venv` and `uv pip install --python <venv>/bin/python --only-binary :all: <fresh engine wheel>` — clean seven-package engine install; PATH contained only a uv symlink and `/usr/bin:/bin`, with PYTHONPATH unset.
- Outside the checkout, `uv run --no-project --python <venv>/bin/python engine_smoke.py` — engine-only complete game passed in 107 decisions, without training packages or Rust on PATH.
- `uv pip install --python <venv>/bin/python --only-binary :all: --torch-backend cpu <both fresh wheels> 'torch==2.10.0' 'etude-sdk==0.2.15'` and `uv pip check --python <venv>/bin/python` — 39 installed packages compatible.
- Outside the checkout, `uv run --no-project --python <venv>/bin/python smoke.py` — passed CPU training, checkpoint reload and alias/catalog checks on CPython 3.13.15, Torch 2.10.0 and Etude SDK 0.2.15; `.runs/gate/training-smoke.log`.
- Final `uv run --no-sync ruff check <seven gate-edited Python/stub paths>` and `uv run --no-sync ruff format --check <same paths>` — passed, seven files formatted.
- `sh -n scripts/llm-game-setup.sh scripts/policy-inference-setup.sh scripts/qwen-mtg183-setup.sh` and `git diff --check` — passed.

## Reused evidence and limits

Only scratch removal separated HEAD from the previous validated source commit
`e03d671e4` before gate edits. All 221 Rust source/lock files in the retained
engine archive matched the current tree; Cargo manifest differences were
maturin's sdist normalization. Gate changes no Rust or rules behavior.

- Prior debug `cargo nextest run --profile ci` with two workers: 1057 passed,
  2 skipped; `/tmp/mtg287-publish-rust.log`. Prior rustfmt and all-target,
  all-feature Clippy passed, recorded at `ab40fe19^:scratch/packaging.md`.
- Prior isolated artifact/Etude dependency suite: 31 passed;
  `/tmp/mtg287-publish-isolated.log`. Integration collection: 1376 core, 560
  evaluation, 134 long; collection is not runtime execution.
- Prior macOS CPython 3.12/3.13 engine-only installs played complete 107-decision
  games with Rust absent. Training passed with minimum dependencies on 3.12
  and latest tested dependencies on 3.13; retained QA is
  `ab40fe19^:scratch/qa.html`. Gate rebuilds current Python bytes for 3.13.

Linux wheels, the complete seven-artifact inventory and broad CI groups remain
CI checks. Existing documented limitation: pip's one-step upgrade from bundled
manabot 0.2.0 can delete engine files; uninstall-first and engine-reinstall
recovery remain necessary. First paired publication still requires the selected
version, both trusted publishers and explicit publication authorization.

Local gate is complete. No servers or watchers were started; all check/build
processes finished. Source changes and these gate notes remain uncommitted.
