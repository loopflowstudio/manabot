# MTG-288 gate — 2026-11-11

Reviewed HEAD `720c5dea6c5c38b2907bdb9969fa94a14554b950` against base
`4c041d6cefe829c5b1b52273ba19991d663ee26b`, the Task directive, agent guide,
and release guidance. No implementation or documentation correction needed.
MIT is present in the repository, Python metadata and all three crate manifests;
release staging retains its bytes. The third-party fixture notice is unchanged.
No runtime, UI or Rust source changed, so no performance measurement applies.

Plan: exercise release staging and tooling, format/lint the changed Python,
validate the current source archive and wheel payloads, and reuse unchanged
locked-sync and consumer-smoke evidence. The changed-path selector selects all
nine CI jobs through manifest/lock filters; broad runtime and native checks stay
with CI, not another local package-wide pass.

Commands and results:

- `git diff --name-only -z 4c041d6cefe829c5b1b52273ba19991d663ee26b HEAD | bash .github/ci/select-tests.sh` — all nine jobs selected; broad suites deferred to CI.
- `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q` — 23 passed in 0.41s.
- `uv run --no-project --with ruff ruff check --fix scripts/release/prepare.py tests/infra/test_release.py` — passed, no edits.
- `uv run --no-project --with ruff ruff format scripts/release/prepare.py tests/infra/test_release.py` — both unchanged.
- `uv run --no-project python scripts/release/check.py` — validated v0.2.1.
- `git diff --check` — passed.
- `git diff --quiet 4c041d6cefe829c5b1b52273ba19991d663ee26b HEAD -- tests/fixtures/seventeenlands/LICENSE` — unchanged.
- `git diff --quiet a1b158987 HEAD -- pyproject.toml uv.lock scripts/release/prepare.py tests/infra/test_release.py` — unchanged; reuse isolated locked-sync and installed consumer-smoke results from `a1b158987:scratch/qa.html`.
- `gh run download 43138667243 -n manabot-sdist -D /tmp/mtg288-gate.N465Xx/dist` — downloaded current-head CI source archive.
- `uv run --no-project python -` (archive inspection) — sdist LICENSE, README and pyproject bytes equal current checkout; MIT in all three archived Cargo manifests.
- `uv build --wheel /tmp/mtg288-gate.N465Xx/dist/manabot-0.2.1.tar.gz --out-dir /tmp/mtg288-gate.N465Xx/dist` — failed because the command selected Python 3.14; PyO3 explicitly supports at most 3.13. This was an invocation error, not a timeout; log `/tmp/mtg288-gate.N465Xx/build.log`.
- `uv build --wheel --python 3.13 /tmp/mtg288-gate.N465Xx/dist/manabot-0.2.1.tar.gz --out-dir /tmp/mtg288-gate.N465Xx/dist` — passed; log `/tmp/mtg288-gate.N465Xx/build313.log`.
- `uv run --no-project --with twine twine check --strict /tmp/mtg288-gate.N465Xx/dist/*` — local wheel and CI sdist passed.
- `uv run --no-project python scripts/release/validate.py /tmp/mtg288-gate.N465Xx/dist` — one wheel and sdist passed.
- `uv run --no-project python /tmp/mtg288-gate.N465Xx/inspect_licenses.py /tmp/mtg288-gate.N465Xx/dist` — exact current LICENSE bytes and MIT metadata passed for both artifacts.
- `gh run download 43138667243 -p 'manabot-*' -D /tmp/mtg288-gate.N465Xx/ci-artifacts` — downloaded all four CI wheels and sdist; copied archives into adjacent `ci-dist` for flat inventory validation.
- `uv run --no-project python scripts/release/validate.py /tmp/mtg288-gate.N465Xx/ci-dist --complete` — complete five-file release inventory passed.
- `uv run --no-project --with twine twine check --strict /tmp/mtg288-gate.N465Xx/ci-dist/*` — all five CI artifacts passed.
- `uv run --no-project python /tmp/mtg288-gate.N465Xx/inspect_licenses.py /tmp/mtg288-gate.N465Xx/ci-dist` — all five CI artifacts contain exact current LICENSE bytes, License-Expression MIT and License-File LICENSE.
- `gh pr view 465 --json title,body,headRefOid,statusCheckRollup` — HEAD matches review; release preflight, four package jobs and both Linux install jobs passed. macOS install and some broader checks remain running; full CI is not yet passed. Snapshot `/tmp/mtg288-gate.N465Xx/pr.json`.

Local gate is complete. Updated local PR copy preserves its Task markers and
records current validation. No merge, publication or release action occurred.
GitHub licence detection and PyPI rendering remain delivery checks. No product
questions or local blockers remain. No server or watcher was started.
