# MTG-285 release decisions — 2026-11-11

Jack Heart: "when this lands, publish 0.2.1 patch release".
Jack Heart: "make sure it runs automatically".

PR #458 (MTG-272) merged at 2026-11-11T23:46:34Z, commit
`0b296e617a87017c4cca71227749eddac0a0481b` (GitHub PR API).
The inherited MTG-272 implementation/audit evidence is retained in
`scratch/questions.md` at that commit; the public audit is
`docs/python-api-audit.md`.

Reversible assumption: keep all three bundled Rust workspace crates at 0.2.1,
consistent with their existing shared 0.2.0 version. Synthetic artifact fixtures
continue to use 0.2.0. Compression keeps tag matching in the existing synthetic
9.9.9 project, avoiding a current-version literal in the tests.
No product questions remain.

Shipping docs prepared before publication: `release/v0.2.1/NOTES.md` and
`docs/development.md`. The previous docs step checked syntax, fences and link
targets; no runtime or install check was performed. Opening the docs returned
"No Desktop Files view attached; nothing opened."

The Task steer authorizes the later release steps without a person: after the
release PR merges and release.yml is green, tag its exact main commit, inspect
all four wheels and the sdist, then approve the pending `pypi` deployment with
comment "0.2.1 artifacts verified". Recheck reviewers/trusted publisher before
tagging. Confirm PyPI, GitHub Release, attestations and clean installs on both
platforms/interpreters; retain a publication record. Stop on a mismatch.
No tag, upload, approval or install proof has happened in implementation.

Realign, 2026-11-11: GitHub compare of the MTG-272 merge above against `main`
returned `identical` (zero commits ahead/behind). `lf task status MTG-285`
reported the active `code` Graph at `realign`, no PR and no Wave. No upstream
change or new product decision alters the release scope. Existing 23-pass
release-tooling evidence is reused; no runtime tests were rerun. Source review
found the workflow tests downloaded wheels before publication, so the plan now
keeps the separate post-publication PyPI install matrix explicit.

PR publication review, 2026-11-11: `lf sync` completed cleanly; the release
diff contains metadata, documentation and release-test changes, with no runtime
source changes. Reused the 23 passing release tests and the macOS 3.12/3.13
sdist-to-wheel consumer proofs recorded in `scratch/qa.html`; inspected both
retained build logs in `/tmp/mtg285-qa.ScDu`. Fresh isolated dependency sync
(`uv sync --locked --extra play --no-install-project`, environment
`/tmp/mtg285-locked-sync.5kFX43/venv`) installed 38 packages successfully,
including Torch 2.10.0 and Etude 0.2.15 on CPython 3.13.7. Release tag metadata
and diff whitespace checks passed. Linux builds, the full five-file inventory
and PyPI publication remain CI/delivery checks; this step publishes only the PR.

## Gate — 2026-11-11

Reviewed PR #463 at `bbc707c64` against the release scope, agent guide and
release instructions. No product, test or documentation fix was needed. The
existing PR description accurately states validation and publication limits.

Validation plan: metadata and lock changes reach staged packaging, installed
consumers and the release-tooling tests; the sole Python change is a release
test refactor. Reuse unchanged build, consumer, locked-sync and 23-test evidence
above and in `scratch/qa.html`; validate the retained artifacts and format/lint
the changed test. Runtime source, browser interactions and performance behavior
are unchanged, so their broad suites stay with CI.

- `git diff --name-only 763b88480 HEAD` — only scratch files; all product, build, documentation and test inputs match the retained QA tree.
- `git diff --name-only -z origin/main...HEAD | bash .github/ci/select-tests.sh` — selects all nine jobs through manifest/lock filters; broad runtime, Linux and browser checks deferred to CI.
- Inspected `/tmp/mtg285-qa.ScDu/build312.log` and `build313.log` — confirms both sdist-derived macOS wheels built successfully; consumer and 23-test results reused from QA, not rerun.
- `uv run --no-project python scripts/release/check.py --tag v0.2.1` — passed.
- `uv run --no-project python scripts/release/validate.py /tmp/mtg285-qa.ScDu/dist` — passed for two macOS wheels and one sdist; complete five-file validation remains CI.
- `uv run --no-project --with twine twine check --strict /tmp/mtg285-qa.ScDu/dist/*` — all three artifacts passed.
- `uv run --no-project --with ruff ruff check --fix tests/infra/test_release.py` — passed, no edits.
- `uv run --no-project --with ruff ruff format tests/infra/test_release.py` — unchanged.
- `git diff --check` — passed.
- `gh pr view --json number,url,headRefOid,state,statusCheckRollup` — PR #463 open at the reviewed HEAD; release preflight, Python Unit Tests and Semantic Kernel Conformance passed. Other jobs still running when inspected; no full CI pass claimed.

Local gate is complete. Delivery still requires green CI, the authorized tag and
deployment sequence, and the separate post-publication PyPI install matrix.
No tag, approval, upload or merge was performed in this gate step.
