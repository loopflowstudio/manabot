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
