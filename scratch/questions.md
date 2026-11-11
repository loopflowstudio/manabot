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
