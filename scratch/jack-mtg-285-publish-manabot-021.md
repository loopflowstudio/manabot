# Publish manabot 0.2.1

Release the merged MTG-271 compatibility changes and MTG-272 scalar/batched
hand snapshots and API audit. Preserve the shipping docs, historical release
records, frozen experiments, world bindings and exact dependency pins.

## Approach

Update Python metadata, the three bundled Rust crate manifests and both
lockfiles to 0.2.1. Update current installation guidance; leave synthetic
0.2.0 fixtures intact. Notes cover CPython 3.12/3.13,
public dependency minimums, HandSnapshot, generation identities and all four
collector/reset breaking changes, with links to the audit and migration guide.

## Delete — do not maintain

Replaced current 0.2.0 package declarations and provisional local-wheel-only
install guidance; local-wheel verification remains supported. Removed the
current-version literal from `tests/infra/test_release.py`: current metadata
validation uses `check(None)`, while the existing synthetic project checks both
matching and mismatched tags. Future releases need no test edit. No remaining
deletion targets; runtime code and historical evidence stay unchanged.

## Remaining work

- Gate: staged build/metadata validation and locked sync;
  release.yml must prove four wheels and outside-checkout consumer installs.
- Delivery: merge green release PR, recheck publisher settings, tag that main
  commit, inspect artifacts and approve pypi, confirm both publication surfaces.
- Verify: PyPI installs on Linux x86-64 and Apple Silicon, CPython 3.12/3.13;
  confirm version/attestations/Release link and retain the publication record.

Publication, landing and verification remain the caller Graph's steps.

Implementation checks: `check.py --tag v0.2.1` and the focused
`test_release_versions_and_tags_agree` passed (1 test); offline uv/Cargo locked
metadata and all 0.2.1 manifest/lock entries agree; `git diff --check` passed.
Compression: `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q` — 23 passed in 4.95 s.
No Rust source changed; staged builds and consumer checks remain for gate.
