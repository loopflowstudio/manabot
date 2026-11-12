# Publish manabot 0.2.1

Release the merged MTG-271 compatibility changes and MTG-272 scalar/batched
hand snapshots and API audit. Preserve the shipping docs, historical release
records, frozen experiments, world bindings and exact dependency pins.

## Implemented

Python metadata, the three bundled Rust crate manifests and both
lockfiles now declare 0.2.1. Current installation guidance uses the PyPI
version; synthetic 0.2.0 fixtures remain intact. `release/v0.2.1/NOTES.md`
covers CPython 3.12/3.13,
public dependency minimums, HandSnapshot, generation identities and all four
collector/reset breaking changes, with links to the audit and migration guide.

Local-wheel verification remains supported. Removed the current-version
literal from `tests/infra/test_release.py`: current metadata
validation uses `check(None)`, while the existing synthetic project checks both
matching and mismatched tags. Future releases need no test edit. No remaining
deletion targets; runtime code and historical evidence stay unchanged.

## Release path

`docs/development.md#publish` and `.github/workflows/release.yml` own delivery.
There are no `release.targets` in `.lf/config.yaml`. The existing workflow
builds one sdist, four wheels and an outside-checkout consumer matrix, then
validates the inventory with `scripts/release/validate.py --complete`.
Only tags publish; the `publish` job waits on the `pypi` environment, followed
by `github-release`. The Task steer supplies approval for unattended delivery.

The workflow's `install` jobs consume downloaded wheels before publication;
they do not prove installation from PyPI. Post-publication verification must
separately run `scripts/release/smoke.py` against fresh PyPI installs on Linux
x86-64 and Apple Silicon, for both 3.12 and 3.13, without paid runs. Headless
execution of that four-combination proof remains to be arranged during delivery.

## Remaining work

- Obtain green release PR CI, including the four wheels and outside-checkout
  consumer installs. Local gate is complete: staged macOS builds, metadata,
  consumer checks and actual isolated locked sync passed (see questions.md).
- Delivery: merge the green release PR, confirm its release workflow result,
  recheck environment reviewers/trusted publisher, then tag that exact main
  commit. Inspect the tag run's five artifacts, version and preflight result;
  approve the pending `pypi` deployment with "0.2.1 artifacts verified".
- Verify: run the separate PyPI install matrix; confirm version, five file
  hashes, attestations and the GitHub Release's notes/PyPI link. Retain a
  publication record like the pinned 0.2.0 record linked from the release guide.

Stop on a mismatch. Never overwrite PyPI files, delete a release or move a used
tag; completion retries retain the original artifacts. No product decision is
open. Publication, landing and verification remain the caller Graph's steps.

## Retained checks

Implementation checks: `check.py --tag v0.2.1` and the focused
pre-compression metadata test passed (1 test); offline uv/Cargo locked
metadata and all 0.2.1 manifest/lock entries agree; `git diff --check` passed.
Compression: `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q` — 23 passed in 4.95 s.
No Rust source changed; staged builds and consumer checks remain for gate.

## Concept review — 2026-11-11

Outcome: nothing to change and no open product choice; advance. Jack Heart:
"when this lands, publish 0.2.1 patch release"; "make sure it runs automatically".
The existing design already assigns unattended delivery to this Task and its
steer. It does not require a second release system or a new package identity.

The consumer installs one `manabot` distribution, reads the explicit collector
migrations, and retains original source for frozen runs. Python metadata, the
bundled engine, source archive and four wheels share the release version;
world and checkpoint identities remain independent. A package upgrade must
not imply that an old run can be resumed under a new world.

Normal path: validate the release PR, tag its exact merged commit, verify the
five artifacts, approve the authorized deployment, complete the GitHub entry,
then prove fresh PyPI installs and retain the publication record. Failure path:
if PyPI succeeds but GitHub completion fails, retry completion using the same
tag and original artifacts. Counterexample: a rebuilt archive with the same
version but different bytes is not a safe retry; stop on a partial inventory or
digest mismatch. These distinctions preserve recovery and cannot be collapsed
into a single successful-upload flag.

The biggest gap remains execution evidence, already identified at realign:
downloaded-wheel CI is not proof of PyPI installation. Delivery must arrange
the four unpaid consumer checks before reporting the Task complete. Show next
the gate results, then the publication record with commit/tag, workflow,
artifact hashes, attestations, Release link and exact installed versions.
No implementation direction changes. This review inspected source and docs;
it ran no builds, runtime tests or external publication checks.
