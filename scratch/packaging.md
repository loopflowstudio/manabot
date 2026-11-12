# Paired Python distributions

Build the engine as `managym`, and training/clients as pure-Python `manabot`
with an exact same-version engine requirement. Preserve decisions, runtime
assets, source receipts and compatibility imports. One tag builds both archives,
four engine wheels and one training wheel; validate both before uploading the
engine first. Exact-digest recovery may resume a missing training publication.
Publication is outside this Task's current authorization; release decisions and
the exact direction are retained in `scratch/questions.md`.

## Delete — do not maintain

- Python implementation and stubs formerly under `manabot/managym`: moved to
  `managym`; only the compatibility import/stub alias remains.
- Root maturin backend and bundled-extension release path: replaced by the
  workspace, Hatchling training wheel and self-contained engine staging.
- Single-project upload checks: replaced by paired inventories, with original
  partial-file/digest mismatch rejection retained.
- Temporary `engine-crate` staging and reverse receipt-path mapping: removed.
  `scripts/release/prepare.py::prepare` copies to final paths and records installed
  source paths in the same pass. The engine smoke discovers installed adapters
  instead of maintaining another module list.
- Late candidate-install qualifier: replaced by guidance before the first
  command in both install entry points. Existing local-wheel commands remain
  the single candidate-install path.

No deletion targets remain. Preserve compatibility aliases, frozen evidence
paths, platform guidance and exact dependency matching.

## Implemented approach

`scripts/release/check.py::check` enforces version lockstep across both Python
projects, all three Rust crates and the exact engine dependency.
`.github/workflows/release.yml` builds wheels from both source archives;
`validate.py::validate` checks the seven-file inventory before either upload.
`complete.py::main` checks both PyPI inventories before emitting upload flags
and requires the complete engine before training upload. This is not atomic
publication: a later upload failure can leave only the engine published.
Recovery requires the original artifacts and exact digests; partial files
within a package still reject.

## Implementation checks — 2026-11-11

- Release-tooling tests: 34 passed; paired inventories, version drift,
  pre-upload failure and exact-artifact retry cases included.
- Focused import/source-binding tests: 4 passed; adapters import with all
  training/Academy imports refused, aliases share identity, moved native bytes
  remain bound to Etude source receipts.
- Fresh `uv sync --locked --python 3.13 --extra dev --extra artifacts`: passed.
- Both sdists and macOS CPython 3.13 wheels built from source archives;
  `/tmp/mtg287-final-dist`, logs `/tmp/mtg287-final-{sdist,engine,bot}.log`.
  These builds precede rustfmt-only wrapping of four PyO3 attributes.
- Artifact validation and strict Twine checks: passed for those four artifacts.
- Engine-only isolated install: all adapters and canonical native module names
  checked; a complete game ended after 107 decisions, without training packages.
- Adding the matching training wheel: import identity, source receipt, both
  curated policy catalogs and `uv pip check` passed, with Torch 2.10.0 and
  Etude SDK 0.2.15. CPython 3.13.7, Apple Silicon.
- Actionlint on both changed workflows and Ruff on changed Python: passed.
- Compression: `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q`
  → 34 passed; focused import/source-binding/CI-selection checks → 58 passed;
  Ruff check/format → passed.
- Rebuilt both sdists and macOS CPython 3.13 wheels from archives after compression:
  `/tmp/mtg287-compress-dist`, logs `/tmp/mtg287-compress-{engine-sdist,engine-wheel,bot}.log`.
  Validation/Twine passed. Fresh engine-only smoke ended after 107 decisions;
  adding training passed dependency checks, alias identity and installed source
  verification. The source receipt still binds the same 235 files and commit.

## Remaining gate/CI work

The local proofs above cover Apple Silicon CPython 3.13 and four artifacts,
not the complete release matrix. Gate still needs the affected suites from
`docs/development.md#validation-by-change-area`, including debug Rust and Clippy
for the PyO3 module-name changes. Reuse the recorded release-tooling and focused
import/source checks unless subsequent changes invalidate them.

CI must build and install on CPython 3.12/3.13, Linux x86-64 and Apple Silicon:
engine-only complete games with Rust unavailable, then matching training wheels
running `scripts/release/smoke.py` (CPU training and checkpoint reload), optional
dependency imports, and minimum/latest-compatible dependency checks. Finally
`validate.py --complete` and strict Twine checks must pass for all seven files.
Neither CPU training nor this complete matrix has been demonstrated locally.

The owner steps are in `docs/development.md#publish`: choose the first paired
version, configure both trusted publishers and explicitly authorize publication.
These are release prerequisites, not permission to tag or upload in this Task.

Final local checks: rustfmt passed after wrapping attributes; the root CPython
3.13 release build passed and its extension was installed atomically. The final
engine-independence probe passed. `lf sync origin/main --manual --json` finished
with local edits retained; main has no source drift from the checkout base.

Realign, 2026-11-11: Task status still limits MTG-287 to packaging; the other
consumer concerns remain MTG-288–292. Remote main was
`4c041d6cefe829c5b1b52273ba19991d663ee26b`, already an ancestor of this checkout;
no new upstream work required a redesign. Corrected the unpublished 0.2.1 notes
and API audit's stale bundled-engine/alias claims and five-file inventory.
No runtime change or new acceptance result was inferred from that correction.

## Install clarification — 2026-11-11

The concept review found that an unpublished candidate's index install can fail
even on a supported interpreter. Both `docs/engine-python.md` and
`docs/development.md#install` now link to the existing Build and verify section
before the first command and qualify PyPI commands as post-publication usage.
Candidate consumers install the engine wheel, then supply both local wheel
paths when adding training. No new installer or release state is needed.
Package versions still do not replace world identity or checkpoint admission.

Compression confirmed the requested clarification was already in the working
tree, retained it, consolidated repeated plan text and updated the stale
proposal status in `scratch/concept-review.html`. No further runtime cut was
warranted. Uncommitted notes before curation are retained at
`/tmp/mtg287-compress-notes.rQn0gz/`.

Documentation check, 2026-11-11: `git diff --check` passed; both qualifiers
precede the first install command and link to `### Build and verify`.
No games, builds or publication checks were rerun for this prose-only pass.
