# Paired Python distributions

Build the engine as `managym`, and training/clients as pure-Python `manabot`
with an exact same-version engine requirement. Preserve decisions, runtime
assets, source receipts and compatibility imports. One tag builds both archives,
four engine wheels and one training wheel; validate both before uploading the
engine first. Exact-digest recovery may resume a missing training publication.

## Delete — do not maintain

- Python implementation and stubs formerly under `manabot/managym`: moved to
  `managym`; only the expressly requested import/stub alias remains.
- Root maturin backend and bundled-extension release path: replaced by the
  workspace, Hatchling training wheel and self-contained engine staging.
- Single-project upload checks: replaced by paired inventories, with original
  partial-file/digest mismatch rejection retained.
- Temporary `engine-crate` staging and reverse receipt-path mapping: removed.
  `scripts/release/prepare.py::prepare` copies to final paths and records installed
  source paths in the same pass. The engine smoke discovers installed adapters
  instead of maintaining another module list.

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

Run the affected suites and full release acceptance matrix once: debug Rust,
CPython 3.12 and Linux wheels, engine-only installs, the existing CPU training
smoke, optional dependency imports and complete seven-file validation. The
workflow contains these consumer checks; this implementation pass did not run
training or claim the cross-platform matrix passed. Review docs in
`docs/engine-python.md` and `docs/development.md` alongside the implementation.

Final local checks: rustfmt passed after wrapping attributes; the root CPython
3.13 release build passed and its extension was installed atomically. The final
engine-independence probe passed. `lf sync origin/main --manual --json` finished
with local edits retained; main has no source drift from the checkout base.
