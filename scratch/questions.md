# MTG-288 — MIT licence, 2026-11-11

Jack Heart: "license should be MIT" (Task directive snapshot).

Reversible assumption: `Copyright (c) 2026 Manabot contributors`; no specific
copyright holder was supplied. No open product question.

The Task brief calls `tests/fixtures/seventeenlands/LICENSE` CC BY 4.0, but it
begins "GNU GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007". README links
that notice without relabelling it; the separate public trophy-deck dataset is
identified there as CC BY 4.0. Preserve the fixture notice and Wizards attribution.

## Plan and rationale

Root `LICENSE` supplies the MIT text; `pyproject.toml` declares SPDX MIT and
`license-files`, and all three Cargo manifests declare MIT. Release staging
(`scripts/release/prepare.py`) carries LICENSE; `.github/workflows/release.yml`
runs for licence-only changes. README links the root text and third-party notice.
Delete — do not maintain: none; this is additive metadata, with no runtime change.

Keep maturin >=1.9.3 consistent across build and development requirements.
The initial file-form metadata build carried LICENSE but emitted no MIT value
in PKG-INFO. Maturin documents SPDX metadata and sdist licence-file fixes through
1.9.3 ([changelog](https://www.maturin.rs/changelog), read 2026-11-11).

Compression groups the three staging byte-preservation assertions in
`tests/infra/test_release.py` into one loop, retaining the failing path in the
assertion message. No production simplification was warranted.

## Evidence and remaining validation

Implementation results (2026-11-11):

- `test_release_projects_preserve_native_paths_and_runtime_data`: 1 passed.
- `uv lock --offline`: passed; only maturin requirement metadata changed, no resolved versions changed.
- `uv build --sdist /tmp/mtg288-license.7y0mfV/spdx-staging --out-dir /tmp/mtg288-license.7y0mfV/spdx-dist`: passed. Inspection verified exact root LICENSE bytes, `License-Expression: MIT`, `License-File: LICENSE`, and MIT in all three crate manifests.
- `git diff --check`: passed; fixture licence unchanged.
- `lf open --host desktop --file README.md`: "No Desktop Files view attached; nothing opened."

Compression verification: `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data -q` — 1 passed; Ruff lint/format and `git diff --check` passed.

Gate still needs `twine check --strict` on the staged sdist and built wheels,
affected release tooling checks and isolated locked dependency sync. Use the
`spdx-dist` directory above, not the earlier file-form trial in
`/tmp/mtg288-license.7y0mfV/dist`. No wheel build or engine suite ran during
implementation; no runtime or Rust source changed.

Inherited MTG-285 release decisions and evidence remain in
`4c041d6cefe829c5b1b52273ba19991d663ee26b:scratch/questions.md`; they concern
that separate release Task. Before curation, the complete uncommitted notes
were moved to `/tmp/mtg288-compress.Xna56V/questions.md`.
