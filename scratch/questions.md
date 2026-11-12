# MTG-288 — MIT licence, 2026-11-11

Jack Heart: "license should be MIT" (Task directive snapshot).

Jack Heart, 2026-11-11: "Make it (c) Loopflow Studio". LICENSE reads
`Copyright (c) 2026 Loopflow Studio`. No open product question.

The Task brief calls `tests/fixtures/seventeenlands/LICENSE` CC BY 4.0, but it
begins "GNU GENERAL PUBLIC LICENSE", "Version 3, 29 June 2007". README links
that notice without relabelling it; the separate public trophy-deck dataset is
identified there as CC BY 4.0. Preserve the fixture notice and Wizards attribution.

## Outcome and implemented approach

Outside consumers can find the MIT grant in the repository and distributed
package, with matching Python and Rust licence metadata. Third-party notices
and Wizards attribution retain their separate scope.

Root `LICENSE` supplies the MIT text; `pyproject.toml` declares SPDX MIT and
`license-files`, and all three Cargo manifests declare MIT. Release staging
(`scripts/release/prepare.py`) carries LICENSE; `.github/workflows/release.yml`
runs for licence-only changes. README links the root text and third-party notice.

Keep maturin >=1.9.3 consistent across build and development requirements.
The initial file-form metadata build carried LICENSE but emitted no MIT value
in PKG-INFO. Maturin documents SPDX metadata and sdist licence-file fixes through
1.9.3 ([changelog](https://www.maturin.rs/changelog), read 2026-11-11).

`tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data`
checks that staging preserves the exact LICENSE bytes alongside the project
metadata and protocol. No runtime or Rust source changed.

## Evidence and remaining validation

Implementation results (2026-11-11):

- `test_release_projects_preserve_native_paths_and_runtime_data`: 1 passed.
- `uv lock --offline`: passed; only maturin requirement metadata changed, no resolved versions changed.
- `uv build --sdist /tmp/mtg288-license.7y0mfV/spdx-staging --out-dir /tmp/mtg288-license.7y0mfV/spdx-dist`: passed. Inspection verified exact root LICENSE bytes, `License-Expression: MIT`, `License-File: LICENSE`, and MIT in all three crate manifests.
- `git diff --check`: passed; fixture licence unchanged.
- `lf open --host desktop --file README.md`: "No Desktop Files view attached; nothing opened."

Compression verification: `uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data -q` — 1 passed; Ruff lint/format and `git diff --check` passed.

Realign findings (2026-11-11): fetched `origin/main` remains
`4c041d6cefe829c5b1b52273ba19991d663ee26b`, already included in HEAD
`9140922a2`; no upstream changes alter this scope. Task status reports no Wave
or PR and no new direction. No code repair was needed.

The retained `spdx-staging` LICENSE, README, pyproject and three Cargo manifests
match the current checkout byte for byte. Comparing the sdist directly instead
failed on `managym/Cargo.toml`: its packaged form adds `readme` and omits
`default-members`, `[lib]` and explicit example tables. Both child manifests
match. This does not establish wheel buildability; that proof remains below.
Existing test results were reused, not rerun.

## Remaining validation

- Run `twine check --strict` on the retained `spdx-dist` sdist above and on
  newly built wheels. The earlier `/tmp/mtg288-license.7y0mfV/dist` trial used
  file-form metadata and is not the accepted artifact.
- Build wheels from the sdist and inspect `License-Expression: MIT`,
  `License-File: LICENSE` and the exact bundled LICENSE bytes. Strict metadata
  checks alone do not prove payload preservation;
  `scripts/release/validate.py::_metadata` compares identity and dependencies,
  not licence fields. `.github/workflows/release.yml` owns the Linux/macOS,
  CPython 3.12/3.13 build/install matrix and strict artifact checks.
- Run the full affected release-tooling tests in `tests/infra/test_release.py`
  and an isolated `uv sync --locked --extra play --no-install-project` with a
  fresh `UV_PROJECT_ENVIRONMENT`, since the maturin minimum changed.
- Broad engine checks remain with CI; no engine suite or wheel build has run
  for this Task. The unavailable Desktop Files view does not block headless
  validation. Publication and the 0.2.1 release sequence belong to delivery and
  MTG-285, not this alignment pass.

Inherited MTG-285 release decisions and evidence remain in
`4c041d6cefe829c5b1b52273ba19991d663ee26b:scratch/questions.md`; they concern
that separate release Task. Before curation, the complete uncommitted notes
were moved to `/tmp/mtg288-compress.Xna56V/questions.md`.

## QA results — 2026-11-11

The remaining validation listed above has now run (`scratch/qa.html`). Fresh staging, an sdist and a
cp313 macOS arm64 wheel built from that sdist: both have Metadata 2.4, `License-Expression: MIT`,
`License-File: LICENSE`, LICENSE bytes identical to the checkout, and pass `twine check --strict`.
The wheel was installed in a fresh venv: `importlib.metadata` reports MIT, and the
release smoke passed (Python 3.13.7, torch 2.15.0). `tests/infra/test_release.py`: 23 passed.
Isolated `uv sync --locked --extra play`: passed (maturin 1.14.1). LICENSE matches the SPDX MIT
text apart from wrapping. Linux and 3.12 wheels remain with the CI release matrix.

## Concept review — 2026-11-11

Outcome for loop-or-next: nothing to change and no open product choice; advance.
The existing design gives an outside consumer one MIT statement for original
source, discoverable from README and carried through package metadata and the
release archive. Third-party notices remain distinct. No runtime concept, API,
user action or release lifecycle needs adding.

Normal path: a consumer follows README to LICENSE, then obtains the same notice
with the package. Failure/recovery: if staging or an archive loses the notice
or MIT metadata, repair packaging and rebuild before publication; the source
notice and third-party attribution must survive. Counterexamples retained:
an MIT metadata label without the notice bytes does not meet this design, and
the root grant must not be presented as replacing third-party notices.

No new implementation direction or usage/skill rewrite is needed. Next show
the wheel payload and strict metadata proofs already listed above. This review
ran no builds or tests and did not publish. The review page is
`scratch/concept-review.html`.

Jack Heart, 2026-11-11: "yeah cut it out of the readme" — the README licence
section no longer mentions the GPL-3.0 17Lands fixture; its own LICENSE and
README stay.
