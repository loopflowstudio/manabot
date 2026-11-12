Manabot had no stated project licence, leaving consumers without clear reuse terms. This PR adds an MIT grant for its original source and carries the notice and matching metadata into release packages.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Add the MIT licence · MTG-288](https://linear.app/loopflow/issue/MTG-288/add-the-mit-licence)
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

## What changes

- Add an MIT `LICENSE` with Loopflow Studio as copyright holder, a one-line README licence section, and MIT in the Python and all three Rust package manifests. The GPL-3.0 notice beside the test-only 17Lands parser fixture is unchanged.
- Preserve the exact LICENSE bytes during release staging, with regression coverage, and run release CI for licence-only changes.
- Raise the maturin minimum to 1.9.3 across build and development requirements so packaging supports SPDX metadata and licence-file handling. Resolved dependency versions remain unchanged.

## Checks

Validation at `720c5dea6c5c38b2907bdb9969fa94a14554b950`, November 11, 2026:

- Release-tooling tests: 23 passed. Ruff lint/format, release version validation and diff whitespace checks passed.
- All four CI wheels (Linux x86-64 and macOS arm64, CPython 3.12/3.13) and the shared sdist passed complete-inventory validation and strict Twine checks. Every artifact contains the exact current LICENSE bytes, `License-Expression: MIT` and `License-File: LICENSE`.
- An additional local CPython 3.13 macOS wheel built from the current CI sdist and passed the same payload and metadata checks.
- Reused the earlier isolated locked dependency sync and installed-wheel smoke evidence; their code and dependency inputs are unchanged.

Release preflight, all four package jobs and both Linux install jobs passed. The macOS install jobs and some broader CI checks were still running at gate completion. No full CI pass is claimed. After merge, check GitHub’s licence detection; after package publication, verify PyPI displays MIT. README’s relative licence link does not resolve on PyPI.
