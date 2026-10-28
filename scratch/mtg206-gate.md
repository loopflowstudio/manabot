# MTG-206 gate — 2026-10-27

Comparison: checkout base `37e5ccd8d` through `a47e1fd00`, initially clean.
Accepted design and previous gate/demo records survive at
`bb3091113:scratch/mtg206.md` and `bb3091113:scratch/mtg206-demo.md`.
Delivery cleanup removed those added scratch files; it did not invalidate their
results. This pass reviews the full diff and checks the subsequent import move.

Validation plan:

- `card_evaluation.py`, `cards.py`, and `mtg206_cards.py` moved the verification
  schedule out of the reporting CLI: run the complete card-evaluation suite,
  Arena-log conformance, and recipe discovery tests for their shared callers.
- Run CI-selection and release-contract tests for the Python API move. CI's
  path classifier selects every group for this branch's shared configuration
  changes; it is not a local test runner and has no content-keyed result cache.
  The full Linux matrix remains CI-owned; run the affected local suites only.
- Reuse the previous 27-test isolated dependency result: `pyproject.toml`,
  `uv.lock`, and the storage/SDK/source/job implementations are unchanged.
  Reuse the prior MTG-186, migration, long-eval/report, Etude evaluation,
  local CLI and package-boundary checks for unchanged implementations.
- Run Ruff check/format over changed Python paths, lock validation and diff
  whitespace checks. CI has no Python type-check job.
- Reuse original-cohort timings, all 1,452 variant comparisons, headless page
  script checks and 96/96 L4 replay audit: query, metrics, page, evidence and
  dependency files are byte-identical to the prior gate. No new rental,
  conversion, timing run or visual approval is needed or claimed.

No implementation defect found in the reviewed diff. Retain the acceptance
measurements in the card-evaluation documentation so delivery scratch cleanup
does not leave their only discoverable account in history.

Results:

- `uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_card_evaluation.py tests/eval/test_arena_log_conformance.py tests/training/test_regimes.py tests/infra/test_ci_selection.py tests/infra/test_release.py -q --durations=8` — 108 passed in 249.98 s; no timeout. Native checkpoint capture (96.40 s) and evaluation/resume/delivery/audit (77.99 s) were the slowest checks.
- `git diff --name-only 37e5ccd8 -- '*.py' | xargs uv run --extra dev ruff check --fix` — passed, no edits.
- `git diff --name-only 37e5ccd8 -- '*.py' | xargs uv run --extra dev ruff format` — 16 files unchanged.
- `uv lock --check` — passed, 171 packages resolved.
- `git diff --check` — passed.
- `git diff --no-renames --name-only -z 37e5ccd8 | bash .github/ci/select-tests.sh` — all groups selected by the dependency configuration changes; the broad Linux matrix remains CI-owned and is not claimed passed here. No rendering environment is available; unchanged page-script behavior retains the prior headless proof, not visual approval.

Only documentation and gate notes changed in this pass. No unresolved local
gate blocker remains. Publication and merge are separate steps. All commands
have exited; no background process was started other than the completed tests.
