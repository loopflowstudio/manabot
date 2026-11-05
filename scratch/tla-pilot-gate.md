# TLA results gate — 2026-11-05 UTC

Comparison: PR #430 targets main; supplied checkout base and current local
origin/main are fe83a6b068dace2187613e1bf124f9677b077ff3. Reviewed committed,
working-tree and untracked changes at HEAD 272ec8c08. No executable code changed.

## Validation plan

- `experiments/data.manifest.json`: 315 new v10 artifact entries. Run the focused
  artifact roundtrip/integrity suite and the mandatory no-tracked-data guard;
  use the real fetch CLI to verify existing bytes; compare prior manifest entries
  and check the report's local links and history references.
- `experiments/tla-trophy-pilot.md`: results and historical summary. Reuse the
  preceding demo's exact real-analysis reproduction and receipt/count audit:
  analysis code, receipt bytes and frozen protocol are unchanged. Inspect result
  claims, qualification, retrieval command and durable historical references.
- `.lf/config.yaml`, Wave GOAL/config migrations, Experiments MEMORY and scratch:
  inspect consistency and preserve prior contributions. The Wave binding migration
  is inherited commit 51b4bd274. Gate does not select a new Wave write.
- Repository selector `.github/ci/select-tests.sh` selects only Integration.
  Leave its broad main/long matrix to CI; no Python/Rust/UI source changes warrant
  recompilation, formatting or whole-package local tests. No additional benchmark,
  training, rental or UI proof is selected.
- Run `git diff --check` on the complete branch and final working changes.

## Review

The report distinguishes launcher success, wall-budget pause and incomplete
scientific acceptance. Endpoint games, failures, missing baseline/milestones and
estimated costs agree with the preceding demo receipts. Corrected the inventory
sentence to state its observation time; replaced a local-only unattached blob
reference with a reachable earlier v10 report commit for fresh-checkout reviewers.
No frozen evidence changed.

## Results

- `uv run --extra dev --extra artifacts pytest tests/training/test_experiment_data.py tests/infra/test_experiment_data_untracked.py -q` — 6 passed in 0.13s.
- `uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/gpu-v10-20261105` — all 315 present files verified; no downloads or evidence changes.
- `uv run python` manifest/link audit — 315 additions exclusively under v10;
  all preexisting entries/store/schema unchanged; all 28 report links resolve;
  both historical report references exist in reachable commit history.
- `git diff --check origin/main` and `git diff --check` — passed.
- Prior demo analysis reproduction is reused: exact equality with retained
  analysis, 92/288 valid games, 196 missing, 64 completed endpoint games. No
  executable, protocol or evidence input changed during this gate.
- PR #430 CI at HEAD 272ec8c08 — Detect Changes passed; both selected Ubuntu
  Integration groups are still running. Their broad validation is deferred to
  CI, not marked passed. Other jobs skipped as selected by the repository filter.

Existing PR copy agrees with the final scope and observed limitations; delivery
can include these additional focused checks. No cached PR copy was created.
No unresolved local gate defect or timeout. Scientific limitations, two native
rules investigations and final billing remain explicitly outstanding. This gate
does not complete MTG-213, claim scientific improvement or select the full-TLA
successor. No background process was started.
