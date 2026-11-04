# TLA resource-recovery gate — 2026-11-04

Reviewed HEAD `4fb8109628a7597cedebe35a4296d5f1aea7cdbf`, PR #426 against
its `main` base. Local main `510b5ec35` separates the repair from inherited
Wave metadata; preserve that metadata. No runtime changes selected by review.
The untracked demo record is part of the review context, not executable input.

## Affected paths and validation plan

- `managym/src/agent/vector_env.rs`, `managym/src/python/vector_env_bindings.rs`,
  `manabot/managym/__init__.pyi`: typed capacity isolation, strict reset/buffer
  boundaries and binding compatibility. Debug native vector tests and retained
  Airship scalar/vector/sharded parity; fmt and all-feature Clippy.
- `manabot/env/failures.py`, `manabot/training/rollout/{collection,producers}.py`,
  `manabot/training/state/game_errors.py`: replay, both-seat discard, unaffected
  stream preservation, ledger restart and threshold. Recovery, game-error
  recovery/ledger, active-training, observation and action-cutoff suites.
- `experiments/tla_pilot_v10.py`, analysis import, v10 protocol/regime and
  associated eval tests: frozen admission, fresh initialization, discovery and
  paired evaluation denominators. Regimes, experiment discovery, TLA pilot,
  current-world admission/monitoring/choice-separability suites.
- `experiments/data.manifest.json`, retained replay fixture and report: artifact
  integrity, historical preservation and untracked experiment-data guard.
- README, training guide and scratch: link/configuration consistency and diff
  whitespace; no UI implementation changed.

`.github/ci/select-tests.sh` selects all nine job families for the native binding
change. This is a job selector, not a local focused runner or content cache.
CI owns the broad integration/browser/package/performance matrix.

## Reused results and remaining checks

Publication validation and its logs are retained in
`tla-pilot-postmerge-verify.md`. Inspection of `git diff 62f20f8c8 HEAD` shows
only scratch review notes changed since the tested source. The working tree
adds only the demo record and this gate record. No executable/test/config input
changed, so affected suite results remain applicable; do not repeat the matrix.

- Required focused Python plan: 320 passed in 447.83s, retained log
  `/tmp/mtg213-publish-python.log`.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2` (managym): 969 passed,
  two ignored; retained log `/tmp/mtg213-publish-rust.log`.
- `cargo fmt --check`; `cargo clippy --all-targets --all-features -- -D warnings`
  with absolute root-venv `PYO3_PYTHON`: passed in publication validation.
  The initial relative-interpreter resolution error was corrected, not a test
  timeout. No unresolved timeout is recorded.
- Real v10 `frozen()` admission: passed; 134 decks, w17, permanent64, v7,
  matched first64 and .25/100/100 recovery. Artifact validation checked all
  166 added files and retained historical manifest/protocol/regime bytes.
- Demo's retained-Airship/collector/threshold selection: 16 passed,
  78 deselected in 3.05s; `/tmp/mtg213-demo-proof.log`.

Gate performs only outstanding changed-file formatting/whitespace checks and
reads current CI. No GPU allocation, training process or server starts here.
Final-source freeze, fresh inventory/budget reconciliation, GPU execution,
trained export, paired evaluation, accounting and teardown remain verification
acceptance. Merge does not complete MTG-213. V9 evidence stays failed/incomplete.

## Decision

Advance to delivery; no code defect or unresolved product choice found.
`uv run --extra dev ruff format --check` over all eleven changed Python files
passed (11 already formatted). `git diff --check origin/main...HEAD` and
`git diff --check` passed. No implementation, protocol or evidence bytes changed.

PR #426 CI at the reviewed HEAD: Rust Tests, Python Unit Tests, both package
builds, Academy Integration, Protocol v1 Conformance, Semantic Kernel
Conformance, Academy UI and preflight passed. Install smoke (Linux/macOS),
main/long integration, Clean-machine Play and Engine Budget remain in progress;
these are pending CI results, not local passes or waived requirements.
Delivery must inspect their final results before landing.
