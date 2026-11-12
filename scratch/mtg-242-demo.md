# MTG-242 demo — 2026-11-12 (headless, against `36196e1f5`)

No person was present. The decision was made standing in for Jack Heart under
the Task's review-only boundary (no merge, no training run).

## Shown through the real paths

- Native rules, debug build (`cargo test --workspace -- restrict scarecrow
  supportive prism powerstone lotus`): every matching test passes, 0 failed.
  These cover restrictions on previously unrepresented subtypes (the Ogre cast),
  legal and illegal purposes (Scarecrow Guide paying its source's activation but
  not a spell), shared taps (Supportive Parents), Powerstone's exclusion, variable
  output (White Lotus Tile) and nested Prism/Treasure completion through
  Commands, forks and stale-command rejection.
- Python, using the extension built from `d0f92723a`, the last commit that
  touches `managym/`: `tests/semantic/test_restricted_mana_input.py` gives
  7 passed. It covers a pool the columns can't tell apart (A), which is refused
  with a typed error, the encodable pool B, and native/Python parity.
- PR #496 CI at `36196e1f5`: Engine Budget, Rust Tests, Python Unit, both
  conformance suites, Academy UI/Integration, the evaluation and long
  integration jobs, and packaging all pass. `Integration (core)` failed:
  `tests/training/test_training_throughput_sizes.py::test_a_size_cell_trains_through_the_real_worker_on_cpu`
  hit the conftest 5-minute limit while waiting on the worker subprocess. Main
  passes the same job on its last five runs. The branch changes only
  `manabot/env/observation.py` in Python, and Engine Budget shows no slowdown.
  The failed jobs were rerun. The result is recorded below.

## Takeaway

The typed restriction/context model works end to end. Arbitrary-restriction
policy rows (questions.md 1) and general `Except` (questions.md 2) stay
open questions for a person to decide. Neither blocks the reviewable PR.

## CI rerun

The failed jobs were rerun on 2026-11-12. In `gh run view 43287154783`, Integration (core)
passed on attempt 2 (20:44–21:01 UTC), and every PR #496 check now passes.
On attempt 1, the job hit the 300 s per-test limit in
`test_a_size_cell_trains_through_the_real_worker_on_cpu`. The parent process
was blocked in `process.wait()` on the CPU training worker. The second leg
(`a4000-a384l8-k1`) trains a 384-wide 8-layer model on a shared runner. The
cause was not proven. Nothing points at this branch: its only Python change is
the encoder's refusal check, and Engine Budget shows no slowdown. If that cell
times out again on any branch, profile its worker log before rerunning.

## Gate (2026-11-12, at `36196e1f5`)

- `cargo fmt --manifest-path managym/Cargo.toml --all --check`: clean.
- `uv run --extra dev pytest tests/semantic/test_restricted_mana_input.py -q`: 7 passed.
- Rust debug suite, clippy, Engine Budget, conformance and Python suites: CI
  green at this tree (managym unchanged since `d0f92723a`).
- `ruff check` reports import-order (I001) findings in `manabot/env/observation.py`
  and `tests/belief/test_state.py`. Those lines are unchanged from main, and CI does
  not enforce the rule.
- The branch is 3 commits behind main. They are training-only and merge cleanly
  (`git merge-tree`); no sync was done, so CI does not need to rerun.

## Decision pass 1 (2026-11-12)

Since `36196e1f5`, only this note changed. PR #496 shows 23 checks passed and 2 skipped,
and `tests/semantic/test_restricted_mana_input.py` passes again (7 passed). The
decision was advance to the review stop. Questions 1 and 2 stay with a person.
