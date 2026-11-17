# MTG-329 evidence

Full earlier commands/results: 3670249eb3:scratch/implementation.md. GPU method,
provider attempts and results: experiments/reports/etude-execution.md.

## Learning comparison method — 2026-11-17, before runs

Exploratory question: does moving execution change uninterrupted learning?
Compare origin/main 3afd92d8f3 with this branch, fresh local baseline runs,
CPU float32, seed 197, three updates, two streams and 16 transitions (complete
actual games, both seats), identical horizon-3 damped-policy schedule and
width-64/depth-2 FiLM model. Compare every raw and EMA tensor, including the
initial policy if available. A bitwise match settles only this short
uninterrupted case; real divergence triggers the planned strength comparison.
The main source is a temporary git archive, not another worktree. Use the same
compiled native extension (engine implementation unchanged), Torch and host;
main keeps its pinned Etude SDK. No runtime/RNG recovery equivalence is claimed.
CPU cost: no rental. Combined remote measurement cap remains USD 10; no remote
measurement has run. Resume strength and CUDA speed/admission remain separate.

## Report cleanup and CPU comparison — 2026-11-17

- Deleted the unreachable flamegraph, phase table and per-run synthetic profile
  hierarchy, plus elapsed/phase/cost payload fields. Removed notebook extra,
  nbformat, matplotlib and their now-exclusive lock dependencies. Updated both
  integration workflows, release install, development guide and AGENTS/CLAUDE
  command together. Historical install commands in the saved-regime guide
  still name the historical revision's notebook extra.
- `uv run --locked --extra dev --extra artifacts pytest tests/training/test_time_cohort.py -q`:
  8 passed (0.26s), after deleting one assertion on the removed phase fields.
  `uv run --isolated --locked --extra dev --extra artifacts --extra reports --extra evaluation pytest tests/training/test_time_cohort.py -q`:
  8 passed (0.81s), with the reduced dependency graph. Ruff check and format
  check pass for the three affected Python files.
- CPU method above completed against main 3afd92d8f3 and branch 9421c945d9,
  CPython 3.13.15, Torch 2.15.0, NumPy 2.4.2, native world w17. All 117
  initial raw tensors match bitwise. After three updates, all 117 raw and
  all 117 EMA tensors match bitwise (zero unequal; maximum absolute error 0).
  Both sides: 6 complete games, 1,421 environment decisions/learner transitions,
  268 optimizer updates. Branch cumulative optimizer updates: 89, 148, 268.
  Step 2 is unnecessary for this short uninterrupted case under the approved
  method; no broader strength or resume equivalence is implied.
- Main used unchanged local_smoke through execute_regime; branch used the real
  etude train CLI. Full commands and temporary paths: 3670249eb3:scratch/implementation.md.
- Required isolated artifact/Etude/recipe integration: 12 passed (15.73s).
  Report JavaScript and diff checks passed. Provider capacity/termination
  evidence is in experiments/reports/etude-execution.md; the original command
  transcript remains at the git reference above.

## Latest dependency and focused checks

Released Etude 0.4.1 replaces git pin in requirements, lock, release installs,
metadata validation and docs. Isolated artifact/SDK/source/recipe and release
checks: 50 passed, one stale deleted-module assertion failed; after replacing
that assertion with baseline.py, isolated test_release.py: 39 passed. Installed
CPython 3.12/3.13 smoke passed in earlier record. Local real CLI resume and
16-game replay preflight passed; no CUDA claim from those checks.

Current implement read CI run 44136662771: only failure was extra discovered
experiment etude_execution (1 failed, 197 passed). Added it to expected set.
`uv run --no-sync pytest tests/eval/test_experiment_files.py -q -k etude_finds_each_current_experiment_and_nothing_else`: 1 passed, 12 deselected (8.54s).

Latest retry (18:57–19:19 UTC): CUDA check passed on Vast 4090; durable-write
stall prevented any update (faulthandler: Etude run.py:412 os.fsync). Rental
terminated, provider lookup confirmed absent; estimate ~USD 0.14. Follow-up
checks found no alternate capacity. Full evidence: experiment report. PR #500
body corrected for deleted regime readers and actual GPU attempts; CI rerun
remains for gate. Training precision guide now matches managed CUDA selection.
