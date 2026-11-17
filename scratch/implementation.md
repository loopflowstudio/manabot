# MTG-329 evidence

Prior transcripts/commands: 4de3a76ba0, 382c653ec8 and f6a2660745:
`scratch/implementation.md`. No paid/provider/service run or landing.

- 2026-11-14: seed 197, two-update PPO/damped raw/EMA, 87 tensors each,
  bitwise equal to retired 6ce618b8ad. SIGINT/SIGTERM saved update 1; resume
  finished 3, rows [1,2,3]. This predates the revised save API.
- 2026-11-16, Etude d9 CPU: 9 agent/native-card retry tests passed. Across
  Etude processes, AgentError/PanicException retain an invalid score-null game
  and failure journal, then seven clean games. Injected clock and Command caps
  keep score-null rows; RetryRequested escapes with its journal. Ordered masked
  Prepared batching and repeated seeded sampling pass. An omitted all-null
  scalar column means unavailable, never zero. Card/world/monitor/head-to-head:
  87 passed; isolated artifacts/SDK/source/recipe: 12 passed. Installed wheels,
  CPython 3.12.12/3.13.15, Torch 2.15.0, Etude d9: two-update training, reload,
  CLI and Academy smoke passed.
- After main sync, a stale binary caused 69 failures. Rebuilt cp313 extension
  atomically (81s), fixed pre-journal seed reads and two stale worker fixtures;
  eleven focused files: 233 passed. Concept cleanup: 139 passed after correcting
  an assembled policy-import fixture; fresh eval.duel/monitor imports load no
  training.regime/records. First compress eval/boundary check: 89 passed.
- Second compress: 269 passed, 2 skipped, 1 failed (101s). The failing recipe
  report's choice-corpus KeyError also occurred at 9336e7aa08. A real one-update
  baseline CLI run located blocks beside `out` under
  `track/baseline/baseline@197`; report_recipe now reads the run's parent by name.

Not shown: CUDA, real Rust panic or native-hang injection, launched/provider runs,
browser proof, full suite, automatic policy→sampler scheduling. Bounded failure
fixtures prove neither native-hang handling nor in-flight collection recovery.

2026-11-16 recipe-report follow-up:
- Audit failure reproduced at seed 82000, 23-action prefix: waterbend focus is
  card 79 of permanent 103. choice_corpus now maps permanent and stack aliases
  to one descriptor, reads characteristic getters and includes stack rows in
  collision inputs; no public-information fallback or strategic labels.
- The report test looked for learning.json outside the run's `out` and gave
  sixteen scalar games plus replay 90 seconds (exhausted, no replay mismatch);
  it now allows 180 under the unchanged 300-second watchdog.
- Deleted unused host/cgroup/RunPod collection; schema round trips remain.
- Compress pass, no code edits: `uv run --extra dev --extra artifacts --extra
  reports --extra notebook pytest tests/verify/test_choice_corpus.py
  tests/verify/test_learning_regression.py tests/infra/test_host_facts.py -q`:
  9 passed (102.92s). Ruff check/format pass for the changed files; an
  unsorted import in tests/verify/test_input_v10.py is not from this branch.
- `lf context`: scratch stays over 12k tokens because of the protected notes
  (questions.md); no limit changed. Gate and publication remain later steps.
