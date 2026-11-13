# scratch

- `mtg333-plan.md`: active plan, receipts and remaining work; next is slice 6b
  (`105e7e2aa` merged as `9db3f1373`; recapture the CPU baseline), then the CUDA checks.
  `questions.md`: reversible assumptions. `mtg325-torch-backend-verification.md`:
  baseline evidence and limits; MTG-325 merged as `ef28f3a58` (PR #501).
- `equivalence/`: MTG-325's CPU oracle (`run.py`, 3,248 tensors;
  `compare.py`; `updates.py`) and MTG-333 L4 launcher, restored from
  `65fa8c7aa:scratch/`. Captures/patches are MTG-325's; regenerate for this Task.
- `speed/`: MTG-325's L4 harness and four receipts (`run*/compare.txt`).
  Patches are MTG-325's; script paths may name its checkout and need updating.

Full receipts of slices 1–6: `1f634d89b:scratch/mtg333-plan.md`.
Uncommitted notes before compress are preserved under
`/tmp/mtg333-compress-before/`; the plan retains current decisions and results.
