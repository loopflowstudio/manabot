# Reversible fixture/runtime choices — 2026-11-12

- The experiment/notebook acceptance test needs real training and complete-game
  monitoring, but not complete-game training. Use one-seat exact-quota collection
  for its eight-row, two-stream workload; keep four attempts and all assertions.
- Hosted CPU Integration needs no CUDA-built Torch. Select the existing
  `play-runtime` group with default groups disabled. Revert by removing those flags;
  dependency definitions and lockfile stay unchanged.

Both choices passed Ubuntu debug/xdist in dispatch 43348710097. Evidence and
remaining delivery are in `ci-under-8-minutes.md`. The superseded wheel-cache
choice is preserved at `e22af480b:scratch/questions.md`. No open product choice.
