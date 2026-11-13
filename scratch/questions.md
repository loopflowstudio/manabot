# Fix-forward fixture choice — 2026-11-12

Assumption: this experiment/notebook acceptance test needs real training and
live complete-game monitoring, but does not require complete-game training.
Use the existing one-seat exact-quota collector for its eight-transition
workload; keep all four attempts and every existing assertion. This bounds
the fixture without changing production defaults or the shared WDL baseline.
Ubuntu debug/xdist run 43345024882 validated the fixture in 151.70 s; its
549-second job still missed the 480-second target because of installation.

# Integration Torch download — 2026-11-12 (QA)

Dispatch 43345024882 at `4db5cd64b` was green, but the experiment job took 549 s:
181 s installing (peers 17–18 s), of which most was CUDA/nvidia wheel downloads
after a pruned setup-uv cache hit. QA identified CPU Torch or retaining the
unpruned cache as alternatives. Only switching to CPU Torch changes the build;
cache retention preserves it.

Implementation choice: retain the existing Torch build with `prune-cache:
false`. All groups restore; `model` alone saves because every group installs
the same extras. This is the simpler reversible repair; it avoids redundant
large uploads. Ubuntu cold/save and warm/restore measurements remain necessary
before claiming the 480-second target.

# CPU Torch for Integration — 2026-11-12 (concept review)

Assumption: Integration does not need the cu132 Torch build on a hosted CPU
runner. Supersedes the cache-retention choice above; evidence and direction are
in `ci-under-8-minutes.md`. Reversible by removing two `uv sync` flags.
