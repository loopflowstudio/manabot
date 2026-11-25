---
extends: implement
---
## Training and evaluation performance

**One profile-backed slice per session.** Implement only improvements backed by
the recorded timing/profile in `scratch/plan.md`. Preserve identical behavior
digests and learning semantics: do not change precision, policy lag, batch
composition, observation capacities or loss denominators to claim a speed win
without separate learning evidence.

**Preserve architecture boundaries.** Library code receives a framework backend
and never imports `manabot.backends.*` outside composition roots; `Prepared`
remains the sole policy input across backends; keep `manabot` free of
`managym.academy` imports and keep evaluation children Torch-free.

**Measure on the same machine with matched workloads.** Re-run the exact
declared benchmark (`scripts/bench_training_update.py` or the plan's matched
workload, writing to ignored `.runs/` paths) with identical recipe, seed,
device and capacities, checking behavior digests and per-phase timings
(separating single-seed latency from packed throughput). Revert changes that
fail to improve the target metric or alter digests, and record negative results
in `scratch/plan.md`.
