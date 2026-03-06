# Rayon Parallelism — Not Now

## Problem

The wave item proposes parallel stepping via Rayon to scale SPS with core count. The premise is that game stepping is the bottleneck and parallelizing it across cores will yield linear speedup.

This premise is wrong for the current codebase. Per-env step cost is ~4.4µs. At that granularity, Rayon's thread distribution overhead (work stealing, task scheduling, cache line bouncing) exceeds the compute saved. This was empirically verified in S02: multi-threading was performance-neutral to slightly negative across 1–256 envs on M4 Max.

Meanwhile, the wave's SPS targets are already met:
- Env-only SPS: **189,000** at 16 envs (target: >100,000)
- 10x improvement over baseline (19K): **achieved**

## Approach

**Skip this sprint.** Rayon parallelism is not beneficial at current per-step costs. The infrastructure to enable it later (GIL release via `py.allow_threads`, `SendSlice` for `Send`-safe buffer access, no Mutex on `VectorEnv`) is already in place. When per-step cost increases — more complex cards, heavier observation encoding, larger game states — rayon can be added with minimal code changes.

The next work is not "find one more SPS optimization." It is to **close the SPS wave cleanly**:

- remove stale product/training references to pre-Rust vectorization
- keep only the code and benchmarks that are still useful long-term
- make benchmark scripts reusable beyond this sprint rather than hard-coded to one ablation pass
- run and record the final measurements
- document what changed, what worked, what did not, and why training is now the active bottleneck

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Add Rayon anyway | Clean parallel code, ready for future | Adds a dependency and complexity for zero measurable gain. Rayon's thread pool competes with PyTorch for cores during training. |
| Rayon with larger batch sizes | More work per dispatch might amortize overhead | Tested up to 256 envs in S02. Still no gain — the per-step cost is the issue, not batch size. |
| std::thread with pinned cores | Avoids Rayon overhead, manual thread management | Also tried in S02. Same result — the work per env is too cheap. |

## Key decisions

1. **Do not add rayon.** The empirical evidence is clear. Adding it would be engineering theater — complexity without benefit.

2. **Keep the `SendSlice` infrastructure.** It's already there, costs nothing at runtime, and makes future rayon integration a one-line change (`iter_mut` → `par_iter_mut`).

3. **Revisit trigger.** Rayon becomes worthwhile when per-env step cost exceeds ~50µs (roughly 10x current). This will happen naturally as the card pool grows and observation encoding gets richer. Document this threshold in the wave README.

## Scope

- In scope: Documenting the decision not to parallelize; reframing the final SPS sprint around cleanup, reusable benchmarking, and publishing results/lessons learned
- Out of scope: Actually adding rayon, chasing further env-only SPS gains for their own sake, game engine optimizations, new card implementations

## Done when

The wave docs clearly state that rayon was investigated and removed, and the closeout sprint is framed around finishing the SPS initiative rather than squeezing one more micro-optimization out of the env.

## Measure

N/A — this sprint produces no performance change by design. The S02 measurements already captured the rayon A/B comparison.
