# SPS: Parallel Stepping — Results

## Finding

Rayon parallelism was implemented, benchmarked, and removed. Per-env
step cost (~4.4µs) is too small for thread distribution overhead to
break even. All configurations (single-thread, multi-thread, rayon)
measured within ~2% of each other at 64 envs (~189k SPS).

## Benchmark data (M4 Max, env-only, zero-copy)

| num_envs | Sequential | Rayon multi-thread | Rayon single-thread |
|----------|------------|-------------------|---------------------|
| 1        | ~82k       | ~74k              | ~82k                |
| 16       | ~189k      | ~189k             | ~209k               |
| 64       | ~189k      | ~195k             | ~227k               |
| 128      | ~180k      | ~172k             | ~225k               |
| 256      | ~163k      | ~163k             | ~205k               |

Controlled back-to-back at 64 envs (3 rounds each):
- Main (std::thread): 189,444 / 189,628 / 190,160 → mean **189,744**
- Rayon multi-thread: 186,897 / 188,237 / 189,134 → mean **188,089**
- Rayon single-thread: 187,133 / 186,934 / 185,004 → mean **186,357**

All within noise. Threading adds no value at current step costs.

## Decision

Removed rayon dependency and thread pool. `parallelize_envs` replaced
with sequential `for_each_env`. `SendSlice` retained for GIL-release
safety. Revisit parallelism when per-step cost increases.
