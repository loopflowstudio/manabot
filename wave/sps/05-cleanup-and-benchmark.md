# 05: Cleanup and Benchmark

## Finish line

`AsyncVectorEnv` path removed. `RustVectorEnv` is the only env
backend. Full benchmark published. Verification ladder passes.

## Changes

### 1. Remove AsyncVectorEnv path

- Delete `manabot/env/env.py` `VectorEnv` class (the `AsyncVectorEnv`
  wrapper)
- Remove `gymnasium.vector` dependency from training
- Update `__init__.py` exports

### 2. Simplify trainer

- `Trainer` uses `RustVectorEnv` directly, no env backend flag
- Remove observation encoding from the rollout loop (Rust does it)
- Rollout step is: `actions = agent(obs)` → `obs, rewards, dones = env.step(actions)`

### 3. Benchmark suite

`manabot/verify/benchmark_sps.py`:
- Measure raw env SPS (step without agent inference)
- Measure training SPS (full rollout + gradient)
- Sweep num_envs: 1, 4, 16, 64, 128
- Report: SPS, CPU utilization, memory usage
- Compare against first-light baseline

### 4. Documentation

Update README training section with new SPS numbers.

## Verification

- Full verification ladder (steps 0-4) passes
- Benchmark shows target SPS improvement
- No code paths reference AsyncVectorEnv
- `ruff check` and `cargo clippy` clean

## Done when

One env backend, clean codebase, published benchmark showing 10x+
improvement. Wave complete.
