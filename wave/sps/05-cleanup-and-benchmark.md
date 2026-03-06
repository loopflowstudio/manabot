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

### 3. Final benchmark

Run the full measurement protocol from `wave/sps/README.md`:

```bash
# Breakdown at multiple scales
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
    python scripts/bench_breakdown.py --num-envs $n --steps 2048 --with-inference
done
```

Publish results in a table: num_envs vs env-only SPS vs training SPS.
Compare against the baseline numbers in the wave README.

### 4. Documentation

Update README training section with new SPS numbers.

## Verification

- Full verification ladder (steps 0-4) passes
- Benchmark shows target SPS improvement
- No code paths reference AsyncVectorEnv
- `ruff check` and `cargo clippy` clean

## Done when

One env backend, clean codebase, published benchmark showing 10x+
improvement over the baseline in the wave README. Wave complete.
