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

Run the full measurement protocol from `wave/sps/README.md` at all
scales:

```bash
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
    python scripts/bench_breakdown.py --num-envs $n --steps 2048 --with-inference
done
```

Fill in the final column of all results tracker tables.

### 4. Ablation

Run the ablation protocol from `wave/sps/README.md` to attribute
gains to each sprint. This isolates individual contributions:

```bash
# Full wave impact: Rust vs AsyncVectorEnv
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5
python scripts/bench_ab.py --a rust --b async --num-envs 64 --rounds 5

# Sprint 03 contribution: zero-copy vs Python-side encoding
python scripts/bench_ab.py --a rust --b rust-python-encode --num-envs 16 --rounds 5

# Sprint 04 contribution: Rayon vs single-threaded
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 16 --rounds 5
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 64 --rounds 5

# Scaling curve
for n in 1 4 16 64 128; do
    python scripts/bench_ab.py --a rust --b async --num-envs $n --rounds 3
done
```

Fill in the ablation table in `wave/sps/README.md`. The table should
show cumulative speedup at each sprint, so the narrative is clear:
baseline → Rust encode → zero-copy → Rayon → final.

### 5. Documentation

Update README training section with final SPS numbers and scaling
curve.

## Verification

- Full verification ladder (steps 0-4) passes
- Benchmark shows target SPS improvement
- Ablation table filled in with all configurations measured
- No code paths reference AsyncVectorEnv
- `ruff check` and `cargo clippy` clean

## Done when

One env backend, clean codebase, published benchmark showing 10x+
improvement over the baseline. Ablation table shows where the gains
came from. Wave complete.
