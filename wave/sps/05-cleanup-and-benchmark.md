# 05: Cleanup and Benchmark

## Finish line

`AsyncVectorEnv` path removed. `RustVectorEnv` is the only env
backend. Full benchmark published. Verification ladder passes.

## Current state

- `VectorEnv` (`manabot/env/env.py`) is already the Rust-backed implementation
  using `managym.VectorEnv` via `step_into()` / `reset_into()` with pre-allocated
  numpy buffers.
- `AsyncVectorEnv` references remain only as comments (env.py:35-36, train.py:461).
- The single-env `Env` class still exists for tests and simulation — it stays.
- Python `ObservationSpace.encode()` still exists for parity testing.

## Changes

### 1. Remove stale AsyncVectorEnv references

Clean up comments referencing AsyncVectorEnv in `env.py` and `train.py`.
Remove any dead code paths related to subprocess-based vectorization.

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
