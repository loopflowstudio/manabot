# 03: Cleanup and Benchmark

**Finish line:** Stale `AsyncVectorEnv` references removed. SPS benchmark published showing 10x+ improvement over first-light baseline (~1,600 SPS).

## Context from shipped sprints

- **Zero-copy buffers (S01):** `PyBuffer` protocol with `mutable_slice_from_buffer()` for raw `&mut [f32]` access. Key files: `vector_env_bindings.rs` (`ObservationFieldSlices`, `WriteBuffers`), `rust_vector_env.py` (pre-allocated numpy buffers).
- **Parallel stepping (S02):** Uses scoped `std::thread` workers, not Rayon (crate unavailable in offline sandbox). `SendSlice` wraps raw buffer pointers for cross-thread use. Thread count precedence: explicit `num_threads` > `RAYON_NUM_THREADS` env var > auto (`max(1, min(num_envs, available_parallelism - 1))`). `Env` is confirmed `Send`. Swapping to Rayon later requires no Python API changes.
- **Thread count tuning unknown.** Auto default uses `std::thread::available_parallelism()` (logical cores, not physical). Whether `-1 core` is optimal under CPU inference remains empirical — measure during benchmarking.
- **Python-side verification was blocked** in S02 (no pytest, pip too old for editable installs). Python tests need to run during this sprint.

## Current state

- `VectorEnv` (`manabot/env/env.py`) is the Rust-backed implementation
  using `managym.VectorEnv` via `step_into()` / `reset_into()` with pre-allocated
  numpy buffers.
- `AsyncVectorEnv` references remain only as comments (env.py:35-36, train.py:461).
- The single-env `Env` class still exists for tests and simulation — it stays.
- Python `ObservationSpace.encode()` still exists for parity testing.

## Changes

### 1. Remove stale AsyncVectorEnv references

Clean up comments referencing AsyncVectorEnv in `env.py` and `train.py`.
Remove any dead code paths related to subprocess-based vectorization.

### 2. Final benchmark

Run the full measurement protocol from `wave/sps/README.md` at all
scales:

```bash
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
    python scripts/bench_breakdown.py --num-envs $n --steps 2048 --with-inference
done
```

Fill in the final column of all results tracker tables.

### 3. Ablation

Run the ablation protocol from `wave/sps/README.md` to attribute
gains to each sprint. This isolates individual contributions:

```bash
# Full wave impact: Rust vs AsyncVectorEnv
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5
python scripts/bench_ab.py --a rust --b async --num-envs 64 --rounds 5

# Zero-copy contribution: Rust encode vs Python-side encoding
python scripts/bench_ab.py --a rust --b rust-python-encode --num-envs 16 --rounds 5

# Parallelism contribution: multi-thread vs single-thread
# Note: implementation uses std::thread but respects RAYON_NUM_THREADS env var
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 16 --rounds 5
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 64 --rounds 5

# Scaling curve
for n in 1 4 16 64 128; do
    python scripts/bench_ab.py --a rust --b async --num-envs $n --rounds 3
done
```

Fill in the ablation table in `wave/sps/README.md`.

### 4. Python verification

Run the Python test suites that were blocked during S02:

```bash
pip install -e managym && pytest tests/env/ tests/verify/ -v
python -m manabot.verify.step0_env_sanity
```

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
