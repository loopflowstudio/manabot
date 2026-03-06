# 03: Cleanup and Benchmark

**Finish line:** Stale `AsyncVectorEnv` references removed. SPS benchmark published showing 10x+ improvement over first-light baseline (~1,600 SPS).

## Context from shipped sprints

- **Zero-copy buffers (S01):** `PyBuffer` protocol with `mutable_slice_from_buffer()` for raw `&mut [f32]` access. Key files: `vector_env_bindings.rs` (`ObservationFieldSlices`, `WriteBuffers`), `rust_vector_env.py` (pre-allocated numpy buffers).
- **Parallel stepping (S02, removed):** Rayon and `std::thread` parallelism were both implemented and benchmarked. Per-env step cost (~4.4µs) is too low for threading overhead to break even — multi-threading was performance-neutral to slightly negative across 1–256 envs. Removed in favor of sequential `for_each_env`. `SendSlice` remains for GIL-release safety (`py.allow_threads`). Parallelism can be revisited when per-step cost grows.
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

Run A/B comparisons to attribute gains:

```bash
# Full wave impact: Rust vs AsyncVectorEnv
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5
python scripts/bench_ab.py --a rust --b async --num-envs 64 --rounds 5

# Zero-copy contribution: Rust encode vs Python-side encoding
python scripts/bench_ab.py --a rust --b rust-python-encode --num-envs 16 --rounds 5

# Scaling curve
for n in 1 4 16 64 128; do
    python scripts/bench_ab.py --a rust --b async --num-envs $n --rounds 3
done
```

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
