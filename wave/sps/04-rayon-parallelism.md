# 04: Rayon Parallelism

## Finish line

`VectorEnv::step_into_buffers()` steps N games in parallel using Rayon.
GIL is released during the parallel section. Linear scaling up to
available cores.

## Context from sprint 03

Sprint 03 built zero-copy buffer writes. Key details that affect this sprint:

- **Buffer access uses `pyo3::buffer::PyBuffer`** (not the `rust-numpy` crate,
  which was unavailable in the sandboxed build). `ObservationBuffers` stores
  `Py<PyAny>` references. `with_write_buffers()` acquires `PyBuffer` handles
  and creates `&mut [f32]` slices via `mutable_slice_from_buffer()`.

- **GIL is NOT released during stepping.** The current implementation holds the
  GIL for both game stepping and buffer encoding. `Python::allow_threads`
  requires the closure to be `Ungil` (i.e. `Send`), but `MutexGuard<VectorEnv>`
  is not `Send`. Sprint 03 works around this by `drop(env)` before encoding,
  but stepping itself still holds the GIL.

- **Buffer write path**: `step_into_buffers` → lock env → `env.step(&actions)`
  → drop lock → `write_step_results_into_buffers` → `with_write_buffers` →
  `encode_into` per env. The encoding loop is sequential over envs.

- **`ObservationFieldSlices`** holds `&mut [f32]` / `&mut [i32]` for all 15
  observation fields. `encoded_row_mut()` returns an `EncodedObservationMut`
  pointing into the appropriate slice for a given env index.

- **Key files**: `managym/src/python/vector_env_bindings.rs` (buffer protocol),
  `manabot/env/rust_vector_env.py` (Python wrapper).

## Changes

### 1. Restructure ownership for GIL release

The `Mutex<VectorEnv>` pattern prevents `allow_threads` because
`MutexGuard` is not `Send`. Options:

- **Extract games into a `Vec` before `allow_threads`**: Lock, take the
  games vec out (replace with empty), release lock, step in
  `allow_threads`, put games back.
- **Use `UnsafeCell` or raw pointer**: More complex but avoids the
  lock/unlock dance. Document safety invariant (single-threaded access
  from Python).
- **Remove the `Mutex` entirely**: `PyVectorEnv` is `#[pyclass]` with
  `&mut self` methods — PyO3 already guarantees exclusive access. The
  Mutex may be unnecessary.

Pick the simplest approach that compiles and is safe.

### 2. Rayon integration

Add `rayon` dependency to `managym/Cargo.toml`.

```rust
// Inside allow_threads:
// Step all games in parallel
self.envs
    .par_iter_mut()
    .zip(actions.par_iter())
    .for_each(|(env, &action)| {
        env.step(action);
    });
```

Game stepping is embarrassingly parallel — each game is independent.

### 3. Buffer writes after parallel stepping

Two options for encoding after parallel stepping:

- **Sequential encoding (simpler)**: Step all games in parallel (GIL
  released), then encode sequentially with GIL held (current pattern).
  This parallelizes the expensive part (game stepping) without changing
  the buffer write path.

- **Parallel encoding (more complex)**: Use `SendSlice` wrappers around
  raw pointers to make buffer slices `Send`, then encode in the same
  Rayon parallel section. Requires `unsafe` for the pointer wrappers
  but buffer slices are disjoint per env.

Start with sequential encoding. Profile to see if encoding is a
bottleneck before adding parallel encoding complexity.

### 4. Thread count configuration

Rayon respects `RAYON_NUM_THREADS` env var natively. No code needed
for configuration. For training on many-core machines, users can set
this to leave room for PyTorch.

## Measurement

Follow the wave measurement protocol (see `wave/sps/README.md`).

### Before

```bash
for n in 1 16 64; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference
```

Record `rust_step` percentage and absolute time. This is the phase
being parallelized. Save to wave results tracker.

### After

```bash
# Scaling sweep (broader range to show parallelism)
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference
```

`rust_step` time should scale sub-linearly with num_envs (near-constant
up to core count, then linear).

### A/B

```bash
# Rayon vs legacy
python scripts/bench_ab.py --a rust --b async --num-envs 64 --rounds 5

# Rayon vs single-threaded Rust (isolates this sprint's contribution)
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 64 --rounds 5
```

The second comparison uses `RAYON_NUM_THREADS=1` on the B side to
measure pure parallelism gains. Rayon respects this env var natively.

### Save results

Update the results tracker tables in `wave/sps/README.md` with S04
numbers. Include the scaling curve data.

### Gate

- Env-only SPS > 100,000 at 16 envs
- Near-linear SPS scaling from 1 to core_count envs
- Training SPS measured and recorded (with inference)
- `python -m manabot.verify.step0_env_sanity` passes
- No data races (ThreadSanitizer clean in debug builds)

## Done when

Parallel stepping works. SPS scales with core count. Documented
scaling curve (envs vs SPS).
