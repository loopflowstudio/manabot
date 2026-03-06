# 02: Rayon Parallelism

**Finish line:** `PyVectorEnv::step_into()` steps N games in parallel using Rayon with GIL released. Linear scaling up to available cores.

## Current state

The architecture was simplified in the latest iteration. Key details:

- **`PyVectorEnv`** is a `#[pyclass]` struct owning `Vec<EnvSlot>` directly —
  no Mutex wrapper. PyO3's `&mut self` methods guarantee exclusive access.

- **Buffer writes use Python `__setitem__`** calls (via `PyAny::set_item`),
  not raw pointer access. Correctness-focused but serializes on the GIL.
  To parallelize, buffer access must move to raw pointers (via `numpy` crate
  or `PyBuffer` protocol) so writes can happen inside `allow_threads`.

- **`ObservationEncoder`** (in `vector_env.rs`) encodes observations into
  `EncodedObservation` structs (owned `Vec<f32>` fields), which are then
  written into numpy buffers one field at a time.

- **Stepping is fully sequential**: `step_into` iterates `envs` with a for
  loop. Each env step + encode + buffer write happens in sequence.

- **Key files**: `managym/src/python/vector_env.rs` (PyVectorEnv, encoder,
  buffer writes), `manabot/env/env.py` (Python VectorEnv wrapper).

- **`num_threads`** is accepted and validated but unused.

## Changes

### 1. Switch buffer writes to raw pointers

Before parallelizing, buffer writes must bypass the GIL. Options:

- **Add `numpy` crate** (`numpy = "0.24"`): provides `PyArray` types with
  `as_slice_mut()` for raw access. Extract pointers while GIL is held,
  pass them into `allow_threads`.
- **Use `PyBuffer` protocol**: `pyo3::buffer::PyBuffer` gives raw pointer
  access without an extra crate dependency. More verbose but works.

Either way: extract raw `*mut f32` / `*mut bool` / `*mut i32` pointers
from numpy arrays while holding the GIL, validate shapes/dtypes/contiguity,
then write directly inside `allow_threads`.

### 2. Rayon integration

Add `rayon` dependency to `managym/Cargo.toml`.

```rust
fn step_into(&mut self, py: Python<'_>, actions, obs_buffers, ...) -> PyResult<()> {
    // Extract raw buffer pointers while holding GIL
    let buf_ptrs = extract_buffer_pointers(py, &obs_buffers)?;

    py.allow_threads(|| {
        self.thread_pool.install(|| {
            self.envs.par_iter_mut()
                .zip(actions_slice.par_iter())
                .enumerate()
                .for_each(|(i, (slot, &action))| {
                    // Step game
                    let result = slot.env.step(action);
                    // Encode + write directly into buffer slice i
                    encode_into_buffer_slice(&result, &buf_ptrs, i);
                });
        });
    });
    Ok(())
}
```

Each env writes to a disjoint slice — no synchronization needed.
Wrap raw pointer + offset in a `BufferSlice` struct that implements
`Send` and documents the safety invariant.

### 3. Thread count configuration

Use a dedicated Rayon `ThreadPool`. Default:
`max(1, min(num_envs, num_cpus::get_physical().saturating_sub(1)))`.
Honor the existing `num_threads` constructor parameter.

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
