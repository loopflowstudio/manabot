# 04: Rayon Parallelism

## Finish line

`VectorEnv::step()` steps N games in parallel using Rayon. GIL is
released during the parallel section. Linear scaling up to available
cores.

## Changes

### 1. Rayon integration

Add `rayon` dependency to `managym/Cargo.toml`.

```rust
fn step_into_buffers(&mut self, actions: &[i64], /* buffer ptrs */) {
    // Each game writes to its own slice — no synchronization needed.
    self.envs
        .par_iter_mut()
        .zip(actions.par_iter())
        .enumerate()
        .for_each(|(i, (env, &action))| {
            let result = env.step(action);
            // Write observation into buffer[i] slice
            encode_into_buffer(result, buffer_slice(i));
        });
}
```

### 2. GIL release

PyO3's `py.allow_threads(|| { ... })` around the Rayon parallel
section. The GIL is held only for the PyO3 boundary crossing, not
during game stepping.

### 3. Buffer safety

Each game writes to a disjoint slice of the output buffer. No mutex
needed. The buffer pointers are `Send` (raw `*mut f32` with known
disjoint ranges).

Wrap the raw pointer + offset in a `BufferSlice` struct that
implements `Send` and documents the safety invariant.

### 4. Thread count configuration

Default to `num_cpus::get()` or a configurable value. For training
on machines with many cores, may want to limit Rayon threads to
leave room for PyTorch.

## Verification

- Benchmark: `num_envs=1` vs `num_envs=16` vs `num_envs=64` SPS
- Verify linear scaling up to core count
- Training produces same learning curves
- No data races (run under ThreadSanitizer in debug builds)
- Verification ladder passes

## Done when

Parallel stepping works. SPS scales with core count. Documented
scaling curve (envs vs SPS).
