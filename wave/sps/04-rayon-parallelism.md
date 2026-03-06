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
