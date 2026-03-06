# 03: Zero-Copy Buffers

## Finish line

Rust writes observations directly into pre-allocated numpy arrays.
No Python objects created per step. No per-step memory allocation.

## Context from sprint 02

Sprint 02 built the Rust observation encoder with a buffer-oriented seam
already in place:

- `encode_into(obs, config, EncodedObservationMut)` writes into caller-owned
  `&mut [f32]` / `&mut [i32]` slices — this is the function sprint 03 calls
  to write directly into numpy memory.
- `encode_observation_into(obs, buffers_dict)` exists as a PyO3 method that
  validates shapes/dtypes and fills preallocated Python arrays. It uses
  runtime numpy module calls (`np.array`, `np.copyto`) rather than the Rust
  `numpy` crate — sprint 03 should switch to direct pointer writes via
  `PyArray::as_slice_mut()` or `unsafe` raw pointer access for true zero-copy.
- The encoder config defaults match `ObservationSpaceHypers`: 60 cards/player,
  30 permanents/player, 20 actions, 2 focus objects.
- PyO3 bindings are in `managym/src/python/bindings.rs`. The `Env` struct
  already has `encode_observation` and `encode_observation_into` methods.
- Observation field dimensions: player=26, card=18, permanent=5, action=7,
  action_focus shape=(max_actions, max_focus_objects) as i32.

## Changes

### 1. Two API lanes (compat + hot path)

- Keep existing `VectorEnv.step()` / `reset_all()` behavior for compatibility.
- Add `set_buffers()`, `step_into_buffers()`, `reset_all_into_buffers()` for
  training hot path.
- Add `get_last_info()` as explicit debug/eval API (not used in rollout loop).

### 2. Buffer protocol

Python allocates numpy arrays with shape `(num_envs, ...)` for each
observation field. These are passed to Rust once via `set_buffers()`.
Rust holds `Py<PyArray>` references (preventing GC).

```python
class RustVectorEnv:
    def __init__(self, num_envs, obs_config, match_config, seed):
        self._rust = managym.VectorEnv(num_envs, ...)
        self.obs = self._allocate_buffers(num_envs, obs_config)
        self._rust.set_buffers(
            self.obs["players"],
            self.obs["cards"],
            self.obs["permanents"],
            self.obs["actions"],
            self.obs["actions_valid"],
        )

    def step(self, actions):
        # Rust writes directly into pre-allocated arrays
        self._rust.step_into_buffers(actions)
        return self._obs_tensors, self._reward_tensor, self._done_tensor
```

`set_buffers()` validates exact dtype, exact shape, C-contiguous, writable.

### 3. Rust buffer writer

`VectorEnv::step_into_buffers()`:
- Release the GIL
- Step all games
- For each env, construct `EncodedObservationMut` pointing into the
  appropriate slice of the shared numpy buffer, then call `encode_into()`
- Write rewards/terminated/truncated into pre-allocated arrays
- Return `()`

The `encode_into()` function from sprint 02 writes into `&mut [f32]` slices
directly — no intermediate `Vec` allocation needed.

### 4. Remove Python encoding from hot path

The trainer reads `self.obs` directly — no `ObservationSpace.encode()`
call. The Python encoder remains for debugging and parity testing.

### 5. Explicit reward + info contracts

- Hot path reward is Rust raw reward only in sprint 03.
- Observation-dependent reward shaping remains out of hot path.
- No per-env info dict allocation in `step_into_buffers()`.

## Measurement

Follow the wave measurement protocol (see `wave/sps/README.md`).

### Before

```bash
# Multi-scale breakdown (env-only)
for n in 1 16 64; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done

# Training SPS baseline
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference
```

Record `encode` and `tensorize` percentages. These are the phases
this sprint eliminates. Save all numbers to the wave results tracker.

### After

```bash
# Zero-copy path (the new hot path)
for n in 1 16 64; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048 --mode zero-copy
done
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --mode zero-copy --with-inference

# Legacy path for comparison (should still work, just slower)
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --mode legacy
```

In zero-copy mode, `step_into_buffers` replaces the old encode/tensorize
phases. `sync_tensors` should be negligible on CPU.

### A/B

```bash
# Zero-copy vs legacy
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5

# Zero-copy vs Python-encode (isolates this sprint's contribution)
python scripts/bench_ab.py --a rust --b rust-python-encode --num-envs 16 --rounds 5
```

Add a `rust-python-encode` mode to `bench_ab.py` that uses Rust stepping
but Python-side encoding (sprint 02 behavior). This enables ablation in
sprint 05.

### Save results

Update the results tracker tables in `wave/sps/README.md` with S03
numbers for all scales measured.

### Gate

- Env-only SPS > 80,000 (encode + tensorize eliminated)
- Training SPS measured and recorded (with inference)
- `python -m manabot.verify.step0_env_sanity` passes
- Parity test passes (buffers produce same values as Python encoder)
- Buffer invariant tests (dtype/shape/contiguous/writable)

## Done when

Training runs end-to-end with zero-copy buffers. SPS improvement
measured and documented. Python encoder removed from training path.
Compatibility APIs still work for tests/debug tooling.
