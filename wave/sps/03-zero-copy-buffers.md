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

### 1. Buffer protocol

Python allocates numpy arrays with shape `(num_envs, ...)` for each
observation field. These are passed to Rust once via `set_buffers()`.
Rust holds `Py<PyArray>` references (preventing GC) and obtains raw
`*mut f32` pointers.

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
        # Rust writes directly into self.obs arrays
        rewards, dones = self._rust.step_into_buffers(actions)
        return self.obs, rewards, dones
```

### 2. Rust buffer writer

`VectorEnv::step_into_buffers()`:
- Release the GIL
- Step all games
- For each env, construct `EncodedObservationMut` pointing into the
  appropriate slice of the shared numpy buffer, then call `encode_into()`
- Return only rewards and dones (small arrays, cheap to allocate)

The `encode_into()` function from sprint 02 writes into `&mut [f32]` slices
directly — no intermediate `Vec` allocation needed.

### 3. Remove Python encoding from hot path

The trainer reads `self.obs` directly — no `ObservationSpace.encode()`
call. The Python encoder remains for debugging and parity testing.

### 4. Rewards and dones as numpy

Return rewards as `np.ndarray(num_envs,)` and dones as
`np.ndarray(num_envs,)` — also via pre-allocated buffers if the
overhead matters, but these are small enough that allocation is fine.

## Verification

- Parity test still passes (buffers produce same values as Python encoder)
- Training SPS measured before/after
- Verification ladder passes
- Memory profiling: no per-step allocations in steady state

## Done when

Training runs end-to-end with zero-copy buffers. SPS improvement
measured and documented. Python encoder removed from training path.
