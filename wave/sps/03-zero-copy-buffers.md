# 03: Zero-Copy Buffers

## Finish line

Rust writes observations directly into pre-allocated numpy arrays.
No Python objects created per step. No per-step memory allocation.

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
- Encode observations directly into the numpy memory
- Return only rewards and dones (small arrays, cheap to allocate)

The observation encoding from sprint 02 is adapted to write into
a buffer slice rather than returning a Vec.

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
