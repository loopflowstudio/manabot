# Zero-Copy Observation Buffers

## Problem

Every training step crosses the Python-Rust boundary per-env: Rust returns
`PyObservation` objects (allocating Python wrappers for every card, permanent,
action), then Python's `ObservationSpace.encode()` converts each observation
into numpy arrays, then `stack_encoded_observations()` stacks them, then
`torch.tensor()` copies to device. This creates O(num_envs) Python objects and
numpy allocations per step.

Sprint 02 built `encode_into(obs, config, EncodedObservationMut)` which writes
directly into `&mut [f32]` slices. The seam exists — we just need to point those
slices at numpy memory.

Wave goals advanced: "3. Zero-copy buffer writes — Rust writes directly into
Python numpy arrays" and "5. 10x SPS improvement on same hardware."

## Approach

Add the `numpy` Rust crate. Python allocates numpy arrays shaped
`(num_envs, ...)` once. Rust holds `Py<PyArray>` references to prevent GC.
On each step, Rust steps all games with the GIL released, then writes encoded
observations directly into numpy memory using `encode_into()`. Python reads
the pre-filled arrays — no encoding, no stacking, no per-step allocation.

### API compatibility and migration

Keep both call paths during sprint 03:

- **Compatibility path**: existing `VectorEnv.step()` / `reset_all()` continue
  returning Python objects + info dicts for tests/debug tooling.
- **Hot path**: new `set_buffers()`, `step_into_buffers()`,
  `reset_all_into_buffers()` for training.

Training (`RustVectorEnv`) uses only the hot path when
`train.use_rust_env=true`. This keeps migration risk bounded while avoiding
adapters in the training loop.

### Rust changes

**Cargo.toml**: Add `numpy = { version = "0.22", optional = true }` gated on the
`python` feature alongside `pyo3`.

**`PyVectorEnv` gains buffer state** (in `vector_env_bindings.rs`):

```rust
struct ObservationBuffers {
    // Py<PyArray> prevents GC. Each array has shape (num_envs, ...).
    agent_player: Py<PyArray2<f32>>,
    opponent_player: Py<PyArray2<f32>>,
    agent_cards: Py<PyArray3<f32>>,
    opponent_cards: Py<PyArray3<f32>>,
    agent_permanents: Py<PyArray3<f32>>,
    opponent_permanents: Py<PyArray3<f32>>,
    actions: Py<PyArray3<f32>>,
    action_focus: Py<PyArray3<i32>>,
    agent_player_valid: Py<PyArray2<f32>>,
    opponent_player_valid: Py<PyArray2<f32>>,
    agent_cards_valid: Py<PyArray2<f32>>,
    opponent_cards_valid: Py<PyArray2<f32>>,
    agent_permanents_valid: Py<PyArray2<f32>>,
    opponent_permanents_valid: Py<PyArray2<f32>>,
    actions_valid: Py<PyArray2<f32>>,
    // Also pre-allocated:
    rewards: Py<PyArray1<f64>>,
    terminated: Py<PyArray1<bool>>,
    truncated: Py<PyArray1<bool>>,
}

#[pyclass(name = "VectorEnv")]
pub struct PyVectorEnv {
    inner: Mutex<VectorEnv>,
    config: ObservationEncoderConfig,
    buffers: Option<ObservationBuffers>,
}
```

**`set_buffers(dict)`**: Validates shapes and dtypes, stores `Py<PyArray>` refs.

**`step_into_buffers(actions) -> ()`**: The hot-path method.

```
1. Lock env
2. py.allow_threads: step all games, collect Vec<StepResult>
3. For each env i:
   a. Get mutable numpy slices for env i via as_slice_mut()
   b. Construct EncodedObservationMut pointing into those slices
   c. Call encode_into(result.obs, config, out)
   d. Write reward/terminated/truncated into their buffers
4. Return () — Python reads from pre-allocated arrays
```

The encoding happens with the GIL held (needed for `as_slice_mut`), but game
stepping — the expensive part — releases the GIL.

**`reset_all_into_buffers(player_configs) -> ()`**: Same pattern for reset.

### Python changes

**`RustVectorEnv.__init__`**: Allocate all observation buffers as contiguous
C-order numpy arrays. Call `self._rust_env.set_buffers(buffers_dict)`.

```python
def _allocate_buffers(self) -> Dict[str, np.ndarray]:
    n = self.num_envs
    c = self.observation_space.encoder  # ObservationEncoderConfig
    return {
        "agent_player": np.zeros((n, 1, PLAYER_DIM), dtype=np.float32),
        "opponent_player": np.zeros((n, 1, PLAYER_DIM), dtype=np.float32),
        "agent_cards": np.zeros((n, c.cards_per_player, CARD_DIM), dtype=np.float32),
        # ... etc for all 15 observation fields
        "rewards": np.zeros(n, dtype=np.float64),
        "terminated": np.zeros(n, dtype=bool),
        "truncated": np.zeros(n, dtype=bool),
    }
```

`RustVectorEnv` also prebuilds stable tensor views once (CPU) and reuses them
every step to avoid repeated `torch.as_tensor(...)` wrapper creation.

**`step()`**: Call `step_into_buffers`, return cached tensor views.

```python
def step(self, actions):
    self._rust_env.step_into_buffers(actions.cpu().tolist())
    obs = self._obs_tensors
    rewards = self._reward_tensor
    terminated = self._terminated_tensor
    truncated = self._truncated_tensor
    return obs, rewards, terminated, truncated, {}
```

`torch.as_tensor` with numpy creates a zero-copy view on CPU. The training loop
copies from these views into its rollout storage buffers via
`obs_buf[key][step] = next_obs[key]` (numpy/torch slice assignment copies data).
This is safe — Rust doesn't write until the next `step_into_buffers` call.

**Reward contract (explicit for sprint 03)**:

- Hot-path reward uses Rust raw reward only.
- Python reward shaping that depends on structured `PyObservation` is out of
  hot path for this sprint.
- If future reward shaping needs observation context, add an opt-in slower path
  rather than silently reading stale/missing structured obs.

**Info contract (explicit for sprint 03)**:

- `step_into_buffers` does not allocate per-env Python info dicts.
- Rust stores raw step metadata internally for the latest step.
- Python can call `get_last_info()` for debugging/eval; this is explicitly
  outside the rollout hot path.
- `VectorEnv.step()` keeps current info behavior for compatibility.

**Buffer safety invariants**:

- All arrays passed to `set_buffers()` must be exact shape, exact dtype,
  C-contiguous, writable.
- Rust holds strong `Py<PyArray>` references for the full env lifetime.
- Returned tensor views are read before next `step_into_buffers()` mutates
  backing memory.

### What gets removed from the hot path

- `PyObservation::from(obs)` — no more Python object wrapping per env per step
- `ObservationSpace.encode()` — no more Python-side feature extraction
- `stack_encoded_observations()` — no more per-step numpy stacking
- `info_dict_to_pydict()` per env — no more info dict conversion
- `add_truncation_flags()` Python call — moved to Rust or removed

### Parity testing

Extend the existing `test_observation_parity.py` to verify that:
1. `step_into_buffers` produces the same numpy values as the
   Python `ObservationSpace.encode()` path
2. Run on multiple random seeds, multiple steps per seed
3. Compare all 15 observation fields element-by-element
4. Compare reward / terminated / truncated parity against `step()` path
5. Compare reset parity (`reset_all_into_buffers` vs encoded reset obs)
6. Validate buffer contract failures (dtype/shape/contiguous/writable)

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Raw pointers without `rust-numpy` | No extra dependency, manual `ctypes` | Reinvents what `rust-numpy` provides; more error-prone, no shape/dtype safety |
| Return flat `Vec<u8>` from Rust, wrap in Python | Simple Rust side | Still allocates per step; not zero-copy |
| GIL-released encoding via raw pointers | Maximum parallelism | Unsafe pointer use across GIL boundary adds complexity; encoding is fast enough with GIL held |
| Rayon parallel encoding | Better multi-core utilization | Separate concern from zero-copy; adds complexity to buffer safety; natural follow-on sprint |

## Key decisions

**`rust-numpy` crate over raw pointer access.** The crate provides type-safe
`PyArray` wrappers with `as_slice_mut()`. It's the standard approach for
PyO3+numpy interop and matches our pyo3 0.22 version. The alternative — manual
buffer protocol via `PyAny` — is strictly worse.

**GIL-held encoding, GIL-released stepping.** Game stepping is the expensive
operation and benefits from GIL release (lets other Python threads run). Encoding
is cheap (just writing floats into arrays) and keeping the GIL simplifies
buffer access. No unsafe raw pointer arithmetic needed.

**No info dict in hot path.** The training loop doesn't use step info during
rollout collection. Dropping it from `step_into_buffers` eliminates O(num_envs)
`PyDict` allocations per step. Available via explicit `get_last_info()` method.

**Pre-allocated rewards/dones/truncated.** Small arrays, but pre-allocating them
means truly zero per-step allocation on the Rust→Python boundary. Return type
becomes `()` — everything is written into buffers.

**No Rayon in this sprint.** Parallel game stepping is wave goal 4 but is
orthogonal to zero-copy buffers. Adding Rayon here would complicate the buffer
safety model. Ship zero-copy first, then parallelize.

## Scope

- In scope:
  - `rust-numpy` dependency
  - `ObservationBuffers` struct in Rust
  - `set_buffers()`, `step_into_buffers()`, `reset_all_into_buffers()` methods
  - `get_last_info()` debug API with explicit non-hot-path semantics
  - Updated `RustVectorEnv` Python wrapper
  - Cached torch tensor views over numpy buffers
  - Parity test extension
  - Buffer invariant tests (contiguous/writable/dtype/shape)
  - Benchmark harness updates (`scripts/bench_managym.py`, `scripts/bench_sps.py`)
  - Type stubs update (`__init__.pyi`)

- Out of scope:
  - Rayon parallelism (wave goal 4, separate sprint)
  - Game engine optimizations ("not here" per wave vision)
  - Removing Python `ObservationSpace.encode()` (kept for debugging/parity)
  - Structured observation access in reward shaping

## Done when

```bash
# Parity test: zero-copy buffers match Python encoder
pytest tests/verify/test_observation_parity.py -v

# Training runs end-to-end with zero-copy path
manabot train --preset simple --set train.use_rust_env=true

# Benchmarks capture before/after SPS and env-step latency
python scripts/bench_managym.py
python scripts/bench_sps.py

# Rust tests pass
cd managym && cargo test && cd ..
```

Training SPS measured and compared to sprint 02 baseline. Target: significant
improvement from eliminating per-step Python object creation and encoding.
Wave goal: ">15,000 on CPU."
