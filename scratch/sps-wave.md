# SPS Wave: Rust VectorEnv

## Problem

Training is env-gated at ~1,600 SPS on CPU. The bottleneck is per-step
Python↔Rust round-trips: each of 2,048 transitions per update (16 envs ×
128 steps) crosses the PyO3 boundary twice, allocates Python objects, and
converts to numpy/torch.

## Target

10x throughput — ~16,000 SPS on the same hardware. Eliminate Python from
the hot path. Only cross the boundary once per rollout step (all envs),
not once per env per step.

## Design

### Rust: `VectorEnv`

A single Rust object owns N `Game` instances and steps them in batch.

```rust
pub struct VectorEnv {
    games: Vec<Game>,
    skip_trivial: bool,
    seeds: Vec<u64>,
    // Pre-allocated observation buffers — one contiguous block per field.
    // Python passes these in; Rust writes directly into them.
}

impl VectorEnv {
    /// Reset all games. Write initial observations into the provided buffers.
    fn reset(&mut self, buffers: &ObservationBuffers);

    /// Step all games with the given actions.
    /// Write observations into buffers. Return (rewards, dones) as numpy arrays.
    fn step(&mut self, actions: &[i64], buffers: &ObservationBuffers)
        -> (Vec<f64>, Vec<bool>);
}
```

### Observation buffers: zero-copy

Python allocates numpy arrays upfront with shape `(num_envs, ...)` for each
observation field. These are passed to Rust once. On each step, Rust writes
directly into the numpy memory via raw pointers (PyO3's `PyArray` supports
this). No Python objects created per step.

```python
class RustVectorEnv:
    def __init__(self, num_envs, obs_space, match, seed):
        self._rust = managym.VectorEnv(num_envs, ...)
        # Pre-allocate observation buffers
        self.obs = {
            "players": np.zeros((num_envs, obs_space.players_dim), dtype=np.float32),
            "cards": np.zeros((num_envs, max_cards, card_dim), dtype=np.float32),
            "permanents": np.zeros((num_envs, max_perms, perm_dim), dtype=np.float32),
            "actions": np.zeros((num_envs, max_actions, action_dim), dtype=np.float32),
            "actions_valid": np.zeros((num_envs, max_actions), dtype=np.float32),
        }
        self._rust.set_buffers(self.obs)  # Pass pointers to Rust

    def step(self, actions):
        # One PyO3 call. Rust writes obs into pre-allocated buffers.
        rewards, dones, infos = self._rust.step(actions)
        return self.obs, rewards, dones, infos
```

### Observation encoding moves to Rust

Currently `ObservationSpace.encode()` is Python — it walks the observation
struct and writes floats into numpy arrays. This moves into Rust as part of
the buffer write. The encoding logic (card features, permanent features,
action features) becomes Rust code that writes directly into the numpy
buffer slice for each env.

### Auto-reset

Games that terminate are automatically reset on the next step (Rust-side).
The returned observation for a done env is the post-reset observation.
Info dict carries `true_terminated` and `winner_index` for the completed
episode.

### Threading

`step()` can use Rayon to step N games in parallel on Rust threads.
The GIL is released during the Rust call. Each game writes to its own
slice of the buffer — no synchronization needed.

## Migration path

1. **`managym::vector_env`** — Rust VectorEnv struct, batch step/reset, no
   buffer writes yet. Returns Vec<Observation> like today.
2. **`managym::observation_encoder`** — Rust-side encoding that matches
   Python's `ObservationSpace.encode()` output exactly. Test by comparing
   outputs.
3. **Zero-copy buffers** — Python allocates, Rust writes. Remove Python
   encode step.
4. **Rayon parallelism** — Thread-per-game stepping within the Rust call.
5. **Training integration** — Replace `AsyncVectorEnv` usage in
   `Trainer` with `RustVectorEnv`.

Each step is independently testable. Step 1-2 can land without changing
training at all — just add the new code alongside the existing path.

## Observation encoding spec

The Rust encoder must produce identical output to `ObservationSpace.encode()`.
Validation: run both encoders on the same raw observation, assert arrays match
within float tolerance. This is the critical correctness gate.

Key encoding details to port:
- Player features: life, zone counts, is_active, is_agent (normalized)
- Card features: card_types flags, mana_cost, power, toughness, zone one-hot
- Permanent features: tapped, damage, summoning_sick, controller match
- Action features: action_type one-hot, focus object indices
- Padding: zero-fill to max_cards/max_permanents/max_actions
- actions_valid: 1.0 for real actions, 0.0 for padding

## Files

| New file | Purpose |
|----------|---------|
| `managym/src/agent/vector_env.rs` | VectorEnv struct |
| `managym/src/agent/observation_encoder.rs` | Rust-side obs encoding |
| `managym/src/python/vector_env_bindings.rs` | PyO3 bindings for VectorEnv |
| `manabot/env/rust_vector_env.py` | Python wrapper (thin) |

| Modified file | Change |
|---------------|--------|
| `manabot/model/train.py` | Use RustVectorEnv instead of AsyncVectorEnv |
| `manabot/env/__init__.py` | Export RustVectorEnv |

## Verification

- Unit test: Rust encoder output == Python encoder output on 1000 random observations
- Integration test: training with RustVectorEnv produces same learning curves (±noise)
- Benchmark: `profile` skill before and after, report SPS
- Step0-step2 verification ladder passes with new env

## Prior art

PufferLib achieves 4M training SPS with this exact pattern (C envs writing
into shared numpy buffers). Their C envs hit 100M raw steps/sec. Our Rust
engine should be comparable. The training SPS ceiling is the agent forward
pass, not the env.
