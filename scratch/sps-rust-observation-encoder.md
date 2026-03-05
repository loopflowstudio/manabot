# Rust Observation Encoder

## Problem

Observation encoding is the hottest Python code in the training loop. Every `step()` call crosses Rust→Python, builds a raw `Observation` with PyO3, then Python iterates over cards/permanents/actions to fill numpy arrays with normalization, one-hot encoding, and object-to-index mapping. This is pure data transformation — no Python flexibility needed.

Moving encoding to Rust eliminates per-step Python overhead and keeps the hot path in compiled code. This is the prerequisite for sprint 03 (swapping the training loop to use Rust-encoded observations directly).

## Approach

Add `ObservationEncoder` in Rust that takes a raw `Observation` and produces fixed-size flat arrays. Expose encoding to Python via `encode_observation(obs)` for parity/profiling in sprint 02 while keeping raw `step()` / `reset()` unchanged.

Also pull **zero-copy prep** into sprint 02 by introducing buffer-oriented encoder internals now (without switching the trainer path yet).

### Encoding spec (must be bit-identical to Python)

**Config constants** (matching `ObservationSpaceHypers` defaults):
- `max_cards_per_player: 60`
- `max_permanents_per_player: 30`
- `max_actions: 20`
- `max_focus_objects: 2`

**Player** — dim=26:
| Index | Field | Normalization |
|-------|-------|---------------|
| 0 | life | / 20.0 |
| 1 | is_active | bool |
| 2–8 | zone_counts (7) | / 60.0 |
| 9–13 | phase one-hot (5) | — |
| 14–25 | step one-hot (12) | — |

**Card** — dim=18:
| Index | Field | Normalization |
|-------|-------|---------------|
| 0–6 | zone one-hot (7) | — |
| 7 | is_mine | bool |
| 8 | power | / 10.0 |
| 9 | toughness | / 10.0 |
| 10 | mana_value | / 10.0 |
| 11 | is_land | bool |
| 12 | is_creature | bool |
| 13 | is_artifact | bool |
| 14 | is_enchantment | bool |
| 15 | is_planeswalker | bool |
| 16 | is_battle | bool |
| 17 | validity flag | 1.0 or 0.0 |

**Permanent** — dim=5:
| Index | Field | Normalization |
|-------|-------|---------------|
| 0 | is_mine | bool |
| 1 | tapped | bool |
| 2 | damage | / 10.0 |
| 3 | is_summoning_sick | bool |
| 4 | validity flag | 1.0 or 0.0 |

**Action** — dim=7:
| Index | Field | Normalization |
|-------|-------|---------------|
| 0–5 | action_type one-hot (6) | — |
| 6 | validity flag | 1.0 or 0.0 |

**Action focus** — shape (max_actions, max_focus_objects), dtype i32:
- Values ≥ 0: global object index
- Value -1: no focus / unused slot

### Object-to-index mapping

Objects are assigned global indices in this order:
1. Agent player → index 0
2. Opponent player → index 1
3. Agent cards → indices [2, 62)
4. Opponent cards → indices [62, 122)
5. Agent permanents → indices [122, 152)
6. Opponent permanents → indices [152, 182)

Card and permanent IDs are mapped to their assigned index. Action focus object IDs are resolved through this map. Unknown IDs map to -1.

### Output structure

`EncodedObservation` contains:
```
agent_player:              [1 × 26]   f32
opponent_player:           [1 × 26]   f32
agent_cards:               [60 × 18]  f32
opponent_cards:             [60 × 18]  f32
agent_permanents:          [30 × 5]   f32
opponent_permanents:       [30 × 5]   f32
actions:                   [20 × 7]   f32
action_focus:              [20 × 2]   i32
agent_player_valid:        [1]        f32
opponent_player_valid:     [1]        f32
agent_cards_valid:         [60]       f32
opponent_cards_valid:      [60]       f32
agent_permanents_valid:    [30]       f32
opponent_permanents_valid: [30]       f32
actions_valid:             [20]       f32
```

Validity arrays are derived from the last element of each object's feature vector.

### Rust implementation

**New file**: `managym/src/agent/observation_encoder.rs`

```rust
pub struct ObservationEncoderConfig {
    pub max_cards_per_player: usize,      // 60
    pub max_permanents_per_player: usize, // 30
    pub max_actions: usize,               // 20
    pub max_focus_objects: usize,         // 2
}

pub struct EncodedObservation {
    pub agent_player: Vec<f32>,           // len = 26
    pub opponent_player: Vec<f32>,        // len = 26
    pub agent_cards: Vec<f32>,            // len = 60 * 18
    pub opponent_cards: Vec<f32>,         // len = 60 * 18
    pub agent_permanents: Vec<f32>,       // len = 30 * 5
    pub opponent_permanents: Vec<f32>,    // len = 30 * 5
    pub actions: Vec<f32>,               // len = 20 * 7
    pub action_focus: Vec<i32>,          // len = 20 * 2
}

pub fn encode(obs: &Observation, config: &ObservationEncoderConfig) -> EncodedObservation

pub struct EncodedObservationMut<'a> {
    pub agent_player: &'a mut [f32],
    pub opponent_player: &'a mut [f32],
    pub agent_cards: &'a mut [f32],
    pub opponent_cards: &'a mut [f32],
    pub agent_permanents: &'a mut [f32],
    pub opponent_permanents: &'a mut [f32],
    pub actions: &'a mut [f32],
    pub action_focus: &'a mut [i32],
}

pub fn encode_into(
    obs: &Observation,
    config: &ObservationEncoderConfig,
    out: EncodedObservationMut<'_>,
) -> Result<(), ObservationEncodeError>
```

The encoder is a stateless function — config is passed in, no mutable state between calls. The `object_to_index` map is built internally per call, same as Python.

`encode()` becomes a thin allocator wrapper over `encode_into()`. Sprint 03 reuses `encode_into()` to write directly into long-lived numpy buffers.

### PyO3 surface

**New encode entrypoint**:
- `encode_observation(obs: Observation) -> dict[str, np.ndarray]`
- `encode_observation_into(obs: Observation, out: dict[str, np.ndarray]) -> None` (prep path; validates shapes/dtypes and fills provided arrays)

The encoded observation is returned as a Python dict of numpy arrays, using the `numpy` crate. In sprint 02, per-call allocation is acceptable for the default method.

`encode_observation_into()` is the sprint-02 zero-copy prep seam: it proves the writer path and buffer validation contract without changing training integration yet.

The raw `step()` / `reset()` methods remain unchanged — Python encoder stays available for validation.

### Parity test

**New file**: `tests/verify/test_observation_parity.py`

```python
def test_observation_parity():
    """Run 200 games, encode every observation both ways, assert match."""
    env = managym.Env(seed=42, skip_trivial=True)
    encoder = ObservationEncoder(hypers)

    for game_i in range(200):
        raw_obs, _ = env.reset(configs)
        encoded_python = encoder.encode(raw_obs)
        encoded_rust = env.encode_observation(raw_obs)

        for key in encoded_python:
            np.testing.assert_allclose(
                encoded_rust[key], encoded_python[key],
                atol=1e-6, err_msg=f"Game {game_i}, key {key}"
            )

        while True:
            action = random_valid_action(raw_obs)
            raw_obs, reward, term, trunc, info = env.step(action)
            encoded_python = encoder.encode(raw_obs)
            encoded_rust = env.encode_observation(raw_obs)

            for key in encoded_python:
                np.testing.assert_allclose(
                    encoded_rust[key], encoded_python[key],
                    atol=1e-6
                )

            if term or trunc:
                break
```

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Return pre-encoded arrays from `step()` directly (replace raw obs) | Simpler API, breaks validation | Need both for parity testing; swap in sprint 03 |
| Encode in Python with vectorized numpy | Could be fast enough | Still per-step overhead; doesn't scale |
| PyTorch C++ extension | Direct tensor output | Adds build complexity; numpy→torch is cheap |
| Single flat `Vec<f32>` | Simpler Rust side | Python must slice by offset; error-prone |

## Key decisions

1. **Stateless encode function.** The Python encoder rebuilds `object_to_index` every call. No reason to carry state.

2. **Keep raw env API untouched in sprint 02.** Use a dedicated `encode_observation(obs)` entrypoint for parity/profiling and defer encoded-step API changes to sprint 03.

3. **Config defaults match Python `ObservationSpaceHypers`.** Hardcoded defaults, overridable via constructor for future use.

4. **`numpy` crate for array return.** The `numpy` crate integrates cleanly with PyO3 and avoids Python-side loops.

5. **Encode explicit observations, not hidden env state.** `Env` does not currently cache the last observation. Passing `obs` explicitly avoids adding mutable cache state just for parity testing.

6. **200-game parity test with random play.** Generates thousands of observations across diverse game states. Random play hits edge cases that scripted games miss.
7. **Add deterministic object-index test.** Unit test explicit object ordering/padding/focus resolution, independent of gameplay randomness.

## Scope

- **In scope**: Rust encoder, PyO3 bindings, parity test, deterministic object-index unit test, `encode_into` writer seam, `encode_observation_into` buffer-fill prep API
- **Out of scope**: Swapping training loop to use Rust encoder (sprint 03), batched encoding across envs, GPU tensor output

## Done when

1. `cargo test` passes with encoder unit tests
2. `pytest tests/verify/test_observation_parity.py` passes on 200 games (thousands of observations verified)
3. `pytest` includes a deterministic test for object-index mapping + padded-slot focus behavior
4. `encode_observation_into()` validates shape/dtype and fills preallocated arrays correctly
5. Raw `step()` / `reset()` still work unchanged
6. No training regression (existing Python path untouched)
