# 02: Rust Observation Encoder

## Finish line

Observation encoding runs in Rust. The Python encoder still exists for
validation but is no longer on the hot path. Output is bit-identical.

## Changes

### 1. `managym/src/agent/observation_encoder.rs`

Port `ObservationSpace.encode()` from Python to Rust. The encoder takes
a raw `Observation` and `ObservationSpaceConfig` and writes float values
into flat arrays.

Encoding spec (must match `manabot/env/observation.py` exactly):

**Players** `(2, player_dim)`:
- life (normalized by 20)
- zone_counts (7 values, normalized by max)
- is_active (0/1)
- is_agent (0/1)

**Cards** `(2, max_cards, card_dim)`:
- card_types flags (creature, land, castable, etc.)
- mana_cost values
- power, toughness (normalized)
- zone one-hot (7 values)

**Permanents** `(2, max_perms, perm_dim)`:
- tapped (0/1)
- damage (normalized)
- summoning_sick (0/1)
- controller_is_agent (0/1)

**Actions** `(max_actions, action_dim)`:
- action_type one-hot
- focus object indices

**Actions valid** `(max_actions,)`:
- 1.0 for real actions, 0.0 for padding

### 2. Rust-side encode method

`VectorEnv::step()` returns encoded observations (flat `Vec<f32>`) in
addition to raw observations. Or: a new `step_encoded()` method.

### 3. Parity test

`tests/verify/test_observation_parity.py`:
- Run 200 games with random play
- At each step, encode via Python and Rust
- Assert all arrays match within `atol=1e-6`

This is the correctness gate. If this test passes, the Rust encoder
is trustworthy.

## Verification

- Parity test passes on 200 games (thousands of observations)
- `cargo test` passes
- No training regression

## Done when

Rust encoder exists, parity test passes, but training still uses
Python encoder (swap happens in sprint 03).
