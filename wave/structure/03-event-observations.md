# 03: Event Observations

## Finish line

The agent sees recent game events in its observation. Events accumulate
between agent decision points and are delivered as a fixed-size buffer
in the observation tensor.

## Context

Events are emitted during play (`GameEvent::DamageDealt`,
`GameEvent::SpellCast`, etc.) and drained by `env.rs`, but never exposed
to the agent. The agent sees state snapshots with no causal context —
it can't tell whether a creature just died, a spell was just cast, or
nothing happened.

This matters for RL because temporal patterns carry information:
- "Opponent just cast a creature" → counter magic is relevant
- "Combat damage was dealt" → explains life total changes
- "A spell was countered" → opponent used resources

MageZero doesn't expose events either, but their MCTS does forward
simulation. PPO agents are purely reactive — they need the observation
to carry context.

## Changes

### 1. Event buffer in Env

```rust
pub struct Env {
    // existing...
    event_buffer: Vec<GameEvent>,
}
```

Between agent decision points (between `step()` calls), buffer all
events emitted by `game.drain_events()`. On each `step()`:
1. Drain events from the game into the buffer
2. Include buffered events in the observation
3. Clear the buffer after building the observation

### 2. EventData in observations

```rust
pub struct EventData {
    pub event_type: i32,
    pub source_card_id: i32,   // -1 if N/A
    pub target_id: i32,        // -1 if N/A (card, permanent, or player)
    pub amount: i32,           // damage, life change, etc. 0 if N/A
    pub zone_from: i32,        // for CardMoved, -1 otherwise
    pub zone_to: i32,          // for CardMoved, -1 otherwise
}
```

Event type discriminants:
```
CardMoved = 0
DamageDealt = 1
LifeChanged = 2
SpellCast = 3
SpellResolved = 4
SpellCountered = 5
TurnStarted = 6
StepStarted = 7
```

### 3. Fixed-size event buffer in observation

```rust
pub struct Observation {
    // existing...
    pub recent_events: Vec<EventData>,  // up to MAX_EVENTS (e.g. 32)
}
```

Capped at a configurable max (default 32). If more events occur between
decisions, keep the most recent. Pad with zeroed EventData if fewer.

### 4. Python bindings

- New `PyEventData` pyclass with all fields
- `PyObservation` gains `recent_events: Vec<PyEventData>`
- ObservationEncoder in Python encodes events as a fixed-size
  `(max_events, event_feature_dim)` float32 array with validity mask

### 5. Rust observation encoder (if sprint 02/03 of sps wave landed)

If the Rust-side `ObservationEncoder` exists, add event encoding there
too. Same shape as the Python encoder.

## Done when

- Stepping through a game produces events in the observation
- Lightning Bolt dealing damage appears as a DamageDealt event
- Spells cast appear as SpellCast events
- Buffer cap works (oldest events dropped when buffer full)
- Events reset between agent decisions (no stale events)
- `cargo test` and `pytest tests/env/` pass
- Python ObservationEncoder handles the new field
