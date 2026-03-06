# Event Observations

## Problem

The agent sees state snapshots with no causal context. Between decision
points, spells resolve, damage is dealt, creatures die, and life totals
change — but the agent only sees the result, not what happened. This
matters for PPO because the agent is purely reactive (no forward
simulation like MCTS), so temporal patterns must be carried in the
observation.

Wave goal 3: "The agent sees recent game events in its observation."

## Approach

Three layers of work, shipped as one commit:

### 1. Expand GameEvent variants

Today only `CardMoved` exists. Add variants that capture the major game
actions an agent needs to reason about:

```rust
pub enum GameEvent {
    CardMoved { card, from, to, controller },       // exists
    DamageDealt { source, target, amount },          // NEW
    LifeChanged { player, old_life, new_life },      // NEW
    SpellCast { card, controller },                  // NEW
    SpellResolved { card, controller },              // NEW
    SpellCountered { card, controller },             // NEW
    AbilityTriggered { source_card, controller },    // NEW
}
```

Emit these at the natural points:
- `DamageDealt` in permanent and player damage application paths
- `LifeChanged` at every player life mutation through a single helper
  (`set_player_life`/equivalent), including damage and life gain paths
- `SpellCast` in `push_spell_to_stack`
- `SpellResolved` after spell resolution completes
- `SpellCountered` in `counter_spell`
- `AbilityTriggered` in `push_triggered_to_stack`

Drop `TurnStarted`/`StepStarted` from the original wave item — the
agent already sees turn/step in `TurnData`. Adding redundant events
wastes buffer slots.

### 2. Dual-drain event accumulator

Events serve two consumers: the trigger system (immediate, within the
game loop) and the observation (accumulated across a full `step()` call).

Add `observation_events: Vec<GameEvent>` to `GameState`. Every event
push goes to both `pending_events` (consumed by `process_game_events`)
and `observation_events` (consumed by `Observation::new`). A helper
method ensures they stay in sync:

```rust
impl GameState {
    pub fn emit_event(&mut self, event: GameEvent) {
        self.pending_events.push(event);
        self.observation_events.push(event);
    }
}
```

Replace all direct `pending_events.push(...)` calls with
`self.state.emit_event(...)`. The `observation_events` vec is drained
by the Env after building the observation — not by the game loop.

### 3. Event encoding in observations

Add `EventData` to the observation:

```rust
pub struct EventData {
    pub event_type: i32,      // EventType discriminant (stable contract)
    pub source_kind: i32,     // EventEntityKind, 0 if N/A
    pub source_id: i32,       // object id, -1 if N/A
    pub target_kind: i32,     // EventEntityKind, 0 if N/A
    pub target_id: i32,       // object id, -1 if N/A
    pub amount: i32,          // damage or life delta, 0 if N/A
    pub controller_id: i32,   // player id who caused this, -1 if N/A
}
```

Add explicit schema enums shared across Rust/Python:

```rust
#[repr(i32)]
pub enum EventType {
    CardMoved = 0,
    DamageDealt = 1,
    LifeChanged = 2,
    SpellCast = 3,
    SpellResolved = 4,
    SpellCountered = 5,
    AbilityTriggered = 6,
}

#[repr(i32)]
pub enum EventEntityKind {
    None = 0,
    Card = 1,
    Permanent = 2,
    Player = 3,
}
```

7 features per event. Fixed buffer of 32 events (configurable via
`ObservationEncoderConfig.max_events`). Most recent events kept when
buffer overflows. Zero-padded with validity mask.

**Observation struct** gains `recent_events: Vec<EventData>`.
`Observation::new` takes `&[GameEvent]` from the accumulated buffer,
converts to `EventData`, truncates to most recent 32.

**Rust encoder** gains `events: Vec<f32>` and `events_valid: Vec<f32>`
in `EncodedObservation`. Shape: `(max_events, EVENT_DIM)` flattened.
New constant `EVENT_DIM: usize = 7`.

**Python bindings** gain `PyEventData` pyclass and
`PyObservation.recent_events: Vec<PyEventData>`. The `encoded_to_dict`
function includes the new `events` and `events_valid` arrays.

**Python encoder** (`manabot/env/observation.py`) adds event encoding:
`(max_events, EVENT_DIM)` float32 array with validity mask, matching
the Rust encoder shape.

### Event lifecycle

```
Game loop:
  state.emit_event(evt)
    → pending_events (consumed by process_game_events for triggers)
    → observation_events (accumulates across ticks)

Env.step():
  game.step(action)               # may emit many events
  events = take(observation_events)  # drain accumulated events
  obs = Observation::new(game, &events)
  return obs
```

`events` is the full event sequence since the previous observation was
returned to the agent (decision-to-decision window), not a full-turn or
full-game log.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| State diffs instead of events | Derive what changed from before/after snapshots | Lossy (can't see intermediate damage that was healed), expensive to diff every step |
| Events as separate step() return | Cleaner separation, obs stays unchanged | Breaks Gymnasium interface; Python side would need to merge anyway |
| Ring buffer on GameState | Game owns the buffer, Env just reads | Complicates lifetime — game doesn't know when Env has consumed events |
| Include turn/step events | More context for the agent | Agent already sees turn/step in TurnData; wastes 2 of 32 buffer slots per turn |

## Key decisions

1. **Dual accumulator over clone-before-drain.** The trigger system
   uses `std::mem::take` which is zero-copy. Cloning the vec before
   every drain would be wasteful. Two vecs with a single `emit_event`
   entry point is simpler and faster.

2. **GameEvent stays Copy.** All variants use `CardId`, `PlayerId`,
   `PermanentId`, `i32` — all Copy. No allocations in the hot path.

3. **7-field flat EventData with explicit entity kinds.** PPO needs
   fixed-size tensors, but mixed id namespaces are ambiguous. Adding
   `source_kind`/`target_kind` removes ambiguity while staying tensor
   friendly.

4. **32-event buffer, keep most recent.** Between two agent decisions,
   a typical Magic turn generates 5-15 events. 32 handles combat-heavy
   turns with headroom. Keeping most recent (not oldest) because the
   agent's next decision is most informed by what just happened.

5. **No AbilityResolved event.** Triggered abilities resolve through
   the same `resolve_top_of_stack` path as spells. The agent sees
   `AbilityTriggered` when it goes on the stack and can infer resolution
   from state changes. Adding a separate resolved event for abilities
   would be noise.

6. **Stable schema contract across Rust/Python.** `EventType` and
   `EventEntityKind` discriminants are explicit and tested, so a reorder
   or variant insertion cannot silently break encoders.

7. **One commit, not six.** The event enum expansion, dual drain,
   observation encoding, Rust encoder, Python bindings, and Python
   encoder are tightly coupled. Shipping them separately would leave
   dead code or broken intermediate states.

8. **Agent memory is complementary, not a substitute.** Recurrent
   memory can infer temporal patterns, but explicit event channels lower
   inference burden and improve sample efficiency for reactive policies.

## Scope

- In scope:
  - New `GameEvent` variants with emission points
  - `observation_events` dual-drain accumulator on `GameState`
  - `EventData` in `Observation` struct
  - Rust `ObservationEncoder` event encoding
  - Python `PyEventData` bindings
  - Python `ObservationEncoder` event encoding
  - Schema contract tests (EventType/EventEntityKind parity in Rust and
    Python)
  - Tests: unit test for event accumulation, integration tests for event
    presence and ordering through spell + combat + trigger sequences
  - Invariant test: every life total change emits `LifeChanged`

- Out of scope:
  - Model architecture changes to consume events (training wave)
  - Event replay or logging infrastructure
  - Reward shaping from events
  - New card types or mechanics

## Done when

- `cargo test` passes
- `pytest tests/env/` passes
- Stepping through a game produces events in the observation
- Lightning Bolt dealing 3 damage produces a `DamageDealt` event
- Casting a spell produces a `SpellCast` event
- Buffer caps at 32 (oldest events dropped)
- Events reset between agent decisions
- Rust and Python encoders produce matching shapes and event type/entity
  discriminants
- Event ordering is preserved inside a decision window
- Any player life total delta emits `LifeChanged`
- `cargo clippy --all-targets --all-features -- -D warnings` clean
