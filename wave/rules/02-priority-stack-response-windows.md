# 02: Priority/Stack + Event System

## Finish line

Priority and stack behavior supports real response windows (not just own-turn
sorcery speed). A `GameEvent` system exists as the foundation for triggers,
replacement effects, and continuous effects in later stages.

## Changes

### Priority/Stack (CR 117, 405)

- Align priority sequencing with CR 117 and stack resolution with CR 405.
- Allow instant-speed casting windows where legal.
- Trace tests for:
  - Active/nonactive pass flow
  - All-pass resolves top object
  - Empty stack advances phase/step
  - Post-resolution priority handoff

### Event System

Introduce `GameEvent` enum and emission points throughout `game.rs`:

```rust
enum GameEvent {
    CardMoved { card: CardId, from: Zone, to: Zone },
    DamageDealt { source: ObjectId, target: ObjectId, amount: u32 },
    LifeChanged { player: PlayerId, old: i32, new: i32 },
    ManaProduced { player: PlayerId, color: Color, amount: u32 },
    SpellCast { card: CardId, targets: Vec<ObjectId> },
    AbilityActivated { source: ObjectId, targets: Vec<ObjectId> },
    TurnStarted { player: PlayerId },
    PhaseChanged { phase: Phase },
}
```

Events fire for zone changes, damage, life changes. Later stages (triggers,
replacement effects) subscribe to these rather than inventing ad-hoc hooks.

This likely requires refactoring parts of `game.rs` (~882 lines) to emit
events at the right points. Scope the refactor to event emission only — don't
restructure the control flow.

### Training smoke test

After enabling instant-speed windows: agent beats random in 100k steps.

## Done when

- Response-window stack interactions are testable and deterministic.
- CR-cited trace tests for core 117/405 behaviors pass.
- `GameEvent` enum exists and fires for zone changes, damage, life changes.
- Training smoke test passes.
