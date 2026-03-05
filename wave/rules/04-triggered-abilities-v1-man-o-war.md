# 04: Triggered Abilities v1 (Man-o'-War)

## Finish line

Triggered abilities exist with queueing + stack placement semantics, built on
the event system from stage 02. Validated with Man-o'-War ETB.

## Context (what's already landed)

Stage 02 shipped the `GameEvent` system (`Vec<GameEvent>` log on `GameState`)
with `CardMoved`, `SpellCast`, `SpellResolved`, `SpellCountered`, `DamageDealt`,
`LifeChanged`, `TurnStarted`, and `StepStarted` events. Events are drained via
`drain_events()`. The `ChooseTarget` action space and single-target
infrastructure are also in place.

`game.rs` is now ~1200 lines. Consider whether trigger integration warrants
extracting resolution logic into a separate module.

## Changes

### Trigger System

- Consume `GameEvent::CardMoved` (via `drain_events()`) for ETB detection
- Pending trigger queue with correct timing: triggers queue when event fires,
  go on stack when a player would next receive priority (CR 603.3)
- APNAP ordering for simultaneous triggers
- Intervening "if" clause support (CR 603.4) — check condition on trigger
  and again on resolution

### Man-o'-War as Driver Card

Add Man-o'-War to card registry with declarative effect:

```rust
Effect::Triggered {
    condition: TriggerCondition::EntersTheBattlefield { source: Source::This },
    effect: Box::new(Effect::ReturnToHand {
        target: TargetSpec::Any {
            filter: Filter::Creature,
            count: 1,
        },
    }),
}
```

Man-o'-War exercises: ETB trigger detection, trigger stacking, target
selection for triggered abilities (reuses `ChooseTarget` from stage 03).

### Trace Tests

- Man-o'-War ETB bounces a creature
- Trigger timing relative to priority/SBA loop
- Multiple simultaneous triggers (two Man-o'-Wars entering)
- Negative: Man-o'-War with no legal targets (must still target if able)
- Trigger ordering with APNAP

### Training smoke test

Agent learns to sequence ETB effects. Track episode length changes.

## Done when

- Man-o'-War ETB behavior is deterministic and CR-cited.
- Trigger queue integrates correctly with priority/SBA loop.
- Triggered ability trace tests pass.
- Training smoke test passes.
