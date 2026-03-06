# 03: Lightning Bolt (Targeting + Instant Interaction)

## Finish line

Lightning Bolt is implemented with declarative effect DSL and engine-side
target legality — validating real instant-speed interaction on the stack.

## Prerequisites already landed

Stage 04 (Man-o'-War) landed targeting infrastructure ahead of this stage:

- `ActionSpaceKind::ChooseTarget` and `Action::ChooseTarget` exist in
  `agent/action.rs`
- `Target` enum (`Player | Permanent`) in `state/target.rs`
- `StackObject` enum (`Spell | TriggeredAbility`) in `state/stack.rs`
- `GameEvent::CardMoved` in `flow/event.rs`
- Sequential target selection flow works for triggered abilities

This means stage 03 is now scoped to:
1. Add `Effect::DealDamage` variant to the DSL
2. Add `TargetSpec` variant for "creature or player"
3. Wire target selection into spell casting (currently only triggered abilities
   use `ChooseTarget`)
4. Add Lightning Bolt to card registry
5. Trace tests for bolt-specific interactions

## Changes

### Declarative Effect DSL Extension

Add damage effect to the existing `Effect` enum in `state/ability.rs`:

```rust
Effect::DealDamage {
    amount: u32,
    target: TargetSpec,
}
```

Add `TargetSpec::CreatureOrPlayer` (or extend existing `TargetSpec::Creature`
to a more general filter model).

### Spell Targeting

Spells with targets need the same `ChooseTarget` flow that triggered abilities
already use. When casting a targeted spell:
1. Agent chooses `CastSpell { card: bolt }`
2. Engine presents `ChooseTarget` with legal targets
3. Agent picks a target
4. Spell goes on stack with target attached

This reuses the existing `ChooseTarget` action space. The main work is wiring
`CastSpell` to check for targets on the spell's effect and present the
targeting step before the spell goes on the stack.

### Card Registry

Add Lightning Bolt (R, Instant, "Deal 3 damage to any target").

### Trace Tests

- Legal/illegal targets (creature on battlefield, player, not own creature if
  no legal — actually any target is legal for Bolt)
- Stack resolution ordering (bolt in response to bolt)
- Lethal player/creature outcomes
- Negative: can't cast bolt with no legal targets (edge case: only if
  literally no creatures or players — unlikely but test it)

### Training smoke test

Agent beats random with bolt-heavy decks. Track action space size increase.

## Done when

- Bolt castable at instant speed via sequential target selection.
- `Effect::DealDamage` exists and resolves correctly.
- Trace tests covering targeting and stack interaction pass.
- Training smoke test passes.
