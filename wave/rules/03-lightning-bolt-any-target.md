# 03: Lightning Bolt (Targeting + Instant Interaction)

## Finish line

Lightning Bolt is implemented with declarative effect DSL and engine-side
target legality — validating real instant-speed interaction on the stack.

## Prerequisites already landed

Stage 04 (Man-o'-War) landed targeting infrastructure ahead of this stage:

Stage 06 shipped:
- `StackObject` enum (`Spell(CardId)` | `ActivatedAbility(ActivatedAbilityOnStack)`)
  in `state/stack_object.rs` — the stack now handles both spells and abilities
- Activated ability definitions on `CardDefinition` (`activated_abilities: Vec<ActivatedAbilityDefinition>`)
- `PriorityActivateAbility` action type for ability activation
- `resolve_top_of_stack()` pops `StackObject` in strict LIFO order, dispatching
  to spell or ability resolution
- Shivan Dragon firebreathing as the first activated ability (cost, stack, resolution)
- Generic `stack_objects` observation lane with controller, source, targets, ability metadata

What's hardcoded now: Bolt, Counterspell, and Shivan firebreathing effects are
implemented directly in `resolve_top_of_stack()` in `game.rs` — no declarative
representation yet. Target metadata is stored as `HashMap<CardId, Target>`
(single target only).

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
