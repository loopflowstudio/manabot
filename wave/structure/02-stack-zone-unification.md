# 02: Stack/Zone Unification

## Finish line

Stack membership is tracked in exactly one authoritative pathway.
`ZoneManager` zone tracking and `GameState.stack_objects` cannot desync
because every stack mutation goes through a single coordination point.

## Context

Cards on the stack are currently tracked in two places:

1. `ZoneManager` knows a card is in the `Stack` zone (via `zone_of(card)`)
2. `GameState.stack_objects: Vec<StackObject>` holds typed stack entries
   (spells and triggered abilities)

These are updated in separate code paths:
- `cast_spell()` creates a `StackObject` and calls `move_card(card, Stack)`
- `resolve_top_of_stack()` pops the `StackObject` and calls `move_card(card, destination)`
- `counter_spell()` removes the `StackObject` and calls `move_card(card, Graveyard)`

If any path updates one but not the other, the game enters an
inconsistent state. Today this works because the code is careful, but
it's not structurally enforced.

## Changes

### 1. Audit current stack operations

Enumerate every code path that touches `stack_objects` or moves a card
to/from the Stack zone. Verify they're paired. Document any that aren't.

### 2. Single coordination point

Create a small set of stack-specific methods that always do both:

```rust
impl Game {
    fn push_spell_to_stack(&mut self, card: CardId, spell: SpellOnStack) {
        self.state.stack_objects.push(StackObject::Spell(spell));
        self.move_card(card, ZoneType::Stack);
    }

    fn push_triggered_ability_to_stack(&mut self, ability: ActivatedAbilityOnStack) {
        self.state.stack_objects.push(StackObject::ActivatedAbility(ability));
        // triggered abilities don't have cards in zones
    }

    fn pop_stack(&mut self) -> Option<StackObject> {
        let obj = self.state.stack_objects.pop()?;
        // caller handles move_card for the resolved card
        Some(obj)
    }

    fn remove_from_stack(&mut self, index: usize) -> StackObject {
        self.state.stack_objects.remove(index)
        // caller handles move_card
    }
}
```

Replace all direct `stack_objects.push()` / `.pop()` / `.remove()` calls
in resolution.rs, game.rs, etc. with these methods.

### 3. Debug assertion

Add a debug-mode consistency check:

```rust
#[cfg(debug_assertions)]
fn assert_stack_consistent(&self) {
    let zone_stack_count = self.state.zones.size(ZoneType::Stack, PlayerId(0))
        + self.state.zones.size(ZoneType::Stack, PlayerId(1));
    let spell_count = self.state.stack_objects.iter()
        .filter(|obj| matches!(obj, StackObject::Spell(_)))
        .count();
    assert_eq!(zone_stack_count, spell_count,
        "stack_objects spell count ({}) != zone stack card count ({})",
        spell_count, zone_stack_count);
}
```

Call this after every `step()`. It runs in tests but not in release builds.

## Done when

- No direct `stack_objects.push/pop/remove` outside the coordination methods
- Debug assertion passes across the full test suite
- `cargo test` and `pytest tests/env/` pass
