# 02: Stack/Zone Unification

## Problem

Cards on the stack are tracked in two independent data structures that
must stay synchronized:

1. `ZoneManager` — per-player `Vec<CardId>` for the Stack zone, plus
   `card_zones[card] = Some(Stack)`. Used by `zone_of()` queries and
   observation encoding.
2. `GameState.stack: Vec<StackObject>` — ordered LIFO stack of spells
   and triggered abilities. Used for resolution and priority logic.

Today these are paired by convention. A future code path can mutate one
without the other and silently desync stack state.

## Approach (clear win)

Keep both data structures, but make desync hard by construction:

1. **Single mutation pathway** for stack operations through `Game`
   coordination methods.
2. **No direct stack writes** at call sites.
3. **Strong debug invariant** (identity equality, not only counts)
   executed after each `step()`.

This preserves current model shape and behavior while materially reducing
risk.

### Coordination methods (in `flow/zones.rs`)

```rust
impl Game {
    /// Push a spell onto the stack and move the card to Stack zone.
    pub(crate) fn push_spell_to_stack(&mut self, card: CardId) {
        self.move_card(card, ZoneType::Stack);
        self.state.stack.push(StackObject::Spell { card });
    }

    /// Push a triggered ability onto the stack.
    /// No zone move: triggered abilities are not cards in zones.
    pub(crate) fn push_triggered_to_stack(
        &mut self,
        source_card: CardId,
        ability_index: usize,
        controller: PlayerId,
        target: Option<Target>,
    ) {
        self.state.stack.push(StackObject::TriggeredAbility {
            source_card,
            ability_index,
            controller,
            target,
        });
    }

    /// Pop top stack object. Caller handles destination for spells.
    pub(crate) fn pop_stack(&mut self) -> Option<StackObject> {
        self.state.stack.pop()
    }

    pub(crate) fn stack_is_empty(&self) -> bool {
        self.state.stack.is_empty()
    }
}
```

### Callers updated

| File | Before | After |
|------|--------|-------|
| `play.rs` | `move_card(...Stack)` + `state.stack.push(Spell)` | `push_spell_to_stack(card)` |
| `resolution.rs` | `state.stack.pop()` | `pop_stack()` |
| `triggers.rs` | `state.stack.push(TriggeredAbility)` | `push_triggered_to_stack(...)` |
| `tick.rs`, `play.rs` | `state.stack.is_empty()` | `stack_is_empty()` |

`agent/observation.rs` remains read-only direct iteration over
`state.stack`.

## Debug invariant (strong form)

Add `assert_stack_consistent()` in debug/test builds and call it at end
of `step()`.

Checks:

1. Build `zone_spell_cards` = all cards in `ZoneType::Stack` from
   `ZoneManager`.
2. Build `stack_spell_cards` = all `CardId`s from
   `StackObject::Spell { card }` entries in `GameState.stack`.
3. Assert exact multiset equality (same cards, same multiplicities).
4. Assert no duplicate spell card IDs in `stack_spell_cards` (belt and
   suspenders for impossible states).

This catches same-count/wrong-card desyncs that count-only assertions miss.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Remove Stack from `ZoneManager` now | True single storage | Larger model/API churn right now; introduces awkward stack-aware zone querying and higher regression risk |
| Do nothing | Zero work | Keeps latent silent-desync risk |

## Scope

- In scope:
  - Coordination methods for stack push/pop/empty
  - Caller migration off direct `state.stack` mutation
  - Strong debug invariant after each `step()`
- Out of scope:
  - Removing Stack from `ZoneManager`
  - New stack mechanics (counterspells/index removal APIs)
  - Representation changes to `StackObject`

## Done when

- `cargo test` passes
- `pytest tests/env/` passes
- No direct `state.stack.push/pop` outside coordination methods
- Priority checks use `stack_is_empty()` in flow paths
- Debug invariant runs on every `step()` in debug/test builds
- `grep -rn 'state\.stack\.push\|state\.stack\.pop' managym/src/flow/`
  returns only coordination methods
