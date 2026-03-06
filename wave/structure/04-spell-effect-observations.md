# 04: Spell Effect Observations

## Finish line

The agent can distinguish spells by what they do. Lightning Bolt is
observable as "deal 3 damage to any target." Counterspell is observable
as "counter target spell." Creatures and lands with no spell effects
have a null marker.

## Context

The agent can already see keywords on cards (flying, deathtouch, etc.)
via `KeywordData`, and stack objects with their targets via
`StackObjectData`. But for spells in hand, the agent sees only card
stats (P/T, mana cost, type flags) — it can't see *what a spell does*
before deciding to cast it.

Two Lightning Bolts and two random 1-mana instants look identical in the
observation. The agent has to learn effects purely from observed outcomes.

This doesn't require migrating spell resolution to the Effect enum.
Spell resolution can stay string-matched for now. We're just annotating
card definitions with metadata that the observation can read.

## Changes

### 1. SpellEffectHint enum

In `state/card.rs`:

```rust
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SpellEffectHint {
    #[default]
    None,
    DealDamage { amount: u8 },
    CounterSpell,
    // Future variants added as cards are added — these are observation
    // hints, not execution logic. No game.rs changes needed.
}
```

### 2. CardDefinition gains the field

```rust
pub struct CardDefinition {
    // existing...
    pub spell_effect: SpellEffectHint,
}
```

### 3. Annotate existing cards

In `cardsets/alpha.rs`:

```rust
// Lightning Bolt
CardDefinition {
    spell_effect: SpellEffectHint::DealDamage { amount: 3 },
    ..
}

// Counterspell
CardDefinition {
    spell_effect: SpellEffectHint::CounterSpell,
    ..
}

// Grey Ogre, Llanowar Elves, lands — all SpellEffectHint::None (default)
```

### 4. Observation encoding

In `agent/observation.rs`, add to `CardData`:

```rust
pub struct CardData {
    // existing...
    pub spell_effect_type: i32,    // SpellEffectHint discriminant (0=None, 1=DealDamage, 2=CounterSpell)
    pub spell_effect_amount: i32,  // numeric parameter (e.g. 3 for bolt), 0 otherwise
}
```

### 5. Python bindings

- `PyCard` gains `spell_effect_type` and `spell_effect_amount` fields
- ObservationEncoder adds 2 features to the card feature vector
- Card feature dimension increases from 18 to 20

### 6. Target type observation

Also add target spec information so the agent can see what a spell
targets before casting:

```rust
pub struct CardData {
    // existing + spell_effect fields...
    pub target_type: i32,  // -1=no target, 0=AnyCreatureOrPlayer, 1=SpellOnStack, etc.
}
```

This mirrors `TargetSpec` from `state/ability.rs` but as an observation
field. The agent can see that Lightning Bolt targets "any creature or
player" while Counterspell targets "spell on stack."

## Done when

- `CardData` for Lightning Bolt has `spell_effect_type=1, spell_effect_amount=3`
- `CardData` for Counterspell has `spell_effect_type=2, spell_effect_amount=0`
- `CardData` for Grey Ogre has `spell_effect_type=0, spell_effect_amount=0`
- Python observation includes the new fields
- `cargo test` and `pytest tests/env/` pass
- Card feature dimension documented in encoder config
