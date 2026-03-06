# 04: Spell Effect Observations

**Finish line:** The agent can distinguish spells by what they do.
Lightning Bolt is observable as "deal 3 damage to any target."
Counterspell is observable as "counter target spell." Creatures and lands
with no spell effects have a null marker.

## Context

The agent sees card stats (P/T, mana cost, type flags) and — since
sprint 03 — game events (damage dealt, spells cast, life changes). But
for spells in hand, the agent can't see *what a spell does* before
deciding to cast it. Two Lightning Bolts and two random 1-mana instants
look identical in the observation.

Sprint 03 established the event observation pipeline: `GameEvent`
variants, dual-drain accumulator (`emit_event` → `pending_events` +
`observation_events`), `EventData` in `Observation`, and matching
Rust/Python encoders with `EVENT_DIM = 7` and configurable `max_events`
(default 32). Schema enums `EventType` and `EventEntityKind` are
`#[repr(i32)]` with stable discriminants.

This sprint adds static spell metadata to the observation — what spells
do, not what happened.

## Codebase state

- `CardData` (observation.rs:54) has 9 fields: zone, owner_id, id,
  registry_key, name, power, toughness, card_types, mana_cost
- `CARD_DIM = 18` (observation_encoder.rs:9)
- No `KeywordData` struct exists yet — keywords are not currently
  observable. Sprint 04 does not add keyword observability.
- `TargetSpec` exists in `state/ability.rs`
- Card definitions live in `cardsets/` (e.g., `cardsets/visions.rs`)

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

In `cardsets/`:
- Lightning Bolt → `SpellEffectHint::DealDamage { amount: 3 }`
- Counterspell → `SpellEffectHint::CounterSpell`
- All creatures, lands → `SpellEffectHint::None` (default)

### 4. Observation encoding

Add to `CardData`:

```rust
pub struct CardData {
    // existing...
    pub spell_effect_type: i32,    // 0=None, 1=DealDamage, 2=CounterSpell
    pub spell_effect_amount: i32,  // e.g. 3 for bolt, 0 otherwise
    pub target_type: i32,          // -1=no target, maps from TargetSpec
}
```

`CARD_DIM` increases from 18 to 21 (3 new features).

### 5. Python bindings and encoder

- `PyCard` gains `spell_effect_type`, `spell_effect_amount`, `target_type`
- Python `ObservationEncoder.card_dim` increases to match
- `_encode_card_features` encodes the new fields

### 6. Rust observation encoder

`encode_card_features` in `observation_encoder.rs` encodes the 3 new
fields. `CARD_DIM` constant updated.

## Done when

- `CardData` for Lightning Bolt has `spell_effect_type=1, spell_effect_amount=3`
- `CardData` for Counterspell has `spell_effect_type=2, spell_effect_amount=0`
- `CardData` for Grey Ogre has `spell_effect_type=0, spell_effect_amount=0`
- Python observation includes the new fields
- `cargo test` and `pytest tests/env/` pass
- `CARD_DIM` updated in both Rust and Python encoders
- Rust and Python card feature shapes match
