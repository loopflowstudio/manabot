# 03: Lightning Bolt (Targeting + Instant Interaction)

## Finish line

Lightning Bolt is implemented with declarative effect DSL, sequential target
selection, and engine-side target legality — validating real instant-speed
interaction on the stack.

## Changes

### Declarative Effect DSL (initial)

Introduce the effect representation model. Lightning Bolt is the first card
to use it:

```rust
Effect::Damage {
    amount: Value::Fixed(3),
    target: TargetSpec::Any {
        filter: Filter::Or(Filter::Creature, Filter::Player),
        count: 1,
    },
    source: Source::This,
}
```

The DSL is declarative — the engine interprets effect trees generically.
Design with observation-encoding in mind (effect trees will be flattened into
features in stage 12).

### Sequential Target Selection

Add `ActionSpaceKind::ChooseTarget` and `PendingChoice::ChooseTarget`.
Casting a targeted spell becomes a two-step sequence:
1. Agent chooses `CastSpell { card: bolt }` (only offered if legal targets exist)
2. Engine presents `ChooseTarget` action space with legal targets
3. Agent picks a target

Target legality checked engine-side in `compute_player_actions` (CR 601.2c).
Agent never sees a spell it can't legally complete.

Single-target only for now. `TargetSpec` includes `count` field to
distinguish required vs optional targets for future extension.

### Card Registry

Add Lightning Bolt with declarative effect. Keep target model extensible for
planeswalkers/battles later.

### Trace Tests

- Legal/illegal targets
- Stack resolution ordering (bolt in response to bolt)
- Lethal player/creature outcomes
- Negative: can't cast bolt with no legal targets

### Training smoke test

Agent beats random with bolt-heavy decks. Track action space size increase.

## Done when

- Bolt castable at instant speed via sequential target selection.
- Declarative effect DSL exists and executes Bolt correctly.
- Trace tests covering targeting and stack interaction pass.
- Training smoke test passes.
