# 03: Declarative Effect DSL + Targeting Expansion

## Finish line

Declarative effect DSL exists and executes Lightning Bolt (and future cards)
generically. Targeting model expands beyond the single-target infrastructure
already landed in stage 02.

## Context (what's already landed)

Stage 02 shipped:
- Single-target infrastructure (`Target::Player | Permanent | StackSpell`)
- `ChooseTarget` action space + pending cast flow (CR 601.2c)
- Lightning Bolt and Counterspell with real target selection at cast time
- `GameEvent` enum and emission log (`Vec<GameEvent>` on `GameState`)
- Priority/stack response windows with instant-speed casting
- CR-cited tests for CR 117/405/601/608

What's hardcoded now: Bolt and Counterspell effects are implemented directly in
`resolve_top_of_stack()` in `game.rs` — no declarative representation yet.
Target metadata is stored as `HashMap<CardId, Target>` (single target only).

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

Migrate Bolt and Counterspell from hardcoded resolution to DSL-driven
resolution. This validates the DSL against cards that already work.

### Broader Target Classes

Extend targeting beyond single-target:
- `TargetSpec` with `count` field for required vs optional targets
- Target filters: creature, player, spell-on-stack, any-target
- Legal target computation driven by `TargetSpec` rather than per-card checks

### Card Registry

Generalize card definition to include declarative effects. Lightning Bolt
becomes the template for effect-bearing cards.

### Trace Tests

- Bolt with declarative DSL resolves identically to hardcoded version
- Legal/illegal target filtering via `TargetSpec`
- Negative: can't cast bolt with no legal targets (already tested, verify DSL path)

### Training smoke test

Agent beats random with bolt-heavy decks. Verify no regression from DSL migration.

## Alternatives considered

From stage 02 design doc:
- Auto-target approach was rejected in favor of real target selection (now landed).
- Full generalized multi-target was deferred — extend here if needed for new cards.
- Event subscribers (pub-sub) deferred — Vec log is sufficient until triggers consume it.

## Done when

- Declarative effect DSL exists and executes Bolt + Counterspell correctly.
- `TargetSpec`-driven target legality replaces per-card target checks.
- Existing CR-cited tests continue to pass (no regressions from DSL migration).
- Training smoke test passes.
