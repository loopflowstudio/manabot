# 04: Triggered Abilities v1 (Man-o'-War)

## Problem

managym has no ability system beyond mana abilities. Cards resolve as vanilla
permanents or go to graveyard — no effects fire. Triggered abilities are the
first class of abilities that interact with game events, and they exercise the
full priority/stack loop: events fire, triggers queue, targets are chosen,
abilities go on stack, opponents can respond, abilities resolve.

Man-o'-War (2U Creature — Jellyfish, 2/2, "When Man-o'-War enters the
battlefield, return target creature to its owner's hand") is the driver card
because it exercises ETB detection, target selection, stack interaction, and a
zone-change effect — all in one card.

This advances wave goals:
- **Goal 2**: "Man-o'-War lands early to force stack/target/trigger behavior."
- **Goal 1**: "Every implemented rule family has focused CR-cited trace tests."
- **Goal 6**: "Declarative DSL enables cross-card generalization."

## Handoff status (for fresh-context implementer)

- Stage 04 is intentionally scoped to land **after** stage 02+03 are complete.
- Event model expectation:
  - **Required runtime queue** for rule processing (`pending_events` + `drain_events()`).
  - Optional persistent `event_history` is debug/test instrumentation and is not
    required for stage 04 correctness.
- Same-controller simultaneous trigger ordering (CR 603.3b player choice) is
  deferred; v1 uses deterministic enqueue order after APNAP partitioning.

## Approach

### Prerequisites from stages 02-03

This design assumes stages 02 and 03 have landed:
- **Stage 02**: `GameEvent` enum with `CardMoved`, `SpellResolved`, etc. plus
  a runtime event queue on `GameState` drained via `drain_events()`.
- **Stage 03**: `ChooseTarget` action space kind, `Target` enum, single-target
  infrastructure, `spell_targets` map on `GameState`.

If these haven't landed, they must ship first.

### Architecture: Three new concepts

**1. Ability definitions on cards** — Extend `CardDefinition` with a
`Vec<Ability>` field. Each `Ability` is an enum:

```rust
// state/ability.rs (new file)

#[derive(Clone, Debug)]
pub enum Ability {
    Triggered {
        condition: TriggerCondition,
        effect: Effect,
        /// CR 603.4 — intervening "if" clause, checked on trigger AND resolution
        intervening_if: Option<TriggerCondition>,
    },
    // Future: Activated, Static
}

#[derive(Clone, Debug)]
pub enum TriggerCondition {
    /// CR 603.6a — "When [this] enters the battlefield"
    EntersTheBattlefield { source: TriggerSource },
}

#[derive(Clone, Debug)]
pub enum TriggerSource {
    /// The permanent with this ability
    This,
    // Future: Any { filter: Filter }, Controller, etc.
}

#[derive(Clone, Debug)]
pub enum Effect {
    /// Return target creature to owner's hand
    ReturnToHand { target: TargetSpec },
    // Future: DealDamage, DrawCard, etc.
}

#[derive(Clone, Debug)]
pub enum TargetSpec {
    /// Target any permanent matching filter
    Creature { controller: TargetController },
}

#[derive(Clone, Debug)]
pub enum TargetController {
    Any,
    // Future: Opponent, You
}
```

Keep the ability DSL minimal — only what Man-o'-War needs. Extend for later
cards (Lightning Bolt adds `DealDamage`, keywords add `Static`).

**2. Trigger queue on GameState** — A `Vec<PendingTrigger>` that accumulates
triggers as events fire, then flushes them onto the stack when a player would
receive priority.

```rust
// flow/trigger.rs (new file)

#[derive(Clone, Debug)]
pub struct PendingTrigger {
    /// Source card at trigger time (survives source leaving battlefield)
    pub source_card: CardId,
    /// Which ability on the card triggered (index into abilities vec)
    pub ability_index: usize,
    /// Controller at time of trigger (for APNAP ordering)
    pub controller: PlayerId,
    /// Monotonic enqueue order for deterministic same-controller ordering in v1
    pub enqueue_order: u64,
}
```

**3. Stack objects for abilities** — Currently the stack only holds cards
(spells). Triggered abilities on the stack are not cards — they're ability
objects (CR 113.3a). Extend the stack to hold `StackObject`:

```rust
// state/stack.rs (new file, or extend zone.rs)

#[derive(Clone, Debug)]
pub enum StackObject {
    Spell { card: CardId },
    TriggeredAbility {
        source_card: CardId,
        ability_index: usize,
        controller: PlayerId,
        target: Option<Target>,
    },
}
```

If stage 02+03 already converted stack storage to `StackObject::Spell`, then
stage 04 only adds `TriggeredAbility` handling. If not, stage 04 performs the
conversion from `Vec<CardId>` to `Vec<StackObject>`.

### Integration into the game loop

The trigger lifecycle follows CR 603:

```
Event fires (CardMoved to Battlefield)
  → check_triggers(): scan all permanents' abilities for matching conditions
  → matching triggers added to pending_triggers queue

Player would receive priority (tick_priority entry)
  → flush_triggers(): APNAP-order pending triggers, put on stack
  → for each trigger: if it has targets, present ChooseTarget action space
  → if no legal targets exist for a required target, do not place ability (CR 603.3d)
  → after all triggers placed, proceed to normal priority

All players pass priority
  → resolve_top_of_stack()
  → if top is TriggeredAbility: execute its effect
  → if top is Spell: existing behavior (permanent→battlefield, etc.)
```

Concrete insertion point in `tick_priority()`:

```rust
fn tick_priority(&mut self) -> Option<ActionSpace> {
    // 1. SBAs (existing)
    if !self.state.priority.sba_done {
        self.perform_state_based_actions();
        self.state.priority.sba_done = true;
        if self.is_game_over() { return None; }
    }

    // 2. NEW: flush pending triggers onto stack (CR 603.3)
    if !self.state.pending_triggers.is_empty() {
        return self.flush_triggers();
    }

    // 3. Priority loop (existing)
    // ...
}
```

### Man-o'-War card definition

```rust
CardDefinition {
    name: "Man-o'-War".to_string(),
    mana_cost: Some(ManaCost::parse("2U")),
    types: CardTypes::new([CardType::Creature]),
    subtypes: vec!["Jellyfish".to_string()],
    abilities: vec![
        Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                source: TriggerSource::This,
            },
            effect: Effect::ReturnToHand {
                target: TargetSpec::Creature {
                    controller: TargetController::Any,
                },
            },
            intervening_if: None,
        },
    ],
    mana_abilities: vec![],
    text_box: "When Man-o'-War enters the battlefield, return target creature to its owner's hand.".to_string(),
    power: Some(2),
    toughness: Some(2),
}
```

### Effect execution

When a `TriggeredAbility` resolves from the stack:

```rust
fn resolve_triggered_ability(&mut self, ability: &TriggeredAbility) {
    let ability_def = self.lookup_ability(ability.source_card, ability.ability_index);

    // CR 603.4 — re-check intervening "if" on resolution
    if let Some(condition) = &ability_def.intervening_if {
        if !self.check_trigger_condition(condition, ability.source_card) {
            return; // fizzles
        }
    }

    match &ability_def.effect {
        Effect::ReturnToHand { .. } => {
            if let Some(target) = ability.target {
                // Validate target still legal (CR 608.2b)
                if self.is_valid_target(target) {
                    self.return_to_hand(target);
                }
                // If target illegal, ability does nothing (fizzles)
            }
        }
    }
}
```

### Module extraction

`game.rs` is ~900 lines now and will grow ~200-300 with triggers. Extract
resolution logic into `flow/resolution.rs`:
- `resolve_top_of_stack()`
- `resolve_triggered_ability()`
- `execute_effect()`

Keep `tick_priority()` and the main loop in `game.rs`.

### Execution order (implementation checklist)

1. Add `Ability`/trigger/effect DSL types and wire card definitions.
2. Add `PendingTrigger` queue + enqueue logic from drained `GameEvent`s.
3. Extend stack object model for triggered abilities (if not already done).
4. Integrate `flush_triggers()` into priority entrypoint (after SBA, before
   player actions).
5. Implement triggered ability target selection + placement legality checks.
6. Implement triggered ability resolution + target revalidation.
7. Add Man-o'-War to registry and pass CR-cited trace tests.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Observer/callback pattern for triggers | Flexible, decoupled | Over-engineered for current needs. MTG triggers have very specific timing rules that don't map to generic observer patterns. Direct scanning is simpler and more correct. |
| Triggers as standalone objects (not on abilities) | Separate trigger registration from card defs | Splits card behavior across two places. Declarative abilities on cards serve the DSL vision (wave goal 6) and keep card definitions self-contained. |
| Generic `Effect` tree with `Box<dyn Effect>` trait objects | Extensible | Trait objects complicate Clone/Debug/serialization. Enum is fine at this scale and the observation encoder needs to pattern-match anyway. |
| Skip stack for ETB triggers (resolve immediately) | Simpler | Wrong per CR 603.3. Triggers must go on stack so opponents can respond. Shortcuts now would need rework for counterspell/Stifle interactions later. |

## Key decisions

**StackObject enum over card-only stack.** The stack currently holds only
`CardId`. CR 113.3a says triggered abilities on the stack are objects, not cards.
We need `StackObject::TriggeredAbility` to properly model this. This is the
right time to make this change — doing it later means more code to retrofit.

**Abilities on CardDefinition, not Permanent.** Abilities are defined on the
card, not the permanent. The permanent references the card for its abilities.
This matches MTG's model (permanents have abilities from their card plus any
granted by effects) and keeps card definitions as the single source of truth.

**Minimal DSL — only what Man-o'-War needs.** `TriggerCondition` has one
variant (ETB). `Effect` has one variant (ReturnToHand). `TargetSpec` has one
variant (Creature). Future cards add variants without changing the architecture.
This avoids the risk of "DSL design locks in too early" (wave risk 5).

**APNAP ordering built in from the start.** Even though our first test case is
single-trigger, the queue sorts by APNAP (active player's triggers first, then
non-active). Within each player, v1 uses enqueue order (deterministic). Full
603.3b "player chooses relative order of their simultaneous triggers" is a
later extension once we have cards that require it. Two-Man-o'-War test
validates APNAP first.

**Intervening "if" clause on the Ability, not the TriggerCondition.** Man-o'-War
doesn't have an intervening "if" but the structure supports it from day one. The
cost is one `Option` field. Cards like "When ~ enters the battlefield, if you
control another creature..." will use it without restructuring.

**Target selection reuses ChooseTarget from stage 03.** When a trigger goes on
the stack, if it has targets, the game presents a `ChooseTarget` action space
to the controller. Same flow as targeting for spells. No new action space kind
needed.

## Scope

**In scope:**
- `Ability` enum with `Triggered` variant
- `TriggerCondition::EntersTheBattlefield`
- `Effect::ReturnToHand`
- `TargetSpec::Creature`
- `PendingTrigger` queue on `GameState`
- `StackObject` enum (Spell + TriggeredAbility)
- `check_triggers()` — scan permanents after zone changes
- `flush_triggers()` — APNAP-order and place on stack with target selection
- Trigger resolution with target validation
- Man-o'-War card definition
- 5 trace tests (ETB bounce, timing, two Man-o'-Wars, illegal target path, APNAP)
- Training smoke test
- Module extraction: `flow/resolution.rs`

**Out of scope (wave "Not here" + stage boundaries):**
- Multi-target abilities
- Activated abilities
- Static abilities / continuous effects
- "Leaves the battlefield" triggers
- Trigger replacement (e.g., "If ~ would enter...")
- Stifle / counterspell for triggered abilities
- Complex intervening "if" conditions (just the plumbing)

## Done when

```bash
cargo test --test rules_tests    # all trigger trace tests pass
cargo clippy --all-targets       # no warnings
cargo test                       # no regressions
```

- `cr_603_triggers.rs` has 5+ passing tests covering:
  - Man-o'-War ETB returns a creature to hand
  - Trigger goes on stack, opponent gets priority before resolution
  - Two Man-o'-Wars entering simultaneously → APNAP ordering
  - Triggered ability target becomes illegal before resolution → ability does nothing
  - Trigger timing: SBAs checked before triggers flush
- Man-o'-War in card registry with declarative `Ability`
- Stack model supports both `Spell` and `TriggeredAbility` objects
- Training smoke test: agent learns Man-o'-War sequencing, episode length tracked
- Advances wave goal 2 ("Man-o'-War lands early") and goal 1 (CR-cited tests)
