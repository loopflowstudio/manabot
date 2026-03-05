# Priority/Stack + Event System

## Problem

The engine currently only supports sorcery-speed actions: the active player
casts spells during main phases with an empty stack, and the nonactive player
can only pass priority. There are no instant-speed response windows — once a
spell is cast, it resolves immediately because `tick_priority` loops through
`pass_count` without allowing the opponent to respond to items on the stack.

This blocks every card that interacts at instant speed (Lightning Bolt,
counterspells, combat tricks) and every triggered ability that uses the stack.
The event system is the prerequisite for triggered abilities, replacement
effects, and continuous effects in later stages.

Who benefits: the RL agent gets a materially richer decision space; the engine
becomes capable of expressing the cards listed in wave goals (Lightning Bolt,
Man-o'-War).

Why now: this unblocks real instant interaction and establishes the minimum
targeting seam needed for stack play (Bolt + Counterspell), while still leaving
triggers/replacement/layers for later stages.

## Approach

### 1. Instant-speed casting in priority windows

Add `can_cast_instants(player) -> bool` alongside `can_cast_sorceries`. During
`compute_player_actions`, when a player has priority and the stack is nonempty
(or it's not their main phase), offer instant-speed spells they can afford. CR
117.1a already defines the timing — we just need to stop gating everything
behind `can_cast_sorceries`.

**Key changes in `game.rs`:**

```rust
pub fn can_cast_instants(&self, player: PlayerId) -> bool {
    // CR 117.1a — Any player with priority may cast an instant.
    true // Priority is already checked by the caller (tick_priority)
}
```

Update `compute_player_actions` and `can_player_act` to check both sorcery and
instant timing:

```rust
// For each card in hand:
if card.types.is_instant() && can_cast_instants {
    // offer CastSpell
} else if is_castable && can_cast_sorceries {
    // offer CastSpell (sorcery-speed, as today)
}
```

`CastSpell` still starts casting, but targeted spells now branch into a
`ChooseTarget` action space before becoming fully cast spells on the stack.

### 2. Card type: add `is_instant()` helper

`CardTypes` already has helpers for every type except a standalone `is_instant`.
Add it. Also add a `speed()` or `is_instant_speed()` method that returns true
for instants (and later, cards with flash).

### 3. Priority pass tracking — per-player

Reframe `PriorityState` so we stop overloading `pass_count` as both
"who has priority" and "how many players have passed."

**New state model:**

```rust
pub struct PriorityState {
    pub holder: PlayerId,
    pub consecutive_passes: usize,
    pub sba_done: bool,
}
```

`holder` is the player who currently has priority. `consecutive_passes` tracks
CR 117.4 pass sequencing.

**Transition helpers (single source of truth):**

```rust
impl PriorityState {
    pub fn start_round(&mut self, active: PlayerId) {
        self.holder = active;
        self.consecutive_passes = 0;
        self.sba_done = false;
    }

    pub fn on_pass(&mut self, next: PlayerId) {
        self.consecutive_passes += 1;
        self.holder = next;
    }

    pub fn on_non_pass_action(&mut self, active: PlayerId) {
        // CR 117.3b — After a player takes an action, AP gets priority.
        self.holder = active;
        self.consecutive_passes = 0;
        self.sba_done = false;
    }
}
```

**Engine impact:**
- `tick_priority()` grants priority to `state.priority.holder` directly
- all-pass check becomes `consecutive_passes >= players.len()`
- `PassPriority` calls `on_pass(next_player)`
- every successful non-pass action calls `on_non_pass_action(active_player)`

This is a small holistic cleanup (state-machine semantics), not a large control
flow rewrite.

### 4. Post-resolution priority (CR 405.5)

After resolving a stack object, the active player receives priority again (CR
405.5). Keep current control flow, but route it through the new transition
helper:

```rust
self.resolve_top_of_stack();
self.state.priority.on_non_pass_action(self.active_player()); // CR 405.5
return self.tick_priority();
```

No control-flow change required here; cover this explicitly with CR 405.5
tests to lock behavior.

### 5. Single-target system + pending cast flow

No fake targeting behavior. Add real target selection now, constrained to
single-target spells.

**Target model:**

```rust
pub enum Target {
    Player(PlayerId),
    Permanent(PermanentId),
    StackSpell(CardId),
}
```

**Action model additions:**
- `ActionSpaceKind::ChooseTarget`
- `Action::ChooseTarget { player, target }`
- `PendingChoice::ChooseTarget { player, card, legal_targets }`

**Cast flow (targeted spell):**
1. `CastSpell { card }` selected from priority space.
2. Engine computes legal targets (CR 601.2c). If none, `CastSpell` is not
   offered.
3. Engine enters `ChooseTarget` action space with explicit legal target actions.
4. On `ChooseTarget`, engine finalizes cast atomically: pay cost, move spell to
   stack, record chosen target metadata, emit `SpellCast`.

**Stack metadata (minimal):**
- Keep existing zone stack order (`Vec<CardId>`) for ordering.
- Add side metadata map for spell targets:
  `HashMap<CardId, Target>` (single target only).
- Resolution reads target metadata for effect legality/application.

This keeps scope bounded while avoiding throwaway auto-target behavior.

### 6. Real instant cards: Lightning Bolt + Counterspell

Implement both cards using real target selection:

- **Lightning Bolt** — `R`, Instant, target `Player | CreaturePermanent`,
  deal 3 damage on resolution.
- **Counterspell** — `UU`, Instant, target `StackSpell`, counter that spell on
  resolution.

**Resolution rules (subset):**
- If a spell’s target is illegal on resolution, that spell is countered by
  rules and has no effect (CR 608.2b subset).
- Counterspell checks target is still on stack; if legal, move target spell to
  graveyard, then Counterspell to graveyard.
- Bolt applies damage to chosen legal target, then goes to graveyard.

No multi-target, no target-retarget effects, no "up to" target counts.

### 7. GameEvent system

Introduce `GameEvent` as a simple `Vec<GameEvent>` log on `GameState`. Events
are appended during game actions and drained at defined points. No subscribers
yet — the log exists for later stages to consume.

**File:** `managym/src/flow/event.rs`

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum GameEvent {
    CardMoved { card: CardId, from: ZoneType, to: ZoneType },
    DamageDealt { source: Option<CardId>, target: DamageTarget, amount: u32 },
    LifeChanged { player: PlayerId, old: i32, new: i32 },
    SpellCast { card: CardId, target: Option<Target> },
    SpellResolved { card: CardId },
    SpellCountered { card: CardId, by: Option<CardId> },
    TurnStarted { player: PlayerId },
    StepStarted { step: StepKind },
}

#[derive(Clone, Debug, PartialEq)]
pub enum DamageTarget {
    Player(PlayerId),
    Permanent(PermanentId),
}
```

**Emit points** (all in `game.rs`, minimal diff):
- `move_card()` → `CardMoved`
- `take_damage()` on player → `DamageDealt` + `LifeChanged`
- `take_damage()` on permanent → `DamageDealt`
- `cast_spell()` → `SpellCast`
- `resolve_top_of_stack()` → `SpellResolved`
- `counter_spell()` (or equivalent path) → `SpellCountered`
- `on_step_start()` for first step → `TurnStarted`
- `on_step_start()` → `StepStarted`

Drop `ManaProduced` and `AbilityActivated` from the original spec — neither is
needed until mana abilities become first-class objects (not this stage).

**Storage:**
```rust
pub struct GameState {
    // ... existing fields ...
    pub events: Vec<GameEvent>,
}
```

Events append to `GameState.events` and are never auto-cleared. Add
`drain_events()` as the single read/reset API:

```rust
pub fn drain_events(&mut self) -> Vec<GameEvent> {
    std::mem::take(&mut self.state.events)
}
```

Tests can assert on drained batches; stage 04 triggers will consume events
through this same API.

**Test policy:** prefer drain-only assertions (`drain_events()`) for rules and
integration tests. Allow direct `state.events` inspection only in narrow
engine-internal tests when required for debugging/diagnostics.

### 8. Test plan

New CR-cited tests in `managym/tests/rules/`:

**cr_117_priority.rs (extend):**
- `cr_117_1a_instant_cast_during_opponents_turn` — nonactive player casts
  Lightning Bolt when active player passes priority with nonempty stack
- `cr_117_3b_after_action_ap_gets_priority` — after casting, priority resets
  to AP
- `cr_117_regression_nap_does_not_retain_priority_after_cast` — explicit
  regression for the old `pass_count` bug

**cr_405_stack.rs (new):**
- `cr_405_1_lifo_resolution` — cast two spells, verify LIFO resolution order
- `cr_405_5_priority_after_resolution` — after top resolves, AP gets priority
  before next item resolves
- `cr_405_counterspell_targets_stack_object` — Counterspell can target top
  stack spell

**cr_601_casting.rs (extend, CR 601.2c subset):**
- `cr_601_instant_timing` — instant can be cast when stack is nonempty
- `cr_601_target_selection_required_for_targeted_spells` — cast leads to
  `ChooseTarget` before spell is put on stack
- `cr_601_spell_not_offered_without_legal_target` — targeted spell absent when
  no legal targets exist
- `cr_601_sorcery_cannot_respond` — sorcery cannot be cast when stack is
  nonempty (negative path)

**cr_608_resolution.rs (new):**
- `cr_608_2b_illegal_target_spell_has_no_effect` — spell fizzles on illegal
  target
- `cr_608_counterspell_moves_target_to_graveyard` — Counterspell counters legal
  target spell

**Event log assertions:**
- `event_log_records_zone_changes` — cast/counter spell, check CardMoved events
- `event_log_records_damage` — Lightning Bolt resolves, check DamageDealt
- `event_log_records_life_change` — damage changes life, check LifeChanged
- `event_log_records_counterspell` — Counterspell emits SpellCountered

**Training smoke test:**
- Agent beats random in 100k steps with Bolt/Counterspell-enabled decks

### 9. Update rules_coverage.yaml

Add entries for:
- CR 405 — Stack (implemented_tested)
- CR 117 — Update notes to reflect instant-speed support
- CR 601 — Update notes to reflect target-selection cast flow
- CR 608.2b (subset) — illegal-target resolution behavior

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Auto-target Lightning Bolt | Fastest implementation of response windows | Rejected: fake card behavior creates semantics debt and false confidence. |
| Counterspell-only (no Bolt) | Exercises stack targeting without permanent/player targeting | Rejected: loses wave goal pressure on damage/any-target interactions. |
| Full generalized targeting + multi-target now | Most future-proof target model | Rejected: scope blow-up; single-target covers current milestone cleanly. |
| Event subscribers (pub-sub) | Proper event system with listener registration | Over-engineering for this stage. Vec<GameEvent> log is simpler, and no consumers exist yet. Add pub-sub when triggers need it. |
| Full stack object refactor now | Strong long-term model (`StackEntry`) | Deferred: side-map target metadata is enough for single-target stage. |
| Refactor game.rs into smaller modules first | Addresses the "882-line monolith" risk | Not the goal of this stage. Event emission is additive — we add `emit()` calls, not restructure control flow. Refactoring as a separate cleanup if needed. |

## Key decisions

**No fake cards.** Lightning Bolt and Counterspell both use real target choice
at cast time. We pay a moderate scope increase now to avoid throwaway behavior.

**Events as a log, not pub-sub.** `Vec<GameEvent>` appended during game
actions. No registration, no callbacks. Triggers (stage 04) will drain the
event log to determine what fires. This avoids borrow-checker complexity from
callback closures.

**Single-target only.** Target system includes `Player`, `Permanent`, and
`StackSpell`, but only one target per spell this stage.

**DamageTarget enum.** Damage can target players or permanents. This replaces
the `ObjectId` approach from the original spec, which conflated card IDs and
player IDs. Explicit enum is safer.

**No `ManaProduced` or `AbilityActivated` events.** These add noise without
consumers. Mana abilities don't use the stack (CR 605.3b). Add them when a
feature needs them.

## Scope

**In scope:**
- Instant-speed casting in priority windows (any player with priority can cast instants)
- Priority-state-machine cleanup (`holder` + `consecutive_passes`)
- Single-target infrastructure (`Target::Player | Permanent | StackSpell`)
- `ChooseTarget` action space + pending cast flow
- Lightning Bolt (real target choice)
- Counterspell (real stack-spell target choice)
- `GameEvent` enum and emission at key points in `game.rs`
- CR-cited tests for stack resolution/timing/targeting (CR 117/405/601/608 subset)
- Training smoke test: agent beats random with Bolt+Counterspell decks

**Out of scope (per wave "Not here" + dependency order):**
- Multi-target and optional-target spells
- Retargeting and target-changing effects
- Full declarative effect DSL
- Triggered abilities consuming events (stage 04)
- Replacement effects (stage 04+)
- Activated abilities other than mana abilities
- Flash keyword
- Multiplayer

## Roadmap adjustment (stage boundary re-scope)

This stage absorbs the minimal real targeting system. Wave stage 03 shifts from
"first targeting implementation" to "targeting + DSL expansion" (broader
target classes, richer legality filters, and declarative effect encoding).

## Done when

```bash
cd managym && cargo test
```

- All existing tests pass (no regressions).
- New CR-cited tests pass:
  - CR 117: instant cast during opponent's turn, priority reset after action
  - CR 405: LIFO resolution, priority between resolutions
  - CR 601 (subset): targeted cast flow requires legal target choice
  - CR 608.2b (subset): illegal target causes no effect
- Event log tests verify CardMoved, DamageDealt, LifeChanged emission.
- Lightning Bolt resolves for 3 damage to chosen legal target.
- Counterspell can target and counter a spell on the stack.
- `rules_coverage.yaml` updated with CR 405 entry and instant-speed notes.

```bash
cd managym && pip install -e . && python -c "
from manabot.model.train import main; main(['--preset', 'simple'])
"
```

- Training smoke test: agent beats random baseline in 100k steps with
  Bolt+Counterspell decks.

**Wave goals advanced:**
- Goal 1: "Every implemented rule family has focused CR-cited trace tests" — adds CR 405 stack tests
- Goal 2: "Lightning Bolt and Man-o'-War land early to force stack/target/trigger behavior" — Lightning Bolt lands with real targeting; Counterspell validates stack targeting
- Goal 5: "Training remains stable as branching factor grows" — smoke test validates
