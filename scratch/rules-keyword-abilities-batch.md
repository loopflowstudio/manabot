# Keyword Abilities Batch

## Problem

The managym engine has 6 cards (5 basics + Grey Ogre + Llanowar Elves) and zero
keyword abilities. Combat is binary: attack, maybe get blocked, simultaneous
damage. The RL agent has no reason to make interesting combat decisions — every
creature is functionally identical except for power/toughness.

Adding 11 evergreen keywords and 11 new cards transforms combat from arithmetic
into strategy: evasion, racing, trading, tempo. This is the single highest-value
change for training signal richness without needing triggers, the stack, or an
effect system.

## Approach

### Keywords struct on CardDefinition

Add a `Keywords` struct with 11 boolean flags to `CardDefinition`. Cards declare
keywords at registration time. `Permanent` accesses keywords through its card
reference (`game.state.cards[permanent.card].keywords`). No keywords on
`Permanent` directly — that's for a future "grant/remove keywords" effect system.

```rust
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Keywords {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
}
```

### Keyword implementation by category

**1. Simple flag checks (5 keywords)**

| Keyword | Where | Change |
|---------|-------|--------|
| Defender (CR 702.3) | `Permanent::can_attack()` | Return false if `keywords.defender` |
| Haste (CR 702.10) | `Permanent::new()` | Set `summoning_sick = false` if `keywords.haste` |
| Vigilance (CR 702.20) | `Permanent::attack()` | Skip `self.tap()` if `keywords.vigilance` |
| Flying (CR 702.9) | `Game::eligible_blockers()` | Attacker has flying → blocker must have flying or reach |
| Reach (CR 702.17) | Same check | Reach allows blocking flyers |

Flying/reach modify blocker legality. Currently `eligible_blockers()` returns all
untapped creatures. Change: when generating blocker actions for an attacker with
flying, filter to only creatures with flying or reach. This is per-attacker
filtering, applied during action space generation.

**2. Menace (CR 702.111) — blocking constraint**

Menace requires 2+ blockers. The current action space declares one blocker at a
time, so we can't validate during declaration. Instead, validate after all
blockers are declared:

- After `DeclareBlockers` completes, check each attacker with menace
- If an attacker with menace has exactly 1 blocker, remove that blocker assignment
  (the block is illegal; the blocker is freed, the attacker is unblocked)
- This matches CR 509.1b: "illegal blocks are removed"

This approach avoids changing the action space structure. The agent learns that
blocking a menace creature alone is wasteful (its block gets removed). Over
training, it learns to either commit 2+ blockers or not block at all.

**Rule-timing constraint (CR 509.1b, 509.1h, 506.4a):**
- Block legality is checked during declare-blockers declaration.
- Once blocks are legally declared, later gain/loss of evasion does not undo that
  block.
- A creature remains blocked even if all blockers are later removed from combat.

Engine policy for this stage:
- Enforce flying/reach legality at declaration time (action generation + defensive
  validation in `declare_blocker`).
- Run menace legality cleanup exactly once at end of `DeclareBlockers` turn-based
  action (before priority), not after later combat changes.
- Do not revalidate block legality after declaration has completed.

**3. Lifelink (CR 702.15) — damage modifier**

In `apply_player_damage()` and `apply_permanent_damage()`, if the source card has
lifelink, the source's controller gains life equal to the damage dealt. This
works for both combat and non-combat damage (e.g., Lightning Bolt from a
theoretical lifelink source, though no current cards combine these).

**4. Deathtouch (CR 702.2) — SBA modifier**

Change `Permanent::has_lethal_damage()`: if the source of any damage marked on
this creature had deathtouch, any amount of damage (>= 1) is lethal.

Implementation detail: track whether a permanent has received deathtouch damage
this turn. Add `deathtouch_damage: bool` to `Permanent`. Set it in
`apply_permanent_damage()` when the source has deathtouch. Check it in
`has_lethal_damage()`. Reset in `clear_damage()`.

**5. Trample (CR 702.19) — combat damage assignment**

Trample requires ordered damage assignment. Currently, the attacker deals its
full power to each blocker (bug: should split, not duplicate). Fix the underlying
damage assignment AND add trample:

For blocked attackers:
1. Order blockers (currently: insertion order from declaration)
2. Assign lethal damage to each blocker in order (lethal = toughness - existing damage; with deathtouch, lethal = 1)
3. Without trample: remaining damage is wasted
4. With trample: remaining damage dealt to defending player

This fixes the existing bug where a 2/2 blocked by two 1/1s deals 2 damage to
each blocker instead of 2 total split between them.

**6. First Strike / Double Strike (CR 702.7, 702.4) — combat sub-steps**

The hardest implementation. Combat damage splits into two sub-steps when any
creature has first strike or double strike.

**Design choice: internal sub-step, not new StepKind.** The `StepKind` enum is
part of the observation contract (tested in `observation_contract_enum_values_stable`).
Adding variants would break Python-side observation parsing. Instead, handle
sub-steps as two internal passes inside one `resolve_combat_damage()` call:

In `resolve_combat_damage()`:
1. Check whether any attacker or blocker has first strike or double strike.
2. If yes:
   - First-strike pass: only first strike + double strike creatures deal damage.
   - Run SBAs immediately after this pass.
3. Normal pass:
   - If first/double strike exists: only non-first-strike creatures + double strike creatures deal damage.
   - If no first/double strike exists: all combatants deal damage (current one-pass behavior).
4. Continue step advancement normally (still one `CombatDamage` step externally).

Double strike creatures deal damage in both sub-steps. First strike creatures
deal damage only in the first sub-step. Normal creatures deal damage only in
the second sub-step.

SBAs between sub-steps mean a 2/1 first striker can kill a 3/2 before it deals
damage. This is where combat gets strategically rich.

**Implementation in the game loop**: The `CombatDamage` match arm still calls
`resolve_combat_damage()` exactly once and returns `None` (no action space needed).
The two internal passes (first-strike pass + SBA + normal pass) happen inside
that method. No priority passes between sub-steps (simplification — CR
technically grants priority between sub-steps, but we skip that for now).

### Card pool expansion

13 new cards total:
- 11 keyword demonstrators (one per keyword minimum)
- 1 vanilla green rate baseline (Craw Wurm)
- 1 non-vanilla finisher with activated ability (Shivan Dragon)

| Card | Cost | P/T | Keywords | First set | Design role |
|------|------|-----|----------|-----------|-------------|
| Wind Drake | 2U | 2/2 | flying | Portal | Evasion baseline |
| Giant Spider | 3G | 2/4 | reach | Limited Edition Alpha | Anti-air |
| Raging Goblin | R | 1/1 | haste | Portal | Tempo, immediate pressure |
| Serra Angel | 3WW | 4/4 | flying, vigilance | Limited Edition Alpha | Multi-keyword, attack + defend |
| Typhoid Rats | B | 1/1 | deathtouch | Innistrad | Trades up, chump-kills |
| War Mammoth | 3G | 3/3 | trample | Limited Edition Alpha | Excess damage to player |
| Wall of Stone | 1RR | 0/8 | defender | Limited Edition Alpha | Pure blocker |
| Boggart Brute | 2R | 3/2 | menace | Magic Origins | Forces multi-block or evasion |
| Youthful Knight | 1W | 2/1 | first strike | Stronghold | Wins combat against equal P/T |
| Fencing Ace | 1W | 1/1 | double strike | Return to Ravnica | Effectively 2 power, survives first |
| Healer's Hawk | W | 1/1 | flying, lifelink | Guilds of Ravnica | Racing, life swing |
| Craw Wurm | 4GG | 6/4 | — | Limited Edition Alpha | Big vanilla ground threat baseline |
| Shivan Dragon | 4RR | 5/5 | flying | Limited Edition Alpha | Flying finisher + repeatable firebreathing |

Deck helpers added for each (e.g., `hawk_deck()`, `spider_deck()`,
`shivan_deck()`, `craw_wurm_deck()`).

### Shivan Dragon firebreathing (CR-correct, requires new scaffolding)

Shivan Dragon's activated ability:
`{R}: Shivan Dragon gets +1/+0 until end of turn.`

To implement this correctly (CR 602, 405, 611), this batch adds a minimal
activated-ability system:

1. **Activated ability definitions on cards**
   - Add `activated_abilities: Vec<ActivatedAbilityDefinition>` to
     `CardDefinition`.
   - For this batch, support one effect kind:
     `SelfGetsUntilEot { power_delta: i32, toughness_delta: i32 }`.

2. **Priority action for ability activation**
   - Add action type: `PriorityActivateAbility`.
   - Generated during priority when player controls an eligible permanent and can
     pay the ability mana cost.
   - Action focus includes the source permanent id.

3. **Unified stack objects (spells + abilities)**
   - Introduce `StackObject` enum in game state:
     - `Spell(CardId)` (existing behavior)
     - `ActivatedAbility(ActivatedAbilityOnStack)`
   - Track canonical stack metadata on every object:
     - controller
     - source card registry key
     - source permanent id (for abilities)
     - chosen targets (possibly empty)
     - ability id/index for activated abilities
   - `resolve_top_of_stack()` pops one `StackObject` in strict LIFO order.
   - Counterspell targeting remains spell-only (`target spell`), so ability stack
     objects are not legal Counterspell targets.

4. **Resolution semantics**
   - On activation: pay `{R}`, put ability on stack, pass priority as normal.
   - On resolution: if source permanent still exists on battlefield as the same
     object, apply `+1/+0 until end of turn`; otherwise ability resolves with no
     effect.
   - Multiple activations stack and resolve independently (LIFO), effects add.

5. **Temporary stat tracking**
   - Add temporary P/T modifiers on `Permanent` (e.g., `temp_power`, `temp_toughness`).
   - Combat damage and lethal checks read effective stats.
   - Clear temporary modifiers in cleanup with damage removal.

### Observation space

Add `KeywordData` to `CardData`:

```rust
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeywordData {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
}
```

Added to `CardData` (which covers cards in all zones). The agent sees keywords
on hand cards, battlefield permanents' cards, graveyard cards, and stack spells.
11 new boolean features per card in the observation vector.

Add a generic stack lane in observation:

```rust
pub enum StackObjectKindData {
    Spell,
    ActivatedAbility,
}

pub struct StackObjectData {
    pub stack_object_id: i32,
    pub kind: StackObjectKindData,
    pub controller_id: i32,
    pub source_card_registry_key: i32,
    pub source_permanent_id: Option<i32>,
    pub ability_index: Option<i32>,
    pub targets: Vec<StackTargetData>,
}

pub enum StackTargetKindData {
    Player,
    Permanent,
    StackObject,
}

pub struct StackTargetData {
    pub kind: StackTargetKindData,
    pub player_id: Option<i32>,
    pub permanent_id: Option<i32>,
    pub stack_object_id: Option<i32>,
}
```

`Observation` gains `stack_objects: Vec<StackObjectData>` ordered top-of-stack
first. This is the canonical way to observe activated abilities on stack.

**Encoding contract detail:** keyword bits are appended in a fixed order
(`flying, reach, haste, vigilance, trample, first_strike, double_strike,
deathtouch, lifelink, defender, menace`) immediately before the card validity
bit in Python encoding. `ObservationEncoder.card_dim` must increase by +11 in
lockstep with Rust `CardData` / PyO3 bindings to avoid silent feature skew.

**Checkpoint compatibility note:** this changes observation shape, so old model
checkpoints are not shape-compatible. Treat as a deliberate breaking change for
training artifacts.

**Current observability decision:** do not duplicate keyword flags onto
`PermanentData` yet and do not add a permanent→card link in this stage. The
existing model can still observe keyword-bearing battlefield cards through
`CardData` entries.

**Duplication decision (accepted):**
- Keep `CardData` entries for cards in `ZoneType::Stack` (spells on stack).
- Also expose the new generic `stack_objects` lane (spells + activated abilities).
- Keep battlefield dual-view (`PermanentData` + battlefield `CardData`) for now.
  This intentionally tolerates overlap and keeps a clean migration path for
  future token support, where permanents are authoritative runtime objects.

**Stack metadata decision (accepted):**
- Include targets and ability metadata in `stack_objects` now (not just kind/source).
- For Shivan specifically, `targets` is empty and `ability_index` identifies
  firebreathing.
- Use stable `stack_object_id` values for object identity/references (including
  stack targets), rather than positional indices.
- This is future-ready for targeted activated abilities/spells without redesigning
  the observation contract again.

**Known limitation (accepted for this stage):** the policy has a projection
layer over card feature vectors ("card embedding"), but no learned per-card
identity token and no explicit permanent↔card binding feature. It can learn from
observable stats/types/keywords, but two cards with identical visible features
are intentionally indistinguishable.

### Test plan

CR-cited trace tests per keyword, following the established pattern in
`tests/rules/`:

**New test files:**
- `cr_702_keywords.rs` — all keyword tests in one file, grouped by CR section
- `cr_602_activated_abilities.rs` — activated-ability stack/cost/duration tests

**Tests per keyword (33 total minimum):**

| Keyword | Positive test | Negative test | Interaction test |
|---------|--------------|---------------|------------------|
| Flying | Flyer can't be blocked by ground | Non-flyer can be blocked | Flyer blocked by reach |
| Reach | Reach creature blocks flyer | Reach doesn't grant evasion | — |
| Haste | Haste creature attacks turn 1 | Non-haste can't attack turn 1 | — |
| Vigilance | Vigilant attacker stays untapped | Non-vigilant taps on attack | — |
| Defender | Defender can't attack | Non-defender can attack | Defender can block |
| Trample | Excess damage to player | No trample: excess wasted | Trample + deathtouch: 1 lethal |
| Deathtouch | 1 damage kills any toughness | Normal: need full toughness | — |
| Lifelink | Damage heals controller | No lifelink: no heal | — |
| Menace | Solo blocker removed | 2 blockers valid | — |
| First strike | First striker kills before normal damage | Normal creatures trade simultaneously | — |
| Double strike | Deals damage in both sub-steps | — | Double strike + trample |

**Activated ability tests (minimum 6):**
- Can activate Shivan ability at priority when controller can pay `{R}`
- Cannot activate without available red mana
- Activation uses stack and resolves only after passes
- Multiple activations resolve LIFO and stack (+N/+0 total)
- Buff expires in cleanup/end turn
- If Shivan leaves battlefield before resolution, no buff is applied
- Observation exposes activated ability object in `stack_objects` while pending
- Stack observation includes target + metadata fields with stable semantics

### Combat damage assignment fix

The current code has a bug: a blocked attacker deals its full power to EACH
blocker independently. A 2/2 blocked by two 1/1s deals 2 to each (should be 2
total split). This fix is prerequisite for trample and is the right thing
regardless.

New logic in `resolve_combat_damage()`:
```
for each attacker:
  if unblocked: deal power to defending player
  if blocked:
    remaining_damage = attacker_power
    for each blocker (in order):
      lethal = blocker_toughness - blocker_damage_marked
      if deathtouch: lethal = min(1, lethal)
      assigned = min(remaining_damage, lethal)
      deal assigned to blocker
      remaining_damage -= assigned
    if trample && remaining_damage > 0:
      deal remaining_damage to defending player
```

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Keywords as enum set (`HashSet<Keyword>`) | More extensible, but allocation per card | Bool flags are cache-friendly, fixed-size, trivially observable. We know exactly which 11 keywords we need. |
| Keywords on Permanent (mutable) | Enables grant/remove effects | No effects system yet. Cards are the source of truth. Add mutable keywords later when needed. |
| New StepKind for FirstStrikeDamage | Cleaner CR mapping | Breaks observation contract enum stability. Internal sub-step is invisible to the agent (correctly — the agent doesn't act between sub-steps). |
| Priority between first/double strike sub-steps | Full CR compliance (CR 510.4) | Adds action space complexity with minimal strategic value at this stage. The agent doesn't have instants that care about between-damage-steps timing. Add later if needed. |
| Menace via action space constraints | Agent can't make illegal blocks | Would require knowing which attackers have menace during per-blocker action generation, and tracking how many blockers each attacker already has. Post-hoc validation is simpler and teaches the agent the same lesson. |

## Key decisions

1. **Keywords on CardDefinition, not Permanent.** Static flags. No runtime
   grant/remove until an effects system exists. This is correct for vanilla
   keyword creatures.

2. **First/double strike as internal two-pass combat resolution.** No new
   StepKind enum variants, no `CombatState` phase flag, no priority between
   sub-steps. Keeps the observation contract stable and the action space unchanged.

3. **Fix multi-blocker damage assignment.** The current "full power to each
   blocker" is wrong. Fix it as part of trample implementation. Ordered damage
   assignment is the correct baseline regardless.

4. **Menace via post-declaration validation.** Remove illegal solo blocks after
   all blockers are declared. The agent learns from the consequence.

5. **Deathtouch tracking via `deathtouch_damage` flag on Permanent.** Minimal
   addition. Cleared with damage in cleanup. Checked in SBAs.

6. **13 new cards with one deliberate non-vanilla exception.** 12 are vanilla
   creatures (11 keyword demos + Craw Wurm), plus Shivan Dragon to force a
   CR-correct activated-ability path.

7. **CR-correct firebreathing over shortcut implementation.** Shivan activation
   is a real activated ability on the stack (not immediate stat mutation).

## Scope

**Packaging decision:** **One more big push** (selected in design review).

- **In scope:**
  - `Keywords` struct on `CardDefinition`
  - All 11 keyword implementations (flag checks, combat modifiers, SBA changes)
  - Fix multi-blocker combat damage assignment
  - First/double strike internal sub-steps (no priority between)
  - 13 new cards (11 keyword demos + Craw Wurm + Shivan Dragon)
  - Minimal activated-ability scaffolding needed for CR-correct Shivan firebreathing
  - `KeywordData` in observation space
  - Generic `stack_objects` observation lane for spells + activated abilities
  - CR-cited trace tests for all keywords
  - CR-cited activated-ability tests for Shivan firebreathing
  - Deck helpers for new cards

- **Out of scope:**
  - Granting/removing keywords dynamically (needs effect system)
  - Priority between first/double strike sub-steps
  - Blocker damage assignment order as agent choice (auto-ordered for now)
  - Training smoke test (separate concern, done after merge)
  - Protection, hexproof, indestructible, or other non-evergreen keywords
  - General activated-ability framework beyond what Shivan requires (targets, tap-cost abilities, loyalty abilities)
  - Battlefield identity/model cleanup (e.g., permanent→card linking refactor for future token-first modeling)

## Done when

```bash
cd managym && cargo test
```

- All 11 keywords functional with CR-cited trace tests
- Multi-blocker damage assignment fixed and tested
- First/double strike sub-steps work (first striker kills before normal damage)
- Deathtouch + trample interaction tested (1 damage = lethal for assignment)
- Shivan Dragon firebreathing is CR-correct (cost, stack, resolution, EOT duration)
- Card pool expanded by 13 cards (including Shivan Dragon and Craw Wurm)
- Observation space includes keyword flags
- Observation includes generic stack objects and shows pending Shivan activations
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- `cargo fmt --check` passes
