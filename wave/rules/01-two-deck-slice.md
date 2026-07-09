# 01: The Two-Deck Slice (Milestone 1)

Two 40-card decks from the elemental cube, playable against each other in the
engine and behind the play UI. Lists blessed 2026-07-09 ("don't care on the
specific 40 too much — I'll be impressed if we can get this set working"), so
counts may flex ±1 during implementation; the CAPABILITY BILL is the contract.

## UR Lessons (17 lands: 9 Island, 8 Mountain)

2 Tiger-Seal · 2 Otter-Penguin · 2 Fire Nation Cadets · 2 First-Time Flyer ·
1 Forecasting Fortune Teller · 1 Dragonfly Swarm · 4 Firebending Lesson ·
2 Igneous Inspiration · 2 Pop Quiz · 2 Divide by Zero · 2 It'll Quench Ya! ·
2 Accumulate Wisdom

Learn note: no sideboard in 1v1 constructed → learn = "you may discard, then
draw" (rules-correct without a wishboard).

## GW Allies (17 lands: 9 Plains, 8 Forest)

2 Water Tribe Rallier · 2 Invasion Reinforcements · 2 Compassionate Healer ·
2 Earth Kingdom Jailer · 2 White Lotus Reinforcements · 2 Earth King's
Lieutenant · 2 Kyoshi Warriors · 2 Badgermole Cub · 1 Suki, Kyoshi Warrior ·
1 South Pole Voyager · 2 Allies at Last · 1 Yip Yip! · 2 Fancy Footwork

(Badgermole Cub added by owner call — earthbend is IN Milestone 1.)

## The capability bill (the contract)

Staged for dispatch; each stage lands with cargo trace-tests per capability
and observation-encoding for anything an agent must be able to see.

**Stage 1 — trigger substrate & board state** (dispatched 2026-07-09):
triggered-ability framework generalized over sources (ETB, death, attack,
upkeep, becomes-tapped, taps-for-mana, another-[type]-enters,
Nth-card-drawn-this-turn, second-time-this-turn gating); **+1/+1 counters**
(state, P/T contribution, on lands too — earthbend prereq); **tokens**
(creature tokens incl. entering tapped-and-attacking; Clue artifact token
with sac-to-draw); **type predicates** (Ally/Lesson checks, graveyard type
counting); keywords: flash, vigilance-correctness, conditional unblockable.
Observation encoding: counter counts, token flag, type tags, gy-type counts.

**Stage 2 — decisions & costs:** mid-resolution player choices (scry,
look-N-select, pay-or-not [kicker / ward / It'll Quench Ya!], modal);
waterbend (tap-permanents-to-help-pay); affinity-style cost reduction;
multi-target spells; spell-bounce (Divide by Zero).

**Stage 3 — specials:** earthbend (land animation + counters + dies/exiled →
return tapped); exile-until-this-leaves linkage (Earth Kingdom Jailer);
static continuous effects (anthem — White Lotus Reinforcements); dynamic P/T
(Suki, Dragonfly Swarm); learn (discard-then-draw mode); until-end-of-combat
mana (firebending); death triggers with gy conditions.

**Stage 4 — registration & validation:** all 25 distinct nonland cards
registered in a `tla` cardset module; per-card trace tests; ≥200
random-vs-random A-vs-B games to terminal with zero errors; decks exposed to
the play UI and the training/eval stack (deck constants next to
INTERACTIVE_DECK); first A-vs-B matchup table (seat-balanced, per-deck).

## Stage 1 RESULT (landed 2026-07-09)

**What landed** (three commits: engine substrate, trace tests, observation
encoding):

- **Trigger framework** (`state/ability.rs`, `flow/triggers.rs`):
  `TriggerCondition` now spans `EntersTheBattlefield`, `Dies`, `Attacks`,
  `BecomesTapped`, `TappedForMana`, `BeginningOfYourUpkeep`, and
  `YouDrawNthCardThisTurn { n }`, each event-based condition parameterized by
  a `TriggerSubject` (`This` / `AnotherYouControl(pred)` /
  `AnyYouControl(pred)`). Events (`StepStarted`, `CardDrawn`,
  `PermanentTapped { for_mana }`, `AttackersDeclared`) are pumped through
  `process_game_events` at the top of the priority loop, so triggers work for
  state changes that happen outside card movement. Attack triggers fire once
  per declared batch (CR 508.1); "one or more you control attack" fires once
  per combat. Triggered abilities carry `Vec<Effect>` (Otter-Penguin does two
  things); targetless triggers go straight onto the stack — the old code
  silently dropped them. Turn-scoped counters (`cards_drawn_this_turn`,
  `ability_resolutions_this_turn`) live on `TurnState` and reset with the
  turn; `Effect::OnNthResolutionThisTurn` implements South Pole Voyager's
  "second time this ability has resolved this turn" gate.
- **+1/+1 counters**: `Permanent::plus1_counters` (any permanent, lands
  included), feeding `effective_power/toughness` and therefore the
  lethal-damage SBA and power-based predicates.
- **Tokens**: registry-defined cards with `is_token`; `Game::create_token`
  supports entering tapped-and-attacking (joins `combat.attackers`, no attack
  trigger per CR 508.4a). SBA removes tokens from non-battlefield zones after
  death triggers enqueue (CR 704.5d). `Ally` (1/1 white, `color_override`)
  and `Clue` ({2}, sac: draw — `sacrifice_source` activation cost) are
  registered.
- **Type predicates**: `state/predicate.rs::CardPredicate` (card type,
  subtype, max power) shared by trigger subjects, block restrictions, and
  `count_graveyard_matching` (gy Lesson counting). Lesson is just a subtype
  string on instants/sorceries.
- **Keywords**: `flash` (via `Card::is_instant_speed`; `CardTypes::
  is_instant_speed` removed), conditional unblockability as
  `Card::block_restriction: Option<CardPredicate>` checked against the
  blocker's *effective* power, plus a `cant_be_blocked_this_turn` permanent
  flag cleared at cleanup. Vigilance attackers don't emit `PermanentTapped`
  (verified); haste attackers fire attack triggers on their entry turn
  (verified).
- **Observation encoding**: CARD_DIM 29→33 (flash, is_token, is_ally,
  is_lesson), PERMANENT_DIM 5→7 (plus1_counters, cant_be_blocked_this_turn),
  PLAYER_DIM 26→27 (graveyard_lessons), mirrored in the Rust encoder,
  pyo3 bindings, `__init__.pyi`, JSON, and `manabot/env/observation.py`.
- **Proof cards** (`cardsets/tla.rs`): Kyoshi Warriors, Avatar Enthusiasts,
  Invasion Reinforcements, Jeong Jeong's Deserters, Tiger-Seal,
  Otter-Penguin, Forecasting Fortune Teller, plus South Pole Voyager (bonus —
  fully expressible and exercises Nth-resolution gating + `GainLife`).
- **Validation**: cargo 135 tests green (107 pre-existing, 28 new: per-card
  traces + substrate tests incl. `tla_cards.rs`), pytest 209 green after
  cp312 rebuild, and a 200-seed random-vs-random smoke on a proof-card deck
  (zero panics, zero stuck games, terminal winner every seed).

**Design decisions Stage 2 must know**

- Triggered abilities support **at most one targeted effect** per ability
  (`Ability::target_spec` takes the first). Multi-target spells (Fancy
  Footwork, Allies at Last) need `TargetSpec` growth, not ability-shape
  growth.
- `EffectContext { source, controller, resolutions_this_turn }` threads
  through effect execution; "you" in effect text = `ctx.controller` (spells
  now use the caster, not the card owner — same thing today, but
  future-proof).
- `intervening_if` was **removed** (was scaffolding, nothing used it).
  Dragonfly Swarm's "if there's a Lesson card in your graveyard" (Stage 3)
  should come back as a proper *state predicate* type checked at trigger and
  resolution time (CR 603.4), not as a `TriggerCondition`.
- `TappedForMana` exists as a stack-trigger condition. Badgermole Cub's
  "whenever you tap a creature for mana, add {G}" is a **triggered mana
  ability** (CR 605.1b — no stack); Stage 2/3's waterbend work should route
  it through mana production directly rather than this condition.
- Trigger-frequency gating at *enqueue* time ("this ability triggers only
  once each turn") is not implemented — no Milestone-1 card needs it;
  resolution counting covers the Voyager pattern.
- `PutCounters` targets creatures only (`TargetSpec::Creature`); earthbend
  (Stage 3) targets lands, which needs a land-capable `TargetSpec` — the
  counter *state* already works on lands.
- Tokens are real entries in `GameState::cards` (cards vec grows mid-game;
  `zone_of == None` after they cease). Anything iterating `state.cards` must
  tolerate zoneless cards.
- `block_restriction` is not yet agent-visible (no encoding); the
  `cant_be_blocked_this_turn` flag is. Revisit if a deck card gains a static
  block restriction (none in Milestone 1; Foggy Swamp Vinebender would be the
  TLA-commons test card).
- No Lesson-typed card is registered yet (all four Lesson spells need Stage-2
  machinery); the gy-Lesson observation path is trace-tested via an injected
  subtype.

## Notes

- Cub + Rallier interaction is a real test case: waterbend taps creatures to
  pay; Cub's trigger adds {G} per creature tapped for mana — the trigger and
  the payment mechanic must compose.
- Milestone 2 (TLA commons complete, `00-pool-audit.md`) reuses every rung
  here; nothing in this slice is throwaway.
