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

## Stage 2 RESULT (landed 2026-07-09)

**What landed** (three commits: decision/cost engine, cards + trace tests,
Python observation mirror):

- **Choice framework** (`flow/decision.rs`): resolutions execute through an
  `EffectFrame` (source, controller, targets + per-target requirement
  indices, kicked flag, trigger-context target, effect queue, finalizer).
  Any effect may return a `Decision`, which parks the frame as
  `GameState::suspended_decision` and surfaces an ActionSpace to the
  deciding player; the answer feeds back through `execute_decision_action`
  and the frame resumes (branch effects push onto the queue front, so
  nested decisions compose). While suspended, nothing else runs — no SBAs,
  no trigger flushing, no priority. Decision kinds: `Scry` (per-card
  keep/bottom, top-down; kept cards keep relative order — no reorder),
  `LookAndSelect` (top-N, select up to K matching a `CardPredicate` to
  hand, `min_select` for mandatory picks, rest to bottom in random order),
  `PayOrNot` (if-paid / if-declined effect branches; pay option only
  offered when `available_mana` covers it), `Modal`, `DiscardThenDraw`
  (learn). New `ActionSpaceKind`s: `Scry`, `LookAndSelect`, `PayOrNot`,
  `Modal`, `DiscardThenDraw`, `Waterbend`; new `Action`s / `ActionType`s:
  `ScryCard` (ScryKeep/ScryBottom), `SelectCard`, `Decline`, `PayCost`,
  `ChooseMode`, `WaterbendTap` (TapForCost). ACTION_TYPE_DIM 7→14.
- **Spells stay on the stack while resolving** (CR 608.2m):
  `resolve_top_of_stack` peeks instead of popping for spells; the
  finalizer removes the stack object and moves the card as resolution's
  last step. This keeps a suspended stack consistent and makes
  "Lessons in your graveyard" exclude the resolving Lesson (Accumulate
  Wisdom does not count itself).
- **Casting pipeline** (`flow/play.rs`): `PendingChoice` grew from a single
  ChooseTarget into a staged pipeline — `KickerChoice` (PayOrNot space,
  only offered when the kicked cost is affordable) → `ChooseTargets` per
  `TargetRequirement {spec, min, max}` ("up to N" adds a Decline action
  once min is met; duplicate targets excluded within a requirement) →
  payment. `SpellOnStack` carries `kicked` and `target_req_indices`;
  CR 608.2b fizzle = all chosen targets illegal for their requirement.
  `TargetSpec` grew `SpellOrPermanent{min_mana_value}` (Divide by Zero),
  `CreatureYouControl`, `CreatureOpponentControls` — target legality is
  controller-relative (`target_is_legal(target, spec, controller)`).
- **Cost mechanics**: affinity-style reduction via
  `CardDefinition::cost_reduction_per` (generic reduced by matching
  battlefield permanents, floor 0, computed in `effective_spell_cost` at
  gating and payment). Waterbend on activated abilities
  (`ActivatedAbilityDefinition::waterbend`): activation opens a
  `Waterbend` payment space — tap any untapped artifact/creature for {1}
  of the generic component (taps emit `PermanentTapped{for_mana: true}`),
  or `PayCost` to settle the remainder with mana; tap actions are gated so
  affordability is preserved (never a dead-end space). **Triggered mana
  abilities** (CR 605.1b, `CardDefinition::triggered_mana_abilities`) fire
  inside `tap_permanent(for_mana=true)` and add to the pool immediately —
  Badgermole Cub's {G} arrives mid-payment and pays the same waterbend
  remainder (trace-tested: 2 taps + Cub → only 1 of 3 forests needed).
- **Ward as a keyword**: `CardDefinition::ward` synthesizes a
  `BecomesTargeted` triggered ability at registration. Casting emits
  `PermanentTargeted` per permanent target; `PendingTrigger` /
  `TriggeredAbilityOnStack` carry the triggering spell as a **context
  target**, and `Effect::CounterUnlessPays` (shared with It'll Quench Ya!)
  reads the frame's primary target (chosen target or context) and asks the
  spell's controller to pay or watch it get countered.
- **Cards registered**: tla.rs — Glider Kids, Firebending Lesson (kicker),
  It'll Quench Ya!, Accumulate Wisdom, Water Tribe Rallier (waterbend
  {5}), Allies at Last (affinity + up-to-2 + 1 multi-target), Badgermole
  Cub, Crossroads of Destiny (invented modal proof card — no M1 deck card
  is modal). strixhaven.rs (new) — Pop Quiz, Igneous Inspiration, Divide
  by Zero (spell-or-permanent bounce; bounced stack spells cease and the
  card returns to its owner's hand), Waterfall Aerialist (ward {2} proof;
  Dragonfly Swarm waits on Stage 3 dynamic P/T).
- **Observation encoding**: library cards revealed by a pending decision
  are added to the *deciding* agent's observation (zone Library) and
  focused by the decision's actions; `Game::determinize` pins them back on
  top of the library (they're known information — flat MC at a scry/look
  decision stays coherent). CardData gained ward/kicker costs; CARD_DIM
  33→37; mirrored in the Rust encoder, pyo3 bindings, `__init__.pyi`,
  JSON, and `manabot/env/observation.py` (+ a new ActionSpaceEnum mirror).
- **Validation**: cargo 164 green (135 preserved + 29 new: 27 per-card /
  machinery trace tests, a 200-seed random-vs-random Stage-2 deck smoke
  with non-empty-action-space asserts, and a flat-MC-at-decision-points
  test); clippy clean; pytest 209 green after cp312 rebuild; 200-seed
  Python-path smoke (UR-decisions vs GW-costs decks, observations encoded
  every step, every new ActionSpace kind exercised, `flat_mc_scores` run
  at live decision points, zero panics / zero stuck games).

**Design decisions Stage 3 must know**

- `EffectFrame.context_target` is how trigger-event payloads reach
  effects. Earthbend's "when this land dies/exiles, return it tapped" and
  Earth Kingdom Jailer's exile-linkage can ride the same channel; extend
  `trigger_context_for_event` for new event kinds.
- Ward triggers on **spells only** — abilities that target (there are
  none in M1 beyond Deserters' friendly counter) don't emit
  `PermanentTargeted`, and countering an *ability* stack object has no
  Target representation yet.
- `Modal` modes must be **targetless** (modes execute mid-resolution;
  mode-specific targeting would need cast-time mode selection, CR 601.2b).
  Fine for M1; revisit if a modal card with targeted modes lands.
- Waterbend is implemented for **activated abilities only** (no M1 spell
  has it) and allows tapping summoning-sick untapped creatures (convoke-
  style; taps aren't {T} activations). Affordability gating is
  conservative: it ignores triggered-mana bonuses (Cub can only make
  things cheaper, never gate out a legal line... but a line affordable
  *only* through Cub mana won't be offered).
- Scry keeps cards in relative order (no reorder sub-decision) and
  scry/look bottom moves are silent library reorders (no CardMoved
  events); look-and-select to hand uses `move_card` (public CardMoved).
- `PutCounters` still targets creatures only; earthbend needs a
  land-capable TargetSpec — `CreatureYouControl`-style controller-relative
  specs show the pattern.
- Multi-target effects read `EffectFrame.targets` directly
  (`TargetCreaturesDealPowerDamageToLastTarget` interprets last = victim);
  per-target requirement indices are on the stack object if a future
  effect needs requirement-aware dispatch.
- Encoded action tensors cap at `max_actions = 20`; learn with a big hand
  or wide waterbend boards can exceed it (raw action path unaffected,
  encoder truncates with a warning — pre-existing behavior, now easier to
  hit). Revisit capacity when Stage 4 wires decks into training.
- Priority-gating for castability still uses `producible_mana` (untapped
  permanents only); payment and decision gating use `available_mana`
  (pool + producible). Until-end-of-combat mana (Stage 3 firebending)
  should extend the pool/`clear_mana_pools` — decisions and waterbend
  already read the pool correctly.

## Notes

- Cub + Rallier interaction is a real test case: waterbend taps creatures to
  pay; Cub's trigger adds {G} per creature tapped for mana — the trigger and
  the payment mechanic must compose.
- Milestone 2 (TLA commons complete, `00-pool-audit.md`) reuses every rung
  here; nothing in this slice is throwaway.
