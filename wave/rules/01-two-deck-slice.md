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

## Stage 3 RESULT (landed 2026-07-09)

**What landed** (three commits: engine substrate, cards + trace tests,
observation mirror + python smoke): every card of both decklists is now
registered and trace-tested — the two decks are fully dealable.

- **Continuous-effect layers** (`flow/statics.rs`):
  `Game::effective_power/toughness/effective_pt(permanent_id)` replaces the
  old permanent-local P/T everywhere (SBA, combat, predicates,
  observations). Layer order, a deliberate simplification of CR 613:
  **base P/T (printed or characteristic-defining, layer 7a) → static
  continuous P/T effects (layer 7c: anthems + conditional self-buffs) →
  until-end-of-turn deltas (also 7c) → +1/+1 counters (7d)**. Every
  Milestone-1 effect is additive, so within-layer timestamp ordering is
  not modeled. Anthem scopes (`StaticScope::OtherYouControl(pred)`)
  match printed characteristics plus animation but never power, so the
  computation cannot recurse. CDAs (`CardDefinition::power_cda`:
  `CreaturesYouControl` for Suki, `GraveyardMatching` for Dragonfly
  Swarm) are recomputed on every read; conditional statics
  (`StaticCondition::GraveyardAtLeast`, shared with trigger gating) are
  likewise pure queries — First-Time Flyer flips between 1/2 and 2/3 as
  Lessons enter/leave the graveyard with zero stored state.
- **Earthbend** (`Effect::Earthbend`, `TargetSpec::LandYouControl`):
  animation is permanent state (`Permanent::animated`) meaning "also a
  0/0 creature with haste, still a land" — the only animation shape in
  M1, with no duration (it lasts as long as the permanent). Base 0/0
  falls out of lands printing no P/T. All creature checks are
  animation-aware (attack/block/tap, lethal SBA, Creature target specs,
  predicates, MassDamage); a new CR 704.5f SBA kills 0-toughness
  creatures, so an earthbent land stripped of counters dies. The
  return is a one-shot **delayed trigger** (`GameState::delayed_triggers`
  watching the card's next battlefield exit): dies/exiled → a
  `PendingTrigger` with **inline effects** (`inline_effects` channel on
  PendingTrigger/TriggeredAbilityOnStack — a real stack object with no
  `Card::abilities` entry) resolving
  `ReturnSourceToBattlefieldTapped`; any other exit drops the watcher.
  Returns enter tapped without a "becomes tapped" event, as a fresh
  object: not animated, no counters, granted trigger gone.
- **Exile-until-this-leaves** (Jailer): `Effect::ExileUntilSourceLeaves`
  + `GameState::exile_links`. CR 603.6e pragmatics: at trigger
  resolution the exile is skipped unless the **source card is still on
  the battlefield** (Banisher-Priest ruling; object-identity check
  deliberately card-level — a same-card new-object race is unreachable
  in M1). When the Jailer leaves by any route, linked cards still in
  exile return immediately under their owners — **no trigger, no
  stack** (CR 610.3-style duration end); cards that left exile earlier
  are simply unlinked. "Up to one target" is a first: triggered-ability
  target choices can now carry a Decline action
  (`Effect::target_optional`), and an optional-target trigger with zero
  legal targets still resolves (as a no-op) instead of being removed.
  `TargetSpec::PermanentOpponentControls{predicate}` rides the
  CardPredicate growth (`card_types_any`, `not_card_types`,
  `min_mana_value`).
- **Until-end-of-combat mana** (firebending): a second pool
  (`Player::combat_mana_pool`) that survives step boundaries and empties
  at end of combat (+ cleanup safety net). Payment spends the combat
  pool first (it expires sooner); `available_mana` and castability
  gating now include pooled mana — previously priority gating used
  producible-only, which would have made firebending mana uncastable.
  Fire Nation Cadets' "has firebending 2 as long as a Lesson is in your
  graveyard" is a **conditionally-granted, on-stack attack trigger**:
  `TriggerCondition::ActiveIf{active_if, condition}` checks the
  condition at fire time (granted ability doesn't exist without it);
  once on the stack it resolves regardless (granted-trigger semantics).
  Dragonfly Swarm's death trigger composes ActiveIf (fire-time) with an
  `IfGraveyardAtLeast` branch (resolution-time) = full intervening-if
  (CR 603.4).
- **Keyword grants + hexproof**: `Permanent::temp_keywords` (until-EOT,
  cleared at cleanup) unioned into `effective_keywords` used by combat
  (flying/reach blocks, first/double strike passes, trample, vigilance)
  and damage (lifelink/deathtouch via the source's permanent when on
  the battlefield). `Keywords::hexproof` blocks opponent targeting in
  `target_is_legal`. New effects: `BuffTarget`, `UntapTarget`,
  `GrantKeywordsToTarget`, `IfTargetMatches` (Yip Yip's Ally rider),
  `ForEachTarget` (per-target sub-effects with per-target legality — a
  Fancy Footwork target that died is skipped individually, CR 608.2b),
  `PutCountersOnEachMatching`, `AddMana`.
- **Cards**: new — Fire Nation Cadets, First-Time Flyer, Dragonfly
  Swarm, Compassionate Healer, Earth Kingdom Jailer, White Lotus
  Reinforcements, Earth King's Lieutenant, Suki Kyoshi Warrior, Yip
  Yip!, Fancy Footwork, Enter the Avatar State; Badgermole Cub gained
  its earthbend-1 ETB (and 2/2 body). Oracle corrections from Scryfall:
  Firebending Lesson is {R} Instant–Lesson kicker {4} targeting a
  creature (was {1}{R} Sorcery kicker {3} any target), It'll Quench Ya!
  is a Lesson, Water Tribe Rallier is {1}{W} 2/2 Soldier, Allies at
  Last is {2}{G} Instant.
- **Observation encoding**: PERMANENT_DIM 7→11 (effective power,
  effective toughness, is_animated, has_exile_link), PLAYER_DIM 27→28
  (combat_mana). Conditional statics/anthems/CDAs are agent-visible
  through effective P/T; exiled cards were already visible in the Exile
  zone and the holding permanent is now flagged. Mirrored in the Rust
  encoder, pyo3 bindings, `__init__.pyi`, JSON, and
  `manabot/env/observation.py`.
- **Deck constants**: `UR_LESSONS_DECK` / `GW_ALLIES_DECK` in
  `manabot/verify/util.py` next to INTERACTIVE_DECK (Stage 4's first
  bullet, pulled forward).
- **Validation**: cargo 194 green (161 rules incl. 28 Stage-3 traces, 13
  engine incl. a 200-seed random-vs-random smoke on the ACTUAL
  decklists, 10 search incl. flat-MC at decision points in the real
  matchup, 10 unit); clippy clean; pytest 209 green after cp312
  rebuild; 200-seed full-stack Python smoke on the real matchup
  (observations encoded every step, zero panics / zero stuck games;
  Scry, LookAndSelect, PayOrNot, DiscardThenDraw, Waterbend and
  ChooseTarget all exercised) with `flat_mc_scores` run at 10 live
  decision points.

**Deviations from CR / oracle (documented on the cards)**

- Suki's hybrid {G/W} pips are registered as {G}{W} (no hybrid-mana
  support; the GW deck runs both colors, so only corner-case payments
  differ). The legend rule is not enforced (single copy in the deck).
- Enter the Avatar State grants its four keywords but does not add the
  Avatar type ("becomes an Avatar in addition to its other types" has
  no mechanical relevance in M1 — no until-EOT type-addition machinery
  was built).
- Dragonfly Swarm's ward {1} triggers on **spells only** (Stage-2 ward
  limitation; oracle says "spell or ability" — no M1 ability targets an
  opponent's permanent).
- CDAs are computed for battlefield permanents; CDA cards in other zones
  observe as printed (CR 604.3 says CDAs apply everywhere, but nothing
  in M1 reads P/T off the battlefield).
- Firebending is an on-stack triggered ability (the TLA reminder-text
  reading), not a mana ability.
- Ancestral Recall: oracle is "Target player draws three cards." — the
  engine has no target-player draw, so the caster always draws (the
  self-target case; the opponent-draw line is unavailable). Added by the
  card-conformance audit, 2026-07-09.
- Learn has no sideboard mode (1v1 constructed): "you may discard a
  card; if you do, draw a card" only — the reveal-a-Lesson-from-outside-
  the-game option is not offered (also noted at the top of this doc).
- Accumulate Wisdom / Water Tribe Rallier bottom the unselected cards in
  a **random** order (Rallier's oracle says random; Accumulate Wisdom's
  says "any order" — no reorder sub-decision, see Stage-2 notes).
- Waterfall Aerialist's ward {2} shares the spells-only ward limitation
  described for Dragonfly Swarm above.

**Card-conformance audit (2026-07-09)** — every registered real card is
now shell-checked against a committed Scryfall snapshot
(`managym/tests/fixtures/scryfall_cards.json`, refreshed via
`uv run scripts/refresh_card_fixture.py`) by
`managym/tests/conformance_tests.rs`; new registrations without a fixture
entry fail. Shell fixes landed by the audit: Glider Kids ({1}{U} 2/1 →
oracle {2}{W} 2/3 Human **Pilot** Ally with flying), Waterfall Aerialist
({2}{U} 2/4 Djinn → oracle {3}{U} 3/1 Djinn **Wizard**), "Grey Ogre" →
**Gray Ogre** (real card's spelling), Raging Goblin gained its Berserker
subtype, and text boxes across the sets were aligned to current oracle
wording ("enters" not "enters the battlefield", Ancestral Recall's
targeting text, Accumulate Wisdom's current wording).

**Stage-4 leftovers**

- Until-EOT keyword grants are visible in engine state but **not in the
  encoded observation** (CardData carries printed keywords; PermanentData
  has no keyword slots). If agents should see a Yip-Yip'd flyer, encode
  effective keywords for battlefield card entries.
- The real GW deck exceeds `max_permanents_per_player = 30` in
  token-heavy games (encoder truncates 34→30 with a warning; raw action
  path unaffected). Combine with Stage 2's `max_actions = 20` note when
  sizing the training encoder.
- The blessed UR list sums to 41 cards (17 lands + 24 spells) — kept as
  written ("counts may flex ±1"); GW is exactly 40.
- Enter the Avatar State is registered and tested but **not in the
  blessed 40** — deck constants follow the wave doc lists.
- Random-vs-random on the real matchup is seat/deck-skewed (GW won
  142/200 as villain seat under uniform-random play) — the Stage-4
  matchup table should seat-balance as specced.

## Notes

- Cub + Rallier interaction is a real test case: waterbend taps creatures to
  pay; Cub's trigger adds {G} per creature tapped for mana — the trigger and
  the payment mechanic must compose.
- Milestone 2 (TLA commons complete, `00-pool-audit.md`) reuses every rung
  here; nothing in this slice is throwaway.
