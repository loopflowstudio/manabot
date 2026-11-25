# MTG-431 — Foundations stack copies, choices, costs and procedure tail

Five of the 56 scoped rows in
`.lf/wave/coverage/research/standard-2026-11/sets/fdn.jsonl` have verified
implementations. Slice 5 delivered Thousand-Year Storm, Bolt Bend and Teach
by Example. Demonic Pact and Maze's End were already playable and have
directed evidence in their ledger rows. MTG-428 covers the same two rows.

**Next: Slice 1, all six entry-choice cards.** The CR 205.3m vocabulary and
the land entry replacement prerequisite are in place. The choice continuation
must preserve ordering when several permanents enter together (below). Each card goes in its first
printing's set file (lookup below). Secluded Courtyard and Uncharted Haven
create `cards/src/neo.rs`, registered in `cards/src/lib.rs`.

Sorting (checked against `managym/state/` and the engine, 2026-11-24, base `cf73d9778f`):

- **25 compose from today's types** (23 in Slice 9, plus Bolt Bend and Teach
  by Example in Slice 5).
- **28 wait on a shared primitive** (Slices 1–8; Curator and Biogenic Upgrade
  each need a new decision shape, the rest are small).
- **Abyssal Harvester** builds on MTG-420's copiable-snapshot authority (PR #593,
  auto-merge requested). Sync main after it lands (`lf sync main`), then
  build Slice 10. It is not a shortcut candidate: if #593 does not land,
  Harvester stays pending in the ledger with "MTG-420 copiable snapshots".

No card here waits on authority owned by MTG-430 (planeswalkers) or MTG-432
(characteristics, replacements and combat constraints). Fiery Annihilation
uses the existing `Modification::ExileIfWouldDie`. Gnarlid Colony uses the
existing `EntersWithPlusOneCounters`.

## First printings (set file, Scryfall id)

grn: Thousand-Year Storm `270a0863-7d07-43f0-925d-a8ce0383a1cb`, Affectionate Indrik `b4c8ddc1-d95c-499f-b1d1-f608f8f07b02` ·
war: Bolt Bend `39b35408-3728-4e1b-9f58-b0775df914d6`, Finale of Revelation `6630c34a-1a97-4e31-9d2c-1150b0aa903e` ·
stx: Teach by Example `aa7fbb9b-50a8-4d18-a667-fe965468ca16`, Deadly Brew `87d33e48-90fc-4aac-b09a-68050bc053b5` ·
eld: Run Away Together `aeffc3c0-567c-442f-ba06-b7d9617c5789`, Heraldic Banner `2e349af5-3f25-46d3-908e-83b2f6028b95`, Charming Prince `dcb94950-3f3e-4876-84f8-d5e4d9cfecee`, Wishclaw Talisman `07c17b01-ee5d-491a-8403-b3f819b778c4` ·
som: Genesis Wave `c920236f-c3d7-421c-b021-103996da790e` ·
neo: Secluded Courtyard `0539b1a5-8704-476f-ba1f-2fe01190e157`, Uncharted Haven `1d4ad89a-3a00-4bf4-a357-4a8a089d4a82` ·
rna: Biogenic Upgrade `0dd73fb2-453f-40b9-8beb-dfa99e6a706e`, Gate Colossus `99767e2f-a558-4d63-b9b6-923d15b433e1` ·
znr: Felidar Retreat `45340647-4d3e-4be1-b0e6-e40cc56a438b`, Gnarlid Colony `7327289d-eed8-44b1-8495-7172e2b49d5f` ·
tsp: Mystical Teachings `a057e5f3-b6bd-4995-b990-714201f36989` · isd: Tribute to Hunger `f77e0f88-2285-4b59-9165-9948c75d77a3` ·
scg: Dragon Mage `7687a201-0ecc-4739-86e3-3b4090d345a8` · xln: Sorcerous Spyglass `85506a24-8d60-475c-9f43-65994caca7d4` ·
bro: Demolition Field `d9c88546-13c9-4d7e-a618-cb2ccd1dbc0f` · m12: Adaptive Automaton `79e42ead-df6e-4181-ae2b-a2abfc3f1d7c` ·
m20: Drakuseth `d09af78f-efde-4107-8406-cb12fd11c686` · m21: Primal Might `1cd8cee8-7ea0-4037-8a3b-39334dc064fb` ·
m19: Diamond Mare `ca600b3f-2c70-489b-b218-6e3245b90114` · psal: Frenzied Goblin `f786de73-98ff-4426-abca-dc789f6848ce` ·
dom: Grow from the Ashes `51d4d1c2-671c-498c-a232-7d076e3dc3bb` · mor: Ballyrush Banneret `a029814e-d84d-43e5-b483-e918871b3333` ·
rtr: Desecration Demon `8242fade-754c-4404-b3fb-f3cccf84b3b6` · one: Ovika `b298cf34-7aa5-4f97-a86c-7f28d2113b87` ·
dtk: Savage Ventmaw `690008d1-d1fe-49ad-810c-84be57cecc6c` · blb: Three Tree Mascot `aaced75b-6e07-457c-8ea2-f74d99710d15` ·
zen: Gatekeeper of Malakir `71db3698-a45c-4eaf-87e6-30502c0c10f4` · mh1: Springbloom Druid `6161d2ed-7cff-4c90-9e74-1d179a6c1498` ·
frf: Flamewake Phoenix `fefd5848-9fe1-4129-a5d7-e51606bf76ef` · mbs: Slagstorm `9e318b03-2aad-462b-a2a9-8b6bdf0e93d6` ·
fdn: Divine Resilience `f3a08245-a535-4d24-b8c0-78759bb9c4b0`, Squad Rallier `65e1ee86-6f08-4aa0-bf63-ae12028ef080`, Cephalid Inkmage `b7e47680-18c7-4ffb-aac4-c5db6e7095ba`, Curator of Destinies `9ff79da7-c3f7-4541-87a0-503544c699b5`, Inspiration from Beyond `b636fe95-664f-4fb1-aab9-28856edeccd6`, Gutless Plunderer `909d7778-c7f8-4fa4-89f2-8b32e86e96e4`, Soul-Shackled Zombie `deea5690-6eb2-4353-b917-cbbf840e4e71`, Fiery Annihilation `54fe00aa-d284-48f9-b5a2-1bd4c5fa8e58`, Strongbox Raider `b2223eb8-59f9-489b-a3f3-b6496218cb79`, Sylvan Scavenging `c35b683c-d3b2-46a1-876a-81b34e8ba2fc`, Zimone `20ccbfdd-ddae-440c-9bc0-38b15a56fdd1`, Arcane Epiphany `06431793-5dfe-4cbf-990b-4bcc960d1f31`, Arbiter of Woe `b2496c4a-df03-4583-bd76-f98ed5cb61ee`, Vampire Gourmand `917514c0-9cd5-4b97-85b9-c4f753560ad4`, Koma, World-Eater `c3b92caa-3401-4b11-9515-152f3e057c05`, Perforating Artist `72980409-53f0-43c1-965e-06f22e7bb608`, Abyssal Harvester `f2e0f538-5825-47e9-883c-3ec6fd5b25ea`

Check each id and append a `managym/tests/fixtures/scryfall_cards.json` row as
`CLAUDE.md` describes. Frenzied Goblin's earliest paper printing is the
Salvat promo; `psal.rs` exists, so follow the repository's convention.

Vocabulary edits that are table data, not primitives:

- Subtypes: the complete Slice 1 vocabulary includes Kithkin and Nightmare.
- Counter kind: `CounterKind::Wish` (append; `record_vocabulary.rs`).
- Tokens in `cards/src/tokens.rs`: `token.raccoon_3_3_g` (Sylvan Scavenging),
  `token.cat_beast_2_2_w` (Felidar Retreat) and `token.phyrexian_goblin_1_1_r`
  (Ovika), each with its first token-printing Scryfall id.

## Build order (riskiest primitive first)

### Slice 1 — Choices made as a permanent enters (6 cards)

Cards: Heraldic Banner, Uncharted Haven, Diamond Mare, Adaptive Automaton,
Secluded Courtyard, Sorcerous Spyglass.

The first pass completed the 367 creature types from CR 205.3m (2026-11-06)
and corrected how the event batch finds an entrant’s own replacements. Zhao
with a conqueror counter turns an entering nonbasic land into a Mountain.
Dark Depths then enters with zero ice counters. Haven and Courtyard must
likewise enter without a recorded choice, even if Zhao later leaves.
`Game::entry_replacements` reads the face that will enter and accounts for
the land-type overwrite at layer 4. Use it for the new replacement.

Verified at `2e7a5bbe46`: 3 subtype tests and 9 focused Zhao, Dark Depths
and Clone tests passed. The rebuilt CPython 3.13 extension exposed 2,868
distinct record names. Both committed pinned pilots (v9 and v11) replayed;
the optional u1952 checkpoint was unavailable.

**Batch-order counterexample.** Genesis Wave can put Spyglass and Heraldic
Banner onto the battlefield together. The controller may inspect the hand
and name a card before choosing Banner’s color, or choose the color first
(CR 101.4c, 614.12a and the two-token example in 616.1g).
`advance_event_batch` asks about members in cursor order, which would fix
these choices in card enumeration order. Retain each answer with its member and
offer the player their outstanding entry choices in either order, observing
APNAP across players. Open the private look only when Spyglass is selected.
Preserve the batch’s original event ordinals and commit it once every
member is ready. Skip the entrant selection when only one remains.

All six definitions and their evidence remain pending.

- **The choice.** `ReplacementEffect::AsEntersChoose(EntryChoiceKind)`, with
  `EntryChoiceKind = Color | CreatureType | CardName`. It runs as a step of the
  entry itself under CR 614.12a, after earlier copy replacements. A lone
  Haven asks once for its color. Use the retained event batch for choices
  across simultaneous entrants.
- **The record.** `Permanent.chosen: Option<EntryChoice>`. It belongs to one
  incarnation: a copy entering makes its own choice, and the choice never
  crosses a zone change.
- **Predicates.** Source-relative `CardPredicate` bits `chosen_color` and
  `chosen_subtype`, modeled on `namesake` (`layers.rs:1606`). They read the
  source's record. The layer read-set (`layers.rs:137`) declares COLORS and
  SUBTYPES reads respectively.
- **Mana.** A chosen-color mana output on the existing dynamic-output path
  (`card.rs:668`), fixed when characteristics are read. Courtyard's
  restriction is fixed when its mana is produced. Add
  `SpendPredicate::CastCreatureSubtype` (creature spell *and* subtype) plus an
  activation-source arm that requires a creature source. `SpendingObject`
  currently snapshots only subtypes. Capture whether the source is a creature
  when its activation is priced, in hand as well as on the battlefield. A
  noncreature Kindred Elf matches neither purpose.
- **Cast triggers.** Diamond Mare's `casts(You, chosen_color)` needs the source
  threaded into `object_matches_predicate` (`triggers.rs:2027`).
- **Creature-type prompt.** The CR 205.3m list exceeds `NAME_BRANCHING`. The
  category table in `managym/state/src/subtype.rs` now includes legal types
  absent from the catalog, including Atog and Servo. Test selecting one.
  The prompt reuses `NameScope`'s grouped navigation (`src/cards/names.rs`)
  over subtypes. It is never one action per type.
- **Spyglass.** It looks at an opponent's hand with the existing private
  `remember_hand` (`zones.rs:189`). The `Decision::NameCard` leaf
  (`decision.rs:3168`) becomes a continuation instead of always queueing
  `ExileNamed`, so the name can land in `Permanent.chosen`. The ban is
  `Restriction::CantActivate { predicate: chosen_name, mana_abilities: false }`
  as a `StaticRestriction`. It covers sources with that name in every zone
  (cycling and graveyard activations included) and spares mana abilities.
  `CantActivateAbilities` stays per permanent.

Checks:

- `cargo test -p managym-cards --test cards -- entry_choice` plus one directed
  test per card in its set file.
- Interactions:
  - Banner choosing red pumps a red creature and not a blue one, and its mana
    is red only.
  - On an ordinary board Haven enters tapped with exactly one prompt. Under
    an active Zhao, neither Haven nor Courtyard asks. Removing Zhao restores
    their printed abilities with no choice recorded for that incarnation.
  - A single batch of Spyglass and Banner admits both choice orders and
    commits neither object early. A fork at the first decision can take the
    other order; neither the unselected look nor the other player's hidden
    cards are exposed.
  - Automaton choosing Elf pumps other Elves and is itself an Elf; a second
    Automaton choosing Construct pumps the first.
  - Courtyard choosing Elf pays for an Elf creature spell and an Elf's
    activated ability, but not a Kindred Elf instant.
  - Spyglass naming a card stops its cycling from hand and its non-mana
    abilities on the battlefield, while its mana abilities still work.
  - A Spyglass naming a card outside both decks is legal.
- Observations:
  - The chosen value reaches policy input as a fact row on the permanent
    (open vocabulary), plus LLM text and learning tokens.
  - Extend `observability.rs` for the new replacement and restriction
    variants.
  - The opponent's view of Spyglass shows the chosen name, not the hand.
  - Replay and forks reproduce the choice.
  - `tests/model/test_pinned_pilot.py` either passes or needs a declared
    column append.
- Stress states:
  - 8–16 Secluded Courtyards with distinct chosen types (distinct restricted
    classes in payment search; compare `distinct-restrictions`).
  - 8 Heraldic Banners or Adaptive Automatons with different choices
    (layer dependency scan).
- Deletes: nothing existing. `NameCard`'s hard-coded `ExileNamed` leaf becomes
  one continuation case.

### Slice 2 — Two piles, chosen by an opponent (Curator of Destinies)

One card, because it alone needs a new decision shape:

- A private partition decision for the controller: per card, face-up or
  face-down.
- A `ChoosePile` decision for an opponent. Its observation shows the face-up
  pile's identities and only the face-down pile's size.
- Knowledge afterwards: a face-down pile that goes to hand stays hidden from
  the opponent; whichever pile goes to the graveyard becomes public.
- Hidden-world sampling keeps the face-up cards fixed.

Do not approximate it with `Modal`: its action space is a bare mode index
(`decision.rs:2134`). `OpponentSplitsRevealed` is a different shape: the
opponent picks among revealed cards, with no piles.

Checks:

- Curator test in `fdn.rs`. Partition sequences cover 0–5 face-up cards.
- The opponent's action space names both piles.
- Reconnect and fork keep face-down identities redacted for the opponent.
- `CantBeCountered` with Flying.
- Stress: the partition is a sequence of per-card binary choices, never
  2^5 enumerated actions. Benchmark the decision once with 5 cards.

### Slice 3 — Counters divided among targets at cast (Biogenic Upgrade)

One card. It announces a per-target share during casting (CR 601.2d):

- Store the shares on the stack object beside `targets`. Copies carry them.
- Read a share per target slot with an `EffectValue::TargetShare`.
- An illegal target's share is lost, never redistributed.
- Replace the resolution-time
  `Effect::DistributeCountersToTappedOpponentCreatures` only if its card's
  oracle text announces at cast. Otherwise leave it alone.

Doubling then composes:
`ForEachTarget { PutCounters { amount: Property { of: Target, Counters(+1/+1) } } }`.

Checks:

- Shares 3 / 2+1 / 1+1+1 resolve to 6 / 4+2 / 2+2+2.
- One target removed in response: the others double their own shares only.
- A copy keeps the shares.
- The action surface states the share per target.
- Stress: three counters over three targets is at most 10 distributions.
  Keep it a per-target choice sequence.

### Slice 4 — Target relations and targeting cardinality (6 cards)

Cards: Drakuseth, Fiery Annihilation, Run Away Together, Divine Resilience,
Charming Prince, Affectionate Indrik (with Primal Might in Slice 9 using the
same fight).

- `TargetRequirement.relation: Option<TargetRelation>`, where
  `TargetRelation = DistinctFromEarlier | AttachedToEarlier(clause) | DifferentControllerFrom(clause)`.
  - Enforce it when spells, triggers and activations choose targets
    (`play.rs:1155` and its equivalents), and again at resolution.
  - `DifferentControllerFrom` uses the last known controller of a target that
    has left.
  - `TargetClause` currently allows the same object in different clauses
    (`ability.rs:1472`); the relation is how a card says otherwise.
- `TargetRequirement.max_if_kicked: Option<usize>`, read in `min_for` /
  `max_for` from `CastChoices.kicked`. Kicker is chosen before targets.
- `TargetSpec::PermanentYouOwn { predicate }` (Charming Prince, with
  `another: true`).
- `Effect::Fight { first: ObjectSelector, second: ObjectSelector }` replaces
  `FightLastTarget`. Keep CR 701.14b (both creatures present and legal) and
  deal both damages as one batch. Convert all 5 uses: `fight_last_target()`
  becomes a constructor.
- Charming Prince's delayed return uses `CreateDelayedTrigger { remember: Moved, EndStep }` (`bro.rs:44`).

Checks:

- `cargo test -p managym-cards --test cards -- target_relation fight`.
- Drakuseth: N targets, with the primary never offered among the secondary
  ones. Retargeting a copy of the trigger keeps them distinct.
- Fiery Annihilation: the Equipment is offered only if it is attached to the
  creature. Moving the Equipment in response makes that target illegal while
  the damage still resolves.
- Run Away Together: a control change in response leaves both creatures
  under one controller, so both targets become illegal.
- Divine Resilience: unkicked offers one target, kicked offers any number.
- Indrik: declining the fight deals no damage. Indrik being gone leaves no
  fight.
- Stress: a Drakuseth trigger with 16 creatures per side plus both players.
  Target enumeration must stay a per-clause sequence.
- Deletes: `Effect::FightLastTarget`.

### Slice 5 — Stack copies and retargeting (complete)

Thousand-Year Storm, Bolt Bend and Teach by Example have directed native play
tests and ledger evidence. Replicate and Storm repeat the existing copy instruction.
A copy can use the last stack appearance of a countered spell and triggers ward
for its final targets. The stress position is `bench_states storm-copies --sizes 8`.

The 17 focused card tests and Scryfall conformance passed. The stress run
offered eight successive retarget decisions. The rebuilt CPython 3.13
extension replayed the four Academy demos (192 Commands) and five navigation
positions. Both bundled pinned pilots passed; the optional u1952 checkpoint
was unconfigured. Current-world admission and the Engine Budget comparison
remain for the integrated gate.

### Slice 6 — Library-top selection (3 cards)

Cards: Gutless Plunderer, Strongbox Raider, Genesis Wave.

- `ObjectSelector::LibraryTop(Box<EffectValue>)`: the top N cards of the
  resolving player's library, fixed when the instruction runs. It records the
  private look, uses the Library selection origin, and limits hidden sampling
  to those positions.
- `ObjectSelector::Unselected`: the enclosing `SelectCards`' `from`, minus
  `Selected` and minus anything no longer present. It is available in
  `effects` and in `if_declined`.
- `SelectCards.max_mana_value: Option<Box<EffectValue>>`, matching
  `LookAndSelect` and `Each`.
- `Effect::GrantPlay { what: ObjectSelector, until: PlayWindow }`. Convert
  `ExileTopCardsToPlay` (1 card) and `GrantGraveyardCastThisTurn` (2 cards)
  where their semantics match, then delete them.

Checks:

- Genesis Wave with X = 5: chosen permanents enter in one batch (each sees
  the others' entry triggers), and the rest go to the graveyard.
- Plunderer: keeping one card leaves it on top and mills nothing else. It is
  not a mill of the kept card and not a surveil.
- Raider: play the chosen card through the end of your next turn, but not
  the other card.
- Viewer privacy holds for the looked-at cards until they are revealed or
  milled.
- Stress: Genesis Wave with X = 16.
- Deletes: `ExileTopCardsToPlay` and `GrantGraveyardCastThisTurn` if fully
  converted.

### Slice 7 — Last-known values of moved objects and shared sacrifice (2 cards)

Cards: Tribute to Hunger, Deadly Brew.

- `Property { of: Moved }` reads the departed permanents' LKI. Record the
  moved-from ids beside `frame.moved`, which today holds the graveyard
  successors and reads 0.
- Generalize `EachPlayerSacrificesCreatures` to `EachPlayerSacrifices { predicate, count }`.
  - It keeps the active player choosing first and one simultaneous event.
  - The resolving player's own sacrifices bind to `Moved`, so
    `Condition::Moved` answers "if you sacrificed".
  - Convert its 1 card.
- `Except::Moved` for "another permanent card" in a graveyard selection.

Checks:

- Tribute gains the sacrificed creature's toughness, including a pumped
  toughness (LKI).
- Deadly Brew with no creature of yours sacrifices nothing, offers no return,
  and cannot return the sacrificed card.
- Deletes: the `EachPlayerSacrificesCreatures` name.

### Slice 8 — Kicked entries, payment obligations and new vocabulary (8 cards)

Cards: Gnarlid Colony, Gatekeeper of Malakir, Ovika, Savage Ventmaw, Three
Tree Mascot, Mystical Teachings, Wishclaw Talisman, Ballyrush Banneret.

- **Kicked on entry.** `kicked` travels on `ZoneEntry` / `Permanent` like
  `cast_x`, and onto pending triggers and their frames. The entry filter
  (`proposed_event.rs:1275`) and intervening-if answer `Condition::Kicked`
  from it (Gnarlid Colony, Gatekeeper of Malakir).
- **Composite ward.** `PaymentProgress` owes a list of components (mana, then
  life), so ward can be "{3}, Pay 3 life" (Ovika). Prefer a list over a
  `ManaAndLife` variant.
- **Mana expiry.** `ManaClass.combat: bool` becomes
  `expiry: Step | EndOfCombat | EndOfTurn`, which `clear_mana_pools` respects
  (Savage Ventmaw). This deletes the bool and its 45 references.
- **Changeling** (CR 702.73) is every creature type in every zone: the layer-4
  subtype view, printed-spell matching (`predicate.rs:256`, `:314`), the
  `SpellSnapshot` used by spend restrictions, and distinct-type counts. Never
  list all subtypes on the card.
- **Keyword predicate.** `CardPredicate.keyword: Option<Keyword>` (Mystical
  Teachings' "card with flash"). Replace the `flying` and `deathtouch` bools
  with it; `not_flying` stays.
- **Data only.** `CounterKind::Wish` (Wishclaw) and the `Kithkin` subtype
  (Ballyrush).

Checks:

- Gnarlid Colony: kicked enters with 2 counters, unkicked with none.
- Gatekeeper: unkicked does not trigger at all.
- Ovika: ward is satisfied only by both {3} and 3 life, and a player at 2
  life cannot pay it.
- Ventmaw: its mana survives into the second main phase and empties at
  cleanup. Ordinary mana still empties at the end of each step.
- Mascot: it counts as an Elf for Adaptive Automaton, Courtyard and
  Ballyrush. Its mana ability works once per turn.
- Ballyrush: a Kithkin Soldier spell costs 1 less, not 2.
- Wishclaw: after the opponent gains control, they can activate it on their
  own turn only.
- Stress: 8 Ventmaw-style end-of-turn pools mixed with combat and restricted
  mana (class count in payment search), and Changelings under 8 chosen-type
  anthems.
- Deletes: `ManaClass.combat` and `CardPredicate.{flying, deathtouch}`.

### Slice 9 — Cards that compose today (23 cards)

Slice 5 delivered Bolt Bend and Teach by Example. Shapes and analogs come
from the surveys:

| Card | Shape |
|---|---|
| Primal Might | Pump X, then `up_to(1)` fight through Slice 4's `Fight` (Earth Rumble, `tla.rs:3143`) |
| Slagstorm | `Modal`: `each(creature)` damage, or `DealDamageToPlayer { EachPlayer }` (one batch; first use, so test it) |
| Zimone | `beginning_of(Combat)` with `up_to(2)` counters; activated `ForEachTarget { PutCountersFrom { from: Target, on: Target } }`. Add a `double_counters` constructor. |
| Soul-Shackled Zombie | `same_graveyard` `up_to(2)` (`tdm.rs:703`); `Objects { of: Moved, creature, in Exile } ≥ 1` gives the drain |
| Frenzied Goblin | Attack trigger targeting, then `OptionalPayment {R}` (Kalastria Highborn, `wwk.rs:23`) |
| Squad Rallier | Water Tribe Rallier (`tla.rs:10916`) |
| Cephalid Inkmage | `Surveil 3` with threshold `CantBeBlockedIf` |
| Inspiration from Beyond | Mill 3, `SelectCards` from the graveyard, Flashback |
| Finale of Revelation | Spirit Water Revival (`tla.rs:8768`); `SelectCards { Each land, max 5 }` untap (first use of `Each` there) |
| Demolition Field | Price of Freedom (`tla.rs:7430`) |
| Grow from the Ashes | Kicker `If` with 1 or 2 basics |
| Springbloom Druid | `may_sacrifice(land)` with `if_you_do` search for two tapped basics |
| Dragon Mage | Combat-damage trigger: each hand to the graveyard, then each player draws 7. Prefer one simultaneous discard batch (see questions). |
| Sylvan Scavenging | End-step `TriggerModes`; the Raccoon mode is an `If ControlsAtLeast(power 4)` at resolution |
| Felidar Retreat | Landfall `TriggerModes`: Cat Beast, or counters on each creature then vigilance until end of turn |
| Arbiter of Woe | `AdditionalCost::Sacrifice` on a creature spell (first creature use, so test it) |
| Vampire Gourmand | `may_sacrifice(another creature)` with `if_you_do` draw and can't be blocked |
| Perforating Artist | Raid end step; per opponent: discard, or sacrifice a nonland permanent, or lose 3. A nested optional; never `Modal`, which would offer impossible modes. |
| Desecration Demon | Each combat, the opponent's optional sacrifice; `If Moved` taps and puts a counter |
| Flamewake Phoenix | Attacks each combat; graveyard ferocious trigger with `OptionalPayment {R}` (Wolfbat, `tla.rs:11197`) |
| Arcane Epiphany | `CostsLess 1` if you control a Wizard |
| Gate Colossus | `CostsLessPerPermanent(Gate)`, `CantBeBlockedBy { max_power: 2 }`, graveyard Gate-entry trigger with optional move to library top (first graveyard entry trigger, so test it) |
| Koma, World-Eater | Koma, Cosmos Serpent shapes (`khm.rs:29`) with `token.komas_coil` |
| Strongbox Raider, Genesis Wave, Gutless Plunderer | Slice 6 |
| Tribute to Hunger, Deadly Brew | Slice 7 |
| Charming Prince, Affectionate Indrik | Slice 4 |
| Gnarlid Colony, Gatekeeper, Ovika, Ventmaw, Mascot, Teachings, Wishclaw, Ballyrush | Slice 8 |

The remaining 23 compose-only cards land as one pass with one directed test
each in their set files. Add `observability.rs` rows where a card is the first
real user of a static or cost row.

### Slice 10 — Abyssal Harvester (after MTG-420 lands)

- A per-turn "put into a graveyard this turn" record that generalizes
  `from_battlefield_this_turn` (`turn.rs:282`, `statics.rs:573`), exposed as a
  `CardPredicate` field.
- `Except::CreatedThisWay` for "exile all other Nightmare tokens you control".
- The rest is `CreateTokenCopy { exceptions: [AddSubtypes(Nightmare)] }` on
  MTG-420's copiable snapshot.
- Checks:
  - A creature milled this turn is a legal target; one from a previous turn
    is not.
  - The new token survives while older Nightmares are exiled.
  - Copying a transformed or face-down card follows MTG-420's rules.

## Integrated validation

- Rust: `CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 --manifest-path managym/Cargo.toml`
  in debug before landing; `cargo fmt --all`.
- Rebuild the extension and refresh Academy fixtures
  (`scripts/refresh_academy_fixtures.py`).
- Current-world admission:
  `uv run --extra dev --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q`,
  plus `tests/model/test_pinned_pilot.py`.
- Stress states listed per slice through `bench_states`. Engine Budget stays
  within 10%.
- Synergies to exercise together:
  - Thousand-Year Storm with Teach by Example (a Teach copy is not cast, so
    it never raises Storm's count).
  - Bolt Bend redirecting a Drakuseth trigger's single-target copy is not
    legal; it is legal on Fiery Annihilation only if that spell has one
    target.
  - Three Tree Mascot with Adaptive Automaton, Secluded Courtyard and
    Ballyrush Banneret.
  - Sorcerous Spyglass naming Three Tree Mascot leaves its mana ability
    working; naming Wishclaw Talisman stops it.
  - Genesis Wave putting Heraldic Banner and Uncharted Haven onto the
    battlefield simultaneously: each asks its entry choice.
  - Deadly Brew against Desecration Demon.
  - Gate Colossus with Maze's End.
- Ledger: update each delivered row with `status: implemented`, `key`,
  `evidence` (test path::name) and `result`, as High Fae Trickster's row does.

## Ledger status

51 scoped rows remain pending, including Abyssal Harvester.
