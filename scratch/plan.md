# MTG-432 · Foundations characteristics, replacements and combat constraints

All 29 remaining FDN rows in this cohort (`primary_batch` characteristics,
replacement_events, combat_constraints) are `definition pending` at base
`cf73d9778f`. None waits on another Task's authority: each slice below builds the
shared primitive it needs here. Oracle text is Scryfall's; set files are the first
printing (Scryfall `order=released`, promos included, as `pm15.rs` does).

Sync with main as soon as #593 (MTG-420) merges (auto-merge is on). It adds
`CreatureTypes::All` / `Characteristics::has_all_creature_types`, which slice 4's
changeling builds on, and it touches `ZoneEntry`, LKI and layers. Do slice 1 before
syncing only if #593 is still blocked; rebase slices 4+ onto it.

Overlap with MTG-431 (stack copies, choices, costs; no plan at its checkout yet):
Three Tree Mascot needs changeling (slice 4) and Dragon Mage needs a simultaneous
discard of entire hands (slice 6). This Task owns those two primitives. Hand
MTG-431 the committed revision and paths once they land (`questions.md`).

## Cards and set files

| Card | File | Slice |
|---|---|---|
| Confiscate | `usg.rs` | 1 |
| Harmless Offering | `emn.rs` | 1 |
| Joraga Invocation | `ori.rs` | 2 |
| Vampire Soulcaller | `fdn.rs` | 2 |
| Knight of Grace, Knight of Malice | `dom.rs` | 3 |
| Elenda, Saint of Dusk | `fdn.rs` | 3 |
| Sphinx of the Final Word | `ogw.rs` | 3 |
| Progenitus | `con.rs` (new) | 3 + 5 |
| Crystal Barricade | `fdn.rs` | 3 + 8 |
| Taurean Mauler | `mor.rs` | 4 |
| Tempest Djinn | `dom.rs` | 4 |
| Blanchwood Armor | `usg.rs` | 4 |
| Dreadwing Scavenger | `fdn.rs` | 4 |
| Dwynen, Gilt-Leaf Daen | `ori.rs` | 4 |
| Garruk's Uprising | `m21.rs` | 4 |
| Crossway Troublemakers | `voc.rs` (new) | 4 |
| New Horizons | `xln.rs` | 4 |
| Darksteel Colossus | `dst.rs` (new) | 5 |
| Dryad Militant | `prtr.rs` (new) | 5 |
| Wilt-Leaf Liege | `shm.rs` | 6 |
| Myojin of Night's Reach | `chk.rs` | 6 + 7 |
| Mossborn Hydra | `fdn.rs` | 7 |
| Heroes' Bane | `pjou.rs` (new) | 7 |
| Wildwood Scourge | `m21.rs` | 7 |
| Prime Speaker Zegana | `gtc.rs` | 7 |
| Giada, Font of Hope | `psnc.rs` (new) | 7 |
| Banner of Kinship | `fdn.rs` | 7 |
| Angel of Vitality | `m20.rs` | 8 |

New set files register in `cards/src/lib.rs`. New cards shift definition ids:
refresh Academy fixtures (`uv run python scripts/refresh_academy_fixtures.py`) and
the state-hash pin in `cards/tests/cards/lea.rs` once per slice that adds cards.
Append each printing to `managym/tests/fixtures/scryfall_cards.json`. Ids are in
`/tmp/mtg432/first.txt` for this run; refetch with `curl` if it is gone.

## Slices, riskiest first

### 1 · Control is layer 2 (Confiscate, Harmless Offering)

Today control is the stored `Permanent.controller`, written by
`zones.rs::change_control`. `Effect::GainControl` writes it and records
`Modification::ControlChanged { previous }`, and `end_effects` hands control back.
`layers.rs::Layer` has no control layer, so a static cannot change control.

- Add `Layer::Control` (CR 613.1b) between copy/text and type. A
  `Modification::Control` record or static part says "the record's (or source's)
  controller controls the affected object". Effects apply in timestamp order
  over the entry controller (`Permanent.entry_controller`, new: the player it
  entered under, CR 110.2).
- Confiscate is `StaticAbility::continuous(StaticScope::Attached, None,
  [Modification::Control])`. When it is detached or leaves, the layer stops
  applying and the next effect, or the entry controller, takes over.
  Competing timestamps (Confiscate, then a later GainControl) resolve by order.
- `Permanent.controller` stays the cached result. One place refreshes it after
  every characteristics revision (stabilization, before priority and SBA). A change
  goes through `change_control`'s consequences: summoning sickness (CR 302.6
  reads it), removal from combat (CR 506.4), and `ControlChanged` events for
  triggers. Debug recomputation asserts that the cache equals the layer result.
- Harmless Offering: "target opponent gains control of target permanent you
  control" needs two target clauses, a player and a permanent.
  `PlayerSelector::Target` reads only the primary target (resolution.rs:753).
  Give `PlayerSelector::Target` the clause index (`TargetClause(n)`) and resolve
  it per clause. Don't use `EachOpponent`: it is wrong in multiplayer and it is
  not a target.
- Obsolete: `Modification::ControlChanged { previous }` (becomes
  `Modification::Control`), `end_effects`' hand-back branch (effects.rs:395-417),
  and `Effect::GainControl` writing `controller` directly.
- Checks: `cargo test -p managym-cards --test cards -- usg:: emn::` plus
  `rules_tests cr_613`. Interactions:
  - Confiscate on an opponent's creature, then the Aura is destroyed: control
    returns, and the creature can't attack the same turn.
  - Confiscate, then a later until-EOT GainControl by the opponent, then EOT:
    the Aura's controller has it again.
  - Confiscate on a land the opponent tapped for mana (the land stays tapped).
  - Harmless Offering into a hexproof opponent (illegal player target), and
    with the permanent gone by resolution.
  - The existing GainControl/threaten cards (`tla.rs:5395`, `frf.rs:38`) keep
    their tests.
  - Replay and forks reproduce control. Observation: `controller` is already
    projected; the ControlChanged fact row moves to the Control record's token
    (`record_vocabulary.rs`, `token_renames.rs` for the retired name).
- Stress: `attachments` plus many Confiscates on one board (control refresh
  is now per revision). Add `control-auras` to `bench_states` (8/16 Confiscates
  swapping creatures) and compare `characteristics` and `step:resolve` rows
  before and after.

### 2 · Block requirements are maximized over the whole declaration (Joraga Invocation, Vampire Soulcaller)

Blocks are offered one blocker at a time (`tick.rs:410-436`). `MustBeBlockedByAllAble`
is a greedy per-blocker filter, duplicated in `combat_preview.rs:258-279` and
`structured_offer.rs:771`. Nothing counts satisfied requirements (CR 509.1c).

- Add `Modification::MustBeBlocked` ("must be blocked if able"): one requirement
  per attacker, satisfied by one or more blockers.
- One function, `Game::block_requirement_bound(partial)`, gives the most
  requirements any completion of the partial declaration can satisfy. It counts
  one requirement per lure attacker per able blocker that blocks it, plus one per
  MustBeBlocked attacker that is blocked. It is a bipartite matching over the
  undeclared blockers and the pairs they can legally make.
  `blocker_can_block_attacker` is the edge test; menace and
  `CantBeBlockedByMoreThanOne` are checked on the completed matching.
- The offer for the current blocker keeps exactly the choices, "no block"
  included, after which the bound is still reachable. Tick, preview and the
  structured offer all call it.
- Vampire Soulcaller composes: `BlockRestriction::CantBlock(default)` plus an ETB
  `return_to_hand(Target(GraveyardCardYouOwn{creature}))`. It is the "can't
  block" witness: it never becomes a matching edge.
- Obsolete: the greedy `required` closure in `tick.rs` and its copies in
  `combat_preview.rs` and `structured_offer.rs`.
- Checks: `cargo test -p managym-cards --test cards -- ori:: fdn::vampire_soulcaller`
  and the existing lure test (`ptk.rs:125`). Interactions:
  - Joraga with three attackers against two untapped blockers: every legal full
    declaration blocks two distinct attackers.
  - One blocker that can only block a flier: it must block the flier.
  - A menace attacker under Joraga with one blocker: no requirement can be met, so
    "no block" is offered.
  - Lure plus Joraga on different attackers.
  - Every completion the offers allow is legal. Brute-force the small cases and
    compare.
- Stress: a combat state with 8/16 Joraga attackers and 8/16 blockers of mixed
  flying/reach. Measure `step:block` and `actions` at each blocker. Matching is
  polynomial, but it reruns for each offered choice.

### 3 · Targeting knows its source; protection and player hexproof (Knight of Grace, Knight of Malice, Elenda, Sphinx of the Final Word, Progenitus' protection, Crystal Barricade's hexproof)

`target_is_legal(target, spec, controller)` (resolution.rs:2769) is the one
legality check, called on selection and resolution. It sees only the targeting
player. There is no protection on permanents, no hexproof-from and no player
hexproof.

- `SourceQuality` (state): `Color(Color)`, `CardType(CardType)` ("instants"),
  `Everything`. It describes the source of a spell or ability (CR 702.11d, 702.16).
- Two layer-6 abilities: `StaticAbility::HexproofFrom(SourceQuality)` and
  `StaticAbility::Protection(SourceQuality)`. Printed, or granted through
  `GrantStatic`. `Characteristics` exposes them, and lose-all-abilities removes
  them. Plain `Keyword::Hexproof` stays a keyword.
- `TargetingSource { controller, colors, card_types }` is built once from the
  `AbilitySource`: the spell's own characteristics on the stack, or the ability
  source's current characteristics or LKI. `target_is_legal` takes it.
  `target_meets_requirement` already holds the `AbilitySource`.
- Protection from everything is DEBT. The same predicate gates damage
  (`damage.rs` prevention), attachment legality (CR 704.5m/n: an Aura or
  Equipment can't stay on it), blocks (`blocker_can_block_attacker`) and targets.
  The player version, `Modification::ProtectionFromEverything` (Teferi's
  Protection), shares the predicate. Its `has_protection_from_everything` becomes
  `player_protected_from(player, &TargetingSource)`.
- Player hexproof: the player-side reader also reads battlefield statics that
  give their controller a player modification:
  `StaticAbility::ControllerHas(Modification)`, used for `Modification::Hexproof`,
  a new player modification. `Game::player_has(player, &Modification)` is the
  one reader of records and statics.
- Uncounterability: `StaticAbility::SpellsCantBeCountered(CardPredicate)` on a
  battlefield permanent covers matching spells its controller controls (Sphinx:
  instant or sorcery). `counter_spell` asks `Game::spell_can_be_countered`, which
  reads the spell's own `CantBeCountered` and those statics.
- Cards: the Knights use `HexproofFrom(Color)`, first strike, and
  `PtBuff{This, condition: BattlefieldAtLeast{1, colors_any}}`. Elenda uses
  `HexproofFrom(CardType(Instant))` with two `Compare(LifeAboveStarting)` static
  continuous effects: +1/+1 and menace, plus an additional +5/+5 at ≥10. Sphinx
  is `CantBeCountered` + flying + hexproof + the static. Progenitus is
  `Protection(Everything)` (its replacement comes in slice 5). Crystal Barricade
  is defender + `ControllerHas(Hexproof)` (its prevention comes in slice 8).
- Obsolete: `has_protection_from_everything` (folds into the source-aware
  predicate) and `StaticAbility::NoMaximumHandSize`, which becomes
  `ControllerHas(Modification::NoMaximumHandSize)` if its readers fold cleanly.
  Do not force it.
- Checks: `dom:: fdn::elenda ogw:: con::` and `flow::target_authority_tests`.
  Interactions:
  - A black removal spell cannot target Knight of Grace, and neither can a black
    creature's activated ability. A colorless spell can.
  - The Knight's own controller can target it with black spells (hexproof is
    opponents only).
  - A spell turned black after targeting rechecks on resolution.
  - Elenda can't be targeted by instants, but sorceries and abilities of instants
    on the stack (copies) can follow the quality rule.
  - Elenda at starting life +1 and +10: P/T, menace and the threshold boundary.
  - Sphinx: an opponent's counterspell on your instant does nothing, and your own
    Sphinx can't be countered.
  - Progenitus: unblockable by any creature, damage prevented, Confiscate can't
    target it, a forced Aura falls off (SBA), and global non-target effects (wrath)
    still destroy it.
  - Crystal Barricade: an opponent's "target player" spell has no legal player
    target, while your own still can.
  - Offers stay complete: protection removes target choices, never hides them.
    Observation, LLM text and tokens carry `HexproofFrom`/`Protection` and their
    quality (`observability.rs` guards for StaticAbility and player status), and a
    directed test reads them back.
- Stress: target enumeration over a board of 16 creatures with mixed
  hexproof-from qualities and 8 different targeting sources. Add
  `quality-targets` and measure `actions` (target offers per spell).

### 4 · Changeling and counted static P/T, with the composing cards (Taurean Mauler, Tempest Djinn, Blanchwood Armor, Dreadwing Scavenger, Dwynen, Garruk's Uprising, Crossway Troublemakers, New Horizons)

- Changeling (CR 702.73) is a characteristic-defining ability working in every
  zone. Add a `Keyword::Changeling`, appended. In layers it applies first in layer
  4 as `CreatureTypes::All` (from #593). Off the battlefield,
  `CardPredicate::matches_printing` treats a creature-type subtype test as
  satisfied when the printing has changeling, and `is_creature_type` picks which
  subtypes it covers. `PrintedCharacteristics::creature_types()` is the one
  reader for both paths. A Changeling in hand counts for "reveal an Elf" or
  library searches.
- Counted P/T: replace `StaticPtBuff`'s three `per_*` options with one
  `per: Option<PtCount>`. Its cases: `CreatureTypes { max }`,
  `ColorsAmong(predicate)`, `GraveyardMatching(predicate)`,
  `YouControl(predicate)` (Tempest Djinn's basic Islands, Blanchwood's Forests;
  "you" is the source's controller) and `SourceCounters(kind)` (slice 7's
  Banner). It reads inside the layer context.
- Composing cards:
  - Dreadwing: two triggers (ETB, attacks) that loot; Threshold
    `GraveyardAtLeast{7}` static continuous +1/+1 and deathtouch.
  - Dwynen: reach; `PtBuff OtherYouControl(Elf creature)`; attack trigger
    `gain_life(AttackingMatching{Elf})`.
  - Garruk's Uprising: an intervening-if ETB `ControlsAtLeast{1, min_power 4}`
    that draws; `KeywordGrant YouControl(creature) trample`; ETB `Matching{You,
    creature min_power 4}` that draws (power is read as it enters).
  - Crossway: `KeywordGrant YouControl(attacking Vampire)` for deathtouch and
    lifelink; `Dies{Matching You Vampire}` → `OptionalPayment{Life(2)} →
    draw`.
  - New Horizons: enchant land; ETB `put_counters` on target creature you
    control; `AttachedHas` with `AddOneColorManaValue{Fixed(2)}`. It is the first
    `AttachedHas` on a land, so test the mana ability on the land.
- Obsolete: `StaticPtBuff.per_creature_type_max`, `per_color_among` and
  `per_graveyard_matching` (convert Diligent Zookeeper, Earthen Ally and Avatar
  Destiny).
- Checks: `mor:: dom:: usg:: fdn:: ori:: m21:: voc:: xln::` and the converted cards'
  existing tests. Interactions:
  - A Changeling in hand found by an Elf/Vampire library search, and buffed by
    Dwynen on the battlefield.
  - Taurean Mauler counts as an attacking Vampire under Crossway, gaining
    deathtouch and lifelink.
  - Tempest Djinn counts only basic Islands: Soulstone/other Islands do not count.
  - Blanchwood on a creature that changes control (slice 1): Forests count for the
    Aura's controller.
  - Dreadwing at 6 and 7 graveyard cards.
  - Garruk's Uprising: an ETB 3/3 with a +1/+1 counter as it enters (4 power)
    draws.
  - New Horizons' land taps for two of one color into a real payment (two-pip
    cost), and the remaining mana is checked.
- Stress: `diverse-statics` plus 8/16 counted buffs over a board of 16 lands
  (Djinn, Blanchwood); `characteristics` rows.

### 5 · Graveyard moves from anywhere consult replacements (Progenitus, Darksteel Colossus, Dryad Militant)

Every move reaches `commit_zone_move` through one proposal path, but
`collect_replacements` reads a card's own replacements only for moves to the
battlefield. The only graveyard redirects are the stack-only
`ExileInsteadOfGraveyard` flag and the battlefield-only `ExileIfWouldDie`.

- `ReplacementEffect::GraveyardInstead { what: GraveyardMover, to: Redirect }`.
  `GraveyardMover` is `This` (the moving card's own text, read from any origin
  zone) or `Matching(predicate)` (a battlefield source's text about other
  cards). `Redirect` is `Exile`, or `OwnerLibraryShuffled { reveal: bool }`.
- `collect_replacements` gains a `ZoneMove { to: Graveyard }` arm for every
  origin: resolving and countered spells, discard, mill, destroy, sacrifice,
  SBA and costs. It reads the moving object's abilities (layered on the
  battlefield, so a creature that lost all abilities dies normally; printed
  elsewhere) and battlefield sources. When more than one applies, the affected
  owner chooses through the existing ordering (CR 616.1).
- Reveal is a public reveal event. The shuffle clears positional knowledge
  (constructed-position rule) and draws from the game rng, so replay and forks
  hold.
- Obsolete: `ReplacementEffect::ExileInsteadOfGraveyard` becomes
  `GraveyardInstead{This, Exile}` attached by the CastFromGraveyard flag, so
  there is one redirect type.
- Checks: `con:: dst:: prtr::` and the existing flashback/CastFromGraveyard tests.
  Interactions:
  - Darksteel Colossus discarded, milled, or destroyed while indestructible is
    lost: each time it is revealed and shuffled.
  - Dryad Militant with a resolving instant, a countered sorcery, and a discarded
    or milled instant: each goes to exile, for both players' cards.
  - A Colossus milled with Dryad out: the redirects don't conflict (Colossus isn't
    an instant).
  - Progenitus wrathed: it shuffles in.
  - The flashback redirect still exiles.
  - The viewer's library knowledge is cleared after the shuffle.
- Stress: `graveyard-routes` with 8/16 Dryad Militants out while milling 16
  cards; `step:resolve`.

### 6 · Discards know their cause; whole-hand discard (Wilt-Leaf Liege, Myojin's discard)

- A discard proposal carries `cause: Option<EventCause>` naming the spell or
  ability and its controller (`EventCause` already exists in
  `CounterProvenance`). Cost discards have their payer as the cause.
- `ReplacementEffect::DiscardedInstead { by: Actor, to: ZoneType }`: the
  discarded card's own text (Wilt-Leaf: `by: Opponent, to: Battlefield`),
  consulted in slice 5's graveyard arm. The battlefield entry uses the ordinary
  entry pipeline.
- `Effect::DiscardHand { who: PlayerSelector }`: each selected player discards
  their entire hand in one simultaneous batch (Myojin's `EachOpponent`;
  MTG-431's Dragon Mage uses `EachPlayer`).
- Wilt-Leaf Liege's anthems: two `PtBuff OtherYouControl(colors_any)` effects
  that stack on a green-white creature.
- Checks: `shm::` and `chk::`. Interactions:
  - Wilt-Leaf discarded to an opponent's Mind Rot effect enters the battlefield;
    discarded to your own looting (Dreadwing) or to cleanup hand size, it goes to
    the graveyard.
  - Wilt-Leaf in hand when Myojin's discard hits: it enters.
  - A G/W creature gets +2/+2.
  - Discard triggers still see the discard, and the battlefield entry triggers
    ETB.
- Stress: none new (a discard batch is linear in hand size); covered by
  `blast-discard`.

### 7 · Entry counters read values; choose a creature type as it enters (Mossborn Hydra, Heroes' Bane, Wildwood Scourge, Prime Speaker Zegana, Giada, Banner of Kinship, Myojin's counter)

Today `EntersWithCounters{count: i32}`, `EntersWithPlusOneCounters{count,
condition}` and `EntersWithXCounters{kind}` are fixed. `ModifyCounters` is
gathered only if the entry already has counters (proposed_event.rs:1343).
`ZoneEntry` carries no cast zone, and nothing makes a choice as an object enters
except `EntersAsCopy`.

- One variant, `ReplacementEffect::EntersWithCounters { on: EntryScope, kind,
  count: EntryValue, condition: Option<Condition> }`:
  - `EntryScope` is `This` or `OtherYouControl(predicate)` (Giada: other
    Angels).
  - `EntryValue` is read as the move commits. Its cases: `Fixed`, `X`,
    `GreatestPower { predicate }` over permanents already on the battlefield,
    excluding the entering object (Zegana), and `ControlledAlready(predicate)`
    (Giada's Angels; Banner's chosen type). It excludes objects entering in the
    same event.
  - Condition gains `EntryCastFrom(ZoneType)` (Myojin). It is read from
    `ZoneEntry.cast_from`, new, which is set from `SpellOnStack.cast_from` at
    resolution.
  - The scan runs for every entry, not only entries that already carry counters.
- Entry choice: `ReplacementEffect::AsEntersChoose(EntryChoice::CreatureType)`.
  The entering controller picks from the CR 205.3m creature-type list
  (`subtype.rs`). Above the branching size (64), the choice uses the naming
  decision's balanced groups (`NameScope`-style sorted groups; only a type
  choice commits), so every type is offered. The choice is stored on the
  permanent as `Modification::ChosenCreatureType(String)`, a record lasting as
  long as the object, like `ModeChosen`. It is public (fact row; text "chosen
  type: Elf"; token).
  - `CardPredicate.chosen_type_of_source: bool` reads it for Banner's static
    scope and count. The choice is made before the counter count is read
    (CR 614.12).
- Cards:
  - Mossborn: fixed 1 counter; landfall puts
    `Property{This, Counters(+1/+1)}` counters, doubling with current amounts
    under Doubling Season/ModifyCounters.
  - Heroes' Bane: fixed 4 counters; activated `put_counters(SourcePower)`.
  - Wildwood Scourge: X counters; `PutsCountersOn{AnyPlayer, +1/+1,
    Matching{You, creature not Hydra, another}}` → 1 counter.
  - Zegana: `GreatestPower` counters, plus ETB `draw(SourcePower)`.
  - Giada: flying, vigilance; `OtherYouControl(Angel)` counters by
    `ControlledAlready(Angel)`; `{T}: Add {W}` restricted with
    `ManaRestriction::spell(["Angel"])`.
  - Banner: choose type, fellowship counters by `ControlledAlready(chosen
    type)`; `PtBuff YouControl(chosen type) per SourceCounters(fellowship)`.
    Add `CounterKind::Fellowship` (append) and divinity if missing.
  - Myojin: a divinity counter if cast from hand; indestructible while it has one
    (`SourceCountersAtLeast`); `remove_counters (Divinity,1)` →
    `DiscardHand(EachOpponent)`.
- Obsolete: `EntersWithPlusOneCounters` and `EntersWithXCounters`, and the
  `entry.counters.is_empty()` gate (convert their users).
- Checks: `fdn:: pjou:: m21:: gtc:: psnc:: chk::` and the converted entry-counter
  cards' tests. Interactions:
  - Two Angels entering together under Giada: neither counts the other.
  - Giada plus a Doubling Season-style ModifyCounters: the counters are scaled
    once.
  - Zegana with no other creatures enters 1/1 and draws 1, then draws by its
    layered power with an anthem.
  - Mossborn with two landfalls: 1 → 2 → 4.
  - Wildwood: one counter event on two creatures is two triggers, while a Hydra
    and Wildwood itself trigger nothing.
  - Banner choosing Elf with three Elves gets three counters and +3/+3 Elves.
    A Changeling (slice 4) counts and is buffed.
  - Myojin reanimated (not cast) has no counter and is destructible, and its
    discard hits a Wilt-Leaf (slice 6).
  - Every creature type is reachable in the choice. Replay and forks reproduce
    the choice, and the observation shows it.
- Stress: `choices` with the creature-type choice (group depth 2), and 8/16
  entering Angels under 4 Giadas; `step:resolve`.

### 8 · Life gain and scoped damage prevention in the proposed event (Angel of Vitality, Crystal Barricade's prevention)

- Add a `LifeChange` arm to `collect_replacements`:
  `ReplacementEffect::ModifyLifeGain { who: Actor, additional: i32 }`. The
  affected player orders several such effects (CR 616.1). A gain of 0 is no event
  (CR 119.10) and isn't increased.
- `ModifyDamage` gets `combat: Option<bool>` and a recipient
  `DamageRecipient { by: Actor, players: bool, permanents: Option<CardPredicate>,
  except_source: bool }` in place of `recipient_by: Actor`.
  `DamageTransform::PreventAll` replaces `Prevent(u32)`, which no card uses.
- Cards: Angel of Vitality is flying, `ModifyLifeGain{You, 1}`, and `PtBuff This`
  under `LifeAtLeast{25}`. Crystal Barricade prevents noncombat damage to other
  creatures you control.
- Obsolete: `DamageTransform::Prevent(u32)`; `ModifyDamage.recipient_by`
  (convert `ons.rs:50`, `fdn.rs:2016`).
- Checks: `m20:: fdn::crystal_barricade`. Interactions:
  - Angel plus lifelink combat damage gains +1 once per event, and two Angels give
    +2.
  - Angel at 24 life gaining 1 becomes 26 (+2/+2).
  - Barricade versus a burn spell on your creature (prevented), on itself (not
    prevented), and in combat (not prevented).
  - Barricade versus a damage doubler: prevention and doubling order is chosen by
    the affected controller.
- Stress: `mixed-life-gain` with 4/8 Angels; `step:resolve`.

## Cross-slice validation (integrated)

Synergies to show together at the end:
- Changeling Taurean Mauler under Banner (any type), Dwynen (Elf), Crossway
  (Vampire) and Giada (Angel counters).
- Confiscate stealing a Giada or Banner: the counted "you" changes with control.
- Progenitus against Confiscate, Joraga (unblockable under protection) and
  Dryad Militant.
- Myojin's discard into Wilt-Leaf Liege.
- Angel of Vitality with Elenda's life thresholds.
- Joraga with Crystal Barricade (defender can still block).

Each slice finishes with:
- the debug nextest suite (CI profile, `CARGO_BUILD_JOBS=2 --test-threads 2`);
- `observability.rs` guards for every new variant;
- the Python observation and text tests on a rebuilt extension;
- `tests/model/test_pinned_pilot.py` (append columns or add a token rename; no
  input version bump);
- the FDN ledger rows marked `source definition found`, naming their directed
  fixture.

Engine Budget runs at gate: new definitions move the decision digest, so record a
new reference with `scripts/engine_budget.py record --reason`.
