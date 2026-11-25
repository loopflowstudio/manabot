# MTG-432 · Foundations characteristics, replacements and combat constraints

All 29 remaining FDN rows in this cohort (`primary_batch` characteristics,
replacement_events, combat_constraints) are `definition pending` at base
`cf73d9778f`. None waits on another Task's authority: each slice below builds the
shared primitive it needs here. Oracle text is Scryfall's; set files are the first
printing (Scryfall `order=released`, promos included, as `pm15.rs` does).

Sync with main as soon as #593 (MTG-420) merges (auto-merge is on; still open and
`BLOCKED` at 2026-11-25T02:23Z). It adds `CreatureTypes::All` /
`Characteristics::has_all_creature_types`, which slice 4's changeling builds on, and
it touches `ZoneEntry`, LKI and layers. Slices 2–3 don't need it; before slice 4,
`gh pr view 593 --json state` and `lf sync main` once merged.

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
refresh Academy fixtures (`uv run python scripts/refresh_academy_fixtures.py`) once per
slice that adds cards. The Clone test in `cards/tests/cards/lea.rs` compares
fork hashes directly at this revision. There is no literal hash to update.
Append each printing to `managym/tests/fixtures/scryfall_cards.json`. First-printing Scryfall ids (name, set | id):

- Dreadwing Scavenger, fdn | e24d838b-ab48-410a-9a50-dbfea5da089b
- Elenda, Saint of Dusk, fdn | 24955f5f-093c-4d33-b0c1-911cd36032ce
- Blanchwood Armor, usg | 9b5f3776-74f4-4626-833b-e1b0921d3cbc
- Dwynen, Gilt-Leaf Daen, ori | 91c143a9-c642-425a-a469-a9d158e43c21
- Garruk's Uprising, m21 | 71a4860a-8bb6-45c0-b00a-b4a42da33ab9
- Crossway Troublemakers, voc | 431711c5-c04f-4d34-97c9-5199cfbf9da9
- New Horizons, xln | 15b12c75-1248-4c81-90cf-28e341a885cf
- Knight of Grace, dom | 7bbbddc0-f8b3-4255-bd82-d50f829ca009
- Knight of Malice, dom | b45266f0-eb4f-4a06-bc64-8c2d774b4cc5
- Harmless Offering, emn | f8f3cc4f-7943-4025-b332-b40653b13014
- Taurean Mauler, mor | d50b5df1-b658-4df0-900e-79c44599b93e
- Confiscate, usg | 7cba6d4a-58d0-42d6-b49b-65c72b86007f
- Sphinx of the Final Word, ogw | 866f92a3-2738-4e0f-adda-3ff9227dc17a
- Tempest Djinn, dom | 3acc883b-3aea-4d0b-ae0f-00d4a08c47c1
- Crystal Barricade, fdn | 905d3e02-ea06-45e7-9adb-c8e7583323a2
- Mossborn Hydra, fdn | 7054a0d7-396f-40b4-ab24-db591c3b08f0
- Banner of Kinship, fdn | a14c16c0-4053-46b0-8fa6-be8b4a7a1c8a
- Giada, Font of Hope, psnc | 8ced9f9b-2d20-41ed-90ff-5b37959a0b3b
- Wildwood Scourge, m21 | 46ff0b33-d153-4b0e-ac48-7e5ed70dea09
- Progenitus, con | bcc764b0-3046-4bde-b424-c0f4e1a6169b
- Myojin of Night's Reach, chk | 13a295b0-535e-4c2d-879d-62603d1f2f1b
- Heroes' Bane, pjou | f4fed10c-74f8-49af-8daa-b9c081f549f8
- Dryad Militant, prtr | d88f8449-748f-4843-b4a2-480f6ebd384d
- Prime Speaker Zegana, gtc | f30dfb8e-f540-45ab-a4e8-63425099646a
- Wilt-Leaf Liege, shm | e6a2881f-e771-47d7-a39e-692054ee727f
- Darksteel Colossus, dst | cbc27b24-f085-48b0-8757-cd11fbf25b91
- Angel of Vitality, m20 | e2f39777-b80a-4618-9310-a9e5b91bb2a2
- Joraga Invocation, ori | 65c89431-0881-4aa6-ac15-d4c13b075273
- Vampire Soulcaller, fdn | 2d076293-3b45-4878-8f67-978927cc1f68

## Slices, riskiest first

**1 · Built** (`40ba5de05a`, review order `cace2b71ee`): Confiscate and Harmless
Offering through CR 613.1b layer-2 control. Directed Rust checks, pilot replays, LLM
checks and Academy replays passed; the full debug nextest suite is deferred to gate.
Player hexproof target tests come with Crystal Barricade (slice 3).

**2 · Implemented.** Joraga Invocation and Vampire Soulcaller use native block
offers that maximize the number of satisfied requirements over the entire
declaration. Mixed flying/reach, overlapping lure, and competing attackers have
directed fixtures. The matching oracle also compares against enumeration on all
two-attacker, three-blocker edge sets with menace, blocker limits and fixed prefixes.

**3 · Implemented.** The Knights, Elenda and Sphinx use source characteristics
on announcement and resolution, including last known information. Progenitus
has protection across damage, attachments, blocks and targets. Crystal Barricade
grants player hexproof. Progenitus still needs its graveyard replacement in
slice 5 and Barricade its noncombat prevention in slice 8.

**What the combat cases changed.** Checking menace after an ordinary matching
can reject a legal tied or lower-scoring completion. The solver instead reserves
a partner for any committed singleton and branches when a relaxed optimum
leaves an undecided menace attacker with just one blocker. The two branches
are leaving it unblocked and requiring a pair. Ordinary Joraga and lure boards
use weighted bipartite matching. Worst-case work with multiple ambiguous menace
attackers grows exponentially in their number.

**Attachment investigation.** Seven release runs on the slice 1 binary measured
the same priority checkpoint at 52 and 100 permanents. Cold characteristics
medians were 51.25 and 93.67 µs, with one board rebuild and zero source plans at
both sizes. Action medians were 31.62 and 65.42 µs, with zero board rebuilds.
The original 296.4 µs sample did not recur.

Seven runs on the new binary gave cold-characteristics medians of 48.83
and 101.75 µs and action medians of 39.96 and 96.50 µs. The sample ranges
overlapped the earlier ranges at both sizes. These observations do not isolate
a cause for the original spike.

**Pass 2 release measurements.** In `bench_states joraga-blocks`, the 8-on-8
and 16-on-16 positions exposed six and fourteen block choices along the sampled
line; the engine completed the forced answers. Enumeration took 9.79–25.00 µs
and 12.92–67.04 µs per decision, respectively. Whole block steps took
66.42–230.71 µs and 67.50–146.00 µs. The `quality-targets` recipe enumerated
10–17 targets per spell for eight spells against sixteen creatures in
1.67–2.12 µs per spell. These are ranges of per-checkpoint three-sample medians
on a shared machine.

**Pass 2 checks.** Directed card fixtures, the 4,096-case matching oracle, native
target authority, menace, combat previews, subtype admission, static/effect
tokens, both viewers’ policy rows, Scryfall conformance and native protocol
schemas passed. On the rebuilt CPython 3.13 extension the v9 and v11 pinned
games replayed, all LLM field checks passed, and a real board with the Knight,
Progenitus and Barricade rendered their effective abilities in text and Academy.
The optional frontier checkpoint was unavailable. Academy replayed all 192
retained Commands and refreshed five navigation positions. Python schema parity
and the Svelte type check passed. Gate owns the full debug suite and Engine
Budget reference.

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
  Reuse the protection prevention transform `DamageTransform::PreventAll` from Pass 2. Retain partial prevention for existing numeric shields and their tests.
- Cards: Angel of Vitality is flying, `ModifyLifeGain{You, 1}`, and `PtBuff This`
  under `LifeAtLeast{25}`. Crystal Barricade prevents noncombat damage to other
  creatures you control.
- Obsolete: `ModifyDamage.recipient_by`
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

## Passes

Passes 1 and 2 built slices 1–3. Three passes remain.

- **Pass 3 · characteristics and graveyard routes:** slices 4 and 5, after #593.
- **Pass 4 · proposed-event replacements:** slices 6, 7 and 8.
- **Pass 5 · integration:** the cross-slice synergies below, ledger rows, Engine
  Budget reference.

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
