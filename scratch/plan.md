# MTG-432 · Foundations characteristics, replacements and combat constraints

All 29 remaining FDN rows in this cohort (`primary_batch` characteristics,
replacement_events, combat_constraints) are `definition pending` at base
`cf73d9778f`. None waits on another Task's authority: each slice below builds the
shared primitive it needs here. Oracle text is Scryfall's; set files are the first
printing (Scryfall `order=released`, promos included, as `pm15.rs` does).

#593 (MTG-420) merged as `45ddf3bca7` and is synced here (`b3485ac26c`): it supplies
`CreatureTypes::All` / `Characteristics::has_all_creature_types` (`flow/statics.rs`),
which slice 4's changeling builds on.

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

**2 · Built** (`6b9f01345d`, review order `df49092f43`): Joraga Invocation and Vampire
Soulcaller; native block offers maximize satisfied requirements over the whole
declaration (4,096-case matching oracle against enumeration).

**3 · Built** (same commits): the Knights, Elenda, Sphinx, Progenitus' protection and
Crystal Barricade's player hexproof, via source characteristics (with LKI) on
announcement and resolution. Progenitus still needs slice 5's graveyard
replacement; Barricade slice 8's noncombat prevention.

Pass 2's directed, pinned-pilot, LLM, Academy (192 Commands) and schema checks
passed on the rebuilt extension; the full debug nextest suite and Engine Budget
reference are deferred to gate.

**Finding that shapes combat code.** Checking menace after an ordinary matching can
reject a legal tied or lower-scoring completion. The solver reserves a partner for a
committed singleton and branches (unblocked / paired) when a relaxed optimum leaves an
undecided menace attacker with one blocker; work grows exponentially in the number of
ambiguous menace attackers. Release `bench_states joraga-blocks`: 9.79–67.04 µs per
decision at 8-on-8 and 16-on-16. The slice-1 attachments spike (296.4 µs at 100
permanents) did not recur in fourteen runs (medians 94–102 µs); no cause isolated.

**4 · Built** (`e8f5bd548a`). All eight cards in the table above are playable. Changeling uses the printed
face in every zone and the layer-4 creature types on the battlefield.
`StaticPtBuff.per` counts from one `PtCount`. The three existing card
programs retain their serialized field names. New Horizons’ enchanted Forest
paid both white pips of Leonin Skyhunter. Both viewers see changeling as a
named v11 keyword fact. The FDN ledger links each card to its play fixture.

The directed Rust checks passed, including the three migrated TLA cards.
The pinned v9 and v11 pilots replayed on the rebuilt CPython 3.13 extension.
The optional frontier checkpoint was unavailable. Academy replayed 192 demo
Commands and refreshed all five retained navigation positions.

Scryfall conformance passed. The release `bench_states` characteristics
measurement took 12.25 µs at priority with 8 counted buffs over 16 lands
and 21.46 µs with 16 buffs. Across priority, attack and block the respective
ranges were 11.17–14.62 and 21.38–51.13 µs (median of three per position).
`diverse-statics` took 9.42–10.21 and 23.25–28.08 µs at the same sizes.

### 5 · Graveyard moves from anywhere consult replacements (Progenitus, Darksteel Colossus, Dryad Militant)

**Pass 3 counterexample from code inspection.** Discarding an instant to pay a
cost goes through `pay_discard_cost → move_object → complete_immediate_batch`.
Under the proposed design, two Dryad Militants would produce two candidates.
The batch would return `OrderReplacement`, which panics in
`advance_immediate_batch`. A resolving spell whose targets all became illegal takes the same
immediate route. Cleanup and several discard decisions also move directly.
A mill redirected back to the same library is rejected by
`proposal_is_current` before its reveal and shuffle can happen.

First exercise a discard payment and a fizzled spell with two
Militants. Retain the payment/decision/resolution continuation across the
affected player's ordering choice and commit its receipt or consequences once.
Audit direct graveyard callers, the upfront death/flashback rewrites, and
the affected permanent or spell's controller when it differs from its owner.
Represent revealing and shuffling a card already in its library without a
spurious zone change. Build the three cards together on that proven route.

- `ReplacementEffect::GraveyardInstead { what: GraveyardMover, to: Redirect }`.
  `GraveyardMover` is `This` (the moving card's own text, read from any origin
  zone) or `Matching(predicate)` (a battlefield source's text about other
  cards). `Redirect` is `Exile`, or `OwnerLibraryShuffled { reveal: bool }`.
- `collect_replacements` gains a `ZoneMove { to: Graveyard }` arm for every
  origin: resolving and countered spells, discard, mill, destroy, sacrifice,
  SBA and costs. It reads the moving object's abilities (layered on the
  battlefield, so a creature that lost all abilities dies normally; printed
  elsewhere) and battlefield sources. When more than one applies, the affected
  object's controller (owner if uncontrolled) chooses through CR 616.1 ordering.
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

Passes 1–3 built slices 1–4. Pass 3 also set out to build slice 5 but stopped at
the counterexample above; no slice 5 code exists.

- **Pass 4 · graveyard routes (slice 5):** first the two-Militant discard-payment
  and fizzled-spell reproductions, then the continuation across the ordering
  choice and the in-library reveal/shuffle, then the three cards. Teaches whether
  immediate batches can suspend on a replacement-order decision.
- **Pass 5 · proposed-event replacements:** slices 6, 7 and 8 (6 reuses slice 5's
  graveyard arm).
- **Pass 6 · integration:** the cross-slice synergies below, ledger rows, Engine
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
