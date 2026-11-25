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
announcement and resolution. Slice 5 completes Progenitus's graveyard
replacement. Barricade's noncombat prevention is planned for slice 8.

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

**4 · Built** (`e8f5bd548a`): the eight changeling and counted-bonus cards;
`StaticPtBuff.per` is one `PtCount`, old serialized field names kept for checkpoints.

**5 · Built** (`dac0dd6d57`): `ReplacementEffect::GraveyardInstead` (replacing
`ExileInsteadOfGraveyard`, token rename recorded) from every origin; payment,
decision and resolution continue across the CR 616.1 ordering choice (the two-Militant
discard-cost and fizzle panics are fixed); a milled Colossus reveals and shuffles in
place with a fresh identity (CR 701.20d). Progenitus, Darksteel Colossus, Dryad Militant.

Finding for slice 6: search determinization (opposing seat, seed 0) moved a private
discard payment out of hand without a receipt. `selection_sampling.rs` now keeps the
announced spell and binds the discard to a matching hand card. Whole-hand discard
batches must get the same sampling coverage.

Slices 4–5 passed directed, pinned v9/v11 pilot, Scryfall, Academy (192 Commands, five
navigation positions) checks; the optional frontier checkpoint was unavailable. Release
`bench_states`: counted buffs 11.17–51.13 µs per decision at 8/16 buffs;
`militant-mill` step:resolve 106.58 / 142.58 µs at 8 / 16.

**6 · Implemented** (Pass 5, `b2da13add6`): Wilt-Leaf Liege and
Myojin's whole-hand discard use the instruction's controller and a simultaneous
batch. The chooser may suspend before any hand card moves. Search preserves
the number of committed hand slots and the replacement choices already offered.

**7 · Implemented** (same pass): Mossborn Hydra, Heroes' Bane, Wildwood
Scourge, Prime Speaker Zegana, Giada, Banner of Kinship and Myojin use the
shared entry-counter replacement. Cast origin and the type chosen for Banner
belong to the pending entry. Giada counts pre-entry Angels and Banner's
public choice survives as a fact on that permanent.

**8 · Implemented** (same pass): Angel of Vitality replaces a positive gain.
Lifelink proposals use damage after prevention and group that damage by
source before the simultaneous batch commits. Crystal Barricade's prevention
selects noncombat damage to other creatures its controller controls.

The first directed run exposed a Tinybones interaction. Its `ForPlayer`
instruction changed the frame's player before the discard, so the Liege went
to the graveyard and Tinybones exiled it. `a4504cdc6b` keeps the activating
player on that frame across the discard choice. The targeted test checks the
Liege both before and after the Tinybones trigger resolves.

## Passes

Passes 1–5 implement slices 1–8.

- Pass 5 passed the directed card positions, learning-token guards and
  Scryfall conformance for every registered real card. The release extension
  replayed the pinned v9 and v11 pilots. The optional frontier pilot was
  unavailable.
- The Python text surface replayed Banner's group and leaf Commands on exact
  forks, encoded every offer, and showed the committed type to both seats.
  Landcycling triggered Tinybones, and one Angel replaced a gain from 24 to
  26. Academy replayed 192 demo Commands and all five navigation positions.
  The board prompt suite passed all 115 cases.
- Release `bench_states` timings below are medians of three steps on a
  contended shared host. Giada's size counts simultaneous entrants with
  four Giadas already present. Banner's size counts Elves, and Angel of
  Vitality's size counts replacement sources.

  | Position | Size | Resolve (µs) | First choice (µs) | Final choice (µs) |
  |---|---:|---:|---:|---:|
  | Giada | 8 | 100.5 | 80.5 | 176.4 |
  | Giada | 16 | 110.0 | 384.6 | 155.9 |
  | Banner | 8 | 47.1 | 64.9 | 109.6 |
  | Banner | 16 | 40.5 | 39.8 | 212.8 |
  | Angel of Vitality | 4 | 52.0 | 43.9 | 75.0 |
  | Angel of Vitality | 8 | 25.5 | 62.7 | 83.7 |

## Pass 6 · Integrated positions complete

| Position | Observable result |
|---|---|
| Taurean Mauler with Giada, Dwynen, Banner and Crossway | Mauler enters with one counter, becomes a 4/4 Elf, counts as Banner's only Advisor, then attacks as a 5/5 Vampire with deathtouch and lifelink. |
| Confiscate on Giada | With two other Angels, the new controller uses Giada's white mana to cast a third Angel. It enters with three counters. Crusader of Odric costs the same and cannot use that mana. |
| Progenitus with Confiscate, Joraga and Dryad Militant | Confiscate must target another permanent. Joraga goes to exile and Progenitus attacks unblocked for 13. Fumigate sends Progenitus back to its library. |
| Myojin into two Wilt-Leaf Lieges | The opponent's Ogre goes to the graveyard and both Lieges enter as 6/6s. The activating player's Liege stays in hand. |
| Angel of Vitality with Elenda | A Hawk dealing one damage takes Elenda's controller from 19 to 21 or from 28 to 30, crossing each threshold by exactly the extra life. |
| Joraga with Crystal Barricade | The Wall's only legal block stops the 5/5 Ogre and costs the Wall its life. |

The focused debug run passed all 16 tests, including these six positions.
The FDN ledger has ten updated rows. All 29 cohort rows were checked for a
source definition and a named directed fixture.

## Pass 7 · Conceptual alignment built (`9fcab2395c`)

- 7.1: wire/API/schemas/text/Academy carry `entry_controller_id`; the internal
  `Permanent.control_differs_from_entry` bit and v11 column stay.
- 7.2: `SourceQuality::{ALL, wire_name, from_wire_name}`; observations carry plain
  names, the encoder maps back to `token_name()` so v11 tokens are unchanged.
- 7.3: `CombatState.blocked` (CR 509.1h); phase-out removes from combat
  (CR 506.4, 702.26b); test `c17::phased_out_blocker_leaves_combat_while_attacker_stays_blocked`.
- 7.4: `Condition::CastFrom { zone, owner }` replaces `EntryCastFrom` and the
  condition `CastFromGraveyard` (token rename recorded).

Passed on the rebuilt release extension: protocol/vocab Rust tests, directed
card tests (329, usg/dom/chk/fdn/con/c17/fin/observability), `combat_preview`,
`test_text_observation.py` (105), pinned v9/v11 pilots (u1952 skipped),
Academy protocol/replay Python tests and UI view/prompt/protocol tests.
Six schemas and five Academy opening fixtures refreshed.

## Coverage review at `92aa1dd819`

No defects. The compress pass (`a0194b3441`) moved phase-out and control-change
combat removal into one `remove_permanent_from_combat`. It also derived the spell's
graveyard-cast flag inline and made the cast-origin text read "cast from exile".
The debug card suite covers usg, dom, chk, fdn, con, dst, prtr, psnc, shm, m20,
m21, ori, emn, c17 and observability. Its 363 tests passed, including the
phased-out blocker and Confiscate departure positions. The Python text tests need
a rebuilt extension, so they are deferred to gate.

## Decomposition · Built (`1eec144255`, `384e717f6a`, `1eedafb796`)

Control, hexproof/protection (`SourceQuality`), graveyard replacement and chosen
creature type already had one path each. Three fused axes were unified:

- **D1** layer-7 quantity: `LayerValue::Count(Box<LayerCount>)` via `Context::value`;
  every P/T static is `StaticAbility::continuous` or `StaticAbility::cda` (`SetBasePt`).
  Deleted `PtCount`, `PowerCda`, `StaticPtBuff`, `StaticAbility::{PtBuff,
  CharacteristicPower, CharacteristicPt}`, `Operation::{Cda, Buff}`, `Game::cda_value`.
- **D2** entry quantity: `EntersWithCounters.count: EffectValue` on a pending-entry
  frame (`effect_value_with_chosen_type` for Banner). Deleted `EntryValue`, `Game::entry_value`.
- **D3** damage replacement: `DamageSource { by, predicate, this }`,
  `DamageRecipient.this`, `DamageTransform::Add(Box<EffectValue>)`. Deleted
  `PreventCombatDamageToOrFromSelf`, `OpponentDamagePlusCounters`,
  `ModifyDamage.{creature_source, source_by}`, `Game::damage_source_facts`.

Finding: newtype enum variants also emit closing `Structure` tokens
(`/Enum::Variant`); retiring both in `token_renames.rs` let pinned v9/v11 pilots
replay unchanged without re-recording. Verified on the rebuilt release extension:
directed card tests (136 + 243), `rules_tests` (30), `managym` unit tests, pinned
v9/v11 pilots (u1952 skipped), `test_text_observation.py` (105). A grep at
`831cb61394` finds none of the deleted names outside `token_renames.rs`.

### Left as they are

- `StaticKeywordGrant` / `StaticTypeGrant` duplicate `continuous` the same way,
  but this cohort added no axis to them and `SpellsYouControl` keyword grants
  are read off the stack by `spell_keywords`. Recorded in `questions.md`.
- `Effect::DiscardHand` beside `Effect::Discard`: one is a simultaneous batch
  with no choice, the other a decision; they share `discard_batch` and
  `DiscardCause`.
- Player protection (`Modification::ProtectionFromEverything`) prevents damage
  in `damage.rs` without a replacement candidate. No ordering against it is
  observable (prevent-all commutes with every transform present).

### Review criterion

Tempest Djinn, Blanchwood Armor, Banner of Kinship, Giada, Prime Speaker
Zegana, Wildwood Scourge and Crystal Barricade are written only with
`StaticAbility::continuous`, `LayerValue`, `EffectValue`, `EntersWithCounters`
and `ModifyDamage`; no type in the Delete list remains; a new "+N/+N for each
X", "enters with counters equal to X" or "prevent/increase damage from A to B"
card needs no engine edit when X, A and B are existing predicates and values.

## Remaining

Gate: the full debug nextest suite (CI profile, `CARGO_BUILD_JOBS=2`,
`--test-threads 2`) and the Engine Budget reference record with its reason.
