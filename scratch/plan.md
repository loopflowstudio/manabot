# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Model as built

- **Profiles.** A physical `CardDefinition` (`managym/state/src/card.rs`) carries its front face plus inset Adventure, Omen or prepare programs. `Object::spell_definition` and `CastChoices::spell` resolve the selected profile, which is bound at announcement and carried on `SpellOnStack`. Targeting, payment, restricted mana, alternative costs, cascade (`matches_printed_spell`), copies and resolution read that profile.
- **Permissions.** Exact-object grants and zone/type allowances live in `managym/src/flow/cast_permission.rs`. Each grant has a caster rule (a fixed player, or whoever controls the source), a duration, and any quota it consumes. Distinct source, quota or cost become distinct native offers. Both viewers see the grants in observations, v11 grant facts, LLM text and Academy inspection.
- **Preparation.** Entry (`flow/zones.rs`) creates the linked exile copy and an exact-source grant. Cleanup (`flow/sba.rs`) consults that grant. Repeat-prepare and unprepare resolve against the live link without a reverse index.
- **Resolution-time casts.** Etali and Cascade resume their parent when the child spell commits (`flow/decision.rs`).

## Delivered (detail in git and the set ledgers)

- FDN 4/17 rows: Zul Ashur `97593abf`, Muldrotha `8d380c05`, Vizier, Etali `192bde78` (fixtures in `managym/cards/tests/cards/fdn.rs`, `rix.rs`, `cmr.rs`).
- HOB: Smaug // Spew Flame and Bilbo `f7aa87fe`, with overlap pricing fixed in `897faeaf` (`hob.rs`).
- TDM 2/13: Twinmaw // Charring Bite `c30088d4`, Scavenger Regent // Exude Toxin `04fbb56a` (`tdm.rs`).
- SOS/FRA 44/62 preparation rows: prepared entry for 31 definitions `1852a68d`, the two Seed Suture sources and transfer `b5ff828c`/`25bf25d7`/`fae97841`, 12 resolution-time sources `a3555c39`, Emeritus of Truce `3beb79c5`.
- Inset spells cast through their own definition (alternative costs, cost predicates, stack matching) `8be13fed`.

After `8be13fed`: 47 focused native tests and 95 Python checks passed (cold replays, semantic text, both bundled pilots, Academy navigation). All 252 navigation prefix Commands replayed with identical offers and revisions.

## Lessons that steer the remaining work

- **Every catalog insertion shifts definition rows.** Recompare the five Academy navigation positions with registry references resolved by name before regenerating digests. Then refresh with the fixture writer (`scripts/refresh_card_fixture.py`) and replay the prefixes.
- **Opening decisions must finish before constructed boards.** Fixtures that build a board need both opening-hand decisions completed first, or state-based actions never run (the Seed Suture repair).
- **Mana in fixtures.** With interchangeable lands the engine offers a payment decision after targeting. Submit the offered automatic payment before passing priority.
- **New hashed field names need tokens.** The runtime vocabulary test caught `exile_instead_of_graveyard`. Run the learning-token guards whenever a rules type gains a field.

## Remaining slices

1. **Preparation rows without new history (SOS/FRA, 12 rows).** Use the existing prepared-source, pending-resolution and source-loss forks:

   | Card | Behavior to distinguish |
   |---|---|
   | Joined Researchers | Opponent hand size checked at end-step trigger and again at resolution |
   | Spiritcall Enthusiast | One or more tokens entering in a batch prepare once |
   | Grave Researcher | Surveil finishes before counting graveyard creatures; Reanimate reads the moved creature's mana value |
   | Kirol, History Buff | One or more cards leaving your graveyard in a batch prepare once |
   | Emeritus of Ideation | Decline, or exile eight distinct graveyard cards, as the attack cost |
   | Harmonized Trio | Tap two other untapped creatures plus the source as one cost |
   | Lluwen, Exchange Student | Sorcery-speed graveyard-creature exile as a cost; the Pest token's attack trigger |
   | Variable Chaser | Each player's optional whole-hand discard, collected in order, then draw seven |
   | Pyre Rhymer | Turn-scoped extra red when tapping Mountains, including ones gained later |
   | Heartwood Crafter | Spend restriction by cast origin: its mana cannot pay for hand spells |
   | Sanar, Unfinished Genius | Treasure activation requires an instant or sorcery cast this turn (`spells_cast` already records the objects) |
   | Codie, Ravenous Codex | Copy an actually cast prepared spell; mass-prepare acts once per eligible source |

2. **Printed programs (HOB/TDM).** Run the other 16 Adventures and 11 Omens through the hand-profile, target/payment and stack-departure fixtures. Check each front and inset program. Group any card that needs a new effect by the effect's owner. Keep the Monica Rambeau // Photon modal casting probe with the profile enumeration checks: chosen-face entry and return to the front face off the battlefield. Battlefield transform belongs to the forms work.
3. **Zone permissions and alternate costs (FDN, 13 rows).** Tinybones (foreign-owned land execution, with control-transfer and prohibition forks), Sphinx of Forgotten Lore, High Fae Trickster, Harbinger of the Tides, Omniscience (paid versus free hand casts with additional costs), Loot, Think Twice, Reassembling Skeleton, Flamewake Phoenix, Suspicious Shambler, Quilled Greatwurm, Redcap Gutter-Dweller, Gate Colossus. Keep each card's other printed abilities.
4. **Blocked on MTG-418 event history (6 rows).** Emeritus of Woe and Bloodline Recollector need creature deaths this turn. Leech Collector needs the first life gain of the turn, and Scheming Silvertongue needs total life gained. Whiplash Wordsmith needs noncombat damage dealt to an opponent. Hexhaven Dueling Arena needs to know which creature attacked. `PlayerTurnHistory` (`managym/src/flow/turn.rs`) records only attacked, sacrifices, land entry, draws, bending and life lost. Once MTG-418 merges, run `lf sync main` and read its history API. Then build these rows on the slice 1 fixtures.

Advance between slices when the ledger gives each completed row a directed witness. Track each row in its set ledger with definition, test names and result.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Muldrotha, artifact creature and a Zul-eligible Zombie | Fork the type choice. Check a new Muldrotha that turn, the next-turn reset, land allowance and a prohibition. With Omniscience, compare paid and free hand casts, required additional costs and manual spending. |
| Tinybones, opposing discarded land and spell with stash counters, second Tinybones | The current controller plays foreign cards that an earlier Tinybones exiled. Check own-turn timing, land plays, ownership versus control, mana spending and a hand-only prohibition. Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |
| Angel and Phytomedic with Seed Suture | Not yet run: remove abilities (stays prepared), copy the permanent (no copied preparation), naming Seed Suture, and Codie's copied spell retargeting. |

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data. Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

Gate owns the full debug workspace, fmt, Clippy and consumer suites.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, then an Adventure and a linked prepare copy through exile access. One permission owner and one casting owner must explain all four, including airbend, flashback and modal regressions. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant.
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
