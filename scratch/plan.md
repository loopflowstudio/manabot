# MTG-419 · Shared casting profiles and play permissions

Task steering, 2026-11-24: “Continue through straightforward cards without returning to review between cards. Realign and review the cohort together. Split only at a concrete shared-rules dependency or genuinely distinct implementation boundary; record that reason and the remaining rows.”

The Task brief requires “one shared casting-profile and play-permission model” and says “Preserve complete legal choices, observations, learning tokens, LLM text, replay/forks, and the 10% Engine Budget.” It also says to “Leave scoped event-history and contextual-query extensions to the history/queries task (defer cards strictly requiring new turn-history counters).” That task is MTG-418, which is open and not merged.

## Shared casting model

A play chooses an exact object, a printed profile and a legal route. The
profile supplies the spell's characteristics, targets and printed costs. The
route supplies permission, timing and any alternative price. Additional costs
apply to the selected combination.

- `ContentPack::spell_definition` resolves the front, Adventure, Omen or modal back for the physical object. Priority and resolution offers use it before filtering. Announcement, payment and targeting use the same definition through `CastChoices::spell`.
- A modal back enters the battlefield back face up. Moving it to another zone restores the front. An Adventure or Omen instead follows its own rule on leaving the stack.
- `PlayRoute` distinguishes ordinary hand casts, free hand casts, top-of-library permission, printed or granted flashback, exact-object grants and Muldrotha's per-type permission. Exact grants separately record who may cast, their duration, preparation and flashback. A land's chosen route determines whether it may be played and who controls it on entry.
- Omniscience replaces the mana cost. A free cast still permits kicker and bargain and requires mandatory additional costs. Harbinger's surcharge buys flash when required. An independent source of flash leaves the surcharge optional.
- Compound casts bind the chosen target with its spell profile and permission. `ContinuationFrame` records the selected profile during announcement. `DecisionContextView` names the spell the acting player chose. The v11 action row carries a single spell-profile category.

## Delivered

The per-set ledgers link each implemented row to its definition and native
fixture. The shared authority first shipped on this branch with Zul Ashur,
Muldrotha, Vizier, Etali, Smaug, Bilbo, Twinmaw and Scavenger Regent. The
first 44 preparation rows include linked copies, source transfers, preparation
during resolution and Emeritus of Truce.

- Yue offers Spew Flame from hand and Solstice Revelations offers Charring Bite from exile. Evaluating the card's front face before enumerating its spell profiles suppressed both legal casts. `f3d84a7abe` moved the restrictions into the per-profile check. The two fixtures in `tla.rs` check the legal offer, the selected spell in the caster's announcement frame and decision context, and the spell's resolution and departure.
- Monica Rambeau and Photon use one front/back definition pair. Photon pays its printed cost, enters back face up and returns to Monica on leaving. A compound cast binds its own face, permission and payable target. The compact priority API retains its recorded candidate addresses.
- The Foundations fixtures cover 11 of 17 rows. Omniscience supplies an alternative hand route. A free Burst Lightning still allows manual spending on kicker. High Fae Trickster supplies timing. Harbinger's surcharge and required sacrifices still apply on a free cast. Sphinx adds an exact-object flashback route alongside Think Twice's printed route. Reassembling Skeleton and Suspicious Shambler exercise graveyard activation and exact source identity.
- Harmonized Trio and Joined Researchers bring preparation to 46 of 62. Their fixtures exercise the complete tap cost, choosing the library order, intervening-if checks and the two players drawing.
- The HOB fixtures cover 12 of 17 Adventures plus Bilbo's separate permission. The TDM fixtures cover 12 of 13 Omens. They exercise both the permanent and the inset. Feral Deathgorger binds both targets to the same graveyard. Till and Tend stacks extra land plays until cleanup. Multiple Omnisciences grant one free route.

## Remaining shared rules

The 31 rows in `be46a3fc8c` close the batch expressible with the casting
profiles, permissions, costs and targeting implemented here. Twenty-two
MTG-419 rows remain across six Foundations cards, five HOB cards, one TDM
card and ten preparation cards.
For example, Claim Territory needs both found cards revealed while the
caster chooses which one enters tapped. The existing library search reveals
a card as it moves to hand, too late for that decision. These rows split at
the shared rules listed below. Six further preparation rows belong to MTG-418.

| Cohort | Cards | Shared rule still needed |
|---|---|---|
| HOB | An Unexpected Party | Choose and retain a creature type as the permanent enters. |
| HOB | Bilbo, Luckwearer | Target two permanents sharing a card type and exchange control only if both sides can change. |
| HOB | Great Ugly-Looking Goblin | Amass handles both creating an Army and choosing an existing Army. |
| HOB | Gandalf, Goblins' Bane | Privately inspect face-down exiled cards and grant conditional play while the caster controls a Wizard. |
| HOB | Glamdring, Foe-hammer | Price an instant or sorcery from its controller's equipped creature's current power. |
| TDM | Bloomvine Regent | Reveal the cards found in a single search, choose one to enter tapped and put the other in hand. |
| Foundations | Tinybones, Bauble Burglar | Discard events and counters on exact exile appearances. A second Tinybones uses the same counter marks, including on foreign lands. |
| Foundations | Loot, Exuberant Explorer | Controller-specific land allowance and private look-six selection whose mana-value limit is the land count at resolution. |
| Foundations | Flamewake Phoenix and Gate Colossus | Identify the graveyard source of a beginning-of-combat or land-entry trigger and retain that appearance through an optional return. |
| Foundations | Quilled Greatwurm | Pay a graveyard casting cost by removing a distributed total of counters from creatures. |
| Foundations | Redcap Gutter-Dweller | Commit the optional sacrifice before the counter and impulse permission. Create the printed Rats with their blocking restriction. |
| Preparation | Spiritcall Enthusiast and Kirol, History Buff | Trigger once per batch of matching entries or graveyard departures. `process_event_batch` currently matches each movement separately. |
| Preparation | Grave Researcher | Surveil before the graveyard count. Charge life from the reanimated object's mana value. |
| Preparation | Emeritus of Ideation and Lluwen, Exchange Student | Distinct graveyard cards as actual costs for an attack trigger or a sorcery-speed activation. Lluwen also creates the printed Pest. |
| Preparation | Variable Chaser | Separate optional whole-hand discards and draws for the two players in order. |
| Preparation | Pyre Rhymer | A turn-scoped mana trigger applies to each Mountain its controller taps for mana, including later arrivals. |
| Preparation | Heartwood Crafter | Restricted mana reads the spell's casting zone in the spending context. |
| Preparation | Sanar, Unfinished Genius | Gate Treasure activation on an instant or sorcery actually cast this turn. Use the cast appearances in `TurnState.spells_cast`, as the first-spell discount already does in `flow/play.rs`. |
| Preparation | Codie, Ravenous Codex | Preserve prepared-cast provenance through announcement and the stack. Copy and retarget the cast spell without granting its copies that provenance. |

**Next slice.** Build the casting-provenance cohort with Heartwood Crafter
and Codie. Use the same linked prepared spell cast from exile as the positive
case for restricted mana and Codie's copy trigger. Fork to an ordinary hand
spell for the spending restriction, then choose separate targets for the
original and stack copy and check that the copy cannot trigger Codie. Pay
through the normal announcement and payment path and restore a pending
Command across replay. Update the FRA ledger after the native fixtures pass.

MTG-418 owns six history-dependent rows. Emeritus of Woe and Bloodline
Recollector need creature deaths this turn. Leech Collector needs the first
life gain, Scheming Silvertongue the total life gained, Whiplash Wordsmith
noncombat damage to an opponent, and Hexhaven Dueling Arena the specific
creature that attacked. The unblock is MTG-418's merged history and query
authority with these facts available to trigger conditions. On that merged
revision, start with
`cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution`
as the preparation control before adding the six directed fixtures.

## Interaction evidence still owed

| Position | Required fork or outcome |
|---|---|
| Tinybones, opposing discarded land and spell with stash counters, second Tinybones | The current controller plays foreign cards that an earlier Tinybones exiled. Check own-turn timing, foreign land control (enters under Tinybones’s controller, not owner), ownership versus control, mana spending, High Fae Trickster non-bypass of own-turn timing, and a hand-only prohibition (blocks spell, allows land). Use Vizier's private top card and foretell's hidden exile as opposite-viewer controls. |

## Validation

**Consumer checks per slice.** Compare native observations for both viewers with the Python bindings, semantic capture/LLM text and Academy offers at priority, at mid-announcement (`AnnounceX`/`ChooseTargets`/`Pay`), with a spell on the stack, and at a nested decision. Use v11 rows and learning tokens. Keep hidden information out of the other viewer's data. Fork-and-replay checks need exact preconditions and stale rejection. Existing controls:

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards airbend_permission
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards flashback_is_exiled_when_countered_after_iroh_leaves
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards sozins_comet_foretell_spending_choices_preserve_color_and_private_identity
cargo test --manifest-path managym/Cargo.toml -p managym --lib preparation_resolution
uv run pytest tests/semantic/test_text_observation.py tests/academy/test_choice_navigation.py tests/model/test_pinned_pilot.py -q
```

The 31 new ledger rows passed their directed checks in a 148-test native
card/observation run. Another ten structured-offer tests, five preparation
resolution tests, the execution vocabulary test and the real-card Scryfall
check passed. Clone Legion copied a prepared creature without duplicating
its cast, and Honest Work removed its abilities while its linked spell
remained castable. Seed Suture was chosen through ordinary naming Commands
and the choice replayed from the same root. The five Academy navigation positions retained all 252 Commands
and offers. The authored match retained its 193 Commands, revealed cards,
costs, terminal board and both presentation tracks. Live, headless and replay
agreed at 194 checkpoints across 1,164 viewer projections; stale object and
stale revision Commands were rejected. The casting-catalog receipts are
stored alongside the original opening receipts.

MTG-422's opening and terminal procedure is already in the branch
(`d12eb496b3`). The later upstream sync at `edee24dad5` changed native snapshot
and replay plumbing. Eight focused Python checks on that merge passed,
covering semantic casting profiles and cold restores for the pending Adventure,
Exude Toxin, Etali, transferred Seed Suture and Waypoint. The native `offers_`
filter passed 11 tests including both nested announcements and the manual
Omniscience payment.

Gate owns the full debug workspace, Clippy and the measured 10% Engine Budget.

## Later phases

**Decomposition.** Follow Muldrotha and Zul from the same graveyard object through payment, Omniscience and Harbinger from hand, Tinybones from counter-marked exile, and an Adventure, Modal BackFace, and linked prepare copy through stack and departure. One permission owner (`PlayRoute`) and one profile owner (`SpellProfile`) must explain all of them, including airbend, flashback and modal regressions. Priority currently derives `CastChoices.free` and `flashback` from the route, while resolution casts set them from the effect. When adding provenance, check whether one cast-choice constructor can derive that treatment for both paths before eligibility and announcement. Remove any exclusive old paths. Review repeated FRA spell programs for card-specific dispatch.

**Performance.** Use the 10% same-runner Engine Budget and the [stress-state protocol](../docs/development.md#stress-states). Add `bench_states` shapes for:
- Muldrotha with a graveyard of duplicate artifact creatures, overlapped by a Zul grant.
- A hand of Dragon/Omen cards with restricted mana and several Charring Bite targets.
- Many Phytomedics with their own Seed Sutures, with one transfer and a zero-link control.

This is where to decide whether the linear grant scan needs an index. Measure first.
