# The 61 TLE bonus-sheet cards in the development world

## Outcome and scope

Jack Heart: "lets file a task for covering all of TLE", then (Task comment,
2026-11-12): "Ok, we should start with only the TLE cards that are bonus sheet
TLA cards" and "no need to remove anything already included though". The same
comment: "All 41 names missing from the 1,000-event corpus are in this list" and
"Keep everything already built (Cabbage Merchant payment model, the 9 non-bonus
TLE cards, the 242-name ledger); don't remove or revert it, but don't extend
non-bonus TLE cards further. The other ~180 non-bonus TLE cards are later work".

Scope: the 61 booster printings pinned in
`managym/cards/tests/fixtures/tle-bonus-sheet.jsonl` (TLE collector numbers 1–61).
Each is one function in its first-printing set file with a Scryfall id, a
directed debug test, observability guards and learning tokens; new mechanics are
new rules-type variants with real card consumers in the same change.

Acceptance: all of the first 1,000 events in
`experiments/data/mtg213/corpus-v1/corpus.json` (manifest SHA256
`22f9b2b9411a69a85ff1270605dd68fba5941601859b752d2a4a0ecf5e4a6043`, 11,034,908
bytes; baseline 922 admitted, not independently reproduced) admit with their real
main decks and sideboards, without substituting names, weakening printed
behavior or construction rules, or moving split assignments. Expanding MTG-282's
saved corpus is subsequent work.

Retained assumptions: the current 1v1 development format with correct
no-commander behavior (Commander/multiplayer is a demo question); open world w17; frozen evidence unchanged.
Decisions and run evidence are in `scratch/questions.md`.

## What exists

`uv run python scripts/check_tle_coverage.py --bonus-sheet`: 58 of 61 defined;
missing only Release to Memory, Lita and Heartbeat of Spring (rebuilt
extension, thirteenth pass). `--allocation-only` checks that
`managym/cards/tests/fixtures/tle-allocation.jsonl` binds all 411 Oracle
paragraphs of `tle-inventory.jsonl`; `tests/infra/test_tle_allocation.py`
protects it. The ledger is planning data, not a behavior proof.

Rules added on this branch, each with directed tests and observation parity
(evidence in `scratch/questions.md`):

- Counted tap costs and `flow/mana.rs::PaymentResources` (The Cabbage Merchant):
  scalar/vector/Academy reconnect, fork and stale-Command proofs
  (`tests/academy/test_merchant_payment_recovery.py`,
  `tests/verify/test_observation_parity.py`). The one-use-per-source planner,
  its bounds and the old finite-source check are deleted.
- -1/-1 counters with CR 704.5q annihilation (Black Sun's Zenith), as counter-kind
  fact rows.
- `Modification::NextCreatureCastFlash` player record (Scout's Warning).
- Same-library `Move` with a stated end reorders without a zone change (Cruel Tutor).
- `Condition::AnyGraveyardAtLeast` (Visions of Beyond);
  `PlayerSelector::TriggeringPlayersOpponents` (Standstill).
- Sheep creature subtype.
- Seventh pass (14 cards; evidence in `scratch/questions.md`): Human token;
  `StaticAbility::CostsLessPerPermanentOnBattlefield`, `CantBeCountered`,
  `AdditionalLandPlayEachPlayer`, `OpponentsCastOnlyFromHand`;
  `Condition::ControlsUnownedAtLeast`, `CardsInHandAtLeast`;
  `EffectValue::GreatestPowerYouControl { predicate }`, `TargetToughness`
  (LKI now keeps toughness); `DealDamage { by }` names the damage source;
  `Destroy` answers `Condition::Moved`; `CardPredicate::colors_any`;
  `TriggerCondition::BeginningOfEachDrawStep`/`BeginningOfEachEndStep` with the
  active player as "that player"; an earlier incarnation of a real card is a
  known damage source (a resolved spell's delayed trigger);
  `Modification::SkipsCombatPhasesNextTurn` player record (Empty City Ruse),
  skipped in `tick.rs`, ended at that player's cleanup, seen as a shared-effect
  row with LLM text and Academy note.

- Eighth pass (6 cards): activation prohibition and must-be-blocked-by-all
  records; `ActivatedAbilitiesCostLess` (floor one); `YouCastThis` cast
  triggers; `Effect::Cascade` (random bottom order); `SpellsCastThisTurn`;
  `CumulativeUpkeep`; Age/Ice counters; `EntersWithCounters`; the `NoCounters`
  state trigger; Koma's Coil and Marit Lage. Free casts from exile or library
  now respect hand-only casting. Detail in `scratch/questions.md`.

- Ninth pass (6 cards): `CreateTokenCopy { of: ObjectSelector }`;
  `SearchDistinctNames` with predicate and end (replaces
  `SearchDistinctLands`); `Effect::EndTheTurn`; `SpellTargetsOnlyThis` and
  `CopyTriggeringSpellForEach`; the exile-from-hand alternative cost with its
  payment choice and Academy proof; `TargetSpec::SpellMatching`,
  `CounterSpell { exile }`; Elf Warrior token. Detail in `scratch/questions.md`.

- Tenth pass (5 cards): `AdditionalCost::Replicate` (count announced as X);
  `ObjectSelector::TargetPlayersZone` and `Game::inspects_library` (Bribery);
  `ReplacementEffect::EntersAsCopy` with `Decision::EnterAsCopy` (code 30),
  printed row restored on leaving; copiable `GrantStatic` exceptions and
  `StaticAbility::LegendRuleDoesntApply` (Sakashima); several activation
  target clauses, `TargetRequirement::opponent_chooses`,
  `before_attackers` (Diaochan). Detail in `scratch/questions.md`.

- Eleventh pass (Teferi's Protection, new `c17.rs`): `EffectDuration::
  UntilYourNextTurn` (shared-effect duration code 4); player records
  `LifeTotalCantChange` (`Game::payable_life` refuses life payments) and
  `ProtectionFromEverything`; `Modification::PhasedOut` with
  `Game::is_phased_out` as the one reader. Detail in `scratch/questions.md`.

- Twelfth pass (The Great Henge, Tarnished Citadel): `mana_choices` accepts
  a leading output plus an untargeted non-mana suffix
  (`ActivatedAbilityDefinition::mana_suffix`), run once in the `flow/mana.rs`
  child frame with the sacrifice cost, before the output choice; damage names
  the land and is preventable. `StaticAbility::CostsLessByGreatestPower`.
  Academy proof `tests/academy/test_citadel_payment_recovery.py`.

- Thirteenth pass (Bloodchief Ascension zen, Brought Back m20, Cityscape
  Leveler bro, Powerstone token): `PlayerTurnHistory.life_lost` (counted in
  `emit` from every `LifeChanged`), `TurnState::graveyard_from_battlefield`
  (exact graveyard incarnations), `Condition::OpponentLostLifeThisTurnAtLeast`,
  `TriggerCondition::CardPutIntoOpponentsGraveyard` ("that player" = graveyard
  owner for CardMoved-to-graveyard), `CardPredicate::from_battlefield_this_turn`,
  unearth as a graveyard activation with `Modification::ExileIfWouldLeave`
  (`normalize_proposal`) and a remembered end-step exile, `CreateToken.tapped`,
  Powerstone mana as `ManaUse::except_nonartifact_spells` (broadest restricted
  class) against a derived, unserialized `casting_nonartifact` spell purpose.
  Policy input v10 built (layout in `docs/observation-and-model.md`).

Twelve non-bonus definitions written before the steer are parked, uncommitted,
on local branch `mtg-286-nonbonus-batch3-parked` for the later Task.

## Remaining: 3 bonus-sheet names (decisions 2026-11-12, `scratch/questions.md`)

| Cards | Disposition |
|---|---|
| Release to Memory; Lita, Mechanical Engineer | Wait for MTG-298 (TokenDefinition, tokens by key, scryfall_id optional only on tokens); allocation rows marked blocked on it |
| Heartbeat of Spring | Split out to MTG-297; don't implement `scratch/tle-production-design.md` here |

Gate on 58 of 61. Corpus admission reports the exact count of events failing
on Release to Memory, Lita or Heartbeat of Spring, and lists any other failure.

Known deviation in a defined card: activated abilities never offer their own
source as a target, so neither player can choose Diaochan herself. Filed as
bug Task MTG-296: the fix (exclude only `TargetRequirement.another`) moves
retained replays (`drill-270` shared-effect transport, T3 observation parity),
so it is not local to this branch.

## Shipping docs (written 2026-11-12)

- `docs/engine.md#tla-decks-with-tle-bonus-sheet-cards`: 58 of 61, the three
  undefined names and what each waits for, the v10 pointer.
- `docs/observation-and-model.md#input-v10-turn-history-and-powerstone-mana`:
  selecting v10, what it adds, v9 refusal. Open questions on v10 beside v9
  are in `scratch/questions.md`.

## Production boundary (split out to MTG-297): Heartbeat of Spring

Heartbeat needs the land's actual produced types and either player's bonus
sources; `state/card.rs::TriggeredManaAbility` holds a fixed `Mana`,
`flow/zones.rs::tap_bonus_sources` scans only the tapping player, bonuses fire
at tap before an output exists, and `Decision::ManaChoice` drops the source.
`scratch/tle-production-design.md` proposes the replacement. Its event
timing/source snapshots and v10 fact/action table need authored review before
any writer; open in `scratch/questions.md`. It blocks only Heartbeat (plus
non-bonus Toph, Uncle's Musings, Animal Attendant).

## Shared observation contract

New state reaches native scalar/vector and Python encoders, semantic capture,
LLM text and learning tokens in the change that adds it; hashed names go in
`agent/record_vocabulary.rs`; open vocabularies are fact rows. Main carries v9;
v10 is explicit, and v9 refuses positions needing v10 state before writing
buffers (open question on the one-input rule in `scratch/questions.md`).

## v10 spells cast and named layout (fourteenth pass, done)

`PlayerData.spells_cast` (from `TurnState.spells_cast`, the one owner) is v10
`player_turn_state` column 1 (/ 10): row 2 × 8 (life lost, spells cast, six
Powerstone-class mana), columns named by `observation_layout::turn_state` and
published by `managym.observation_layout(*, policy_history, turn_state)`, in
native scalar/vector and Python encoders, LLM text ("cast N spells") and the
guard. Academy `LegacyPlayerView` gains `spells_cast` and the missing
`life_lost` (Rust, Python, projection, five schemas, TS, `PlayerArea`). V9 adds
no refusal (reversible assumption, `scratch/questions.md`).

Result: debug cards 684 passed; managym encoder/schema/protocol 16 passed;
pytest named-layout, input-v10, parity, text observation, experience protocol
passed; cp313 rebuilt atomically. Academy vitest/svelte-check not run (no
`node_modules`).

## QA result (2026-11-12, `scratch/qa.html`)

Corpus admission: 995/1000; the 5 failures are Heartbeat of Spring (3) and
Lita (2); no dropped cards. Admit `Decklist(t.decklist.source, t.decklist.filename)`:
the saved `Trophy.decklist` keeps the old world's issues (134/1000). QA fixed
{X} text capture (`ManaCost.x`) and the Academy `CardState` conformance failure.

## Acceptance checks (gate)

- Bonus-sheet coverage reports 58/61 (the three waiting names missing); every name has a unique key,
  first-printing Scryfall id, learning tokens and a directed debug test. Debug
  Rust suite (`cargo nextest --profile ci`), conformance, observability, fmt, clippy.
- Regenerate `conformance/semantic-kernel-w17-v2` evidence: the pack
  `content_digest` moves with every added definition, so
  `checked_semantic_conformance_evidence_reproduces` fails until it is
  regenerated (eleventh pass: 1148 passed, 2 skipped, this one failed). The
  Academy precision demos in `tests/academy/test_decision_surface.py` pin a
  content/setup identity that moves the same way (three fail, ninth pass).
- World-change Python suites from `CLAUDE.md` and
  `docs/development.md#validation-by-change-area`; Academy pending choices,
  stale Commands, reconnect and replay/fork for each new decision. Browser
  checks headless after frontend build, isolated ports.
  Includes Academy vitest (`permanent-notes.test.ts` gained phased-out and
  life-lock cases, not yet run) and reconnect/fork with phased-out rows.
- Fetch the pinned corpus via `manabot.data` with R2 credentials from Doppler;
  admit all `Corpus.trophies[*].decklist` of the first 1,000 events through
  `managym.admit_decklists` (not `Corpus.eligible` or saved `Trophy.deck`).
  Record input digest and per-event results: 1,000 passes, no dropped cards.
  Publish before tracking any receipt under `experiments/data/`.
- The guide's Scout's Warning example prints `[[]]` on the thirteenth-pass
  extension (wording now present tense); its "expected to fail" corpus
  sentence awaits the admission run. CI's Engine Budget covers speed.

## Gate result (2026-11-12)

Fixed in gate: clippy (`-D warnings`, local 1.98) on branch code — boxed the
`CardPredicate` payloads of `PowerCda::GraveyardMatching`,
`DelayedEvent::YouCastSpell` and `DelayedTriggerEvent::YouCastSpell` (serde
output unchanged), `checked_div` for counted tap uses, a useless `into_iter`,
`as_chunks` in two TLE tests; an unused parity-test import; the coverage
script reads `catalog_tokens`. `test_equal_rule_names_learn_opposite_public_preferences`
pairs Gray Ogre with Glory Seeker, since new names moved Hill Giant to the
next name group. Re-recorded the content-bound evidence per
`conformance/README.md`: semantic-kernel w17-v2 (14 cases, identical command
counts and outcomes, new digest `f90667d4…`), Academy fixtures (offer labels
replayed unchanged), authored-match receipt (same 135 decisions; new
`AUTHORITY_SHA256`/`TERMINAL_STATE` pins), both parity receipts, belief v3 pins.

Checks (current tree): cargo fmt clean; clippy all targets/features clean;
`cargo nextest run --workspace --profile ci` 1164 passed, 2 skipped; arena
admission/monitoring suite 65 passed; model, belief, Learn setup, choice
navigation, decision surface, authored/public parity, text observation and
parity 506 passed except below; env/agent/llm/action cutoff/active training
352 passed; protocol, trigger order, forced decisions, shared-effect transport,
record coverage, infra 180 passed; Academy svelte-check 0 errors, vitest 127
passed; `--bonus-sheet` 58/61, `--allocation-only` clean. Engine Budget and
Playwright release proofs left to CI.

Not passing, not in CI: `tests/belief/test_learning.py::test_real_population_learning_is_held_out_and_beats_p0`.
Passes at base f95949149 (held-out NLL improvement +0.126 nats); here −0.046.
The belief schema's card vocabulary is every world definition (360 → 460), so
the 16-step toy model grows from 2,980 to 3,780 parameters and its seeded
result moves; population and supervision are identical. Not retuned here.
