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
no-commander behavior (Commander/multiplayer is a demo question); the shipping
guide (`docs/engine.md#tla-decks-with-tle-bonus-sheet-cards`) keeps its
planned-support wording until gate/demo; open world w17; frozen evidence unchanged.
Decisions and run evidence are in `scratch/questions.md`.

## What exists

`uv run python scripts/check_tle_coverage.py --bonus-sheet`: 52 of 61 defined,
9 missing, no other issues (rebuilt extension, tenth pass). `--allocation-only` checks that
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

Twelve non-bonus definitions written before the steer are parked, uncommitted,
on local branch `mtg-286-nonbonus-batch3-parked` for the later Task.

## Remaining: 9 bonus-sheet names by rule family

| Family | Cards |
|---|---|
| Production/spending (blocked, review) | Heartbeat of Spring, The Great Henge, Tarnished Citadel |
| Token identity (blocked, decision) | Release to Memory; Lita, Mechanical Engineer |
| Observed turn history (blocked, layout) | Bloodchief Ascension (life lost this turn), Brought Back (left the battlefield this turn) |
| Restricted mana shape (blocked, layout) | Cityscape Leveler (Powerstone; its cast/attack trigger and unearth are otherwise ready) |
| Phasing and life lock (contract written) | Teferi's Protection |

Eight are blocked. Teferi's Protection is unblocked: implement the contract
below.

Known deviation in a defined card (tenth pass): activated abilities never offer
their own source as a target, so neither player can choose Diaochan herself,
although the card allows it (the opponent's clause makes this strategic). It
predates this branch; per the playable-not-bug-free rule it does not block the
card. It needs a fix here or a bug Task before gate.

## Teferi's Protection contract (proposed, tenth pass)

Every new state is an `Effect::ApplyContinuous` record, per the duration rule,
so v9's shared-effect rows (header, details, one-based members) carry it with
no new column or fact kind:

- `EffectDuration::UntilYourNextTurn`: ends as the record controller's next
  turn begins (before the untap step's phasing, CR 502.1).
- Player records on the caster: `Modification::LifeTotalCantChange` (life
  gain, loss, damage-as-loss and setting are no-ops; life costs other than 0
  cannot be paid, CR 119.8) and `Modification::ProtectionFromEverything`
  (CR 702.16j: cannot be targeted, all damage to them prevented).
- Phasing (CR 702.26): `Modification::PhasedOut` on each permanent the caster
  controls, plus anything attached to them (phased out indirectly, 702.26g),
  ending when its controller's untap step phases it in (702.26a). No zone
  change, no triggers, counters and attachments kept (702.26d-e).
- One authority: `Game::is_phased_out(id)` reads the effect index by object;
  the battlefield iterators (`iter_battlefield_permanents`,
  `battlefield_permanents`, `layer_views` scans, trigger index, SBAs, combat,
  target legality, mana sources, counts) skip phased-out permanents. Permanent
  rows stay in the observation (public), marked by the effect membership; the
  LLM text names the record; Academy notes it.
- Card: instant {2}{W}, `c17.rs`, effects = the two player records, the
  phase-out record over `each_you_control`, then `Move This` to exile.
- Directed tests: the caster's life stays put against Lightning Bolt and a
  life payment is refused; their creatures are not attackable/targetable/
  counted (Volcanic Torrent misses them), and phase in, before untapping, on
  the caster's next untap step with counters kept; no ETB triggers fire.

## Token identity (decision open)

`managym/src/cards/content_pack.rs` registers every definition under a unique
name (`cards_by_name`), and `token.spirit` is the TLA Spirit with its blocking
restriction. Release to Memory needs a plain 1/1 colorless Spirit. Lita's
Zeppelin has no token printing on Scryfall, so it has no `scryfall_id`. Two
decisions are open:
(a) whether tokens resolve by key, so that two tokens can share a printed name,
or the content pack stays keyed by name;
(b) which `scryfall_id` a token without a printing carries.
These block only those two cards.

## Production boundary (open review)

Heartbeat of Spring needs the land's actual produced types and either player's
bonus sources; `state/card.rs::TriggeredManaAbility` holds a fixed `Mana`,
`flow/zones.rs::tap_bonus_sources` scans only the tapping player, bonuses fire at
tap before an output exists, and `Decision::ManaChoice` drops the source. Great
Henge's life gain and Citadel's damage run inside the mana ability, so an
evaluator that accepts only a final AddMana cannot discard those suffixes.
`scratch/tle-production-design.md` proposes the replacement (production
occurrences, actual-output triggers, canonical mana classes, spending receipts).
Its event timing/source snapshots and v10 fact/action table need authored review
before any writer; open in `scratch/questions.md`. It blocks only these three cards.

## Shared observation contract

New state reaches native scalar/vector and Python encoders, semantic capture,
LLM text and learning tokens in the change that adds it; hashed names go in
`agent/record_vocabulary.rs`; open vocabularies are fact rows. Main carries only
v9; do not widen it or add version branches. A layout that v9 cannot express
waits for a world freeze (the production review's v10 question).

## Acceptance checks (gate)

- Bonus-sheet coverage reports 61/61; every name has a unique key,
  first-printing Scryfall id, learning tokens and a directed debug test. Debug
  Rust suite (`cargo nextest --profile ci`), conformance, observability, fmt, clippy.
- Regenerate `conformance/semantic-kernel-w17-v2` evidence: the pack
  `content_digest` moves with every added definition, so
  `checked_semantic_conformance_evidence_reproduces` fails until it is
  regenerated (eighth-pass compress run: 1131 passed, this one failed). The
  Academy precision demos in `tests/academy/test_decision_surface.py` pin a
  content/setup identity that moves the same way (three fail, ninth pass).
- World-change Python suites from `CLAUDE.md` and
  `docs/development.md#validation-by-change-area`; Academy pending choices,
  stale Commands, reconnect and replay/fork for each new decision. Browser
  checks headless after frontend build, isolated ports.
- Fetch the pinned corpus via `manabot.data` with R2 credentials from Doppler;
  admit all `Corpus.trophies[*].decklist` of the first 1,000 events through
  `managym.admit_decklists` (not `Corpus.eligible` or saved `Trophy.deck`).
  Record input digest and per-event results: 1,000 passes, no dropped cards.
  Publish before tracking any receipt under `experiments/data/`.
- Run the shipping guide's Scout's Warning example on the rebuilt engine; then
  replace its planned-support wording. CI's Engine Budget covers speed.
