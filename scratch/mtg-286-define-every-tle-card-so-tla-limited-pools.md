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

`uv run python scripts/check_tle_coverage.py --bonus-sheet`: 34 of 61 defined,
27 missing, no other issues (rebuilt extension, seventh pass). `--allocation-only` checks that
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
- Seventh pass (13 cards; evidence in `scratch/questions.md`): Human token;
  `StaticAbility::CostsLessPerPermanentOnBattlefield`, `CantBeCountered`,
  `AdditionalLandPlayEachPlayer`, `OpponentsCastOnlyFromHand`;
  `Condition::ControlsUnownedAtLeast`, `CardsInHandAtLeast`;
  `EffectValue::GreatestPowerYouControl { predicate }`, `TargetToughness`
  (LKI now keeps toughness); `DealDamage { by }` names the damage source;
  `Destroy` answers `Condition::Moved`; `CardPredicate::colors_any`;
  `TriggerCondition::BeginningOfEachDrawStep`/`BeginningOfEachEndStep` with the
  active player as "that player"; an earlier incarnation of a real card is a
  known damage source (a resolved spell's delayed trigger).

Twelve non-bonus definitions written before the steer are parked, uncommitted,
on local branch `mtg-286-nonbonus-batch3-parked` for the later Task.

## Remaining: 27 bonus-sheet names by rule family

Grouping from the allocation ledger; each family lands with its cards.

| Family | Cards |
|---|---|
| Production/spending (blocked, below) | Heartbeat of Spring, The Great Henge, Tarnished Citadel |
| Copy entry, token copies, spell copies | Clone, Clone Legion, Sakashima of a Thousand Faces, Rhys the Redeemed, Mirrorwing Dragon |
| Cascade (new variant; keep `ExileUntilNonlandCastFree`) | Volcanic Torrent |
| End the turn (turn/stack owner, cleanup, surviving delayed triggers) | Sundial of the Infinite |
| Phasing and life lock (object presence, not exile; attachments, no leave/enter triggers) | Teferi's Protection |
| Combat skip and requirements | Empty City Ruse, Taunting Challenge |
| Counter and zone-history triggers | Bloodchief Ascension, Dark Depths, Brought Back |
| Cast-time costs (alternate exile cost, replicate, unearth, cumulative upkeep) | Force of Negation, Shattering Spree, Cityscape Leveler, Mystic Remora |
| Scoped costs | Training Grounds (activated-cost reduction, floor one) |
| Token identity (open, `scratch/questions.md`) | Release to Memory (plain Spirit vs the TLA Spirit name), Lita, Mechanical Engineer (no Zeppelin token printing) |
| Modal activation, activation prohibition, Coil token | Koma, Cosmos Serpent (uncounterable now exists) |
| Selection constraints and chooser/owner separation | Three Dreams (distinct names), Bribery (opponent's library, your battlefield) |
| Split target choosers | Diaochan, Artful Beauty |

Cascade test cases: continue past expensive nonlands, decline, no qualifying
card, illegal targets, random bottom order after the cast decision, replay/fork at
each suspended choice. End-turn and phasing need their state/observation
contracts written before the writer. Non-bonus families in the ledger (player
experience counters, the Masters, Toph, Uncle's Musings, Animal Attendant) are
outside this Task.

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
