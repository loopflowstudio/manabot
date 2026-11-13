<!-- Draft body for PR #491. Keep the loopflow task-pr-context block the PR already has. -->

The engine marked tokens and copies of spells with one `is_token` flag stored on the object's definition, and called every runtime object a `Card`. Each game object now has a kind taken from the Comprehensive Rules (`ObjectKind::Card`, `Token` or `SpellCopy`), and the runtime record is named for what it is: `Object`, named by `ObjectId`. Card-only consumers narrow through `GameState::card_id` and carry `CardId`, so a token or a copy of a spell cannot enter those operations.

This builds on [MTG-298 · Give tokens their own `TokenDefinition` type](https://github.com/loopflowstudio/manabot/pull/476), which split token definitions from card definitions and has merged into main.

## What changes

- **Checked card ids.** `CardId` has a private index and is created only after checking the object kind. Deck/sideboard rosters and card abilities carry it; card-only candidates and play/cost entry points use it. Conversion to `ObjectId` is free, indexing asserts the kind in debug, and serialization retains the same numeric row.
- **Kind.** The runtime record has a private `kind`, set once when the object is made. A copy of a permanent spell becomes a token as it enters the battlefield (`resolve_into_token`, CR 111.12, 608.3f); nothing else changes a kind.
- **Readers moved to the kind.** State-based actions (a token off the battlefield and a spell copy off the stack cease to exist, CR 704.5d-e), predicate and layer `token` filters, exile cast grants, "choose a card" selection, the return-to-battlefield guard, and `TriggerCondition::PutIntoGraveyard` (`moved.is_card()`). `become_copy_of` no longer saves and restores a token flag. The remaining `is_token` reads are catalog rows, deck and sideboard admission, the observation column and the hash's wire bytes.
- **Rename.** `Card` -> `Object` (in `state::game_object`), `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`, `GameState::cards` -> `objects`, `card_to_permanent` -> `object_to_permanent`. The old u32 `ObjectId` (the match-unique public number for players, objects and abilities) is now `PublicId`. Helpers that move or match any object follow: `move_object`, `move_object_with_entry`, `capture_object_move`, `subject_matches_moved_object`, `remove_object`, `object_zones`, `matches_object`, `object_power`, `object_mana_value`.
- **`card` stays where the rules mean a card:** card definitions, decklists, sideboards, checked card selection helpers (`library_top_card`, `selection_zone_cards`, `foretold_cards`, ...), draw rules, exact graveyard targets, and the observation encoders and Python bindings, whose names are published.
- **Docs.** `docs/engine.md` has a new section, [Objects and zones](docs/engine.md#objects-and-zones): one Spirit token through a turn, the CR term -> engine type table, what is built, the planned split, and where the engine departs from the rules.

## Decisions a reviewer should check

- **Library and hand store checked cards plus explicit visitors.** One ordered `CardZoneEntry` sequence stores `Card(CardId)` or `Transient(ObjectId)`. Tokens and spell copies retain their position until state-based actions; there is no parallel membership or ordering list. Borrowed reads and serialization project the original numeric ids, while card-only reads use stored checked ids. Reordering preserves entry kinds, and shuffling the full sequence preserves RNG consumption. This replaces the earlier all-ObjectId fallback.
- **Graveyard and exile still store `ObjectId`.** They hold the same contents as library and hand (cards plus transient non-cards), but only library and hand record the kind per entry; card-only readers of graveyard and exile narrow through `GameState::zone_cards` on each read. Typing them the same way would give the four card zones one shape, at twice the entry width in state that search clones. Not done here.
- **Byte compatibility remains an acceptance requirement.** Serde keeps the old field names (`#[serde(rename = "card")]`). Observation `is_token` is `!is_card()`, and the match-state hash records a non-card object with `is_token: true`, as before. The earlier rename reproduced `semantic-kernel-w17-v2`; Semantic Kernel Conformance passed on 8e5e8e62b; final seeded QA remains. `Debug` output of the renamed types does change.
- **Card-only choices.** Transient tokens stay visible in their zones but are excluded from card-only play, activation, discard and search choices. `SelectCards` narrows to cards only when it reads a zone or milled cards; `CreatedThisWay` still selects the new token (Kyoshi Battle Fan). Direct card-only commands reject them too. Generic movement, inspection and draw ordering still include them. Predicate `token` filters match only `ObjectKind::Token`; spell copies used to match, though no defined card filters spells by token status.
- **No emblem or copy-of-a-card kind.** No defined card creates either; the docs say "none yet".

## Conversion cleanup

Object lookups, references, predicates, moves, casting helpers and projection
helpers and zone membership accept `impl Into<ObjectId>`, the one convention;
a wrapper hands the id to the object function it calls without converting
first. Observation definition counting accepts either id kind.
Card-only functions retain `CardId` and checked construction is unchanged.

The steer's exact `git grep -nE` expression now reports **76 before → 43
after** (baseline `66e8b7ae5`, working tree after this cleanup). Both include
17 card-name string conversions in `possible_worlds.rs`; excluding those gives
**59 → 26 CardId widening sites**. These are matching source lines, not calls.
The earlier broader 88→41 audit was historical and used a different scope.
Removing four widened locals increases the grep count by one because concrete
transport fields now convert independently. No object API needs a performance
exception.

Every retained match involving CardId is listed below (paths under `managym/`):

| Sites | Reason |
| --- | --- |
| `src/agent/observation.rs:2114` | Combine card and permanent activation sources as `Option<ObjectId>`. |
| `src/flow/action.rs:379,412,430,448,489,606,619` | Fill concrete Action or Payment object fields. |
| `src/flow/decision.rs:1909,1919,3443` | Fill object-based Action fields, including a rejection fixture. |
| `src/flow/decision.rs:2553,2608,2656,3037` | Compare checked candidates with object-based command identities. |
| `src/flow/decision.rs:3364,3693,3770` | Private test helper accepts concrete command transport. |
| `src/flow/decision.rs:3387,3978` | Compare checked cards with event/Action object fields in assertions. |
| `src/flow/play.rs:999` | Exclude the casting object from checked alternative-cost candidates. |
| `src/flow/play.rs:2043,2154` | Combine card and permanent payment sources as `Option<ObjectId>`. |
| `src/flow/play.rs:2405` | Fill `ActivatedAbilityOnStack.source_object`. |
| `state/src/game_object.rs:271,284` | Implement checked indexing through the ObjectId index. |

Additional widenings outside that regex remain at standard collection or return
boundaries: `cards/tests/cards/stx.rs:573` (mixed hand membership),
`src/flow/play.rs:173` (compare library-top card with object),
`src/flow/cast_permission.rs:226`, `src/flow/search.rs:100` and
`src/flow/selection_sampling.rs:141` (collect or extend mixed object lists), and
`tests/rules/cr_400_identity.rs:209,277,308,353,392` (numeric identity and typed
scan assertions). None converts a card merely to pass it to an object API.

Earlier conversion cleanup and its compression passed 44 focused debug tests (1.242 s)
and workspace all-target/all-feature Clippy with warnings denied. Formatting and whitespace
checks passed. The cp313/arm64 wheel rebuilt, its matching extension was
installed atomically, and a fresh import passed. Final-head conformance, replay
bytes and Engine Budget (unchanged
digests) still belong to gate. Existing seeded QA below predates this pass.

The final four-local cleanup passed six focused debug tests (checked card/zone
identity, transient occupants, hidden reassignment, landcycling and Force of
Negation's alternative cost) and all-target/all-feature workspace Clippy with
warnings denied. The cp313/arm64 release wheel rebuilt, its extension was
installed atomically, and a fresh import, rustfmt and whitespace checks passed.
Exact command and run ID are in `scratch/mtg-320-plan.md`.

## Seeded QA before the conversion cleanup and main merge

538 seeded random games (authored setups, Fire Lord Azula copies, 400 TLA pool human builds; 118,781 decisions) produce byte-identical observations for both seats and identical state digests on this branch and main 6ce618b8a, including four matching `PyAgentError` endings. That comparison first caught a regression in Kyoshi Battle Fan's token attach, fixed here. Two Academy games played to Game Over. Details: `scratch/qa.html` (not published).

## Remaining before review

Main is merged through 1d7b5fde9 (its engine tree equals 22ed4a733 and
2c47f59ae), with main's explicit mana-spending fixtures adapted to runtime
object ids (91ba66c3a4). At 9c9124627, 568 seeded games (108 authored, 30
Azula, 30 Heartbeat/Lita/Release to Memory mirrors, 400 TLA pools; 137,518
decisions) matched main 22ed4a733 byte for byte in observations and state
hashes; the one `PyAgentError: source cannot pay tap cost` occurs identically
on main. Two Academy games reached Game Over. CI passed on c52d81b16. The
widened-local cleanup (4ef4a0829: `flow/action.rs`, `flow/play.rs`,
`state/src/zone.rs`) came after that QA; its CI run, Engine Budget numbers
and authored reviews remain.

## Left for later

Not in this PR:

- a separate physical `Card` store beside the object store (`CardId` now exists as a checked reference to an object row);
- a new object per zone change (today `ObjectId` is the row that survives a zone change, and `ObjectRef` is the appearance that goes stale);
- the CR 400.7 carry-over record (cast, kicked, linked effects);
- measuring what a new id per zone change costs in layer-view and `TriggerIndex` invalidation;
- whether the later physical-card split changes the storage of library and hand.

## Checks

Compression passed 31 focused debug tests, all-target/all-feature workspace
Clippy with warnings denied, rustfmt and whitespace checks. The cp313/arm64
release wheel rebuilt, its extension was installed atomically, and a fresh
import passed. It removes the custom zone iterator and conditional card-scan
chains, reuses scry ordering, and restores inspected libraries in one placement
pass. Storage and serialized fields stay unchanged. Exact commands and receipts
are in `scratch/mtg-320-plan.md`.

Before compression, the typed-zone/projection pass passed 17 focused debug tests
covering numeric bytes, checked scans, transient tokens/spell copies, shuffle
RNG, hidden reassignment, draws, ordering and projection. Workspace Clippy passed;
the cp313 release wheel rebuilt, was installed atomically, and passed a fresh
import. Rustfmt and whitespace checks passed. Four merged payment fixtures were
also repaired and all 15 focused payment tests passed. These checks predate
the subsequent merge of main's payment fast paths (6ce618b8a).

The implementation and compression were checkpointed in c76f01f02 and published
with merge head 96d3aa14e. Realignment found one more newly merged payment fixture
using the removed `card: CardId(index)` field; it is corrected locally to
`object: ObjectId(index)`. All 19 focused debug payment tests and workspace
Clippy passed; the current cp313/arm64 extension rebuilt, was installed
atomically, and passed a fresh import. Current-head
[CI](https://github.com/loopflowstudio/manabot/actions/runs/43375879983) confirms
that fixture caused its Rust compile failure; semantic conformance, Python
unit tests, Academy integration and Clean-machine Play passed. Other jobs and
[package checks](https://github.com/loopflowstudio/manabot/actions/runs/43375879939)
were still running at readback and do not include this repair.

[CI run 43369147820](https://github.com/loopflowstudio/manabot/actions/runs/43369147820)
on 8e5e8e62b passed Semantic Kernel Conformance, Python Unit Tests, Academy
Integration and Clean-machine Play; Rust Tests failed to compile the four
fixtures since repaired. Engine Budget and all Integration groups passed;
[package run 43369147921](https://github.com/loopflowstudio/manabot/actions/runs/43369147921)
completed successfully. These results cover 8e5e8e62b only.

Earlier evidence, **not validation of typed ids or compression**:

- [CI run 43325427802](https://github.com/loopflowstudio/manabot/actions/runs/43325427802)
  on 2a3eae3b0 passed all 13 jobs, including debug Rust, conformance, Engine
  Budget, Clean-machine Play, Python and Academy.
- Its Engine Budget compared 2a3eae3b0 with rebuilt 7ef6ce588 on the same
  runner: search total 173.303 vs 173.206 s, worst reply 5.102 vs 5.123 s;
  steps 12,174.4/s vs 12,398.8/s. All digests matched and the 10% budget passed.
- QA at a45df9d4b compared 190 seeded games with MTG-298 parent 4bc48211e:
  both viewers' observations and full state digests matched for 25,732
  decisions, including spell copies and permanent-spell-copy tokens.
  Another 60 random games finished. Treasures remained in graveyards during
  mana choices until SBAs, as required. See `scratch/qa.html`.
- Four Academy games finished on 79aa9881e. A later table run stalled at
  a45df9d4b; its cause is unknown. It coincided with high host load, which is
  not evidence of causation. A clean final-head Academy run remains for QA.

Not reached in Academy: a permanent-spell copy (no matchup has Azula) or a spell
copy bounced to hand (no defined card does it). Older test receipts and superseded
CI status are in `df58f2bff:scratch/pr-491-body.md`.
