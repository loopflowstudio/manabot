<!-- Draft body for PR #491. Keep the loopflow task-pr-context block the PR already has. -->

The engine marked tokens and copies of spells with one `is_token` flag stored on the object's definition, and called every runtime object a `Card`. Each game object now has a kind taken from the Comprehensive Rules (`ObjectKind::Card`, `Token` or `SpellCopy`), and the runtime record is named for what it is: `Object`, named by `ObjectId`. Card-only consumers narrow through `GameState::card_id` and carry `CardId`, so a token or a copy of a spell cannot enter those operations.

This builds on [MTG-298 · Give tokens their own `TokenDefinition` type](https://github.com/loopflowstudio/manabot/pull/476), which split token definitions from card definitions and lands first.

## What changes

- **Checked card ids.** `CardId` has a private index and is created only after checking the object kind. Deck/sideboard rosters and card abilities carry it; card-only candidates and play/cost entry points use it. Conversion to `ObjectId` is free, indexing asserts the kind in debug, and serialization retains the same numeric row.
- **Kind.** The runtime record has a private `kind`, set once when the object is made. A copy of a permanent spell becomes a token as it enters the battlefield (`resolve_into_token`, CR 111.12, 608.3f); nothing else changes a kind.
- **Readers moved to the kind.** State-based actions (a token off the battlefield and a spell copy off the stack cease to exist, CR 704.5d-e), predicate and layer `token` filters, exile cast grants, "choose a card" selection, the return-to-battlefield guard, and `TriggerCondition::PutIntoGraveyard` (`moved.is_card()`). `become_copy_of` no longer saves and restores a token flag. The remaining `is_token` reads are catalog rows, deck and sideboard admission, the observation column and the hash's wire bytes.
- **Rename.** `Card` -> `Object` (in `state::game_object`), `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`, `GameState::cards` -> `objects`, `card_to_permanent` -> `object_to_permanent`. The old u32 `ObjectId` (the match-unique public number for players, objects and abilities) is now `PublicId`. Helpers that move or match any object follow: `move_object`, `move_object_with_entry`, `capture_object_move`, `subject_matches_moved_object`, `remove_object`, `object_zones`, `matches_object`, `object_power`, `object_mana_value`.
- **`card` stays where the rules mean a card:** card definitions, decklists, sideboards, checked card selection helpers (`library_top_card`, `selection_zone_cards`, `foretold_cards`, ...), draw rules, exact graveyard targets, and the observation encoders and Python bindings, whose names are published.
- **Docs.** `docs/engine.md` has a new section, [Objects and zones](docs/engine.md#objects-and-zones): one Spirit token through a turn, the CR term -> engine type table, what is built, the planned split, and where the engine departs from the rules.

## Decisions a reviewer should check

- **Library and hand lists hold `ObjectId`s, not card ids.** A bounced or tucked token is in that zone until state-based actions are checked (CR 111.7-111.8, 704.5d), and a viewer can see it there: QA saw Treasures in a graveyard at the mana-choice decision right after they were sacrificed. Making those zones card-only would remove the token inside the move and change observations. Decklists and sideboards carry `CardId`; `GameState::zone_cards` narrows mixed zone lists for card-only consumers. Hidden sampling and observations still use the ordered object lists.
- **Observations, wire names and evidence keep their bytes relative to MTG-298.** Serde keeps the old field names (`#[serde(rename = "card")]`). Observation `is_token` is `!is_card()`, and the match-state hash records a non-card object with `is_token: true`, as before. The earlier rename reproduced `semantic-kernel-w17-v2`; final-head conformance remains for gate. `Debug` output of the renamed types does change.
- **Card-only choices.** Transient tokens stay visible in their zones but are excluded from card-only play, activation, discard and search choices. Direct card-only commands reject them too. Generic movement, inspection and draw ordering still include them. Predicate `token` filters match only `ObjectKind::Token`; spell copies used to match, though no defined card filters spells by token status.
- **No emblem or copy-of-a-card kind.** No defined card creates either; the docs say "none yet".

## Left for later

Not in this PR:

- a separate physical `Card` store beside the object store (`CardId` now exists as a checked reference to an object row);
- a new object per zone change (today `ObjectId` is the row that survives a zone change, and `ObjectRef` is the appearance that goes stale);
- the CR 400.7 carry-over record (cast, kicked, linked effects);
- measuring what a new id per zone change costs in layer-view and `TriggerIndex` invalidation;
- whether the later physical-card split changes the storage of library and hand.

## Checks

After compression, 53 focused debug tests passed across checked ids, transient
occupants, card choices, casting, capture and relevant card behavior. Workspace
Clippy passed with every target and feature and warnings denied; rustfmt and
whitespace checks passed. The CPython 3.13 release wheel built, its matching
extension was installed atomically, and a fresh import passed. Exact commands and run receipts are in
`scratch/mtg-320-plan.md`. Final-head conformance, seeded QA and Engine Budget
remain for gate.

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
