The engine marked tokens and copies of spells with one `is_token` flag stored on the object's definition, and called every runtime object a `Card`. Each game object now has a kind taken from the Comprehensive Rules (`ObjectKind::Card`, `Token` or `SpellCopy`), and the runtime record is named for what it is: `Object`, named by `ObjectId`. Card-only consumers narrow through `GameState::card_id` and carry `CardId`, so a token or a copy of a spell cannot enter those operations.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Derive runtime object types and zone contents from the MTG rules, so tokens are not Cards · MTG-320](https://linear.app/loopflow/issue/MTG-320/derive-runtime-object-types-and-zone-contents-from-the-mtg-rules-so)
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

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
- **Byte compatibility remains an acceptance requirement.** Serde keeps the old field names (`#[serde(rename = "card")]`). Observation `is_token` is `!is_card()`, and the match-state hash records a non-card object with `is_token: true`, as before. Current-head Semantic Kernel Conformance passed; the final seeded comparison below also matches observations and state hashes. `Debug` output of the renamed types does change.
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

## Validation

- [Current-head CI](https://github.com/loopflowstudio/manabot/actions/runs/43493623873) at `349d643e52`: Rust Tests, Python Unit Tests, all six Integration groups, Academy Integration, both conformance jobs, Academy UI and Clean-machine Play passed. Engine Budget also passed; its measurements are below. [Package checks](https://github.com/loopflowstudio/manabot/actions/runs/43493623564) passed for CPython 3.12/3.13 on Linux x86_64 and macOS arm64, including installs and artifacts.
- Local gate at the same head: 22 focused debug tests passed (checked ids, numeric serialization, transient-zone ordering/RNG, hidden reassignment, stale references, SBA, card-only choices, landcycling, Force of Negation and Kyoshi Battle Fan). Workspace all-target/all-feature Clippy with warnings denied, workspace rustfmt and whitespace checks passed. Full Rust coverage is the current-head CI result above.
- Final seeded QA at `6b28b642d6`: 568 games and 137,518 decisions matched main's observations for both seats and full state hashes byte for byte. The engine tree is unchanged between that QA and `349d643e52`. Three Academy games reached Game Over without page or console errors. No frozen evidence was regenerated.

### Engine Budget

[Current-head run 43493623873](https://github.com/loopflowstudio/manabot/actions/runs/43493623873) measured `349d643e52` against reference `d53220d36`, both rebuilt on the same GitHub-hosted Ubuntu AMD EPYC 9V74 runner (4 vCPU), in release. The paired candidate/reference/reference/candidate order uses median durations; the downloaded artifact also passed the local comparison command.
| Measurement | Reference | Candidate | Change |
| --- | ---: | ---: | ---: |
| Search total | 90.484 s | 98.364 s | +8.71% |
| Worst search reply | 2.598 s | 2.818 s | +8.48% |
| Steps per second | 19,542.0 | 18,495.2 | -5.36% |

**The 10% budget passes.** All three search digests, the step digest, 512-game count and 102,209-step count match. These figures compare with the recorded reference, not main; they do not isolate this refactor's cost from intervening main changes. The artifact retains load/CPU/scheduling measurements. Search timings are close to the limit; this passes the existing budget rather than establishing a speedup.

## Limits and review handoff

- Graveyard/exile retain mixed object lists as described above. Physical-card storage, new objects per zone change, CR 400.7 carry-over records and their cache-invalidation measurements belong to the separate follow-up.
- One seeded TLA pool game ends with `PyAgentError: source cannot pay tap cost` on both main and this branch. An earlier Academy table stall did not recur in the final three games; its cause remains unknown.
- Academy did not exercise a permanent-spell copy (no matchup has Azula) or a spell copy bounced to hand (no defined card does it). Emblems are unimplemented.
- Earlier local `test_capture_replays_both_seat_update` exhausted its configured 120-second training wall deadline (141 seconds overall). Current-head CI ran the same test in 28.83 seconds; the operator steer explicitly left this check to CI. The earlier timeout's cause remains unknown; no deadline was relaxed.
- No human demo or new authored review was collected in the headless gate. Stop at review; auto-merge stays disabled.
