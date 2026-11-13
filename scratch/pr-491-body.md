<!-- Draft body for PR #491. Keep the loopflow task-pr-context block the PR already has. -->

The engine marked tokens and copies of spells with one `is_token` flag stored on the object's definition, and called every runtime object a `Card`. Each game object now has a kind taken from the Comprehensive Rules (`ObjectKind::Card`, `Token` or `SpellCopy`), and the runtime record is named for what it is: `Object`, named by `ObjectId`. Effects that say "card" ask `is_card()`, so a token or a copy of a spell is never treated as a card.

This builds on [MTG-298 · Give tokens their own `TokenDefinition` type](https://github.com/loopflowstudio/manabot/pull/476), which split token definitions from card definitions and lands first.

## What changes

- **Kind.** The runtime record has a private `kind`, set once when the object is made. A copy of a permanent spell becomes a token as it enters the battlefield (`resolve_into_token`, CR 111.12, 608.3f); nothing else changes a kind.
- **Readers moved to the kind.** State-based actions (a token off the battlefield and a spell copy off the stack cease to exist, CR 704.5d-e), predicate and layer `token` filters, exile cast grants, "choose a card" selection, the return-to-battlefield guard, and `TriggerCondition::PutIntoGraveyard` (`moved.is_card()`). `become_copy_of` no longer saves and restores a token flag. The remaining `is_token` reads are catalog rows, deck and sideboard admission, the observation column and the hash's wire bytes.
- **Rename.** `Card` -> `Object` (in `state::game_object`), `CardId` -> `ObjectId`, `CardVec` -> `ObjectVec`, `GameState::cards` -> `objects`, `card_to_permanent` -> `object_to_permanent`. The old u32 `ObjectId` (the match-unique public number for players, objects and abilities) is now `PublicId`. Helpers that move or match any object follow: `move_object`, `move_object_with_entry`, `capture_object_move`, `subject_matches_moved_object`, `remove_object`, `object_zones`, `matches_object`, `object_power`, `object_mana_value`.
- **`card` stays where the rules mean a card:** card definitions, decklists, sideboards, hidden-card worlds, hand/library/graveyard/exile selection helpers (`draw_cards`, `library_top_card`, `Target::GraveyardCard`, ...), and the observation encoders and Python bindings, whose names are published.
- **Docs.** `docs/engine.md` has a new section, [Objects and zones](docs/engine.md#objects-and-zones): one Spirit token through a turn, the CR term -> engine type table, what is built, the planned split, and where the engine departs from the rules.

## Decisions a reviewer should check

- **Library and hand lists hold `ObjectId`s, not card ids.** A bounced or tucked token is in that zone until state-based actions are checked (CR 111.7-111.8, 704.5d), and a viewer can see it there: QA saw Treasures in a graveyard at the mana-choice decision right after they were sacrificed. Making those zones card-only would remove the token inside the move and change observations. Only decklists and sideboards are card-only.
- **Observations, wire names and evidence keep their bytes relative to MTG-298.** Serde keeps the old field names (`#[serde(rename = "card")]`). Observation `is_token` is `!is_card()`, and the match-state hash records a non-card object with `is_token: true`, as before. `semantic-kernel-w17-v2` conformance reproduces. `Debug` output of the renamed types does change.
- **One narrowing.** Predicate `token` filters match only `ObjectKind::Token`; spell copies used to match. No defined card filters spells by token status.
- **No emblem or copy-of-a-card kind.** No defined card creates either; the docs say "none yet".

## Left for later

Not in this PR:

- a separate `Card`/`CardId` store beside the object store;
- a new object per zone change (today `ObjectId` is the row that survives a zone change, and `ObjectRef` is the appearance that goes stale);
- the CR 400.7 carry-over record (cast, kicked, linked effects);
- measuring what a new id per zone change costs in layer-view and `TriggerIndex` invalidation;
- whether library and hand then become card-id lists.

## Checks

- [CI run 43325427802](https://github.com/loopflowstudio/manabot/actions/runs/43325427802) on 2a3eae3b0 (before the final helper renames): all 13 jobs passed, including debug Rust, both conformance jobs, Engine Budget, Clean-machine Play, Python Unit, Academy and all three Integration partitions.
- Engine Budget on that run compared 2a3eae3b0 with 7ef6ce588 rebuilt on the same runner: `bench_search` total 173.303 s versus 173.206 s, worst reply 5.102 s versus 5.123 s; `bench_steps` 12,174.4/s versus 12,398.8/s. All search/decision digests matched; all measurements passed the 10% budget. These measurements precede the helper cleanup.
- Current published head a45df9d4b includes the helper cleanup and merges parent 4bc48211e (scratch-only changes). Release packages 43330293430 and CI 43330293531 were still in progress when checked on 2026-11-12; final-head checks remain to be confirmed.
- After the final helper renames (names and comments only): `cargo build --tests --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` clean; debug nextest for zone identity, stack, triggers, state-based actions and scenarios (30 passed) and token and Azula card tests (41 passed).
- After simplifying `assert_stack_consistent` to compare sorted object lists: clippy clean; focused debug nextest for stack, copy, counter and cr_6 tests, 160 passed.
- Earlier, before the MTG-298 merge: debug `cargo nextest run --workspace --profile ci`, 1166 passed; 4 timed out under machine load and passed alone.
- Directed test: a copy of an instant from Fire Lord Azula is a `SpellCopy` and ceases to exist off the stack.
- Earlier pytest over env, belief, semantic, verify, Academy and the world-change list: 907 passed, 1 failed (`test_v10_vector_buffers_reach_the_model`, "policy forward requires an observation from admit_observation"), 1 deselected for the five-minute limit. The earlier plan attributed the failure to inherited code because Python matched main then; that is not an isolated reproduction of its cause. Later CI on 2a3eae3b0 passed.

QA on the release extension: 60 random games over five authored packs, 150 scripted Fire Lord Azula games and four Academy games all finished. No token or copy was seen off the battlefield or stack at a decision except Treasures mid mana choice, which is correct (CR 704.3).

## Not reached

- A token copy of a permanent spell in Academy (no Academy matchup plays Fire Lord Azula).
- Bouncing a spell copy to hand (no defined card does it).
