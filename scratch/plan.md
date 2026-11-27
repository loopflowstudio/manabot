# MTG-502: counterfactual state edits as a replay intervention

Tracks https://github.com/loopflowstudio/manabot/issues/630.

Design exploration pages: [scratch/design-guide.html](scratch/design-guide.html) (settled choices), [scratch/explore.html](scratch/explore.html) (round 2), [scratch/explore-belief-memory.html](scratch/explore-belief-memory.html) (belief/memory review).

## Outcome

A Python caller holding a live `managym.Env` or `managym.GameSnapshot` at step *t* applies one or more typed `GameStateEdit` operations in a single journaled intervention and reads the refreshed legal actions and observation of the same decision on the edited state. Turn, step, stack, mana pools, graveyards, exile, summoning sickness, every other permanent, the 32-row `policy_history` window, decision memory, and replay provenance stay intact.

Two doors share the exact same Rust `Intervention::GameStateEdits` path:

```python
# Door 1: quick probe on a clone (1 clone, memory capture frozen, returns refreshed Observation)
what_if = env.clone_env()
blocker = next(p for p in obs.opponent_permanents if p.characteristics.name == "Gray Ogre")
obs2 = what_if.edit_game_state([
    GameStateEdit.move_permanent(blocker.id, to="library_bottom"),
])

# Door 2: fork from a snapshot (1 clone, preserves memory capture, composes with VectorEnv SlotLoad)
snap = env.snapshot()
branch = snap.restore(
    RestoreOptions.exact(
        game_state_edits=[
            GameStateEdit.set_tapped(land.id, True),
            GameStateEdit.set_life(obs.opponent.player_index, 5),
        ]
    )
)
replay_json = branch.snapshot().to_replay_json()  # cold replay verifies digest
```

## Public API (`GameStateEdit`)

One pure-data value type `GameStateEdit` in `managym`, five constructors, extensible later without changing the type or journal envelope:

| Constructor | Effect |
| --- | --- |
| `GameStateEdit.move_permanent(permanent, to="graveyard")` | battlefield permanent to owner's `"graveyard"`, `"exile"` or `"library_bottom"`; cards are conserved, tokens cease to exist |
| `GameStateEdit.set_tapped(permanent, tapped)` | sets the tapped flag (no stun-counter spend) |
| `GameStateEdit.set_damage(permanent, damage)` | sets marked damage (`0` clears the deathtouch mark); `dealt_damage_this_turn` untouched |
| `GameStateEdit.set_counters(permanent, kind, count)` | sets one counter kind through `TrackedState::change_counters` (keyword-counter timestamps, revisions); `0` removes |
| `GameStateEdit.set_life(player, life)` | sets player `0`/`1` life; `turn.history` untouched |

- `env.edit_game_state(edits) -> Observation`: validates, applies, journals one `"game_state_edits"` intervention, republishes the current decision once.
- `RestoreOptions.exact/shuffle_libraries/determinize(..., game_state_edits=None)`: edits applied **before** `mode`; non-empty edits with `forced_first_action` are rejected. Stored as `Option<Arc<[GameStateEdit]>>` so `RestoreOptions` stays cheap to clone (it loses `Copy`; `SlotLoad` callers clone).
- `permanent` is the observed `Permanent.id` (`managym/src/agent/observation.rs:1844`). `kind` is `CounterKind::name`; note Python `Permanent.counters` omits `plus_one_plus_one` (`observation.rs:1910`, exposed as `plus1_counters`), so the docs say so. `player` is `0`/`1`.
- Errors are `managym.AgentError`, leaving Env and provenance unchanged: unknown or phased-out permanent, duplicate move of one permanent or editing a permanent after moving it earlier in the same batch, unknown destination/counter kind, negative damage/count, bad player, before reset, game over, `opening()`/`pending_choice`/`suspended_decision` present, unsupported decision kind, declare-step edit leaving the focal declarer ineligible or an empty action space, edits + `forced_first_action`.

## Replay journal

`{"at_action": 41, "op": "game_state_edits", "edits": [{"kind": "move_permanent", "permanent": 42, "to": "graveyard"}, ...]}` in `GameProvenance.interventions` (`managym/src/agent/env/provenance.rs`); empty `Arc` in normal play. Mirrored by `Intervention::GameStateEdits` (Rust), a strict `GameStateEdits` model with a tagged edit union in `managym/recording.py` (round-trips through `RecordedVectorEnv._materialize` and `managym.replay.walk`), `BRANCHING_INTERVENTIONS` in `managym/academy/trace.py:240`, and Academy UI `BranchKind` (`academy/ui/src/lib/recording.ts:54`) + `BRANCHES` label `'game state edited'` (`academy/ui/src/lib/board/replay.ts:257`, else Watch shows "Branch · undefined"). Cold replay re-applies through the same `apply_intervention` and verifies `state_digest`.

## Settled (Jack Heart, 2026-11-27)

Choices `1a, 2a, 3a, 4a, 5a, 6a` with: *"GameStateEdit?"*, *"interventions should be called game_state_edits right / or whatever naem we use, more consistent between journal and api"*, *"i feel like remmove_permanent should be move_permenant and take a zone"*, *"so things like the cards in the deck still all accounted for"*, `hidden_neutral` → *"drop it"*, *"sounds good lets proceed"*.

1. One noun `GameStateEdit` across Rust, Python, journal (`game_state_edits`), `Env.edit_game_state`, `RestoreOptions(game_state_edits=)`.
2. Policy history: expire live links to the moved object, append no `Arrival`, evict nothing; tap/damage/counters/life leave history alone. Decision memory intact; `clone_env()` freezes capture, `restore()` keeps it.
3. Legacy `scenario_*` untouched; no aliases.
4. `edit_game_state` returns `Observation`; no `hidden_neutral`; `PossibleWorldSpace` identity stays strict.
5. The five edits above; no card leaves the game.
6. Validate before mutating; Priority path is 1 clone + 1 republish.
SBAs wait for the game's next check; refreshable kinds are `Priority`, `DeclareAttacker`, `DeclareBlocker`; suspended decisions refuse.

## Concept review (scratch/concept-review.html) — how each finding landed

Verified in code during realign: `policy_history::record` clears only `context_id` (`policy_history.rs:136-160`); `restore` binds `forced_first_action` before interventions (`snapshot.rs:144-185`); `RestoreOptions` is `Copy` (`snapshot.rs:58`); TS `BranchKind`/`BRANCHES` are closed maps.

- **`commit_zone_move` is a rules transition, not a primitive** (emits `CardMoved` → history row + presentation event, token placement, exile-link followups, card capture, melds). Replaces the plan's mark-and-restore wrapper with a dedicated `quiet_move_permanent`: remove from combat (`remove_permanent_from_combat` **and** `combat.remove_defender`), snapshot LKI, clear battlefield records, end effects on it, drop dead leaves-battlefield delayed triggers, then token → cease, card → `mint_appearance` into destination (`library_bottom` at index 0 with a valid bottom pin). Never calls `emit()` or `commit_zone_move`. Melded permanents: both cards move to the destination together (my call; a split leaves an orphan on the library top).
- **History rule has one owner.** Extract `policy_history::expire_links(game, object, presentation_id)` from `record` and call it from both; it clears `context_id` only, keeps historical `zone` (plan previously said clear zone too — corrected).
- **Separate module.** `managym/src/flow/state_edit.rs`; `flow/scenario.rs` stays the fixture builder.
- **Declare-step atomicity.** Cheap validation cannot foresee Menace/must-block or layer/control changes (`block_requirements.rs:425`). Declare steps take `rollback_snapshot()` (`flow/card_capture.rs:283`), mutate, `refresh_control()`, rebuild `*_to_declare`, rebuild the focal actions, and roll back with a named error if the focal declarer is ineligible or no actions remain. Priority keeps the in-place path.
- Small mismatches (counter routing, `plus_one_plus_one`, `dealt_damage_this_turn`, boundary guards, TS branch label, strict recording union) are folded into the API and journal sections above.

**Review choices adopted for ship-review, not settled** (headless; they bind replay digests, so confirm before merge — also in `scratch/questions.md`):
- **Q1 → 1a.** Tokens cease on any move; cards linked by `exile_links` stay exiled and the dead link is dropped. Passed over: 1b quietly returning linked cards (one edit would change several permanents), 1c rejecting tokens to library (special case in generic loops).
- **Q2 → 2a.** Edits before `mode`; reject edits with `forced_first_action`; `library_bottom` stays pinned, so `shuffle_libraries` does not reshuffle it (the earlier note that shuffling composes with `library_bottom` was wrong: `search.rs:56-58` skips pinned/known cards). Passed over: 2b mode first (determinize before the move exposes it), 2c an unpinned destination (changes the unseen pool).
- **Q3 → 3a.** At declare steps, newly eligible unprompted creatures (not focal, committed or declined) join `*_to_declare`, so untapping a blocker lets it block. Passed over: 3b "never add", which shows an untapped creature that cannot block.
- **Q4 → 4a.** Phased-out permanents are rejected (CR 702.26b).

## Done when

- `uv run python -m pytest tests/env/test_game_state_edits.py -q` after a rebuild: a seeded game to a mid-game `DeclareBlocker` and a later `Priority`, every edit through both doors; original digest unchanged; turn/step/stack/mana/other permanents/decision memory and the full 32-row history (links expired, no eviction, no `Arrival`) as expected; legal actions updated; cold replay matches digest, actions, history and continuation and rejects pre-edit Commands; `VectorEnv` `SlotLoad.snapshot` with edits; `determinize` and possible worlds on edited envs (bottom pin kept, token absent); pre-validation and declare rollback leave Env and provenance unchanged; edits + `forced_first_action` rejected; untapped blocker gets prompted.
- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test scenario_tests --test-threads 2` (debug): dies-trigger creature moved → empty stack, no pending triggers, `creatures_died == 0`; attacking/blocking creature and attacked planeswalker moved mid-combat, combat damage clean; enchanted creature moved → Aura to graveyard on next SBA; token moved → gone from all zones; exile-link source moved → linked card stays exiled; 32-row window preserved; debug cache and `assert_stack_consistent` hold across steps.
- Academy `academy/ui/src/lib/board/replay.test.ts` covers the new branch label.
- Existing consumers pass: `tests/verify/test_competency.py`, `tests/env/test_decision_group.py`, recording/replay tests, `managym/tests/source_taps.rs`. `uv run python scripts/public_api.py check` clean (`record` for additions only).

## Exclusions

More edit variants (hand, graveyard, exile, placement, mana, sickness, control); opt-in SBA settle; `BeliefTracker` carry-over/double scoring (separate manabot Task); suspended-decision re-evaluation; Academy What-If UI.

## Docs written before implementation

- `docs/engine-python.md` — `### Counterfactual state edits` plus `RestoreOptions` `game_state_edits` / `forced_first_action` notes.
- `docs/engine.md` — table entry in Choosing a Python API pointing to `docs/engine-python.md#counterfactual-state-edits`.
- `docs/academy.md` — mentions counterfactual state edits and the **Branch · game state edited** move-list label in Replay.

## Passes

1. **Core edit engine** landed on this branch in `f118a1571`; five focused native edit tests and four existing policy-history tests pass in debug.
2. **Republish + journal** landed in `a1e8d430d`. Three native regressions pass for priority refresh, blocker rollback and eligibility, and cold replay of an object-bound Command through exact, shuffled and determinized restores.
3. **Python, recording, Academy** landed in `70e45c2a5`. Seven focused Python tests, 30 Academy move-list tests and the public API check pass.

The focused Python checks ran against a CPython 3.13 debug wheel. The full
scenario and consumer suites in Done when are left to CI. Ship-review still
needs the decisions recorded under Concept review.
