# MTG-483 — Grouped input: one managym compiler from a whole plan to sequential Actions

This plan replaces the browser-side chunked-runner plan (`chunked.ts`,
`draft-store.ts`, `PartitionPanel.svelte`, the `piloting`/`rejected` props).
None of that was built; nothing of it remains to remove.

Direction, verbatim (Jack Heart, 2026-11-25, Option A):

> Stop the browser-side chunked Command runner build and pivot MTG-483 to a
> first-class managym grouped-input API.
> - Keep core engine `Action` / `step()` flat and sequential (never add atomic
>   grouped `Action` variants that change RL/search action spaces,
>   observations, digests, or split human vs bot semantics).
> - Promote the pattern started in `managym/src/flow/combat_preview.rs`
>   (`Game::combat_preview`) into a first-class `managym` grouped-input
>   compiler layer above sequential `Action`s:
>   1. Structured grouped-decision descriptor / metadata from `managym` (on
>      `ExperienceFrame` / Python API) so callers never scrape Rust `Debug`
>      strings (`pending`) or label regexes (`FACE_UP_RE`) and never
>      re-implement engine rules (`skip_trivial`, lethal minimums,
>      `max_share + targets - 1`, distinct-name constraints) in TypeScript.
>   2. Read-only `preview_group(plan)` (clone-and-step compiler): validates a
>      typed grouped plan against the real engine rules, returns
>      requirements/legality/refusal reasons and the compiled sequential steps.
>   3. First-class grouped submission (`apply_group(plan)` in `managym` +
>      Academy server endpoint) bound to `(revision, frame_hash)`: validates
>      the whole plan on a clone first, then executes the exact same sequential
>      `Action`s in one server/engine call—recording identical per-step
>      trace/replay actions, digests, presentation events, Study forks, and
>      policy-probability attachment points while eliminating multi-RTT browser
>      send loops, disconnect/reload half-commits, and double `command_outcome`
>      races.
>   4. Honest purity boundary: groups stop cleanly at any hidden-info reveal,
>      shuffle/draw, or opponent choice (e.g., Surveil before OrderLibrary,
>      OrderTriggers when modes/targets interleave).
>   5. Shared decision surface & MTG-441 invariants preserved: raw single-step
>      offers stay reachable under the collapsed disclosure in live play (never
>      hidden permanently); Replay/Study keep raw per-step offers visible;
>      nothing commits until Confirm; preserve MTG-441 drag/combat UX and check
>      whether stacking on MTG-431 is still needed or if MTG-483 should base
>      cleanly on main / wait cleanly for MTG-441.

## Outcome

A person playing live in Academy drafts a multi-part decision whole, sees the
engine's own verdict as they edit it, and confirms once. Confirm is one
`group_command` message and one authority call: the whole plan lands or none
of it does. This holds for combat declarations and for Scry, Surveil and
OrderLibrary. The recording is byte-for-byte the per-step recording: one
`ReplayDecision` per native step, so Replay, Study forks and policy
probabilities attach where they always did.

A Python caller of `managym` uses one decision surface on `Env`, with no
Academy involved (Pass 4, replacing Pass 1's `group`/`preview_group`/
`apply_group` side channel after the ship-review in
`scratch/ship-review-notes.md`):

- `env.decision(viewer=None) -> Decision | None`: the pending decision, always.
  Grouped kinds (`DeclareAttackers`, `DeclareBlockers`, `Scry`, `Surveil`,
  `OrderLibrary`) carry remaining items plus `committed*` `GroupCard`s, so
  half-way through raw attacker steps it reports what is declared and what
  remains; anything else is `SingleDecision` with raw `Offer`s.
- `env.preview(choice, *, at=None) -> DecisionPreview`: read-only clone-and-step.
  `choice` is an offer id or a typed grouped `Choice`; `at` is a
  `DecisionAddress(viewer, revision, fingerprint)`.
- `env.submit(choice, *, at=None)`: validates on a clone, then steps the live
  game through the same sequential `Action`s, all or nothing.
- `env.step(action: int)` stays one flat RL step.
- `compound_offers`, `compound_commands_json`, `structured_offers`,
  `step_structured` and `combat_preview_json` / `managym/combat.py` warn for
  removal in `0.3.0`; `structured_search_offers` stays.

Types live in `managym/group.py`; the native layer is `managym/src/flow/group/`
(`mod.rs` driver, `combat.rs`, `library.rs`); Academy reads it through
`GameSession.live_group()` and `/group-preview` in `managym/academy/server.py`.

## Decisions that still explain the approach

- Based on main, not MTG-431 (`lf sync main`, `e2224c3988`); the pile split is
  a follow-up kind. MTG-441 is merged in (`3460ec64c5`).
- `protocol/experience-v1.schema.json` is untouched: `ExperienceFrame` feeds
  `frame_hash`, which retained replays re-check. The live `group` travels
  beside `frame` on WebSocket messages, attached per recipient in
  `_response_with_participant` only for roles with `SUBMIT_LIVE_COMMAND`
  (watchers never see private Scry/Surveil cards; a new pilot gets it on
  `role_changed`). `group_command` lives in `protocol/testing-house-v1.schema.json`.
- Built on `combat_preview`'s clone-and-step mechanism; `compound_*` was not
  widened (Study branches and `manabot/sim/action/benchmark.py` used it), only
  deprecated.
- Purity boundary: between steps of one group the clone must add no
  `CardDrawn`/`CardRevealed`, leave `state.rng` unchanged and reveal no new
  hidden object; effects after the group ends (Preordain's draw) are fine.
- Per-step recorded command ids are `authority.group.<id>.<n>`; duplicate
  `command_id` retries reuse the existing `accepted_commands` path.
- Remaining assumptions are listed in `scratch/questions.md`.

## Passes

1. Native layer with combat parity and library kinds — `b3ae75976c`.
2. Academy authority and `testing-house-v1` schema — `b510844e42`.
3. Browser and docs — `c870fb9349`.
4. Unified `Env.decision`/`preview`/`submit`, `GroupCard` `committed*`
   labels, `0.3.0` deprecations, Academy alignment, PR #623 CI fixes
   (`SpaStaticFiles` `/api/*` 404; `showEachCreaturesChoice` for library
   groups) — `716d7ed377`. Recorded passing: `cargo nextest`
   (`group | combat_preview`), `scripts/public_api.py check`, pytest
   (`test_group`, `test_group_command`, `test_shared_table_protocol`,
   `test_public_api_contract`, `test_package_boundaries`, `test_study_branch`,
   `test_structured_policy`), `svelte-check`, 709 UI unit tests, Playwright
   `grouped-decisions`, `combat-presentation`, `drag-play`, `keyboard-play`
   (34/34). `laptop-comfort` hit a pre-existing `BEAT_MS`/`Travel` race fixed
   on main by `9f20178dbf`.

No build pass remains.

## Remaining (gate / ship-review)

- Full CI on PR #623 after Pass 4 (checks had no conclusions at realign,
  2026-11-25) and the live demo: `./scripts/play`, attack with several
  creatures and Confirm (one `group_command` in the network panel), then Scry 2
  and arrange both cards before one Confirm.
- Ship-review: answer Jack's API questions against the new surface and file
  the follow-ups in `scratch/questions.md` (pile split, "pick", "amounts",
  OrderTriggers, discard instruction text, `0.3.0` removals).
- Nit: `managym/group.py` names internal helpers `_parse_decision`,
  `_parse_preview`, `_dump_choice`, `_dump_address`, `_DecisionBase`, against
  the no-underscore style rule; `managym/src/python/bindings.rs` looks them up
  by name, so renaming needs an extension rebuild.

## Done when

- Confirming a drafted attack, block, scry, surveil or library order sends one
  `group_command`, and the game is wholly past that decision or exactly where
  it was with the engine's reason shown.
- A game played with grouped Confirms and the same game played one Command at
  a time record the same decisions and reach the same state digest.
- Replay and Study show every per-step choice as before;
  `protocol/experience-v1.schema.json` is unchanged.
- Disconnecting or reloading mid-Confirm reconciles via `accepted_commands`
  without double-applying.
- The browser holds no send loop, no step verification and no legality rule;
  it reads `group` and `/group-preview`.
- `uv run python scripts/public_api.py check` passes.
- `docs/academy.md` carries the reclassified audit and names the follow-up
  kinds.
