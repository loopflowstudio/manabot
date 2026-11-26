# MTG-483 — Unified interactive decisions and native grouped-input compiler

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

A person playing live in Academy drafts any supported multi-part decision whole
(combat declarations, Scry, Surveil, OrderLibrary, and PartitionPiles), sees the
engine's own verdict as they edit it, and confirms once. Confirm is one
`group_command` message and one authority call: the whole plan lands or none of
it does, with zero browser-side send loops or label regex scraping left anywhere
in Academy. The recording is byte-for-byte the per-step recording: one
`ReplayDecision` per native step, so Replay, Study forks and policy
probabilities attach where they always did.

A Python caller of `managym` uses one self-describing decision surface on `Env`,
with no Academy involved:

- `env.decision(viewer=None) -> Decision | None`: the pending decision, always.
  - Every `Decision` carries its optimistic-concurrency `address`
    (`DecisionAddress(viewer, revision, fingerprint)`) and its complete raw
    single-step legal `offers` (`list[Offer]`), where each `Offer` carries a
    human-readable `label` (e.g. `"Play Mountain"`, `"Cast Gray Ogre"`,
    `"Attack with Raging Goblin"`, `"Pass priority"`) and its viewer-safe
    `source: SubjectRef | None` populated from native offer subjects — without
    changing `semantic_decision_frame()` fingerprints.
  - Grouped kinds (`DeclareAttackers`, `DeclareBlockers`, `Scry`, `Surveil`,
    `OrderLibrary`, `PartitionPiles`) carry remaining items plus immutable
    `committed*` state (with `GroupCard` labels on all card-based groups), so
    halfway through raw stepping it reports what is committed and what remains;
    anything else is `SingleDecision`.
- `env.preview(choice, *, at=None) -> DecisionPreview`: read-only clone-and-step.
  `choice` is an `int` offer id or a typed grouped `Choice`; `at` is an optional
  `DecisionAddress`.
- `env.submit(choice, *, at=None)`: validates on a clone, then steps the live
  game through the same sequential `Action`s, drains `observation_events`, and
  settles any built-in opponent policies so mixed-policy envs match manual
  `env.step` identically.
- `env.step(action: int)` stays one flat RL step.
- `compound_offers`, `compound_commands_json`, `structured_offers`,
  `step_structured`, and `combat_preview_json` / `managym/combat.py` warn for
  removal in `0.3.0`; `structured_search_offers` stays.

Types live in `managym/group.py`; the native layer is `managym/src/flow/group/`
(`mod.rs` driver, `combat.rs`, `library.rs`); Academy reads it through
`GameSession.live_group()`, `POST /api/sessions/{id}/group-preview`, and WS
`group_command` in `managym/academy/server.py`.

## Decisions that still explain the approach

- `protocol/experience-v1.schema.json` is untouched: `ExperienceFrame` feeds
  `frame_hash`, which retained replays re-check. The live `group` travels beside
  `frame` on WebSocket messages, attached per recipient in
  `_response_with_participant` only for roles with `SUBMIT_LIVE_COMMAND`
  (watchers never see private Scry/Surveil/PartitionPiles cards; a new pilot
  gets it on `role_changed`). `group_command` lives in
  `protocol/testing-house-v1.schema.json`.
- Built on `combat_preview`'s clone-and-step mechanism; `compound_*` was not
  widened (Study branches and `manabot/sim/action/benchmark.py` used it), only
  deprecated for `0.3.0`.
- Purity boundary: between steps of one group (only while `continues()` is
  true) the clone must add no `CardDrawn`/`CardRevealed`, leave `state.rng`
  unchanged and reveal no new hidden object; effects after the group ends
  (Preordain's draw, `PartitionPiles`'s reveal on transition to `ChoosePile`)
  complete cleanly.
- Per-step recorded command ids are `authority.group.<id>.<n>`; duplicate
  `command_id` retries reuse the existing `accepted_commands` path.
- Remaining assumptions and follow-up families (Pick, Amounts, `OrderTriggers`,
  `0.3.0` removals) are tracked in `scratch/questions.md`.

## Passes

1. Native layer with combat parity and library kinds — `b3ae75976c`.
2. Academy authority and `testing-house-v1` schema — `b510844e42`.
3. Browser and docs — `c870fb9349`.
4. Unified `Env.decision`/`preview`/`submit`, `GroupCard` `committed*` labels,
   `0.3.0` deprecations, Academy alignment, PR #623 CI fixes — `716d7ed377`,
   compressed in `bf6209ae7c`, QA'd in `dda8c2d532`.
5. **Next pass (from post-Pass 4 concept review, `scratch/concept-review.html`):**
   Close the two real conceptual mismatches left at `dda8c2d532` plus one small
   UX polish caught in Pass 4 QA:

   - **5a. Self-describing `Offer`s on `Env.decision()` (zero fingerprint drift):**
     - In `managym/src/flow/group/mod.rs`, switch `DecisionOffer.source` from
       `crate::agent::structured_offer::SubjectRef` to
       `crate::experience::SubjectRef` (matching `managym/group.py`, which
       already imports `SubjectRef` from `managym.experience`).
     - Enrich `project_offers()` in `Game::build_decision` using
       `self.decision_offer_subjects(actor)` and `self.action_space()` without
       touching `structured_search_offers()` or `semantic_decision_frame()`.
       Populate `source` with the offer's primary `SubjectRef` (`Object`,
       `Stack`, or `Player` when present) and format readable `label`s when
       `action_description(action)` is `None` (e.g. `"Play <card>"`,
       `"Cast <card>"`, `"Attack with <creature>"` / `"Don't attack with <creature>"`,
       `"Block <attacker> with <blocker>"`, `"Select <card>"`,
       `"Pass priority"`, `"Pay cost"`).
     - Verify in `managym/tests/rules/group.rs` and `tests/env/test_group.py`
       that two different lands/spells in hand produce distinct readable labels
       and non-`None` `source`s on `SingleDecision.offers`, and that
       `semantic_decision_frame().fingerprint` is completely unchanged.

   - **5b. Port `PartitionPiles` (`Decision::PartitionPiles`, Curator of Destinies) onto the unified `Decision` / `group_command` pipeline and delete the last browser send loop:**
     - Foundations (`10217a2f08`) merged `Decision::PartitionPiles` onto `main`,
       leaving `Board.svelte:455-543` running a legacy browser-side
       `partitionRunning` send loop with `FACE_UP_RE` regex scraping,
       `Stop sending`, `academy.table.partition-draft`, and `Rail.svelte`'s
       `stopped` prop beside the new `group_command` path.
     - Rust (`managym/src/flow/group/mod.rs` + `library.rs`): add
       `DecisionSpec::PartitionPiles { items: Vec<GroupCard>, committed_face_up: Vec<GroupCard>, committed_face_down: Vec<GroupCard> }`
       (threshold `remaining.len() >= 2`),
       `DecisionChoice::PartitionPiles { face_up: Vec<ObjectRenderId> }`, and
       `DecisionOutcome::PartitionPiles { face_up: Vec<ObjectRenderId>, face_down: Vec<ObjectRenderId> }`.
       Compile each remaining card in cursor order (`looked[*next..]`) to
       `Action::ChooseMode { mode: 0 }` (face-up) or `mode: 1` (face-down).
       Validate `face_up` IDs are distinct members of `items` (refusing unknown
       or already-committed cards). `continues()` stays true while
       `suspended_decision` is `Decision::PartitionPiles { player, .. }` for
       `viewer`; on the final card it transitions to `ChoosePile` and emits
       `CardRevealed` after `!continues()`, keeping the inter-step purity
       boundary clean.
     - Python & protocol: add `PartitionPilesDecision`, `PartitionPilesChoice`,
       `PartitionPilesOutcome` to `managym/group.py`, update
       `protocol/testing-house-v1.schema.json` (and fixture if needed), and
       verify `uv run python scripts/public_api.py check`. Academy's
       `live_group()`, `/group-preview`, and `hero_group()` require no new
       transport or recording code.
     - Browser (`academy/ui/src/lib/`): add `PartitionPilesGroup` /
       `PartitionPilesPlan` / `PartitionPilesOutcome` to `group.ts`; drive the
       existing two-pile partition UI in `Board.svelte` from `activeGroup`
       (`kind === 'partition_piles'`), `/group-preview`,
       `academy.table.group-draft`, and `ongroupact` (`group_command`). Delete
       `partitionRunning`, `stopPartitionSending()`, `FACE_UP_RE`,
       `FACE_DOWN_RE`, `partitionContext()`, `partitionOffer()`,
       `partitionSteps()`, `savePartitionDraft`/`loadPartitionDraft`
       (`academy.table.partition-draft`), and the `stopped` prop on
       `Rail.svelte`. Update `partition.test.ts` and
       `academy/ui/e2e/partition-piles.spec.ts` to assert 1-RTT atomic
       `group_command` submission, preview, refusal, and raw-offer disclosure.
     - Add Rust (`managym/tests/rules/group.rs`), Python
       (`tests/env/test_group.py`), and Academy
       (`tests/academy/test_group_command.py`) tests for `PartitionPiles`
       parity, mid-group raw stepping, watcher privacy, and `ChoosePile`
       transition.

   - **5c. Clear stale rail rejection banner on draft edit & sync docs:**
     - When `Board.svelte` calls `dismissRejection()` on editing a drafted group
       after a `group_refused` / `group_incomplete` rejection, clear the
       matching rejection error on `gameStore` (in `+page.svelte`) so the red
       banner in the rail disappears along with the inline panel refusal.
     - Update `AGENTS.md`, `docs/academy.md`, `managym/README.md`, and
       `scratch/questions.md` to reflect that `PartitionPiles` is ported (no
       browser send loops remain) and `Offer.label`/`source` are populated.

## Remaining (after Pass 5, in gate / ship-review)

- Full CI on PR #623 and live demo: `./scripts/play`, attack with several
  creatures and Confirm (one `group_command`), Scry 2 and arrange both cards
  before one Confirm, and verify Curator of Destinies pile split submits in one
  `group_command`.
- Ship-review: answer Jack's API questions against the Pass 5 surface and file
  the remaining follow-up families in `scratch/questions.md` ("pick", "amounts",
  `OrderTriggers`, discard instruction text, `0.3.0` removals).

## Done when

- `env.decision().offers` returns human-readable labels and populated
  `SubjectRef` sources on `SingleDecision` (e.g. distinguishing two lands or
  spells in hand) with zero changes to `semantic_decision_frame()` fingerprints.
- Confirming a drafted attack, block, scry, surveil, library order, or pile
  partition sends one `group_command`, and the game is wholly past that decision
  or exactly where it was with the engine's reason shown (and editing the draft
  clears the rejection banner cleanly).
- A game played with grouped Confirms and the same game played one Command at a
  time record the same decisions and reach the same state digest.
- Replay and Study show every per-step choice as before;
  `protocol/experience-v1.schema.json` is unchanged.
- Disconnecting or reloading mid-Confirm reconciles via `accepted_commands`
  without double-applying.
- The browser holds zero chunked send loops, zero step-verification loops, zero
  label regex scrapers (`FACE_UP_RE`), and zero legality rules; every drafted
  decision reads `group` and `/group-preview` and uses `academy.table.group-draft`.
- `uv run python scripts/public_api.py check` passes.
- `docs/academy.md` and `managym/README.md` match the built surface and name the
  remaining follow-up kinds.
