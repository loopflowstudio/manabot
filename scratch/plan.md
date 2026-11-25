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
engine's own verdict on the draft as they edit it, and confirms once. Confirm
is one message and one authority call: the whole plan lands or none of it
does. Nothing about a half-sent sequence can be shown, because there is none.

After this change that holds for combat declarations (attackers, blockers) and
for arranging looked-at cards (Scry, Surveil, OrderLibrary). The recording of
such a turn is byte-for-byte the per-step recording it is today: one recorded
decision per native step, each with its frame, offers and retained root, so
Replay, Study forks and policy probabilities attach exactly where they do now.

A Python caller of `managym` gets the same three calls (`group`,
`preview_group`, `apply_group`) with no Academy involved.

## Base (done in kickoff; choice made for review)

- **Based on main, no stack.** `lf sync main` ran in kickoff (merge
  `e2224c3988`); `stack_parent` is now null and the branch differs from main
  only in `scratch/`. Undo: `lf sync MTG-431`.
  Why: the native layer is additive (`flow/group/`), and every first consumer
  exists on main — `combat_preview.rs`, `Decision::Scry`, `Surveil`,
  `OrderLibrary`. MTG-431 (PR #605) is `CONFLICTING` with main and still
  planning engine slices; stacking would hold this Task behind it for a pile
  split that is one more group kind.
  Passed over: staying stacked to migrate the pile split here. It becomes a
  follow-up (see `scratch/questions.md`).
- `docs/academy.md` is main's again. The "Drafted decisions" section committed
  earlier described the chunked runner and the MTG-431 pile split; the audit
  it carried is kept below and rewritten in pass 3.
- MTG-441 (PR #613) merged 2026-11-25 and is in this branch (`3460ec64c5`);
  `academy/ui/src/lib/board/Board.svelte` declares `interaction:`.

## What was found

### Two grouped mechanisms already exist; one generalizes

- **`Game::combat_preview`** (`managym/src/flow/combat_preview.rs`, shipped in
  0.2.6 as `Env.combat_preview_json`, `managym/combat.py`). Checks the semantic
  address `(viewer, revision, fingerprint)`, clones, asks the clone's own
  action space for the action matching the plan's answer for the creature
  asked, records `(revision, fingerprint, action_index)`, steps, and stops
  when the decision is about none of the creatures still to declare. Coupled
  rules (menace, must-attack, must-block) need no special case: the engine
  refuses or repairs on the clone and the preview reports it.
- **`compound_offers` / `compound_commands` / `step_structured`**
  (`managym/src/agent/structured_offer.rs`). A static product descriptor
  (`ChoiceStep::Select { min, max, ordered, distinct }`) decoded and lowered on
  a fork. It returns the plain sequential surface whenever choices couple
  (`structured_blocker_offers` gives up on menace, blocking limits and
  must-be-blocked), has no refusal reasons, and is a policy surface: Study
  branches (`managym/academy/study_branch.py`) and `manabot/sim/action/benchmark.py`
  execute through it. Widening it would change what those consumers can
  submit, which the direction rules out.
  **Decision:** build on the `combat_preview` mechanism. `compound_*` is left
  exactly as it is and the new layer shares only its id types
  (`ObjectRenderId`, `SubjectRef`).

### Protocol schema split: `experience-v1` stays frozen, `testing-house-v1` adds `group_command`

- `ExperienceFrame` (`protocol/experience-v1.schema.json`) is hashed into
  `frame_hash` by `_build_frame` (`managym/academy/server.py`) and re-checked
  against retained replays by `replay_index.py`. Adding a `group` field inside
  `ExperienceFrame` would move every combat `frame_hash` and break retained
  replays.
  **Decision:** `protocol/experience-v1.schema.json` (`ExperienceFrame`,
  `Command`, `CommandReceipt`, `RecoveryEnvelope`, `ReplayDecision`) is
  untouched. The live `group` descriptor travels beside `frame` on live
  WebSocket messages, outside the canonical frame (like `log`, `auto_passed`,
  `boards`). Replay and Study frames never carry it, keeping raw per-step
  offers the only thing shown there.
- `shared_table_protocol.py` (`protocol/testing-house-v1.schema.json` and
  `protocol/fixtures/testing-house-control-v1.json`) is the closed WebSocket
  request envelope checked by `tests/academy/test_shared_table_protocol.py`
  (it already evolved for `rematch`, `retry_mirror`, etc.). Adding the
  `group_command` WebSocket request updates `shared_table_protocol.py`,
  `protocol/testing-house-v1.schema.json`, and
  `protocol/fixtures/testing-house-control-v1.json` together.

### The authority's per-step path and early-commit lifecycle are already what a group needs

- `hero_command` → `_apply_bound_command` → `_step_and_record` writes one
  `ReplayDecision` (frame, offer, command, retained `clone_env` root,
  presentation drain) per native step, then `committed(...)` sends early
  confirmation before `_advance()` runs bots and automatic passes. Inside a
  group the next decision is the same person's non-priority decision, so
  `_advance()` does nothing between steps today. Applying a group is
  therefore: validate on clone, then for each compiled step publish the prompt,
  verify native `(revision, fingerprint)`, `_apply_bound_command`; then emit
  one receipt under the group's `command_id`, call `committed(...)`, run one
  `_advance()`, and return the settled `command_outcome`.
- `accepted_commands` and `record.accepted_command_submitters` already handle
  idempotent retries (`status: duplicate`) before role authorization in
  `_dispatch_table_request`. Sharing that exact path for `GroupCommandMessage`
  and persisting pending `group_command` envelopes in `socket.svelte.ts`'s
  `ResumeCredentials.pending_command` gives grouped Confirms the exact same
  disconnect/reload recovery as single Commands.
- `_response_with_participant` runs for every outgoing message per recipient
  (including `observation`, `command_outcome`, `role_changed` on
  `TransferPilot`, and `_cache_table_recovery`). Attaching `payload["group"]`
  there — present only when `SUBMIT_LIVE_COMMAND` is in the recipient's role
  capabilities, `None` for watchers — prevents private Scry/Surveil cards from
  leaking to watchers AND hands a live group immediately to a newly promoted
  pilot without a reload.

### Native facts the compiler relies on (verified against `decision.rs`, `tick.rs`, `zones.rs`, `identity.rs`)

- **Object identities match across battlefield and inspected library cards:**
  `ObjectRenderId.entity` is `permanent.id` for battlefield permanents and
  `provenance(object).0` (`PublicId`) for inspected library cards (`identity.rs:84-101`),
  which equals `LegacyPermanentView.id` and `LegacyCardView.id` in
  `ExperienceFrame`.
- **`skip_trivial` (`flow/tick.rs:110`):** automatically executes any
  one-action space, so the last card of `OrderLibrary` (and any forced
  attacker/blocker) is never asked. Under `skip_trivial = true`, a 2-card
  `OrderLibrary` (such as immediately after keeping 2 cards on Surveil 2) asks
  only 1 step. The compiler clones with the game's `skip_trivial` setting, so
  it lists the exact steps that will be asked; no caller counts steps.
- **Scry bottom ordering (`flow/decision.rs:2806`):** `Decision::Scry` puts
  `top` via `put_on_top_of_library` (`top[0]` drawn first) and `bottom` via
  `for object in bottom.iter().rev() { reposition(..., 0); }`, which places
  `bottom[0]` at index `0` (the *very bottom*, drawn last) — the opposite of
  `OrderLibrary { end: Bottom }` (`decision.rs:2938`), where the first pick
  sits *above* later picks. `GroupPlan::Scry.bottom` uses consistent
  top-to-bottom reading order (`bottom[0]` drawn first among bottomed cards,
  `bottom.last()` on the very bottom), and the Scry compiler emits
  `ScryCard(..., to_bottom=true)` in `bottom.iter().rev()` order so the engine
  commits the exact order the user drafted.
- **Surveil (`flow/decision.rs:2091, 3145, 3685`):** offers `SelectCard(c)` for
  each remaining card for the graveyard plus `Decline` ("Keep on top"). When
  all remaining cards are chosen for the graveyard, it finishes automatically
  without `Decline`; otherwise `Decline` commits the graveyard batch and
  immediately opens `Decision::OrderLibrary { end: Top }` if 2+ cards were
  kept. That `OrderLibrary` is a separate decision and therefore its own group.
- **Hidden-info signals (`flow/zones.rs:181`, `flow/event.rs`):**
  `shuffle_library` and `remember_library_objects` emit no `GameEvent`, and
  `observation_events` does not include `CardDrawn` or `CardRevealed`. Also,
  the *final* step of a Scry (e.g., Preordain: "Scry 2, then draw a card")
  calls `run_frame` and draws immediately after Scry completes. Therefore the
  purity boundary checks **between step `i` and step `i+1` (only while
  `continues()` is still true)** that: (1) no `CardDrawn` or `CardRevealed`
  was appended to `state.events`, (2) `state.rng` did not advance (catching
  every shuffle/randomization), and (3) the viewer's visible hidden-zone
  objects (hand + `inspected_library_objects`) gained no unseen `ObjectId`.

## Design

### Source of truth

The engine's sequential `Action`s. A group is a *reading* of the pending
decision plus a *compiler* from a typed plan to the action indices the engine
will accept, in order. It adds no `Action`, no `ActionSpaceKind`, no
observation field, no game state and no digest input. `Game::step` is the only
thing that changes a game.

### Native layer: `managym/src/flow/group/`

`mod.rs` holds the shared envelope types and the one clone-and-step driver; one
file per family (`combat.rs`, `library.rs`).

The **envelope** (`Group`, `GroupStep`, `GroupRefusal`, `GroupPreview`) is
identical across every kind. The **descriptor** (`GroupSpec`), **plan**
(`GroupPlan`), and **outcome** (`GroupOutcome`) are tagged enums by kind so
every kind carries exact domain fields (such as `OrderLibrary`'s `end`,
`DeclareBlockers`'s `can_block`, and already-committed choices) with no
overloaded `required` boolean, no dummy `Answer::Next`, and clean extension to
future families (`PartitionPiles`, `SelectCards`, `Distribute`).

```rust
/// A card item in a library arrangement group.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GroupCard {
    pub item: ObjectRenderId,
    pub label: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AttackerItem {
    pub item: ObjectRenderId,
    pub label: String,
    pub must_attack: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BlockerItem {
    pub item: ObjectRenderId,
    pub label: String,
    pub can_block: Vec<ObjectRenderId>,
    pub must_block: bool,
}

/// The grouped decision `viewer` faces now, bound to its native semantic address.
#[derive(Clone, Debug, Serialize)]
pub struct Group {
    pub revision: u64,
    pub fingerprint: String,
    #[serde(flatten)]
    pub spec: GroupSpec,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroupSpec {
    DeclareAttackers {
        declarable: Vec<AttackerItem>,      // in engine asking order
        committed: Vec<ObjectRenderId>,     // already declared (skip_trivial or raw offer)
    },
    DeclareBlockers {
        declarable: Vec<BlockerItem>,       // in engine asking order
        committed: Vec<PlannedBlock>,       // already declared blocks
    },
    Scry {
        items: Vec<GroupCard>,              // remaining cards to place
        committed_top: Vec<ObjectRenderId>, // already placed on top, top-first
        committed_bottom: Vec<ObjectRenderId>, // already placed on bottom, top-of-bottom first
    },
    Surveil {
        items: Vec<GroupCard>,              // remaining cards to decide
        committed_graveyard: Vec<ObjectRenderId>, // already marked for graveyard
    },
    OrderLibrary {
        end: LibraryEnd,                    // Top or Bottom (serialized snake_case)
        items: Vec<GroupCard>,              // remaining cards to order
        committed: Vec<ObjectRenderId>,     // already placed prefix, drawn-first first
    },
}

/// The whole plan a caller drafts for the current group.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupPlan {
    DeclareAttackers {
        #[serde(default)]
        attackers: Vec<ObjectRenderId>,
    },
    DeclareBlockers {
        #[serde(default)]
        blocks: Vec<PlannedBlock>,
    },
    /// `top` is drawn-first at index 0. `bottom` is top-to-bottom of the bottom
    /// group (`bottom[0]` drawn first among bottomed cards; `bottom.last()` on
    /// the very bottom), matching `OrderLibrary { end: Bottom }`.
    Scry {
        #[serde(default)]
        top: Vec<ObjectRenderId>,
        #[serde(default)]
        bottom: Vec<ObjectRenderId>,
    },
    /// Cards to put into the graveyard; all remaining looked-at cards stay on top.
    Surveil {
        #[serde(default)]
        graveyard: Vec<ObjectRenderId>,
    },
    /// Remaining cards in desired order, drawn-first at index 0.
    OrderLibrary {
        #[serde(default)]
        order: Vec<ObjectRenderId>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupStep {
    pub revision: u64,
    pub fingerprint: String,
    pub action_index: usize,
    /// The object answered at this step, or `None` for a group-ending `Decline` (Surveil).
    pub item: Option<ObjectRenderId>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupRefusal {
    pub item: Option<ObjectRenderId>,
    pub label: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroupOutcome {
    DeclareAttackers { attackers: Vec<ObjectRenderId> },
    DeclareBlockers { blocks: Vec<PlannedBlock> },
    Scry { top: Vec<ObjectRenderId>, bottom: Vec<ObjectRenderId> },
    Surveil { graveyard: Vec<ObjectRenderId>, kept: Vec<ObjectRenderId> },
    OrderLibrary { end: LibraryEnd, order: Vec<ObjectRenderId> },
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupPreview {
    pub group: Group,
    pub steps: Vec<GroupStep>,
    pub refused: Option<GroupRefusal>,
    pub complete: bool,
    pub outcome: Option<GroupOutcome>,
}
```

#### Activation threshold (domain rule, not internal step count)

`Game::group(viewer) -> Option<Group>` returns `None` when `viewer` is not the
current actor, the game is over, or the pending decision is not one of the
supported kinds above its natural domain minimum:

- `DeclareAttackers` / `DeclareBlockers`: `declarable.len() >= 1` (even a
  single attacker or blocker keeps board drafting and MTG-441 drag/Confirm
  consistent).
- `Scry` / `Surveil` / `OrderLibrary`: remaining `items.len() >= 2` (2-card
  `OrderLibrary` after Surveil 2 stays in `ArrangePanel` even though
  `skip_trivial` collapses it to 1 native step; 1-card Scry/Surveil stays a
  1-click raw offer).

#### Entry points on `Game`, mirrored on `Env`

- `group(viewer) -> Option<Group>`: reads the pending decision for `viewer`.
- `preview_group(viewer, revision, fingerprint, &plan) -> Result<GroupPreview, AgentError>`:
  errors on stale/non-actor address, no active group, or a plan whose `kind`
  differs from the active group; a plan the engine will not accept is
  `refused`, and a structurally incomplete plan (unplaced card in Scry or
  OrderLibrary) returns `complete: false, refused: None, steps: [], outcome: None`
  without cloning. The live game is only read.
- `apply_group(viewer, revision, fingerprint, &plan) -> Result<GroupPreview, AgentError>`:
  runs `preview_group`; a `refused` or `!complete` preview is an `AgentError`
  and the live game is untouched; otherwise steps the live game through
  `steps`, verifying each step's `(revision, fingerprint)` before
  `step(action_index)`.

#### Kind trait & shared clone-and-step driver (`flow/group/mod.rs`)

Each kind implements one small internal trait selected by a `match` on the
pending decision:

- `describe(&Game, viewer) -> Option<GroupSpec>`: reads live fields and
  committed state.
- `validate(&Game, &GroupSpec, &GroupPlan) -> PlanCheck` (`Ready` |
  `Incomplete` | `Refused(GroupRefusal)`): checks plan size cap, duplicates,
  unknown/unaskable items, and missing required items before cloning.
- `next_step(&Game /* clone */, &GroupPlan, &mut Progress) -> NextStep`
  (`Step { action_index, item }` | `Done` | `Refuse(GroupRefusal)`): picks the
  matching action from the clone's own `ActionSpace`.
- `continues(&Game /* clone */, viewer, &Progress) -> bool`: true while the
  clone still asks this same group of `viewer`.
- `finish(&Game /* live */, &Game /* clone */, events_from, &GroupPlan, &Progress) -> (GroupOutcome, Option<GroupRefusal>)`:
  reads what the engine actually committed (e.g., `CombatAttackersDeclared`,
  `BlockersDeclared`, final library/graveyard placement) and reports any
  post-declaration refusal (menace removing a lone block, skipped must-attack).

`combat.rs` is today's `combat_preview.rs` moved under this interface, its
refusal sentences unchanged. `Game::combat_preview`, `Env::combat_preview`,
`Env.combat_preview_json` and `managym/combat.py` stay as wrappers over
`preview_group` (mapping `CombatPlan`/`CombatPreview` and reading
`outcome` even on `undeclared` refusal so `declared_blocks` parity holds),
tagged for removal in `0.3.0` with a `DeprecationWarning` on the Python call
(shipped in 0.2.6; main stays patch-clean).
`managym/tests/rules/combat_preview.rs` stays unedited as the proof of combat
parity.

### Purity boundary

A group is the consecutive steps of one pending decision by one player. The
driver stops, and sets `complete = true`, when `continues` becomes `false`
(another decision, another chooser, or game over). Surveil ends when its
graveyard batch commits; the `OrderLibrary` that follows is its own group with
its own Confirm. `OrderTriggers` is excluded because modes and targets can
interleave between picks.

Before executing any **subsequent** step (i.e., after step `i` when
`continues()` is still `true`, before compiling step `i+1`), the driver
asserts the clone crossed no hidden-information boundary:

1. No `GameEvent::CardDrawn` or `GameEvent::CardRevealed` was appended to
   `position.state.events` during step `i`,
2. `position.state.rng` state did not change during step `i` (catches every
   `shuffle_library` and random ordering),
3. Every hidden-zone object visible to `viewer` (hand + `inspected_library_objects`)
   was already visible at step 0.

If any check trips while `continues()` is still `true`, the driver stops and
returns what it compiled so far with `complete: false, outcome: None` (so
`apply_group` / `hero_group` refuse it). Effects that fire *after* the final
step exits the group — such as Preordain drawing a card right after Scry 2
finishes — happen when `continues()` is `false` and complete cleanly with
`complete: true`. Pass 1 tests both every kind's fixture AND Preordain
explicitly.

`steps` carry native fingerprints of states the viewer has not reached yet;
they belong to the authority. Academy's REST preview endpoint returns
`{ match_id, revision, prompt_id, frame_hash, viewer, complete, refused, outcome }`
and never exposes `steps`.

### Python surface

`Env.group_json(viewer)`,
`Env.preview_group_json(viewer, revision, fingerprint, plan_json)`,
`Env.apply_group_json(viewer, revision, fingerprint, plan_json)` (which
honours `reject_guarded_mutation` as `step_structured` does), typed in
`managym/group.py` with Pydantic models matching `Group`, `GroupSpec`,
`GroupPlan`, `GroupOutcome`, `GroupPreview`, declared in `managym/__init__.pyi`,
and recorded with `uv run python scripts/public_api.py record` (non-breaking
additions only).

### Academy authority (`managym/academy/` & `protocol/testing-house-v1`)

- **Descriptor.** `GameSession.live_group()` reads `Env.group_json(self.human_seat)`
  for the published prompt and wraps it with the Academy address
  `{ match_id, revision, prompt_id, frame_hash }`, cached alongside
  `published_prompt` and cleared on `_advance_protocol_revision()`.
  `_response_with_participant` attaches `payload["group"]`:
  - `record.game.live_group()` (object or `None`) when
    `TableCapability.SUBMIT_LIVE_COMMAND in ROLE_CAPABILITIES[participant.role]`,
  - `None` otherwise (watchers never receive private Scry/Surveil descriptors).
  Because `_response_with_participant` runs on every outgoing response
  (`observation`, `command_outcome`, `table_snapshot`, `role_changed`, and
  `_cache_table_recovery`), a Watcher promoted via `TransferPilot` immediately
  receives the live group descriptor with no reload.
- **Preview.** `POST /api/sessions/{id}/group-preview` accepts
  `{ match_id, revision, prompt_id, frame_hash, plan }`, runs the same
  participant/role/staleness checks as `/combat-preview` (`403
  group_preview_forbidden`, `409 group_preview_stale`, `409
  group_preview_no_group`), calls `preview_group`, and returns
  `{ match_id, revision, prompt_id, frame_hash, viewer, complete, refused, outcome }`
  (never `steps`). Read-only.
- **Apply.** Add `GroupCommandMessage` (`type: "group_command"`,
  `grant_revision`, `command: { command_id, match_id, expected_revision,
  prompt_id, frame_hash, plan }`) to `shared_table_protocol.py`,
  `protocol/testing-house-v1.schema.json`, and
  `protocol/fixtures/testing-house-control-v1.json`.
  `DISPATCH_CAPABILITIES["group_command"] = TableCapability.SUBMIT_LIVE_COMMAND`.
  `_dispatch_table_request` runs the same pre-auth duplicate-receipt and
  submitter-ownership check for `(CommandMessage, GroupCommandMessage)`.
  `GameSession.hero_group(raw, committed=None)`:
  1. Reject reserved `command_id` prefixes (`"authority."`, `"compound-"`).
  2. Duplicate `command_id` in `accepted_commands` → stored receipt
     (`status: duplicate` + `recovery`).
  3. Match/prompt/actor/revision/`frame_hash` checks → `wrong_match`,
     `authority_busy`, `stale_revision`, `stale_prompt`, `not_actor` (with
     `recover=True` on stale/busy, matching `hero_command`).
  4. `preview_group`:
     - `refused` → `_reject_command(..., "group_refused", message=refused.reason, refused=refused, recover=False)`;
     - `!complete` → `_reject_command(..., "group_incomplete", message=..., recover=False)`;
     - `AgentError` (wrong kind / no group) → `stale_revision` with recovery.
  5. Record `base_revision = context.revision` and `batch_cursor = self.presentation.next_seq`.
     For each `(step_index, step)` in `enumerate(preview.steps)`:
     - publish the current prompt (`_publish_current_prompt()`),
     - assert its native `(revision, fingerprint)` equals `(step.revision, step.fingerprint)`
       and `context.action_by_offer.get(step.action_index) == step.action_index`
       (a mismatch raises `RuntimeError`, indicating clone divergence),
     - build the bound step command with collision-free
       `command_id = f"authority.group.{command_id}.{step_index}"`,
     - call `_apply_bound_command(context, step_command, source="client")`.
  6. Mint one receipt under the group's `command_id` (`accepted_at` = first
     step's revision, `resulting_revision` = `self.revision`,
     `resulting_frame_hash` = `committed_frame["frame_hash"]`), store in
     `accepted_commands`, call `committed(...)` if supplied (emitting the early
     commitment frame across `base_revision -> resulting_revision` before any
     slow bot runs), run one `self._advance()`, and return the settled
     `_accepted_outcome(...)`.
  No `ReplayDecision`, `ExperienceFrame`, or `experience-v1.schema.json`
  changes: the recorded rows are identical to N single Commands.
- `/combat-preview`, `CombatPreviewRequest` and `tests/academy/test_combat_preview.py`
  go; the test's cases move to `tests/academy/test_group_command.py`.

### Browser (`academy/ui/`), on top of MTG-441

- `gameStore` keeps `liveGroup` whenever a server message carries `'group' in message`
  and `group?.frame_hash === protocolFrame?.frame_hash`; otherwise `null`.
  No group: the board is purely raw offers.
- `socket.svelte.ts`:
  - Adds `sendGroupCommand(plan)` sending `type: "group_command"` bound to the
    active frame's `(match_id, revision, prompt.id, frame_hash)` with a fresh
    `command_id`.
  - Widens `PendingCommand` → `PendingSubmission = CommandMessage | GroupCommandMessage`,
    persisted in `ResumeCredentials.pending_command` and reconciled in
    `reconcileCommand(recovery)` against `recovery.accepted_commands` and
    `(match_id, revision, prompt.id, frame_hash)` so a disconnect/reload mid-Confirm
    behaves identically to a single Command.
  - Replaces `fetchCombatPreview` with `fetchGroupPreview` calling
    `POST /api/sessions/{id}/group-preview`.
  - Extends `CommandRejection` with optional `refused?: GroupRefusal | null`.
- Combat keeps MTG-441's draft, drag, marquee and rail controls. Underneath:
  `group.kind === 'declare_attackers' | 'declare_blockers'` supplies
  `declarable` and `committed` immediately with no bootstrap fetch; draft edits
  call `fetchGroupPreview`; Confirm calls `sendGroupCommand`; and the `running`
  send loop, `stepOffer`, `Stop sending` and every partial-progress sentence
  are deleted. A `group_refused` rejection shows the engine's reason on the
  draft and keeps it for editing.
- One `ArrangePanel.svelte` for `scry`, `surveil`, and `order_library`, driven
  solely by `group` (`GroupSpec`) and the preview response (zero label scraping
  or `pending` text parsing):
  - `scry`: two ordered buckets ("Top of library — drawn first at top",
    "Bottom of library — top-most at top"); remaining cards start unplaced,
    Confirm disabled until every card is placed and preview says `complete`
    without `refused`; shows read-only "Already placed" summary when
    `committed_top`/`committed_bottom` is non-empty.
  - `surveil`: two buckets ("Keep on top", "Graveyard"); remaining cards start
    in "Keep on top" (matching `Action::Decline` = "Keep on top") with an
    explicit Confirm label naming the exact commitment (`"Keep all N on top"`,
    `"Put K in graveyard, keep M on top"`, `"Put all N in graveyard"`); shows
    read-only "Already in graveyard" summary when `committed_graveyard` is
    non-empty.
  - `order_library`: one ordered list titled from `group.end` ("Top of library
    — drawn first at top" or "Bottom of library — drawn first at top");
    remaining cards start unplaced, Confirm disabled until all are ordered;
    shows read-only "Already ordered" prefix when `committed` is non-empty.
  - Full click and keyboard reordering/moving (parity with Academy a11y).
- The rail's collapsed disclosure of raw offers (MTG-441's
  `per-creature-choices`) takes its summary from `group.kind` and is shown for
  every active group. Raw offers remain clickable ordinary `command`s; the next
  frame brings the updated `group` showing what committed and what remains.
- Drafts persist in `sessionStorage` under one key scoped to
  `(match_id, frame_hash, kind)`, replacing the combat-only key.
- While a `group_command` is pending, the draft stays drawn and disabled, the
  way any pending Command leaves the board (MTG-441's `pending`).

### Choices made for review (see `scratch/concept-review.html`)

- **Based on main; pile split deferred to a follow-up after MTG-431.**
- **Tagged `GroupSpec` / `GroupPlan` / `GroupOutcome` over a flat `Vec<Choice>`:**
  keeps `OrderLibrary`'s `end`, committed state, and per-kind outcomes explicit
  and extends cleanly to `PartitionPiles`, `Pick`, and `Distribute`. Passed
  over: flat `GroupItem { answers, required }` with dummy `Answer::Next`.
- **Domain activation thresholds (`>= 1` creature for combat; `>= 2` remaining
  cards for Scry/Surveil/OrderLibrary):** keeps 1-creature combat drafting
  (MTG-441) and 2-card `OrderLibrary` (after Surveil 2) consistent regardless
  of `skip_trivial`. Passed over: suppressing `group()` whenever `steps <= 1`.
- **First non-combat kinds are Scry, Surveil and OrderLibrary:** one arrange
  panel, exercises ordered plans and `skip_trivial` tails, and Scry is printed
  across `managym/cards/src/tla.rs`. Passed over: shipping native kinds with no
  browser consumer.
- **Surveil starts in "Keep on top" with explicit Confirm label; Scry and
  OrderLibrary start unplaced:** mirrors the engine's actual `Decline` ("Keep
  on top") semantics and Combat's `"No attack"` default without silently
  choosing for Scry/OrderLibrary.
- **Scry `bottom` plan order is top-to-bottom of the bottom group** (matching
  `OrderLibrary::Bottom` and human reading order); the Scry compiler reverses
  when emitting `ScryCard(to_bottom=true)` to match `decision.rs:2806`.
- **Apply is all-or-nothing.** A person who wants one step uses the raw offer.
- **`experience-v1.schema.json` frozen; `testing-house-v1.schema.json` adds
  `group_command`.** Live `group` travels beside `frame` in
  `_response_with_participant` so watchers never receive private library cards
  and `TransferPilot` delivers the active group immediately.
- **Per-step recorded command ids use `authority.group.<id>.<n>` with
  `source: "client"`:** collision-free with client command ids, zero replay
  schema change.
- **`apply_group` exists natively even though Academy loops `steps` itself,**
  because Python `managym` callers get atomic one-call submission while Academy
  captures a frame, retained Study root, and presentation drain around each
  step. Pass 1 pins them digest-equal.

## Acceptance check at gate

```
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -E 'test(group) | test(combat_preview)' \
  && uv run python scripts/public_api.py check \
  && uv run --extra dev --extra play python -m pytest tests/academy/test_group_command.py tests/academy/test_shared_table_protocol.py -q \
  && (cd academy/ui && npm run check && npm test \
      && npx playwright test e2e/grouped-decisions.spec.ts e2e/combat-presentation.spec.ts e2e/drag-play.spec.ts e2e/keyboard-play.spec.ts)
```

All pass. Scenarios pinned across layers:

1. **Native** (`managym/tests/rules/group.rs`):
   - Three-creature attack, 1-creature attack, multi-block, menace refusal,
     must-attack/must-block: `apply_group(plan)`, stepping
     `preview_group(plan).steps` by hand, and the unedited `combat_preview`
     test's `follow` reach the same `state_digest`. Refused, incomplete, and
     stale calls leave the digest untouched.
   - **Scry 3** (with 1 top, 2 bottom — explicitly verifying bottom card
     library order matches top-to-bottom!), **Preordain** (Scry 2 then draw 1 —
     verifying the post-group draw does NOT trip the purity boundary),
     **Surveil 2 → OrderLibrary (2 cards, `skip_trivial=true` and `false`)**,
     **OrderLibrary (`Top` and `Bottom`)**, and **mid-group raw step** (take 1
     raw step, check `group()` exposes `committed_*` and remaining items, then
     finish via `apply_group`): all match manual stepping digest-for-digest and
     pass the inter-step hidden-info boundary assertion.
2. **Authority** (`tests/academy/test_group_command.py` + `test_shared_table_protocol.py`):
   - Two sessions from one seed and config: one sends three `command`s, the
     other one `group_command`. Their `canonical_decisions` agree row for row
     on viewer, source, offer, projection, frame hash and revision, and their
     native state digests are equal.
   - Over WebSocket: sending 1 `group_command` commits all 3 steps atomically
     (early commit `command_outcome` across `rev -> rev+3` before `_advance()`,
     followed by settled `command_outcome` if the bot moved — zero intermediate
     half-declared frames).
   - Refused plan returns `group_refused` with the engine's sentence and
     `refused` payload, `recover=False`, 0 rows written.
   - Incomplete plan returns `group_incomplete`, 0 rows written.
   - Resending the same `command_id` returns `duplicate` (even after pilot
     transfer for the original submitter) and writes 0 rows; another
     participant reusing that `command_id` gets `forbidden`.
   - Watcher receives `group: null` during Scry/Surveil; after `TransferPilot`,
     the `role_changed` message delivers the live `group` to the new pilot and
     `null` to the former pilot.
3. **Browser** (`e2e/grouped-decisions.spec.ts`):
   - Drafting 3 attackers (and 1 attacker) and pressing Confirm sends exactly
     one `group_command` socket message.
   - Scry 2 opens `ArrangePanel`, Confirm is disabled until both cards are
     placed, and one `group_command` is sent.
   - Opening the disclosure lists raw per-card offers; clicking one sends an
     ordinary `command` and updates the panel with the committed card.
   - Surveil 2 → OrderLibrary transitions cleanly across two grouped Confirms.

Demo: `./scripts/play`, attack with several creatures and press Confirm (network
panel shows one `group_command` out), then cast a Scry 2 spell and arrange both
cards before confirming once.

## Absent and error states

- No group (not a grouped kind, below domain threshold, not the chooser,
  Replay, Study, Watch, recording, watcher role): `group` is `null` or absent;
  raw offers only.
- Stale preview (`409`): browser drops the answer and asks again at the current
  frame; a draft whose `frame_hash` no longer matches is discarded.
- `group_refused`: nothing applied, frame unchanged, draft kept, engine's
  reason shown on the refused item.
- `group_incomplete`: nothing applied, frame unchanged, draft kept, naming what
  is unplaced. The browser disables Confirm while incomplete; the code exists
  for direct clients.
- Stale `group_command`: rejected with recovery, nothing applied, draft
  discarded with the frame it belonged to.
- Connection lost or tab reloaded during apply: `PendingSubmission` is stored
  in `etude.gui.resume`. On reconnect, `reconcileCommand` checks
  `recovery.accepted_commands` (clearing pending if committed) or resends the
  exact `group_command` if the recovered frame still matches `(match_id,
  revision, prompt.id, frame_hash)`; duplicate receipt prevents double-apply.
- A compiled step that does not match the live game: `RuntimeError` (clone
  divergence bug; steps already applied stay recorded as ordinary decisions).
- Storage unavailable: no saved draft, no error.

## Operational boundary

- Preview: no clone when structurally incomplete or over size cap; otherwise
  one `Game::clone` and at most `items + 1` steps. Debounced in the browser as
  combat is today.
- Apply: one clone plus the same steps, under `record.lock` for the whole call,
  as a single Command is.
- No engine hot path changes; `managym/perf/baseline.json` digests must not
  move. A moved digest means the layer touched state and is a defect.

## Exclusions

- New `Action` or `ActionSpaceKind` variants, observation or policy-input
  changes, any change to `compound_*` / `step_structured`, `ExperienceFrame`,
  `ReplayDecision`, or `protocol/experience-v1.schema.json`.
- Group kinds beyond the five above (table below), and the MTG-431 pile split.
- Grouped input in Study, Replay or Watch.
- Resuming or undoing anything: there is no partial state to resume.

## What goes

- `managym/src/flow/combat_preview.rs` as a module of logic (its body moves to
  `flow/group/combat.rs`; a `0.3.0` wrapper remains).
- Server: `/combat-preview`, `CombatPreviewRequest`, `tests/academy/test_combat_preview.py`.
- Browser: `fetchCombatPreview`, the `running` send loop and its `$effect` in
  `Board.svelte`, `stepOffer` and `DeclarationStep` in `combat.ts`, the
  `Stop sending` control and stop note with their test ids, the combat-only
  draft storage key.
- `managym/README.md`'s combat-preview paragraphs (rewritten as the grouped
  input section).

Survives unchanged: every `Action`, digest and recorded row; `experience-v1`
schema; MTG-441's drag and draft UX (including 1-creature combat); combat's
refusal sentences; `compound_*`; `Env.combat_preview_json` until `0.3.0`.

## The audit, reclassified

With a native compiler the earlier A1/A2/B split disappears: "needs typed
fields" is `GroupSpec` and "needs clone-and-step" is the driver. What
separates decisions now is the shape of the plan and where the group ends.

| Decision (stage) | Plan shape | Group ends | Status |
|---|---|---|---|
| DeclareAttackers / DeclareBlockers | attackers list / blocker→attacker pairs | declaration event | this Task (ported) |
| Scry (`scry`), 2+ cards | ordered `top` + ordered `bottom` | scry finishes | this Task |
| Surveil (`surveil`), 2+ cards | `graveyard` subset | picks commit; `OrderLibrary` follows as its own group | this Task |
| OrderLibrary (`order_library_top` / `_bottom`), 2+ cards | ordered list + `end` | last card placed (`skip_trivial` may skip final step) | this Task |
| PartitionPiles (`partition_piles`, MTG-431) | two piles | split finishes; `ChoosePile` is the opponent's | follow-up after MTG-431 |
| SelectCards with max > 1, SearchDistinctNames, Discard of 2+, OpponentSplitsRevealed, KeepCreatures (count or total power) | subset within engine limits | selection finishes | follow-up: "pick" kind and panel |
| DistributeCounters, DivideAmongTargets (`target_shares`), AssignCombatDamage | amount per recipient (with lethal minimums) | amounts exhausted | follow-up: "amounts" kind and panel |
| OrderTriggers | order | only while no trigger asks a mode or target between picks | follow-up, behind the boundary test |
| Retarget, ManaUpfront, ManaSpend, OrderEntryChoices, OrderReplacement | — | — | stays sequential: later options are legality- or planner-filtered, usually one or two steps |
| LookAndSelect, ManifestDread, ChooseToExile, DiscardFromRevealedHand, OrderMeldedCards, ExileLinked, LegendRule, ChoosePile | — | — | one step |

A follow-up kind is one `flow/group/` file + `GroupSpec`/`GroupPlan`/`GroupOutcome`
variant plus a panel for its shape; it needs no server, protocol or recording
change.

## User docs written in place

- `docs/academy.md`: added **Grouped decisions** (combat drafting, Scry / Surveil / OrderLibrary arrangement panel, raw per-step disclosure, reload/reconnect/watcher/recording invariants, purity boundary, and reclassified audit table) and updated **Playing on the board** to remove the browser send loop.
- `managym/README.md`: replaced **Previewing a whole combat declaration** with **Grouped input** (`managym.group` `group`, `preview_group`, `apply_group`, plan/descriptor/outcome types, Scry bottom reading order, purity boundary, and `0.3.0` deprecation note for `combat_preview`) while preserving the combat rules notes under **Combat declaration and damage rules**.

## Passes

1. **Native layer with combat parity and the library kinds** — done (`b3ae75976c`).
   Realign 2026-11-25: `tests/env/test_group.py` (2 tests) and
   `scripts/public_api.py check` pass; the 7 Rust tests in
   `managym/tests/rules/group.rs` were not rerun here (gate runs them in debug).
2. **Academy authority & `testing-house-v1` schema** — done (`b510844e42`).
   `tests/academy/test_group_command.py` (6 tests),
   `tests/academy/test_shared_table_protocol.py` (4 tests),
   `tests/academy/test_shared_table.py` (6 tests), and
   `scripts/public_api.py check` pass.
   Realign 2026-11-25: the server no longer serves `/combat-preview`, but the
   browser still calls it (`socket.svelte.ts:1079` `fetchCombatPreview`,
   `Board.svelte` `combatPreview` prop, `+page.svelte:554`, mocked in
   `e2e/drag-play.spec.ts:345`). Live combat drafting in the browser is broken
   until pass 3 replaces it; this branch must not ship without pass 3.
3. **Browser and docs** — done (`c870fb9349`). `svelte-check` (0 errors, 0
   warnings), unit tests (`design`, `socket`, `shared-table-protocol`, `group`,
   `arrange`, `combat`), and Playwright acceptance specs
   (`grouped-decisions`, `combat-presentation`, `drag-play`, `keyboard-play`,
   34/34) pass. All passes complete; ready for gate.

## Done when

- Confirming a drafted attack, block, scry, surveil or library order sends one
  `group_command` message, and the game is either wholly past that decision or
  exactly where it was with the engine's reason shown.
- A game played with grouped Confirms and the same game played one Command at
  a time record the same decisions and reach the same state digest.
- Replay and Study of such a game show every per-step choice as they do now,
  and `protocol/experience-v1.schema.json` is unchanged.
- Disconnecting or reloading mid-Confirm reconciles via `accepted_commands`
  without double-applying or leaving a half-committed group.
- The browser holds no send loop, no step verification and no rule about
  which answers are legal; it reads `group` and `/group-preview`.
- `uv run python scripts/public_api.py check` passes with the three new `Env`
  methods and `combat_preview_json` still present.
- `docs/academy.md` carries the reclassified audit and names the follow-up
  kinds.
