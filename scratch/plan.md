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
- `scratch/concept-review.html` (the chunked-runner concept review) is deleted.
- **MTG-441 (PR #613) must be on main before pass 3.** It has auto-merge on
  and is `MERGEABLE`, with `Academy UI` failing as of 2026-11-25. It rewrites
  the combat draft in `Board.svelte`, `Rail.svelte` and `+page.svelte`. Passes
  1 and 2 touch no file under `academy/ui/`. Pass 3 starts with `lf sync` and
  checks that `Board.svelte` declares `interaction:`; if it does not, pass 3
  has not got its input and the build says so and stops there.

## What was found

### Two grouped mechanisms already exist; one generalizes

- **`Game::combat_preview`** (`managym/src/flow/combat_preview.rs`, shipped in
  0.2.6 as `Env.combat_preview_json`, `managym/combat.py`). Checks the semantic
  address `(viewer, revision, fingerprint)`, clones, asks the clone's own
  action space for the action matching the plan's answer for the creature
  asked, records `(revision, fingerprint, action_index)`, steps, and stops
  when the decision is about none of the creatures still to declare. Coupled
  rules (menace, must-attack, must-block) need no special case: the engine
  refuses or repairs on the clone and the preview reports it. Nothing in the
  loop is combat-specific except "which creature is asked", "which action
  answers it" and "what did the engine declare".
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

### The descriptor cannot go inside `ExperienceFrame`

`_build_frame` (`managym/academy/server.py`) hashes the whole frame into
`frame_hash`; `ExperienceFrame` is `deny_unknown_fields` and byte-compared with
`protocol/experience-v1.schema.json`; `replay_index.py` compares a re-derived
frame's hash with the recorded address. A `group` field would move the hash of
every combat frame and break retained replays.
**Decision:** the descriptor travels beside the frame on live envelopes, the
way `log`, `auto_passed` and `boards` already do (`_drain_surface_metadata`:
"outside the canonical frame"). Replay and Study frames never carry it, which
is also what keeps their raw per-step offers the only thing shown.

### The authority's per-step path is already what a group needs

- `hero_command` → `_apply_bound_command` → `_step_and_record` writes one
  `ReplayDecision` (frame, offer, command, retained `clone_env` root,
  presentation drain) per native step, then `_advance()` runs bots and
  automatic passes. Inside a group the next decision is the same person's
  non-priority decision, so `_advance()` does nothing between steps today.
  Applying a group is therefore: for each compiled step, publish the prompt,
  check it is the step the compiler predicted, `_apply_bound_command`; then
  one `_advance()`.
- `accepted_commands` already gives a retried `command_id` its receipt
  (`status: duplicate`). A group's one client `command_id` gets the same
  treatment, so a response lost to a disconnect cannot double-apply.
- `_send_table_response` is the one place a table message leaves the server,
  per recipient. It is where the descriptor is attached, for recipients whose
  role has `SUBMIT_LIVE_COMMAND` only: a Scry descriptor names the looked-at
  cards.

### Native facts the compiler relies on (audit, carried over)

- `skip_trivial` answers any one-action space and loops (`flow/tick.rs`), so
  the last step of a sequence is often never asked (last card of
  `OrderLibrary`, a creature with one legal answer). The compiler steps a
  clone with the same setting, so it lists exactly the steps that will be
  asked. No caller counts steps.
- Scry offers every remaining card with both sides each step; pick order is
  final order. Surveil is "select cards for the graveyard, then `Decline`
  ('Keep on top')"; the graveyard picks commit as one batch and a separate
  `OrderLibrary` decision follows for two or more kept cards.
- Limits (`min`/`max`, `remaining`, lethal `minimum`) are typed fields of the
  pending `Decision`; only their `Debug` rendering reaches
  `decision_context.pending` today. A descriptor reads the fields.
- Mid-`Decision` steps do not draw, shuffle or pass the choice to another
  player: triggers and state-based actions wait for the resolving instruction
  to finish. This is the claim pass 1 tests rather than assumes (see
  Boundary).

## Design

### Source of truth

The engine's sequential `Action`s. A group is a *reading* of the pending
decision plus a *compiler* from a typed plan to the action indices the engine
will accept, in order. It adds no `Action`, no `ActionSpaceKind`, no
observation field, no state and no digest input. `Game::step` is the only
thing that changes a game.

### Native layer: `managym/src/flow/group/`

`mod.rs` holds the types and the one driver; one file per kind
(`combat.rs`, `library.rs`).

```rust
pub enum GroupKind { DeclareAttackers, DeclareBlockers, Scry, Surveil, OrderLibrary }

/// The grouped decision `viewer` faces now, read from the pending decision.
pub struct Group {
    pub kind: GroupKind,
    pub revision: u64,          // native semantic address it was read at
    pub fingerprint: String,
    pub items: Vec<GroupItem>,  // what will be asked, in asking order where the engine fixes one
    pub ordered: bool,          // the order of a plan's choices is part of the answer
}
pub struct GroupItem {
    pub item: ObjectRenderId,
    pub label: String,
    pub answers: Vec<Answer>,   // legal for this item from here
    pub required: bool,         // no default: must attack, must block, must be placed
}
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Answer {
    Attack, Block { attacker: ObjectRenderId },   // left out: stays home / does not block
    Top, Bottom,                                  // scry; no default
    Graveyard,                                    // surveil; left out: kept
    Next,                                         // order_library; no default
}
pub struct GroupPlan { pub kind: GroupKind, pub choices: Vec<Choice> }
pub struct Choice { pub item: ObjectRenderId, pub answer: Answer }

pub struct GroupPreview {
    pub group: Group,
    pub steps: Vec<GroupStep>,        // (revision, fingerprint, action_index, item, answer)
    pub refused: Option<GroupRefusal>,// first item the engine will not answer as planned, and why
    pub complete: bool,               // following `steps` finishes this group
    pub outcome: GroupOutcome,        // what the engine did: declared attackers and blocks
}
```

Entry points on `Game`, mirrored on `Env`:

- `group(viewer) -> Option<Group>`: `None` when the viewer is not the chooser,
  the decision is not a grouped kind, or it will ask at most one step.
- `preview_group(viewer, revision, fingerprint, &plan) -> Result<GroupPreview, AgentError>`:
  errors on a stale or non-actor address or a plan of another kind; a plan the
  engine will not take is `refused`, never an error. The live game is only read.
- `apply_group(viewer, revision, fingerprint, &plan) -> Result<GroupPreview, AgentError>`:
  runs `preview_group`; a `refused` or incomplete preview is an error and the
  game is untouched; otherwise steps the live game through `steps`, checking
  each step's `(revision, fingerprint)` before `step(action_index)`.

A kind is the small interchangeable part (a trait with one implementation per
kind, chosen by a `match` on the pending decision):

- `describe(&Game) -> Group`
- `next(&Game /* the clone */, &plan, progress) -> Answer step | Finish | Refuse(reason)`:
  the action in the clone's own action space that gives the planned answer for
  the item now asked. Legality is whatever that action space offers.
- `continues(&Game) -> bool`: the clone still asks this group's decision of
  this player.
- `outcome` / `unmet`: read what the engine did and name a plan it did not
  follow (menace removing a lone block).

`combat.rs` is today's `combat_preview.rs` moved under that interface, its
refusal sentences unchanged. `Game::combat_preview`, `Env::combat_preview`,
`Env.combat_preview_json` and `managym/combat.py` stay as wrappers over
`preview_group` that map `CombatPlan`/`CombatPreview`, each tagged for removal
in `0.3.0` with a `DeprecationWarning` on the Python call (it shipped in
0.2.6; main stays patch-clean). `managym/tests/rules/combat_preview.rs` stays
unedited: it is the proof that the generalization kept combat's behavior.

### Boundary

A group is the consecutive steps of one pending decision by one player. The
driver stops, and `complete` is true, when `continues` is false: another
decision, another chooser, or game over. Surveil ends when its graveyard picks
commit; the `OrderLibrary` that follows is its own group with its own
Confirm. `OrderTriggers` is not a kind here because modes and targets are
asked between picks.

The driver also refuses to step the clone past an event that gives the viewer
new hidden information (a draw, a shuffle, a library look or reveal) and
returns what it compiled so far with `complete: false`. No kind in this Task
should ever reach that branch; a test over every kind's fixture asserts the
clone's event tail holds none of those events between a group's first and
last step. If that test fails for a kind, the kind's boundary is wrong and it
is dropped from this Task rather than patched around.

`steps` carry the fingerprints of states the viewer has not reached. They are
for the authority. Academy's REST preview returns `group`, `refused`,
`complete` and `outcome` and never `steps`.

### Python surface

`Env.group_json(viewer)`, `Env.preview_group_json(viewer, revision, fingerprint, plan_json)`,
`Env.apply_group_json(...)` (which honours `reject_guarded_mutation` as
`step_structured` does), typed in `managym/group.py` as `combat.py` is today,
declared in `managym/__init__.pyi`, recorded with
`uv run python scripts/public_api.py record` (additions only).

### Academy authority (`managym/academy/`)

- **Descriptor.** `GameSession.live_group()` returns `Env.group_json` for the
  published prompt plus the Academy address `(match_id, revision, prompt_id,
  frame_hash)`, cached with the prompt. `_send_table_response` sets
  `response["group"]` (object or `null`) on every message that carries a
  frame, for `SUBMIT_LIVE_COMMAND` recipients.
- **Preview.** `POST /api/sessions/{id}/group-preview` with the address and a
  `GroupPlan`, same authorization and staleness checks as `/combat-preview`
  has now (`409 group_preview_stale`, `403 group_preview_forbidden`,
  `409 group_preview_no_group`). Read-only.
- **Apply.** WebSocket `{"type": "group_command", "command": {command_id,
  match_id, expected_revision, prompt_id, frame_hash, plan}}`, authorized like
  `command`. `GameSession.hero_group`:
  1. duplicate `command_id` → the stored receipt (`status: duplicate`);
  2. address checks → `stale_revision` / `stale_prompt` with recovery;
     `not_actor`; `frame_hash` mismatch → `stale_revision`;
  3. `preview_group`; `refused` → `rejected`, code `group_refused`, message
     the engine's reason, the refused item beside it, no recovery (the frame
     has not moved); not `complete` → `group_incomplete`;
  4. for each step: publish the prompt, require its native
     `(revision, fingerprint)` to equal the step's, `_apply_bound_command`
     with `source: "client"` and a command id derived from the group's
     (`<command_id>.<n>`); a mismatch is a `RuntimeError`, as the offer
     enumeration check in `/combat-preview` is today;
  5. one receipt under the group's `command_id` (`accepted_at` the first
     step's revision, `resulting_revision` the last), `committed(...)`, one
     `_advance()`, one `command_outcome` whose presentation runs from the
     first step.
  No `ReplayDecision`, trace or protocol schema changes: the rows are the
  rows N Commands write.
- `/combat-preview`, `CombatPreviewRequest` and `tests/academy/test_combat_preview.py`
  go; the test's cases move to `tests/academy/test_group_command.py`.

### Browser (`academy/ui/`), on top of MTG-441

- `gameStore` keeps `group` only while `group.frame_hash === frame.frame_hash`.
  No group: the board is exactly the raw offers.
- Combat keeps MTG-441's draft, drag, marquee and rail controls. What changes
  is underneath: the preview fetch calls `/group-preview`, Confirm sends one
  `group_command`, and the `running` send loop, `stepOffer`, `Stop sending`
  and every partial-progress sentence are deleted. A `group_refused` rejection
  shows the engine's reason on the draft and keeps it for editing.
- One `ArrangePanel.svelte` for Scry, Surveil and OrderLibrary, driven only by
  `group.items[].answers` and `group.ordered`: a row of destinations, cards
  moved between them and reordered by click or keyboard, Confirm disabled
  until the preview says `complete` without `refused`. It names nothing from
  a label and parses no `pending` text.
- The rail's collapsed disclosure of raw offers (MTG-441's
  `per-creature-choices`) takes its summary from the group kind and is shown
  for every group. Raw offers stay clickable and stay ordinary Commands; the
  next frame brings a descriptor for what remains.
- Drafts persist in `sessionStorage` under one key scoped to
  `(match_id, frame_hash, kind)`, replacing the combat-only key.
- While a `group_command` is pending the draft stays drawn and disabled, the
  way any pending Command leaves the board (MTG-441's `pending`).

### Choices made for review

- Based on main; pile split deferred to a follow-up after MTG-431 (above).
- First non-combat kinds are Scry, Surveil and OrderLibrary rather than the
  pick or amount families: they are one panel, they exercise an ordered plan
  and a `skip_trivial` tail, and Scry is printed across the selected matchup's
  set (`managym/cards/src/tla.rs`). Passed over: shipping native kinds with no
  browser consumer.
- Left-out items take a default only where the game already has one (a
  creature not declared stays home or does not block; a surveilled card not
  binned is kept). Scry and OrderLibrary have none: an unplaced card makes the
  plan incomplete and Confirm stays disabled. Nothing is selected silently.
- Apply is all-or-nothing. A person who wants one step uses the raw offer.
- The descriptor is live-only and beside the frame (above).
- `apply_group` exists natively even though Academy follows the compiled
  steps itself, because each recorded step needs its own frame, retained root
  and presentation drain between native steps. Both follow the one `steps`
  list; pass 1 pins them equal.

## Acceptance check at gate

```
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -E 'test(group) | test(combat_preview)' \
  && uv run python scripts/public_api.py check \
  && uv run --extra dev --extra play python -m pytest tests/academy/test_group_command.py -q \
  && (cd academy/ui && npm run check && npm test \
      && npx playwright test e2e/grouped-decisions.spec.ts e2e/combat-presentation.spec.ts e2e/drag-play.spec.ts e2e/keyboard-play.spec.ts)
```

All pass. The one scenario that crosses every consumer is a three-creature
attack:

1. **Native** (`managym/tests/rules/group.rs`): from one position,
   `apply_group(plan)`, stepping `preview_group(plan).steps` by hand, and the
   unedited `combat_preview` test's `follow` all reach the same
   `state_digest`. A refused plan and a stale address leave the digest where
   it was. The same three-way equality holds for a Scry 3 with one card
   bottomed and for Surveil 2 → OrderLibrary.
2. **Authority** (`tests/academy/test_group_command.py`): two sessions from one
   seed and config. One sends three `command`s, the other one `group_command`.
   Their `canonical_decisions` agree row for row on viewer, source, offer,
   projection and revision, their native state digests are equal, and the
   second session received exactly one `command_outcome`. A refused plan
   returns `group_refused` with the engine's sentence and writes no row.
   Resending the same `command_id` returns `duplicate` and writes no row.
3. **Browser** (`e2e/grouped-decisions.spec.ts`): drafting three attackers and
   pressing Confirm sends exactly one socket message of type `group_command`;
   a Scry 2 opens the arrange panel, Confirm is disabled until both cards are
   placed, and one message is sent; opening the disclosure lists the raw
   per-card offers and clicking one sends an ordinary `command`.

Demo: `./scripts/play`, attack with several creatures and press Confirm (the
network panel shows one frame out), then cast a Scry 2 spell and arrange both
cards before confirming once.

## Absent and error states

- No group (single-step decision, not a grouped kind, not the chooser, Replay,
  Study, Watch, recording): `group` is `null` or absent; raw offers only.
- Stale preview (`409`): the browser drops the answer and asks again at the
  current frame; a draft whose `frame_hash` no longer matches is discarded.
- `group_refused`: nothing applied, frame unchanged, draft kept, engine's
  reason shown on the refused item.
- `group_incomplete`: same, naming what is unplaced. The browser should never
  send one; the code exists for other clients.
- Stale `group_command`: rejected with recovery, nothing applied, draft
  discarded with the frame it belonged to.
- Connection lost during apply: the authority finishes or never started. On
  reconnect the recovery frame is the truth and a resend of the same
  `command_id` is a `duplicate`.
- A compiled step that does not match the live game: `RuntimeError`. It means
  the clone and the game diverged, which is an engine bug; steps already
  applied stay recorded as the ordinary decisions they are.
- Storage unavailable: no saved draft, no error.

## Operational boundary

- Preview: one `Game::clone` and at most `items` steps, as `/combat-preview`
  costs now. The browser asks once per draft edit, debounced as combat is.
- Apply: one more clone plus the same steps, under `record.lock` for the whole
  call, as a single Command is. A plan with more choices than the group has
  items is refused before any clone.
- No engine hot path changes; `managym/perf/baseline.json` digests must not
  move. A moved digest means the layer touched state and is a defect.

## Exclusions

- New `Action` or `ActionSpaceKind` variants, observation or policy-input
  changes, any change to `compound_*` / `step_structured`, `ExperienceFrame`,
  `ReplayDecision` or the protocol schemas.
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

Survives unchanged: every `Action`, digest and recorded row; MTG-441's drag
and draft UX; combat's refusal sentences; `compound_*`;
`Env.combat_preview_json` until `0.3.0`.

## The audit, reclassified

With a native compiler the earlier A1/A2/B split disappears: "needs typed
fields" is the descriptor and "needs clone-and-step" is the driver. What
separates decisions now is the shape of the plan and where the group ends.

| Decision (stage) | Plan shape | Group ends | Status |
|---|---|---|---|
| DeclareAttackers / DeclareBlockers | answer per creature | declaration event | this Task (ported) |
| Scry (`scry`), 2+ cards | destination per card, ordered | scry finishes | this Task |
| Surveil (`surveil`) | subset to graveyard | picks commit; `OrderLibrary` follows as its own group | this Task |
| OrderLibrary (`order_library_top` / `_bottom`) | order | last card placed (the final one is never asked) | this Task |
| PartitionPiles (`partition_piles`, MTG-431) | pile per card | split finishes; `ChoosePile` is the opponent's | follow-up after MTG-431 |
| SelectCards with max > 1, SearchDistinctNames, Discard of 2+, OpponentSplitsRevealed, KeepPermanents (count or total power) | subset within engine limits | selection finishes | follow-up: "pick" kind and panel |
| DistributeCounters, DivideAmongTargets (`target_shares`), AssignCombatDamage | amount per recipient | amounts exhausted | follow-up: "amounts" kind and panel |
| OrderTriggers | order | only while no trigger asks a mode or target between picks | follow-up, behind the boundary test |
| Retarget, ManaUpfront, ManaSpend, OrderEntryChoices, OrderReplacement | — | — | stays sequential: later options are legality- or planner-filtered, usually one or two steps |
| LookAndSelect, ManifestDread, ChooseToExile, DiscardFromRevealedHand, OrderMeldedCards, ExileLinked, LegendRule, ChoosePile | — | — | one step |

A follow-up kind is a `flow/group/` file implementing the kind interface plus
a panel for its shape; it needs no server, protocol or recording change.

## Passes

1. **Native layer with combat parity and the library kinds.** The pass that
   could prove the plan wrong. `flow/group/` (types, driver, `combat.rs`,
   `library.rs`), `Game`/`Env` entry points, the `combat_preview` wrappers,
   bindings, `managym/group.py`, `__init__.pyi`, public API record, extension
   rebuild; `managym/tests/rules/group.rs`.
   Shows: the unedited `combat_preview` tests pass through the wrapper, so one
   driver serves combat; Scry, Surveil and OrderLibrary fit the same kind
   interface without a special case in the driver; `apply_group` equals
   following `steps`; the boundary test finds no hidden-information event
   inside any group. If a library kind needs the driver to know about it, the
   interface is wrong and is reshaped here, before any consumer exists.
2. **Academy authority.** `live_group`, the `group` field on live messages,
   `/group-preview`, `group_command` and `hero_group`,
   `shared_table_protocol.py`; `tests/academy/test_group_command.py`; removal
   of `/combat-preview`. Shows: acceptance step 2, the recording equality.
3. **Browser and docs.** Needs MTG-441 on main. `gameStore` group, combat on
   one-shot Confirm, `ArrangePanel.svelte`, rail disclosure, draft storage,
   `e2e/grouped-decisions.spec.ts`, updates to MTG-441's combat specs for the
   removed stop control. `docs/academy.md` gains "Grouped decisions" (what a
   person sees, the contract, the audit table above) and its combat
   paragraphs lose the send loop; `managym/README.md` documents `group`,
   `preview_group`, `apply_group` and the boundary; `AGENTS.md` "Decision
   ownership" gains the sentence that grouped input compiles to sequential
   Actions and never widens an action space. Shows: acceptance step 3 and the
   demo.

## Done when

- Confirming a drafted attack, block, scry, surveil or library order sends one
  message, and the game is either wholly past that decision or exactly where
  it was with the engine's reason shown.
- A game played with grouped Confirms and the same game played one Command at
  a time record the same decisions and reach the same state digest.
- Replay and Study of such a game show every per-step choice as they do now.
- The browser holds no send loop, no step verification and no rule about
  which answers are legal; it reads `group` and the preview.
- `uv run python scripts/public_api.py check` passes with the three new `Env`
  methods and `combat_preview_json` still present.
- `docs/academy.md` carries the reclassified audit and names the follow-up
  kinds.
