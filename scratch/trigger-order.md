# Controller-selected simultaneous trigger order

**MTG-195 · Design accepted · 2026-10-26.** Jack Heart accepted the reviewed
design with “seems reasonable” in the MTG-195 primary conversation. This accepts
the sequential native representation, second-class scheduler fixture proof limit,
and maintained learnability checks described below. Jack Heart explicitly authorized
implementation with “implement for sure” in this conversation. On 2026-10-27 Jack Heart selected concept-review option 1a and authorized
publication and landing after that refinement. Training experiments and paid
compute remain excluded.

## Problem and intended experience

Human players and training agents cannot choose the order of waiting triggered
abilities they control. `pop_next_pending_trigger` sorts by active-player rank
and enqueue counter. This loses a strategic choice: at 2 life, putting gain-2
below damage-2 loses the game; putting damage-2 below gain-2 survives at 2.
The last ability placed resolves first, with state-based actions between resolutions.

The outcome is one complete native controller choice consumed by human Commands,
ordinary bot actions, search and replay. Academy explains placement and resolution
order without selecting an order or reconstructing legality. Correct two-part APNAP
placement, target timing, source identity and committed costs remain authoritative.

This serves Rules' faithful shared-world and exact-authority objectives in
`wave/rules/GOAL.md`. The Task authorizes planning a missing-mechanic correction,
not deck expansion. Wave memory's historical world numbers and deferred provider
work do not select new work. No quantitative performance claim is commissioned.

## Demo

Implemented headless demo:

```sh
uv run --extra dev --extra play pytest tests/academy/test_trigger_order.py::test_survival_orders_cross_consumers -q
```

Create an explicit native scenario root using existing registered cards: controller
at 1 life, Bastion of Remembrance and Callous Inspector on the battlefield, two
ready Swamps, Deadly Dispute in hand, opponent above 1 life. Setup uses existing
scenario helpers before capturing the root. The prefix then casts Dispute and
sacrifices Inspector through normal Commands. Its death creates Bastion's
drain-1/gain-1 and Inspector's self-damage-1/Clue triggers. The cross-consumer
witness injects no pending triggers and changes no card definition.

Academy displays “Choose the next ability to put on the stack,” both labeled
abilities, and explains that later placements resolve first. Choosing Inspector
first places Bastion above it: gain life, then take damage, remain alive at 1.
Choosing Bastion first places Inspector above it: lose at 0 before Bastion resolves.
Dispute's sacrifice and mana payment remain committed. The opponent must be above
1 life to avoid a confounding terminal outcome.

Run both orders through GameSession's real Command boundary and ordinary Env actions;
compare final state and ordered consequences. Replay recorded Commands from the
same explicit root, and fork/return at order and target suspensions. The browser
companion uses the same server path for keyboard selection and disconnect recovery.
A test-only ASGI fixture installs the root through existing Python APIs before play;
no production injection endpoint or mocked acceptance is added. The retained `academy/fixtures/trigger-order-w14.json` records the root
recipe, world/content identities and initial digest: seed-only replay
cannot reproduce an injected root. Ordinary authored-match conformance is separate.

Keep the exact original **2-life / gain-2 / damage-2** synthetic native witness as
an independently specified regression. The existing-card demo demonstrates the
same consequence through actual consumers without changing authored decks.

## Investigation and resolved risks

- **Current main inspected:** checkout `558e0186b585228dd2130cf14d9cb40de217e66f`;
  fetched main `2f5b7cbcd` on 2026-10-26 adds only paper changes. Trigger source is
  byte-identical; PR #354's audit is present. Task status identifies this running
  `task-design` graph and no current PR. No graph was restarted. Recheck overlapping
  open PRs before implementation as required by the Task.
- **Pinned rule verified:** the [September 25, 2026 Wizards rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt)
  download hashes to
  `8d860e451f20f38865b725b42d82feb714c725373dd8f3b32b8652b3eeb070ca`, matching
  `docs/rules/audit-2026-10/source.json`. CR 603.3b first traverses APNAP for
  triggers not conditioned on another ability triggering, then traverses APNAP
  again for the remaining class; each controller chooses within its group. SBA
  checks and newly waiting abilities follow before priority. CR 603.3d locates
  target selection at placement. CR 405.2, 117.5 and 704.5a explain the opposite
  outcomes. Cite these rules beside the owning code.
- **Missing choice reproduced:** the distinct-outcomes witness passes; explicitly
  running the ignored choice witness fails at “must suspend for the controller's
  order choice.” Its later unchanged stack-length assertion is correctly 2:
  the fixture already contains two spells, not an empty stack.
- **Two-part scheduling is absent; causality is narrower than event naming:**
  `TriggerCondition` has no condition for another ability triggering.
  `GameEvent::AbilityTriggered` is emitted at stack placement, not collection;
  it cannot serve as trigger-time causation. The proposed scheduler has both
  classes, tested with typed native fixtures. Current collection/delayed-trigger
  producers all select the ordinary class. Future trigger-on-trigger support
  needs actual collection-time matching, typed content admission and its own
  evidence. Injecting a second-class scheduler fixture proves scheduling, not
  that missing matching feature. This limited scope was accepted at design review.
- **Suspension crosses authorities today:** `pending_triggers` collects arrivals;
  `pending_trigger_choice` holds one trigger's targets; `flush_triggers` consumes
  both. `stabilize_before_priority` checks SBAs on re-entry. Resume must retain
  the placement round and not interleave another batch or an SBA between placements.
- **The consumer route exists:** `ActionSpace` projects through structured offers
  and compound lowering to `DecisionFrame`/`Command`, with revision/offer binding.
  Academy maps these to Experience offers and generic direct-choice buttons.
  Experience has an order-shaped DTO, but no native permutation authority or
  ordinary policy permutation decoder follows merely from that DTO.
- **Shared observations need work:** `continuation_frames` includes stack,
  announcement and suspended-resolution state, not the pending trigger target
  slot or waiting triggers. Source-card focus alone collapses multiple abilities
  or occurrences with different contexts. Human labels alone would leave policies
  blind. Extend the native projection and action-to-trigger join for both consumers.
- **Old acceptance tooling was deliberately removed:** PR #372,
  `b3681297f16e27a9e42b5e49f4fe928292fe0bb2`, deleted `manabot.verify.learnability`,
  `learning_control`, their workflows and positive-control tests at Jack Heart's
  request. Current `AGENTS.md` prescribes maintained arena/monitor,
  choice-separability and ordinary PPO/checkpoint checks. Use those, extended for
  triggers, as the accepted interpretation of the Task's “full learnability”
  wording. This does not establish playing strength; old w12/w13 receipts do not
  validate the changed world.

## Chosen mechanism

### One placement continuation

Keep `GameState.pending_triggers` as the arrival queue. Replace
`pending_trigger_choice` with `trigger_placement: Option<TriggerPlacement>` in
`flow/trigger.rs`. This is pre-priority state, not an `EffectFrame` pretending an
ability is resolving. It uses the existing native action/Command decision pipeline.

`TriggerPlacement` owns the detached round's remaining triggers, captured active
player, pass/controller cursor and `placing: Option<PendingTrigger>`. The latter
owns the current requirement and chosen targets already carried by PendingTrigger;
there is no second target-progress record. Add typed
`TriggerOrderingClass::{Ordinary, AbilityTriggered}` at enqueue. Controller and
class are captured facts, not recomputed from a source that may have left play.

Promote the enqueue identity to typed `TriggerId`: occurrence identity, never
preferred order. Mint with checked increment and explicit exhaustion failure,
replacing saturating increment. No second counter. Neither CardId, ability index,
vector position nor a live ObjectRef identifies an occurrence. Source ObjectRef/LKI
still describes the source; source departure does not delete its trigger.

Progression:

1. Finish committed-event collection and the required SBA fixed point. With no
   active placement, detach the waiting set into a round. Traverse ordinary AP,
   ordinary NAP, then ability-triggered AP, ability-triggered NAP. Group the set
   waiting since priority, not just triggers caused by a single event.
2. In the current group, expose **one next-to-stack action for each remaining
   trigger** when there is a choice. Selection moves that occurrence into `placing`.
   Empty groups advance; singleton ordering is forced and may advance automatically,
   but its required target choices remain explicit. Exercise both `skip_trivial`
   settings. Do not coalesce apparently identical occurrences or omit their contexts.
3. Finish this trigger's existing target requirements, clause-local distinctness
   and optional-target decline, then use the existing stack placement authority.
   If required legal targets cannot be chosen, consume/remove this trigger under
   CR 603.3d and continue. Invalid submissions preserve progress. No earlier
   placements rewind; ordering offers no pass, cancel, undo or default branch.
4. Continue the same group and then the next. Resume active placement **before**
   restarting SBA processing. No priority or effect resolution occurs mid-round.
   Facts emitted during placement stay queued for the later fixed-point round;
   they cannot enter a group already traversed. After both passes finish,
   stabilization performs its next SBA/collection iteration.

For n occurrences in a controller/pass group, all n! orders are expressible with
n-1 deliberate selections, O(n) stored occurrences and at most n offers per decision.
The final position is forced. Generating all menus across an order can be O(n²);
there is no factorial materialization or general large-board latency guarantee.

### Actions, shared input and versioning

Add `Action::ChooseTrigger { player, trigger_id }`, `ActionType::ChooseTrigger`,
`ActionSpaceKind::OrderTriggers` and matching structured prompt/verb variants.
Each offer privately binds to an exact occurrence through existing offer-set binding.
`Command` retains its revision/offer envelope and empty answers. Wrong actor, group,
occurrence or revision rejects atomically. Canonical display order may use TriggerId;
that order never silently becomes the strategic choice.

Extend the native continuation projection with the current pass/controller and
pending occurrences: definition/ability, exact visible source or LKI definition,
causal context, current targets and which occurrence is being placed. Action metadata
joins to a **decision-local trigger frame**, not an unbounded counter cast to float.
A departed source has a definition/ability description without a fabricated live
object. Candidate labels include ability meaning and relevant context. Only the
actor receives pending candidate details; other viewers retain public stack facts
and committed consequences without another player's private candidates.

Derive `decision_context`, decision capture, text and Academy prompt explanations
from those frames. For ordinary policy input, add an explicit action-to-continuation
row index and gather that row's contextual embedding into the action embedding.
Existing pooled continuation context alone cannot associate two same-source offers
with different frames. Reuse `ContinuationEncoder` row encoding for pooling and
this gather, with zero for absence and checked references. Raw occurrence counters
are not strategic features. Native and Python reference encoders agree exactly.

Implemented contracts: fresh policy input v3, semantic decision schema 7, capture
version 3 and next unallocated rules world (w14 at inspection). Advance canonical
state-hash version for changed trigger state. Action ABI includes the added family
and trigger-row join. Update corresponding runtime/model architecture manifests
and semantic learning schema if its serialized vocabulary changes. Preserve
`ObservationConfig.from_checkpoint` semantics: missing version still means v1,
and existing serialized v1/v2 configurations retain their meaning. Reject old-world
checkpoints for current-world play; do not reinterpret or pad their heads to imply
compatibility. Versioned evidence does not require a second live ordering engine.

Include all round fields, occurrence identity/counter and partial targets in clone,
canonical hashing and undo. Replace the old target-slot journal entry. Search
fork/apply/rollback and Study return must restore offers, observations, RNG, event
cursor, stack and digest at every new suspension. Replay stores ordinary Commands
and existing transition receipts; no parallel order tape or browser-owned draft.

### Absent/error states and operational boundary

- No waiting triggers means no ordering prompt. Singleton ordering introduces no
  strategic choice. Impossible mandatory targets consume only that selected trigger;
  invalid Commands consume nothing.
- Stale Commands fail before mutation. Existing receipt handling returns duplicate
  results without applying another occurrence. Reconnect exposes the native cursor,
  offer set and partial targets. Priority-stop automation never answers OrderTriggers.
- Terminal loss is evaluated at normal resolution/SBA boundaries. Pending gain does
  not rescue a player already at 0. Costs, targets and revelations remain committed.
- Native/semantic choices remain uncapped. Bounded tensor admission fails explicitly
  if all actions or relations do not fit; no truncation, first-N menu, random fallback
  or automatic order. Test a 65-occurrence native menu without factorial traversal,
  default encoder rejection, and successful larger capacity where configurable.
  Fixed relation overflow remains an explicit model admission limit, not an engine
  legality limit. General ragged-policy redesign is excluded.
- No network/service dependency is added to placement. Tests use bounded fixtures;
  no load generation or paid benchmarks. Existing consumer timeouts remain intact.

## Alternatives and failure analysis

| Mechanism | Decision |
| --- | --- |
| Sequential next-to-stack actions | Accepted: complete orders, existing action decoder, ordinary target continuation. Each selection is a visible committed revision. |
| One atomic permutation Command | Possible, but needs native ordered-occurrence grammar, policy decoder and editing/recovery semantics. An existing browser Order DTO does not supply them. Disproportionate here. |
| One action per full permutation | Reject: factorial menus invite caps and missing legal choices. |
| Browser reorder or fixed “good” order | Reject: second authority or silently removed agency. |

Success is a player understanding the reverse resolution order, choosing either
outcome by keyboard or pointer, and recovering the exact unfinished choice; a policy
can learn either preference from that same state. Multiple abilities on one source
make the shared input improvement particularly useful.

Failure is a UI that lists every card while collapsing ability occurrences, a target
resume that enters another APNAP batch too early, or identical tensor actions for
different trigger contexts. Opposite outcome witnesses, same-source/context
separability and phase-boundary rollback tests address those failures. Browser
snapshots and replay self-consistency alone cannot establish rules correctness.

## Affected surfaces, removals and exclusions

Owners: `flow/trigger.rs`, `triggers.rs`, `stabilization.rs`, `action.rs`, `game.rs`,
`setup.rs`, `undo.rs`, `state/hash.rs`; native action/observation/structured offers,
continuation, decision capture/context; PyO3 and `manabot/managym` enums/types and
Experience projection; native/Python encoders, `model/continuation.py`,
`actor_critic.py`, configuration/checkpoint admission; semantic text, search prompt
mapping, replay/Study; Academy prompt instructions/labels and generated wire types.
Reuse generic action controls with the new prompt/context.

**Delete — do not maintain:**

- Remove `pop_next_pending_trigger` and its APNAP/enqueue minimum selection.
  Replace callers directly; no legacy ordering flag or fixed-order fallback.
- Remove `GameState.pending_trigger_choice`, setup/clone/hash field,
  `UndoEntry::PendingTriggerChoice` and its journal helper when target progress
  moves into `TriggerPlacement.placing`. Cut target helpers/action dispatch over
  in that same change. Preserve legality, clauses, source identity and LKI.
- Remove the ignore marker from the audit choice regression. Change the
  distinct-outcomes test to submit choices instead of changing enqueue order;
  preserve independent expected outcomes and the old two-controller target-owner
  APNAP witness.
- Replace live fixture pointers with fresh world-bound evidence, retaining earlier
  corpora, authored decks, checkpoints, architecture receipts and experiments.
- Never restore the intentionally removed learnability verifier, learning-control
  module, candidate-bundle loader or their exclusive tests/workflows.

Excluded: replacement/prevention ordering (separate queued work), general trigger
condition expansion, modal-trigger completeness beyond existing supported semantics,
new cards/decks, priority automation redesign, human-only undo, multiplayer, LLM
harness changes and new training experiments. Broader source/target/cost changes
need separate evidence rather than being hidden in this mechanism.

Forbidden: legacy/new live engines, UI order authority, duplicated target state,
factorial actions, strategic deduplication, placement events treated as trigger
causes, reconstructed recovery menus, hidden-state labels, shape-only checkpoint
admission and a locally passing consumer that sees only part of the legal menu.

## Internal implementation slices

One coherent architectural change; intermediate cuts are not separately shippable.
No compatibility/default path may make a partial migration appear complete.

1. **This slice — replace ordering authority.** Introduce placement and occurrence
   identity; delete fixed sorting/old target slot while cutting action dispatch,
   projection, hash/undo and clone over together. Implement two passes and target
   continuation. Focused check:
   `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml --lib trigger_order`.
   New tests under that prefix cover n=0/1/2/3 (all six orders for three), both active
   seats, pass-before-controller precedence, source departure, optional/impossible/
   multiple target clauses, arrivals during placement, no intervening priority/SBA,
   rejection and nested rollback. Enable the old audit test. Trigger-on-trigger
   condition matching remains explicitly unproved.
2. **Complete shared inputs and consumer cutover.** Add action/frame joins, versions
   and Python mirrors; derive Academy/text context; update search and ordinary policy.
   Add same-source/different-ability and same-ability/different-context collision
   tests, encoder parity, permutation invariance and capacity failures. Implement
   the existing-card demo and real WebSocket/browser recovery. Finish generation,
   typecheck and build before starting browser servers.
3. **Bind the world and evidence.** Select the next world after checking overlapping
   changes, keep decks/frozen evidence unchanged, record fresh conformance and
   affected demo/navigation roots, and update live fixture/CI selectors. Run the
   gate matrix once using the accepted learnability interpretation. This
   plan selects no publication or landing procedure.

## Done when / headless gate

Both legal outcomes are selectable through native and human/agent paths. Every
controller has every within-group order, correct two-part APNAP placement and exact
target continuation. Human, bot, capture, replay and forks agree without lost costs
or hidden-state exposure. Fresh world-bound checks establish these facts; compilation
or an added enum does not satisfy the outcome.

Validation commands, from the repository root:

```sh
CARGO_BUILD_JOBS=2 cargo fmt --manifest-path managym/Cargo.toml --check
CARGO_BUILD_JOBS=2 cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings
CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml
```

Expected: full debug suite passes, including the enabled audit, independent outcomes,
two-part/APNAP fixtures, fork/undo checks and fresh current-world conformance. Record
the new corpus under a versioned root with the existing `semantic_conformance record
--root` operation during preparation; gate checks, not overwrites, it. Historical
roots remain byte-identical.

Build the extension from the root with
`CARGO_BUILD_JOBS=2 uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
and install the cp312 extension by atomic rename per `AGENTS.md`. Then:

```sh
uv run --extra dev --extra play pytest tests/academy/test_trigger_order.py tests/semantic/test_decision_contract.py tests/semantic/test_text_observation.py tests/academy/test_decision_surface.py tests/academy/test_optional_payment_recovery.py tests/academy/test_study_branch.py tests/academy/test_choice_navigation.py tests/academy/test_learn_setup.py -q
uv run --extra dev pytest tests/model/ tests/verify/test_continuation_input.py tests/training/test_active_training.py -q
uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q
npm --prefix academy/ui run check
npm --prefix academy/ui run test:e2e:release
```

Expected: crossing demo and sibling isolation pass; stale/foreign choices mutate
nothing; duplicate receipts add no event; reconnect at ordering and target cursors
returns the same bound frame; nonactors see no private candidates; text/tensors
distinguish relevant occurrences. Extend the maintained choice-separability matrix
with ordering, retaining its two fixed opposite-preference fits, 180 updates,
>=0.95 selected probability after save/reload and permutation assertions. The
Elemental check exercises ordinary PPO and reload of its final checkpoint. Arena
and monitor admit new-world artifacts and reject historical ones. These prove
capacity/integration, not playing strength or the deleted full diagnostic.

Add `trigger-order.spec.ts` to release Playwright `testMatch` so the release command
actually selects it. Use an isolated built SPA and real ASGI test fixture, one worker
and unused configured API/frontend ports. Assert both orders, keyboard access,
reverse resolution explanation and interrupted recovery. Retain existing release,
combat and navigation cases; settle narration before terminal assertions. Missing
local browser/toolchain checks defer explicitly to capable CI without blocking
source work or substituting mock acceptance. Collect new uniquely named Python
modules with their existing CI invocation to catch import collisions.

**Implementation — 2026-10-26.** Implemented locally under Jack Heart's
“implement for sure” direction, followed by “continue.” The fixed enqueue-order
selection and old target slot are removed. One native placement continuation now
owns controller choices, APNAP passes, target progress, identity and rollback;
Commands, Academy, text capture and ordinary policies consume its shared facts.
The final selected-trigger context also supplies Academy's target source.

The model joins each action to every relation of its trigger frame; source,
ability and causal-context distinctions survive object/offer permutations.
Occurrence counters and pending frame ordinals are excluded from numeric policy
features, and missing joins reject input. Native menus remain complete; oversized
tensor inputs fail admission explicitly. The existing-card recipe and initial
identity are retained in `academy/fixtures/trigger-order-w14.json`.

**Validation:** focused native ordering/target/rollback/capacity checks pass;
Python model, two-preference fitting, capture/replay/reconnect, current-world PPO
and checkpoint reload, Academy fixtures and package boundaries pass. The release
browser suite passes all 21 cases; arena/monitor passes 33 checks after adding w14
to the registration vocabulary. Clippy, formatting/lint, frontend check/build and
CI selection collection (669 tests) pass. The full debug Rust suite also passes,
including all 316 rule tests, exhaustive search rollback and the fresh w14
conformance corpus. Final join/inference checks pass (33 tests); the final native
ordering follow-up passes (6 tests). Local command output is retained in
`/tmp/mtg195-*`.

The broader runs caught stale test assumptions about automatic ordering, a missing
Academy instruction display, and the omitted arena world registration; these are
fixed. Historical corpora and fixtures remain unchanged. The two-pass scheduler is
fixture-proved; trigger-on-trigger matching remains outside this implementation.
No new training experiment, paid compute, publication or merge occurred. Task
status remains open with no live Graph; disposition is **available** for local
implementation review in the
primary conversation. Periodic Task automation and the CI watcher are disabled.

## Concept review — 2026-10-27

Jack Heart requested concept-review after local implementation. This review does
not reopen the accepted sequential native choice, authorize publication, or
restart implementation. The core concepts remain a waiting occurrence, a placement
continuation and the public stack. No user-owned order draft is required.

**Consequential mismatch:** the human chooses an occurrence, but the current
button describes its entire source card. A fresh native demo read shows Bastion's
death-trigger offer also contains its entry-trigger text; Inspector's death-trigger
offer includes Menace and the Clue activation reminder. An ability index identifies
a slot but leaves the human to identify what this occurrence actually does. The
policy's source/ability/context join already makes the distinction. The generic
Academy control renders `offer.label` verbatim. The existing reverse-resolution
instruction is accurate and should stay.

**Refinement accepted as option 1a, 2026-10-27:** make each offer read “Put [this ability] on
the stack next,” with its exact source and relevant viewer-safe context. Show the
selected ability's meaning; retain full source text in inspection. Stack entries
remain top-first and authoritative. Selection commits placement order immediately;
finish any required targets before the next placement. In the two-trigger demo,
the other position is forced, but no ability has resolved at that point. Do not
relabel the operation “resolve first” or predict effects as though priority and
responses cannot intervene.

Derive ability descriptions once at the native projection owner, from the exact
registered ability or inline/delayed effects, rather than hand-writing Academy
paraphrases or splitting card text by newline. Keep occurrence identity and the
action/frame join; two occurrences of one source must never collapse. This is
presentation derived from the existing state, not another persisted ability or
order record. Reuse the same description in offers, stack inspection and text
capture where applicable; policy tensors continue to consume typed facts.
Experience remains a consumer of bound offers. Presentation changes may alter
frame/capture hashes; update current fixtures deliberately without rewriting frozen
evidence. Shared rules, APNAP, privacy, target timing and checkpoint admission
remain the accepted requirements.

**Draft Academy usage paragraph — proposed copy for `academy/README.md`:**
> When several of your abilities are waiting, choose one ability to put on the
> stack next. Inspect its source for the full card text. Choosing commits its
> place in the order; if it needs targets, finish those choices before continuing.
> Later placements sit above earlier ones and resolve first if the stack proceeds
> without responses. The last remaining position is forced. Reload restores the
> actual pending order or target decision, or the current stack if placement has
> finished. A lost response does not undo a submitted choice.

**Draft Task demo instruction — proposed, no global skill change:** demonstrate
both existing-card orders using explicit Commands; distinguish placement from
resolution; interrupt one submission before acknowledgment and recover the native
frame. Do not select a preferred order for the person, simulate refunds, or present
a prototype state as a live game. Existing generic demo/Command guidance needs no
rewrite.

**Failure and counterexamples:** an accepted choice whose response was lost
recovers committed state; a duplicate receipt does not place again; a stale offer
refreshes to the actual frame rather than inventing a retry. A vanished source
keeps its ability identity without a fabricated live object. Same-source occurrences
with different causal subjects, delayed inline effects, and optional/multiple target
clauses invalidate a card-name-only or whole-text summary. Do not call the round
finished while a forced final trigger still needs targets. A requested editable
whole-order preview would change the accepted native protocol and recovery model;
it is not a browser-only affordance to add during this refinement.

**Accepted product choice:** Jack Heart selected 1a: refine shared ability
descriptions and explicit place-next controls before delivery. He then authorized
landing with “after that you can land.” Option 1b is superseded. An editable whole-order interaction is a distinct design
change, not approved by either answer. Next proof for 1a: inspect a pending ability
without committing; identify which ability fires and where it is placed; then
recover a lost order/target response through the same Command path. Include
same-source/context and inline-effect examples. The review illustration is
`scratch/concept-review.html`, not an executable rules proof.

Task/Graph status refresh currently fails in `lf` with a serialization error,
`missing field interactive`. Session and process inventories read successfully;
no execution was launched. Last verified implementation results remain dated
above; this review does not claim a fresh Task execution disposition.
