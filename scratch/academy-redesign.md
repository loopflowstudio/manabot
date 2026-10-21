# Academy as the agent's decision surface

Live demo revision, Jack Heart, 2026-10-21: board first, play selection primary,
and logs/history/details available but collapsed. This supersedes the original
decision-first narrow layout below. Implemented with a single initial start
action, visible pending state and batched persistence during checked demo setup.
The restarted live demo responded in 0.77 seconds; 390px/1500px and the mana-cost
alternative browser checks passed. Human demonstration remains available here.

Accepted implementation direction, 2026-10-21: Jack Heart said “just build it and
then i will demo”. This authorizes the complete proposal, including explicit offer
commitment after inspection, acting-view Watch by default and decision-first narrow
layout. Human demonstration and usability acceptance remain separate. Publication,
deployment, training and Task completion are excluded.

The runtime now has native read-only decision explanations, exact typed offer
subjects, public pools, private inspection and ordered stack entries. One shared
surface renders live, Watch, replay and Study decisions. Board navigation never
submits; the existing Command authority commits each offer. Study uses the same
one-action semantic seam and returns to its retained source. Precision and Learn
have checked ordinary-Command prefixes. Frozen frames/erd1 evidence remain unchanged.

The native observation audit found pools, inspection candidates and ordered stack
entries already available to agents. Explanations summarize actor choice history
and current continuations; they do not add tensor features or legal semantics.
Academy uses the existing native `skip_trivial=False` preset to expose sole mana
and payment decisions. Configured priority-stop automation stays distinct. Training
presets, w9 identities and frozen artifacts keep their original settings. Current
Learn/Precision prefixes bind the explicit preset; historical navigation reproduction
uses its original native preset only in its compatibility test.

## Usage documentation

Start **Play** by choosing your deck, opponent and opponent deck. See the actual
world and setup compatibility before starting. A disabled checkpoint explains its
mismatch; a failed replacement leaves the existing game available. **Watch** uses
the same decision surface with bot controllers. **Games** opens saved outcomes;
**Study** explores a recorded decision without changing the live game.

At the table, read **Who decides**, **What is happening**, and **Your choices**.
Turn ownership and decision ownership are separate: an opponent's turn may require
your target or response. The board shows public zones, your private cards, known
opponent cards and unknown counts. Inspected cards appear in a private inspection
area only while the authoritative observation permits them. Card text explains the
card; legal offers decide what can be done now.

Click a card to inspect it and locate its offers. That click does not play it.
Search or group the legal offers; the count always says how many of the complete
set are displayed. Clear filters to see every offer. Identical copies remain
individually selectable and highlighting identifies the exact object. Activate an
explicit offer button to commit one decision. There is no hidden suggested payment,
implicit color selection, or automatic selection of a sole strategic choice.

When casting, the table follows the actual sequence: modes, targets, additional
costs, mana decisions, then the stack and legal responses. A context strip shows
what is announced, what is selected but unpaid, and what is already committed.
It is a description of shared rules state, not an editable local transaction.
Back clears navigation only; it cannot undo a submitted Command, a sacrificed
Treasure, spent life, or revealed cards. A resolving sacrifice is labeled as an
effect, distinct from a casting cost.

After a choice, **What happened** shows the authority's consequences and the next
pending decision. Animations may finish or be skipped without controlling legality.
A lost connection disables commitment. Reconnect restores the actual pending
frame, not a locally rebuilt payment draft. An expired table says the live game is
unavailable and offers its saved history and an explicit new-game action. A saved
screen is not evidence that an expired engine can resume.

Replay uses **Decision before / Chosen action / Consequences after**. Automatic
passes are labeled as automation rather than human choices. Study's **Try another
line** operates on an isolated retained root; **Return to recorded decision** and
**Return to live** are explicit destinations. Missing roots or policy outputs show
unavailable states. Watch's privileged **Show both hands** is separate analysis;
**Acting bot's view** is the default proposed inspection lens for this redesign.

## Concrete screen and interaction proposal

Desktop table, wide layout (plain-text wireframe; no rendered prototype claimed):

```text
Academy   Play | Watch | Games       Connected · Live table · Pilot
Temur vs BG · world/pack             Settings: priority stops / motion
Turn 5 · opponent's main phase       DECISION: YOU · choosing a target
┌─────────────────────────────────┬──────────────────────────────────┐
│ Opponent: life, mana, known      │ What is happening                │
│ cards + unknown hand count      │ Announcing Deadly Precision      │
│ Battlefield / public zones      │ Mode → TARGET → Cost → Payment   │
│                                 │ Announced: spell                 │
│ Ordered stack / resolution      │ Committed: none                  │
│                                 │ Your choices     9 of 9           │
│ Your battlefield, mana           │ [Find an action] [Clear filters] │
│ Your hand / private inspection  │ [Target Inspector, object A]     │
│                                 │ [Target Inspector, object B] ... │
├─────────────────────────────────┴──────────────────────────────────┤
│ What happened: committed events · automatic passes marked          │
│ Inspect selected card | recorded decisions | advice (optional)     │
└────────────────────────────────────────────────────────────────────┘
```

The sequence strip contains only applicable stages from authoritative context;
it is never a clickable back/forward wizard. No predicted stage promises an offer
that rules have not produced. Exact mana symbols, restrictions and available
pools replace an undifferentiated payment total. Opponent pools are public;
private choice candidates never cross to the other viewer.

In Watch/replay, an expandable **Why this choice** region sits beside the same
offer list: recorded policy probability per exact offer, selected offer, value
with its recorded semantics, search scores when used, and acting-view belief
summary. Compound-policy conditional factors retain their separate sequence
labels; they are not presented as native microchoice marginals. Missing critic,
historical outputs or belief evidence says **Not recorded** with the supplied
reason, never zero or a freshly rerun estimate. Newly requested advice is marked
as a new computation with its own decision/model/provenance, separate from the
historical output. Private sampled hands and branch audits are not display data.

On a narrow screen: status and decision heading first, then current context and
choices, then board and consequence history. A persistent **Inspect board / Back
to choices** link preserves keyboard position and filters without submitting.
A focused action highlights its source/targets across collapsed zones; text
identifies those objects even without color or motion. No modal hides a required
choice behind advice. Pointer, touch and keyboard activate the same buttons.

Entry and history screens:

| Screen | Primary content and action | Authority/error behavior |
|---|---|---|
| Play setup | Decks, exact opponent identity, compatibility; Start | Catalog digest, selected compiled pack and full main/sideboard setup; revalidate bytes on start |
| Watch setup | Both controllers, decks, seed; Start / Step / Play | Existing watch controller admission; never turn inspection into live human Commands |
| Games | Outcome, seats, provenance, time; Open replay | AttemptStore permissions and pagination; active/abandoned is never labeled completed |
| Replay | Decision selector, same board/context/choices, chosen offer and consequences | Viewer-projected canonical rows; historical choices read-only |
| Study | Recorded origin, alternate decision, live/recorded destination | Existing source-bound branch receipts and retained roots; unavailable means unavailable |

## Current experience and consequential evidence

Source inspected at base `a04dcdc27d6d705cb2daade8b6071b7e29c35537` on 2026-10-21.
This is a concept review of source paths and executable interactions, not a human
usability session or visual review.

* `GameBoard.svelte` arranges opponent/stack/hero; `+page.svelte` places a 300px
  Actions rail next to it. `ActionPanel.svelte` already supports Learn navigation,
  search at eight choices, full offer IDs and focus. Keep that work's semantics.
* `handleBoardTargetSelect` immediately submits the only matching action outside
  Learn. Proposed change: board selection always inspects/narrows; offer activation
  commits. This deliberate extra gesture makes commitment legible but needs review.
* `GameStore.applyFrame` maps offers back to `ActionOption`; `action_type` and flat
  `focus` are compatibility bridges. Focus mixes player/card/permanent domains;
  it is insufficient for exact source/target roles or incarnation-safe display.
  `GameSession._build_decision_context` also sets every offer's source to null
  and choices to empty, and supplies the generic instruction **Choose an action**.
  Native `decision.rs` already has typed offers, object-candidate addresses and a
  fixed-viewer semantic Observation. Reuse those authorities rather than making
  another Academy interpretation. Its `structured_search_offers` decision seam
  and `structured_offers` compound presentation API have different sequencing;
  preserve the current native decision/offer correspondence during this redesign.
* Native `agent/observation.rs` exposes mana and restricted Lesson mana; Python
  `experience_projection._serialize_player` drops these. Inspected library cards
  exist natively but the browser serializer groups only hand/graveyard/exile/stack
  and battlefield. Thus action labels alone are not a complete observation UI.
* Native `flow/action.rs` supplies sacrifice, search and color labels. The general
  `PAY_OR_NOT`, `LOOK_AND_SELECT` and `MODAL` categories conflate distinct contexts;
  client family instructions cannot explain whether costs are already committed.
* `GameSession.hero_command` validates match/revision/prompt/offer and empty
  answers, then lowers the exact offer through `step_semantic_command`. Duplicate
  command IDs return a receipt before stale checks. Do not add a second submit path.
* `_step_and_record` retains client/policy `ReplayDecision`s and clones Study roots.
  Automatic passes use authority Commands and `TraceEvent(auto=True)`, but do not
  receive canonical decision ordinals. A canonical row list is not a full execution
  tape and must never be treated as deliberate human decision count.
* `PresentationProjector` specializes authored Lightning Bolt and combat/turn
  sequences. Existing readable consequence history cannot supply missing general
  sacrifice, mana-activation or payment facts. Extend native typed event projection
  and retire the Bolt-name branch as generic facts replace its full coverage;
  do not infer Precision narration from labels or board differences.
* `_experience_frame` persists snapshots through `_persist_attempt` to the existing
  `play.sqlite`; Trace retains pre-action observations, actions, selected indices,
  auto marks and presentation, plus private canonical replay. Persistence occurs
  at surfaced frames/close, not as an atomic engine transition/acknowledgement log.
  Command receipts and engine roots live in memory. No process-crash resume or
  policy-free full reconstruction guarantee follows from this trace.
* `SESSION_REGISTRY` retains the live engine with a 15-minute lease. Recovery
  contains the current frame, receipt ledger and presentation tail; checkpoint is
  null. `shared_table_protocol.py` already owns pilot/watcher grants, table modes,
  restore/retry/return and capabilities. Do not invent another session/identity DB.
* `GameConfig.to_rust` validates main decks AND sideboards against the selected
  curated pack. `new_game` resets a disposable environment before replacing the
  current game. `opponent.py` re-hashes checkpoint bytes; `model/world.py` binds the
  actual setup/content/input schema. `teacher1_evidence.runtime_fingerprints()`
  defaults to an Interactive mirror; it is not an Allies/Lessons identity probe.
  No candidate bytes were supplied, so real checkpoint admission is unproven here.

## Chosen architecture: enrich the shared decision, render it once

Keep one `GameSession`, native legality owner, SQLite attempt store and canonical
replay. Add viewer-safe explanatory fields to the shared observation/decision
contract, sourced from native typed `PendingChoice`, `Payment`, suspended decision,
stack and transition facts. Both controller inspection and browser consumers see
that contract. The frontend derives layout, filters and wording from it; it cannot
infer rules from card names, parse Oracle text, recompute feasibility or assemble
an alternative payment state.

Proposed shared context: exact actor and decision purpose; source reference;
announced mode/target clause bindings; payment base/alternative and restrictions;
reserved sacrifice; mana pool and outstanding obligation; committed cost facts;
private inspection candidates; and parent continuation. Each fact has a typed
reference and visibility scope. Expose only facts known to that viewer. Distinguish
null/absent historical evidence from an actual empty selection. Context is a
projection of existing state, never a second mutable pending-action store.
Audit native observations/encoders first: where context is new information rather
than a readable projection of existing agent data, make it available through the
shared agent observation and version its policy input/world binding. Never silently
load an old checkpoint against changed input semantics. This plan changes no legal
choice timing; any later shared cancellation or decision aggregation needs its own
approved contract and identity migration.

| Display / action | Source of truth | Boundary and consumer changes |
|---|---|---|
| Turn, life, public board, private hand and unknown counts | Fixed-viewer native Observation and redaction | Extend projection/types; preserve known-hand counts and both viewer privacy checks |
| Mana / restrictions / inspection area | Native player pools and chooser-only suspended observation | Typed native/Python/TS DTOs; policy observation audit; never expose full library |
| Who decides, why, announced intent and cost status | Semantic decision frame + PendingChoice/Payment/suspended continuation | Shared typed explanatory context; Academy validates and forwards; live/replay/Study render identically |
| Every legal choice, source and target roles | Native offer identity and exact bound subjects | One button per offer, grouping retains all IDs; typed role references replace overloaded numeric focus |
| Commit / stale / retry | Command bound to match/revision/prompt/offer, existing receipts | Transport retains submitted ID while uncertain; stale recovery replaces frame and clears navigation; never retry against a new offer |
| Consequences / cost commitments | Native transition events and PresentationProjector | Extend typed events where sacrifice/mana/payment meaning is absent; ordered persistence and viewer projection, not inferred board diffs |
| Saved decision / fork / return | Canonical replay, ed2 pre-command address, separately validated landmark Command and retained root | Existing Study provider; keep erd1 historical binding and missing-root errors |
| Table role / permission / reconnect | Shared-table grant revision + session lease | Reuse current access control; observing an offer grants no capability to submit |
| Model identity / game history | Verified digest/setup binding and AttemptStore | No new identity authority; preserve record provenance and immutable historical bytes |
| Why the bot chose / optional advice | TraceEvent.bot_decision and existing source-bound advice identities | Preserve action-index-to-offer binding at capture, value semantics, conditional factors and unavailable reasons; no policy invocation to populate history |

Versioning: use optional, omitted-when-absent extensions for explanatory fields
that preserve existing semantics; keep historical frame hashes and addresses
byte-identical. A change to required meaning, action schema, policy input or world
requires a new explicit version/binding, rejection at incompatible readers, and new
current-world fixtures. Update all embedding schemas: replay, Study protocol/index/
recorded decisions, advice and experience. Regeneration is not permission to rewrite
frozen evidence. Historical frames without context say **Details not recorded**.

### Delete — do not maintain

The board auto-submit branch, duplicated table retry buttons, Bolt-name dispatch,
`academy/learn_demo.py`, `StudyBranch.current_observation`, and
`academy/ui/src/lib/action-map.ts` with its exclusive unit tests are removed. All demos now use `PlayDemo` and one checked-prefix
executor; validation aliases preserve the retained Learn fixture format. The
surviving offer navigation tests cover typed subjects and historical focus.
Presentation staging consumes `InteractionOffer` directly: remove numeric-focus
identity guesses and zero-incarnation fabrication rather than maintaining two
identity authorities. After casting, retain only the map from native source IDs to stack IDs;
source/controller/targets have already produced their presentation facts.

No further unconditional deletion targets remain. Keep `GameStore.applyFrame`'s
raw action projection for its existing socket/notes consumers and retained APIs;
table navigation uses direct offers. Keep `DECISION_PROMPTS` only as historical
presentation fallback when native instructions are absent. Keep
`StudyBranch.structured_offers`/`submit`: their independent benchmark contract and
isolation/rejection tests in `docs/benchmarks/study-branch-contract-v1.md` and
`tests/academy/test_study_branch.py` still require that API. Current table retries
use the semantic path. Removing those retained surfaces requires a separate
consumer cutover, not repairs or extensions here. Existing combat, death, turn,
reduced-motion timing, skip/history, permissions and source identities survive.

Forbidden: parallel HumanSession/payment draft, human-only legal actions, local
payment suggestions, dual writes, a Legacy/New table split, hidden strategic choices,
new meanings for historical offer IDs, and calling an automated game human evidence.

## Exact Deadly Precision walkthrough

This is the existing native sequence, with proposed human explanation alongside
it. Directed source: `managym/tests/rules/sacrifice_cards.rs`,
`precision_has_a_real_four_mana_alternative` and
`treasure_completes_an_announced_cost_without_a_priority_window`; payment ownership
is `flow/play.rs`, `flow/mana.rs`, `flow/decision.rs`. No retained Academy prefix
for this position exists yet. Native scenario coverage is not browser certification.

Use a legal position with Deadly Precision, Callous Inspector as target and
sacrifice candidate, a Treasure, and no pooled mana. A separate branch has BGGGG
pooled mana to test the four-mana alternative. This is a directed test setup, not
an assertion that an ordinary seeded draw reaches it.

| Step / committed Command | Human display proposed | Agent/native choices and actual commitment |
|---|---|---|
| Cast spell offer | Announcing Deadly Precision | CastSpell starts the pending announcement; no UI-generated compound action |
| Choose target Inspector | Target: Inspector, exact object reference | ChooseTarget fixes the announced target before additional cost/payment |
| Choose sacrifice or extra mana | Additional cost: sacrifice eligible creature OR pay {4} extra, base {B} still due | SelectCard reserves sacrifice; PayCost selects mana alternative. Only feasible alternatives are offered. Selecting sacrifice does not kill it yet |
| Select Inspector sacrifice branch | Reserved: Inspector; pending base {B}; Treasure available | ManaPayment excludes the reserved sacrifice as a mana source. If no chosen-mana ability remains, engine may finish payment immediately; UI must not manufacture another confirmation |
| Activate Treasure | Committed: Treasure sacrificed; choose mana for this spell | ActivateAbility pays tap/self-sacrifice immediately, then suspends ManaChoice; no response window opens. Inspector remains reserved, not sacrificed |
| Choose Add {B} | Pool: {B}; finish announced cost | The directed fixture offers exactly Add {B}, because other colors cannot complete payment. Keep that explicit single choice; no human-only auto-selection |
| Pay cost | Committed: {B} spent, Inspector sacrificed; spell on stack at original target | PayCost commits full payment once, pushes spell, and native priority/trigger ordering resumes. No response between Treasure/color/payment steps |
| Responses then resolution | Show actual pending priority owner and sacrifice/death triggers, then resolution consequences | Target can have been sacrificed as cost. On resolution a spell with no legal remaining target does not destroy another object. Never redraw a target or claim it died from Precision |

Four-mana branch: with BGGGG pooled, choose the same target then **Pay {4}
instead of sacrificing**. Payment spends the additional four generic plus base
black, preserving black for the base obligation. In the existing test no separate
mana ability remains, so this Command completes payment immediately. Inspector
stays on battlefield until the spell resolves and destroys it. With additional
chosen-mana abilities available, a pending payment can instead remain: render
whatever frame the engine returns, not a fixed page sequence.

Reconnect checkpoints in the Treasure branch: (1) before activation, Treasure is
present and Inspector reserved; (2) after activation, Treasure is gone and color
choice is pending, with no black added yet; (3) after color, black is present and
PayCost pending; (4) after payment, Inspector is gone and priority/trigger processing
has resumed. Resume each exact live frame and submit the original stale Command
once: no extra sacrifice, mana, cost or revision. Lost acknowledgement resends the
same Command ID and gets the existing receipt while retained. Lease expiry is an
unavailable-live-session state, not a refund or reconstructed payment.

## Challenges and counterexamples

| Case | Required behavior and what would invalidate the design |
|---|---|
| Ordinary land/play/pass | Enter game, inspect hand, activate exact Play land, see next actor and event; card inspection must not submit or imply every card is playable |
| Large choice / duplicate copies | Search nine-target retained fixture, locate both same-name copies, clear filter and compare complete offer-ID set; a hidden legal offer or merged copy fails |
| Learn | Navigation Take a Lesson / Discard and draw / Decline, previews and local Back; only final exact offer commits. Back cannot undo discard/draw or reveal sideboard to opponent |
| Modal/targeted cast | Modes precede target clauses and payment; independent clauses can choose same object when legal. A UI enforcing global distinctness changes strategy and fails |
| Combat | Show exact attacker/blocker declaration and nonparticipation offers with object identity; no local legal-block calculator or hidden finish/decline. State clearly which declaration has committed |
| Cost vs effect sacrifice | Deadly Dispute/Precision reserve then commit additional costs before responses; Cycle of Renewal sacrifices at resolution. Shared context must distinguish these despite common SelectCard shape |
| Treasure/color | Immediate activation commits before color, supports payment continuation and exposes every feasible color. Showing all five when only black is legal fails |
| Private fetch/search | Inspector shows chooser's live candidates and typed entry rule, including legal failure-to-find; opponent sees permitted outcomes only. Search candidates are not knowledge of fixed library order |
| Stale / lost receipt | Existing frame binding rejects without mutation; duplicate ID does not execute twice. Refresh clears filters/context caches; never relabel old offer with current meaning |
| Reconnect / permissions | Actual pending actor/context restores at each payment boundary; watcher and stale pilot grants cannot submit. Disconnected UI cannot claim submission succeeded |
| Replay / Study | Read recorded frame/offer/events without invoking policy. Exact branch return restores source address/frame and leaves live game untouched; canonical rows alone cannot reconstruct omitted automatic transitions |

## Alternatives, success and failure

A visual polish of the current Actions rail is cheap but leaves missing pools,
private inspection and overloaded decision purpose unresolved. A human-only casting
wizard conflicts with the accepted shared-action principle and commit boundaries.
A wholesale session/event-store replacement duplicates working authority and would
make this redesign depend on crash recovery research. Choose shared context plus
one table renderer; defer process-crash resumption as separate work.

Wild success: a person can explain whose decision this is, identify all available
information, locate a specific same-name target and distinguish paid from reserved
costs without a coach; Watch teaches the identical decision sequence. Wild failure:
an attractive payment strip lies after reload, hides a one-option decision, or
predicts a destruction that never happened. Countermeasures are exact native
context and boundary fixtures, not more frontend rules.

Game alignment: `wave/game/GOAL.md` calls Academy a research instrument for loading
checkpoints, watching, inspecting and forking decisions. This redesign serves those
four measures through the shared table. It does not expand the product into a
general Magic client, deck builder, chat or hosted service. The Task's named
Elemental/sacrifice cases are directed contract challenges, not ten-deck admission.
`wave/game/MEMORY.md`'s advice latency budgets and evidence limitations remain
unchanged; advice restart and Wave memory maintenance are excluded.

## Implementation sequence and demo

**Demo:** `./scripts/play` opens Play; a directed Deadly Precision demo follows the
normal GameSession/Command path. Inspecting a card exposes choices without playing
it, Treasure disappears on its committed activation, reload returns to color
selection, and saved replay/Study display the same context and consequences.
Launcher: `./scripts/play --demo precision`; the mana alternative uses
`./scripts/play --demo precision-mana`.
Both demos return an initial recovery envelope at their final prefix position.

1. **This slice — shared context and fixture:** inventory the native/player/encoder
   observation fields, add typed viewer-safe decision/payment and inspected-card
   projections, typed offer subjects and missing consequence facts at the existing
   owners. Version only boundaries whose meaning changes. Add a reproducible
   GameSession Precision prefix in both cost branches and a pre/post-Treasure
   recovery fixture. Focused proof: exact actor, target/reserved/committed status,
   pools and offer IDs agree with native state for both viewers at every step;
   stale/duplicate Commands cannot mutate twice. Preserve historical bytes.
2. Cut over live table in one coherent frontend change: direct offers, explicit
   commit, board inspection, full-set search/grouping, context/mana/private cards,
   accessible narrow/wide layout. Delete board auto-submit and superseded bridges
   in the same cut. Preserve Learn's accepted navigation and priority-stop semantics.
3. Cut over Replay/Watch/Study to the same renderer and retained context, with mode
   permissions and unavailable historical details. Keep analysis visibility separate;
   preserve branch return and live state. Do not add another replay authority.
4. Gate the complete experience, update usage docs, and present the design/runtime
   distinction with limitations. No publishing/deployment/training is authorized.

Each internal cut is part of this complete design; none alone establishes the
redesign outcome. Human demo acceptance remains separate from implementation.

## Done when / unattended proof plan

The design deliverable is ready for review when this document describes all
principal screens, authority mappings, counterexamples, exact payment/recovery
sequence and unresolved choices, and worker guidance records only accepted policy.
Runtime acceptance below is future work, not a pass ledger.

* Native debug tests preserve legal choice sets, full replay/rollback, target
  incarnation and private search behavior: `cargo test --manifest-path
  managym/Cargo.toml sacrifice_cards` and affected observation/semantic tests.
* One integration scenario crosses native state → GameSession frame → Command →
  SQLite → viewer replay → Study return at every Precision boundary. Add
  `tests/academy/test_decision_surface.py`; expected identical legal offer IDs,
  current actor/context, zero unauthorized fields, stale no-op and duplicate
  single execution. Run `uv run --extra dev pytest
  tests/academy/test_decision_surface.py tests/academy/test_choice_navigation.py
  tests/academy/test_shared_table.py tests/academy/test_recorded_attempts.py
  tests/academy/test_study_runtime.py -q` once at gate.
* Add `academy/ui/e2e/decision-surface.spec.ts` using the real demo prefix/server;
  test pointer/keyboard, 390px/wide layouts, reduced motion, both cost branches,
  disconnect before/after activation and color, lost ack, replay and exact return.
  Run `npm --prefix academy/ui run test:e2e -- decision-surface.spec.ts learn.spec.ts
  choice-navigation.spec.ts shared-table.spec.ts`; expected no hidden choices,
  no navigation Commands, correct privacy/commitment text and no accessibility
  violations. Include it in the release selection explicitly. Visual judgment
  and pinned Linux snapshots remain a capable CI/review boundary.
* Run `npm --prefix academy/ui run check` and `npm --prefix academy/ui test`, plus
  three-language schema/embedded reader tests. New private context is tested in
  imported replay/Study/advice readers as well as live redaction.
* If native observation meaning/shape changes, rebuild extension from root and
  run AGENTS.md's arena/monitor checks and full learnability diagnostic; those
  costs belong to implementing a shared contract change, not this design step.

Operational boundary: preserve existing one-step Commands and live-session lease;
no new subprocess/model call in normal render/recover, no uncapped client-local
rules search. No numerical performance improvement is claimed without measurement.
Keep the full legal set; benchmark dense fixtures before selecting virtualization.

Proof boundaries: synthetic checkpoints certify loader plumbing only; a compatible
real digest plus supplied bytes is needed for candidate admission. Automated
completion is software evidence. Dated human completion and later return are
separate evidence, unavailable in this headless run. Existing ETU-75 Learn approval
is not reopened and is not promoted to whole-product usability approval.

Earlier implementation evidence (2026-10-21): Study protocol/runtime,
Python/Rust/TypeScript embedded readers, retained navigation, native Treasure debug
assertions and 101 frontend unit tests passed. Real Precision browser scenarios
passed at 390px and 1500px with Axe, reload, duplicate receipt and the four-mana
alternative. The default-play import test and demo-origin assertion passed. These
results remain software evidence; human usability acceptance is separate.

Validation: `uv run --no-sync pytest tests/academy/test_presentation.py tests/academy/test_decision_surface.py tests/academy/test_learn_setup.py -q -x` — 17 passed; final presentation-only rerun 3 passed; focused offer/ActionPanel unit tests 3 passed; frontend check/build, Ruff and `git diff --check` passed. Gate owns the design-wide suites, browser acceptance and outstanding product-language review in `scratch/questions.md`.


## Accepted and excluded interactions

Jack Heart, 2026-10-21, authorized the full proposal after the original concept
review: explicit offer activation after inspection, acting-view Watch default and
the decision-first narrow layout. Shared undo/cancellation and bulk/revisable
combat declarations remain excluded. They require shared explicit actions with
information, timing, replay and irreversible-cost rules.

## Gate evidence — 2026-10-21

Gate polished the shared decision language and duplicate-copy accessibility,
replaced live navigation proofs with current explicit-decision prefixes (keeping
historical fixtures intact), and made empty offer-subject serialization consistent
between authority and canonical rows. Synchronous trace saves now borrow existing
JSON payloads instead of recursively copying all historical observations again;
policy/belief dataclasses still convert at their boundary. Persistence remains
synchronous and writes the same record.

A local five-sample benchmark of conversion plus JSON encoding for a 160-event
trace measured median **85.87 ms → 7.81 ms**, with identical **1,214,494 bytes**.
This measures serialization only, not network or end-to-end response latency.
Production performance signal: Command-to-applied-frame p95 on dense decisions,
with pilot/watcher broadcast lag recorded separately. The existing headless shared
20-action probe uses a 250 ms p95 budget and at most one revision of lag; retain
that signal for this redesign. No new telemetry service or Wave write was added.

Check ledger (results from this gate, superseding earlier implementation notes):

- Changed Python files: `uv run --no-sync ruff check <changed .py files>` and `uv run --no-sync ruff format --check <changed .py files>` — passed; unrelated existing formatting differences were left alone.
- `cargo fmt --manifest-path managym/Cargo.toml --check` — passed.
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed.
- `uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; cp312 extension replaced atomically.
- `cargo test --manifest-path managym/Cargo.toml` — affected native unit/observation, rules (289), and experience suites passed before interruption in unrelated long `search_state_contract` stress tests; full-suite pass is not claimed.
- `cargo test --manifest-path managym/Cargo.toml --test experience_protocol_tests --test canonical_replay_tests --test study_protocol_tests --test study_index_protocol_tests` — 19 passed, including embedded-reader privacy and generated schema checks.
- `uv run --no-sync pytest tests/academy/test_decision_surface.py tests/academy/test_learn_setup.py tests/academy/test_presentation.py -q` — 18 passed after the persistence change.
- `uv run --no-sync pytest tests/academy/test_choice_navigation.py -k current_explicit -q` — 2 passed.
- `npm --prefix academy/ui test` — 102 passed across 19 files.
- `npm --prefix academy/ui run check` — zero errors/warnings; `npm --prefix academy/ui run build` — passed.
- `npm --prefix academy/ui run validate:contrast` — all 37 token pairs meet WCAG AA.
- `npm --prefix academy/ui run test:e2e -- decision-surface.spec.ts --output test-results/surface-repair` — 3 passed (390px, 1500px, four-mana alternative); two subsequent capture scenarios passed with PNGs under `academy/ui/test-results/gate-captures/`.
- `npm --prefix academy/ui run test:e2e -- choice-navigation.spec.ts --grep 'CHOOSE_TARGET|live retained' --output test-results/navigation-repair` — 3 passed after prefix/search-copy repairs; the original run's remaining mocked navigation and Learn scenarios passed.
- Original browser selection: Learn's three outcomes/reconnect and direct demo passed; two history/import Watch tests passed. Its obsolete ID/numbering/default-view expectations were fixed and checked separately. Runs sharing a Vite server while generated files changed were not accepted as final runtime evidence.
- `uv run --no-sync pytest tests/academy/test_public_commitment_parity.py -q -x` — failed: frozen Command tape is rejected by the new explicit-decision preset. Original-collapse diagnostic still failed with `Villain offer policy selected unknown offer: 2`. Frozen authored receipt regeneration also differs in offer subjects/presentation; no frozen artifacts were rewritten.
- Broad Python attempt — interrupted after 19 passes in the fixed authored-match helper; scoped integration completed its first 33 checks (protocol/server/navigation) before being interrupted at the same helper. Neither interrupted command is a suite pass.
- `git diff --check` — passed.

Ship readiness is **blocked** by historical receipt regeneration/preset binding.
Retained reader round-trips and current runtime proofs do not close that gap.
Pinned Linux visual references and subjective visual/human usability review remain
CI/review boundaries. No full learnability diagnostic was run: native tensor shape,
rules/world identity and training presets are unchanged, and the brief excludes
training. No publication, deployment, landing or Task completion occurred.

Final scoped results:

- `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/academy/test_shared_table.py tests/academy/test_recorded_attempts.py tests/academy/test_study_runtime.py tests/academy/test_study_protocol.py tests/academy/test_study_index.py tests/academy/test_replay_index.py tests/test_package_boundaries.py tests/arena/test_models.py tests/arena/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q -k 'not live_decision_summaries_match_full_replay_without_reconstruction'` — **91 passed, 1 deselected**; the deselected fixed-match helper remains an explicit verification gap.
- `uv run --no-sync pytest tests/arena/test_models.py tests/arena/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q` — **30 passed** (also covered by the scoped command above).
- `ACADEMY_API_PORT=8013 ACADEMY_FRONTEND_PORT=5185 npm --prefix academy/ui run test:e2e -- shared-table.spec.ts watch.spec.ts study-unavailable.spec.ts --output test-results/final-mode-checks` — Watch **3 passed** and reduced-motion shared Study **1 passed**; terminal shared probe was invalidated by concurrent generated-file reload, and the Study helper's raw socket had no owner credential. Both were addressed in the final repair run.
- Stable terminal shared-table repair — **passed**, 20-action p95 **45 ms**, maximum broadcast lag **1**; deliberate/automatic decisions retain their actual ordinal in accessibility announcements.

The Study source helper now supplies the browser's existing player credential;
shared nonparticipant replay permissions remain unchanged. Frontend generation
must finish before Vite browser proofs; AGENTS.md records this harness invariant.
- Final stable shared-table/Study repair command (`ACADEMY_API_PORT=8013 ACADEMY_FRONTEND_PORT=5185 npm --prefix academy/ui run test:e2e -- shared-table.spec.ts study-unavailable.spec.ts --grep 'terminal Study|normal Study' --output test-results/gate-final-repairs`) — terminal shared table passed; Study exposed its missing raw-socket participant credential. The helper now retains that returned credential alongside the player's profile credential, using the existing record access path.
- `ACADEMY_API_PORT=8013 ACADEMY_FRONTEND_PORT=5185 npm --prefix academy/ui run test:e2e -- study-unavailable.spec.ts --output test-results/gate-study-owner` — **1 passed**: owner-authorized restore, retry, sealed/unavailable evidence, keyboard focus and exact return. This supersedes the earlier helper-credential failures.

All requested current browser scenarios now have passing evidence across the
focused repair runs, including explicit spectator reveal in Watch. The frozen
receipt blocker and fixed-match summary exclusion remain unresolved; these are
not waived by the passing current-runtime scenarios. Linux visual-reference
acceptance, full learnability and the interrupted native stress suite are not
claimed as passed.

### Follow-up gate after bounded battlefield grouping — 2026-10-21

The new unscored explicit navigation fixture now records native permanent types
and token flags. Refreshing it replayed every original Command and asserted
unchanged offer sets, state digests and all previously recorded projection fields.
Historical fixtures remain byte-identical; their compatibility comparison omits
only the added optional permanent metadata, as it already does for other new
explanation fields. No frozen receipt assertions were relaxed.

An explicit **Compare advice** action now opens the collapsed Advice section,
focuses its summary and scrolls it into view. Initial advice loading still leaves
it collapsed. The existing shared-table browser tests now assert that keyboard
focus reaches Advice. The usage README also reflects board-first ordering at all
widths and bounded creature/land/other/token lanes.

The summary characterization test uses the existing small two-seat shared-table
match, through ordinary bound Commands to a terminal state. It retains all
original comparisons against the full canonical replay, both-seat coverage and
the guards forbidding replay reconstruction. It no longer depends on the frozen
authored-match policy's historical execution preset.

The v1 public-commitment verifier still fails. A separate process loaded the
unmodified HEAD server, presentation and Learn-prefix modules and reproduced
the historical-collapse failure: revision 18, `DECLARE_ATTACKER`, tape offer 2,
available offers 0/1. That producer/world binding issue predates this diff; it
does not waive the current runtime's verifier failure or prove compatibility.

Current check results (each command runs from the repository root):

- `npm --prefix academy/ui test` — 105 passed in 20 files.
- `npm --prefix academy/ui run check` and `npm --prefix academy/ui run build` — passed; repeated only after the explicit Advice navigation repair, zero Svelte errors/warnings.
- `npm --prefix academy/ui run validate:contrast` — all 37 token pairs meet AA.
- `uv run --no-sync ruff check <changed .py files>` and `uv run --no-sync ruff format --check <changed .py files>` — passed for 25 changed/new Python files after standard formatting of the changed verifier test.
- `cargo fmt --manifest-path managym/Cargo.toml --check` and `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed.
- `uv build --wheel --out-dir /tmp/academy-gate-dist` — passed; archive inspection found only the six immutable UI pack files under `academy/ui`, all byte-identical, with no Node application/dependencies.
- `uv run --no-sync pytest tests/academy/test_choice_navigation.py -q` — 7 passed after the optional-field fixture repair.
- `uv run --no-sync pytest tests/academy/test_shared_table.py::test_live_decision_summaries_match_full_replay_without_reconstruction -q` — 1 passed; closes the earlier excluded summary check.
- `uv run --no-sync pytest tests/academy/test_public_commitment_parity.py -q -x` — failed, frozen Command rejected; separate HEAD/historical-collapse diagnostic reproduced the revision-18 offer divergence described above.
- `cargo test --manifest-path managym/Cargo.toml --test experience_protocol_tests --test canonical_replay_tests --test study_protocol_tests --test study_index_protocol_tests --test rules_tests` — **308 passed** (289 rules, 19 schema/replay/Study); debug build, no native test deferral in this selection.
- `uv run --no-sync pytest tests/arena/test_models.py tests/arena/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q` — **30 passed**.
- `uv run --no-sync pytest tests/academy/test_decision_surface.py tests/academy/test_choice_navigation.py tests/academy/test_recorded_attempts.py tests/academy/test_study_runtime.py tests/academy/test_shared_table.py -q -k 'not live_decision_summaries_match_full_replay_without_reconstruction'` — initial run **21 passed, 6 failed, 1 deselected**; all six navigation failures repaired by the seven-test navigation rerun above, and the previously deselected summary passes separately. The combined focused results cover all 28 selected cases without the previous summary exclusion.
- `ACADEMY_API_PORT=8027 ACADEMY_FRONTEND_PORT=5197 npm --prefix academy/ui run test:e2e -- decision-surface.spec.ts learn.spec.ts choice-navigation.spec.ts shared-table.spec.ts --output test-results/gate-20261021` — initial **10 passed, 3 failed**; `ACADEMY_API_PORT=8027 ACADEMY_FRONTEND_PORT=5197 npm --prefix academy/ui run test:e2e -- choice-navigation.spec.ts shared-table.spec.ts --grep 'live retained CHOOSE_TARGET|shared authority through terminal Study|reduced motion preserves' --output test-results/gate-final-repair` — **3 passed** after the two repairs. All 13 cases have passing evidence across those runs.
- `uv run --no-sync pytest tests/academy/test_learn_setup.py tests/academy/test_presentation.py tests/academy/test_experience_protocol.py tests/academy/test_study_protocol.py tests/academy/test_study_index.py tests/academy/test_replay_index.py tests/academy/test_server.py tests/academy/test_clean_machine_verifier.py tests/academy/test_play_launcher.py tests/academy/test_play_modes.py tests/semantic/test_compiler.py tests/test_package_boundaries.py -q` — **152 passed, 2 failed**; `uv run --no-sync pytest tests/academy/test_play_modes.py -q -k 'curated_combat_tape or hero_view_swaps'` — **2 passed, 10 deselected** after binding the frozen combat test to its original collapse preset and updating the serializer mock's keyword contract. The exact frozen event tape still compares byte-for-byte; no event/revision assertion changed. All 154 affected cases have passing results across the initial and focused repair runs, including complete Search/random/checkpoint matches, replay restoration, Python schema/privacy, launcher/cleanup and package boundaries.
- `git diff --check` — passed; no frozen fixture, conformance receipt or checkpoint bytes changed.

The first current browser selection passed 10 of 13 cases, including both
Precision sizes, mana alternative, Learn outcomes/recovery and retained discard.
The three failures exposed the missing fixture metadata and collapsed Advice
navigation; their focused repair run owns final results. Its 20-action shared
probe measured p95 **64 ms**, maximum broadcast lag **1**, within the existing
250 ms / one-revision budget. This is a headless local interaction measurement,
not a production latency claim.

Pinned Linux visual-reference comparison remains a CI deferral. Existing v7
reference images still describe the earlier layout; capable Linux CI must
generate/review the new layout's references before a passing visual gate can be
claimed. Local headless Precision captures live under
`academy/ui/test-results/gate-20261021/`. Human visual/usability acceptance and
the full learnability diagnostic are not claimed. No publication or landing
occurs as part of this gate step.
