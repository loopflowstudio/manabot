# LLM observation language

Status: PR #376 published at the earlier capture-v1 revision; accepted shared
native architecture implemented in `2e1a14fc0`, reconciled on 2026-10-26 after
merge `ffec2957d` of main `a0dead2b5`. Correctness gate is complete. Runpod's
27-command matched comparison passes Jack Heart's revised shipping acceptance,
including the small measured overheads. The earlier overloaded-host timings
remain diagnostic evidence, superseded for acceptance by the Runpod section below.
Current code uses policy input v2 and capture v2 while retaining semantic schema
6 and legacy checkpoint input v1. Representation and prompt presentation are
accepted; trained comparison/history selection and model experiments remain draft.
MTG-184 owns representation; MTG-185 owns the parallel model harness. No Wave
is selected. Copied from the llm-finetune design conversation on 2026-10-24.

Jack Heart's accepted intent is a small pretrained LLM that fits on one L4,
trained only enough to connect to the action space, then measured for relatively
raw playing ability. On 2026-10-24 he accepted exact action-reference output,
one repair attempt and seeded uniform legal fallback; directed matching the
non-LLM input without writable memory for the untrained version; and requested
performance and model-size comparisons. Experimental mechanisms remain proposals.

After concept review and the comparison additions, Jack Heart requested “launch
pursue” on 2026-10-24, superseding the opening hold on execution and authorizing
representation implementation and review with native fixtures. This does not
select a model, authorize paid experiments
or transfer harness ownership. Experimental choices stay with their follow-ups.

## What to build

### Conversational concept review — 2026-10-26

Jack Heart invoked concept-review while gate was running. This review leaves
that verification undisturbed and adds no implementation or publication launch.
Accepted requirements remain shared native facts, complete legal choices,
unchanged legacy checkpoint meaning and measured encoding nonregression.

The experience has three concepts: the acting player's observation (including
current pending choices), its legal decision, and the Command that commits a
choice. "Continuation" is current decision context, not another memory system
or a separate model-controlled state. Tensor rows and English are projections
of native facts; capture is an atomic evidence/execution binding, not another
game authority. Native fact extraction/encoding and Python model orchestration
are separate responsibilities; moving PyTorch or text formatting to Rust is
not required merely to avoid duplicate schemas.

Draft usage explanation: new training receives current decision context through
the default observation configuration and native reusable buffers. Loading a
checkpoint selects the input contract recorded for those weights; it never
silently upgrades an old policy. A text caller captures one decision, passes
its complete text to the model, and submits the returned A reference through
the same captured handle. The caller does not coordinate semantic-schema,
capture and checkpoint version numbers during ordinary action selection.

Concrete interaction: after choosing a Deadly Precision target, the sacrifice
decision must still identify that target in both projections. Submitting the
selected action commits one native Command. Malformed text leaves the game
unchanged and can receive the already-designed bounded protocol repair. A stale
or foreign capture is a different failure: discard the answer and obtain a new
decision; never reinterpret its A reference against a replacement action list.
Historical JSON can render but cannot regain a live submission handle.

Most consequential remaining distinction: preserved v1 checkpoints still lack
the new facts. Shared native ownership is not a claim of matched information
when comparing an LLM to those policies. A matched-input experiment needs a
policy trained on v2 and an explicit matched history treatment; old checkpoints
remain useful labeled historical baselines. Such training remains separate and
is not authorized by this review. Card text versus semantic catalog encoding
also remains a disclosed representation difference.

Review judgment: the accepted architecture is coherent; no new product choice
or redesign is required at this boundary. Clarify the usage vocabulary as above.
Performance remains an unfulfilled acceptance test, not a conceptual assurance:
report observation construction/encoding separately from the extra model work
needed to consume context. Encoding timings alone cannot establish unchanged
end-to-end training throughput. Existing gate owns final-code correctness and
paired performance evidence. No human approval beyond earlier accepted direction
is inferred from this review entry.

### Accepted architecture and performance direction — 2026-10-25

After the tensor-input collision was demonstrated, Jack Heart accepted fixing
the shared observation and versioned tensor encoding, with the requirement:
"not in any way hurting observation encoding's performance, and moving as much
as necessary to rust". This supersedes treating the LLM-only capture as the
completed shared-input solution. PR #376 remains unmerged; this is further work
in the same Task, with the accepted text presentation preserved.

The ordinary training path already uses Rust `observation_encoder::encode_into`
through `vector_env_bindings` and reusable buffers. Python's reference encoder
is not the training hot path. The revised architecture must preserve that:

Jack Heart explicitly requested "ok, lets pursue this" on 2026-10-25,
authorizing implementation, review and publication of this revision in the
existing checkout and PR. Implementation and compression are complete; the
recovered sync does not create another implementation slice.

- Own viewer-safe continuation facts in native typed observation structures,
  derived from the existing rules authority. Audit stack relationships and
  pending source, target/clause, mode, sacrifice and payment context rather
  than patching only the demonstrated target case.
- Encode these facts directly into bounded, reusable tensor buffers in Rust.
  Integrate them into the policy's actual model input; adding unused tensors is
  not a fix. Avoid per-step JSON, Pydantic validation, repeated public-definition
  construction, strings and whole-capture hashing on the tensor training path.
- Share native facts between tensor and text projections. Python owns model
  orchestration and the public call surface, not a second full observation
  schema. Remove the redundant observation/frame hierarchy introduced in
  `manabot/managym/decision_capture.py` when consumers are moved to the shared
  native contract; preserve offline validation and origin/staleness rejection.
- Keep text formatting and portable evidence serialization outside the tensor
  fast path. Stable public definitions should be reused by content identity.
- Version changed policy inputs explicitly. Frozen checkpoints and experiment
  evidence retain their original shapes and meaning; fresh training must use
  the revised inputs end to end. Do not silently add fields to old checkpoints
  or claim that old policies can use them without training.

Performance acceptance: retain before/after native encode-only and full vector
step measurements on identical seeded decision corpora, compiler profile,
hardware and buffer sizes. Include ordinary priority, target/payment and large
legal-action positions, both single-env and training batch sizes, with warmup
and repeated paired runs. Measure allocations, bytes written, latency and
throughput; separate the cost of newly represented facts from refactoring cost.
Use `scripts/bench_breakdown.py` for the real vector path and a native benchmark
for encode-only attribution. No regression beyond measurement noise is an
acceptance target, not an unmeasured promise. Resolve material regressions before
publication; an overloaded host cannot establish a nonregression claim.

Implemented proof: the retained counterexample produces distinct inputs through
both Rust and Python encoders; native/Python outputs agree, model outputs differ
and gradients reach the continuation projection, while v1 retains the collision.
The bounded layout and checkpoint admission are documented
in `docs/recent-events.md#current-continuation-input`. The active revision below
retains the remaining performance acceptance; implementation is not a timing claim.

A deterministic, viewer-safe English rendering of one native observation and
its complete legal decision, usable by an untouched LLM through its existing
tokenizer and bound back to the engine's Commands.

### Implementation finding — 2026-10-24

The original collision proof (retained at
`e929d4c7c:scratch/observation_collision.py`) disproves `Observation` as the sole renderer
input. From an ordinary priority root with Deadly Precision, a Swamp and two
Gray Ogres (one per player), real Commands cast the spell and choose different
targets in cloned branches. At the additional-cost decision, **complete schema-6
Observations compare equal**: revision, viewer-state hash, decision fingerprint,
offers and action descriptors. The same bound sacrifice Command succeeds in
both branches but produces different stack targets. Scenario helpers only
establish the root; the collision follows native Commands.

Found at `912771b9484bddf282bd11183d2f14dfbbbb8d67` and confirmed after sync to
`1e601527094601e5f152d866fbdf9f924c33e954`: `Game::semantic_observation` hashes
only `AgentObservation::to_json`. The separate shared `Game::decision_context` /
`Env.decision_context_json` exposes the acting viewer's selected targets and
clauses, but is absent from that input and hash. Neither a definition resolver
nor history-off/snapshot treatment recovers continuation facts. Refusing these
decisions would leave payment acceptance unfinished.

Capture v1 first supplied continuation through the separate presentation context.
The accepted 2026-10-25 revision supersedes that ownership: native
`agent/continuation.rs` owns typed current facts, consumed by tensor input v2,
capture v2 and `decision_context` presentation. The capture atomically binds
schema-6 Observation, shared frames, public definitions, world/content identity
and optional bounded policy history under the engine lock. Every included fact
is hashed; stable public definitions/content hashes are cached per native pack.
Presentation `recent_facts` is excluded; declared public policy history v1 is
the optional suffix (demo: 32 rows). No writable model memory is added.

The embedded semantic schema remains byte-compatible at version 6; capture
inclusion/order/digest changes use `DECISION_CAPTURE_VERSION` (now 2). Existing
worlds and checkpoints retain their meanings. The collision proves missing
input facts, not broken revision rejection: ordinary Commands check revision
and legality, while text submission additionally binds originating game/branch,
reset generation and complete capture digest. Revisions alone do not identify
a game across clones. Earlier implementation detail remains at
`e929d4c7c:scratch/llm-finetune.md`.

This is an additive series. The keystone is the representation and binding;
a parallel model harness consumes that contract. Fine-tuning and playing-strength
evaluation follow, with their own designs.

## The demo

Render an actual native decision, inspect its text, return an action reference,
and execute the corresponding revision-bound Command. Repeat at target, combat,
payment, and selection decisions, including two copies of the same card. Show
that every legal alternative remains selectable. A text response stub suffices
for the representation demo; the parallel harness substitutes a real LLM.
Retain exact inputs, bindings, answers, selection origins and execution receipts,
labeled with world, representation and memory treatment. If history is selected,
include expired references and equal current states with different public events.

Illustrative layout, not a captured observation or final field-complete fixture:

```text
You are choosing one legal action in the current game.
Return only its action reference, such as A2.

Turn 4. Your first main phase. You have priority.
You: 14 life. Opponent: 11 life; 3 cards in hand, identities hidden.
Your mana pool: empty.

Your hand:
C1 — Lightning Bolt [D1]
Your battlefield:
P1 — Mountain [D2]; untapped.
P2 — Mountain [D2]; tapped.
Opponent battlefield: empty.
Stack: empty.

Card definitions:
D1 — Lightning Bolt. Cost {R}. Instant.
Rules: Lightning Bolt deals 3 damage to any target.
D2 — Mountain. Basic land — Mountain.

Legal actions:
A0 — Cast C1, Lightning Bolt. Further choices occur when offered.
A1 — Pass priority.
```

The real renderer derives every line from the bound game, including its actual
card text and legal payment/target sequence. This example supplies no rule for
guessing castability, effects, or future choices.

## Output protocol and recovery

Accepted by Jack Heart on 2026-10-24: one initial response, at most one repair
response, then a seeded uniform draw from the bound legal actions. This enables
untouched-model play before deciding whether fine-tuning is needed.

The response is exactly one listed reference, such as `A1`, allowing surrounding
whitespace. Reject commentary, multiple references, unknown IDs and empty or
incomplete output. Give both calls fixed generation limits within one decision
deadline. On a protocol error, retain the original observation and failed answer
and append: “Invalid response. Return exactly one action reference from the
legal-actions list, with no other text.” Supply no tactical advice. The retry
does not advance the game or change the offered action set.

After two invalid answers, the harness emits an explicitly marked fallback
selection. Uniform sampling avoids tying fallback strategy to native action
order; retain its RNG seed/state and selected reference. The resolver builds
the same ordinary bound Command for either origin.

Only protocol failures receive this recovery. Representation overflow, model
runtime errors, exhausted deadlines, stale bindings and engine rejection stop
the attempt with evidence; they are not fixed by choosing a different action.
Never retry an uncertain execution or replay a committed cost.

Record raw responses, validation errors, attempts, tokens, latency and selection
origin (`initial`, `retry`, `fallback`) at each decision. Report first-response
validity, validity after repair, fallback decisions and games with any fallback.
Full-cohort playing scores describe the LLM plus this recovery policy, never
pure model play. Do not discard fallback games to improve the reported score.

## Representation

Use templated English with fixed section order and explicit field labels.
Retain ordinary words, names, numbers and mana notation. Apply the selected
model's unchanged tokenizer afterward. JSON is the storage envelope, not the
primary model prompt; dense tensor dumps and learned special tokens would
require learning meanings before testing pretrained knowledge.

1. Perspective, turn/phase/step, current decision kind and terminal status.
2. Both players' visible quantities: life, zone counts, mana including
   restrictions/expiry, and publicly known hand minima. Distinguish unknown,
   zero and empty. Include sideboard information only as projected to this viewer.
3. Visible cards by zone, private own hand and currently revealed selections;
   battlefield state including effective statistics, damage, counters, tapped
   state, summoning sickness, temporary effects and keywords.
4. Ordered stack entries, controllers, modes, abilities and visible targets;
   current continuation source, selected targets/clauses, chosen mode, reserved
   sacrifice and outstanding payment from the shared capture, even before the
   spell reaches the stack. These facts are required in either history treatment.
5. Card definitions referenced above or by legal choices: name, characteristics,
   native rules text, costs and printed keywords, deduplicated by bound definition.
6. Every current legal action, with its distinct reference, verb, visible focus,
   declaration, mode and ability selectors. Never infer legality from prose.

Use decision-local aliases for objects and definitions; distinct copies remain
distinct. Keep native IDs, revision and fingerprint in the binding envelope,
outside model text. Resolve aliases only within that envelope. Preserve ordered
relations; normalize only order that carries no game meaning. No importance
ranking, strategic annotations, recommendations or human-only shortcuts.

Audit every source field as rendered, binding-only or redundant with a named
rendered field. Unresolved references, unsupported fields/decisions and token
overflow produce typed failures, never clipped states or smaller action menus.
An absent upstream fact remains unavailable. Resolve ambiguous cost, object or
pending-decision meaning at the shared observation authority if necessary;
do not reconstruct private state or add an LLM-only rules solver.

## Memory direction: match the non-LLM input

### Verified tensor-input omission — 2026-10-25

During review Jack Heart asked whether the capture implies missing information
for ordinary agents. Directly encoded both native `collision_branches()` using
`ObservationSpace.encode(env.observation_for_player(actor))`: all 19 tensor
fields are identical with both `policy_history_version=0` and `=1`, while the
selected continuation targets differ. Receipt:
`.runs/tensor-continuation-collision.json`. This extends the earlier JSON-only
counterexample to the actual tensor encoder. It proves missing pending-target
information in this position, not a measured playing-strength loss or the full
extent of omitted information. The LLM capture exposes facts the current tensor
policy does not receive. This finding led to the accepted 2026-10-25 shared observation/encoding revision
above. The local revision now separates this counterexample; full LLM-versus-policy
input parity still depends on the selected history and representation treatment. Existing
checkpoints must retain their original input meaning.

The intended experiment asks about relatively raw playing ability. A fresh
current-state-only prompt also measures a deliberately memoryless policy. That
can be a useful baseline, but the transport's present omissions should not decide
the experiment implicitly. Current card text and board state do not generally
recover the order of prior casts, which spell was countered, or the sources of
earlier damage and life changes.

Composite `events` and `recent_events` are empty on the inspected semantic
observation path; its JSON omits `policy_history`. That omission does not select
a memory treatment. Never fabricate history from audit logs. Previous-state
lists remain an alternative experiment, not an accepted baseline.

The ordinary ActorCritic's optional `recent_events=True` path consumes native
public history v1: at most 32 event records covering arrivals, damage, life,
casts, resolutions and counters. A "row" is one such record: event kind, signed
amount and source/target references where available. One action can emit several
records; an event with multiple targets expands into one row per target. The
window therefore counts neither turns, decisions nor previous states.
`RecentEventEncoder` is stateless across decisions;
it does not retain prior observations or model-written notes. `recent_events`
defaults to false, so this history is an available treatment, not evidence that
every existing checkpoint used it. The older transition-event path is different
again; history-off must not be equated with all event input being absent. These
contracts are described in `docs/recent-events.md` and implemented by
`manabot/env/observation.py` and `manabot/model/recent_events.py`.

Proposed baseline: render current state plus the same admitted native history
rows as a declared history-on non-LLM configuration, using the same configured
window (1..32 rows), chronological order and available facts. It is neither full
viewer history nor all public events. Keep unavailable sources and expired object
links explicit; never follow an old object reference through a hidden zone or a
new incarnation. Select the comparison configuration before claiming matched
inputs. Natural-language card rules versus semantic catalog programs are another
representation difference to disclose; sharing history is not full input parity.

Consequences if selected: expose a typed, versioned projection through the shared
observation authority; bind the included history to observation identity and
retained input; render it in manabot; let MTG-185 consume the resulting text without
reconstructing events. Freeze the event coverage, order and bound as a declared
memory treatment. Measure its token contribution. This requires shared API and
viewer-safety proof; it is not permission to serialize an audit log or change
frozen worlds. No persistent model chat or LLM-generated memory is proposed.

Alternative: retain current-state-only v1 and explicitly call its experiment a
snapshot baseline. Include all currently projected known facts, including known
hand minima; current-only must not mean deleting remembered facts already present
in the authoritative state. A history treatment can then be compared separately.

Counterexamples to preserve: two histories can lead to the same displayed state
but supply different public evidence; a bounded suffix can omit an older relevant
event; an object that leaves and re-enters play cannot inherit its former alias;
an earlier private observation cannot be exposed to the other player. These limit
the claims and the renderer even if basic action parsing succeeds.

The LLM receives a fresh prompt per decision, rebuilt from authoritative inputs.
No model-generated notes or chat transcript carry across decisions. Only the
accepted single repair reuses a failed response within that same decision.
Adding a previous-state list would need a separately declared sampling/boundary
and visibility contract; a model-written buffer is excluded from the untrained
baseline. These additions are not needed merely to express the existing public
event input in English.

## Current system

- `managym/src/agent/observation.rs` owns visible cards, native `text_box`,
  effective permanents, stack and raw action descriptors; `to_json` projects them.
- `managym/src/decision.rs` composes viewer state and `DecisionFrame`.
  `structured_search_offers` emits one offer per native action in the same order.
  Its labels can be enum names, sources are null, and offers have empty choices.
- `manabot/managym/decision.py` mirrors Observation/DecisionFrame/Command.
  `manabot/semantic/decision_contract.py` binds and submits Commands.
- `manabot/env/observation.py` is the fixed tensor encoder, not the text source.
  `manabot/managym/experience_projection.py` already formats some action labels;
  inspect/reuse suitable shared formatting rather than importing Academy.
- No existing LLM runtime or fine-tuning implementation was found. Compound
  decoding is an existing alternative interface, not this initial integration.

Upstream reconciliation, 2026-10-26: merge `ffec2957d` retains main through
`a0dead2b5`, including batched checkpoint evaluation/resume (`b7b49467b`) and
failed-game-only journal retention (`aa6f4652d`). `eval.match.play_cell` now admits
resident policies before spawning bounded games; `sim.serving.PolicyStore`
loads through `model.checkpoint`, verifies checkpoint bytes and observation ABI,
and batches compatible requests. `local_checkpoint_file` keeps verified local
artifact admission and the TYPE_CHECKING-only managed Agent import; the local
CLI must still work without training/cloud/callback imports.

Checkpoint loading and catalog discovery use `ObservationConfig.from_checkpoint`.
So do ordinary/scenario replay roots; the new `sim.replay.replay_prefix` inherits
that admission for retained-input calibration. Prefix reconstruction is not a
complete trace-verification receipt. Full replay and pending-failure evidence
keep their existing owners and frozen meanings.

Reconciliation repair: `eval.match` serializes the admitted configuration with
`model_dump`, which intentionally omits v1. The new `sim.match` worker had used
fresh-default `model_validate`, silently reconstructing v2. It now uses
`from_checkpoint`; bounded serving/replay tests cover absent-version v1 and
explicit v2. The fresh-input serving fixture now passes an explicit fresh space
instead of relying on absent retained configuration to mean v2. Neither repair
changes checkpoint bytes, architecture-receipt exclusions or replay identities.

The LLM follow-up can reuse scheduling, serving measurements and replay owners,
but still needs an explicit adapter: runners receive integer actions from
`act(env, obs)` or inference replies, not text-binding/response-attempt evidence.
The adapter must retain captured Commands, selection origin and failure policy.
Shared serving is not an LLM integration or this revision's encode-performance
proof. `912771b94` removed supervised policy training; interface fitting remains
a new design. The newly retained MTG-181/186 research does not select a comparator
or authorize relabeling its v1 evidence as v2.

Labels alone are insufficient. Pair offers with action descriptors from the
same composite observation using their native order contract; validate count,
schema and binding, and render visible focus/selector distinctions. Maintain
every microdecision in v1, including committed costs and later choices.

## Data structures and key functions

Implemented owner: `manabot/semantic/text_observation.py`. Native observation
and continuation types validate portable inputs; `manabot/managym/decision_capture.py`
keeps the small content/presentation envelope and reuses `decision.Observation`.

- `TextDecision`: representation version, prompt text, typed legal references,
  source Observation identity and immutable Command bindings.
- `TextAction`: reference, description and one exact bound Command.
- `TextEncodingError`: unsupported schema/field/reference or unavailable meaning.
- `render_decision(capture_json: str) -> TextDecision`: pure projection of
  validated, digest-checked capture v2. Bound definitions are inside the capture.
  `capture_text_decision(env, history_capacity=...)` attaches the opaque native
  handle. Offline JSON can render but cannot regain submission authority.
- `resolve_action(decision: TextDecision, response: str) -> Command`: accept
  exactly one listed reference after trimming whitespace; unknown IDs,
  commentary and malformed answers fail. `submit_response(env, decision, response)`
  checks the native origin Arc, reset generation and entire current capture digest
  under the same lock as ordinary Command execution. The harness handles repair.
- Harness `SelectionResult`: chosen reference, origin, ordered response attempts,
  validation errors and fallback RNG evidence; never an invented successful reply.

Prompt and artifact versions bind native world/content/schema and renderer
identity. The harness separately binds model/tokenizer revisions, chat template,
precision, decoding settings and actual token IDs. This adds a derived format,
with its own capture-v2 compatibility/versioning boundary. Historical worlds and checkpoints retain their existing
meaning; the renderer version cannot stand in for a shared schema migration.

## Delete — do not maintain

Completed deletion cut: removed capture's duplicate Python Observation, Frame,
ViewerState and card/player/action hierarchy; retained native validation and the
shared semantic Observation. Replaced `decision_context`'s independent fact
extraction with projection of native continuation frames. Frozen schema-6 JSON,
legacy tensor layouts and origin/stale rejection remain.

Removed the superseded `scratch/observation_collision.py`; its native setup and
assertions now live in `verify.text_observation.collision_branches()` and the
text/continuation regression tests. The original remains at
`e929d4c7c:scratch/observation_collision.py`. No exclusive fixture is needed.
Removed the unused `model.world.POLICY_INPUT_VERSION`; `ObservationConfig`
owns the selected input version. No remaining deletion target is identified.

Owned encoder buffers expose one mutable view used by allocating encoding and
benchmarks. Python encoding reads only the supplied observation. The model's
`shared_context` combines history and continuation. Native collection separates
public stack, announcement and suspended frames; the renderer separates player,
card, permanent and definition sections. These boundaries keep visibility/order
and prompt assembly reviewable without mixing each field's formatting into them.
Capture reuses the native input returned by semantic projection and feeds its
frames directly to presentation. Removed its second viewer projection and the
third continuation extraction, along with building then clearing prose history.
Presentation retains its own chronological 12-fact suffix without intermediate
reverse/collect vectors. The renderer derives stack aliases directly from native
order; its redundant stack-ID lookup is removed. No further obsolete code or
exclusive fixture is identified.

Pre-edit source, notes and full prompt/capture pairs remain in
`.runs/compress-input-2026-10-25/` and `.runs/compress-review-2026-10-25/`.
The latter also retains both viewers' v1/v2 native/reference input arrays.

Historical note removals:

- `scratch/elemental-baseline-preflight.md`, `scratch/elemental-mana-repair.md`
  and historical MTG-174/176/180/182 questions remain at
  `912771b9484bddf282bd11183d2f14dfbbbb8d67:scratch/`.
- `scratch/17lands-for-bots.md`, `scratch/17lands-for-bots-evidence.md` and
  upstream questions restored by sync remain at
  `1e601527094601e5f152d866fbdf9f924c33e954:scratch/`.

Uncommitted originals remain in `.runs/compress-input-2026-10-24/` and
`.runs/implement-input-2026-10-24/` (including sync conflict markers). Inputs to
this consolidation remain in `.runs/compress-review-2026-10-24/`. Keeping MTG-184's
questions through that conflict selects no card-evaluation work or new contract.

The ordinary tensor policy now consumes the revised input; its v1 layout remains
only for explicitly admitted legacy checkpoints. Compound decoding, shared
projection utilities and frozen experiment evidence remain. Any temporary
hand-written harness examples are superseded by generated fixtures at integration;
they must not become a second observation renderer.

## Internal slices

1. **Implemented:** native current stack/continuation frames, Rust reusable
   relation buffers, matching Python reference encoding, model consumption and
   explicit v1/v2 checkpoint admission. Legacy schema-6 bytes remain unchanged.
2. **Implemented:** capture v2 and native offline validation; removal of duplicate
   Python observation/frame hierarchy and independent presentation extraction.
   Origin-bound submission and complete text choices remain. Gate and reliable
   paired performance acceptance are outstanding.
3. **MTG-185/follow-up:** token-length census with the selected tokenizer and
   declared game corpus; actual model context budget and inference measurements.
   The complete-prompt token-budget hook rejects overflow without truncation.

## Implemented field audit and demo

Jack Heart reviewed an actual blocker-prompt excerpt and accepted omitting
"nones" on 2026-10-24. The renderer now omits absent optional metadata, empty
keyword/subtype lists, absent definition power/toughness and absent continuation fields. Attack/block verbs express
their declaration directly. Present modes, abilities, targets, costs and other
facts remain explicit; unknown/unavailable references and meaningful zero counts
remain distinguishable. This is the parent compression pass and human demo
feedback, not approval of a playing-strength result.

Capture models reject unknown fields recursively. Native enums without a
renderer meaning, nonempty unsupported transition increments, nonacting/terminal
inputs, incomplete offers, missing definitions/links and overflow fail explicitly.
Native IDs never appear as identifiers in model text; each visible copy has a
local C alias, each outside candidate O, stack entry S and definition D.
Maps are normalized by definition key; ordered native lists retain their order.

| Source fields | Treatment |
| --- | --- |
| `capture_version`, `world_version`, `content_digest`, `digest` | Binding-only, retained with exact input; digest checked before rendering. |
| Observation identity: schema/revision/viewer/hash; frame schema/revision/actor/fingerprint | Binding-only; actor/revision consistency checked, perspective rendered. |
| `events`, `recent_events`, presentation continuation `recent_facts` | Must be empty in the text contract. Legacy prose history deliberately excluded; optional policy history is the sole declared suffix. |
| `game_over`, `won`; turn number/phase/step/active/agent | Rendered explicitly; player IDs resolve to You/Opponent. |
| Player life, zone counts, graveyard Lessons, mana, restricted Lesson mana, combat expiry total, known-hand minima, initial/remaining sideboard counts | Rendered with units and definition aliases, including zeros/empties. `is_agent`/`is_active` rendered; `id`/`player_index` bind roles. |
| Cards: name, text, zone, owner, power/toughness, token/legendary/Ally/Lesson, ward/kicker values, all card-type and keyword flags, mana cost/value/hybrid | Rendered. False type flags stated collectively; absent keywords mean false. IDs bind aliases, registry keys bind D references. Effective battlefield state remains separate from printed card facts. |
| Outside candidates: candidate/owner/definition/name/text | O aliases and rendered owner/definition/rules; never fabricated in-game objects. |
| Permanents: controller/tapped/damage/sickness/counters/effective P/T, animation, exile link, exile-if-dies, unblockability, temporary subtypes/effective keywords | Every field rendered; IDs and registry keys resolve to the same C/D aliases as backing cards. |
| Stack: ordered IDs/kind/controller/definition/source permanent/ability/mode/targets | S aliases, native top-to-bottom order; all selectors and ordered target relations rendered. Missing source stays unavailable. |
| Continuation: native frames plus actor/stage/instruction/source, targets with clauses, sacrifice reservation, mana cost, mode | Shared native frames and presentation are rendered; frame order, parents, mana outputs/restrictions and resolving ability count remain explicit. Entity/display/incarnation IDs only resolve exact aliases. Expired objects do not acquire a returned object's alias. |
| Bound definitions: name/rules/mana/types/supertypes/subtypes/keywords/P/T/kicker/ward/token | All rendered; cost mana value is redundant with cost symbols. These are native public characteristics, not a second executable rules program. |
| Visible `objects` card/permanent/incarnation joins | Binding-only alias joins; projected only for already-visible cards. |
| History version/capacity and ordered kind/amount/source/target; reference kind/definition/owner/zone/current link | All rendered; unavailable and expired links explicit. Uses existing native history admission and at most 32 rows, no ledger reconstruction or model memory. |
| Actions: type/focus/declaration/mode/ability/description | All rendered with A references, distinct copies and explicit nonparticipation. |
| Offers: id/actor/verb/source/choices/label/help/confirm/public commitment; object-candidate addresses | ID/actor and addresses binding-only. Action-aligned verb/description/focus cover label, confirm and public commitment meaning. Source/help null, choices empty in this native seam; other structures fail admission. All offers bind one exact immutable ordinary Command. |

Runnable proof: `uv run python -m manabot.verify.text_observation --out .runs/text-observation`.
The seed-23 fixture helper declares each custom deck and scenario injection in
code, then uses native Commands exclusively. Retained per-case `input.json`,
`prompt.txt`, `bindings.json`, `checks.json`, `executions.json` include exact
inputs, stub responses, initial selection origin, Commands, receipts, and
replay comparison. Native handles are deliberately not reconstructible from
JSON. `summary.json` reports 33 executed/replayed alternatives across priority,
target, attackers/blockers, private selection, both collision branches, mana payment,
five mana colors, alternative sacrifice/mana costs, three spell modes and optional kicker.
Two ordinary Bolt sequences swap public target order and reach identical
current observations with distinct public-history prompts and expired links.
Malformed, foreign-branch, stale and reset submissions cannot mutate state.
`collision_branches()` and the text/continuation regression tests preserve the
original schema-6 counterexample, its divergent stack targets and distinct inputs.

## Parallel increment and follow-ups

**Parallel harness:** load one pinned small instruction model; accept text and
legal references; apply its chat template; tokenize without truncation; generate
a bounded response; return raw output, parsed selection/failure, token counts,
latency and peak GPU memory. Apply the bounded repair/fallback policy above.
The harness owns neither observation semantics nor Commands. Fixtures allow
independent work before the renderer lands. Test
untouched inference first. Keep dependencies optional and model loading out of
the core package import path.

Candidate, not a final model decision: Qwen3-4B-Instruct-2507, a non-thinking
instruction checkpoint ([model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)).
One L4 has 24 GB ([NVIDIA specifications](https://www.nvidia.com/en-us/data-center/l4/)).
Actual inference/training fit depends on context and runtime and must be measured.
Quantized adapter training is a later option
([PEFT guide](https://huggingface.co/docs/peft/developer_guides/quantization)),
not evidence that the proposed workload already fits.

**Follow-up: raw baseline and interface fitting.** Run untouched and adapted
models on the same representation. Decide from observed failures whether any
fitting is needed. If needed, distinguish grounding an explicitly requested
action from learning which action is strategically good; do not train on random
or expert policy choices and label the resulting strategy untouched. Grammar
constraints can improve answer validity but must be a declared treatment shared
across comparison arms. Training data, objective, stopping rule and resource
allocation remain undesigned.

**Follow-up: playing evaluation.** Paired games with swapped seats against fixed
anchors, separate interface errors from strength, retain Commands and replays.
No search, rewards, strategy distillation or teacher rationales in the initial
raw-performance comparison. Card knowledge in pretraining and changes caused by
interface fitting remain confounds, not proof of general game reasoning.

## Performance and model-size comparison

Jack Heart requested performance testing and a size comparison with the models
currently being trained (2026-10-24). The requirement is accepted; the measurement
protocol below is proposed. This adds evaluation requirements, not a run or model
allocation. MTG-184 supplies representation fixtures and token measurements;
MTG-185 supplies inference measurements; the playing-evaluation follow-up combines
them with admitted trained checkpoints.

**Pin the reference before measuring.** Use an actual current training checkpoint
with its artifact hash, resolved architecture, world/content binding, history
configuration, training exposures and inference settings. The checked-in ordinary
`baseline-v3` recipe is width 64, two attention layers, four heads, semantic card
input, scalar critic and FiLM policy (`manabot/recipes/baseline.py`). This is a
concrete configuration reference, not proof of which model is currently running
or the best trained checkpoint. Do not substitute a historical parameter count
or randomly initialized model for the trained reference. Report a history-on
variant separately if the selected trained checkpoint has history off; never
change its expected inputs and call it the same checkpoint.

Historical architecture measurement, 2026-10-24: a fresh instance of the then
`baseline-v3` configuration has **193,794 parameters**, **775,176 FP32 parameter
bytes** and **37,410 buffer bytes**. The resolved configuration and architecture
receipt are in `.runs/llm-observation-baseline-size.json`. This counts the current
input-v1 architecture without loading trained weights; it is not the current
input-v2 parameter count, playing evidence or the selected trained comparator.
A fresh v2 architecture measurement remains necessary for a v2 size comparison.
The selected LLM's exact count and measured
serving memory remain with MTG-185's model selection.

| Comparison | Retained measurements |
| --- | --- |
| Model size | Exact distinct parameter count, architecture, parameter/buffer bytes, stored checkpoint bytes, serving precision/quantization, and LLM-to-policy parameter and storage ratios. |
| Adaptation size | Parameters actually optimized during interface fitting, adapter bytes and fitting exposures; zero fitting for the untouched LLM. Separate these from total pretrained parameters and the non-LLM policy's trained parameters. |
| Decision speed | Warm p50/p95/p99 end-to-end decision latency and decisions/second at batch one and declared batch/concurrency settings. Break out rendering/encoding, tokenization, prefill, generation, parsing and repair; also report model-only timing. |
| Memory and startup | Load time, peak process RAM, GPU allocated/reserved memory, batch size, context length and generated tokens; include the largest admitted prompt and report OOM/overflow failures. |
| Playing performance | Paired seat-swapped W/D/L and score with uncertainty against the pinned trained policy and fixed anchors, complete-game wall time, and initial/repair/fallback rates. Retain incomplete/failed attempts separately rather than silently dropping them. |

Count parameters from the actual models, counting tied/shared parameters once.
Use the non-LLM `ArchitectureReceipt` where available and validate it against the
loaded model. Count logical parameters independently of quantized storage bytes;
the smaller adapter or a frozen inference model does not make the LLM's base
parameters disappear. Serving memory also includes activations, runtime overhead
and the LLM's KV cache, so parameter bytes alone cannot establish single-L4 fit.
The draft 4B model candidate is not an exact measured count or a selected model.

Use one frozen corpus of native decisions for timing both representations,
covering ordinary play, long prompts, many legal actions, history and awkward
payment/target/selection decisions. Retain corpus identity, decision-family mix,
prompt lengths and action counts. Run untimed warmup followed by repeated timed
passes; synchronize accelerator work when measuring it. Pin hardware, runtime,
precision, threads, batching, decoding limits and cache policy. Compare on the
same device where feasible, and label CPU/GPU or precision differences explicitly.
Measure cold loading separately. Count repair time in operational latency and
include failures in the report; report first-call timing separately for diagnosis.
Game throughput is a second measurement because different policies visit different
states and take different numbers of decisions per game.

For playing strength, freeze the world, decks, deals, seat swaps, inference
budgets, game count and failure policy before seeing scores. Report the full
LLM-plus-recovery cohort with draw-aware score and paired uncertainty; a game
containing fallback remains in that cohort. Stop conditions and infrastructure
failures remain explicit, not invented wins or losses. The untouched and any
later interface-fitted LLM are separate arms, both without writable memory in
the untrained treatment. Record input/history differences rather than claiming
fairness from matching the action protocol alone.

Present a compact result table with one row per pinned model/treatment and
columns for parameters, relative size, serving memory, decision latency,
throughput, playing score and fallback/failure rates. The question is how much
playing ability the larger pretrained model buys at its inference cost. Parameter
count is descriptive, not equal training compute: report manabot training exposure
and LLM interface-fitting exposure separately, and mark upstream pretraining
cost/data as unavailable unless documented. Do not infer a size-scaling law from
one LLM and one small policy. Exact comparator checkpoint, model, hardware,
sample sizes and performance targets remain to be selected before execution.

## Constraints and forbidden outcomes

Preserve viewer safety, the complete action set, exact continuation and stale
rejection. Never disguise a fallback as a model decision, silently automate a choice,
look up hidden cards, summarize with another LLM, or use external card text as
the engine's rules authority. Keep manabot independent of Academy. No claim of
single-L4 fit before measuring the largest admitted inputs. No paid run is part
of this design work.

## Parent gate evidence and limits

The pre-revision gate passed 101 Python cases after syncing to `66b8fb3e7`
(`.runs/mtg-184-post-sync-tests.log`); integration CI collected 614 tests.
The prior complete debug Rust suite, including 316 rules tests, and Clippy passed.
These predate the native policy-input-v2 revision and do not replace its gate.
Detailed commands and parent receipts remain at `e929d4c7c:scratch/llm-finetune.md`.

Contrary evidence remains: `.runs/learnability/mtg-184-gate/report.md` records
an error after all three seeds passed capacity/held-out fits and PPO collection
exceeded 300 seconds. Wall/CPU times were 402.5/71.5 seconds; observed contention
is a possible, unverified cause. PR #372 (`b3681297f16e27a9e42b5e49f4fe928292fe0bb2`)
subsequently removed this verifier at Jack Heart's request. Do not restore it or
relax its budget; use current AGENTS.md admission/monitoring/choice-separability
checks. No successful PPO result follows from the removal.

Human prompt review is accepted. MTG-185 owns actual tokenizer/chat-template
budgets, model loading, repair/fallback and runtime/GPU measurements. Jack Heart's
2026-10-26 direction authorizes landing after the final gate and the bounded
Runpod test recorded below. Parent owns rental cleanup and waited landing;
this gate performs no publication, merge or additional paid execution.

## Measure and evidence

### Accepted prompt review — 2026-10-24

Jack Heart reviewed the complete native blocker prompt and accepted its
presentation and omission of absent optional metadata ("ok this is fine").
One Gray Ogre could block either distinct attacking Gray Ogre or decline; all
three actions remained explicit. Exact prompt/capture:
`.runs/text-observation/interactive/0-{prompt.txt,input.json}`. The review used
native text without a visible board or LLM; no human action was executed.
Execution evidence remains the automated 33-alternative replay proof. This
accepts presentation, not model performance or PPO verification. Historical
interaction detail remains at `2e1a14fc0:scratch/llm-finetune.md`.

Measure prompt tokens p50/p95/max by decision family, section contribution,
action count and overflow fraction. Later report initial/repaired response rates
and fallback usage separately from game outcomes, alongside latency and memory.
Set no playing-strength gate for the representation itself.

`scratch/questions.md` indexes remaining experimental choices.

## Current revision and remaining gate — 2026-10-26

Implemented at `2e1a14fc0`: native stack/announcement/resolution and nested mana
frames, 64×32 reusable relation rows, object/catalog embedding joins, actual
mana outputs/restrictions and resolving ability count. Capture v2 reuses native
facts and cached definitions; duplicate Python observation models and independent
presentation extraction are removed. Layout: `docs/recent-events.md`. The
post-sync serving repair above completes the bounded reconciliation change.
No Wave is associated; Wave files remain read-only.

Retained implementation evidence: collision, parity, gradients and checkpoint
checks passed; historical ActorCritic at `e929d4c7c` exactly matched v1 weights
and policy/value outputs on 12 fixtures; all 228 legacy native arrays matched
(`.runs/continuation-revision/{legacy-model,legacy-input}.json`). Compression
retained 24 passing Python cases, 27 native decision cases, and 24 exact
capture/prompt/context/schema-6 comparisons (`.runs/compress-snapshot-reuse/`).
The old 621-test collection predates the evaluation sync. These are focused
proofs, not the current full gate. The formerly interrupted 200-game parity soak is now complete: six tests
passed in `observation-gate-20261026-final/parity.log`; see Gate execution below.

### Gate-owned correctness commands

From the repository root, against the current extension. Native rebuild when
needed uses `uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
and the AGENTS.md atomic cp312 wheel-member install; no in-place overwrite of
a loaded extension. Debug checks remain required even with a release extension.

```sh
cargo fmt --manifest-path managym/Cargo.toml --check
cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path managym/Cargo.toml
uv run --extra dev --extra play --extra notebook --extra artifacts --extra reports --extra evaluation pytest tests/env/ tests/agent/ tests/model/ tests/test_package_boundaries.py tests/semantic/test_text_observation.py tests/semantic/test_decision_contract.py tests/verify/test_continuation_input.py tests/verify/test_checkpoint_scenarios.py tests/academy/test_decision_surface.py -q
uv run --extra dev pytest tests/verify/test_observation_parity.py -q
uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q
uv run --extra dev --extra artifacts pytest tests/training/test_recovery.py tests/training/test_game_error_recovery.py tests/training/test_learning_state.py -q
uv run --extra dev --extra artifacts --extra evaluation --extra reports pytest tests/eval/ tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py tests/training/test_checkpoint_queue.py -q
uv run python -m manabot.verify.text_observation --out .runs/text-observation-gate-v2
uv run --extra dev ruff check manabot/sim/match.py tests/eval/test_batched_eval.py
uv run --extra dev ruff format --check manabot/sim/match.py tests/eval/test_batched_eval.py
git diff --check
```

The parity-file command includes `test_observation_parity` (200 games) and all
buffer/vector checks. Current `.github/workflows/manabot-integration.yml` owns
its exact main/long/evaluation selections and collection; the historical count
is not a gate target. Changed-file lint and the affected correctness selections completed below;
the broader integration matrix retains its own CI result. Headless browser Perf proof belongs to CI's existing
`./scripts/verify-eval-performance-report`; it does not block this reconciliation.

### Gate-owned performance evidence

Existing `.runs/continuation-revision/` timings are historical diagnostics only.
They retain baseline `e929d4c7c`, corpus hashes, allocation/byte counters and
three alternating before/v1-refactor/v2 pairs at batch 1/64. Measurements were
highly variable under contention and predate final nested-parent ordering and
helper/snapshot reuse corrections. A lower later load reading is not acceptance.
The measured relation buffer adds 8,192 bytes per observation; native encode
allocation count was one. Neither fact establishes full-vector cost.

The historical passive/action-zero recipe has been superseded by the matched
five-case vector runner and native encoder receipts documented in
`docs/recent-events.md`. The measured baseline is a fresh export of `e929d4c7c`
with only the retained measurement overlay. Both native examples use the
`benchmark` feature; both measured extensions use `python,benchmark`, release,
CPython 3.12 and the same compiler. Counting is disabled during timing, though
the opt-in allocator wrapper remains compiled into both arms. Normal wheels
exclude that wrapper and the fixture APIs.

The paired command and completed results are recorded below. Unlike the earlier
recipe, it checks per-family inputs, Commands, legacy tensors and resulting
states across all arms, including a 33-action **output**. Source manifests,
executable/extension hashes, compiler/runtime identities and allocation/byte
attribution accompany every run. No historical receipt was rewritten.

Acceptance still requires identical seeded inputs and action sequences across
before/v1-refactor/v2 arms, both batch sizes, same release compiler/profile,
hardware, threads and buffer settings; allocations, bytes written, latency and
throughput; per-family paired samples and observed noise, including failed
attempts. Compare before→refactor separately from refactor→v2. Any material
regression needs repair and remeasurement before publication. No synthetic load
is permitted (TESTING.md); only bounded actual-workload measurement. No paid
execution or playing-strength experiment is selected.

No representation product decision is needed. Comparator/history/model choices
remain draft follow-ups in questions. Publication remains conditional on gate
and performance acceptance; merge is not authorized.

Check, 2026-10-26: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_batched_eval.py::test_served_checkpoint_preserves_input_version tests/eval/test_batched_eval.py::test_residency_catalog_batching_parity_and_eviction tests/eval/test_local_evaluation.py::test_cli_completes_without_training_cloud_or_callback -q` → 4 passed; focused Ruff → pass; full gate/parity soak/performance → deferred to gate.

### Gate execution — 2026-10-26

Fresh evidence owner: `.runs/observation-gate-20261026-final/`. Historical
`.runs/continuation-revision/` receipts were preserved. The baseline was freshly
exported from `e929d4c7c` into `before/`, without a worktree; its measurement-only
source overlay is retained in `baseline-instrumentation-final.patch`.

Gate repairs: the object-permutation verifier now remaps both continuation
object-reference columns with the object table. The observation-key fixture
expects fresh v2 relations; the incompatible checkpoint test now actually
changes v2 to v1. No model invariant or learning threshold was relaxed. The
continuation module's two Clippy findings were fixed without changing its
layout. The existing uncommitted serving/replay compatibility fix was preserved.

Measurement repair: `benchmark` is an opt-in native feature, absent from normal
wheels. It counts `alloc`, `alloc_zeroed` and `realloc` requests over complete
vector calls, separately from timing, and permits untimed exact fixture forks.
The existing allocation test/example share that counter so all-feature builds
have a single allocator. Native receipts add final encoded/native-input hashes;
vector receipts retain per-family inputs, Commands, continuation/legacy tensor
hashes, post-state hashes, allocation scopes and output/copy bytes. The real
Python failure-capture, synchronization and reward path is included. Each
positional action is checked against its semantic Command and all output rows
against the scalar native transition, including transition events. The first
instrumentation smoke caught an erroneous reread that lost those events; that
receipt is retained and the measurement code now uses the step's returned input.

The paired runner records source manifests including new files, binary hashes,
compiler/runtime identities, exact commands, host load and wall/CPU times. It
rejects mismatched inputs/Commands/legacy outputs and changes during collection.
Current/baseline smoke inputs, Commands, legacy tensors and resulting states
match in all four families. The 64-row refactor smoke also passes. These are
instrumentation checks under concurrent validation, not performance acceptance.

Checks so far (final outcomes below supersede this in-progress list):
- Affected env/agent/model/semantic/Academy correctness command above → initial
  339 passed, 1 skipped, 5 failures (`correctness.log`); the five stale verifier
  expectations were repaired and included in the remaining-suite run.
- `uv run --extra dev pytest tests/verify/test_observation_parity.py -q` →
  **6 passed in 53.94s**, including the complete 200-game native/Python soak.
- `uv run --no-sync python -m manabot.verify.text_observation --out .runs/observation-gate-20261026-final/text-proof`
  → **33 legal alternatives executed/replayed**, continuation collision and
  history separation retained (`text-proof.log`).
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings`
  → **pass** (`clippy-resolved.log`), after resolving the two continuation
  lints and allocator conflicts with existing measurement targets.
- Ruff check and format-check on all 22 changed/new Python files → **pass**.
- Exact current CI main/long/evaluation selections collected successfully;
  main **621**, long **27**; logs retain the evaluation per-file counts.
- The first instrumented build omitted the required `python` feature and failed
  with `unknown binding type`; the corrected command uses
  `--features python,benchmark`. Both release extensions are atomically installed
  from fresh wheels, never overwritten while mapped.

The debug long-game/branch-contract progress warnings were investigated, as
was the long evaluation run (bounded process sample: `eval-sample.txt`). Both
selections completed successfully below. No check timeout was overridden.
Timing collection began after both validation jobs finished.

Completed correctness outcomes:
- Combined required admission, monitoring, choice-separability, recovery,
  learning-state, full evaluation and Etude/head-to-head/checkpoint-queue
  selection, including the repaired observation/world/FiLM files → **308 passed,
  1 CUDA-only skip in 1965.01s** (`remaining.log`). This includes all five nodes
  that failed the initial affected-suite run; no learning threshold was changed.
  The exact selection is retained in `remaining-collection.log` (309 nodes).
- `cargo test --manifest-path managym/Cargo.toml` → **539 passed, 1 ignored**
  across 26 targets (`rust-tests.log`). The long debug branch-contract target
  passed in 291.77s; its 60-second warnings were progress warnings, not timeouts.
- The changed native Python stub also needed formatting. Ruff lint/format,
  `cargo fmt --check` and `git diff --check` now pass; exact static command arrays
  and results are in `static-checks.json`. Subsequent benchmark-only edits were
  linted/formatted separately before timing.
- `./scripts/verify-eval-performance-report` remains owned by capable CI under
  the accepted plan; it was not run or marked passed locally. No frontend or
  accepted prompt presentation changed in this gate.

The last instrumentation audit added a fifth full-vector case: an ordinary
priority pass returns and encodes 33 legal actions. The original large-input
case remains, distinguishing its offer/capture overhead from large-output
encoding. Batch-64 smoke passed all five cases (`output-coverage-smoke.json`).
The exact measured baseline overlay is `baseline-instrumentation-paired.patch`.

Fresh serial timing command:
`uv run --no-sync python scripts/bench_observation_pair.py --before .runs/observation-gate-20261026-final/before --out .runs/observation-gate-20261026-paired --steps 128`.
It started only after the correctness process exited. Initial measured host load
was 108.7/114.2/82.1 on 16 logical CPUs, despite only one benchmark child running.
This is an explicit environment limitation for reliable performance acceptance;
no other processes were altered and no artificial load was induced.

### Final paired evidence and remaining acceptance

All **27 serial measurement commands passed**: three before/refactor/current
pairs, each with encode-only and full-vector batches 1/64. The runner confirmed
identical native inputs, Commands, legacy input/output tensors and resulting
states in every vector family, and unchanged source/binary identities over the
whole collection. V2 continuation and tensor digests are retained and checked
across repeats. Receipts: `.runs/observation-gate-20261026-paired/`; the full
comparison is `comparison.json`, commands/load/CPU/wall records are
`commands.json`, and `identity-verified.json` records the final identity check.

These are **diagnostic timings, not a performance pass**. One-minute host load
ranged **84.40–112.86 on 16 logical CPUs** (Apple M4 Max / Mac16,5). The serial
children consumed **95.33 CPU-seconds over 268.50 wall-seconds**. Same-source
variation is large: batch-64 large-input refactor medians were 276.46, 1294.35
and 267.20 µs/env; its p90 samples reached 2944.68 µs/env. This is consistent
with severe contention; it does not prove every slowdown is noise.

The batch-64 large-input before→current paired ratios were **1.06, 1.64, 2.09**;
the large-output ratios were **1.06, 1.73, 1.09**. These remain unresolved
regression signals. A lower-tail diagnostic for large input was roughly
165–167 µs/env before versus 171–172 current; that does not waive the requirement
or establish a statistically meaningful regression under this host condition.
The Python vector orchestration and native stepping implementation are unchanged
from the baseline; the measured changes are shared fact projection, encoder
refactoring and relation-buffer binding. No speculative optimization was made
from these unstable timings.

The v1-refactor arm uses current native facts but the old tensor layout. Thus
before→refactor includes shared native projection costs; refactor→current isolates
the additional relation-buffer path. It is not a claim that native fact
construction itself has zero cost. Timings use identical allocator wrappers
with counting disabled; allocation attribution runs separately.

Medians of the three run medians below are µs/environment. Paired ranges are
current/before (above 1 is slower); retain the individual pairs in the JSON
rather than treating this summary as a noise-adjusted acceptance statistic.

| Path / batch / case | Before | V1 refactor | V2 current | Paired current/before range |
| --- | ---: | ---: | ---: | ---: |
| vector/1/priority | 39.69 | 40.08 | 40.12 | 1.01–1.05 |
| encode/1/priority | 7.28 | 2.10 | 1.78 | 0.19–0.99 |
| vector/1/target | 39.27 | 40.81 | 39.90 | 0.96–1.03 |
| encode/1/target | 9.62 | 1.67 | 2.37 | 0.24–0.54 |
| vector/1/additional_cost | 46.02 | 38.94 | 39.67 | 0.83–1.02 |
| encode/1/additional_cost | 4.61 | 2.18 | 2.49 | 0.54–1.11 |
| vector/1/large_legal | 200.21 | 201.04 | 200.04 | 0.98–1.03 |
| encode/1/large_legal | 24.03 | 5.09 | 7.76 | 0.29–1.31 |
| vector/1/large_legal_output | 108.00 | 122.85 | 117.04 | 0.97–1.13 |
| vector/64/priority | 18.65 | 19.01 | 18.45 | 0.91–1.06 |
| encode/64/priority | 8.77 | 4.12 | 4.91 | 0.33–0.97 |
| vector/64/target | 17.33 | 17.53 | 18.59 | 0.80–1.30 |
| encode/64/target | 6.21 | 4.48 | 4.58 | 0.41–2.60 |
| vector/64/additional_cost | 17.05 | 18.63 | 17.45 | 0.81–1.17 |
| encode/64/additional_cost | 9.96 | 5.00 | 4.36 | 0.43–2.05 |
| vector/64/large_legal | 169.05 | 276.46 | 345.19 | 1.06–2.09 |
| encode/64/large_legal | 24.00 | 7.95 | 9.22 | 0.34–0.40 |
| vector/64/large_legal_output | 91.57 | 113.80 | 127.00 | 1.06–1.73 |

Rust allocation and logical buffer-byte evidence is deterministic across the pairs:
- Encode-only: priority/target/additional-cost allocations drop **3→1**
  (276→152 requested bytes); large-legal drops **5→1** (1156→584 bytes).
  V2 adds no native encode allocations over v1-refactor.
- Full-vector current versus refactor adds **one 80-byte Rust allocation per
  batch**, at both batch sizes. Full vector includes fact extraction and Python
  failure capture, not just encode-only counters.
- At batch 64, large-input Rust requests are **135459→135267→135268** and
  requested bytes **17934496→17965472→17965552**. Large-output requests are
  **76195→75939→75940**, bytes **8792736→8756128→8756208**. These counters do
  not explain a twofold timing increase, but cannot certify latency by themselves.
- Dense native output is **38130→46322 bytes/environment**: the accepted v2
  buffer adds **8192 bytes**, or **524288 per batch of 64**. Tensor-sync destination
  bytes remain **518/environment**. These are logical buffer/copy bytes, not
  physical bus traffic or a count of repeated stores.
- Torch CPU allocation events/bytes and Python traced peak/live bytes are
  retained per family. Python traced peaks are explicitly not cumulative
  allocation traffic; native allocations are counted by the Rust allocator.

Identity: current HEAD `ffec2957d293348dda1a646b9fe668a6c42e8e67` plus the
retained uncommitted source manifest/diff; baseline `e929d4c7c` plus
`baseline-instrumentation-paired.patch`. Both used release Rust **1.98.1**,
LLVM **22.1.8**, aarch64-apple-darwin, CPython 3.12, one intra-op and one inter-op Torch thread,
Apple clang **17.0.0**, SDK **26.2**, extension deployment target **11.0**.
`identity.json` and `toolchain-supplement.json` retain full identities. SHA-256
of sorted compact source-manifest JSON:
- before: `ec21a9538357f46cb341a7276f4c009f87dab4bea75d6b0422009f66bba4352b`.
- current: `4406638f8f821703605b1961b266bcaca76462764add6a7b59008722b948b57e`.

**Remaining blocker:** reliable no-regression acceptance, including resolution
of the batch-64 large-input/output signals, requires repeating the same fixed
protocol on a sufficiently quiet equivalent host and profiling/fixing any
persistent regression. No threshold, corpus or checkpoint semantics may be
relaxed to obtain a pass. This environment cannot provide that evidence while
its observed load is 84–113 on 16 CPUs; no authority was given to stop other
work or acquire resources. Correctness, the interrupted soak and benchmark
instrumentation are completed here, not deferred back to a future gate.
Publication remains blocked; no merge, publication, new Graph or worktree was
performed. The existing serving fix and other contributions remain uncommitted.

After measurement, the normal release extension was rebuilt with only the
`python` feature and atomically installed. Its benchmark fixture APIs are absent.
The measured instrumented binary remains retained in `wheels-identity/`; it is
not relabeled as the normal binary. Normal extension SHA-256:
`67fe50ec871040491b5d18f62dc12d4a698f9f9b2da0b7242a088c319137ebfa` (`normal-extension.json`).

Final executable source manifests still equal the measured manifest.
`uv run --no-sync pytest tests/verify/test_continuation_input.py` plus the five
explicit buffer/vector/parity fixture nodes → **10 passed in 3.05s** on that
normal extension (`normal-wheel-check.log`). The earlier
`normal-extension-check.log` ran before the wheel installation completed and is
not counted as a normal-build proof. The complete 200-game soak remains the
separate successful `parity.log` receipt. Final `git diff --check` → pass.

### Runpod performance acceptance — 2026-10-26

Jack Heart approved shipping and up to $2 for Runpod, then explicitly directed shipping if results look normal-ish. This supersedes exact no-regression as the delivery blocker; retain small measured overheads honestly. A 32-vCPU / 64-GB CPU pod at $0.96/hour, bounded to one hour and $1.50, ran the unchanged 27-command protocol on AMD EPYC 7763 / Linux x86_64 with Rust 1.98.1, Python 3.12, Torch 2.10 CPU. Both snapshots were rebuilt on the same machine. Only benchmark runner extension-path discovery changed for Linux. Sources were transferred from the current checkout and the already instrumented baseline, with all source/binary hashes retained. The remote git commit identifies a frozen source snapshot, not the published branch head; original checkout HEAD is retained in `.runs/runpod-performance/head.txt`.

All 27 commands passed, with matching inputs, Commands, legacy outputs and post-states; final source/binary identity verification passed. At batch 64, large-input ratios were 1.008/1.018/1.008 and large-output 0.994/0.993/0.992. Other vector ratios ranged 0.978–1.051; current median large-input 173.19 versus 171.77 microseconds/environment, large-output 84.49 versus 85.02. Large-action encode-only improved approximately 20%; small-vector cases retain up to approximately 5% measured overhead. This is normal under the user's revised acceptance, not proof of zero overhead. Host has 256 logical CPUs, the container has 32 assigned CPUs and no quota. It is not an Apple-host equivalence claim, model-throughput benchmark, or playing-strength experiment.

Receipts: `.runs/runpod-performance/evidence/paired/{comparison,commands,identity,identity-verified}.json`; complete build/run log and hardware records in its parent. Rental `cpu32-e23510bb389b`, provider pod `t0sy0vlgm74ozu`; termination requested immediately after downloading the evidence. The full prior correctness gate remains valid; only benchmark-path portability and documentation changed afterward. Continue through final gate and waited landing.

### Final shipping gate — 2026-10-26

**Pass under the revised shipping acceptance.** No production code changed in
this gate. The branch comparison is the Task's recorded base `a0dead2b5` through
the current working tree; prior gate receipts are reused rather than repeating
full suites. The Task API still reports the historical stopped sync failure;
the recorded merge and completed gate evidence above establish the subsequent
repair. No Graph was launched or historical effects replayed.

- `uv run --no-sync python -` (runner `source_identity` against both retained manifests) → all 425 current source/content hashes match Runpod; only `scripts/bench_observation_pair.py` differs from the prior local gate manifest.
- `uv run --no-sync python -` (runner `compare` using the receipt's recorded Linux extension key) → downloaded comparison reproduces exactly; all 27 command return codes are zero and final source/binary verification is true.
- `uv run --extra dev ruff check scripts/bench_observation_pair.py scripts/bench_observation_support.py scripts/bench_breakdown.py` → pass.
- `uv run --extra dev ruff format --check scripts/bench_observation_pair.py scripts/bench_observation_support.py scripts/bench_breakdown.py` → all three already formatted.
- `uv run --no-sync python scripts/bench_observation_pair.py --help` → pass; separate `EXT_SUFFIX` path assertion resolves the installed macOS cp312 extension. Linux execution is covered by the completed remote protocol.
- `git diff --check` → pass.
- Prior `cargo test`, affected Python selections, complete parity soak, text proof, Clippy and static receipts → reused on unchanged production source; 539 Rust tests, 308 remaining Python tests, 200 games and 33 legal alternatives as detailed above.
- `./scripts/verify-eval-performance-report` → remains deferred to capable CI, confirmed in `.github/workflows/manabot-integration.yml`; CUDA-only skip remains unverified locally.

Durable delivery context is in `docs/recent-events.md`: measured overheads,
source manifest digest, v1 checkpoint semantics, matched-input comparison
limits, unavailable mana activation source, buffer cost and CI deferrals.
The earlier `scratch/pr-review.html` is a dated source walkthrough, not the
current gate verdict; its pre-gate claims are superseded here. Publication and
waited landing remain with the parent, as does rental cleanup. No paid execution,
publication or merge occurred in this gate.
