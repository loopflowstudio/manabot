# LLM observation language

Status: representation demo available, 2026-10-24. Latest omission cleanup
passes 13 focused tests and all 33 native execute/replay alternatives.
Broader delivery verification remains as recorded below; nothing published or merged.
MTG-184 owns representation; MTG-185 owns the parallel model harness. No Wave
is selected. Copied from the llm-finetune design conversation on 2026-10-24.

Jack Heart's accepted intent is a small pretrained LLM that fits on one L4,
trained only enough to connect to the action space, then measured for relatively
raw playing ability. On 2026-10-24 he accepted exact action-reference output,
one repair attempt and seeded uniform legal fallback; directed matching the
non-LLM input without writable memory for the untrained version; and requested
performance and model-size comparisons. Mechanisms below remain proposals.

After concept review and the comparison additions, Jack Heart requested “launch
pursue” on 2026-10-24, superseding the opening hold on execution and authorizing
representation implementation and review with native fixtures. This does not
select a model, authorize paid experiments
or transfer harness ownership. Experimental choices stay with their follow-ups.

## What to build

A deterministic, viewer-safe English rendering of one native observation and
its complete legal decision, usable by an untouched LLM through its existing
tokenizer and bound back to the engine's Commands.

### Implementation finding — 2026-10-24

`scratch/observation_collision.py` disproves `Observation` as the sole renderer
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

**Shared-source revision selected for implementation below:** capture and hash
continuation source, targets with clauses, chosen mode, reserved sacrifice and
outstanding payment alongside the legal decision. Reuse `decision_context` as
the facts' owner; do not reconstruct them through an LLM-only side channel.
Its `recent_facts` are a separate history treatment, not implicitly part of
current continuation. Bind public definitions to world/content identity: the
current composite supplies neither, and the manifest has names, not full text.

Capture v1 now supplies these facts without changing existing schema-6
consumers, worlds or evidence. Optional non-LLM history matching remains a
separate comparison choice.

Implementation decision after Jack Heart requested getting this ready to demo:
use a separately versioned shared native decision capture that binds the existing
Observation, continuation, public definitions and optional bounded policy history
atomically. This is an engineering choice under that request, not a claim that
Jack Heart selected the API spelling. Capture under the engine lock, hash all
included facts and bind world/content identity. Reuse existing fact owners; the
capture is an aggregate input, not a second rules authority. The renderer consumes
it end to end. Preserve the existing schema-6 contract and frozen evidence.
Use the existing optional public history v1 (up to 32 records) for the demo,
declaring that treatment explicitly; selecting a trained comparison remains
separate from this local functional proof. No writable model memory is added.
Changing the embedded Observation requires `SEMANTIC_DECISION_VERSION`;
changing capture inclusion/order/digest semantics requires
`DECISION_CAPTURE_VERSION`. Capture v1 hashes every included fact; schema 6
remains byte-compatible. No existing consumer migration is required. Existing Commands check revision and
offer legality, not a complete input digest. The collision proves missing input
facts, not a failure of revision-staleness rejection. A text binding must retain
its originating game/branch as well as its captured decision; revisions alone
do not identify a game across clones.

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

Upstream reconciliation, 2026-10-24: `1e6015270` moved paired evaluation to
`manabot/eval/match.py` and game execution/registration to `manabot/sim/`.
The playing-evaluation follow-up can reuse those schedules and evidence owners,
but needs an explicit integration: the current runner receives an integer action
from `act(env, obs)` and ordinarily calls `env.step(action)`. It does not yet
accept the proposed text binding and response-attempt evidence. A future adapter
must preserve the captured Command, selection origin and failure policy; existing
paired scheduling alone is not an end-to-end LLM evaluation. `912771b94` removed
supervised policy training, so later interface fitting also remains a new design,
not an available training path.

Labels alone are insufficient. Pair offers with action descriptors from the
same composite observation using their native order contract; validate count,
schema and binding, and render visible focus/selector distinctions. Maintain
every microdecision in v1, including committed costs and later choices.

## Data structures and key functions

Implemented owner: `manabot/semantic/text_observation.py`; strict shared Python
input models live in `manabot/managym/decision_capture.py`.

- `TextDecision`: representation version, prompt text, typed legal references,
  source Observation identity and immutable Command bindings.
- `TextAction`: reference, description and one exact bound Command.
- `TextEncodingError`: unsupported schema/field/reference or unavailable meaning.
- `render_decision(capture_json: str) -> TextDecision`: pure projection of
  validated, digest-checked capture v1. Bound definitions are inside the capture.
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
with its own capture-v1 compatibility/versioning boundary. Historical worlds and checkpoints retain their existing
meaning; the renderer version cannot stand in for a shared schema migration.

## Delete — do not maintain

Completed removal of unrelated inherited notes; no deletion targets remain:

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

No existing production path is replaced. Do not delete tensor policies, compound
decoding, shared projection utilities or frozen experiment evidence. Any temporary
hand-written harness examples are superseded by generated fixtures at integration;
they must not become a second observation renderer.

## Internal slices

1. **Complete:** atomic capture v1, bound public definitions, continuation and
   optional native public history v1; schema-6 consumers remain unchanged.
2. **Complete:** strict pure renderer, origin-bound submission, field audit,
   native fixture/demo and focused verification. Retained collision still has
   equal schema-6 observations, now distinct capture digests/continuation/text.
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
| `events`, `recent_events`, continuation `recent_facts` | Must be empty in v1. Legacy prose history deliberately excluded; optional policy history is the sole declared suffix. |
| `game_over`, `won`; turn number/phase/step/active/agent | Rendered explicitly; player IDs resolve to You/Opponent. |
| Player life, zone counts, graveyard Lessons, mana, restricted Lesson mana, combat expiry total, known-hand minima, initial/remaining sideboard counts | Rendered with units and definition aliases, including zeros/empties. `is_agent`/`is_active` rendered; `id`/`player_index` bind roles. |
| Cards: name, text, zone, owner, power/toughness, token/legendary/Ally/Lesson, ward/kicker values, all card-type and keyword flags, mana cost/value/hybrid | Rendered. False type flags stated collectively; absent keywords mean false. IDs bind aliases, registry keys bind D references. Effective battlefield state remains separate from printed card facts. |
| Outside candidates: candidate/owner/definition/name/text | O aliases and rendered owner/definition/rules; never fabricated in-game objects. |
| Permanents: controller/tapped/damage/sickness/counters/effective P/T, animation, exile link, exile-if-dies, unblockability, temporary subtypes/effective keywords | Every field rendered; IDs and registry keys resolve to the same C/D aliases as backing cards. |
| Stack: ordered IDs/kind/controller/definition/source permanent/ability/mode/targets | S aliases, native top-to-bottom order; all selectors and ordered target relations rendered. Missing source stays unavailable. |
| Continuation: actor/stage/instruction/source, targets with clauses, sacrifice reservation, mana cost, mode | All facts rendered; entity/display/incarnation IDs only resolve exact aliases. Expired objects do not acquire a returned object's alias. |
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
The original `scratch/observation_collision.py` also now checks distinct captures
and text while retaining its original schema-6 counterexample.

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

Local architecture measurement, 2026-10-24: a fresh instance of the checked-in
`baseline-v3` configuration has **193,794 parameters**, **775,176 FP32 parameter
bytes** and **37,410 buffer bytes**. The resolved configuration and architecture
receipt are in `.runs/llm-observation-baseline-size.json`. This counts the current
architecture without loading trained weights; it is not playing evidence or
the selected trained comparator. The selected LLM's exact count and measured
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

## Parent gate and demo review

Parent evidence, 2026-10-24: the combined representation, decision-contract,
package-boundary, current-world admission and monitoring regression run passed
**89 tests** in 640.44 seconds. Rust formatting and Clippy passed. Full debug
Rust tests subsequently completed successfully (debug `cargo test`, exit 0,
including all 316 rules tests).

The full learnability diagnostic retained an **error**, not a pass, at
`.runs/learnability/mtg-184-gate/report.md`: all three seeds' capacity fit and
held-out checks passed, then reward-driven PPO collection exceeded its 300-second
deadline. The total diagnostic took 402.5 seconds wall time and 71.5 seconds CPU;
concurrent local test/training processes were observed. Contention is a possible
cause, not a verified diagnosis. Do not widen frozen diagnostic budgets or
substitute the `--pr` subset. A successful full diagnostic remains a delivery
verification dependency; it does not prevent inspecting the retained native
representation demo. No model inference or playing-strength result is claimed.

No publication or landing is authorized in this implementation step. The focused
checks below are completed; parent gate owns the broader checks once:

- `uv run --extra dev pytest tests/semantic/test_text_observation.py tests/semantic/test_decision_contract.py tests/test_package_boundaries.py -q`
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings`
- `cargo test --manifest-path managym/Cargo.toml` (debug, including frozen conformance).
- Shared-input regression: `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q`.
- `uv run python -m manabot.verify.learnability --out .runs/learnability/mtg-184-gate` (full diagnostic, no `--pr`). Existing tensor shape/world are unchanged; retain this as the design's conservative shared-input regression boundary.

Review the retained prompts for understandable labels and field coverage. No
browser is required. MTG-185 owns real tokenizer/chat-template overflow checks,
model loading, bounded repair/fallback orchestration and runtime/GPU measurements;
these are not representation-demo blockers. No model or paid resource was used.

## Measure and evidence

### Interactive demo — 2026-10-24

Jack Heart invoked `demo` after accepting omission of absent optional fields.
The live native blocker fixture is now presented in this conversation: one
2/2 Gray Ogre can block either of two distinct attacking 2/2 Gray Ogres or
decline to block. The complete generated prompt is retained at
`.runs/text-observation/interactive/0-prompt.txt`, with exact capture at
`.runs/text-observation/interactive/0-input.json`. The displayed excerpt keeps
all three legal actions; the linked file is the actual full model input.

Jack Heart asked where the game was; the presentation was clarified as a local
native process with a chat/text interface, without a visible board. At his
request the complete prompt above was opened in the default text editor. He
then said "ok this is fine". This accepts the displayed full prompt and its
omission of absent optional metadata; no further text redesign is requested.

No human action reference was selected or executed in this review. The
interactive driver `.runs/text-observation/interactive/demo.py` was launched
with an origin-bound environment (tool session 22618); its continued liveness
must be checked before any later interaction. No LLM was loaded. Action
execution evidence remains the automated 33-alternative native replay proof,
not a human playthrough. Recommended next action: retain this accepted prompt
format for harness integration and resolve the recorded full learnability
verification dependency before delivery. Acceptance here does not establish
model performance or successful PPO verification.

Measure prompt tokens p50/p95/max by decision family, section contribution,
action count and overflow fraction. Later report initial/repaired response rates
and fallback usage separately from game outcomes, alongside latency and memory.
Set no playing-strength gate for the representation itself.

Check, 2026-10-24: focused `pytest tests/semantic/test_text_observation.py -q` → 13 passed; debug `cargo test --manifest-path managym/Cargo.toml decision_capture::tests -- --nocapture` → 1 passed; root `uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` → built and cp312 extension atomically replaced; demo and original collision proof → pass. exact main CI invocation with `--collect-only -q` → 614 collected (`.runs/text-observation/ci-collection.log`); Ruff check/format, `cargo fmt --manifest-path managym/Cargo.toml --check` and `git diff --check` → pass. Parent gate owns the remaining commands above.

`scratch/questions.md` indexes remaining experimental choices.
