# LLM harness and shared observation design

Status: harness implemented; Qwen3.5-9B BF16 integration accepted on 2026-10-24.
Qwen runtime delta implemented and focused verification passed.
Gate/demo review and measured L4 capacity proof remain. Later direction below
supersedes the historical model/context blocker.
Placement: MTG-185, standalone Task; no Wave is associated.
This checkout owns the harness design. Representation sections describe the
shared draft boundary; MTG-184 owns that implementation and design.
Copied from the llm-finetune design conversation on 2026-10-24.
Jack Heart's accepted intent is a small pretrained LLM that fits on
one L4, trained only enough to connect to the action space, then measured for
relatively raw playing ability. The output protocol and recovery policy below
were accepted by Jack Heart on 2026-10-24; other mechanisms remain proposals
unless explicitly accepted below.

The original conversation excerpts remain in
`.runs/llm/context-before-compress/llm-finetune.md`; the decisions below are the
current design record.

## What to build

A deterministic, viewer-safe English rendering of one native observation and
its complete legal decision, usable by an untouched LLM through its existing
tokenizer and bound back to the engine's Commands.

This is an additive series. The keystone is the representation and binding;
a parallel model harness consumes that contract. Fine-tuning and playing-strength
evaluation follow, with their own designs.

## Representation demo — MTG-184 proposal

Render an actual native decision, inspect its text, return an action reference,
and execute the corresponding revision-bound Command. Repeat at target, combat,
payment, and selection decisions, including two copies of the same card. Show
that every legal alternative remains selectable. A text response stub suffices
for the representation demo; the parallel harness substitutes a real LLM.

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
the same ordinary bound Command for either origin. First-action fallback was
considered; the accepted policy uses seeded uniform sampling.

Only protocol failures receive this recovery. Representation overflow, model
runtime errors, exhausted deadlines, stale bindings and engine rejection stop
the attempt with evidence; they are not fixed by choosing a different action.
Never retry an uncertain execution or replay a committed cost.

Record raw responses, validation errors, attempts, tokens, latency and selection
origin (`initial`, `retry`, `fallback`) at each decision. Report first-response
validity, validity after repair, fallback decisions and games with any fallback.
Full-cohort playing scores describe the LLM plus this recovery policy, never
pure model play. Do not discard fallback games to improve the reported score.

## Representation — MTG-184 draft

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
4. Ordered stack entries, controllers, modes, abilities and visible targets.
5. Card definitions referenced above or by legal choices: name, characteristics,
   native rules text, costs and printed keywords, deduplicated by bound definition.
6. Every current legal action, with its distinct reference, verb, visible focus,
   declaration, mode and ability selectors. Never infer legality from prose.

Use decision-local aliases for objects and definitions; distinct copies remain
distinct. Keep native IDs, revision and fingerprint in the binding envelope,
outside model text. Resolve aliases only within that envelope. Preserve ordered
relations; normalize only order that carries no game meaning. No importance
ranking, strategic annotations, recommendations or human-only shortcuts.

The proposed v1 is current-state-only; Jack Heart has not accepted that boundary.
Composite `events` and `recent_events`
are empty on the inspected native semantic-observation path, and its JSON omits
the separate `policy_history`. Do not fabricate a history from audit logs or
call this full observation-history parity with existing policies. History is
a later, separately versioned input treatment.

Audit every source field as rendered, binding-only or redundant with a named
rendered field. Unresolved references, unsupported fields/decisions and token
overflow produce typed failures, never clipped states or smaller action menus.
An absent upstream fact remains unavailable. Resolve ambiguous cost, object or
pending-decision meaning at the shared observation authority if necessary;
do not reconstruct private state or add an LLM-only rules solver.

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
- `manabot/llm/` now owns the standalone runtime and protocol evidence. There is
  no fine-tuning implementation. Compound decoding remains a separate interface.

Labels alone are insufficient. Pair offers with action descriptors from the
same composite observation using their native order contract; validate count,
schema and binding, and render visible focus/selector distinctions. Maintain
every microdecision in v1, including committed costs and later choices.

## Representation data structures and harness boundary

Proposed reusable owner: `manabot/semantic/text_observation.py`.

- `TextDecision`: representation version, prompt text, typed legal references,
  source Observation identity and immutable Command bindings.
- `TextAction`: reference, description and one exact bound Command.
- `TextEncodingError`: unsupported schema/field/reference or unavailable meaning.
- `render_decision(observation: Observation) -> TextDecision`: pure projection;
  no engine mutation, model, tokenizer or hidden-state access. A necessary public
  definition resolver must be explicit and content-bound, not a global catalog.
- `resolve_action(decision: TextDecision, response: str) -> Command`: accept
  exactly one listed reference after trimming whitespace; unknown IDs,
  commentary and malformed answers fail. Submission still goes through native
  stale/legality checks. The harness handles bounded repair separately.

Implemented harness types live in `manabot/llm/schema.py`: `DecisionInput`
contains text, legal-reference tuples, representation ID and opaque string-valued
source identity. `SelectionResult` retains that input, configuration, runtime
identity, ordered attempts, selection origin, fallback evidence or failure.
`run_decision(decision, config)` supervises inference. None of these types binds
or submits Commands; the renderer integration remains future work.

Prompt and artifact versions bind native world/content/schema and renderer
identity. The harness separately binds model/tokenizer revisions, chat template,
precision, decoding settings and actual token IDs. This adds a derived format,
not a world migration or a reinterpretation of historical checkpoints.

## Delete — do not maintain

Completed: the optional Transformers 4.57.6 pin and root-only context check
were replaced by Transformers 5.3.0 and text-config admission. The worker's
redundant latest-result copy is removed: selection owns inference evidence,
and the parent retains progress across unexpected worker exit. No deletion
targets remain for this additive harness.

Preserve generic causal loading, CPU fixtures and protocol evidence on one
runtime path. Do not introduce a separate Qwen runner or registry, restore retired
supervised-policy training, or delete tensor policies, compound decoding, shared
projections or frozen evidence. Handwritten harness examples remain temporary;
MTG-184's generated inputs will replace them without a second renderer here.

## Accepted harness direction — 2026-10-24

Jack Heart selected the first demo as one decision with a fully inspectable
record; full games come later. After reviewing `scratch/llm-harness-review.html`,
he said “ok lg2m launch a pursuit,” authorizing the harness pursue Graph. That
HTML remains the approved historical design artifact, not current status.
The opening hold on implementation and later model/context blocker are superseded.

After reviewing model/memory options, Jack Heart said “sure, whatever works.
that all sounds fine.” Qwen3.5-9B at BF16 is the accepted first playing-model
integration, with pinned model/tokenizer commits and thinking disabled for the
exact-reference protocol. His context direction was “smaller probably better”
and “Get it working, then optimize”: a provisional explicit budget is sufficient
for implementation and code review. Complete inputs and repairs must fit without
truncation; actual L4 fit remains an acceptance dependency.

Reversible engineering settings are Transformers 5.3.0, 4,096 total context
tokens, 32 new tokens per response, and 120 seconds for the entire decision.
`configs/llm/qwen3.5-9b.json` pins both revisions to
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` and `enable_thinking=false`.
These settings are not a measured corpus limit or a capacity claim.

Jack Heart wants to “sample a lot of different models once we have this more
up and running.” Preserve explicit comparable configuration and evidence;
“sample” does not select stochastic decoding. Broad sweeps, reasoning support
and quantization are later increments, without a prebuilt registry or scheduler.
The accepted continuation includes the remaining authored pursue work and
publication. No paid allocation, fine-tuning, full-game integration or scientific
campaign is selected; the earlier CPU proof must not be recreated as new evidence.

Jack Heart's Task steer `6d8447cf-776e-434b-a4c9-cee500f135d6` confirms this
direction on 2026-10-24. It supersedes the Task description's older statement
that model choice and runtime remain draft. Publication and Task completion are
distinct: the latter still requires measured L4 capacity.

## Implementation and review boundary

`manabot/llm/{schema,selection,transformers_runtime,runner}.py` and
`uv run --extra llm python -m manabot.llm` implement one decision. See
`docs/llm-harness.md` for inputs, pinned CPU/CUDA configurations and result fields.

- The caller supplies complete viewer-safe text/references and provenance.
  The harness has no Observation or Command dependency. MTG-184 retains rendering,
  reference binding, native stale/legality checks and replay acceptance.
- Optional Transformers loads pinned Hub snapshots without remote Python code,
  adapters or training. The same auto-loader extracts the original Qwen text
  weights from composite checkpoints without constructing the vision encoder.
  Text-config admission checks context and decoder-only architecture; both root
  and text configs reject quantization. Runtime identity records the loaded class.
- Both prompts use the native chat template and unchanged tokenizer. The optional
  thinking flag applies to initial and repair prompts; true is rejected. Greedy
  generation uses explicit settings retained in full, with exact token IDs.
- Selection publishes immutable snapshots from a local attempt list. Pending
  latency/memory is unavailable, not zero. The small frozen schema base stays
  local because the shared infra base eagerly imports Torch.
- The spawned worker owns runtime initialization and forwards selection evidence.
  The parent owns the wall deadline, including imports/downloads/loading and
  both attempts. It waits on the pipe; the sole sender's exit exposes EOF after
  queued evidence, without liveness polling. Parent failures clear selection
  while preserving the last received snapshot.
- CUDA memory is peak allocated tensor bytes since loading, not total device
  usage or reserved memory. CPU memory fields are null. The CLI reserves a new
  output file and writes the final record after supervision returns; this is
  not durable resume or protection against losing the parent process.

One selection record contains up to two response attempts. EOS-terminated `A1`
succeeds initially; commentary receives the unchanged protocol-only repair;
a second invalid answer yields marked seeded-uniform fallback. An initial
prompt may fit while repair overflows, or a legal-looking response may arrive
late: both stop without selection or fallback. Address the cause and start a
fresh invocation/output path. A future game caller must not retry uncertain
Command execution or erase committed costs or revealed information.

Upstream `1e601527094601e5f152d866fbdf9f924c33e954` moved game execution to
`manabot.sim`, evaluation to `manabot.eval`, and adopted Etude SDK 0.2.2. Harness
CI preserves those paths and includes `tests/llm/` with the optional `llm` extra.
Future game integration belongs at the sim/Command boundary. Historical inherited
context is referenced in `scratch/questions.md`.

## Retained evidence and remaining acceptance

Concept review, 2026-10-24: no design change or unresolved product choice within
the accepted one-decision harness. Jack Heart's selected experience remains
explicit input/configuration → one invocation → one inspectable selection record.
The usage contract in `docs/llm-harness.md` already makes that interaction concrete;
no proposed replacement instructions are needed.

The useful concepts are the supplied decision, inference settings and result;
response attempts belong inside that result. `run_decision` owns execution,
`select` owns protocol recovery, and the CLI owns the output file. A repair is
neither a new game decision nor a resumable job. Keep the process/pipe machinery
private and retain one generic runtime path; no service, registry or additional
lifecycle is justified by this experience.

The decisive counterexamples remain a repair that exceeds context and a legal
answer received after the deadline: neither may become a fallback or success.
Recovery preserves received evidence, corrects the cause and uses a fresh output
path. Parent-process loss can leave an empty reserved file; durable recovery is
outside this one-shot contract and must not be implied by worker supervision.
Future Command execution still needs its own stale/replay checks and must never
retry an uncertain execution. These boundaries preserve the accepted behavior
without turning selection into game orchestration.

The next proof remains the affected gate and actual pinned-model L4 measurements
described below. Missing GPU authorization/capacity evidence is an execution and
acceptance dependency, not a reason to reopen the accepted model or protocol.
The unresolved history choice belongs to MTG-184 and does not block this
independently supplied-text harness. This review changes no implementation and
does not establish L4 fit or Task completion.

- `.runs/llm/cpu-smoke/decision-cleanup.json`: pinned SmolLM2-135M-Instruct revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, produced under Transformers 4.57.6.
  Answer `A0\nA1`, repaired to `A0`; origin `retry`, 88/124 prompt tokens,
  6/3 output tokens, 18.94 seconds including loading, null GPU memory. The first
  cold run remains beside it as `decision.json`. This is plumbing evidence,
  not the selected playing model, playing-strength evidence or L4 fit.
- `.runs/llm/qwen-tokenizer/receipt.json`: pinned metadata/tokenizer-only proof,
  handcrafted initial/repair messages and exact IDs, 32/65 prompt tokens,
  thinking disabled. These are not `select`'s fully assembled prompts, which
  append a reference list and output instruction; those counts do not establish
  harness context usage. No 9B weights were downloaded.
  Offline random-weight GPT2 and composite Qwen
  tests verify template behavior, exact text-weight extraction, hybrid-layer
  cached generation, overflow and nested context limits. Random weights establish
  runtime plumbing only.
- Controlled protocol/supervisor tests cover initial/repaired validity, fallback,
  incomplete output, initial/repair overflow, late results, runtime failure,
  deadline termination and queued evidence on worker exit. Dependency isolation,
  lint/format, build and focused Qwen tests passed before this compression.

Gate owns full affected suites and CI collection. Demo owns inspection of the
retained one-decision evidence and measured largest-input/repair latency and
memory on an L4 using the actual harness prompts and pinned pretrained Qwen
weights. No retained receipt yet shows pretrained inference under Transformers
5.3.0; the earlier CPU proof and random-weight tests cover different boundaries.
Capacity requires an authorized execution environment and
complete inputs covering the provisional budget; neither CPU nor tokenizer
proof completes that Task acceptance. `tests/remote/` is absent here, so its
repository-prescribed isolated dependency check is unavailable; the isolated
harness tests provide the applicable dependency check.

MTG-184 separately owns the renderer and proposed acceptance:
`uv run --extra dev pytest tests/semantic/test_text_observation.py tests/semantic/test_decision_contract.py tests/test_package_boundaries.py -q`.
Require deterministic complete viewer-safe rendering, distinct duplicate copies
and microdecisions, unsupported/overflow failures, and native replay/stale
rejection. Current-state-only versus visible history remains an unaccepted
representation choice; native events were empty and `policy_history` absent in
source/API inspection on 2026-10-24 (schema 6, reset seed 23, Mountain/Gray Ogre/
Lightning Bolt, two legal offers). This establishes no renderer behavior.

## Follow-up designs

Measure prompt-token p50/p95/max by decision family, section contribution,
action count and overflow fraction across complete games and targeted edge cases.
Use the existing tokenizer and freeze an admitted context only from that evidence.
No playing-strength gate is selected for representation itself.

Jack Heart requested noting “allowing some reasoning output” for later. Proposed:
retain reasoning separately from a strictly delimited final legal reference,
with native template/channel handling, explicit reasoning-token/deadline budgets
and repair retention. Settle those choices before implementation; never extract
an arbitrary reference from prose. Compare reasoning and direct-answer modes as
declared treatments. Reasoning-enabled benchmark rankings do not predict current
greedy action-only performance.

Untouched play precedes deciding whether interface fitting is necessary. If
needed, distinguish grounding an explicitly requested action from learning which
action is strategically good. Random action targets teach randomness; expert
choices teach strategy. Training data, objectives, stopping rules and allocation
need separate designs. Grammar constraints must be declared and shared across
comparison arms. Later paired-game evaluation should swap seats against fixed
anchors, retain Commands/replays and separate interface failure from strength.
No search, rewards, strategy distillation or teacher rationales are selected for
the raw-performance comparison; pretraining card knowledge remains a confound.

Earlier model research compared Qwen3-4B, Qwen3.5-4B/9B, quantized Qwen3.8-27B
and GPT-OSS-20B. The Qwen3.5-9B selection supersedes those candidates, not their
capacity caveats: weight-size arithmetic excludes caches/buffers, and the cited
GPT-OSS MXFP4 recipe requires Hopper or later. Publisher benchmarks with differing
conditions do not prove L4 capacity or playing strength. Full uncommitted research,
prior blocker text and duplicate review prose were preserved before consolidation
at `.runs/llm/context-before-compress/llm-finetune.md`; original questions are
beside it. No historical experiment receipt was changed.

Check, 2026-10-24: `git diff --no-index --check /dev/null scratch/llm-finetune.md` → no whitespace errors (exit 1 denotes the new file); documentation-only reconciliation reuses the preceding `uv run --locked --extra dev --extra llm pytest tests/llm/test_llm_runner.py tests/llm/test_llm_selection.py -q` result (34 passed) and isolated 2-test runtime proof. Gate owns full affected suites; demo owns L4 capacity proof.
