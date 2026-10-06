# Engine-owned decisions, model-owned preferences

**ETU-107 · 2026-10-05 · Draft for Jack Heart's architecture review.**
Source inspection: `338cccd5fe0dd8d4038821a2f724e14ffd71866b`.
Jack Heart requested a consistent model-facing decision interface across Magic
choices. This contribution prepares that design only. The implementation slices,
new model inputs and validation budgets below are proposals awaiting review;
ETU-104's approval does not approve them. No training or benchmark ran here.

## Recommendation and demo

Evolve managym's `InteractionOffer`/`StructuredOfferSet` into the typed source of
model-facing choices. Keep `DecisionFrame` and `Command` as the canonical
execution boundary. Make simple offer scoring, object selection and conditional
compound construction instances of one **bound decision distribution**, with
explicit root, prefix, support and sampling semantics. Reuse the current compound
GRU and native lowering; give a new, explicitly selected decoder typed role and
runtime object inputs. Do not introduce a framework, a second rules interpreter,
or a list of all complete actions.

A flat checkpoint remains a one-factor distribution over its admitted atomic
offers. An existing compound checkpoint retains its exact offer/subset tape and
weights. A new typed compound checkpoint can distinguish two same-name creatures
by their visible state and role without learning object IDs. All three execute
through the same native Command authority. “One interface” does not mean these
policies induce the same distribution or take the same inputs.

The proposed headless demo loads a saved fixture policy through the ordinary
loader, samples a Bolt target, a modal choice, an attacker subset, blocker
assignments and waterbend payment, then teacher-forces each retained tape.
It prints factor probabilities, summed log probability, native Command receipts
and replay equality. A 65-attacker offer stays linear in candidates, with no
`2^65` action list. A second same-name target with different visible damage gets
a distinct candidate embedding. A payment with effects stays sequential.
A bounded train/export/reload/arena fixture then exercises the same interface.
No demonstration is a strength or human-challenger claim.

This serves the chapter's reproducible playable-checkpoint path. Full-game
improvement, human play, ETU-91's frozen campaign, broader content, belief/history
research and scientific decoder comparisons are outside this delivery.

## What exists, and the gaps that matter

| Owner/source | Observed behavior | Design consequence |
| --- | --- | --- |
| [Native offers](../managym/src/agent/structured_offer.rs) | `structured_search_offers()` emits one offer per legal action. `compound_offers()` replaces eligible casts and declarations with structured choices; all other actions stay available. | Extend this owner, not the prototype-only `structured_offers()` subset. Keep canonical frames action-aligned. |
| [Semantic authority](../managym/src/decision.rs), [Python mirror](../managym/decision.py) | Schema 6; `DecisionFrame` binds revision, actor, fingerprint and object candidates. Commands name offer IDs and expected revision. Python offer contents are still mappings. | Parse closed typed values at the native boundary. Offer IDs have local meaning, not a learned vocabulary. |
| Native candidate schema | `CandidateValue` currently has only subject/object/player; `ChoiceStep` only distinct/select. Dependency fields exist, but do not establish a general continuation service. | Modes, amounts, ability identity and role subjects need native semantic projection. Do not infer them from labels. |
| [Ragged transport](../manabot/sim/structured_policy.py) | `flatten_projection` rejects ordered selections, non-distinct choices, and `depends_on`; validates cardinalities and IDs. | Do not claim dependent-role support. Admit only native-proven static products initially. |
| [Agent](../manabot/model/agent.py) | Flat `_gather_informed_actions` uses action features plus bounded object focus. `Agent.compound` reuses these for ordinary offers; grouped declarations/payments instead use pooled visible context. | Flat focus is a compatibility input, not the new semantic contract. Expose object embeddings with identity joins for new candidates. |
| [CompoundDecoder](../manabot/model/compound.py) | Samples offer then canonical include/exclude factors with a GRU. Public UTF-8 labels, position/count and pooled context encode set candidates. `tokens` rescoring and `prefix` completion already exist. | Preserve its probability law; add typed features under a new decoder identity. Candidate labels alone cannot locate runtime subjects. |
| [Serving](../manabot/sim/compound.py) | `sample_compound` lowers on a fork; `CompoundPolicy` drains a Command deque via action indices. It checks revision and resets on engine changes. | Strengthen envelope checks; retain one execution owner. Never sample a pending suffix again. |
| [Local search](../manabot/sim/local_compound.py) | Retains root observation, offer projection, tape prefix and Commands. `project_compound` checks replay and maps disjoint decoder prefixes to next Commands, including exact zero mass. | Reuse its marginalization and selected branch runtime. Do not restart the policy at intermediate microsteps. |
| [ETU-104 accepted cut](../docs/plans/modular-architecture-recipes.md#11-delivery-cut-and-acceptance-after-review) | Python recipes build `TrainingRegime.agent: AgentSpec`; ordinary builders/loading and serialized `agent_hypers` remain owners. Broader module/spec hierarchy is unapproved. | Add only the decoder configuration needed by this slice. No registry, new trainer or backend. |

Paths above are repository-relative from `scratch/`. Evidence is source and
existing fixtures, not freshly executed measurements.

## Representative traces and scope

Letters below stand for IDs returned by the engine, never constants supplied by
the model. `r -> r'` means the actual recorded revisions; forced engine steps can
make them non-consecutive. Model construction is pure until native submission.

### Priority, cast and target

At revision r, actor 0 sees pass, land, activation and cast offers. For the
eligible non-kicked Lightning Bolt, native `structured_priority_offers` supplies
source ObjectRef and role 1, selecting exactly one creature/player. The checked-in
[Bolt fixture](../managym/tests/fixtures/structured_priority_bolt_offer.json)
contains Hero, Villain and Gray Ogre candidates.

Trace: select cast B; include/exclude target candidates in native order until the
one-target answer is complete; submit `{offer_id:B, answers:[role 1 -> target T]}`.
`compound_commands_json` lowers to cast at r, then target at r' if the engine
surfaces it. A sole target may already be auto-resolved. Do not submit it twice.
Priority/response resumes after construction. Pass has just its offer factor;
play-land is likewise a current atomic choice.

Kicker is excluded from this grouped cast path. Counterspell's spell target and
multi-target casts are not covered by the Bolt bridge: use the complete atomic
cast then the newly published target decision. Safe sequential means the same
selected decoder scores a fresh native offer, not a flat-model fallback.

### Ability, mode and amounts

`Action::ActivateAbility` already retains permanent and ability index, but its
search offer drops those into the internal binding and presents a generic label.
The new projection must expose the public source and a content-bound ability
reference. Activation remains one native decision; a subsequent payment, target
or resolution choice is another root unless native grouping explicitly covers it.

Current modal trace: cast/resolve Crossroads of Destiny, then the engine publishes
two `ChooseMode` actions. Mode 0 gains 3 life; mode 1 draws a card, as exercised in
[stage2_cards](../managym/tests/rules/stage2_cards.rs). Model scores the two typed
mode candidates at that modal root. Choosing draw executes resolution and reveals
a card to its owner; no cached choices extend across that draw. Modes here occur
at resolution, not in an invented universal cast wizard.

Proposed fields: mode reference = content program identity plus mode ordinal;
ability reference = definition/program identity plus ability ordinal; amount =
quantity plus unit and role, when native semantics actually supplies one. A
fixed cost such as waterbend's remaining generic requirement is a descriptor,
not a freely chosen number. Current `Action` has no general choose-X or damage
allocation constructor. Arbitrary X values, multi-mode combinations and general
numeric allocations are unsupported until Rules supplies exact support. Neither
Python nor the model synthesizes numeric ranges from card text.

### Attackers and blockers

Attack root: one `DeclareAttackers` offer, role 1, eligible object candidates
[A,B,C], min 0/max 3. Tape `[offer=0, include A, exclude B, include C]` denotes
exactly subset {A,C}. Native lowering supplies the per-creature Commands in
engine order, stopping when declaration ends. No opponent block choices are
predicted as part of this action.

Block root: one `DeclareBlockers` offer with a role per blocker. For blocker X,
select zero or one native legal attacker; repeat for blocker Y. Proposed typed
role subjects bind X and Y explicitly (currently their identities are conveyed
to the model mostly by role labels). Global state remains authoritative in Rust.
Independent assignments can share an attacker where native rules allow it;
there is no fabricated global distinctness constraint across blockers.

Menace currently falls back to sequential declarations because native resolution
can remove an illegal singleton assignment. The sequential policy's action is
its actual Command path; do not call several paths producing the same final
board distinct *semantic declarations*. Grouped menace support requires native
prefix-completion constraints that exclude illegal/aliased declarations before
sampling. That is deferred, not repaired by Python masking after sampling.

### Payment

Pay-or-not (kicker/ward/unless): native support is pay if affordable, plus decline.
Expose typed cost and purpose where public; score the current prompt only. A
ward trigger, opponent payment or subsequent reveal starts a new decision.

Waterbend root: native code determines legal tap candidates, remaining generic
cost G and minimum taps m needed given available mana. For an illustrative
G=3, m=1 and candidates [A,B,C], answers are subsets of sizes 1..3. Selecting
{A,C} lowers to tap A, tap C and the native remainder completion where still
surfaced. Native auto-resolution may consume a forced suffix. The decoder never
pays a second time and never invents an ordering permutation.

This grouping currently excludes candidate mana abilities, battlefield triggered
effects/triggered mana and delayed triggers; those prompts use atomic waterbend
choices. End-of-payment effects may reveal information. No downstream choice is
cached past that boundary. See native `structured_waterbend_offers` and existing
wide payment/reveal tests in [test_compound](../tests/training/test_compound.py).

Learn/outside-game candidates must retain their own typed addresses; they are
not in-game ObjectRefs. Scry, discard, legend choice and other current prompts
remain atomic until their structured grouping is separately justified. Unknown
mechanics with no supported projection fail explicitly; they never become random
moves, silently dropped candidates or permissive masks.

## Typed boundaries: extend existing owners

These are proposed records and methods, not claims of existing APIs. Prefer
closed Rust enums and frozen typed Python mirrors in existing modules. Use
Pydantic at configuration/wire admission, dataclasses for hot-path records;
validate once rather than copy every tensor on every forward call.

| Boundary | Proposed data and responsibilities |
| --- | --- |
| Native projection | Extend the model projection of `StructuredOfferSet` with typed role kind, role subject, candidate semantics and public fixed parameters. `CandidateValue` variants: subject, outside candidate, content-bound mode/ability, boolean and native-supported amount. Preserve opaque routing IDs separately. |
| Root binding | A bound decision combines existing Observation identity, canonical frame fingerprint/revision/actor, structured projection digest and factorization version. Match/episode identity is supplied by the existing caller. A local offer ID alone is never sufficient. |
| State encoding | Named view over visible object rows `[N,D]`, validity `[N]`, root context `[D]` and a native viewer-safe subject-to-row mapping. Include player/stack/outside distinctions. Physical IDs perform joins only and never enter embeddings. Missing visible subject joins are errors. |
| Candidate encoding | Ragged typed role/candidate rows `[sum C,D]`, role offsets, source/target/assignment/payment links and explicit numeric units. Join dynamic subjects to visible embeddings; mode/ability references join the existing checked semantic program catalog. No second content vocabulary inferred from names. |
| Decoding | `bind` creates an immutable bound distribution. `sample`, `score(tape)`, `complete(prefix)` and `next_command_distribution(prefix)` share that root, factorizer and masks. Output retains typed submission, tape, factor supports/probabilities/log probabilities and value coordinates. Terminal is a separate result, never empty softmax. |
| Learning | Loss consumes recorded behavior factors and explicit terminal/bootstrap targets, not engine objects. Retain current grouped/sequential credit choices and scalar prefix-value contract. Decoder changes do not select a new objective, discount clock or value-token architecture. |
| Execution | Existing native lowering validates submission on an exact fork; one runner drains bound Commands against the live authority and records actual receipts. It owns cancellation/reset. The model owns no engine mutation or pending command queue. |

A distribution cannot expose complete-action logits for exponentially large sets.
Only its current factor or canonical next-Command probabilities are materialized.
Batching concatenates candidate rows with offsets; memory scales with observed
objects, roles and candidates. Implementation limits (today some native
cardinalities are u16, and object encoding has capacity) stay explicit admission
errors, not arbitrary truncation or a promise of infinite capacity.

### Prefix legality without another rules interpreter

For the initial slice, native code exports a closed factorization: atomic choice,
unordered distinct subset with certified cardinality, or product of independent
assignment roles. Every admitted factor alternative must have a legal completion.
Move the authoritative support calculation into native offer logic; Python may
vectorize its masks but may not own independent legality equations. Cardinality
support is cheap because the native bridge has already proved interchangeability.
Model-dependent zero probability is separate from absence in legal support.

A proposed native prefix query over those admitted forms returns the next typed
alternatives, completion, or explicit unsupported/stale error. It does not step
the live game, peek at hypothetical future private observations, or explore every
complete assignment to establish legality. Existing native lowering remains the
final validation. This is a bounded extension of `StructuredOfferSet`, not a new
world/session abstraction. Historical decoder tapes can be translated to this
support query without changing their ordering or probabilities.

General `depends_on` graphs and ordered/repeated selections remain rejected for
grouping. If an existing atomic path covers the mechanic, publish that path before
sampling. Otherwise reject admission. A grouping failure after sampling is a
failed decision; retry-until-valid would renormalize the policy and is forbidden.
Future coupled support must come from Rules with a constructive continuation
contract before manabot can score it.

## One probability and identity contract

For bound root x, prefix z<t and native legal factor support L_t:

`p(z | x) = product_t p(z_t | x, z<t, L_t)`

`log p(z | x) = sum_t log p(z_t | x, z<t, L_t)`.

Illegal entries have exact zero probability. A singleton factor has probability
1, log probability 0, entropy 0 and no actor gradient. Native auto-steps are not
extra sampled factors. Record them through execution receipts and separate
counters. Nonterminal empty support and non-finite scores fail explicitly.
Deterministic argmax is an execution mode, not an on-policy categorical sample;
its induced behavior must not be labeled with stochastic policy likelihoods.

**Identity.** Store root binding, decoder/factorization identity, canonical
submission and semantic candidate identities alongside the token tape. Integer
tokens alone are uninterpretable after reordering. Command IDs identify executions,
not trainable actions. Multiple equal-definition physical objects remain distinct
when the engine distinguishes them. Unordered subsets have one path in native
canonical order; reordered answer lists canonicalize to that same identity.
Changing candidate order changes a recurrent policy and is schema-significant.

**Aliases.** Do not assign one path's probability to a semantic action represented
by several paths. Its probability is the sum over those paths. Prefer a unique
canonical path for supported grouped actions. Where ordering actually changes
effects, retain distinct sequential Command paths. Do not merge by final board
hash, or by display label. Canonical include/exclude order handles present
attack/payment subset aliases without factorial enumeration.

**Rescoring and teacher forcing.** Rebuild the exact root projection, validate
its digest and subject bindings, and walk exactly the recorded tape using the
same masks. Current weights provide differentiable log probabilities; collection
weights/probabilities stay frozen data. Reject missing/trailing tokens, impossible
prefixes, unknown candidates and changed schemas. An external declaration target
needs native canonicalization to a unique tape, or explicit probability
marginalization; do not choose a convenient ordering and call it the target.
Rescoring at a later microstep must retain the original root and prefix.

**Entropy and KL.** Joint entropy is the expectation under the policy of the sum
of conditional entropies. A sampled path's sum is an estimator, not exact joint
entropy. Similarly joint `KL(p||q)` is the p-prefix expectation of conditional
KLs when both use the same canonical factorization and compatible support.
Existing sums of reverse-KL penalties at retained behavior prefixes remain
sampled-prefix regularizers, not exact joint reverse KL. A zero q where p is
positive makes this KL infinite; do not smooth it away. Different flat/compound
factorizations require a shared canonical action distribution or separately
named losses, not direct factorwise comparison. This interface does not change
existing estimators or authorize stale behavior data.

**Search.** Extend `project_compound`, `advance_compound` and their receipts;
keep the original viewer root and actual prefix. For next Command c, sum the mass
of disjoint decoder-prefix events that lead to c, integrating over unchosen
suffixes. Existing payment projection already does this without enumerating
subsets. At a retained prefix, skipped earlier taps stay at zero mass even when
they remain engine-legal. Preserve v3 local-update positive-support restriction,
zero targets/counts and unavailable unvisited Q; no flat fallback or epsilon.
Lower every supported route through native authority and use the selected branch
session's fork/snapshot/apply operations. Do not clone an active selected branch
behind its runtime. Search budgets, hidden-world sampling and leaf values remain
the existing planner's responsibility.

## Information, execution and persistence

Grouping needs an engine-certified interval, not merely the same actor twice.
The certificate covers a known declaration/payment/eligible cast, its allowed
internal steps and its stop condition. A reveal, opponent decision, trigger
requiring a new choice, priority response, or changed economics ends it. Effects
at the final step may occur normally; no prefetched choice consumes their new
information. Same-kind/same-actor checks alone cannot prove this for future
mechanics. Unknown intervals use sequential execution.

The existing runner owns one pending suffix per game/branch. Bind it to episode,
actor, root fingerprint, projection and expected Command revisions. Before each
step validate actual frame/route and any object incarnation preconditions. On
mismatch discard the suffix and report a typed interruption; do not silently
resample to make a stale action appear successful. On reset, new game, branch
switch, policy replacement, termination or cancellation clear both prefix and
queue. Unsubmitted drafts can be abandoned freely. Once Commands are committed,
cancellation cannot roll them back; retain the executed prefix and a failed or
interrupted record, and only then start a new decision if the caller requests it.
Fork-based lowering does not make the live multi-Command drain crash-atomic.

Reuse [GameSession](../etude/server.py), [Trace](../etude/trace.py) and
[AttemptStore](../etude/attempts.py). GameSession turns policy choices into bound
commands and calls `step_semantic_command`; canonical deliberate decision rows
exclude automatic passes, while Trace events retain `auto` and presentation.
`_wire_message` persists attempts at surfaced frames and terminal;
`_persist_attempt` saves the current trace/canonical replay to SQLite. This is
not a demonstrated fsync-per-microstep or compound crash-resume guarantee.
Record model/tape provenance with the existing training/search evidence owner,
linked to actual Commands; do not invent another session or identity database.
Never put private sampled-world evidence in viewer-facing table payloads.

The [experience protocol](../etude/experience_protocol.py) already has richer
mode/payment/boolean candidate variants than the agent-native projection. It is
a separate presentation DTO, not permission to infer engine legality. Add native
meaning once and project it into consumers as needed; no table wire overhaul is
required for the first delivery.

## Alternatives considered

| Mechanism | Benefit | Concrete limitation | Recommendation |
| --- | --- | --- | --- |
| Enumerate completed actions and score each | Simple categorical likelihood and search edge | 65 attackers give `2^65` subsets before blocks/payments; violates scope | Reject as universal representation. Keep categorical scoring for genuinely atomic alternatives. |
| Keep flat focus everywhere; enrich more slots | Smallest flat-model change | Does not express role dependencies, action construction or semantic amounts; fixed slots remain the contract | Preserve admitted weights only. |
| Typed ragged candidates + current GRU factorization | Reuses sampling/rescoring, native lowering, training and search; richer runtime inputs | Sequential cost scales with candidates; new input features require new model identity | Recommended first implementation. |
| Query/key categorical pointer per role, with stop/repeat controls | Direct variable-object scoring; can avoid binary scan for one-of-N | New tape and likelihood law; subset stop/order aliases and search mapping need new proof | Later decoder behind the same contract, not silently substituted for the GRU. |
| General native dependent-action cursor now | Could eventually cover coupled targets/allocations | Current `depends_on` is not such a service; broad Rules constraints and boundaries are unsolved | Limit cursor to existing certified forms. Keep other decisions sequential. |

The successful result is a policy consumer using typed choices without decoding
card names, with one shared likelihood implementation across learning and search.
The failure to avoid is a “universal” API backed by Python legality, every caller
reconstructing probabilities, or attractive typed fields disconnected from native
rules. Richer embeddings do not by themselves establish better play.

## Incremental migration and checkpoint admission

**Review now:** approve the bounded semantic contract, canonical probability
semantics and proposed delivery scope before implementation.

1. **This slice after review — native typed projection through serving/search.**
   Add typed source/role/candidate data for the examples above and native prefix
   support for existing admitted forms. Preserve the canonical frame/Command
   schema and existing projection serialization where feasible through an
   explicit model-projection entry point on the same offer owner. Add the Python
   typed mirror and bound-distribution facade around admitted flat/compound
   policies. Migrate serving, teacher forcing and local-search projection together
   so they share tape identity/support. Prove unchanged old-policy outputs and
   native replay before introducing learned feature changes. This additive
   projection is a compatibility boundary, not a duplicate legality owner.
2. **Typed candidate model and ordinary learning slice.** Add the smallest
   reviewed decoder selector to `AgentSpec` and existing recipe helpers. Use the
   visible object joins and semantic role/parameter features with the current
   GRU; keep old initialization/state keys and decoder inputs untouched. Integrate
   existing `train_compound`, export, `load_checkpoint_agent`, arena and configured
   player paths. Keep unsupported objective/value/belief combinations rejected.
   Do not claim ordinary PPO trains compound policies just because inference
   shares an interface. Full bounded workflow demonstrates this slice.
3. **Remove migrated duplication.** Move next-factor support and route ownership
   out of Python verb dispatch in `local_compound.project_compound` into the
   native-backed distribution; retain its tests and semantics. Replace mapping
   accesses/validation duplication in `flatten_projection` with the typed parser.
   Share suffix lifecycle validation between serving and rollout without forcing
   search to own a serving queue. Keep deterministic `RaggedPolicyDecoder` only
   as a named parity/benchmark instrument while it has consumers. Do not delete
   admitted flat `_add_focus`, old compound label features or their tests: those
   are required weight interpretations until their artifacts are retired.

Use serialized `agent_hypers` and ordinary strict loading. Missing new decoder
metadata resolves only to the historically defined flat/compound interpretation
based on existing fields; new recipes serialize their explicit decoder, model
projection schema, factorization/order version and required input capability.
Bind these along with existing world/setup/content/input identities. Reject
conflicts between the new selector and old `compound_decisions` compatibility
field; do not maintain two independent truth sources in new recipes.
Equal-shaped tensors do not authorize weight transfer. New typed features are a
new model choice, not a state-dict port or automatic upgrade.

[checkpoint_world](../manabot/model/world.py) currently checks the exact full
binding, including decision schema and actual setup. Avoid bumping the canonical
schema merely for an additive model projection; if a real semantic/ABI change
is needed, follow WORLDS.md and retain old policies only in their admitted
runtime. Never relax that guard to make old bytes load. Extend new checkpoint
metadata separately and validate it strictly. Preserve old raw/EMA distinctions.
The selected-world loader must use authored Allies/Lessons decks **and
sideboards**, complete semantic input and pinned actual bytes. The default
`runtime_fingerprints()` Interactive mirror is not a selected-world identity;
its explicit `match_hypers`/observation arguments are required for that setup.

No PR is published by this contribution. Later implementation PR descriptions
should name the concrete supported choices, any newly versioned input and
compatibility evidence. Task completion follows the full software acceptance,
not design publication or one successful synthetic checkpoint.

## Acceptance and cost proposal

Extend existing suites rather than invent a second harness. Proposed gate
commands (not executed in this design contribution):

```bash
cargo test --manifest-path managym/Cargo.toml --test rules_tests structured
uv run pytest tests/training/test_compound.py tests/sim/test_local_update_compound.py tests/sim/test_structured_policy.py tests/semantic/test_decision_contract.py
```

The existing `rules_tests` target includes the structured fixtures; run debug,
not release-only. Rebuild the native extension using repository instructions
after Rust edits before Python integration checks.

| Required observation | Acceptance |
| --- | --- |
| Sample vs teacher-force | Same frozen root/weights/tape gives matching conditional probabilities, summed log probability and submission; gradient checks retain score-function/finite-difference agreement. |
| Joint normalization | Enumerate only tiny offer/subset/assignment fixtures; total mass 1 within numeric tolerance; aliases canonicalize or sum correctly. Reject zero-completion prefixes before sampling. |
| Dynamic features | Same-name distinct objects join to correct rows; role subject, source, target, mode and public amounts survive projection. Hidden-world swaps with identical viewer inputs leave policy bytes unchanged. IDs are not learned features. |
| Wide support | 65 attackers, 65 blockers with 35 targets and 65 payment candidates; zero omitted candidates, no complete-action enumeration, canonical replay equality. Record object-capacity errors separately. |
| Lifecycle | Stale revision/fingerprint/incarnation, engine replacement, policy reset, terminal and interruptions clear queued work and fail explicitly. Native automatic steps are counted separately. Tests cross a real reveal/opponent/trigger boundary. |
| Search | Exact canonical next-Command marginal sums to 1; preserves unreachable payment zeros, root/prefix identity, selected branch ownership, target reader and receipt replay. |
| Admission | Old admitted flat and compound fixture outputs/state keys unchanged; new checkpoints round-trip; malformed metadata, changed order/world/setup or missing typed inputs fail before play. |
| Complete workflow | Existing regime trainer, raw export/reload, ordinary player and arena with exact replay; both selected deck/seat assignments with actual admitted fixture bytes. No synthetic artifact is advertised as a trained challenger. |

Proposed bounded integration ceiling after approval: one CPU thread, at most
15 minutes per complete fixture attempt, no paid compute. Failures remain recorded;
no automatic budget extension. This is a new proposal, not inherited permission
from ETU-104. Gate can run fixture optimizations only after review authorizes the
software proof; no full campaign is needed.

Measure projection, state/candidate encoding, factor decoding, lowering and full
player decision time separately on identical retained roots, warmup and seeds.
Compare the preserved compound decoder and typed decoder at 3/35/65 candidates,
report median/p95 and peak allocation method, plus full-game throughput. Extend
`structured_benchmark` rather than quote its old prototype timing as learned
inference. Suggested performance review threshold: over 20% median full-player
slowdown on matched roots triggers an explicit design decision before default
adoption; correctness and full candidate coverage may not be traded for speed.
This number is a review proposal, not a measured result or strength criterion.

## Decisions for Jack Heart

1. **Scope:** approve existing native safe groupings plus typed atomic mode,
   ability, target and payment semantics, with coupled/ordered mechanics staying
   sequential. A general dependent-action engine would be separate Rules work.
2. **Model choice:** approve runtime-object/role features on the existing GRU as
   an explicit new decoder, preserving old flat/compound meanings. Defer pointer
   architecture and empirical comparison to separately owned experiments.
3. **Delivery:** approve the integrated serving/rescoring/search contract first,
   then the bounded train/export/reload proof above, including the proposed
   15-minute cap and performance review threshold. Review is not supplied by
   this headless contribution and ETU-107 remains open.

Check result: `uv run python` local-link/fence check passed (18 links, two files);
`git diff --check` passed. Runtime validation is deferred to reviewed implementation.
