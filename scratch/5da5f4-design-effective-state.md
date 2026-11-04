# Effective state, shared effects and policy memory

Draft experiment design · 2026-11-03 · Task 5da5f4 · Training Wave

## Authorized training experiment continuation

Jack Heart authorized up to **US $25 total** to experiment in this conversation,
adding “but only use as much as you need.” This supersedes the earlier no-paid-
compute statements for this Task alone. MTG-213's allocation, running pilot and
frozen evidence remain separate. The completed kickoff is retained; the next
outcome is an executed, reproducible representation comparison with actual
multi-seed training and a report, or concrete evidence why the budget cannot
support an interpretable comparison. Broad production replacement is not approved.

Proposed execution choice: compare the current working representation, typed
shared state pooled into addressable objects/frames, and the same typed state
with variable-length record encoders. Keep temporal memory out of the first
comparison. Preserve viewer-safe information parity between the two candidate
arms and document any baseline information gap as a treatment confound. Implement
only the coherent experimental integration needed for runnable agents through
the existing TrainingRegime/collector/learner; no second training loop.

Proxies are sanity checks for correctness and feasibility. Training results
across independent seeds should drive the decision. Use common seed sets and
held-out paired deals/both seats, report cross-seed uncertainty separately from
evaluation-game uncertainty, learning against both steps and cost, and every
attempt/failure. Do not claim statistical sufficiency from an arbitrary seed
count. Prefer a useful replicated comparison to many undertrained arms; document
any necessary narrowing before scoring rather than silently changing treatments.

Before rental, inspect the existing Etude launch/cost-control path, obtain a
current provider quote, and commit a concrete protocol with predictions, endpoint,
seed allocation, evaluation/selection rule, stopping rules and total cost cap.
Account for setup, training, evaluation, idle time, storage and failed attempts
within the same $25 ceiling; reserve enough for termination and final evaluation.
Use enforced lease/deadline/spend controls and verify cleanup. Never treat the
ceiling as a spending target. A minimal ordinary-work canary may calibrate cost;
declare whether it is excluded exploratory work or part of the frozen cohort.
If available resources cannot yield useful precision at meaningful run length,
report that limitation without exceeding the cap. Preserve provenance and publish
result bytes through the existing artifact-store procedure before delivery.

The user authorized experiment execution, not automatic merge or promotion of a
winning representation to production defaults. Review results in this conversation.

## Implementation and launch status — 2026-11-03

**Partial implementation; canary admitted, scientific cohort not yet frozen.** The experimental
`ModelConfig.state_encoder` selects `rows` (unchanged default), `typed_pool` or
`typed_gru`. Both candidates run through the ordinary ActorCritic, collector,
Learner, exports and checkpoint loader. `experiments/state_representation.py`
declares the recipes and bounded jobs; it allocates nothing on import or recipe
construction. Model choice enters architecture identity; historical default
serialization omits it. No native schema, pilot configuration or world changed.

The first comparison deliberately isolates **same-input encoding**. The candidates
factor complete v5 fact *fragments*, preserving memberships and multiplicity;
this is not yet native effect-payload transport. Their frame encoder removes
repeated headers, retains ordered targets and public stack rank, follows parent
and source/object joins, and pools stack/execution/triggers separately. One
within-record reducer pools tokens; the other packs variable-length header/edge
sequences into a GRU. Object vectors and exact action-frame joins survive.
Fact records have five typed tokens each. No state survives a forward call.
The Python prototype performs per-observation grouping and routing; its CUDA
synchronization cost is not yet measured or optimized.

All arms still have v5's information omissions and 128-fact/64-relation admission
limits. They cannot establish sufficiency or fix the original overflow. This is
an engineering narrowing before any scoring, not approval of a new production
contract. The full native source/controller/ability projection and capacity
experiment remain pending review. No obsolete production path is replaced in
this comparison; remove the unselected experimental encoders after that decision,
retaining frozen checkpoint readers and behavioral fixtures.

A native Ty Lee cast exposed a Python reference bug: WhileYouControlSource was
mapped to 2 while the existing native encoder emits 3. The Python mapping now
matches native for both viewers. The current FactEncoder's third-duration pattern
was correctly described in kickoff, but its assumed Python parity was not.
This correction restores the existing native v5 meaning; exact source identity
is still absent.

`uv run --extra dev python -m manabot.verify.representation --out PATH` now runs
12 constructed native fixtures, both viewers' scalar/Python parity, and one
CPU forward/backward per arm/fixture. It retains input NPZs, source/native hashes,
architecture identities, counts and timings in a fresh directory. It is a
same-input engineering proof, not the full source-A/source-B native replacement
proof proposed below. Directed tests additionally cover object/frame relabeling,
parent joins, meaningful stack order, padding, duplicate multiplicity, duration,
checkpoint rejection/reload, action binding and ordinary one-update training.

The native Drill fixture resolves six activations against ten permanents:
60 native records, 120 keyword fact fragments and 12 shared fragment payloads,
with all six timestamps retained. Seven activations produce the expected v5
capacity rejection in both encoders. This is a newly constructed Command prefix,
not a replay of the original pilot seed. The input buffer remains 69,856 bytes
for every arm. Generic width64/two-layer fixture models have 1,014,402 / 1,031,490 /
1,081,410 parameters (rows / typed_pool / typed_gru); these are not the compiled
training catalog's parameter totals. The measurements are single passes without
warmup, precision intervals, CUDA allocation accounting or strength conclusions.

Local engineering evidence is under `.runs/representation/authority-proof/`;
preceding bounded runs remain under `same-input-proof/` and `native-effects-proof/`.
The first lacked active effects; the second added Drill; the third adds the
source-duration parity case. They are all exploratory, uncommitted-source proofs,
not scientific cohort attempts. No remote data publication has been attempted;
result bytes must enter the artifact store before scientific delivery.

**Launch preflight (updated 2026-11-04 02:17 UTC).** Jack Heart's credential
correction supplies nested Doppler injection: artifact credentials in
`manabot/prd`, RunPod credentials in `etude/prd`. The earlier missing-key result
was an incomplete environment, not an unavailable credential. The corrected
provider quote is L4, minimum 6 vCPUs/48 GB, available at $0.59/hour, with the
configured ceiling $0.60/hour. No pilot machine or state is reused. A separate
Task-local Etude home `.runs/representation/etude` contains only copied provider/
storage/template metadata and the new `representation-5da5f4` on-demand, max one
machine initially. It contains no existing machines, jobs or cohorts.

Etude 0.2.12's job/cohort reservations, bounded setup, machine deadline and
finalization/cleanup safeguards bound rental charges. Source inspection confirms
that transfer estimates omit storage byte-hours. Current official
[RunPod pricing](https://docs.runpod.io/pods/pricing) lists container storage at
$0.10/GB-month and per-second billing; the 50-GB container has no persistent
network volume. [R2 pricing](https://developers.cloudflare.com/r2/pricing/)
lists Standard storage at $0.015/GB-month, Class A/B at $4.50/$0.36 per million,
free egress and rounding to billing units. Do not assume unused free tiers.
Reserve **$6 ancillary** for one rounded million of each operation class
($4.86), up to 10 GB-month of experiment evidence ($0.15), 50-GB container disk
through 15 aggregate hours (<$0.11), and rounding/termination margin. This covers
execution plus the first month's evidence storage; repository artifact retention
continues under its existing policy. Source bundles, logs and exports are bounded
by fixed runs/updates; inspect retained bytes/usage before the main cohort and
stop before 10 GB or 100,000 operations of either class. No extra persistent
storage, bucket or independent background evaluator is allocated.

### Frozen excluded canary — committed before rental

The canary uses `runs(revision=COMMIT, canary=True)`: all three arms, seed 73200,
one real complete-game update on 64 streams, original width64/two-layer model
and unchanged damped-policy schedule. It is excluded exploratory feasibility,
never a learning replicate. Each job has a 300-second learning watchdog,
60-second finalization reserve, 0.2-hour job limit, $0.30 cap and one attempt.
The dedicated on-demand additionally allows at most 900 seconds of setup within
its paid rental reservation. Cohort limit **$0.90**, parallel one. No retries.
Prediction: finite updates/exports for all arms, each update below 90 seconds,
GRU collection+learning time no more than 1.5 times the row arm. Read actual
transition counts, complete-game counts, time and numerical health. The first
update includes cold costs; it is not a throughput distribution or proof of
learning. Any invalid numerical state, failure, or forecast unable to fit a
meaningful replicated comparison stops the dependent cohort while preserving
all attempts. A revised scientific protocol requires a new pre-run commit.

The aggregate reservation is $0.90 canary + at most $12 training + at most $5
final evaluation + $6 ancillary = **$23.90**, leaving $1.10 unallocated under
Jack Heart's $25 cap. These are ceilings. Current quote implies less than these
ceilings; no unused allowance is a spending target. Keep setup, idle, failures,
final exports and cleanup charged. Verify all task-owned rentals deleted at the
end. Paid spend remains $0 before this preflight commit.

### Proposed frozen learning protocol (not launched or accepted as sufficient)

- Three arms above, four common independent seeds 73201–73204; treatment order
  rotates within seed. One same-world Allies/Lessons recipe: width64, two
  attention layers, FiLM, no history, both seats, 64 streams, one completed game
  per stream/update, 128 updates; existing damped-policy schedule. Learning
  clocks record actual transitions, games, optimizer exposures and total cost.
  This matches games/updates, not exact transition counts or hardware time.
- Primary contrast: typed_gru minus typed_pool final raw-policy score against the
  frozen heuristic; row baseline contrasts are secondary/descriptive. Prediction:
  primary mean gain is at least 0.03 at this horizon, with no more than 25% extra
  measured training cost. Final raw export only; no best-checkpoint selection.
  Interior exports at 32/64/96 describe curves, not additional selection attempts.
- Proposed held-out deals 84001–84032 with both deck assignments and both seats
  (128 scheduled games/checkpoint). Evaluation goes through the existing local
  evaluation API and retains every outcome/failure. The common per-game limits
  and actual compute envelope still need freezing with the priced allocation.
- Across-seed inference uses the four paired seed-score differences, a two-sided
  Student-t interval with 3 degrees of freedom, and all four individual values.
  Deal-paired bootstrap intervals within each seed are separate evaluation noise,
  never method replication. No winner unless the primary interval excludes zero,
  the mean clears 0.03 and the cost bound holds; otherwise inconclusive. Four seeds
  do not by themselves certify adequate precision, and a null result is not
  equivalence. No post-hoc extra seeds or endpoint changes within this cohort.
- Proposed scientific reservation: at most $12 training ($1/job, 1 hour/job,
  45-minute learning watchdog, one attempt) and $5 evaluation, within the
  aggregate $23.90 reservation above. These are ceilings, not spending targets. No automatic retries; a failed
  or incomplete run prevents a complete-cohort conclusion. An unpriced component
  or uncovered lease prevents launch. Never reuse MTG-213's machine/allocation.
- Before execution: obtain current quotes, bound storage retention/bytes and
  requests, establish a dedicated machine allocation, calibrate one ordinary
  excluded update for useful run length, freeze the final protocol in git, and
  enforce the combined $25 reservation. The local test smokes are excluded from
  science and do not establish this forecast. If meaningful replication cannot
  fit, record that conclusion without a weakened "successful" substitute.

Remaining implementation: priced launch/canary, any demonstrated GPU feasibility
fix, frozen evaluation/reduction wiring, actual cohort, cross-seed report and
artifact publication. Full native transport and temporal memory remain separate
review decisions. Gate owns the affected suites; the code Graph stops at demo.

**Focused check result:** new model checks (14), same-source trigger/action-binding
checks (6), native source-duration parity (1): 21 passed; bounded native proof:
36 fixture/arm forward/backward attempts, 0 failures. Ordinary one-update training
and checkpoint reload passed for both candidates. Scientific replicates: 0.

## Problem, authority and accepted direction

Jack Heart's subsequent clarification: representation-size, execution-cost and
small-fixture proxies are useful sanity checks, with less weight in architecture
selection. Actual training across independent seeds, with useful statistical
precision, should drive that decision. Correctness and feasibility checks remain
necessary; favorable proxy results do not establish learning quality, and modest
proxy disadvantages alone do not rule out a viable training arm. This supersedes
the proxy-led selection recommendations below. Training arms, allocation, seed
count and scoring protocol still need specification. The later $25 authorization
above permits experiment execution once those details and spend controls exist.

Jack Heart requested this investigation on 2026-11-03 after MTG-213's fact-capacity
failure: make decision-relevant state understandable and learnable without
repeating every effect on every permanent. Consider effective characteristics,
shared effects, variable-length state encoding, distinct continuation structures,
and history/memory together. His subsequent “rex” correction names the Task
conversation environment; it does not request a Rust neural-network rewrite.
These are accepted questions and experiment scope, **not an accepted production
architecture**. Review measured examples before broad replacement. No paid
compute or MTG-213 GPU allocation is selected here.

The immediate repair remains MTG-213's prospective v6 pilot with 256 facts.
Frozen checkpoints, observation versions, worlds and experiment evidence keep
their original identities. This checkout is `c823c9c8f53df06c723e93d6d3e7a8676a71fa9d`,
which has v1–v5 configuration and fixed 128 facts. The transferred
[architecture snapshot](observation-architecture-snapshot.html) describes source
`1af50386fdc4d96b936e81a8c32fae3054dc2c5c` with v6/default 512 facts; its pilot
capacity is 256. It is reference evidence, not this checkout's runtime.

The retained failure at `2adbd0758e40518947daf91569e1938b060d4346`, stream 181,
seed 31482, action 87, has seven Drill activations, ten affected permanents,
70 native records and 140 distinct per-keyword facts. No exact duplicates were
removable. This kickoff has not replayed that historical failure. The bounded
probe below reproduces its record arithmetic with constructed data.

The beneficiary is the developer comparing representations and ultimately the
agent choosing among the complete native legal actions. Academy and the LLM
must receive the same viewer-safe distinctions if a later production contract is
adopted. No human-only semantics, silent choice, cancellation, stale-command
relaxation or forced-decision change is included.

Training's current chapter asks for reproducible playable checkpoints and
full-game improvement at declared cost. This experiment establishes neither KR;
it supplies an auditable representation and an execution decision for a later
frozen learning comparison. There are no supplied numeric chapter targets.
Training memory's seat-routed credit, train-mode admission and recovery boundaries
remain binding. MTG-239's five-card work is the checkout base; its native
continuation and text contracts survive. MTG-225 owns broader Sagas/faces/copies/
turn scheduling; MTG-231 owns revealed-library belief transport. This plan does
not implement those outcomes or assume their completion.

## Demo and result available now

Run from the repository root:

```sh
uv run --no-project --python 3.12 scratch/representation_probe.py
```

The standalone, typed, standard-library probe prints PASS plus exact record and
byte counts. It checks lossless expansion of shared effects, duplicate
multiplicity, object renaming, fixed membership, expiry, history eviction,
source identity, timestamp ordering, typed frame/action joins and an explicit
byte-budget rejection. It makes no native imports and launches no games,
training, cloud jobs or subprocess workers. The probe's restricted keyword
algebra illustrates information loss; it is not a replacement rules resolver.

| Constructed example | Before | Candidate representation | What this establishes |
|---|---:|---:|---|
| Seven Drill-like applications to ten objects | 70 payload copies / 140 keyword facts | 7 complete payloads + 70 memberships | Reversible factoring, preserving all seven timestamps |
| Same effects, compact JSON including exact object references | 16,871 bytes | 6,033 bytes | 64.2% fewer JSON bytes for this schema and fixture |
| Same effects, existing active float32 fact payload | 6,720 bytes | No candidate tensor ABI yet | JSON savings are not a float-tensor comparison |
| Three frames with four target edges | 7 active 49-float rows: 1,372 bytes; full padded table: 12,544 | Typed JSON: 478 bytes | Header duplication can be removed; not a throughput result |

Grouping plus JSON serialization took 149.333 microseconds in the initial
single pass. This is descriptive evidence for a tiny fixture, with no warmup,
repeat distribution or performance threshold. A rerun naturally changes it.
The full production encode/forward/backward costs remain unmeasured.

## Current architecture and consequential findings

```text
Native Game: effects, characteristics, stack, suspended choices, trigger state
  -> Observation::for_player(viewer) + authoritative decision capture
     -> native scalar/vector tensors <-> Python reference encoder
        -> facts summed into addressable object vectors
        -> continuation row projection + global/per-frame pools
        -> object attention -> legal action focus joins + critic
     -> portable decision capture -> LLM text / Academy viewer surfaces
CardDefinition -> learning_catalog -> per-definition token GRU -> catalog joins
Public event append -> bounded history suffix -> optional stateless event pool
```

| Authority / consumer | Existing mechanism and implication |
|---|---|
| `managym/state/src/effect.rs`, `src/flow/effects.rs` | `EffectRecords` owns ordered records and derived indexes. `record_continuous` shares a timestamp across a fixed affected set. Each record retains controller and exact `source_object` for source-dependent duration. Do not replace canonical engine storage to optimize a policy transport. |
| `src/flow/statics.rs`, `agent/observation.rs` | `Game::characteristics` already owns effective values and activated abilities. Observation includes effective P/T, keywords, ability loss, copy metadata and raw per-object effects. Full effective activated-ability descriptors are not a `PermanentData` field: the experiment must audit grants, grantor and ability indexes through this authority, not infer them from printed card lists. |
| `agent/observation.rs::ContinuousEffectData` | Carries modification, details, duration, timestamp, source definition/name. **Omits controller and exact source incarnation.** Two records tethered to different instances of one definition can project identically although removing one source ends only one record. This is a verified field-level loss; a reachable paired-position proof remains required. Memory cannot repair a missing initial distinction reliably. |
| `agent/observation_encoder.rs`, `manabot/env/observation.py` | v5 facts split a modification into keyword/subtype details and repeat duration/order/source per affected object. Widths and defaults are frozen. Timestamp ranks express meaningful order without raw IDs. |
| `manabot/model/facts.py` | Facts are summed into object embeddings, not independent attention tokens. Three 4,096-bucket embeddings include hashed vocabulary IDs and source definition. The two duration flags distinguish EndOfTurn and Indefinite; WhileYouControlSource uses neither flag. It is a third pattern, not accidentally identical to Indefinite. Hashing and pooling are neural compression, not injective semantic transport. |
| `agent/continuation.rs` | Ten frame kinds: three public stack kinds, announcement, resolution, obligation, waiting trigger, placing trigger, placement cursor and delayed trigger. One frame expands to `1 + targets + context + sacrifice + hybrid + 2*mana_outputs` rows. The 64 limit is on expanded rows, not frames. Nonacting viewers get public stack and delayed triggers, not the acting viewer's private decisions. |
| `model/continuation.py`, `model/actor_critic.py` | Global valid-row mean and per-frame means, with two object and two definition joins. Waiting/placing/delayed frame positions are suppressed as learned features. Other frame/parent integers enter features; the generic frame index therefore mixes public stack rank and routing. Adding a target also reweights a repeated frame header in the global mean. Action continuation selectors are checked joins and must survive any split. |
| `src/cards/tokens.rs`, `model/semantic_cards.py` | Rules definitions generate complete token sequences, including referenced definitions. GRU encodes each program within one call. This is not temporal memory. Embedding allocates padded catalog tokens before packing the GRU, so “variable length” alone does not establish a memory reduction. |
| `agent/policy_history.rs`, `model/recent_events.py` | Native public suffix has at most 32 records: arrival, damage, life, cast, resolved, countered. It is not an effect-creation ledger. Links expire on zone moves; forks copy on write. Optional neural encoder adds ordinal age then pools; no hidden state survives a forward call. Pilot history is disabled. |
| `src/decision_capture.rs`, `manabot/semantic/text_observation.py`, `verify/text_observation.py` | Portable current decision and text preserve frame context, eligibility and public history. Text is a consumer of authority, never a second visibility filter or source for tensor parsing. |
| `training/rollout/collection.py`, `training/state/learning_state.py`, checkpoint/world modules | Current policy context is recomputed per observation. Persistent recurrence would change collection, sequence sampling and recovery contracts; adding a GRU in `ActorCritic` alone is insufficient. |

The mixed relation table is a flattened typed structure, not intrinsically
wrong. Its concrete problems for this experiment are repeated headers, coupled
capacity, overloaded index/order features, and uniform pooling across unrelated
structures. Separate namespaces and explicit joins address these without losing
an ordered stack or a pending choice.

### Counterexamples and native proof anchors

- **Expiry:** identical effective flying now from EndOfTurn versus Indefinite
  grants diverges at cleanup. Probe passes the distinction; native anchor:
  `tests/rules/cr_613_continuous_effects.rs::cr_514_2_an_end_of_turn_record_ends_at_cleanup`.
- **Source dependency:** equal-looking records attached to source A versus B
  diverge when A leaves or changes controller. Native `end_effects_on` and
  `end_effects_for_control_change` compare exact references/controller. Existing
  `cards/tests/cards/tla.rs::ty_lee_chi_blocker_locks_a_creature_down_while_you_control_her`
  exercises source departure. The new paired-observation test must expose the
  missing source link and survive leave/reenter of the same physical card.
- **Timestamp/layers:** add-then-remove versus remove-then-add flying has the
  same unordered modification multiset and different effective result. Probe
  distinguishes it. Native grant/ability-loss ordering test is
  `cr_613_7_a_grant_later_than_ability_loss_applies_and_an_earlier_one_does_not`.
  Preserve order within each layer and between effects and static sources;
  transport factoring must not implement its own layer calculation.
- **Membership:** ten objects affected at resolution do not include an eleventh
  entrant or a returned incarnation. `Affected::Object(ObjectRef)` owns this
  distinction; replacing membership with “opponents' permanents” is incorrect.
- **Copy versus effective state:** a copy inherits copiable values and ordered
  exceptions, not all current counters/status/effects. Native anchors are
  `ran_and_shaw_copy_records_its_copy_effect_and_exception` and
  `ran_and_shaw_copy_ignores_the_sources_status_counters_and_effects` in `tla.rs`.
  Copy exceptions cannot be merged into ordinary expiring effects.
- **Ability loss/grants:** current effective activated-ability order determines
  activation/payment indexes. Loss of the granting Equipment/Aura may remove an
  action even where P/T and keywords stay equal. The native proof must compare
  actual offers and focus bindings, not just tensor inequality.
- **History eviction:** the probe appends 32 identical public events after two
  different grants. Equal suffixes and equal current keywords still have
  different cleanup futures. Existing history is even less sufficient because
  it never records continuous-effect payload creation in the first place.
- **Continuation:** public stack rank must matter; presentation order of waiting
  triggers must not. Preserve clause ordinal, retained legality and parent
  execution edges. `tests/verify/test_continuation_input.py` already tests target
  distinctions after equal costs, retained eligibility, viewer safety and
  trigger-choice invariance. Extend these fixtures, do not replace them.

These anchors were read, not rerun during kickoff. They justify the experiment's
required native counterexamples without claiming a proof of all Magic states.

## Chosen experimental contract

**Recommendation (engineering choice, 2026-11-03):** a lossless typed,
variable-length viewer projection with object-addressable outputs, effective
characteristics and shared active-effect payloads. Keep a stateless current-state
arm as the reference. Learn optional historical context on top. Do not make a
network responsible for remembering an ongoing rules obligation omitted from its
input. “Sufficient” means preserving the viewer's decision-relevant observable
state; it does not make hidden cards observable or prove a fully observed MDP.

```text
One viewer-safe native projection
  objects: effective characteristics + local typed facts + ability references
  effects: full payloads + exact affected-object membership edges
  stack: ordered entries
  execution: parent-linked announcement/resolution/payment frames
  triggers: waiting occurrences + placement state + delayed conditions
  history: optional viewer-visible chronological suffix
       |
       +--> pooled typed encoder (control)
       +--> per-record token encoder + typed joins / object attention (candidate)
       +--> candidate + per-seat persistent memory (later experiment)
       |
       -> per-object vectors + per-frame vectors + shared context
          -> unchanged semantic Command/focus binding -> actor and critic
```

Proposed records use local routing handles, typed references and integer offsets;
no raw object/card allocation ID is a learned numeric feature. A reference names
an exact visible incarnation, a public definition without a live link, or explicit
unavailability. Never invent a link from a source definition to one of several
copies. Hidden/private facts are removed by native viewer authority *before*
lengths, tokenization, hashing and grouping; even counts cannot reveal them.

An effect payload contains modification and ordered detail tokens, values,
duration, controller role, source definition, optional exact visible source,
and layer/timestamp order. Group only complete equal payloads, including source
and timestamp; retain each membership and multiplicity. The probe retains original
record ordinals so expansion is exact. Production may omit those only after a
native equivalence proof. Seven Drill timestamps stay seven records, not one.
This is structural factoring of repeated payloads, not rules simplification.

Static continuous abilities remain attached to their authoritative definitions
and effective source objects. They must not acquire fixed memberships like a
resolved effect: later entrants can be affected by a static scope. Dynamic
resolved effects, copy exceptions and static sources have distinct types.

Separate stack rank from handles. Execution parent links remain explicit, and
waiting trigger identity is a join into action context. Target clauses retain
position and eligibility even after an object leaves. Stack/frame/trigger
summaries do not remove object vectors or the action's precise frame join.

Batching packs record/token arrays with offsets and masks; batch padding is an
execution detail, not a semantic cap. Prototype budgets are **2,048 records and
8,192 edges/tokens per decision, 1 MiB wire bytes, batch at most four**, CPU only,
width64. These are engineering admission limits, not bounds on reachable games.
Validate sizes before allocation; report which visible structure exceeded its
limit. Never truncate, substitute zeros or fall back to old inputs. Gather natural
length distributions later before choosing production budgets. In the initial
probe only explicit byte-limit rejection is exercised; packed allocation bounds
remain to implement.

## Alternatives kept distinct

| Mechanism | Assessment and next discriminating experiment |
|---|---|
| Larger fixed fact/relation tables | Control for admission cost; MTG-213 already owns the fact repair. Does not remove duplication or source-identity loss. Never compare overflowing v5 to a candidate as a speed win. |
| Typed shared records, per-type pooling into existing objects/frames | Cheapest candidate baseline; linear message accumulation in record/edge count, same object attention count. Exact transport can still be compressed by a learned sum. Compare joins and masked outputs on native fixture pairs. |
| Typed per-record variable token GRU + sparse joins + object attention | Preferred candidate to test. Ordered payload programs use within-record sequences; unordered objects/effects use equivariant joins with explicit semantic ordering features. Addresses are preserved outside token values. Avoid serializing the whole game into a single final hidden vector. |
| All object/effect/frame tokens in global attention | Viable small-fixture comparison, but changes attention cost from object count N squared to `(N+E+F)` squared. Membership edges need not become attention tokens. Measure actual forward/backward allocation; compact transport alone is no justification. |
| One whole-observation GRU / pooled global vector | Reject as the sole interface: loses direct target addressability and imposes arbitrary enumeration order unless extra retrieval/ordering machinery is added. A summary may supplement explicit object outputs. |
| Effective state plus finite event window, omitting active effects | Blocked by demonstrated expiry/source/window counterexamples. Recovering exact active records from that suffix is as hard as the missing state, and impossible when the initiating event is absent. |
| Persistent recurrent memory plus complete typed current state | Viable later learning arm for inference from past observations/opponent behavior; not a repair for lossy rules state. Requires the lifecycle below and matched training costs. |

Prior art informs mechanisms, not a strength prediction: [Deep Sets](https://arxiv.org/abs/1703.06114)
studies permutation-invariant set functions; [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)
models set interactions and reduces attention cost using inducing points.
Neither guarantees injective compressed embeddings for these game records.
[R2D2](https://willdabney.com/publication/r2d2/) identifies recurrent-state staleness
and representational drift; this motivates explicit sequence/recovery ownership,
not adoption of its off-policy algorithm into this learner.

## History and memory contract

The preferred first arm has **no persistent policy state**. Current active effects,
known-hand minima, deck disclosure and pending obligations survive history eviction
in native state. Optional history keeps its declared chronological window; old
rows disappearing is explicit and does not change current legal decisions.
A within-observation GRU resets each forward call and is not advertised as memory.

A later persistent arm requires all of these together:

- Memory belongs to `(match instance, stream generation, viewer seat, policy
  generation)`, not whichever player acted last in a vector slot. Updates ingest
  only that seat's authorized observations/events since its committed cursor,
  including public events between retained decisions. No opponent private frames,
  sampled possible worlds or future trajectory facts enter it.
- New game, reset, rematch or slot reuse starts empty. Ordinary turn boundaries
  preserve memory. Seat swaps route to the corresponding independent state; they
  do not relabel an existing player's private memory. A rejected/stale command
  commits no new policy transition. Native trivial skipping stays on.
- Forks clone memory and cursor at the fork point and evolve independently.
  Replay reproduces the same ordered viewer inputs and model weights. Departed
  object handles never reconnect to returned physical cards.
- Exact restoration stores hidden state, event cursor, viewer/match identity,
  observation schema, model/checkpoint hash and relevant RNG with committed
  learning state. A weights-only checkpoint is not a mid-game memory snapshot.
  If state is absent, exact restore requires replay from the true initial viewer
  prefix with matching weights; a last-32-event burn-in cannot claim equivalence.
  Missing prefix is an explicit unavailable restoration, not a zero-state fallback.
- Training uses contiguous same-seat sequences, episode/padding masks and declared
  gradient truncation/burn-in. Loss uses admitted learning rows only; burn-in
  reconstructs context without introducing opponent rewards. Parameter changes
  require a declared memory refresh policy. Existing seat-routed return/terminal
  rules remain unchanged. Serving and collection call the same transition API.

No recurrent arm proceeds on a mere random-weight forward smoke. Required
lifecycle evidence includes alternating seats, vector reset, invalid command,
fork, resume and history-window eviction. Explicit symbolic known information
stays native-owned; do not build a second Python knowledge store.

## Execution sequence, scope and deletion

**This slice — kickoff:** source audit, reversible representation probe, measured
examples and this draft design. Complete for planning; focus check is the demo
command above. No production code or default changed. The rest is proposed work
for design review, not permission manufactured by launching a draft.

1. **Native paired-position proof.** Extend existing constructed fixtures through
   Commands and `Game::characteristics`. Produce both viewers' captures for
   expiry/source/control/entry/order/copy/grant cases. Add source controller and
   viewer-safe exact source references in an experimental projection at native
   authority; do not guess them in Python. Show a source-A/B observational
   collision in the old projection and divergence after the same source-removal
   Command. Record full payloads privately and publish only safe summaries.
2. **Bounded representation and neural comparison.** Implement the three admitted
   current-state arms in one research harness, consuming the same captures and
   catalog. CPU, one seed, at most twelve paired fixtures and four observations
   per batch; one forward/backward per arm per batch, no training campaign.
   Compare tensor bytes (including padding), tokens, edge count, parameters,
   activation/allocation accounting and ordinary encode/forward/backward times.
   Reordering nonsemantic handles with repaired joins must permute object outputs
   and preserve corresponding action logits (tolerance 1e-6); semantic stack/order
   changes must remain distinct at the input. Verify gradients through source,
   affected object and action frame joins. Random-output difference is a smoke,
   not proof of learnability. Report any budget rejection as a failed admission.
3. **Review evidence and choose.** Inspect native distinctions, coverage gaps and
   measured costs. Prefer typed shared records with local token encoders if the
   integration cost is justified; retain typed pooling if tokens show no advantage.
   No effect-history omission is eligible. A future learning study must freeze
   worlds, independent seeds, complete-game budgets, held-out deals, evaluation
   controls and compute accounting before running; it needs its own allocation.
   Persistent-memory arm is a separate choice after its lifecycle design passes.
4. **Production cutover only after review.** Allocate the next unused input schema
   after integrating MTG-213; do not preempt v6 or reinterpret it. One native
   viewer projection feeds scalar/vector encoding, Python parity, capture/text,
   learning tokens, model joins, checkpoint receipt and relevant Academy display.
   Generate new identities and keep historical readers explicit. Run normal
   package admission/monitoring checks before switching fresh defaults.

**Delete — do not maintain:** no current engine record/index or frozen decoder is
removed in this design slice. In an accepted production cutover, remove the new
schema's dependence on per-object effect expansion (`effect_facts`/Python twin),
repeated-header relation packing and the untyped continuation projection. Replace
the candidate path completely with the typed projection; do not modernize the old
path first. Frozen v1–v6 readers retain their existing code and receipts as the
explicit historical compatibility exception: selected only by saved schema, never
as a live fallback or dual write. Remove research-only rival encoders and this
scratch probe from the shipping path after the experiment decision; preserve
native behavioral fixtures and measured evidence. No second canonical game state,
copy resolver, event ledger or browser-side visibility implementation is allowed.

Missing catalog entries, dangling current handles, invalid offsets/parents,
nonfinite numeric payloads or unsupported schema are errors. Zero records is a
valid empty set; absent exact source is an explicit unavailable reference, not
proof that there is no source dependency. Historical evidence cannot be enriched
with invented links. Public summaries do not expose private captures or hidden
record counts. Checkpoint incompatibility fails admission before weights execute.

Success looks like a developer seeing why two currently identical creatures have
different futures, and a policy retaining the exact object to act on while shared
effect payloads stay compact. Failure would be smaller JSON paired with larger
attention allocations, an order-sensitive arbitrary serialization, or a memory
model that forgets an active obligation on reconnect. Native differential tests,
full allocation accounting and explicit restoration boundaries address these
specific failure modes.

## Done when and headless acceptance

For this design slice, the demo command exits zero, prints the counts above,
and leaves a reviewable recommendation with accepted decisions separate from
proposals. The source gap and unexecuted native/learning work stay explicit.

For the next bounded prototype slice, propose
`uv run --extra dev python -m manabot.verify.representation --out .runs/representation/proof`
(**not implemented by kickoff**) as one headless command. It must create a
versioned receipt binding source/native bytes, fixture/capture hashes, schema,
catalog, seed and model config; count all fixtures and failures; and exit nonzero
on lost distinctions, viewer leaks, broken joins or budget failures. Its central
scenario is: construct source-dependent effect pairs at native authority, capture
both viewers, encode through native scalar/vector and Python paths, render text,
join to model actions, apply the same legal source-removal Command and observe the
predicted divergence. A second case retains the effect past history eviction.
No browser or human step is needed to establish these source-to-consumer facts.

Reuse these checks at that slice's gate:

```sh
uv run --extra dev pytest tests/verify/test_continuation_input.py tests/verify/test_observation_parity.py tests/semantic/test_text_observation.py tests/model/test_semantic_cards.py -q
```

Rust changes require debug `cargo test` plus fmt/clippy as in
`docs/development.md#validation-by-change-area`, the pinned root native rebuild,
and native/Python parity before interpreting a result. A production input/shape
cutover additionally requires `tests/model/`, `tests/training/test_active_training.py`
and the AGENTS.md current-world arena/monitoring suite, with all required extras.
Preserve original checkpoint defaults/omissions and prove incompatible schemas
are rejected. No world is relabeled, no evidence is regenerated to conceal loss,
and experiment result bytes go through the existing artifact-store procedure.
Unavailable native tooling is reported; capable CI owns the missing checks.

**Check result (2026-11-03):** `uv run --no-project --python 3.12 scratch/representation_probe.py` — PASS; 70→7 payloads with 70 memberships, exact roundtrip and counterexamples; native paired replays, neural costs and learning quality not tested.

### First CUDA canary outcome and bounded revision

Source `a8c1cbbff182`, excluded seed 73200, all three arms completed one CUDA
update and exported admitted checkpoints. Rows / typed_pool / typed_gru used
11.981 / 38.354 / 48.257 seconds of collection plus learning, with
11,133 / 11,427 / 11,427 learner decisions and 64 complete games each; no
numerical failure was recorded. These are one-update feasibility measurements,
not scores or a speed distribution. The first rental
`representation-5da5f4-6168654d2c3f` is confirmed deleted at 02:35:17 UTC;
SDK full-window estimate including setup/idle is $0.07083924. Its receipts are
`.runs/representation/canary/<arm>/run.json`. A diagnostic listing call after
successful termination used the wrong argument; termination and receipt fetch
both completed, and `etude machines --all` independently confirms deletion.

The unoptimized candidates project beyond the initial 45-minute horizon.
Source inspection identified per-sample CUDA mask/index synchronizations even
for observations without facts/frames. The revised implementation routes only
active samples once per batch; equations, parameter layout, memberships and
ordering stay the same. Focused native/model/action checks passed (20). No
scientific seed has run, so the training horizon remains unselected.

Second excluded canary: same seed, three arms, update count and watchdogs, source
at the next commit; cohort `representation-5da5f4-canary-batched-empty`, **$0.60
maximum** with one machine. First settled rental plus this entire reservation is
$0.67084, below the existing $0.90 canary envelope. Prediction: both candidate
collection+learning times below their first canary; no numerical or export
regression. Retain all results even if the prediction fails. This is a distinct
implementation calibration, not a retry or replacement of first-canary evidence.
The source-preparation failure `b65b1912e41a` remains recorded separately at $0.

Second-canary prelaunch amendment (2026-11-04): L4 availability is now zero;
the quote allocated nothing. RTX 4000 Ada is available at $0.28/hour. Create
`representation-5da5f4-ada` with the same 6-vCPU/48-GB host minimum, 50-GB disk,
pinned image, 900-second startup and 300-second delivery, ceiling $0.30/hour,
max one. Keep the same excluded recipe and $0.60 cohort cap. The hardware
change confounds before/after speed attribution; use the result only to
calibrate the final hardware/horizon, not to claim the sparse-routing speedup.

The Ada offer disappeared during launch admission `5c8cef228edb`; no machine
or run was created. A fresh catalog offers RTX 3090 at $0.50/hour (LOW), A40
at $0.59 and RTX 4090 at $0.89. Use the cheapest available, RTX 3090, through
new Task-local `representation-5da5f4-3090` with the same safeguards and a
$0.50/hour ceiling. Same excluded calibration and $0.60 cohort maximum; no
scientific protocol change or run. If this offer also disappears, preserve the
availability failures without claiming training results.
