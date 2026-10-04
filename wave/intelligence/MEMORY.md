# Intelligence memory

## Operating principle

> Lead with building, not burden of proof.

The research loop is prototype → measurement → surprise or confound → focused
diagnostic → revised prototype. Every primary Project must yield a runnable
agent, teacher, search system, or training loop. Katas and formal probes have a
real place when a working system produces an ambiguous result, but are not a
gate before integration.

## Durable evidence

- The order-invariant semantic encoder is mathematically unable to distinguish
  equal-token programs and remains at 50% on the structural suite.
- The first relational semantic encoder proved that explicit structure matters:
  it reached 82.1% overall and 100% on order and hierarchy. Its throughput and
  trainability were unacceptable.
- The follow-up discriminator ruled out “just train longer” for that encoder
  family and ended in `KILL_REDESIGN structural_capacity`. These results kill
  bag pooling and the first relational-pooling design; they do not require more
  static katas before a plausible semantic policy can be built.
- The structured command prototype handled 35 target choices and 64 attacker
  declarations with zero illegal outputs or trace mismatches at roughly a 4%
  game-throughput cost. Structured decoding is ready to be used in a learned
  prototype.
- Viewer-safe semantic program projection is not currently the bottleneck:
  real selected-match observations project and batch at tens of thousands per
  second with explicit ragged structure and no silent truncation.
- Teacher-0 established a runnable data → policy/value student → arena path.
  Its 512-game immutable snapshot trained both arms in 8.63 minutes; joint
  value supervision materially improved value calibration without reducing
  batch throughput. It is flat Monte Carlo evidence, not MCTS strength.
- Rust vector stepping and zero-copy observation buffers moved environment-only
  throughput from roughly 24k to 183k SPS at 16 environments. With inference
  enabled, model inference consumed 97% of step time. Model layout is now a
  first-order systems question.

## Decisions

- Preserve the structural katas as regression tests and diagnostic fixtures.
  Do not extend their static proof ladder without an observed prototype
  ambiguity.
- The next semantic experiment is an end-to-end policy over real engine state,
  typed programs, runtime bindings, and `InteractionOffer` values, using a
  plausible structural encoder and the shipped structured decoder.
- Put ablations inside runnable prototypes: identity versus semantics,
  structured versus legacy decoding, intact versus shuffled structure, and
  policy-only versus search augmentation.
- Search-teacher work and semantic-policy work can proceed in parallel. Neither
  is permission for the other; their eventual integration is another runnable
  prototype.
- Information-set honesty is part of the executable system, not an analysis
  cleanup. Default training, evaluation, and Study evidence use only the acting
  viewer's historical information.

## Architecture convergence (accepted 2026-07-17)

- managym owns the authoritative match, semantic Commands, canonical viewer
  Observation stream, replay/forks, possible-world meaning, typed `WorldQuery`
  grammar, compatible-deal measure, and materialization. Intelligence must not
  create parallel meanings for hands or actions.
- manabot owns agent memory, conditional priors and learned beliefs over the
  managym world domain, planning, policy/value learning, teacher evidence,
  datasets, checkpoints, opponents, arena evaluation, and Study evidence.
- The product proof is a complete play distribution changing under typed
  conditions such as `Has(Bolt)` without revealing actual truth. A
  `ConditionalStrategyResult`, not one best action, is the primary result.
- The accepted student architecture conditions policy and value on the canonical
  compatible-deal prior restricted by curriculum queries. Actual hidden truth
  supervises the belief head separately; policy is not trained as a clairvoyant
  one-world model.
- The original delivery order was: authoritative `PlanningProblem`, conditional
  teacher/result, conditional trajectories/shards/student, supervised belief
  head plus INT-9 adaptation, immutable self-play populations plus INT-6, and
  conditional Study evidence.

## Results-first reprioritization (2026-07-18)

- The convergence push was described as landing every planned instrument:
  world/query kernel, exact-range tracker/player, conditional search, advice,
  shards, visit teacher, and arena. That description missed an integration gap:
  temporal beliefs fed the specialized search player, root beliefs fed search,
  and positional condition tags fed the neural student. The general neural
  agent still lacked a semantic belief lifecycle. The architecture itself had
  not rejected that lifecycle.
- Measured basis: the shipped belief comparison moves the advice policy by at
  most 0.125 and never flips the top action (`top_action_changed` 0.0 in every
  retained condition); INT-7 value heads improved held-out Brier while
  weakening every complete player; INT-8 killed one-seed learned priors inside
  PUCT (paired score −0.15 chosen / −0.10 visit vs uniform); the INT-4
  production harness has never run (frozen Teacher-0 control bytes absent);
  the arena has anchors and one challenger but no rating run.
- Decision: the unit of progress is a frozen result on an existing instrument.
  The R1–R4 results ladder in `docs/plans/results-first-roadmap.md` supersedes
  the I1–I6 build ordering.
  At that time the supervised belief head was deferred with wider content and
  new planners. The belief-forming branch supersedes the head deferral and
  teacher-first integration order; it preserves the results' evidence gates.
- Near-term strongest-player hypothesis: belief-weighted, uniform-prior
  determinized PUCT at a larger declared budget. Learned components re-enter
  only through arena admission.

## Ownership boundaries

- **Rules:** authoritative semantics, state, legal offers and commands,
  viewer-safe projection, exact forks.
- **Intelligence:** learned policy, search, training data, opponents,
  evaluation, policy/search evidence.
- **Study:** human-facing retry, reveal, comparison, branching, and research
  consent.

## Open tensions

- Exact enumeration retains correlations but must meet real support-size and
  latency budgets. A later autoregressive joint scorer or correlation-aware
  policy view must preserve world/query semantics and frozen evidence.
- An ordered history encoder and arbitrary mid-game attachment remain separate
  changes: the current learned updater is order-invariant and consumes typed
  commitments collected from the initial root. Additional public memory needs
  an explicit intervention-safe contract; no private belief bypass is allowed.
- A plausible Transformer or graph/tree encoder must preserve structure without
  destroying the inference throughput needed for rollout-heavy training.
- Real dynamic binding—joining static program roles to runtime objects and
  offers—is more important than another static classification result.
- Early arena results are useful for iteration but remain vulnerable to weak
  opponents, one-seed variance, and hidden-information mistakes.
- Conditional search can improve labels while making data expensive or
  information-set inconsistent; belief calibration, strategy quality, cost,
  and planner honesty must be measured separately in the running teacher.

## Belief-forming branch reconciliation (2026-09-24)

The branch supplies `Manabot.decide`, `evaluate_under_belief`, `ManabotPlayer`,
a reference compatible-deal updater, schema-bound marginal encoding, and a
bounded learned exact-world updater. The configured `checkpoint` player
selects belief capability from serialized Agent fields. Observation-only
checkpoints still exist; this is not evidence that every training or serving
path now forms beliefs. Belief-enabled play generates a belief on every
ordinary decision, and evaluating the same belief explicitly yields identical
policy/value bytes without replacing autonomous memory.

The three semantic layers are managym's possible worlds, manabot's normalized
weights over those worlds, and managym's typed predicates. Conditioning is
`b_Q(w) = b(w) * 1[Q(w)] / P_b(Q)`; empty mass fails explicitly. Full world
weights preserve correlations for queries and search. The policy projection
contains canonical owner/zone/CardDefId count marginals, validity, entropy,
and effective support. Equal distributions encode equally regardless of query
spelling or provenance; marginals cannot authoritatively answer conjunctions.
Teacher policy/value labels are separate from a supplied belief.

The implementation shares the visible-state attention/decision core, with a
schema-bound card-definition embedding for belief rows. The existing visible
card encoder has numeric features but no CardDefId/name identity to share.
The learned updater has its own embeddings and head; full semantic parameter
sharing remains an architectural target, not an implemented claim. Belief and
strategy losses remain separate, and no private recurrent activation crosses
the intervention boundary. Joint gradients require a measured future choice.

Belief-enabled checkpoints require both `belief_schema_identity` and
`belief_content_manifest_identity`. The schema binds world/content identities,
count buckets, and every ordered owner/zone/CardDefId/name row; equal dimensions
are insufficient. Runtime validates it before inference. PPO and BC share
`belief_checkpoint_fields` admission checks. Positional condition Agent fields,
neutral fallback, aliases, and state-dict ports are removed; old experiment
bytes are retained as evidence, not made loadable through semantic adaptation.
Loader changes intentionally trip INT-8 loader-source and arena drift guards;
new experiments must freeze current-ABI models and contracts.
The same source binding now rejects serving the historical INT-15 flip fixture
through `/api/advice` with `advisor_artifact_mismatch`. Its frozen positive
response remains an offline reference; this branch does not regenerate it or
restore a production conditional-advice result.
The default live-posterior resolver also rejects its retained INT-7 checkpoint
because that artifact declares the removed `max_conditions=0` field. The ed2
address/replay path survives, but successful default posterior resolution now
needs a newly registered current-ABI likelihood checkpoint. Frozen checkpoint
bytes and their admission rules remain unchanged.

### Learned world correction and its measured boundary

managym supplies the current exact physical-deal measure `p0`: a hand count
vector has weight `product choose(c_i, h_i)`, normalized over compatible hands.
The current world domain represents opponent hand counts and the complementary
library multiset, not library order or arbitrary hidden state. Candidate index
meaning is local to each decision. Training packs `world_counts[sum W, C]`,
`log_p0[sum W]`, candidate-to-decision indexes, offsets, and supervision-only
local target indexes. The scorer normalizes `log_p0 + s_theta(world, history)`
within each decision and trains whole-world NLL. Materialized truth is a sample
label, never an inference input or a reason to collapse every posterior.
No smoothing or entropy bonus replaces the proper joint scoring rule.

Without typed commitments the deployed updater returns `p0` exactly. managym
has no mulligan/Keep event; `ScryKeep` is unrelated. Current typed history uses
viewer-relative actor, public commitment kind (pass, cast, play-land, discard,
decline-discard), and canonical public card name. Kind set and vocabulary are
schema-bound. Event hashes, command identities, policy versions, and episode
receipts are provenance only. A standalone mid-game Observation has opaque
earlier event identities, not decoded commitments; arbitrary attachment needs
a managym semantic-history replay projection. Do not infer meaning from hashes.
The v2 updater sum-pools events with square-root count normalization: it learns
actor/kind/card/multiplicity sensitivity but is explicitly invariant to order.

`uv run manabot belief-learn-demo` samples 160 opening worlds with replacement
from exact `p0`. One frozen opponent plays its most-held land (canonical-name
tie break), otherwise passes, after refreshing its legal offers. Each deal
contributes one real post-transition label; whole episodes split 128/32. The
held-out test checks joint NLL improvement >0.1 nat, improved inclusion Brier
and ECE, 90% credible-set coverage >=0.8, and broad posterior support. The same
scripted population appears in both arms: this proves population supervision,
not opponent transfer, order sensitivity, multi-seed calibration, or strength.
Future splits must also hold out opponent versions; the updater must not
require the acting policy at inference. Known-policy Bayes remains diagnostic.

### What the branch proves and what remains

On 2026-09-24 the focused belief/state/runtime/learning and BC round-trip suite
passed 27 tests in 20.74 s. It exercises the real keystone and population demos,
exact generated/supplied output equality, semantic query effects, hidden-world
swap invariance, and checkpoint binding failures. The prior compression check
also preserved four reference/learned model, belief, and receipt identities
and all existing checkpoint rejection messages. Reference receipt history
ranges count opaque events; learned ranges count typed semantic commitments.
Those coordinates deliberately differ even with one receipt constructor.

The next research proof remains paired conditional teacher measurements at
real roots under a declared budget, then multi-seed policy-only distillation
on held-out roots with baseline-belief and shuffled-belief controls. Record
policy distance, stable top-action flips, decision type, label cost, and source
identities before training; absent stable signal, retain
`KILL_NO_CONDITIONAL_TEACHER_SIGNAL`. A nonzero random-network policy delta or
a curated teacher fixture cannot establish a trained student's strategic flip.
Learned-belief continuation needs held-out NLL, query calibration and coverage,
zero incompatible mass, pre-event equality to p0, hidden-truth invariance, and
latency; then evaluate the full autonomous loop at matched compute.

The historical evidence explains the gap: beliefs were proposed in July 9's
PBS design (`dd66516`), diagnostic-gated on July 10 (`ede07b3`), split from
teacher work on July 15 (`c562fe3`), and accepted as policy/value inputs on
July 17 (`1f79603`, `04af5a1`). INT-14 (`8ca40e2`) used positional tags while
citing a superseded dormant-beliefs premise; INT-9 (`3efd25b`) supplied a
separate exact-range player. July 18's results-first closure (`823b0a3`) obscured
the missing composition. INT-15 later froze a post-hoc curated flip, INT-17
retained a systems failure (quadratic support enumeration, no calibration
curves), and INT-18's rating omitted the exact-range comparison. These are
separate claim boundaries, irrespective of a task's completion status.

### Planning reconciliation blocked by repository migration

`lf pm show --wave intelligence` on 2026-09-24 refused refresh because legacy
`pm.provider`/`pm.linear_team` bindings require repository-wide migration owned
by PRD-44. `--no-sync` exposed an 11-day-old cache only. No Linear definitions,
KRs, or task states were changed; that cache is not live authority. Refresh
through `lf pm` after the migration before applying any reconciliation.

The cached Search Teacher & Student Arena definition needs the explicit
agent boundary and held-out conditional policy-only action-change proof above;
Belief-Aware Play's blanket learned-head deferral no longer describes this
branch. Its live-advice, multi-game calibration, and arena strength KRs remain
unproven by the bounded demo. ETU-34 names only architecture mapping and needs
its actual delivered scope reconciled after branch acceptance; ETU-31's
production multi-seed teacher comparison is not completed here. Check existing
work before filing the conditional atlas/distillation continuation or broader
belief calibration/transfer work, to avoid duplicates. The semantic-history
replay and shared card vocabulary gaps above are narrower provider follow-ups,
not grounds for inventing parallel Rules meaning.

ETU-21 and Game's ETU-14 already cover the live-advice continuation. The `ed2`
address and posterior resolver now exist, so the claim that a live address is
wholly missing is stale. `/api/advice` still calls the fixture provider, so the
end-to-end finish line remains open. The carried design and remaining release
gates are in [the live-advice plan](../../docs/plans/live-belief-advice.md).

## Corrected-world training binding (2026-09-29)

`scripts/train_challenger.py` remains the receipt-bound training-to-demo runner.
The regime executor described below adds staged training; it does not yet
replace that demo admission path. It builds
teacher games with `MatchHypers.authored`, so sideboards are present and Learn
offers a Lesson; deck constants alone give the old setup. The play server
rejects a candidate whose content manifest was taken without sideboards. The
runner records rules-runtime, content, setup, Lesson-pool and
observation/action ABI digests, requires admission (engine legal-offer count
equals encoded rows at every decision, at least one Learn decision offering a
Lesson, search cap hits at most 1%), and completes only after the candidate
plays both deck assignments through `configured_opponent` and `GameSession`.
`train_search_supervised` now takes `observation_hypers`.

Two eight-game executions completed on 2026-09-29 in 200 s and 139 s with zero
omitted choices over 993 and 824 decisions (18 and 17 Learn decisions). In
both, held-out policy KL stayed at its untrained value (0.0218 → 0.0222,
0.0197 → 0.0212): PUCT-64×4 visit targets were close to uniform (top share 47%
versus 39% uniform) and each run took about 56 optimizer steps. Which of those
limits learning is untested. Treat these checkpoints as pipeline proof only.
Details and unreviewed decisions are in the
[2026-09-29 record](../../docs/evidence/corrected-world-training-2026-09-29.md).
That record predates the current w4 declaration. New runs resolve the native
world and exact ABI identities; ETU-75 owns the shared checkpoint/setup binding.


## Training regime reconciliation (2026-10-04)

Jack Heart split training work into ETU-89 (regime/run infrastructure), ETU-90
(RL correctness/treatments), and ETU-91 (studies and analysis), then requested
three dedicated workers. The study recipes and proposed scientific allocations
are in [the comparison protocol](../../experiments/training-regimes.md) and
[the ablation protocol](../../experiments/ataraxos-mtg-ablations.md).
Jack Heart subsequently authorized scientific execution after integration and
landing within 168 active laptop hours total, including calibration and the
ablation screen. The root session owns the single launch; workers must not
start competing training. Counts and allocations must be frozen from measured
integrated CPU calibration before scoring; smoke does not freeze them.

The current chapter remains Trained Challengers; older conditional-teacher
priorities are research background, not authorization for a new costly cycle.
Infrastructure workers run only bounded proof; no paid compute is authorized.

`TrainingRegime` and `TrainingRun` now execute the existing search-supervised
and self-play trainers through `manabot train --regime`. VerifyStore owns status;
JSON is an export. Same-run supervised rounds retain Adam and immutable
whole-game split membership. Self-play continuation retains the latest live
collector at an exact update boundary; it cannot branch from old collector
state. Checkpoints do not promise process resume or byte-identical training.
Resolved recipes, source/runtime identities, artifacts and phase costs are the
reproducibility contract. Schedules use whole-run elapsed budget, not stage age.

The collector defect was a contract mismatch: transition-end flags were passed
to episode-start GAE, and banked rows made next-state bootstrap stale. Corrected
end-marker GAE and paused streams address those mechanisms. The trace scale
`1/(1-gamma*lambda)` is not a hard learning horizon. Historical PPO/search
comparisons do not settle which repaired recipe wins at matched current cost.

ETU-89 now integrates ETU-75's mandatory checkpoint world/setup contract,
semantic tensor propagation and ETU-90's complete-state EMA helper. A retained
two-stage execution completed 13 games and 1,024 learner transitions with four
ordinary raw/EMA reloads on the semantic ABI. ETU-90 owns EMA complete-state
correctness; ETU-91 owns final replayed arena/notebook evidence after integration. Study smoke success cannot
close infrastructure, RL correctness, strength or human-play acceptance for
another Task. The implementation contract is in
[training regimes](../../docs/training-regimes.md); scientific proposals and
limits remain in the [study protocol](../../experiments/training-regimes.md).

Compare study scores only over overlapping observed cost ranges, using the last
checkpoint available at each cutoff. Search generation remains on the cost axis;
no overlap means equal-cost comparison is unavailable. One seed and a four-leg
deal block prove workflow, not method-level uncertainty. ETU-91 owns final
replayed study and offline-regeneration evidence on the integrated code.

ETU-91 now has integrated semantic study proof: final learning-speed and
five-arm smokes completed in 91 and 204 seconds with 24 and 72 exact-replayed
games, two checkpoints per arm, and unchanged offline-regenerated metrics and
reports. Both use the executor's cumulative checkpoint clock and have observed
cost overlap. Earlier inherited-ABI evidence remains preliminary. A later
five-arm attempt failed when a merge exposed conflict markers to a child
process; that attempt remains retained separately. Never edit imported source
while a multiprocessing measurement is running.

Scientific plan generation scales measured updates and teacher games, retains
cumulative fitting, reserves 168 hours across calibration, both three-seed
studies, evaluation and recovery, and checks projected storage against a 4 GiB
reserve. The three fixed anchors, untouched endpoint deals and full nine-cell
main comparison remain explicit. Count extrapolation is not long-run timing
proof. No scientific study ran here; root owns calibration and the unique launch.
Post-training evaluation resume retains failed cells without replacement, uses
remaining original time and cannot turn an incomplete cohort into a strength
claim. Training checkpoints do not support process resume.

Jack Heart requested keeping ETU-91 and its checkout open to preserve ignored
`.runs` evidence. Detailed attempts, commands and evidence limits live in the
[study protocol](../../experiments/training-regimes.md). This evidence does not
satisfy ETU-82's demo-run contract, ETU-85's improvement comparison, S1–S5
competence, or the chapter's human-challenger outcome.

## Direct self-play treatment contracts (2026-10-04)

Self-play transitions use end-of-transition terminal flags. Stock PPO's
start-of-episode GAE convention cannot consume them unchanged. Collection must
pause fast streams at their exact next learner observation, preserve every
observation tensor and recompute actions under the next collection policy;
banking surplus transitions across updates breaks that contract. Collection
KL uses the saved full legal behavior distribution, not reconstructed updated
weights. Independent policy/value estimators, filtering, reference choices and
schedules are runnable treatments, not exact Ataraxos reproduction.

EMA is an evaluation artifact with a collect/update-iteration clock, including
empty-filter skips. Its helper averages parameters and copies buffers without
changing learner/behavior weights. Local ETU-90 proofs exercise an empty-filter
continuation: learner weights stay fixed while the evaluation average advances.
Collector match metadata passes through the Trainer env shim for ordinary
checkpoint admission.

ETU-90 independently validated the integrated semantic ABI on 2026-10-04 after
syncing published parent `9a1b90df`. The retained normal and empty-filter runs
in `.runs/etu90-semantic-final` each completed 14 games and 1,024 learner
transitions across two stages. All eight raw/EMA artifacts passed the ordinary
loader with semantic inputs and authored sideboards. Empty-filter continuation
retained learner weights with zero optimizer exposures while EMA advanced;
collection, learning and export costs remained recorded. The affected Python
suite passed 65 tests (one notebook dependency skip), and six native debug
vector tests passed after rebuilding the extension. These are current-ABI
workflow and treatment-correctness proofs, not strength or human-play results.
ETU-91 retains ownership of final replayed study/notebook evidence.

## Compound decision implementation (2026-10-04)

Jack Heart authorized ETU-94's bounded implementation and landing separately
from the frozen ETU-91 campaign. `train_compound` now connects a trainable
recurrent legal-offer decoder, complete-game collection, explicit sequential or
grouped credit, outcome or bootstrapped estimators, and ordinary world-bound
checkpoint reload/serving. The four-arm study uses the existing arena/report
path. The sequential arm is a conditional-factor credit control with the same
decoder, not the historical flat policy. No scientific allocation is inherited
from ETU-91; the campaign plan generator rejects that reuse.

Canonical DecisionFrames remain action-aligned. Native `compound_offers` and
`compound_commands_json` reuse the existing structured bridge to lower one
complete declaration on an exact fork into revision-bound Commands. Python does
not reconstruct combat legality. Atomic attackers and eligible single-target
casts group; blockers, payments and other newly published observations remain
separate. Broader atomic blocker/payment support needs native authority work.
Ordinary Etude and arena consumers drain the sampled suffix without resampling;
stale/interrupted suffixes fail closed.

The useful estimator boundary is explicit: terminal reward per seat, no update
within a game/declaration, grouped joint log probability versus sequential
conditional factors, detached targets/behavior, and gamma=1 in comparison
recipes. Trace/discount clocks otherwise change with grouping. Forced decoder
factors and native optionless auto-resolution have separate counters. Summed
conditional reverse-KL terms at retained prefixes are sampled-prefix
regularizers, not exact joint reverse KL. Prefix values are scalar and the
policy uses a GRU; neither is an exact Ataraxos setup-network reproduction.

Focused evidence includes normalized joint distributions, score-function and
finite-difference gradients, 65-attacker/35-target native parity, payment/blocker
boundaries, terminal and interrupted-game credit, hidden-world swap invariance,
and ordinary learned checkpoint reload. Native debug comparison must account
for canonical execution consuming its observation-event queue; comparing an
undrained raw atomic bridge to a drained Command stream compares different
ownership points. No physics change or frozen evidence rewrite was needed.

Two retained one-thread workflow executions completed in 268 and 224 seconds;
the latter followed integration with ETU-92 and ETU-96. Each trained four arms
for two games apiece, exported eight admitted checkpoints, and exact-replayed
56 arena games with no failed cells. Offline reports/metrics regenerated
unchanged. These one-seed receipts are retained under this Task checkout's
ignored `.runs/etu94-compound-smoke-{1,final}`; they are not method uncertainty
or authorization for further scoring.

Scientific outcome-versus-bootstrap and sequential-versus-grouped improvement
remain unmeasured. Multi-seed, held-out, equal-cost complete-game scoring needs
its own frozen protocol and budget. Candidate runtime-object representation in
set-valued choices is still limited to public labels plus pooled viewer state;
this is runnable mechanism evidence, not strategic-strength or challenger proof.
The [compound contract](../../docs/training-regimes.md#compound-decisions) and
[follow-up protocol](../../experiments/training-regime-followups.md) own details.

## Ataraxos source correction and move recipe (2026-10-04)

ETU-92 checked the final Nature paper and actual supplement S3.4, equations
(5)–(6), Table S7. The move-learning loss explicitly uses clipped per-action
importance ratios plus reverse KL to collection and magnet policies. A claim
that PPO clipping disqualifies this mechanism is incorrect; scheduled entropy
alone still does not establish fidelity. This is not DeepNash R-NaD.

The selectable `ataraxos_move` recipe adds categorical outcome lambda targets,
unnormalized advantages, inclusive quantile filtering, timestep-grouped updates
and iteration schedules. Scalar MSE remains a separate representation ablation;
ordinary PPO controls retain their elapsed-budget schedules. Checkpoint metadata
binds scalar versus categorical heads; serving still returns signed expected
value. Behavior likelihoods and outcome distributions are recorded at collection,
not reconstructed after updates. MTG same-viewer transitions and action-type
magnet, CPU architecture and boundary conventions are declared adaptations in
[the fidelity table](../../docs/ataraxos.md).

A bounded one-thread categorical execution completed five games and 512 learner
transitions with 46 optimizer exposures in 2.15 seconds, exporting four admitted
raw/EMA artifacts. Analytic mixed-policy gradients, categorical terminal/reset
boundaries and real reload checks establish mechanism/wiring evidence only.
No matched-cost multi-seed comparison or challenger strength is established;
scientific scoring needs its separate protocol and budget. ETU-91's frozen
campaign and retained checkout were not altered or restarted.

## Frozen-policy sampler mechanism (ETU-96, 2026-10-04)

`collect_belief` and `train_belief` extend TrainingRegime with explicit frozen
raw/EMA policy → whole-game private dataset → sampler dependencies. They do not
mutate ETU-91 or feed a belief loss back into the behavior policy. VerifyStore
retains attempts/costs; sampler admission binds exact bytes, policy, dataset,
schema and world. Ordinary last-complete-raw policy selection is unchanged.

managym now exposes hand constraints without enumerating the support, using the
same pool/minima source as exact possible worlds. Autoregressive count masks
preserve hand size, known minima and residual card capacity. The physical-deal
baseline removes known cards before dealing unknown slots. Joint NLL uses truth
only as a label; inference sees public constraints and a lossy, order-invariant
commitment summary, not full board/ordered history. Out-of-roster generated
cards fail explicitly. This is a bounded mechanism, not a calibrated posterior
on every viewer information set.

Exact-tracker comparison and foreign/adversarial cohort measurement are exposed
as instruments. Synthetic checks do not replace independent-seed calibration,
real foreign-policy histories, history-dropout comparisons or matched-time
complete-game search evidence. These scientific comparisons remain open and
require their own frozen protocol/budget; no new costly run or paid compute was
started. Coordinate sampled-hand consumption with the post-RL search Task and
preserve the exact tracker as a tractable reference. Wider pools still require
new world-bound complete-loop measurements. See the
[sampler guide](../../docs/belief-sampler.md).

The bounded compound follow-up admits `train_compound` raw exports into the
same belief dependency chain and resets queued Commands at game boundaries.
Ataraxos raw/EMA already has end-to-end coverage. Compound EMA and mixing flat
policy training into compound recipes remain rejected. Both collector paths
enforce the saved observation capacity, even though standalone compound play
can expand action encoding. This is compatibility evidence, not wider-pool or
scientific acceptance; ETU-96 remains open for those comparisons.

## Frozen-policy local update boundary (ETU-95, 2026-10-04)

Jack Heart authorized implementation and landing while leaving scientific
acceptance open and ETU-91 untouched. `collect_local_update` freezes raw/EMA
policy bytes, labels frozen-policy trajectories with viewer-safe rollout/value
estimates, and retains the two-KL local update from final Ataraxos S3.7 (7)–(8).
Policy, likelihood, rollout and value identities agree; arena opponents need not
match that model. Scalar critic units must be explicitly signed outcomes;
old supervised win-logit checkpoints cannot silently enter this search.

The bounded compatible-prior recipe completed four tiny-pool games and 318
labeled decisions in 9.94 seconds, including hard/soft/allocation targets and a
second teacher/student round with cumulative immutable data. This is workflow
proof, not selected-matchup strength, compounding or a calibrated posterior.
Allocation frequencies from balanced flat rollouts are not PUCT visits.

Real-game verification exposed a provider gap: the retained tracker cannot form
an exact full-history posterior when opponent combat/target choices have no
public likelihood identity; ordinary discard refresh is also unsupported. The
new exact arm fails explicitly instead of inheriting the tracker's partial-history
claim. Rules must supply these capabilities before full-game exact-versus-prior
scoring. Source roots plus belief history remain required for rollout replay;
private receipts are not a replacement for GameSession/Trace authority.

The arena lifecycle retains private targets and actual command replay, including
commands preceding failed belief updates. Etude has a typed viewer-safe projection,
not a new live advisor registration. Genuine same-root PUCT-visit controls,
archived-root relabeling, learned-sampler integration, independent-seed strength
and human-play acceptance remain open. See [local-search contracts](../../docs/local-policy-search.md)
and the separately budgeted [follow-up protocol](../../experiments/training-regime-followups.md).

## Omitted learning controls (ETU-93, 2026-10-04)

Jack Heart authorized mechanism implementation and landing with bounded proof,
not another scientific allocation. Independent resolved contrasts now cover
discount, policy trace, paper estimator settings, reference, collection KL,
filter threshold/ties/scope, raw/EMA evaluation and behavior, separate schedule
decay and conditional leave-one-out follow-ups. They use the existing regime
executor and arena/report path; ETU-91's live checkout and frozen cohort remain
untouched. The [separate protocol](../../experiments/ataraxos-omitted-controls.md)
keeps every scientific technique disposition unresolved.

EMA behavior uses averaged weights for both collection seats, saved likelihoods
and paused-tail bootstrap; raw weights receive gradients, then EMA advances.
Evaluation EMA alone does not test this mechanism. Actor-only filtering trains
the critic on all rows even with no retained actor rows; shared encoder changes
can still alter policy outputs. Raw/EMA variants remain correlated per seed.
Paper schedules retain their iteration clock in executor receipts.

Selection diagnostics cross action type with observed terminal distance and
retain raw/selected signed advantage quartiles and target-residual means.
Unfinished batch tails are censored, not distant. These residuals use lambda
targets; they cannot prove independent terminal-outcome critic error for
unfinished games. Whole-game held-out analysis and multi-seed matched-cost
scoring remain required. One bounded EMA-behavior comparison completed 48 exact
replayed games in 117 seconds; this proves the integrated workflow, not strength,
demo admission, human completion or a technique retention decision.

The main integration preserves frozen-opponent and compound studies. Compound
training rejects the new quantile/actor-only knobs rather than silently ignoring
them; ETU-93's executable contrasts target ordinary self-play. The merged
omitted-control/frozen-opponent/study paths passed 39 focused checks.

## Learned sampler to local search (ETU-95 continuation, 2026-10-04)

Jack Heart requested the complete ETU-96 sampler → local teacher → distillation
connection without support enumeration. Native direct count materialization
shares indexed placement while validating current observation, pool, public
minima, hand size and viewer-root/offer invariance. Learned play maintains only
viewer history and never starts an exact tracker. TrainingRegime admits the
sampler's exact bytes/dataset/schema/world/generating-policy identities and
requires the same raw/EMA policy for labels, rollouts and signed leaf values.

Approximate learned-hand receipts retain sampled counts, probabilities, seeds
and branch tapes; they never fabricate canonical world indexes or full-support
weights. The current public commitment summary remains lossy and order-invariant;
`True` is supported, conditional learned queries fail explicitly. Compound and
belief-enabled rollout policies remain unsupported. These boundaries do not
complete the missing Rules public-commitment/exact-history work.

A two-label-game pilot reached distillation but failed because the fixed
whole-game held-out partition was empty after three belief games. The bounded
recipe now collects eight label games to reach that existing partition; it does
not redefine splits or discard the failed attempt. Current proof and limits live
in [local-search contracts](../../docs/local-policy-search.md#learned-joint-hand-search).
No scientific allocation, paid compute or ETU-91 change is authorized by this
mechanism proof; ETU-95 remains open for scientific acceptance.

## Complete-loop calibration boundary (ETU-98, 2026-10-04)

PR207's bounded complete-state recovery remains intact. The separate
[calibration entry point](../../docs/training-calibration.md) reuses TrainingRun,
VerifyStore and the study arena for two CPU self-play checkpoints and eight
exact-replayed games. It distinguishes native microsteps, learner transitions
and optimizer sample exposures, retains failed attempts, records actual model
device/threads, and reports replay as a subset of arena cost. MPS/CUDA are
explicitly unsupported on this complete path; inference rates cannot substitute.
Host load/RSS observations do not correct contention or measure throttling.
The active ETU-91 campaign was not edited. Concurrent bounded checks prove
workflow only; uncontended multi-seed hardware calibration, clone/sampling
costs and conservative cohort projections remain open within the unchanged
168-hour campaign ceiling. No new scientific allocation or paid compute follows
from this command. Its costs must be included in campaign accounting.
