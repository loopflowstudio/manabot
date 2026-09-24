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
