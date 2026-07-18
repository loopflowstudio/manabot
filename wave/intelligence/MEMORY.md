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
- The first supervised student conditions policy and value on the canonical
  compatible-deal prior restricted by curriculum queries. Actual hidden truth
  supervises the belief head separately; policy is not trained as a clairvoyant
  one-world model.
- The high-priority order is: authoritative `PlanningProblem`, conditional
  teacher/result, conditional trajectories/shards/student, supervised belief
  head plus INT-9 adaptation, immutable self-play populations plus INT-6, and
  conditional Study evidence.

## Results-first reprioritization (2026-07-18)

- The convergence push landed every planned instrument — world/query kernel,
  exact-range tracker and player, conditional determinized PUCT with paired
  sampling, versioned advice, conditional shards, visit teacher, world-pinned
  arena — with zero committed production results across all of them.
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
  The supervised belief head, wider content pools, factorized-marginal belief
  degradation, and new planner families wait for those artifacts.
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

- Decide the first belief-head representation—fixed categorical exact support
  or structured hypothesis scorer—without changing the normalized
  `BeliefState` contract.
- Decide whether history reaches policy/value only through belief or also
  through a separate agent-memory path; match protocol remains memory-free.
- A plausible Transformer or graph/tree encoder must preserve structure without
  destroying the inference throughput needed for rollout-heavy training.
- Real dynamic binding—joining static program roles to runtime objects and
  offers—is more important than another static classification result.
- Early arena results are useful for iteration but remain vulnerable to weak
  opponents, one-seed variance, and hidden-information mistakes.
- Conditional search can improve labels while making data expensive or
  information-set inconsistent; belief calibration, strategy quality, cost,
  and planner honesty must be measured separately in the running teacher.
