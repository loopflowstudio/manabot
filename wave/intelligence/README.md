# Intelligence

> Architecture priority updated 2026-07-17 after the interactive search and
> learning review. The top-down contract is
> [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md). Predecessor
> strategy/history remains in [03-search-era-notes.md](03-search-era-notes.md),
> the earlier beliefs exploration in [02-beliefs-design.md](02-beliefs-design.md),
> and the cycle log in [01-experiment-loop.md](01-experiment-loop.md).

## Thesis question

> **Can a manabot learn and plan strong strategies conditioned on explicit,
> viewer-safe beliefs, and make the strategic value of hidden-information
> assumptions legible at practical inference cost?**

The immediate proof is not a generic public-belief solver. It is a running
conditional teacher and student over the authoritative managym world: the same
position produces complete play distributions for the compatible-deal prior,
`Has(Bolt)`, `Lacks(Bolt)`, and other typed conditions, with exact provenance
and no hidden-truth leak. Information-set-consistent planning remains a major
research frontier after this belief/query/strategy boundary is executable and
measured.

## Standing constraints

- **Worlds** (`/WORLDS.md`): shape changes are versioned; measurements are
  never ported across worlds; baselines re-run per freeze.
- **Pacing**: a cycle closes when its result is integrated into
  `paper/understanding.md` at whiteboard-defensible level. One cycle in
  flight; un-integrated results queue.
- **Reproduction** (`experiments/repro/`): headline claims have end-to-end
  scripts; the suite re-runs on world freezes. A claim without a repro script
  is provisional.
- **Discipline** (`experiments/README.md`): predict-first-in-git, name the
  confound, mechanism over aggregates, numbers trace, protect the instrument.
- **Statistics**: the experimental unit is the *training seed*, not the
  evaluation game. Game-level CIs quantify eval noise of one checkpoint;
  claims about a *method* need independent seeds and cross-seed uncertainty.
- **Authority** ([docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)):
  managym owns Commands, Observations, replay, forks, possible-world meaning,
  and materialization. manabot owns memory, probability, beliefs, search,
  learning, opponents, and evaluation. Intelligence prototypes may use a
  temporary adapter but may not establish a competing world ontology.

## Metrics, in order of authority

1. **Exploitability** (approximate best response, standing per-checkpoint —
   promoted from one-off probe): the only strength number that cannot be
   gamed by opponent choice. In the microgame (below), exact NashConv.
2. **Competency scenarios** (`manabot/verify/competency.py`): mechanism-level
   capability. Capability claims cite these, never win rates alone.
3. **Matched wall-clock comparisons**: any "X beats Y" is at equal
   wall-clock or the imbalance is stated.
4. **The search ladder** — *demoted to a compute-calibrated baseline.* It
   anchors cost comparisons; it is not the objective. Measured reason: the
   ladder's own anchor dominates random opponents while scoring worse than
   random on delayed-control scenarios (exp-09).

## High-priority next tasks

These tasks supersede the old “diagnose, then treat” ordering. The existing D1
and D2 diagnostics remain valuable, but they no longer gate the foundational
conditional-belief architecture or the player-facing strategy comparison.

### I1 — Consume the authoritative planning root

Define the manabot `PlanningProblem` adapter around one canonical viewer
Observation/history, semantic legal offers, a normalized world distribution,
planner/evaluator identities, budget, and seed plan. Move determinized PUCT and
one teacher path behind it. Search must exact-fork and execute semantic Commands
through the selected managym branch API; raw `Game` access and action-index
identity do not cross the boundary.

This task can begin against the Rules R1/R2 compatibility adapter. It is done
only when one running teacher consumes the new managym authority and preserves
the current legality, replay, and per-world evidence checks.

### I2 — Ship the first conditional strategy teacher

Add `ConditionalWorldPrior` and `ConditionalStrategyResult` over the managym
world/query kernel. For one root, run paired `True`, `Has(Bolt)`, `Lacks(Bolt)`,
and `Q`/`Not(Q)` conditions with the same planner, evaluator, budget, and paired
seed plan. Return aligned complete semantic action distributions, per-action Q,
root value, condition mass/support, uncertainty, realized compute, and
provenance. Never query whether the actual hidden hand satisfies a condition.

The demo is the product-shaped result: the play distribution visibly changes
under at least one strategically meaningful condition, and exact replay
reconstructs every root and sampled-world receipt.

### I3 — Version conditional evidence and train the first student

Create `QuerySamplerSpec`, conditional teacher trajectories, and immutable
shards. Always include `True`; pair useful conditions and complements; sample
definition presence, land-count buckets, and stable semantic tags; reject
empty, redundant, nearly vacuous, and initially vanishingly rare conditions.
Rows bind full targets, belief/query identity, condition mass, search receipt,
played Command, source trajectory, and access-controlled actual-world target.

Train policy and value conditioned on the normalized canonical conditional
prior. Query text is provenance, not a direct model feature. Equivalent
queries inducing the same distribution must produce the same result. A
`CheckpointManifest` binds world, Observation/action and world-hypothesis
schemas, history strategy, targets, dataset, seed, exact bytes, and inference
profile.

### I4 — Land the supervised belief head and adapt INT-9

Map viewer Observation history to a normalized `BeliefState` over the same
managym hypothesis domain. Supervise it from actual hidden worlds retained in
private self-play audit data; never feed hidden truth at inference or train
policy only on one-hot truth. Adapt INT-9's exact range, public-action
likelihood, replay, and calibration work to this common contract instead of
landing a parallel hand ontology.

Measure log loss, calibration, support/normalization error, query-mass error,
and downstream conditional strategy quality separately. Then mix learned,
non-uniform beliefs into policy/value training so serving on the learned head
is not out of distribution.

### I5 — Make self-play populations and INT-6 the admission path

Replace mutable/dict opponent specs with immutable `PlayerRegistration` and
`OpponentPoolManifest` values. Record the selection policy, selected opponent,
and probability for every match. Preserve teacher mirrors and true self-play,
then add immutable champions, historical checkpoints, nearby-skill opponents,
search teachers, and exploiters only when the population justifies them.

Land INT-6's world-pinned paired-deal arena as the sole promotion authority.
Retain the full payoff matrix, competencies, calibration, latency/throughput,
regularized population rating, paired-deal uncertainty, connectivity, and
residuals. Development pairwise matches remain explicitly non-admission.

### I6 — Project conditional strategy into Study

Accept an exact managym historical Observation/decision plus player-selected
typed queries, resolve the exact model/planner artifacts, and emit attributable
baseline and conditional `DecisionEvidence`. Etude owns query construction and
explanation; manabot owns the distributions and uncertainty. Missing artifacts
or quantities remain typed unavailable. Raw sampled worlds and actual-query
truth never enter viewer-facing evidence.

## Diagnostics for the later planner decision

The architecture above makes belief error, continuation error, and
information-set inconsistency separately measurable. D1 and D2 choose the next
planner family after I1/I2 provide the common conditional interface.

### D1 — the 2×2 ablation (information × continuation)

> **Launch spec ready:** [04-2x2-diagnosis.md](04-2x2-diagnosis.md) — fully
> self-contained brief for dispatching this cycle from a fresh context.

On competency scenarios and micro-format matchups, cross:
{hidden information, full information} × {random continuation, oracle/exact
continuation}.

| outcome pattern | diagnosis | treatment |
|---|---|---|
| fails even at full-info + exact search | scenario/search implementation bug | fix it first |
| fails at full-info + random continuation | planning/rollout deficit | value or tree search over rollouts |
| full-info exact succeeds, hidden-info fails | information-set inconsistency (true fusion) | CFR-style / public-belief solving |
| hidden-info improves with oracle beliefs | belief estimation deficit | likelihood-weighted determinization |

This also settles the terminology debt: exp-09's S2-class failures are only
"strategy fusion" if the hidden-information column isolates them.

### D2 — the exactly solvable microgame

A Magic subgame small enough to solve: counterspell bait, held removal, one
represent/bluff decision; tiny decks, capped turns. Compute the equilibrium
and exact best responses (NashConv). Every planner we own gets an exact
exploitability number there before we trust its approximate one at scale.
Ten thousand approximate games against random are worth less than one game
where the equilibrium is known. The full engine is for scalability, not
ground truth.

### Then: one new planner treatment, chosen by D1/D2 — not before

Value/tree search for a planning deficit; likelihood-weighted determinization
for a belief deficit; CFR/public-belief solving (per
[02-beliefs-design.md](02-beliefs-design.md), with its Sokota-conditions
caveat) for genuine fusion. Implement the *simplest* indicated baseline,
compare per the metrics hierarchy, write the causal conclusion into
`paper/understanding.md`. That conclusion — not a ladder rung — is the unit
of progress.

## What is already known (one line each, repro-gated)

Search-256 with random rollouts is the unbeaten teacher (exp-02); distillation
beats matched-cost PPO (exp-03/07); pay-per-event shaping was harmful
(exp-04); the crank does not compound by default — label economics (exp-07);
the pilot cannot play delayed control (exp-09); scalar V is blind exactly in
interaction states (exp-10); the opponent installs the strategy, self-play
wins, the student is robust to a matched-budget exploiter (exp-11); per-seat
claims before the deal fix are suspect (exp-06).

## Exits (inherited, still armed)

Exit 2 (model-free game-theoretic pivot) remains live if the chosen treatment
fails its own gate; Exit 4 (wrap the platform paper) unchanged. Exit 1's
tripwire is superseded by D1 — diagnosis replaces the plateau proxy.

## Not here

Rules-authority implementation (owned by the active Rules wave), Etude UI
implementation (owned by Game/Study), cube optimization (north star, not now),
multiplayer/general-sum play, and hidden decklists.
