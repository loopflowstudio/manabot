# Manabot research

Manabot's research program builds increasingly strong Magic: The Gathering
agents. The aim is agents that play real `managym` games, are measured
honestly, and are understood well enough that we know why they improve or
fail, repeated until a bounded superhuman claim is earned.

This document says what we are doing now, what we know, what we plan to do
later, and how we judge results. Preregistered predictions, results and
provenance live in [`experiments/`](../experiments/README.md); the system
design is in [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). Live task
status lives in Linear, not here.

## What we are doing now

### The training loop

Self-play reinforcement learning is the main loop. One network plays both
decks against current copies of itself from terminal reward only. The update
is a clipped policy gradient with two regularizers that do different jobs: a
KL term toward the policy that collected the data limits how far one update
moves, and a KL term toward a fixed reference policy shapes where the policy
is pulled. The reference is uniform over action types, then uniform over the
actions of that type, so a spell with dozens of target encodings does not get
dozens of times the prior weight of a single attack. Advantage filtering,
annealed step sizes and regularization, and an EMA of the weights for
evaluation complete the recipe, which adapts
[Ataraxos](https://doi.org/10.1038/s41586-026-11036-y). Each piece is a
switch we test on Magic rather than adopt by default. Many games step in
lockstep so inference is batched; the GPU baseline is
`batched_baseline()` in `experiments/batched_baseline.py`.

### Current questions

- **Curriculum.** The main open question is what the agent should learn
  from. We are comparing which matchups it trains on (mirrors versus balanced
  cross-deck play), which opponents (self-play, the scripted greedy bot, or a
  mix), how many decks, and whether a smaller card pool or fewer card
  identities make the same game easier to learn.
- **Scale points.** Default recipes at fixed budgets of money and wall time,
  starting at $10 for one day and $50 for seven days, tracked as the recipe
  changes. Alongside them: which model size wins at 64 streams, and where GPU
  training time goes (collection is most of it).
- **Evaluation anchors.** Scripted greedy is saturating. We are measuring
  whether the rule-based `heuristic` bot or frozen trained checkpoints order
  checkpoints of different strength, and playing chosen checkpoints against
  each other in head-to-head
  [Tournaments](../experiments/head-to-head.md).
- **More decks.** A ten-deck Elemental world is being built so training and
  evaluation can move beyond one matchup.
- **Inspection.** Academy is being shaped into a lab for loading a trained
  checkpoint, watching it play, and inspecting its decisions.

Evaluation stays narrow, just enough to compare curricula, rather than testing
current agents across a wide range of environments.

## What we know

All numbers are world-relative and trace to frozen experiment reports. Most
self-play results so far use few training seeds and weak fixed opponents, so
they guide the next experiment rather than settle a method.

### Self-play

| Finding | Evidence | Consequence |
| --- | --- | --- |
| A 10,000-update self-play cohort improves on monitoring deals, mostly by learning to pilot Lessons. | Mean win rate against scripted greedy rose from 30.3% to 54.3%; Allies saturated near 87–88% by update 1,240 while Lessons rose from 5/150 to 32/150 ([current baseline](../experiments/current-baseline.md)). | The held-out final comparison is still pending. Lessons is where learning shows. |
| Batching many games per update is a large, reliable speedup, and the smallest model learns as well as larger ones at this scale. | 64×2 took 109–111 minutes at 4 streams and 4.8–5.2 at 256 ([longer learning](../experiments/longer-learning-017.md)). | Batched collection is the default. Two seeds are not enough to rank model sizes. |
| Opponent choice drives the measured gap between the decks. | Switching the greedy opponent's deck moved Lessons results by 40 to 60 points on five deals ([mirror curriculum](../experiments/mirror-curriculum.md)). | Curriculum and evaluation opponents are first-order questions. |
| Several adapted techniques have not yet shown a benefit on Magic. | Actor-only advantage filtering: +3.33 points, interval [−4.33, +11.33], at 1.45× wall time ([technique screen](../experiments/ataraxos-technique-screen.md)); value-token and pooling variants unresolved ([value token](../experiments/value-token-screen.md), [pooling follow-up](../experiments/pooling-filter-followup.md)). | Keep them as switches; promote only on evidence. |
| The observation encoding no longer hides decision-relevant differences. | The learnability search found 0 clean input collisions ([learnability](../experiments/learnability-search.md)). | A nightly check guards it; see [WORLDS.md](../WORLDS.md). |
| Collection dominates GPU training time. | Collection was 82.5% of active time in the profiled run ([training time](../experiments/training-time.md)). | Engine and inference throughput are the cost levers. |

Belief sampling and local search run end to end. In their
pilots, 840 held-out sampled hands had no support violations and local search
replayed 183 commands with no mismatch. Neither has shown a strength gain over
the policy alone yet.

### Earlier: the search era

These come from earlier encodings and are not comparable with today's
checkpoints.

| Finding | Evidence |
| --- | --- |
| Search supplies useful intelligence without training. | Flat search dominated early policies and scaled with simulations ([exp-02](../experiments/exp-02-flat-mc.md)). |
| Distillation beat the early PPO recipe at equal cost. | 90.5% versus 52.7% against random ([exp-03](../experiments/exp-03-distillation.md)). |
| Distillation does not improve a weak teacher. | Round 1 lost 74–26 to round 0 ([exp-07](../experiments/exp-07-expert-iteration.md)). |
| Reward shaping hurt. | Terminal-only reward beat shaped rewards by about 15 points ([exp-04](../experiments/exp-04-potential-shaping.md)). |
| Win rate hid incompetence. | The search pilot could not play a control deck ([exp-09](../experiments/exp-09-control-competency.md)). |
| Information boundaries must be executable. | Hidden-pool ordering once changed search evidence between viewer-equivalent states. Viewer equivalence, replay and leakage checks are now hard gates. |

## What we plan to do later

### Search on top of the policy

```text
self-play policy (frozen checkpoint)
        |                     |
        v                     v
 belief sampler over    frozen-policy continuations
 the opponent's hidden          |
 cards                          |
        +----------+------------+
                   v
       local search at the decision
                   |
                   v
     distillation back into the policy
```

The sampler and search exist ([belief sampler](../docs/belief-sampler.md),
[local policy search](../docs/local-policy-search.md)); distillation back into
the policy is not currently implemented. The plan is to make them improve play. Search should adjust a strong learned policy, not replace
it. At a decision it samples hidden hands from the belief, plays out each
candidate action with the frozen policy, and applies one regularized update to
the policy's distribution:

```text
π_search(a) ∝ π_policy(a)^(1/(1+ηα)) · ρ(a)^(ηα/(1+ηα)) · exp(η·Q̂(a)/(1+ηα))
```

Here `π_policy` is the policy before search, `ρ` is the fixed reference and
`Q̂` the sampled action values. With `α = 0` this is a plain KL step toward
better actions. We will compare that with the version that also pulls toward
the reference, at matched worlds, rollouts and compute, and test whether more
search ever makes the policy more confidently wrong. Noisy `Q̂` with a large
`η` turns small estimation errors into large changes in behavior.

Constraints that apply throughout:

- Sampling a hidden world for simulation is not letting the policy see it.
  Continuations act from each simulated player's own observation, and action
  values are averaged across sampled worlds before choosing one action.
- Belief quality is measured separately from search strength: calibration and
  likelihood on held-out games, and how search strength degrades under a
  deliberately wrong belief.
- A shallow search with a strong continuation policy may beat deeper search
  with a weak one, so depth, rollout count and continuation quality are
  measured rather than assumed.
- Distilled targets must keep genuine mixed strategies; hard targets cannot
  represent bluffing.

The protocols for sampler quality, search with learned sampling, distillation
and exploiters are preregistered
([transfer protocols](../experiments/ataraxos-transfer.md)).

### Training dynamics

- **Regularization roles.** Measure the two KL terms, entropy (by decision
  type and number of legal actions), clip fraction and gradient norms
  separately, and compare them alone and together at matched compute.
- **Schedules and exploration.** Test whether coordinating the step-size and
  regularization schedules beats tuning each independently, and whether
  reopening exploration after a plateau recovers lost strategic diversity.
  The second is our hypothesis, not an established result. A plateau or an
  entropy drop is not diagnostic on its own: it can mean collapse or
  genuinely decisive play.
- **Advantage filtering.** Filtering changes which decisions the policy
  learns from, so we measure strength per environment step, per training
  example and per dollar, and which decision types it drops.

### Durable progress

Nontransitivity (A beats B beats C beats A), temporal cycling (training
revisits old behavior) and exploitability (losing to a best response) are
different things. We plan to play a time-ordered cohort of checkpoints against
each other and against frozen anchors, keep the full payoff matrix, and add a
fixed suite of interpretable decision states to compare action distributions
across checkpoints. Population self-play and dedicated exploiters come after
that matrix shows where the latest policy is weak.

### Throughput

The target is durable strength per unit of total compute, not raw engine
steps per second. If a fraction `f` of time is simulation and it gets `s`
times faster, the whole run speeds up by `1 / ((1 − f) + f/s)`, so inference
and data movement matter as much as the engine. Planned work, in order:
multicore CPU collection, cheaper observation and inference batching, and
undo-based branching for search instead of cloning. A GPU rules backend is
considered only against a strong CPU baseline at equal card coverage.

### Generalization and Limited

Cards are typed programs, so a policy can embed what a card does as well as
its name. Held-out cards and compositions will test whether strength reflects
card reasoning rather than memorization. The long-term target is Limited:
building decks from a sealed pool or a draft, then piloting them.

## How we measure

Every candidate is judged on multiple axes. No single number licenses a
strength claim. Three kinds of progress are easy to confuse: optimization
progress (loss, KL, entropy), relative progress (wins against the latest
opponent), and durable progress (strength across historical, diverse and
held-out opponents). The first two can improve while the third gets worse.

1. **World-pinned skill rating** is the hill-climbing objective.
2. **Competencies and behavioral profiles** show what kind of play produced
   the rating and catch strategically important failures hidden by aggregate
   wins.
3. **Approximate exploitability** tests whether a rating is inflated by the
   chosen opponent cohort. Exactly solvable microgames use exact NashConv.
4. **Cost** includes native and matched p50/p95 decision latency, training and
   label compute, throughput, and memory.
5. **Integrity** requires legal commands, viewer-safe information, exact
   replay, and pinned engine/content/model identities.
6. **Transfer** measures held-out cards, compositions, decks, and eventually
   broader formats.

### Skill rating contract

Strength is a population rating, not a win rate against random. The arena
(`manabot/eval/`) fits a seat-aware Bradley-Terry model and expresses its log-odds
parameters on the familiar Elo scale; [Tournament](../experiments/head-to-head.md)
files play chosen checkpoints against each other on shared deals.

A rating belongs to an exact tuple:

```text
(world, content/matchup suite, information boundary, compute class, arena version)
```

Ratings never cross world or arena-version boundaries. Within one arena:

- games are seat-balanced and paired by deal seed;
- the frozen anchor cohort includes random, scripted players, several fixed
  flat-search budgets, and frozen learned incumbents;
- one anchor fixes the arbitrary origin of the scale; rating differences, not
  the absolute number, carry meaning;
- uncertainty is estimated over paired deal blocks, not by pretending every
  correlated game is an independent training seed;
- the complete matchup matrix and model residuals remain visible because
  Magic policies and decks can be non-transitive;
- policy-only and search-augmented agents are compared in explicit compute
  classes rather than allowing unlimited test-time search to masquerade as a
  better policy.

The first rating arena should stay inside one symmetric selected matchup so it
measures play rather than deck selection. Later arena versions can add a fixed
deck suite and, eventually, human opponents. Rating is a navigation signal,
not evidence of equilibrium play or a superhuman claim by itself.

Training opponents and evaluation opponents are different instruments. The
evaluation cohort stays frozen for an arena version. Training may sample the
checkpoint archive adaptively, but a scheduler change cannot rewrite the
scoreboard. Every admitted checkpoint is immutable and remains available for
regression matches.

### The league ladder

Build the league only as far as the current population justifies:

1. **Fixed anchors:** rate runnable candidates against random, scripted, flat
   search, and frozen learned players at one compute class.
2. **Checkpoint archive:** admit materially stronger candidates without
   deleting their predecessors; retain the full payoff matrix.
3. **Informative scheduling:** once the archive spans useful skill, spend
   matches near uncertain boundaries rather than repeatedly proving 95-5
   outcomes.
4. **Exploiters:** train or search specifically against the incumbent and add
   successful counter-strategies to the archive. Promote general strength,
   not the exploiter itself, unless it also survives the cohort.
5. **Wider anchors:** add fixed deck suites and human cohorts only after the
   corresponding world, information, and compute contracts are stable.

Promotion requires a preregistered rating improvement at fixed compute plus no
material integrity or competency regression. Elo chooses promising builds;
the payoff matrix and exploiters decide whether the gain is robust.

### Experiment principles

- **One mechanism at a time.** If several ideas change together, a stronger
  checkpoint does not say which worked. Where an interaction is the
  hypothesis, use a factorial comparison.
- **Behavior and outcomes.** Report optimization diagnostics, action
  distributions on fixed decision probes, strength with uncertainty against
  historical and held-out opponents, and total compute.
- **Semantics first.** managym stays the authority for legality and state;
  a speedup that changes rules behavior or replay is not a valid comparison.
- **Negative results are evidence.** Record them with the same identities,
  seeds and intervals as positive ones.

## What successful ladders teach

The useful lineage is a set of engineering patterns, not a recipe to copy.
The detailed source review lives in
[`docs/research/intelligence-development-ladders.md`](../docs/research/intelligence-development-ladders.md).

| System | How it hill-climbed | What manabot should borrow |
| --- | --- | --- |
| [Neural MMO](https://arxiv.org/abs/2311.03736) | Goal-conditioned policies train on task curricula and compete on unseen tasks, maps, and opponents under explicit compute limits. Its baseline stack includes fixed and adaptive curricula, policy zoos, replays, and multi-policy evaluation. | Separate integrated strength from diagnostic tasks; hold out task/content/opponent combinations; publish the compute envelope; make curricula generators and samplers first-class, but only after a runnable policy exposes a learning bottleneck. |
| [CoGames](https://github.com/Metta-AI/metta-public/tree/main/packages/cogames) to [Coworld](https://github.com/Metta-AI/coworld) | CoGames paired scripted and learned players with deterministic diagnostic missions, procedural integrated missions, fixed replacement pools, and a hosted league. Coworld generalizes the loop into game, player, commissioner, replay, report, diagnosis, and optimization contracts. Its [Magic world](https://github.com/Metta-AI/coworld-mtg) reverses deck assignments and preserves seeded, replayable traces. | Keep the player-improvement loop artifact-first: every match yields results, replay, logs, and identities. Use fixed scenarios to locate deficits found in integrated play. Treat scheduling and rating as replaceable arena policy, not rules-engine behavior. |
| [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) | Human replay imitation bootstrapped a competent population; league reinforcement learning then used adaptive strategies, counter-strategies, and prioritized fictitious self-play before final Battle.net MMR evaluation. | Bootstrap from teacher data, preserve a population, sample informative weaknesses, and create dedicated exploiters. Keep the full matrix: AlphaStar reported millions of strong rock-paper-scissors cycles involving exploiters, so one MMR could not describe the league. |
| [KataGo](https://arxiv.org/abs/1902.10565) | Neural-guided search generated self-play targets, new networks re-entered search, and fixed-compute matches plus ablations measured progress. Auxiliary score/ownership/future-policy targets, varied search budgets, and later [policy-surprise weighting](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#policy-surprise-weighting) made the loop far more sample-efficient. | Make the value head explain useful substructure, not only final wins; mix cheap trajectories with strategically deep labels; upweight positions where search most changes the policy; judge every idea by rating gained per self-play and label compute. |
| [OpenAI Five](https://cdn.openai.com/dota-2.pdf) | Continuous self-play used the current policy 80% of the time and archived opponents 20% of the time. A fixed reference pool supplied continual TrueSkill, near-skill opponents concentrated evaluation information, and human/pro matches anchored the scale. Policy-preserving surgery kept a strong agent alive while the environment evolved. | Start archiving now, evaluate continuously against immutable nearby references, and reserve a stable human anchor for mature worlds. When `managym` changes, preserve or explicitly fork artifacts and arena versions rather than drawing one curve across incompatible environments. |

The common shape is:

```text
runnable baseline
  -> immutable checkpoints and replayable matches
  -> compute-pinned population rating + payoff matrix
  -> targeted curricula, search labels, and exploiters
  -> promoted generalist
  -> external human/competition anchor
  -> repeat on a deliberately widened world
```

Two anti-patterns recur. Training only against the latest self can forget old
strategies, and optimizing only the published scalar can overfit the arena.
The archive protects the first; held-out suites, competencies, and external
anchors protect the second.

## Claim ladder

1. **Runnable:** acts legally through authoritative `managym` games.
2. **Measured:** has a world-pinned rating, competencies, behavioral profile,
   calibration, and cost.
3. **Improving:** a training change raises rating at fixed compute across
   training seeds without competency or integrity regression.
4. **Belief-aware:** public-history ranges are calibrated and improve play
   against informative opponents.
5. **Strategically robust:** approximate best responses do not expose a
   material exploit; exact microgames approach equilibrium.
6. **Transferable:** gains survive held-out cards, compositions, decks, and a
   declared broader content boundary.
7. **Bounded superhuman:** under a pinned format, information boundary,
   compute budget, and human cohort, the manabot exceeds elite human play and
   survives the declared exploitability battery.

Limited play (building decks from a sealed pool or a draft, then piloting
them) adds capabilities beyond piloting fixed decks. They are not
prerequisites for the first bounded play-policy claim.

## Sources of truth

| Source | Owns |
| --- | --- |
| This document | What we are doing, what we know, what we plan, and how we measure |
| [`experiments/`](../experiments/README.md) | Preregistrations, immutable reports, raw evidence links, and reproduction |
| [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) | System design and ownership |
| [`WORLDS.md`](../WORLDS.md) | Observation/action worlds and artifact compatibility |
| Linear | Projects, Tasks, owners, dependencies, and live execution state |

Every completed experiment should update its frozen report and, when the
conclusion changes the program, one row or plan in this document. Do not copy
PR status, task checklists, or live ownership here.
