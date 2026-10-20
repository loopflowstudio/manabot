# Manabot research

Manabot's research program builds increasingly strong Magic: The Gathering
agents. The aim is agents that play real `managym` games, are measured
honestly, and are understood well enough that we know why they improve or
fail, repeated until a bounded superhuman claim is earned.

This document is the map of that program: the approach, the evidence we
accept, the open questions, and the claims we intend to earn. Preregistered
predictions, results and provenance live in
[`experiments/`](../experiments/README.md). The system design is in
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md); the
[learning research guide](LEARNING_RESEARCH_GUIDE.md) works through the
learning questions in depth.

## The builder's loop

The unit of progress is a stronger runnable system:

1. build the thinnest training loop or player that acts in the real engine;
2. place it on a common skill-and-cost scoreboard;
3. inspect behavior, competencies, calibration, and information safety;
4. when a result is surprising or confounded, run the smallest ablation or
   kata that distinguishes the live explanations;
5. change the running system and measure it again.

Katas are diagnostic instruments, not a substitute for gameplay. Expensive
comparisons are preregistered so their thresholds cannot move after the
result is known.

## The approach

Self-play reinforcement learning is the main loop. One network plays both
decks against current copies of itself from terminal reward only, and the
update is a clipped policy gradient regularized toward the collection policy
and a fixed reference policy, with advantage filtering, annealed step sizes
and an EMA of the weights. Many games step in lockstep so inference is
batched. The recipe adapts
[Ataraxos](https://doi.org/10.1038/s41586-026-11036-y); each of its pieces is
a separate switch that we test on Magic rather than adopt on the paper's
authority.

Hidden information and search sit on top of that policy:

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

- **Beliefs.** A sampler learns a distribution over the opponent's hidden
  cards from the frozen policy's own games
  ([belief sampler](../docs/belief-sampler.md)). Exact hand enumeration is
  kept only as a reference.
- **Search.** Local search samples hands from that belief, plays out
  continuations with the frozen policy, and updates the action distribution
  at the decision ([local policy search](../docs/local-policy-search.md)).
- **Distillation.** Search choices are distilled back into the policy so the
  next round of self-play starts stronger.

Search, beliefs and opponent populations are the improvement operators. A
policy can only compress the intelligence present in its targets: in the
search era, a cheaper but weaker teacher produced a weaker next student
([exp-07](../experiments/exp-07-expert-iteration.md)).

## The scoreboard

Every candidate is judged on multiple axes. No single number licenses a
strength claim.

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
(`manabot/arena/`) fits a seat-aware Bradley-Terry model and expresses its log-odds
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

Belief sampling and local search with distillation run end to end with zero
replay mismatches, but neither has shown a strength gain over the policy
alone yet.

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

## Open questions

### Curriculum

The main open question is what the agent should learn from: which matchups,
which opponents, and what mix of self-play, for example mirror matches versus
balanced cross-deck play. The first mirror-inclusive comparison was negative
on a small pilot and its 10,000-update run is incomplete
([mirror curriculum](../experiments/mirror-curriculum.md)). Evaluation stays
narrow, just enough to compare curricula.

### Evaluation anchors

Scripted greedy is saturating as an anchor. A stronger rule-based `heuristic`
bot and frozen trained checkpoints are candidates, and head-to-head
Tournaments already separate checkpoints that look identical against a
scripted opponent ([head-to-head](../experiments/head-to-head.md)).

### Scale

No model size has won yet, and one large configuration underperforms for
reasons we have not explained ([longer learning](../experiments/longer-learning-017.md)).
Separating the effects of more updates, filtering and staleness in batched
training is still open.

### Beliefs and search

The next steps are preregistered but unrun
([Ataraxos transfer protocols](../experiments/ataraxos-transfer.md)): sampler
quality, search with learned sampling, distillation, and exploiters trained
against the incumbent. Later belief-consistent search must emit, and students
must preserve, genuine mixed strategies, since hard targets cannot represent
bluffing.

### Semantic generalization

Cards are typed programs, so a policy can embed what a card does as well as
its name. Held-out cards and compositions will test whether strength reflects
transferable card reasoning rather than memorization.

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

Avatar Cube Team Sealed later adds deck construction, teammate allocation, and
a three-by-three series. Those are additional capabilities, not prerequisites
for the first bounded play-policy claim.

## Sources of truth

| Source | Owns |
| --- | --- |
| This document | Approach, accepted evidence, open questions, and the claim ladder |
| [`experiments/`](../experiments/README.md) | Preregistrations, immutable reports, raw evidence links, and reproduction |
| [`paper/understanding.md`](../paper/understanding.md) | Causal conclusions mature enough to support publication |
| [`wave/intelligence/GOAL.md`](../wave/intelligence/GOAL.md) | Intelligence's objective, measures, bounds, and operating loop |
| Linear | Projects, Tasks, owners, dependencies, and live execution state |
| [`WORLDS.md`](../WORLDS.md) | Observation/action worlds and artifact compatibility |

Every completed experiment should update its frozen report and, when the
conclusion changes the program, one evidence row or open question in this
document.
Do not copy PR status, task checklists, or live ownership here.
