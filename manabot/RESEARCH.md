# Manabot research

Manabot is Etude Fantasia's program for building increasingly strong Magic:
The Gathering agents. The objective is not to accumulate isolated model
results. It is to build agents that play real `managym` games, measure them
honestly, understand why they improve or fail, and repeat until a bounded
superhuman claim is earned.

This document is the durable map of that program. It records the research
thesis, accepted evidence, current capability frontiers, and the decisions
future experiments must make. Execution status and ownership live in Linear;
frozen predictions, results, and provenance live in [`experiments/`](../experiments/README.md).

## The builder's loop

The unit of progress is a stronger runnable system:

1. build the thinnest teacher, player, or training loop that acts in the real
   engine;
2. place it on a common skill-and-cost scoreboard;
3. inspect behavior, competencies, calibration, and information safety;
4. when a result is surprising or confounded, run the smallest ablation or
   kata that distinguishes the live explanations;
5. change the running system and measure it again.

Katas are diagnostic instruments. They do not grant permission to build an
agent, and success on them is not a substitute for gameplay. Expensive
comparisons remain preregistered so their thresholds cannot move after the
result is known.

## The intelligence system

Three coupled capabilities form the path from a working player to a
superhuman one:

```text
authoritative game + viewer-safe public history
                       |
                       v
                beliefs / ranges
                       |
                       v
          slow teacher: search or solve
             | policy       | value
             v              v
             fast policy/value student
                  |             |
                  +----> next teacher
                              |
                            repeat
```

- **Value learning** amortizes the future: estimate which positions or public
  belief states are favorable without playing every continuation to terminal.
- **Teacher-to-student distillation** amortizes deliberation: turn expensive
  local search into a fast policy that generalizes to unseen positions.
- **Belief modeling** makes both honest under hidden information: infer ranges
  from public history and eventually value and solve public belief states
  rather than independent sampled worlds.

Search, solving, and opponent populations are the improvement operators. A
value head or student can only compress the intelligence present in its
targets. Exp-07 demonstrated the failure mode: a cheaper but weaker teacher
produced a weaker next student.

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

### Skill rating contract to establish

Manabot needs a reproducible population rating rather than another win rate
against random. The initial instrument should fit a batch Bradley-Terry model
and express its log-odds parameters on the familiar Elo scale.

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

All numbers below are world-relative and trace to frozen experiment reports.

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Search supplies useful intelligence without training. | Exp-02's flat search dominates early policies and scales with simulations. | Keep search as a baseline and source of targets, but do not confuse its aggregate strength with strategic completeness. |
| Search distillation is much more effective than the early PPO recipe. | Exp-03's student reached 90.5% versus random and roughly search-8 strength at about 1 ms/decision. | Teacher-to-student learning is a primary path. |
| Distillation does not improve a weak teacher. | Exp-07 round 1 lost 25.8% head-to-head to round 0 after affordable policy-rollout search reduced label quality. | Prove teacher improvement at matched cost before buying another corpus. |
| The current distilled student resisted one matched-budget exploiter. | Exp-11's dedicated PPO attackers reached only 23.5-26.0%. | Preserve adversarial evaluation, while recognizing this is not a best-response certificate. |
| A scalar value can support policy improvement but is not yet a rollout substitute. | Exp-10 search-over-V beat V-greedy 60.25%, yet lost to random-rollout search at matched wall time. | Judge values by search utility, not calibration alone; target quality and hidden information remain open. |
| Joint value supervision is trainable and inexpensive. | Teacher-0's immutable 512-game prefix improved value Brier to 0.167 without reducing batched throughput or small-sample gameplay. | Carry policy-only controls, but keep building joint students. |
| Visit-based policy/value learning and neural PUCT execute end to end. | INT-4's engineering smoke produced legal replayable labels, four matched student arms, a neural-search arena, and Study evidence. | Run the production-strength iteration; the smoke makes no strength claim. |
| Information boundaries must be executable. | INT-4 exposed and fixed hidden-pool ordering that changed search evidence between viewer-equivalent authorities. | Viewer equivalence, replay, and leakage checks remain hard gates for every teacher. |
| Static semantic structure matters, but the first structural encoder family was not viable. | W2-214 and INT-1 killed bag pooling and the first relational-pooling design. | Build a plausible semantic policy in real play; use the katas only to diagnose its failures. |

## Capability frontiers

### Value learning

The current head predicts one scalar from the acting viewer's observation.
The next useful build is a student whose value improves PUCT at matched cost,
with terminal outcomes, teacher root values, and a preregistered blend compared
on the same frozen trajectories.

The longer path changes the type of the value function:

```text
scalar outcome value
  -> useful leaf evaluator
  -> search-trained and reanalyzed value
  -> range-conditioned per-hand counterfactual values
  -> equilibrium continuation values from deeper public-belief solves
```

Teacher-root imitation and terminal-outcome calibration remain separate
claims. A low Brier score matters only if the resulting decisions improve.

### Teacher-to-student distillation

Chosen-action cloning is the working baseline. The next build learns normalized
root visits and value targets, returns the student to PUCT, and measures whether
the complete loop improves across training seeds.

```text
chosen action
  -> visit distribution
  -> joint policy/value student
  -> student-guided search
  -> stronger teacher
  -> repeated population-aware improvement
```

Hard targets can be effective for noisy flat search, but they cannot represent
the calibrated mixing required for bluffing. Later belief-consistent teachers
must emit and students must preserve genuine mixed strategies.

### Belief modeling

Current search samples viewer-safe worlds uniformly and builds a separate tree
inside each. That is a necessary baseline, not information-set-consistent
planning. The current selected deck's 10,832 possible seven-card hands make an
exact first range tracker tractable.

The first belief build should be a complete player, not a standalone posterior
demo: maintain an exact range from public history, update it through canonical
action likelihoods and chance events, sample weighted determinizations, and
play the selected matchup. Calibration and a uniform-sampling ablation explain
the gameplay result.

```text
viewer-safe uniform determinization
  -> exact calibrated range tracker
  -> likelihood-weighted search player
  -> public-belief search with mixed policies
  -> safe continual resolving at useful Magic boundaries
```

If the working player produces an ambiguous failure, the information x
continuation matrix and exactly solvable counterspell/bait microgame determine
whether the next treatment is better continuation, better beliefs, or
information-set-consistent solving.

### Semantic generalization

Semantic programs and structured commands are a cross-cutting substrate rather
than a fourth independent improvement loop. A semantic policy must play real
games and enter the same rating arena. Held-out identities and compositions
then test whether its strength reflects transferable card reasoning rather
than memorization.

## Parallel build map

Independent work should share frozen inputs rather than silently create
different experiments:

```text
freeze world, arena, teacher, seeds, and evaluation cohort
                         |
       +-----------------+------------------+
       |                 |                  |
       v                 v                  v
 skill-rating       visit/value       belief-aware
 instrument         iteration         search player
                         |
                  freeze one corpus
                         |
             +-----------+------------+
             |                        |
             v                        v
       value-target arms       teacher-guidance arms
             |                        |
             +-----------+------------+
                         v
                    combined arena
```

The rating instrument, visit-based baseline, belief-aware player, and semantic
policy can be built concurrently. Once a corpus is frozen, policy and value
training arms can run independently from that same read-only artifact.
World/contract freezes, corpus generation, matched-host latency calibration,
and the final combined arena remain serialized.

On the current Apple host, run only one MPS training job at a time. CPU work can
proceed alongside non-timing-sensitive training, but latency, throughput, and
memory cells require a quiet host.

## Claim ladder

1. **Runnable:** acts legally through authoritative `managym` games.
2. **Measured:** has a world-pinned rating, competencies, behavioral profile,
   calibration, and cost.
3. **Improving:** a teacher/student iteration raises rating at fixed compute
   across training seeds without competency or integrity regression.
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
| This document | Research thesis, accepted evidence, capability frontiers, and decision graph |
| [`experiments/`](../experiments/README.md) | Preregistrations, immutable reports, raw evidence links, and reproduction |
| [`paper/understanding.md`](../paper/understanding.md) | Causal conclusions mature enough to support publication |
| [`wave/intelligence/GOAL.md`](../wave/intelligence/GOAL.md) | Intelligence's objective, measures, bounds, and operating loop |
| Linear | Projects, Tasks, owners, dependencies, and live execution state |
| [`WORLDS.md`](../WORLDS.md) | Observation/action worlds and artifact compatibility |

Every completed experiment should update its frozen report and, when the
conclusion changes the program, one evidence row or frontier in this document.
Do not copy PR status, task checklists, or live ownership here.
