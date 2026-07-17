# Intelligence ladder research

## Question

What should Etude Fantasia borrow from Neural MMO, Softmax's CoGames/Coworld,
AlphaStar, KataGo, and OpenAI Five when building a measurable road to a
superhuman manabot?

## System understanding

These projects converge on a layered improvement system:

```text
real environment and runnable player
              |
              v
replayable matches + immutable checkpoints
              |
              v
rating, payoff matrix, and diagnostic slices
              |
              v
training opponent/curriculum/search scheduler
              |
              v
stronger generalist + targeted exploiters
              |
              v
human or public competition anchor
```

The environment, training scheduler, and evaluator are separate systems. A
training curriculum may change quickly. A rating cohort must remain frozen
long enough to make progress comparable. Rules or observation changes fork the
world and therefore the rating scale.

## Development ladders

### Neural MMO

Neural MMO 2.0 turns a large partially observable multi-agent world into a
multi-task benchmark. Goal-conditioned teams train on curricula and are tested
over unseen tasks, maps, and opponents. The competition constrained training
compute, used many rounds and similarly skilled lobbies, and separated an RL
track from a curriculum track so architecture and task-sampling improvements
could be studied independently. The baseline exposes fixed and Syllabus-driven
curricula, an agent zoo, PvE/PvP evaluation, replay generation, and experiment
tracking.

The strongest design choice is not the MMO scale. It is treating task
generation, task sampling, opponent diversity, and generalization as explicit
axes while keeping a common runnable policy and compute budget. The danger is
that task completion can become its own benchmark game; integrated adversarial
play still needs to be the north star.

Sources: [Neural MMO 2.0 paper](https://arxiv.org/abs/2311.03736),
[competition results](https://arxiv.org/abs/2508.12524),
[baseline repository](https://github.com/NeuralMMO/baselines), and
[documentation](https://neuralmmo.github.io/).

### CoGames and Coworld

The public CoGames stack combined a real player loop with deterministic
diagnostic missions, procedural integrated missions, scripted baselines,
learned policies, fixed replacement pools, and hosted competition. Its VORP
metric asked how much a candidate improved a mixed team relative to a
replacement policy, which is more appropriate for cooperative populations
than a naive individual win rate. The diagnostic/integrated split is especially
healthy: a complete policy plays the game first, and fixed missions locate the
broken capability afterward.

CoGames is now retired. Its successor, Coworld, makes the improvement loop an
explicit runtime architecture: game, player, commissioner, reporter, grader,
diagnoser, and optimizer roles exchange typed episode artifacts. The
commissioner can own Swiss, round-robin, adaptive, Elo-seeded, or other league
logic without putting tournament policy in the game. Results, replays, logs,
seeds, and policy identities are durable episode outputs. The current Coworld
Magic implementation reverses deck assignments across variants and can pin a
root seed to reproduce library order and later randomness.

The transfer to Etude is organizational: make the rules world authoritative,
the arena replaceable, and every improvement debuggable from exact artifacts.
Do not copy the retired package or its infrastructure wholesale.

Sources: [CoGames history](https://github.com/Metta-AI/metta-public/tree/main/packages/cogames),
[CoGames tombstone](https://github.com/Metta-AI/cogames),
[Coworld](https://github.com/Metta-AI/coworld), and
[Coworld Magic](https://github.com/Metta-AI/coworld-mtg).

### AlphaStar

AlphaStar bootstrapped from human replay imitation, then entered a league of
continually adapting strategies and counter-strategies. Prioritized fictitious
self-play focused training on opponents that exposed useful weaknesses.
Specialized exploiters searched for holes while main agents pursued general
strength. Final evaluation moved outside the internal population to the human
Battle.net ladder.

The payoff matrix is the decisive lesson. AlphaStar's main-agent lineage was
mostly transitive, but its published league contained roughly three million
strong rock-paper-scissors cycles involving exploiters. Elo/MMR is an excellent
navigation signal and a poor complete description of a strategic population.
Manabot should therefore store the matrix, use exploiters as tests, and promote
a generalist only when its improvement survives the cohort.

Source: [AlphaStar Nature paper](https://www.nature.com/articles/s41586-019-1724-z).

### KataGo

KataGo kept the core AlphaZero loop—search self-play, train a policy/value net,
put the net back into search—but optimized information efficiency. Playout-cap
randomization supplied many cheap game trajectories while reserving deep
search for policy targets. Forced exploration was pruned out of the final
policy target. Future-action, score, and ownership auxiliary targets taught
useful substructure. Later policy-surprise weighting sampled positions more
often when search substantially disagreed with the prior. Fixed-search and
fixed-wall-clock matches, pooled Elo, and component ablations quantified what
actually accelerated the climb.

This directly informs manabot's value and distillation work. Terminal outcome,
root value, per-zone/card consequences, future actions, and public-belief
statistics can be separate targets. Positions where PUCT changes the student's
policy are likely more valuable than another easy agreement position. Search
budget should vary intentionally rather than applying an expensive teacher
uniformly.

Sources: [KataGo paper](https://arxiv.org/abs/1902.10565),
[current methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md), and
[training history](https://github.com/lightvector/KataGo/blob/master/TrainingHistory.md).

### OpenAI Five

OpenAI Five used a single long-running PPO/self-play system. Eighty percent of
training games used the current parameters and twenty percent used archived
versions to prevent strategy collapse. Past opponents were sampled
adaptively. Separately, a fixed pool of 83 reference agents supported continual
TrueSkill: evaluators preferred opponents within a useful skill band, collected
hundreds of games per evaluated version, and left anchor ratings fixed. Human
and professional matches supplied the external meaning of the scale.

The less celebrated lesson is build continuity. More than twenty
policy-preserving "surgeries" let the team expand observations, actions, model
shape, and Dota support while retaining a high-skill agent. This tightened the
feedback loop for features that matter only at high skill. It also exposed the
metric hazard: agents from different game versions cannot be placed honestly
on one curve without translating them into a common final environment, which
can bias earlier agents.

Source: [OpenAI Five paper](https://cdn.openai.com/dota-2.pdf) and
[project overview](https://openai.com/index/openai-five/).

## Tensions

- A scalar makes progress legible; non-transitivity makes it incomplete.
- Frozen anchors make measurements comparable; adaptive opponents make
  training efficient. They must be separate cohorts.
- Katas make mechanisms visible; integrated gameplay decides whether the
  capability matters.
- Search produces intelligence; uniform deep search can waste most label
  compute on unsurprising positions.
- A stable world supports a long curve; a builder must keep improving the
  rules, observations, and agent interface. World changes need explicit forks
  or behavior-preserving migrations.
- A symmetric matchup isolates play skill; Magic mastery ultimately requires a
  deck and matchup population where cycles will be common.

## Recommendations

1. Build a v1 arena around one symmetric selected matchup with paired
   seat/deal seeds, immutable checkpoints, explicit compute classes, batch
   Bradley-Terry/Elo, uncertainty, and the full payoff matrix.
2. Keep random, scripted, flat-search, and learned anchors frozen. Do not use
   the training scheduler's changing opponent distribution as the rating
   dataset.
3. Archive every admitted player now. Add near-skill scheduling only after the
   archive spans useful strength; add PFSP-like sampling and dedicated
   exploiters only after there is a population worth exploiting.
4. For the next visit/value iteration, compare terminal, search-root, and
   blended value targets on one frozen corpus. Measure the resulting student's
   value as a PUCT leaf evaluator, not only by calibration.
5. Cross policy prior quality with search budget. The central teacher question
   is whether the student helps PUCT discover better targets per node and per
   wall-clock second.
6. Add search-disagreement buckets to the corpus. If they predict improvement,
   try KataGo-style surprise-weighted sampling before generating a much larger
   corpus.
7. Build the exact-range belief tracker into a complete weighted-determinization
   player. Use posterior calibration and uniform-determinization as explanations
   of its arena result.
8. Preserve results, replay, logs, seeds, world, model, and opponent identities
   for every rating match. A league result that cannot be replayed is not
   research evidence.
9. Delay a human rating anchor until the world and timing contract are stable,
   but design the arena so human matches can enter as another frozen cohort.

## Open questions

- Which batch rating model behaves best with Magic draws, paired deals, and a
  sparse non-transitive matrix: Bradley-Terry, Davidson, or an OpenSkill model?
- What is the smallest checkpoint population at which adaptive opponent
  sampling beats uniform archive sampling?
- Does root policy surprise predict useful manabot learning, or merely tactical
  positions with more legal actions?
- Which value subtargets are viewer-safe and cheap enough to improve search:
  life delta, material/board summaries, future public events, or range-aware
  counterfactual values?
- Can player behavior be preserved across a `managym` world change, or should
  Etude prefer cheap rerating in a deliberately new arena version?
