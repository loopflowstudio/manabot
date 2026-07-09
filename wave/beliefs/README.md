# Beliefs

> **Status: dormant, trigger-armed** (2026-07-09). This wave activates when
> the search wave's **Exit 1 tripwire** fires: two consecutive 10x rollout
> increases gain < 2 points on the ladder AND an exploiter reaches ≥ 65%
> against the frozen search-derived policy — the signature that strategy
> fusion, not compute, is the binding constraint. Informally: build the
> listener when there is someone worth listening to. Until then, uniform
> determinization is *exactly correct* against uninformative opponents, and a
> belief tracker pointed at a random player converges, correctly and
> uselessly, to the prior we already have.

## Vision

Lift play from states to **public belief states** (PBS): public game state
plus both players' ranges. In PBS space, hidden-information Magic becomes
effectively perfect-information — values are well-defined, the AlphaZero /
expert-iteration loop applies again, and bluffing becomes a representable,
measurable strategy rather than a folk concept.

This is the DeepStack → ReBeL → Student of Games lineage, transplanted. The
transplant is unusually viable here, for reasons measured in this repo:

- **The belief space is poker-sized.** The current INTERACTIVE_DECK admits
  exactly **10,832 distinct 7-card hands** (11.81 bits of entropy vs. poker's
  1,326 combos / ~10.4 bits) — computed 2026-07-09, generating-function exact.
  A full explicit range vector with exact Bayes updates is trivially tractable
  *today*. The count scales like C(names+6, 7): ~250k at 20 card names, ~10M
  at 40. Factorized marginals (per-card-name presence probabilities + land
  count histogram) are the pre-planned graceful degradation for larger pools —
  Magic decisions hinge on marginal/threshold questions, so the joint is
  rarely needed.
- **Ground truth is loggable.** Self-play records real hidden hands, so belief
  models train *supervised* and calibration is a SQL query — a luxury poker
  never had.
- **Counterfactual queries are already cheap.** Cards enter the network as
  feature vectors (guard this forever), so encoding a hypothetical hand is
  writing rows into the observation tensor, and sweeping all 10,832 hands is
  one batched forward pass on a 100k-param net.

### Not here

- Multiplayer / general-sum anything (the entire game-theoretic license below
  is two-player zero-sum; it does not survive outside it)
- Opponent exploitation / adaptive modeling (safety first; exploitation is a
  conscious later trade)
- Hidden decklists (known lists are a standing assumption of the search wave)

## The design (captured from the 2026-07-09 advisor session)

### State: PBS

PBS = (public state, hero range, villain range). "Value of my observation" is
a **type error** in hidden-information games — the same private hand in the
same public spot has different values under different ranges. Values exist
only at the PBS level.

### Update rule

r′(h) ∝ r(h) · σ̃(a | h), with:

- **σ̃ = (1−ε)σ + ε·uniform(legal(h))** — the ε-floor (trembling hand as an
  implementation detail). Keep the two zeros distinct: *logical* zeros (hand
  cannot legally take the action — card removal, exact, flows through the same
  multiplication) vs *behavioral* zeros (model says "wouldn't"), which must be
  floored or one mispredicted action permanently annihilates true hands
  (particle death).
- **Chance events update via convolution, not σ**: draws convolve the range
  with the hypergeometric deal from the opponent's unseen pool (their decklist
  minus their public zones — separate decks mean no cross-player blockers;
  poker's blocker machinery has no analog here). Reveals collapse support.
- **Canonical action IDs** (the Magic-specific engineering poker never
  needed): the legal action *list* depends on the hidden hand, so updates
  cannot be over positional indices. Actions need content-based identity —
  (type, card name, target identities) — matched per hypothetical hand;
  no match ⇒ logical zero.
- σ is queried by **batched counterfactual forward passes**: swap hypothetical
  hands into the observation, one batch over the support, softmax per hand's
  legal list, extract the matching action's probability.

### License: why ranges are "public"

Ranges are computable from public history only *relative to a strategy*; the
real assumption is **common knowledge of the strategy** (the hand→frequency
map, not the hand). The game-theoretic license: **in two-player zero-sum,
equilibrium strategies are announcement-proof** — publishing the full mixing
document costs nothing, because equilibrium frequencies make the opponent's
best responses indifferent. Asymmetry to keep in mind: *my* range against my
actual strategy is exact; *their* range is a model. Under disagreement my
beliefs about them go wrong but the equilibrium guarantee survives — wrong
ranges cost unexploited profit, not safety. In self-play training the question
dissolves: both sides share σ by construction.

### Bluffing, technically

A bluff induces **no false beliefs**. It is the low-strength side of an
equilibrium **pooling** action (strong and weak hands take the same action),
mixed at the frequency that makes the opponent's posterior payoff-indifferent
across responses. The value exists only at the range level — which is exactly
why per-world evaluators (determinized search) cannot see it (strategy
fusion), and why near-deterministic policies cannot express it. Magic form:
"representing the Counterspell" = passing with UU open, pooled across
holding/not-holding at calibrated frequency; the price is paid in tempo.

Consequences for representation: the policy head must output and be trained as
a genuine **mixed strategy**. Hard-target (argmax) distillation collapses
posteriors and makes bluffing unrepresentable — distillation targets must
become **distributions** (search's per-action scores — `flat_mc_scores`
already returns them — or CFR's average strategy).

### Where σ comes from

Policy nets: batched counterfactual queries (above). Search players: per-hand
MCTS is dead on arrival (10k searches per observed decision); **CFR-style
solvers compute the whole range's strategy in one solve** — this is the
structural reason PBS methods pair with CFR, not MCTS. The policy net is the
amortized solver table.

### Value head

- **Input**: PBS — the ranges go *into* the network.
- **Output**: per-hand counterfactual value **vector** (one value per hand in
  the range, both players; zero-sum consistency as free regularization), not a
  scalar — the solver above a depth limit needs per-hand values to compute
  regrets.
- **Targets**: root values of deeper solves (value iteration whose backup
  operator is an equilibrium solve). Training PBSs from **self-play** (ReBeL),
  not random generation (DeepStack) — density where play actually goes.

### Off-model actions

Bayes is silent on zero-probability events (off-equilibrium-path beliefs; the
refinements literature). Response ladder: (1) the ε-floor keeps posteriors
defined — degrades to card-removal-only inference; (2) **safe re-solving**
from the post-action PBS (the Libratus answer): don't interpret the deviation,
construct a response under which no interpretation of it profits; (3) never
hard-code action translation (the falsified middle road — poker's
translation-mapping exploits). Magic gift: discrete enumerable actions mean
the off-*tree* problem (poker's continuous bet sizes) does not exist — only
off-*model* remains. Magic twist: "never" actions are occasionally correct
(Bolt-own-Man-o'-War class), so the same ε is also exploration mass for
finding heresies in our own play. **Surprise ledger**: accumulate per-opponent
log-likelihood of observed actions under σ̃ — in self-play it doubles as a bug
canary (should be ≈ expected entropy; drift = canonicalization or tracker
bug), against foreign opponents it detects when the equilibrium license is
falsified.

## The open research problem

**Subgame boundaries in Magic.** Poker has streets — natural public-state
roots where solves anchor. Magic has no streets; the stack and priority weave
decisions continuously. Turn boundaries are the coarse candidate, priority
windows the fine one; nobody has published the right decomposition because
nobody has gotten this far. If this wave produces a paper, this is the paper —
not "we ran ReBeL on Magic."

## Goals (when activated)

1. Canonical action IDs (engine-side naming; small).
2. Belief tracker: support enumeration from decklist, exact Bayes with
   ε-floor, chance convolutions, batched counterfactual queries. Calibration
   harness against logged ground truth from day one.
3. Soft-target distillation (loss-function change; prerequisite from C4/C5
   even before activation — flag at the C5 design point).
4. Likelihood-weighted determinization: reweight sampled worlds by σ̃ of the
   opponent's observed actions (particle filter). Measure the
   **uniform-vs-weighted search gap** — the value of listening.
5. Range-conditioned value net (per-hand output vector), solver-target
   training; subgame decomposition experiments.

## Pre-registered predictions (2026-07-09, while agnostic)

- Uniform-vs-weighted determinization gap: **~0 against random opponents,
  material and growing against every rung of the self-play ladder.** Flat gap
  as opponents strengthen falsifies the whole mechanism.
- Rung-1 scalar value head (AlphaZero-naive, pre-PBS) climbs several ladder
  rungs, then stalls with a measurable per-bucket bias concentrated in
  "behind on board, holding interaction" states; range-conditioning recovers
  it. No stall = a surprising, publishable fact about how little Magic's
  hidden information binds at this deck complexity.
- Milestone worth the name: **the first bot to bluff in Magic with a
  verifiable bluffing frequency** — a pooling frequency in the logs, audited
  against ground-truth hands.

## Metrics

- Belief calibration vs logged ground truth (reliability curves per card name)
- Surprise ledger ≈ expected entropy in self-play (bug canary)
- Uniform-vs-weighted determinization win-rate gap, per ladder rung
- Pooling/bluff frequencies at mana-open decision points, vs ground truth
- Per-bucket value error (aggro-favoring vs control-favoring states) — the
  goal-4 gate metric, inherited from the search wave

## References

- Moravčík et al., *DeepStack* (2017) — counterfactual value networks,
  continual re-solving
- Brown & Sandholm, *Safe and Nested Subgame Solving* (NeurIPS 2017) — the
  off-tree answer
- Brown & Sandholm, *Libratus / Superhuman AI for heads-up no-limit poker*
  (Science 2018)
- Brown et al., *ReBeL* (2020) — self-play PBS value learning, the loop this
  wave transplants
- Schmid et al., *Student of Games* (2023) — the unified treatment
- Kreps & Wilson, *Sequential Equilibria* (1982) — off-path beliefs
- wave/search/README.md — Exit 1 (activation trigger), goal-4 gate,
  strategy-fusion section
- reports/exp-02-flat-mc.md — the searcher whose blind spots this wave exists
  to fix
