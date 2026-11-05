# Architecture

Manabot is three packages with one direction of dependency and one kind of
truth per package. This page says who owns what, which contracts the code
enforces today, and what exists beside the training loop without being on it.
How each part works is in its own guide; start from
[Start here](start-here.md).

![The rules engine sends each seat's view to the policy network, which returns a chosen action. Decisions and results go to the learner, which sends new weights to the network. Checkpoints go to evaluation, which never feeds back into training.](images/system.svg)

- [Ownership](#ownership)
- [Contracts the code enforces](#contracts-the-code-enforces)
- [Beside the loop](#beside-the-loop)
- [Non-goals](#non-goals)

## Ownership

- **managym is the world** (`managym/`, Rust; `import managym` in Python). It
  owns card meaning, match state, legality, decisions and Commands, committed
  events, viewer-safe observations, deterministic replay, exact forks, and the
  hidden-hand domain. It never chooses among a learner's legal actions
  ([the rules engine](engine.md)).
- **manabot is the agent** (`manabot/`, Python). It owns the policy and value
  network, self-play training, checkpoints, search, beliefs, evaluation and
  experiment records. Inside it, `sim` runs games and registered players,
  `eval` scores and rates, `training` learns, `experiment` declares and
  executes comparisons, and `infra.identity` owns canonical byte identities.
- **Academy is the inspection UI** (`academy/`, `academy/ui/`). It owns
  transport, interaction drafts, reconnect behaviour, presentation and replay
  navigation. It reconstructs no rules meaning, and nothing in training
  depends on it ([Academy](academy.md)).
- **Etude launches and records** (`loopflowstudio/etude`). Remote training and
  evaluation go through it; the machine is chosen at launch, never in the
  experiment ([training](training.md#remote-training)).

Imports follow the same order: Academy may import manabot and managym, manabot
imports managym and never Academy, managym imports neither, and `manabot.sim`
imports none of `eval`, `training`, `experiment` or `reports`.
`tests/test_package_boundaries.py` enforces all of it. Telemetry, profilers,
caches and UI state may observe or accelerate these authorities; they never
become a second source of truth.

## Contracts the code enforces

**One rules authority.** Only managym commits a state transition. A transition
is `(revision N, decision N) + Command(expected=N) → committed events +
revision N+1`; a Command against another revision fails closed
(`SemanticError::StaleRevision`). Academy, a person or a manabot selects a
Command; none patches state, invents legality or re-derives what a viewer may
see. Academy's live play still lowers its protocol-v1 `Command` to a
positional `step` through `DecisionContext.action_by_offer`; the semantic
`DecisionFrame` / `Command` / `TransitionReceipt` contract in
`managym/src/decision.rs` is the engine API, not yet Academy's wire path
([experience protocol v1](architecture/experience-protocol-v1.md)).

**Forced decisions are played by the engine.** A decision reaches a player
only with at least two legal actions (`skip_trivial`, on everywhere). Every
decision count in this repository excludes forced choices.

**Observations are viewer-safe at the source.** `Observation::for_player`
holds everything one seat may see and nothing else; legal actions appear only
in the chooser's view; decklist disclosure lives on `PlayerConfig.disclosure`.
Viewer-equivalent states produce byte-identical observations. The policy
tensor, the LLM text and the learning tokens all derive from the same
observation, and a mechanic reaches all three in the change that adds it
([observations](observation-and-model.md)).

**Identity is typed.** An `ObjectRef` carries its incarnation and fails after
a zone change. An action index is valid only in one decision epoch. The
observation and action encoding is a *world* (`managym.WORLD_VERSION`,
[WORLDS.md](../WORLDS.md)); checkpoints bind world, deck setup and input
schema and load only where the digest matches. Every durable value has a
canonical serialization and digest (`manabot.infra.identity`), and locators
are not identity.

**Replay is exact.** A game regenerates from its deal seed, configuration and
Commands, and every pre- and post-state digest must match. RNG streams derive
from recorded seeds, so scheduling cannot select an outcome. Evaluation scores
only replay-verified terminal games
([forks, replay and determinization](engine.md#forks-replay-and-determinization)).

**Branching is a full clone.** Search forks a `Game` by cloning it and steps
the clone; a rules-only witness checks the source is untouched. Undo journals
and copy-on-write pages were rejected on preregistered thresholds
([decision record](https://github.com/loopflowstudio/manabot/blob/055dddab1/docs/benchmarks/search-branching-decision-v1.md)).
Determinized PUCT and flat Monte Carlo are named as such: separate
perfect-information trees per sampled world, aggregated at the root, with no
claim to information-set consistency ([beside the loop](#beside-the-loop)).

**Hidden information has one domain.** `managym/src/possible_worlds.rs` owns
the hidden-hand domain, the five-variant `WorldQuery`, compatible-deal weights
and materialization; `manabot.belief` owns probability over that domain. A
query never reveals whether the actual hidden hand satisfies it, and empty
support is an explicit result.

**Training is declared data.** A `TrainingRegime` (one module per experiment
under `experiments/`) names the world, decks, model and ordered stages; one
runner executes it, writes world-bound raw and EMA checkpoints, records every
attempt in `VerifyStore`, and ships metrics through Etude. Regimes refuse
belief inputs; the trained model sees the current observation only
([how a manabot learns](learning.md), [training](training.md)).

**Evaluation never promotes.** A player is an immutable `PlayerRegistration`;
a rating belongs to one `EvaluationKey` (world, content suite, viewer
boundary, arena version, compute class) and never crosses it. Candidates play
the same deals with seats and decks swapped, uncertainty resamples whole
deals, and a seat-aware Bradley-Terry fit gives local Elo coordinates with
`heuristic` and frozen checkpoints as working anchors. No win rate, rating or
report promotes a checkpoint; scores never enter the loss
([evaluation](evaluation.md)).

**Evidence is preregistered.** The question, prediction, kill criteria and
cost cap are committed before a run; the training seed is the unit of
evidence for a method; refutations stay on the record
([experiments](experiments.md)).

## Beside the loop

These exist and run, and none of them is on the training path today.

- **Search players** (`manabot/sim/search`). `flat_mc.make_player` builds
  every kind from a spec: `search` (flat Monte Carlo over determinized worlds,
  no network), `determinized_puct` and `agent_puct` (one tree per sampled
  world, values as win scores, not information-set-consistent),
  `policy_search`, `value_greedy`, `value_search`, and `local_update`.
  `LocalUpdateTeacher` is the current search-improvement code: for one
  checkpoint it maximizes `Σ p(a)Q(a) − α KL(p‖uniform) − β KL(p‖base)` over
  rolled-out worlds, following Ataraxos supplement S3.7, and retains a
  replayable receipt. Worlds come from the compatible-deal prior, a learned
  sampler, or the small exact tracker. None has shown a strength gain over the
  policy alone, and every determinized method shares the strategy-fusion
  caveat above.
- **Beliefs** (`manabot/belief`). `managym/src/possible_worlds.rs` owns the
  hidden-hand domain; `manabot.belief` puts probability on it: `BeliefState`
  over one `PossibleWorldSpace`, a tracker fed by viewer observations, and the
  `AutoregressiveBeliefSampler`, which learns opponent hand counts from a
  frozen policy's games. The `collect_belief` and `train_belief` stages fit it
  on an immutable dataset (`uv run manabot train --regime-name belief_sampler
  --seed 421 --out .runs/belief-pilot` is the bounded pilot), and
  `python -m manabot.reports.belief` scores a sampler by held-out joint NLL,
  Brier and calibration. Consumers are search players and a demo seam only;
  `ActorCritic` has an optional belief channel that regimes reject, and
  Academy's live belief tracking was removed. Detail:
  [search-and-beliefs.md at 055dddab1](https://github.com/loopflowstudio/manabot/blob/055dddab1/docs/search-and-beliefs.md).
- **Academy Study** (`academy/study_branch.py`, the replay page) forks a
  retained engine root, executes structured Commands on the clone, proves
  source and sibling isolation, and returns through a consuming receipt.
  Its decision evidence is fixture-served.
- **LLM players** ([evaluation](evaluation.md#llm-players)) read the text observation and choose
  Commands through the same authority as everyone else.

## Non-goals

- A general-purpose Magic rules library independent of the training worlds.
- Natural-language parsing or LLM authority over legality at runtime.
- Multiplayer, general-sum play, Commander breadth or broad format legality.
- A second semantic authority, in Academy or anywhere else, even as a
  compatibility layer.
- Claiming that exact beliefs solve strategy fusion, or that one rating
  describes every matchup.
