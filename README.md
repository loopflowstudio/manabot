# Manabot

Manabot is a project to train superhuman
[Magic: The Gathering](https://magic.wizards.com/) bots. A manabot learns by
playing itself: it sees only what a player would see, chooses among the legal
actions, and finds out who won when the game ends. Nobody tells it which plays
were good, and three things make that hard:

- **The opponent learns too.** The best play depends on a policy that keeps
  changing, so a good win rate against the latest self says little.
- **Actions aren't a fixed list.** A legal action can name any object in play,
  so the network scores what's offered rather than filling a fixed output layer.
- **Cards change the rules.** Each pairing of decks is close to its own game,
  and a new card can add a rule.

> **Built with two companion projects from Loopflow Studio:**
>
> - **[Loopflow](https://github.com/loopflowstudio/loopflow)** is how
>   coding agents are coordinated to write the code.
> - **[Etude](https://github.com/loopflowstudio/etude)** launches training and evaluation
>   runs and tracks what they record.

- [The system](#the-system)
- [What it trains on now](#what-it-trains-on-now)
- [How a manabot learns](#how-a-manabot-learns)
- [How strength is measured](#how-strength-is-measured)
- [Contracts the code enforces](#contracts-the-code-enforces)
- [Beside the loop](#beside-the-loop)
- [Scale](#scale)
- [What we know](#what-we-know)
- [Not built yet](#not-built-yet)
- [Quickstart](#quickstart)
- [Experiment data](#experiment-data)
- [Guides and map](#guides-and-map)

## The system

![The rules engine sends each seat's view to the policy network, which returns a chosen action. Decisions and results go to the learner, which sends new weights to the network. Checkpoints go to evaluation, which never feeds back into training.](docs/images/system.svg)

- **The rules engine** (`managym/`, Rust; `import managym` in Python) runs full
  games of Magic, many at once, deterministically from a seed. It owns card
  meaning, match state, legality, decisions and Commands, committed events,
  viewer-safe observations, replay, exact forks and the hidden-hand domain. It
  plays forced choices itself and never chooses among a learner's legal
  actions. Every card is one Rust function in the engine's own rules types, so
  a new card is content and new behavior is a new rules variant, never
  card-specific logic ([the rules engine](docs/engine.md)).
- **The policy network** (`manabot/model/`) embeds every visible object,
  including what each card *does*, mixes them with attention, and scores each
  legal action with the objects it names through one action head. A value
  head predicts the outcome
  ([observations and the policy network](docs/observation-and-model.md)).
- **The learner** (`manabot/training/`) turns finished games into one weight
  update per batch ([how a manabot learns](docs/learning.md)).
- **Evaluation** (`manabot/eval/`) measures strength on held-out deals from
  outside the loop. Its scores never enter the loss
  ([evaluation](docs/evaluation.md)).

Two more pieces sit around the loop. **Academy** (`academy/`, `academy/ui/`)
is the browser UI for inspecting a bot or a replay by hand; it owns transport,
drafts and presentation, reconstructs no rules meaning, and nothing in
training depends on it ([Academy](docs/academy.md)). **Etude** launches remote
training and evaluation and records what they produce; the machine is chosen
at launch, never in the experiment ([training](docs/training.md#remote-training)).

Inside `manabot/`, `sim` runs games and registered players, `eval` scores and
rates, `training` learns, `experiment` declares and executes comparisons, and
`infra.identity` owns canonical byte identities. Imports follow one direction:
Academy may import manabot and managym, manabot imports managym and never
Academy, managym imports neither, and `manabot.sim` imports none of `eval`,
`training`, `experiment` or `reports` (`tests/test_package_boundaries.py`
enforces all of it). The observation and action encoding is versioned as a
*world* ([WORLDS.md](WORLDS.md)); checkpoints load only in the world they were
trained in.

## What it trains on now

The current world is *Avatar: The Last Airbender* Limited. Every card in the
set is defined as a typed rules program (336 supported in
[world w17](WORLDS.md)), and training draws both seats of each game at random
from real trophy decks: 134 seven-win Premier Draft lists admitted whole from
the [17Lands public dataset](https://www.17lands.com/public_datasets) (CC BY
4.0), built by [`experiments/tla_corpus.py`](experiments/tla_corpus.py) and
frozen in the pilot's corpus and
[regime](experiments/regimes/tla-trophy-213-v10.json)
([protocol](experiments/protocols/tla-trophy-213-v10.md)).

The pilot's question is whether the recipe learns on a population of real
decks at all. Its first run learned for five updates before a capacity limit
stopped it, and no trained checkpoint on this population exists yet
([pilot evidence](experiments/tla-trophy-pilot.md)). The curriculum questions
that follow, which matchups, opponents and mix of self-play to learn from, are
in the [research program](manabot/RESEARCH.md). Earlier runs trained on one
fixed matchup, [UR Lessons versus GW Allies](content/DECKS.md), which the older
reports and the local evaluation check still use.

## How a manabot learns

![One training iteration: play, assign credit, filter and update, then repeat with new weights. Weights also feed an EMA, checkpoints and evaluation, which never feed back into the loss.](docs/images/learning-loop.svg)

The only feedback is who won. Each iteration has four steps:

1. **Play.** Many games step in lockstep so inference is batched, and one
   network plays them. By default both seats learn.
2. **Assign credit.** The game result becomes a per-decision advantage and a
   value target, computed once with generalized advantage estimation.
3. **Filter.** Only decisions with the largest advantages, about a quarter of
   the batch, stay in the update.
4. **Update.** One pass of a clipped policy gradient, regularized toward both
   the policy that played the batch and a fixed reference policy. The batch is
   then discarded.

An EMA of the weights and annealed step sizes complete the recipe. Each piece
has a source and is a separate switch we test on Magic rather than adopt on a
paper's authority ([influences](docs/learning.md#influences)). Reward is win
or loss only: shaped rewards bought nothing over it
([exp-04](experiments/exp-04-potential-shaping.md)).
[How a manabot learns](docs/learning.md) walks each step down to the code.

## How strength is measured

Two players meet on the same seeded deal four times, swapping decks and seats,
and only finished, replay-verified games count. Uncertainty resamples whole
deals. A rule-based `heuristic` bot and frozen checkpoints anchor the scale,
ratings are fit across the whole population, and tactical scenarios and
bounded attacks probe what a win rate hides ([evaluation](docs/evaluation.md)).

A result counts as evidence only when its question, numeric prediction, kill
criteria and cost cap were committed to git before the run, and claims about a
method need independent training seeds. Refutations stay on the record
([experiments](docs/experiments.md)).

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
([experience protocol v1](docs/architecture/experience-protocol-v1.md)).

**Forced decisions are played by the engine.** A decision reaches a player
only with at least two legal actions (`skip_trivial`, on everywhere). Every
decision count in this repository excludes forced choices.

**Observations are viewer-safe at the source.** `Observation::for_player`
holds everything one seat may see and nothing else; legal actions appear only
in the chooser's view; decklist disclosure lives on `PlayerConfig.disclosure`.
Viewer-equivalent states produce byte-identical observations. The policy
tensor, the LLM text and the learning tokens all derive from the same
observation, and a mechanic reaches all three in the change that adds it
([observations](docs/observation-and-model.md)).

**Identity is typed.** An `ObjectRef` carries its incarnation and fails after
a zone change. An action index is valid only in one decision epoch. Checkpoints
bind world, deck setup and input schema and load only where the digest
matches. Every durable value has a canonical serialization and digest
(`manabot.infra.identity`), and locators are not identity.

**Replay is exact.** A game regenerates from its deal seed, configuration and
Commands, and every pre- and post-state digest must match. RNG streams derive
from recorded seeds, so scheduling cannot select an outcome. Evaluation scores
only replay-verified terminal games
([forks, replay and determinization](docs/engine.md#forks-replay-and-determinization)).

**Branching is a full clone.** Search forks a `Game` by cloning it and steps
the clone; a rules-only witness checks the source is untouched. Determinized
search is named as such: separate perfect-information trees per sampled world,
aggregated at the root, with no claim to information-set consistency.

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
([training](docs/training.md)).

**Evaluation never promotes.** A player is an immutable `PlayerRegistration`;
a rating belongs to one `EvaluationKey` (world, content suite, viewer
boundary, arena version, compute class) and never crosses it. Candidates play
the same deals with seats and decks swapped, uncertainty resamples whole
deals, and a seat-aware Bradley-Terry fit gives local Elo coordinates with
`heuristic` and frozen checkpoints as working anchors. No win rate, rating or
report promotes a checkpoint; scores never enter the loss.

**Evidence is preregistered.** The question, prediction, kill criteria and
cost cap are committed before a run; the training seed is the unit of
evidence for a method; refutations stay on the record.

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
  policy alone.
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
- **LLM players** ([evaluation](docs/evaluation.md#llm-players)) read the text
  observation and choose Commands through the same authority as everyone else.

## Scale

Training scale per agent, measured in **learner decisions**: the decisions
the learning policy made and trained on. Decisions with a single legal action
are played automatically and not counted, and neither are a frozen opponent's;
when both seats learn, both seats count. This unit does not depend on model size or batching,
unlike iterations or Adam steps, and does not depend on hardware, unlike
dollars or hours. We add a row when a run sets a new high.

| Agent | Date | Model | Parameters | Learner decisions | Decisions per iteration × iterations | Hardware |
|---|---|---|---:|---:|---:|---|
| first-light PPO ([report](experiments/first-light-run-1.md)) | 2026-03 | width 64, no attention | not recorded | 262k | 512 × 512 | CPU |
| matched-cost PPO ([exp-03](experiments/exp-03-distillation.md)) | 2026-07 | width 64, no attention | 50,306 | 5.8M | 2,048 × 2,850 | M4 Max |
| `small-26k` ([capacity](experiments/model-capacity.md), [head-to-head](experiments/head-to-head.md)) | 2026-10 | 64 wide × 2 attention layers | 188,482 | 13.3M | 512 × 26,000 | 1× L4 |
| `small-63k`, continued | 2026-10 | 64×2 | 188,482 | 32.3M | 512 × 63,000 | 1× L4 |
| `small-100k`, continued | 2026-10 | 64×2 | 188,482 | 51.2M | 512 × 100,000 | 1× L4 |
| scale ladder v1, one seat ([ladder](experiments/scale-ladder.md)) | 2026-10 | 64×2 | not retained | 196.6M | 16,384 × 12,000 | 1× L4, $13.34 |
| MTG-183, ten Elemental decks, both seats ([protocol](experiments/protocols/elemental-183.md)) | 2026-10 | 64×2 | 196,482 | 331.2M | ≈55,700 (complete games) × 5,950 | 1× L4, four rental segments, $50 cap |

The parameter count of a 64×2 network depends on its inputs, not only its
width and depth. The card encoder's token vocabulary adds 64 parameters per
distinct learning token, card names among them, which is the 8,000 between the
two-deck and ten-deck runs. The fact encoder (input v5 and later) and the shared-effect encoder
(input v7) each hold about 0.8M parameters in hash-bucket embedding tables, so
the current TLA recipe has 1,883,650 with the same 64×2 attention core.
The first two rows predate the current world and
encoding, so they record scale only. The one-seat ladder's $10 checkpoint
(151.6M decisions) scored 0.69 against the `heuristic` anchor and 0.70 as
Lessons against Allies; it was closed before its $50 tier once self-play
moved to learning from both seats, and MTG-183 is the two-seat run that
followed.

Learning-rate schedules currently count iterations, not decisions. The same
recipe run at a different batch size therefore decays at a different number
of decisions: about 18M for `small-100k` versus about 575M for the ladder recipe.

## What we know

All numbers belong to the world they were measured in, and most come from one
training seed, so they guide the next experiment rather than settle a method.
Earlier findings are tabulated in the
[research program](manabot/RESEARCH.md#what-we-know).

- **One seed on ten decks kept improving for 331M decisions.** MTG-183 trained
  a 64×2 network from both seats on the ten-deck Elemental world (w13). Its
  score against the `heuristic` anchor on the full 10×10 deck grid rose from
  0.065 to 0.801 over 5,950 updates, and its rating from 525 to 1,248 Elo with
  the heuristic at 1,000. When the heuristic score flattened after update
  3,600, newer checkpoints still beat older ones: the endpoint scored 0.591
  [0.569, 0.612] against update 2,000 and 0.532 [0.511, 0.554] against update 3,600, with no
  significant intransitive triangles among 27 rated checkpoints
  ([protocol](experiments/protocols/elemental-183.md); rating report in
  `experiments/data/mtg183-rating/`).
- **Claude Haiku played that checkpoint to a draw.** Haiku 5.5 with high
  reasoning won 51 of 100 games against MTG-183's final policy, at $0.37
  per game and a median 1.4 seconds per decision
  ([Haiku versus MTG-183](experiments/haiku-mtg183.md)).
- **The anchors are moving targets.** A rewrite of the `heuristic` bot took it
  from 18.3% to 62.0% as UR Lessons against MTG-183's update-1,750 checkpoint,
  and from 10.8% to 34.3% as each of the ten decks against its GW and UB seats,
  without slower decisions
  ([heuristic strength](experiments/heuristic-strength.md)). Ratings are
  pinned to an anchor version and a world, never compared across them.
- **Learning from both seats replaced one-seat training.** The one-seat ladder
  reached 0.69 against the heuristic at $10
  ([scale ladder](experiments/scale-ladder.md)); matched per learner decision
  over four seeds per arm, two-seat learning was no worse and reached the same
  score about 1.4× sooner
  ([seat comparison](experiments/protocols/seat-comparison-v1.md)).
- **Batching is the reliable speedup, and the smallest model kept up.** 64×2
  took 109–111 minutes at 4 streams and about 5 at 256; larger models showed no
  advantage at this horizon, on two seeds
  ([longer learning](experiments/longer-learning-017.md)). Collection is 82.5%
  of training time ([training time](experiments/training-time.md)).
- **The trophy population is untested.** The TLA pilot's initialization scored
  7–25 against the heuristic before its run stopped; there is no trained
  checkpoint on random real decks yet
  ([pilot evidence](experiments/tla-trophy-pilot.md)).
- **Search and beliefs run end to end but add no strength yet**
  ([beside the loop](#beside-the-loop)).

## Not built yet

- **A trained agent on the trophy population.** The pilot's first run stopped
  at a permanent-capacity limit after five updates; its successor raises the
  capacity and recovers from oversized games, but hasn't run
  ([pilot evidence](experiments/tla-trophy-pilot.md)).
- **The full trophy population.** The pilot corpus was frozen when 134 of
  1,000 sampled lists admitted; the current world admits all 1,000, and no
  protocol or run uses the larger population yet.
- **Limited.** The long-term target is sealed and draft play: building decks
  from a pool, then piloting them. Piloting random trophy decks is the first
  half; nothing in the repository drafts or builds decks yet.

Not goals: a general-purpose Magic rules library independent of the training
worlds; natural-language parsing or LLM authority over legality at runtime;
multiplayer, general-sum play, Commander breadth or broad format legality; a
second semantic authority anywhere, even as a compatibility layer; claiming
that exact beliefs solve strategy fusion, or that one rating describes every
matchup.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a Rust toolchain.

```bash
git clone git@github.com:loopflowstudio/manabot.git
cd manabot
uv sync --python 3.12 --extra dev
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python

uv run manabot train                       # two-update CPU smoke, capped at two minutes
uv run python scripts/inspect_batch.py     # one real batch and how credit reaches each decision
```

`uv run manabot train` trains a small agent and saves checkpoints under
`.runs/`. Training recipes are typed *regimes* composed in Python, one module per
current experiment in [experiments/](experiments/), from reusable parts in
[manabot/recipes/builders.py](manabot/recipes/builders.py). The current GPU
baseline is
`uv run manabot train --regime-name batched_baseline --out .runs/batched/run`.
[docs/training.md](docs/training.md) covers running, monitoring and recovering
runs, and [manabot/README.md](manabot/README.md) covers the package.

There is also a local browser table (`./scripts/play`, needs Node) for playing
one game against a checkpoint or stepping through a replay. It is for
inspecting what a bot does, not a way to play Magic.

Versioned PyPI release preparation, platform support, CPU/CUDA installation
and maintainer publishing are in [the release guide](docs/development.md#releases).
[Manabot 0.2.0 is available on PyPI](https://pypi.org/project/manabot/0.2.0/).

## Experiment data

Every number above traces to a report in [experiments/](experiments/README.md)
and the files behind it. Reports are in git; the result files are not.
[`experiments/data.manifest.json`](experiments/data.manifest.json) records
each file's path, size and SHA-256, and the bytes live once per digest in the
team's R2 store, so a checkout is small and a report's evidence is still exact:

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg183-rating   # one report's files
uv run --extra artifacts python -m manabot.data fetch                                  # everything
```

`fetch` needs the `R2_*` credentials from Doppler `manabot/prd` (prefix the
command with `doppler run -p manabot -c prd --`), verifies every digest and
refuses to overwrite a file whose bytes differ. Training and evaluation runs
themselves are tracked by [Etude](https://github.com/loopflowstudio/etude):
`uv run etude report experiments/ --source r2://manabot/jobs --serve` renders
every experiment's curves and matches from the job store, and
`uv run etude eval experiments/elemental.py --source r2://manabot/jobs` plays
whatever matchups a report still wants. How a question becomes a report, and
what makes a result count, is in [experiments](docs/experiments.md).

## Guides and map

Each part of the system has one guide, written the same way: what it does, one
worked example, the key terms with the code that implements them, then what we
know and where the code lives.

| Guide | What |
|---|---|
| [engine.md](docs/engine.md) | The rules engine: one decision, forced choices, who chooses, observations, forks, replay, cards as programs, decks, worlds |
| [observation-and-model.md](docs/observation-and-model.md) | What the policy sees, how it scores legal actions and estimates value, policy input versions |
| [learning.md](docs/learning.md) | How a manabot learns: play, batch boundaries, credit assignment, filtering, the loss, reading a run, down to the code |
| [evaluation.md](docs/evaluation.md) | One deal, what counts, anchors and ratings, scenarios, frozen-policy attacks, LLM players, shared inference, card data |
| [experiments.md](docs/experiments.md) | How a question becomes a report: the claim rules, a worked example, failures worth reading, declaring a comparison |
| [training.md](docs/training.md) | Running training: regimes and stages, remote training, monitoring, numerical health, recovery, game failures |
| [experiment-metrics.md](docs/experiment-metrics.md) | Metric definitions the experiment reports read |
| [development.md](docs/development.md) | Environment, validation by change area, CI, releases, clean-machine play, retained evidence |
| [academy.md](docs/academy.md) | The browser UI: running it, the shared decision surface, presentation, trained challengers, the proof gate |

| Path | What it is |
|---|---|
| `managym/` | Rust rules engine, vectorized env, and PyO3 bindings ([README](managym/README.md)) |
| `manabot/` | Agents: models, training, search, beliefs, evaluation, verification ([README](manabot/README.md)) |
| `experiments/` | Preregistered experiment reports, protocols, and data |
| `content/` | Deck lists and card content notes |
| `conformance/` | Active rules and replay fixtures; [historical corpora in S3](docs/development.md#retained-evidence-and-archived-corpora) |
| `docs/` | The guides above, the [rules authority](docs/rules/), the [Academy wire protocol](docs/architecture/experience-protocol-v1.md) and dated [research](docs/research/) |
| `paper/` | A working paper draft |
| `deploy/` | Hosted service deployment |
| `academy/`, `academy/ui/`, `protocol/` | Local browser UI for testing a bot by hand and reviewing replays |
| `.lf/` | Loopflow Waves (`.lf/wave/`), chapters and prompts that organize agent work |

## Development

Read [TESTING.md](TESTING.md) before running checks. Never intentionally induce
load for testing.

```bash
# Rust (CI runs cargo test in debug, so validate in debug before landing)
cd managym && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test && cd ..

# Python
uv run --extra dev pytest tests/
```

Contributor and agent conventions are in [AGENTS.md](AGENTS.md). Package style
guides are in each package README.

## Names

- **manabot**: the bot and the Python library. You train *a* manabot.
- **managym**: the rules environment the bot lives in.
- **Academy**: the local bot-testing UI (`academy/`, `academy/ui/`).
- **Etude**: the separate system that launches and records experiments
  ([`loopflowstudio/etude`](https://github.com/loopflowstudio/etude)). Older
  documents also use *Etude* and *Fantasia* for this project.
- **Loopflow**: the agent development tool
  ([`loopflowstudio/loopflow`](https://github.com/loopflowstudio/loopflow)).

Manabot is an independent research project and unofficial Fan Content
permitted under the Wizards of the Coast
[Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy).
Not approved or endorsed by Wizards. Magic: The Gathering and its card names
are property of Wizards of the Coast LLC.
