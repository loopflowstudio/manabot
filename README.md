# Manabot

Manabot is a project to train superhuman
[Magic: The Gathering](https://magic.wizards.com/) bots. A manabot learns by
playing itself: it sees only what a player would see, chooses among the legal
actions, and finds out who won when the game ends.

> **Built with two companion projects from Loopflow Studio:**
>
> - **[Loopflow](https://github.com/loopflowstudio/loopflow)** is how
>   coding agents are coordinated to write the code.
> - **[Etude](https://github.com/loopflowstudio/etude)** launches training and evaluation
>   runs and tracks what they record.

## The system

![The rules engine sends each seat's view to the policy network, which returns a chosen action. Decisions and results go to the learner, which sends new weights to the network. Checkpoints go to evaluation, which never feeds back into training.](docs/images/system.svg)

- **The rules engine** (`managym/`, Rust) runs full games of Magic, many at
  once, deterministically from a seed. It decides what is legal, plays forced
  choices itself, shows each seat only what that seat may see, and can clone,
  replay or re-deal a game exactly. Every card is one Rust function in the
  engine's own rules types, so a new card is content and new behavior is a new
  rules variant, never card-specific logic
  ([the rules engine](docs/engine.md)).
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

The observation and action encoding is versioned as a *world*
([WORLDS.md](WORLDS.md)); checkpoints load only in the world they were trained
in. **[Start here](docs/start-here.md)** walks from this picture down to the
code of each training stage.

## What it trains on now

The current world is *Avatar: The Last Airbender* Limited. Every card in the
set is defined as a typed rules program (336 supported in
[world w17](WORLDS.md)), and training draws both seats of each game at random
from real trophy decks: 134 seven-win Premier Draft lists admitted whole from
the [17Lands public dataset](https://www.17lands.com/public_datasets) (CC BY
4.0), frozen in [`experiments/tla_corpus.py`](experiments/tla_corpus.py) and
the [pilot protocol](experiments/protocols/tla-trophy-213-v10.md).

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
or loss only: our shaped rewards lost to it
([exp-04](experiments/exp-04-potential-shaping.md)).

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

## Scale

Training scale per agent, measured in **learner decisions**: the decisions
the learning policy made and trained on. Decisions with a single legal action
are played automatically and not counted, and neither are the opponent seat's
decisions in self-play. This unit does not depend on model size or batching,
unlike iterations or Adam steps, and does not depend on hardware, unlike
dollars or hours. We add a row when a run sets a new high.

| Agent | Date | Model | Learner decisions | Decisions per iteration × iterations | Hardware |
|---|---|---|---:|---:|---|
| first-light PPO ([report](experiments/first-light-run-1.md)) | 2026-03 | width 64 | 262k | 512 × 512 | CPU |
| matched-cost PPO ([exp-03](experiments/exp-03-distillation.md)) | 2026-07 | 100k params | 5.8M | 2,048 × 2,850 | M4 Max |
| `small-26k` ([capacity](experiments/model-capacity.md)) | 2026-10 | 64×2, 188k params | 13.3M | 512 × 26,000 | 1× L4 |
| `small-63k`, continued | 2026-10 | 64×2 | 32.3M | 512 × 63,000 | 1× L4 |
| `small-100k`, continued | 2026-10 | 64×2 | 51.2M | 512 × 100,000 | 1× L4 |
| scale ladder v1, one seat ([ladder](experiments/scale-ladder.md)) | 2026-10 | 64×2 | 196.6M | 16,384 × 12,000 | 1× L4, $13.34 |

The first two rows predate the current world and encoding, so they record
scale only. The ladder run's $10 checkpoint (151.6M decisions) scored 0.69
against the `heuristic` anchor across the four deck matchups, and 0.70 as
Lessons against Allies. The run was closed before its $50 tier, once
self-play moved to learning from both seats.

Learning-rate schedules currently count iterations, not decisions. The same
recipe run at a different batch size therefore decays at a different number
of decisions: about 18M for `small-100k` versus about 575M for the ladder recipe.

## What we know

All numbers belong to the world they were measured in. The
[research program](manabot/RESEARCH.md#what-we-know) keeps the full table.

- **Self-play learns.** A 10,000-update cohort rose from 30.3% to 54.3%
  against scripted greedy on monitoring deals, mostly by learning to pilot the
  harder deck ([current baseline](experiments/current-baseline.md)).
- **Batching is the reliable speedup, and the smallest model kept up.** 64×2
  took 109–111 minutes at 4 streams and about 5 at 256
  ([longer learning](experiments/longer-learning-017.md)). Collection is 82.5%
  of training time ([training time](experiments/training-time.md)).
- **Borrowed techniques haven't shown a benefit yet.** Actor-only filtering
  measured +3.33 points, interval [−4.33, +11.33], at 1.45× wall time
  ([technique screen](experiments/ataraxos-technique-screen.md)).
- **Search and beliefs run end to end but add no strength yet**
  ([search and beliefs](docs/search-and-beliefs.md)).

From the earlier search era, in older encodings:

- **Search was strong and cheap.** Flat determinized Monte Carlo with 256
  rollouts beat random 99% of the time
  ([exp-02](experiments/exp-02-flat-mc.md)).
- **Win rate hid incompetence.** That same pilot could not play a control
  deck, and random beat it in 3 of 5 competency scenarios
  ([exp-09](experiments/exp-09-control-competency.md)). Matchup tables mostly
  measured the pilot and card quality
  ([exp-08b](experiments/exp-08b-ancestral-dose.md)).
- **Distillation beat PPO at equal cost**, 90.5% versus 52.7% against random
  ([exp-03](experiments/exp-03-distillation.md)), but expert iteration did not
  compound: round 1 lost 74–26 to round 0
  ([exp-07](experiments/exp-07-expert-iteration.md)).

## Not built yet

- **A trained agent on the trophy population.** The pilot's first run stopped
  at a permanent-capacity limit after five updates; its successor raises the
  capacity and recovers from oversized games, but hasn't run
  ([pilot evidence](experiments/tla-trophy-pilot.md)).
- **The rest of the set's decks.** 866 of the 1,000 sampled trophy lists were
  rejected for cards or rules the world doesn't yet define. Most are content
  rather than engine changes.
- **Limited.** The long-term target is sealed and draft play: building decks
  from a pool, then piloting them. Piloting random trophy decks is the first
  half; nothing in the repository drafts or builds decks yet.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a Rust toolchain.

```bash
git clone git@github.com:loopflowstudio/manabot.git
cd manabot
uv sync --python 3.12 --extra dev
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python

uv run manabot train                       # small CPU training run, under a minute
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

## Map of the repository

| Path | What it is |
|---|---|
| `managym/` | Rust rules engine, vectorized env, and PyO3 bindings ([README](managym/README.md)) |
| `manabot/` | Agents: models, training, search, beliefs, evaluation, verification ([README](manabot/README.md)) |
| `experiments/` | Preregistered experiment reports, protocols, and data |
| `content/` | Deck lists and card content notes |
| `conformance/` | Active rules and replay fixtures; [historical corpora in S3](docs/development.md#retained-evidence-and-archived-corpora) |
| `docs/` | The system, part by part; training operations; rules; research notes ([index](docs/README.md)) |
| `paper/` | A working paper draft |
| `deploy/` | Hosted service deployment |
| `academy/`, `academy/ui/`, `protocol/` | Local browser UI for testing a bot by hand and reviewing replays |
| `.lf/` | Loopflow Waves (`.lf/wave/`), chapters and prompts that organize agent work |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the full system design. In
short, managym owns the rules and legality, manabot owns inference and
decisions, and the play UI only displays what managym allows the viewer to
see.

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
