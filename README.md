# Manabot

Manabot is a project to train superhuman
[Magic: The Gathering](https://magic.wizards.com/) bots.

It has two halves:

- **managym** (Rust): a deterministic rules engine built for learning. It
  handles fast batched stepping, exact clone/fork for search, replay, and
  per-player observations that keep hidden information hidden. Python bindings
  come through PyO3.
- **manabot** (Python): the agents. It includes self-play training, search,
  belief models over hidden cards, and an arena for evaluating checkpoints
  against fixed opponents.

We are currently focusing on a very small but fairly complicated set of decks:
[UR Lessons versus GW Allies](content/DECKS.md), from the *Avatar: The Last
Airbender* set.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a Rust toolchain.

```bash
git clone git@github.com:loopflowstudio/manabot.git
cd manabot
uv sync --python 3.12 --extra dev
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python

uv run manabot train          # small CPU training run, under a minute, no W&B
```

`uv run manabot train` trains a small agent and saves checkpoints to
`.runs/local/`. Training recipes are JSON files under
[experiments/regimes/](experiments/regimes/). The current GPU baseline is
`uv run manabot train --regime experiments/regimes/batched-baseline.json`.
[manabot/README.md](manabot/README.md) covers presets, remote GPU runs,
simulation, and evaluation.

There is also a local browser table (`./scripts/play`, needs Node) for playing
one game against a checkpoint or stepping through a replay. It is for
inspecting what a bot does, not a way to play Magic.

## How it works

**The environment.** managym runs a full game of Magic: priority, the stack,
combat, triggers, and state-based actions. At every decision it gives each
player an observation of only what that player can see: players, cards, and
permanents as typed objects, plus the list of legal actions with the objects
each one targets. Cards are authored as typed programs, not parsed from Oracle
text, so the model can embed what a card does as well as its name. The
observation and action encoding is versioned as a *world*
([WORLDS.md](WORLDS.md)). Checkpoints only load in the world they were trained
in, and a nightly learnability check catches encodings that make two
different choices look identical.

**The agent.** One network encodes every visible object, mixes them with
attention, and scores each legal action together with its focus objects. A
value head predicts the game outcome. Every decision, from casting and
targeting to attacking, blocking, and paying costs, goes through that single
action head.

**Learning.** The main loop is self-play from terminal reward. The agent plays
both decks against current copies of itself, and many games step in lockstep
so that inference is batched. The update is a clipped policy gradient,
regularized toward both the collection policy and a fixed reference policy,
with advantage filtering, annealed step sizes, and an EMA of the weights for
evaluation. Those choices follow
[Ataraxos](https://doi.org/10.1038/s41586-026-11036-y) (Sokota et al., 2026),
and each one is a separate switch that we test on Magic rather than adopt on
the paper's authority ([research program](manabot/RESEARCH.md)).
Training recipes are declared as JSON *regimes*
([experiments/regimes/](experiments/regimes/)) and can run locally or on
rented GPUs ([remote training](docs/remote-training.md)).

**Hidden information and search.** Belief samplers learn a distribution over
the opponent's hidden cards from a frozen policy's own games
([belief sampler](docs/belief-sampler.md)). Search samples hands from that
belief, uses the frozen policy for continuations, and distills its choices
back into the policy ([local policy search](docs/local-policy-search.md)).
This runs end to end but has not yet shown a strength gain over the policy
alone.

**Evaluation.** Trained agents play each other. A *Tournament* file names a
roster of checkpoints and fixed opponents, and the runner plays every pairing
on the same held-out deals, with both decks and both seats
([head-to-head](experiments/head-to-head.md)). Head-to-head play separates
checkpoints that look the same against a scripted opponent, and the arena
fits ratings across the whole population (`manabot/arena/`). Fixed anchors
(random, scripted greedy, and a stronger rule-based `heuristic` bot) give a
stable reference point across runs. Every game is replay-verified, claims
about a training method use independent training seeds, and experiments are
preregistered: the prediction and kill criteria are committed before the run
([experiments/](experiments/README.md)).

Right now the main open question is the training curriculum: which matchups,
opponents, and mix of self-play the agent should learn from (for example,
mirror matches versus balanced cross-deck play). Evaluation is kept narrow,
just enough to compare curricula, rather than testing current agents across a
wide range of environments.

## Scale

Training scale per agent. We add a row when a run sets a new high.

| Agent | Date | Hardware | Learner transitions | Updates |
|---|---|---|---:|---:|
| `small-100k`, 64×2, 64 streams ([capacity](experiments/model-capacity.md)) | 2026-10 | 1× L4 | ~51M | 100,000 iterations (~800k Adam steps) |

Learner transitions are agent decisions.

## Earlier results

Before self-play became the main loop, the project leaned on search. These
results come from that period and from earlier encodings, so their numbers are
not comparable with today's checkpoints:

- **Search was strong and cheap.** Flat determinized Monte Carlo with 256
  rollouts beat random 99% of the time. Every early trained policy was weaker
  than 16-rollout search ([exp-02](experiments/exp-02-flat-mc.md)).
- **Distillation beat PPO at equal cost**, 90.5% versus 52.7% against random
  ([exp-03](experiments/exp-03-distillation.md)). Naive expert iteration did
  not compound: round 1 lost 74–26 to round 0
  ([exp-07](experiments/exp-07-expert-iteration.md)).
- **Reward shaping hurt.** Terminal-only reward beat our shaped rewards by
  about 15 points ([exp-04](experiments/exp-04-potential-shaping.md)).
- **Win rate hid incompetence.** The search pilot could not play a control
  deck: it found the right line at most 39% of the time in competency
  scenarios, and random beat it in 3 of 5
  ([exp-09](experiments/exp-09-control-competency.md)). Matchup tables mostly
  measured the pilot and card quality
  ([exp-08b](experiments/exp-08b-ancestral-dose.md)).

## Not built yet

- **More cards.** Cards are authored as typed programs rather than parsed
  Oracle text ([content/semantic/](content/semantic/README.md)), so that most
  new cards can be added as content without engine changes. The pool is still
  two decks.
- **Limited.** The long-term target is sealed and draft play: building decks
  from a pool, then piloting them. Nothing in the repository drafts or builds
  decks yet.

## Map of the repository

| Path | What it is |
|---|---|
| `managym/` | Rust rules engine, vectorized env, and PyO3 bindings ([README](managym/README.md)) |
| `manabot/` | Agents: models, training, search, beliefs, arena, verification ([README](manabot/README.md)) |
| `experiments/` | Preregistered experiment reports, protocols, and data |
| `content/` | Card and deck definitions compiled to a typed IR |
| `conformance/` | Rules conformance fixtures |
| `docs/` | Architecture, training operations, rules, and research notes ([index](docs/README.md)) |
| `paper/` | A working paper draft |
| `ops/`, `deploy/` | Remote training hardware and deployment |
| `academy/`, `frontend/`, `protocol/` | Local browser UI for testing a bot by hand and reviewing replays |
| `wave/` | Working plans for the active areas of work |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the full system design. In
short, managym owns the rules and legality, manabot owns inference and
decisions, and the play UI only displays what managym allows the viewer to
see.

## Development

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
- **Academy**: the local bot-testing UI (`academy/`, `frontend/`).
- **Etude**: a separate system used to launch and record experiments
  ([`loopflowstudio/etude`](https://github.com/loopflowstudio/etude)). Older
  documents also use *Etude* and *Fantasia* for this project.

Manabot is an independent research project and unofficial Fan Content
permitted under the Wizards of the Coast
[Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy).
Not approved or endorsed by Wizards. Magic: The Gathering and its card names
are property of Wizards of the Coast LLC.
