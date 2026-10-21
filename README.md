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

> **Built with two companion projects from Loopflow Studio:**
>
> - **[Loopflow](https://github.com/loopflowstudio/loopflow)** runs the
>   development. Coding agents do most of the implementation here, organized
>   into Waves, Tasks, and Flows. [How we use it ↓](#development-with-loopflow)
> - **[Etude](https://github.com/loopflowstudio/etude)** launches training
>   runs and tracks what they record. [How we use it ↓](#experiments-with-etude)

## Technical approach

**A rules engine built for learning.** managym runs a full game of Magic:
priority, the stack, combat, triggers, and state-based actions. It is
deterministic given a seed, steps many games in lockstep, and can clone a game
state exactly. That makes batched self-play, search over possible futures,
and replay verification cheap. At every decision it gives each player an
observation of only what that player can see: players, cards, and permanents
as typed objects, plus the list of legal actions with the objects each one
targets.

**Cards as typed programs.** Cards are not parsed from Oracle text and not
hand-coded per card name. Each card is authored as a typed program
([content/semantic/](content/semantic/README.md)), compiled to an IR, and run
by a generic interpreter that rejects anything it cannot execute. The model can
embed what a card *does* as well as its name, and most new cards are content
changes rather than engine changes. Rules behavior is pinned by conformance
fixtures ([conformance/](conformance/)).

**Versioned worlds.** The observation and action encoding is versioned as a
*world* ([WORLDS.md](WORLDS.md)). Checkpoints only load in the world they were
trained in, and a nightly learnability check catches encodings that make two
different choices look identical.

**One network, one action head.** One network encodes every visible object,
mixes them with attention, and scores each legal action together with its
focus objects. A value head predicts the game outcome. Every decision, from
casting and targeting to attacking, blocking, and paying costs, goes through
that single action head.

**Self-play from terminal reward.** The agent plays both decks against current
copies of itself, and many games step in lockstep so that inference is
batched. The update is a clipped policy gradient, regularized toward both the
collection policy and a fixed reference policy, with advantage filtering,
annealed step sizes, and an EMA of the weights for evaluation. Those choices
follow [Ataraxos](https://doi.org/10.1038/s41586-026-11036-y) (Sokota et al.,
2026), and each one is a separate switch that we test on Magic rather than
adopt on the paper's authority ([research program](manabot/RESEARCH.md)).
Reward is win or loss only: our shaped rewards lost to it
([exp-04](experiments/exp-04-potential-shaping.md)).

**Hidden information and search.** Belief samplers learn a distribution over
the opponent's hidden cards from a frozen policy's own games
([belief sampler](docs/belief-sampler.md)). Search samples hands from that
belief, uses the frozen policy for continuations, and distills its choices
back into the policy ([local policy search](docs/local-policy-search.md)).
This runs end to end but has not yet shown a strength gain over the policy
alone.

**Evaluation by head-to-head play.** Trained agents play each other. A
*Tournament* file names a roster of checkpoints and fixed opponents, and the
runner plays every pairing on the same held-out deals, with both decks and
both seats ([head-to-head](experiments/head-to-head.md)). The arena fits
ratings across the whole population (`manabot/arena/`), and a rule-based
`heuristic` bot gives a stable reference point across runs. Every game is
replay-verified.

**Preregistered experiments.** The question, numeric prediction, kill
criteria, and cost cap are committed to git before a run; without that commit
a result is exploratory, not evidence. Frozen contracts record the hashes of
their inputs, claims about a training method need independent training seeds,
and refutations stay on the record ([experiments/](experiments/README.md)).

Right now the main open question is the training curriculum: which matchups,
opponents, and mix of self-play the agent should learn from (for example,
mirror matches versus balanced cross-deck play). Evaluation is kept narrow,
just enough to compare curricula, rather than testing current agents across a
wide range of environments.

## Development with Loopflow

Most of the code in this repository is written by coding agents, and
[**Loopflow**](https://github.com/loopflowstudio/loopflow) (`lf`) is how that
work is organized and run:

- **Waves** are standing areas of work, each with a charter, running memory,
  and numbered working documents under [wave/](wave/README.md): `intelligence`
  (training, search, evaluation), `rules` (managym), and `game` (Academy).
  Each wave declares which paths it owns.
- **Tasks** are the units of work inside a wave, tracked in Linear. Task IDs
  (`MTG-…`, `INT-…`, `RUL-…`) show up in experiment and report names.
- **Flows** chain agent steps such as kickoff, implement, review, gate, and
  land, and can wait on other Tasks before starting
  ([.lf/flows/](.lf/flows/)). Repo-specific skills and steps live in
  [.lf/skills/](.lf/skills/) and [.lf/steps/](.lf/steps/).
- Each agent session works in its own git worktree and lands through a PR.
  The conventions agents follow are in [AGENTS.md](AGENTS.md).

## Experiments with Etude

[**Etude**](https://github.com/loopflowstudio/etude) launches training runs on
your own or rented machines and tracks what they record. Manabot is its first
heavy user:

- **Run tracking.** Training-time reports read Etude's run records directly,
  from local storage or S3, with `uv run --extra reports manabot
  training-report` ([experiment metrics](docs/experiment-metrics.md)). The
  `etude-sdk` client is an optional `reports` extra.
- **Remote training.** Regimes run on rented GPUs as jobs that survive
  disconnects, write checkpoint snapshots to S3, enforce a dollar ceiling, and
  shut themselves down ([remote training](docs/remote-training.md)). The
  launcher currently lives in manabot (`manabot deploy`, `manabot/remote/`)
  and targets RunPod.
- **Records that outlive the run.** Experiment IDs, contracts, and receipts
  are frozen evidence and are never renamed.

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
`.runs/local/`. Training recipes are JSON *regimes* under
[experiments/regimes/](experiments/regimes/). The current GPU baseline is
`uv run manabot train --regime experiments/regimes/batched-baseline.json`.
[manabot/README.md](manabot/README.md) covers presets, remote GPU runs,
simulation, and evaluation.

There is also a local browser table (`./scripts/play`, needs Node) for playing
one game against a checkpoint or stepping through a replay. It is for
inspecting what a bot does, not a way to play Magic.

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

- **More cards.** Because cards are typed programs, most new cards can be
  added as content without engine changes, but the pool is still two decks.
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
| `wave/`, `.lf/` | Loopflow Waves, Flows, and skills that organize agent work |

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
