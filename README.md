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

## How we train now

This section describes the approach as of October 2026.

**Self-play RL is the main loop.** Until mid-2026 the project leaned on
search. Determinized Monte Carlo was strong and cheap, and we distilled it into
networks. But search spends compute on every decision, it handles hidden
information poorly, and naive expert iteration on top of it did not compound
(results below). Self-play on a fast engine scales with training compute
instead, so it is now the primary path. Search and belief models are being
rebuilt as policy-improvement steps on top of it.

**Ataraxos is a starting point, not a blueprint.** Sokota et al.'s
[Stratego agent](https://doi.org/10.1038/s41586-026-11036-y) is the clearest
recent example of conventional deep RL reaching champion level in a large
imperfect-information game. The current baseline adapts its recipe:

- policy updates damped by KL to the collection policy and to a fixed
  reference policy (uniform over action types, then over actions)
- annealed learning-rate and regularization schedules
- advantage filtering
- an EMA of the weights for evaluation

Each piece is a separate switch in the regime file, so it can be tested on its
own. So far, none has shown a measured benefit for Magic on its own.
Restricting advantage filtering to the actor gained +3.3 points
[−4.3, 11.3] for 45% more training cost
([ETU-105](experiments/ataraxos-technique-screen.md)). Pooling and
filter-floor variants showed no demonstrated benefit
([ETU-106](experiments/pooling-filter-followup.md)). Magic also differs from
Stratego in ways that matter: structured actions (targets, modes, payments),
a large and changing card pool, and deckbuilding. The
[learning research guide](manabot/LEARNING_RESEARCH_GUIDE.md) separates what
the paper shows from what we are only conjecturing.

**Throughput comes from batching.** The trainer steps many games in lockstep
and batches inference on one GPU. Etude launches rented GPUs (RunPod) and
records every run. At equal experience, a 64×2 attention model trains in
about 110 minutes with 4 streams and about 5 minutes with 256. Larger models
have not helped yet. At 524k transitions the small model scored 46–60%
against scripted greedy and 74–89% against random
([longer-learning-017](experiments/longer-learning-017.md)).

**Evaluation uses fixed opponents.** Checkpoints play fixed anchors (random,
scripted greedy, and a stronger `heuristic` bot) on held-out paired deals,
from both decks and both seats, and every game is replay-verified. Independent
training seeds, not games, are the unit for claims about a method. Greedy is
nearing saturation: one checkpoint scored 79.5% against greedy but 57% against
`heuristic` on the same deals. A world-pinned rated arena across many
players exists from earlier work, but has not been rerun on the current
encoding.

**Encoding audits catch silent capacity bugs.** Two audits in October found
choices the policy literally could not tell apart. Attack and decline had
identical inputs. So did different targets, modes, and abilities on the same
source. Both are fixed (worlds w5 and w6 in [WORLDS.md](WORLDS.md)), and a
nightly learnability check now catches regressions.

**Beliefs and search are being rebuilt for the RL policy.** Belief samplers
over the opponent's hidden cards train from a frozen policy's own games
([belief sampler](docs/belief-sampler.md)). Local search can use a frozen
policy for continuations and sampled hands for the hidden state, then distill
the result back into the policy
([local policy search](docs/local-policy-search.md)). Both run end to end and
neither has shown a strength gain yet.

## Earlier results

Every experiment has a report under [experiments/](experiments/README.md),
with its prediction and kill criteria committed before it ran. Negative
results stay on the record. These came from the search-first period and from
earlier encodings, so their checkpoints are not comparable with today's:

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
