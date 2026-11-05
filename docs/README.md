# docs/

Start with [How a manabot learns](learning.md): one training iteration, from a
game's first decision to the weight update. Then read
[ARCHITECTURE.md](ARCHITECTURE.md) for the system picture and who owns which
truth. Experiment reports live in
[experiments/](../experiments/README.md), and world versioning lives in
[WORLDS.md](../WORLDS.md).

## Current system

| Doc | What |
|---|---|
| [learning.md](learning.md) | How a manabot learns: play, batch boundaries, credit assignment, filtering, the loss, reading a run |
| [observation-and-model.md](observation-and-model.md) | What the policy sees and how the network produces action probabilities and values |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Packages, authorities and data flow across managym, manabot and Academy |
| [development.md](development.md) | Environment, validation by change area, CI, releases, clean-machine play, retained evidence |
| [training.md](training.md) | Regimes and stages, experiments, remote training, monitoring, numerical health, recovery, Ataraxos, policy inputs |
| [evaluation.md](evaluation.md) | Arenas, shared inference, card data, checkpoint scenarios, frozen-policy attacks |
| [experiment-metrics.md](experiment-metrics.md) | Metric definitions the experiment reports read |
| [search-and-beliefs.md](search-and-beliefs.md) | Forking, search players, local policy search, possible worlds, the belief sampler |
| [llm.md](llm.md) | LLM players: selection protocol, local and hosted models, games, offline inference reports |
| [academy.md](academy.md) | The browser UI: running it, the shared decision surface, choice navigation, presentation, trained challengers, the proof gate |

## Engine and rules

| Where | What |
|---|---|
| [rules/](rules/) | Rules authority: [proposed events](rules/proposed-events-v1.md), [trigger/SBA stabilization](rules/stabilization-v1.md), the comprehensive rules PDF |
| [architecture/experience-protocol-v1.md](architecture/experience-protocol-v1.md) | The Academy experience protocol |
| [architecture/match-state-hash-v1.md](architecture/match-state-hash-v1.md) | The deterministic state-hash domain |
| [research/semantic-kernel.md](research/semantic-kernel.md) | Typed card programs, object identity, structured offers |
| [benchmarks/](benchmarks/) | Preregistered benchmark contracts, including the search-branching decision (fork by full clone) |

## Evidence, research notes and plans

- [evidence/](evidence/) holds dated records that reports cite.
- [research/](research/) has comparisons and deep dives, for example
  [Etude vs Phase](research/etude-vs-phase.md),
  [observation robustness](research/metta-observation-robustness.md), and
  [intelligence development ladders](research/intelligence-development-ladders.md).
- [plans/](plans/) has cross-cutting plans, for example
  [training-time cohorts](plans/training-time-cohorts.md). Most plans there
  predate the October move to self-play and are kept as history.
