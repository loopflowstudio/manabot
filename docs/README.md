# docs/

Begin with [Start here](start-here.md): the four parts of the system, then one
training iteration stage by stage, down to the code. Each part has its own
guide below, written the same way: what it does, one worked example, the key
terms with the code that implements them, then what we know and where the code
lives. Experiment reports live in [experiments/](../experiments/README.md), and
world versioning lives in [WORLDS.md](../WORLDS.md).

## The system

| Doc | What |
|---|---|
| [start-here.md](start-here.md) | The overview: the system, one iteration, play, credit, filter, update, what win rate hides, open questions |
| [engine.md](engine.md) | The rules engine: one decision, forced choices, who chooses, observations, forks, replay, cards as programs, worlds |
| [observation-and-model.md](observation-and-model.md) | What the policy sees, how it scores legal actions and estimates value, policy input versions |
| [learning.md](learning.md) | How a manabot learns: play, batch boundaries, credit assignment, filtering, the loss, reading a run, influences and fidelity |
| [evaluation.md](evaluation.md) | One deal, what counts, anchors and ratings, scenarios, frozen-policy attacks, shared inference, card data |
| [experiments.md](experiments.md) | How a question becomes a report: the claim rules, a worked example, failures worth reading, declaring and executing a comparison |
| [search-and-beliefs.md](search-and-beliefs.md) | Forking, search players, local policy search, possible worlds, the belief sampler, and what remains unmeasured |

## Operating it

| Doc | What |
|---|---|
| [training.md](training.md) | Running training: regimes and stages, remote training, monitoring, numerical health, recovery, game failures |
| [experiment-metrics.md](experiment-metrics.md) | Metric definitions the experiment reports read |
| [development.md](development.md) | Environment, validation by change area, CI, releases, clean-machine play, retained evidence |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Packages, authorities and data flow across managym, manabot and Academy |
| [llm.md](llm.md) | LLM players: selection protocol, local and hosted models, games, offline inference reports |
| [academy.md](academy.md) | The browser UI: running it, the shared decision surface, choice navigation, presentation, trained challengers, the proof gate |

## Engine and rules

| Where | What |
|---|---|
| [rules/](rules/) | Rules authority: [proposed events](rules/proposed-events-v1.md), [trigger/SBA stabilization](rules/stabilization-v1.md), the comprehensive rules PDF |
| [architecture/experience-protocol-v1.md](architecture/experience-protocol-v1.md) | The Academy experience protocol |
| [architecture/match-state-hash-v1.md](architecture/match-state-hash-v1.md) | The deterministic state-hash domain |
| [benchmarks/](benchmarks/) | The search-branching contract and decision (fork by full clone) |

## Evidence and research

- [evidence/learnability/](evidence/learnability/) holds the learnability
  qualification records that the nightly baseline and learnability reports cite.
- [research/](research/) has dated investigations: the
  [engine speed investigation](research/engine-speed-2026-11.md) and the
  [Standard milestone research](research/standard-2026-11/README.md).
