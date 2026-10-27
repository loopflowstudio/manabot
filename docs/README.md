# docs/

Start with [ARCHITECTURE.md](ARCHITECTURE.md): the system picture, the
training loop, and who owns which truth. Experiment reports live in
[experiments/](../experiments/README.md), and world versioning lives in
[WORLDS.md](../WORLDS.md).

## Training

| Doc | What |
|---|---|
| [training-regimes.md](training-regimes.md) | `TrainingRegime` recipes and stages: self-play, compound decisions, policy import, belief fitting |
| [training-experiments.md](training-experiments.md) | Declaring comparisons between recipes |
| [ataraxos.md](ataraxos.md) | The Ataraxos-adapted move-learning rule, and how it differs from the paper |
| [numerical-health.md](numerical-health.md) | Policy numerics and update-health diagnostics |
| [recent-events.md](recent-events.md) | Optional viewer-safe recent-event history as model input |
| [training-calibration.md](training-calibration.md) | Measuring complete-loop training cost |
| [training-recovery.md](training-recovery.md) | Recovering interrupted CPU self-play |
| [remote-training.md](remote-training.md) | GPU training on rented machines through Etude |
| [training-monitoring.md](training-monitoring.md) | Dashboards, W&B, and S3 checkpoint storage |
| [distributed-rl.md](distributed-rl.md) | Distributed RL research and CPU feasibility evidence |

## Beliefs, search, and evaluation

| Doc | What |
|---|---|
| [belief-sampler.md](belief-sampler.md) | Hidden-hand samplers fit from a frozen policy's games |
| [local-policy-search.md](local-policy-search.md) | Frozen-policy local search at play time |
| [frozen-policy-attacks.md](frozen-policy-attacks.md) | Training exploiters against a frozen policy |
| [checkpoint-scenarios.md](checkpoint-scenarios.md) | Tactical scoring of checkpoints on constructed positions |
| [experiment-execution.md](experiment-execution.md) / [experiment-metrics.md](experiment-metrics.md) | Running an Experiment comparison and reading its report |

## Engine and rules

| Where | What |
|---|---|
| [rules/](rules/) | Rules authority: [proposed events](rules/proposed-events-v1.md), [trigger/SBA stabilization](rules/stabilization-v1.md), the comprehensive rules PDF |
| [research/semantic-kernel.md](research/semantic-kernel.md) | Typed card programs, object identity, structured offers, fork/rollback |
| [benchmarks/](benchmarks/) | Preregistered benchmark contracts for search branching and clone/undo |
| [choice-navigation.md](choice-navigation.md) | Large legal-choice lists |
| [architecture/match-state-hash-v1.md](architecture/match-state-hash-v1.md) | The deterministic state-hash domain |

## Research notes and plans

- [research/](research/) has comparisons and deep dives, for example
  [Etude vs Phase](research/etude-vs-phase.md),
  [observation robustness](research/metta-observation-robustness.md), and
  [intelligence development ladders](research/intelligence-development-ladders.md).
- [plans/](plans/) has cross-cutting plans, for example
  [training-time cohorts](plans/training-time-cohorts.md). Most plans there
  predate the October move to self-play and are kept as history.

## Play UI

The Academy browser table is an optional inspection tool:
[experience protocol v1](architecture/experience-protocol-v1.md),
[presentation runtime](architecture/presentation-runtime.md),
[clean-machine play](clean-machine-play.md),
[experience proof](experience-proof.md), and the
[local challenger workflow](local-trained-challenger.md).
