"""Compose independent training regimes without executing them.

Recipes are ordinary Python functions returning existing specification values.
Variation revalidates the whole regime, including model/objective compatibility,
and snapshots nested values so editing one arm cannot change its baseline.
TrainingRegime remains the serialized owner; no recipe registry is involved.
"""

from collections.abc import Sequence
from typing import Literal

from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.training.models import (
    AtaraxosMoveLearning,
    Execution,
    TrainingRegime,
    TrainSelfPlay,
)

ValueOutput = Literal["scalar", "categorical_wdl"]
ValueAggregation = Literal["historical_mean", "masked_mean", "value_token"]


class _Unset:
    pass


_UNSET = _Unset()


def with_agent(regime: TrainingRegime, *, id: str, agent: AgentSpec) -> TrainingRegime:
    """Replace only model settings and label; reject incompatible stage targets."""
    return TrainingRegime.model_validate(
        {**regime.model_dump(), "id": id, "agent": agent.model_dump()}
    )


def with_recent_events(
    regime: TrainingRegime, *, id: str, enabled: bool
) -> TrainingRegime:
    """Select public recent-event context independently of model size and learning."""
    agent = AgentSpec.model_validate(
        {**regime.agent.model_dump(), "recent_events": enabled}
    )
    return with_agent(regime, id=id, agent=agent)


def with_value_output(
    regime: TrainingRegime, *, id: str, output: ValueOutput
) -> TrainingRegime:
    """Use the trainer's existing scalar/WDL target and loss dispatch.

    The learning rule is preserved. In particular, converting ordinary PPO to WDL
    fails regime validation instead of silently switching to Ataraxos move learning.
    """
    agent = AgentSpec.model_validate(
        {**regime.agent.model_dump(), "value_kind": output}
    )
    return with_agent(regime, id=id, agent=agent)


def with_value_aggregation(
    regime: TrainingRegime, *, id: str, aggregation: ValueAggregation
) -> TrainingRegime:
    """Change pooling only, preserving output representation and learning rules."""
    agent = AgentSpec.model_validate(
        {**regime.agent.model_dump(), "value_aggregation": aggregation}
    )
    return with_agent(regime, id=id, agent=agent)


def with_capacity(
    regime: TrainingRegime,
    *,
    id: str,
    width: int,
    depth: int,
    heads: int,
    feedforward_dim: int | None | _Unset = _UNSET,
) -> TrainingRegime:
    """Select delivered width/depth/head fields without changing information inputs.

    Omission preserves the baseline expansion; None restores heads × width.
    Invalid head divisibility and compound combinations fail ordinary AgentSpec validation.
    """
    expansion = (
        {}
        if isinstance(feedforward_dim, _Unset)
        else {"attention_feedforward_dim": feedforward_dim}
    )
    agent = AgentSpec.model_validate(
        {
            **regime.agent.model_dump(),
            **expansion,
            "hidden_dim": width,
            "attention_layers": depth,
            "num_attention_heads": heads,
        }
    )
    return with_agent(regime, id=id, agent=agent)


def value_outputs(
    regimes: Sequence[TrainingRegime], outputs: tuple[ValueOutput, ...]
) -> dict[str, TrainingRegime]:
    """Cross explicitly requested outputs; an invalid cell fails the whole plan.

    Each regime ID supplies its cell prefix; a second label would duplicate that
    authority. The caller binds the cells to an EvaluationProtocol with an explicit
    budget and cohort. Nothing is allocated or executed here.
    """
    if not regimes or not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("regimes and unique value outputs must be nonempty")
    if len({regime.id for regime in regimes}) != len(regimes):
        raise ValueError("value cross requires unique regime IDs")
    result: dict[str, TrainingRegime] = {}
    for regime in regimes:
        for output in outputs:
            name = f"{regime.id}-{output.replace('_', '-')}"
            result[name] = with_value_output(regime, id=name, output=output)
    return result


def ataraxos_baseline(
    *,
    id: str,
    world: str,
    match: MatchHypers,
    observation: ObservationSpaceHypers,
    agent: AgentSpec,
    checkpoints: int,
    updates: int,
    transitions: int,
    streams: int,
    stage_seconds: float,
    wall_seconds: float,
) -> TrainingRegime:
    """Build a one-thread CPU move-learning baseline with cumulative stages.

    Setup, model and workload are explicit; this constructor grants no allocation.
    Model output chooses the existing scalar or categorical value treatment. Stage
    continuations retain the executor's optimizer, collector and EMA semantics.
    """
    if checkpoints < 1:
        raise ValueError("at least one checkpoint is required")
    regime = TrainingRegime(
        id=id,
        world=world,
        match=match,
        observation=observation,
        agent=agent,
        stages=[
            TrainSelfPlay(
                operation="train_self_play",
                id=f"policy-{index}",
                initial=f"policy-{index - 1}" if index else None,
                updates=updates,
                streams=streams,
                transitions=transitions,
                learning=AtaraxosMoveLearning(gradient="ataraxos_move"),
                execution=Execution(wall_seconds=stage_seconds),
            )
            for index in range(checkpoints)
        ],
        wall_seconds=wall_seconds,
    )
    return TrainingRegime.model_validate(regime.model_dump())
