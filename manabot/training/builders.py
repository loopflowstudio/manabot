"""Reusable parts for composing training regimes without executing them.

Experiment modules in `experiments/` assemble a `TrainingRegime` from
these parts; anything more than one recipe file needs lives here. Builders
return fresh values and carry no experiment identity: a recipe supplies the
regime `id`. Variation helpers (`with_*`) revalidate the whole regime, including
model/objective compatibility, and snapshot nested values so editing one arm
cannot change its baseline. TrainingRegime remains the serialized owner.
"""

from collections.abc import Sequence
from typing import Literal

from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.training.models import (
    DampedPolicyLearning,
    Execution,
    IterationSchedule,
    Learning,
    Operation,
    TrainingRegime,
    TrainSelfPlay,
)

# Semantic pack of the selected Allies/Lessons matchup that current recipes train on.
PACK = "ur-lessons-vs-gw-allies"
GIB = 1024**3

ValueOutput = Literal["scalar", "categorical_wdl"]
ValueAggregation = Literal["historical_mean", "masked_mean", "value_token"]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def move_learning(schedule: IterationSchedule | None = None) -> DampedPolicyLearning:
    """Ataraxos move learning; a schedule pins its curves to a reference horizon."""
    return DampedPolicyLearning(gradient="damped_policy", iteration_schedule=schedule)


def move_stage(
    id: str,
    *,
    updates: int,
    seconds: float,
    initial: str | None = None,
    streams: int = 4,
    transitions: int = 64,
    memory_bytes: int = 32 * GIB,
) -> TrainSelfPlay:
    """One CPU self-play stage of Ataraxos move learning."""
    return TrainSelfPlay(
        id=id,
        operation="train_self_play",
        initial=initial,
        updates=updates,
        streams=streams,
        transitions=transitions,
        learning=move_learning(),
        execution=Execution(wall_seconds=seconds, memory_bytes=memory_bytes),
    )


def move_checkpoints(
    count: int,
    *,
    updates: int,
    seconds: float,
    streams: int = 4,
    transitions: int = 64,
    memory_bytes: int = 32 * GIB,
) -> list[Operation]:
    """Cumulative move-learning stages `policy-0` … each continuing the previous.

    Continuations retain the executor's optimizer, collector and EMA semantics;
    `updates` and `seconds` apply to each stage.
    """
    if count < 1:
        raise ValueError("at least one checkpoint is required")
    return [
        move_stage(
            f"policy-{index}",
            initial=f"policy-{index - 1}" if index else None,
            updates=updates,
            seconds=seconds,
            streams=streams,
            transitions=transitions,
            memory_bytes=memory_bytes,
        )
        for index in range(count)
    ]


def ppo_checkpoints(count: int, learning: Learning) -> list[Operation]:
    """Cumulative PPO self-play stages `policy-0` … with independent copies of `learning`."""
    if count < 1:
        raise ValueError("at least one checkpoint is required")
    return [
        TrainSelfPlay(
            id=f"policy-{index}",
            operation="train_self_play",
            initial=f"policy-{index - 1}" if index else None,
            transitions=64,
            learning=learning.model_copy(deep=True),
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


def gray_ogre_match() -> MatchConfig:
    """The small Gray Ogre against Llanowar Elves matchup for belief and local search."""
    return MatchConfig(
        hero_deck={"Mountain": 8, "Gray Ogre": 8},
        villain_deck={"Forest": 8, "Llanowar Elves": 8},
    )


# ---------------------------------------------------------------------------
# Regimes and variations
# ---------------------------------------------------------------------------


class _Unset:
    pass


_UNSET = _Unset()


def ataraxos_baseline(
    *,
    id: str,
    world: str,
    match: MatchConfig,
    observation: ObservationConfig,
    agent: ModelConfig,
    checkpoints: int,
    updates: int,
    transitions: int,
    streams: int,
    stage_seconds: float,
    wall_seconds: float,
) -> TrainingRegime:
    """Build a one-thread CPU move-learning baseline with cumulative stages.

    Setup, model and workload are explicit; this constructor grants no allocation.
    Model output chooses the existing scalar or categorical value treatment.
    """
    regime = TrainingRegime(
        id=id,
        world=world,
        match=match,
        observation=observation,
        agent=agent,
        stages=move_checkpoints(
            checkpoints,
            updates=updates,
            seconds=stage_seconds,
            streams=streams,
            transitions=transitions,
        ),
        wall_seconds=wall_seconds,
    )
    return TrainingRegime.model_validate(regime.model_dump())


def with_agent(
    regime: TrainingRegime, *, id: str, agent: ModelConfig
) -> TrainingRegime:
    """Replace only model settings and label; reject incompatible stage targets."""
    return TrainingRegime.model_validate(
        {**regime.model_dump(), "id": id, "agent": agent.model_dump()}
    )


def with_recent_events(
    regime: TrainingRegime, *, id: str, enabled: bool
) -> TrainingRegime:
    """Select public recent-event context independently of model size and learning."""
    agent = ModelConfig.model_validate(
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
    agent = ModelConfig.model_validate(
        {**regime.agent.model_dump(), "value_kind": output}
    )
    return with_agent(regime, id=id, agent=agent)


def with_value_aggregation(
    regime: TrainingRegime, *, id: str, aggregation: ValueAggregation
) -> TrainingRegime:
    """Change pooling only, preserving output representation and learning rules."""
    agent = ModelConfig.model_validate(
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
    Invalid head divisibility and compound combinations fail ordinary ModelConfig validation.
    """
    expansion = (
        {}
        if isinstance(feedforward_dim, _Unset)
        else {"attention_feedforward_dim": feedforward_dim}
    )
    agent = ModelConfig.model_validate(
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
