"""Frozen attack plans map target/seed pairs to ordinary TrainingRegimes.

An update ladder retains every rung of one continued attacker, not independently
reinitialized candidates. Evaluation deals are shared across its checkpoints;
training seeds remain the units of replication. Execution and arena reporting
consume this validated plan rather than inferring targets from mutable aliases.
"""

from typing import Literal

from pydantic import Field, model_validator

from manabot.training.models import (
    FrozenOpponent,
    Strict,
    TrainingRegime,
    TrainSelfPlay,
)


class AttackTarget(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,40}$")
    policy: FrozenOpponent
    producer_seeds: tuple[int, ...] = Field(min_length=1)


class AttackPlan(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,40}$")
    schema_version: Literal[1] = 1
    targets: tuple[AttackTarget, ...] = Field(min_length=1)
    attacker_seeds: tuple[int, ...] = Field(min_length=2)
    cumulative_updates: tuple[int, ...] = Field(min_length=2)
    final_deal_seeds: tuple[int, ...] = Field(min_length=1)
    template: TrainingRegime
    total_seconds: float = Field(gt=0)
    evaluation_seconds: float = Field(gt=0)
    game_seconds: float = Field(default=30, gt=0)
    max_commands: int = Field(default=10000, ge=1)
    prediction: str = Field(min_length=1)

    @model_validator(mode="after")
    def cohort(self) -> "AttackPlan":
        if len({target.id for target in self.targets}) != len(self.targets):
            raise ValueError("target IDs must be unique")
        if len({target.policy.sha256 for target in self.targets}) != len(self.targets):
            raise ValueError("duplicate target bytes are not independent targets")
        if any(n <= 0 for n in self.cumulative_updates) or any(
            a >= b for a, b in zip(self.cumulative_updates, self.cumulative_updates[1:])
        ):
            raise ValueError("update ladder must be positive and strictly increasing")
        for seeds in (self.attacker_seeds, self.final_deal_seeds):
            if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError("seed cohorts require distinct nonnegative values")
        producers = {seed for target in self.targets for seed in target.producer_seeds}
        if any(seed < 0 for seed in producers) or producers.intersection(
            self.attacker_seeds
        ):
            raise ValueError("producer and attacker seeds must be distinct")
        # Reserve executor's initialization/collection/minibatch/evaluation
        # namespaces. This prevents literal seed reuse, not a proof that PRNG
        # trajectories cannot overlap; actual game identities remain in traces.
        reserved = {
            seed + offset
            for seed in producers | set(self.attacker_seeds)
            for offset in (0, 10000, 20000, 30000)
        }
        if reserved.intersection(self.final_deal_seeds):
            raise ValueError("final deals overlap training seed namespaces")
        if len(self.template.stages) != 1 or not isinstance(
            self.template.stages[0], TrainSelfPlay
        ):
            raise ValueError("attack template requires one self-play stage")
        if self.template.stages[0].initial is not None:
            raise ValueError("attack template must initialize an independent attacker")
        training_cap = (
            len(self.targets) * len(self.attacker_seeds) * self.template.wall_seconds
        )
        if training_cap + self.evaluation_seconds > self.total_seconds:
            raise ValueError("full attack cohort exceeds total allocation")
        return self

    def regime_for(self, target: AttackTarget) -> TrainingRegime:
        """Allocate increments while preserving one collector and optimizer."""
        if target not in self.targets:
            raise ValueError("target is not in the frozen cohort")
        source = self.template.stages[0]
        assert isinstance(source, TrainSelfPlay)
        stages: list[TrainSelfPlay] = []
        previous = 0
        for index, cumulative in enumerate(self.cumulative_updates):
            stage = source.model_copy(deep=True)
            stage.id = f"attack-{index}"
            stage.initial = stages[-1].id if stages else None
            stage.behavior = "frozen"
            stage.opponent = target.policy
            stage.updates = cumulative - previous
            stages.append(stage)
            previous = cumulative
        payload = self.template.model_dump()
        payload.update(
            id=f"{self.id}-{target.id}", stages=[s.model_dump() for s in stages]
        )
        return TrainingRegime.model_validate(payload)
