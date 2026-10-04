"""Frozen protocol for the bounded training-regime workflow studies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    study: Literal["learning-speed", "ataraxos-ablations"]
    purpose: Literal["workflow-smoke"] = "workflow-smoke"
    regime_digests: tuple[str, ...]
    training_seed: Literal[197] = 197
    paired_deals: tuple[int, ...] = (910001,)
    anchor_deals: tuple[int, ...] = (920001,)
    checkpoint_count: Literal[2] = 2
    anchor: Literal["random-smoke-anchor"] = "random-smoke-anchor"
    seat_deck_legs: Literal[4] = 4
    inference: Literal["stochastic-policy-cpu-one-thread"] = (
        "stochastic-policy-cpu-one-thread"
    )
    game_seconds: float = Field(default=120, gt=0, le=120)
    max_commands: int = Field(default=10000, gt=0, le=10000)
    process_seconds: float = Field(default=900, gt=0, le=900)
    selection: Literal["all-completed-cutoffs-raw"] = "all-completed-cutoffs-raw"
    uncertainty: Literal["cross-seed-unavailable"] = "cross-seed-unavailable"
    incomplete: Literal["fail-retain-all-attempts"] = "fail-retain-all-attempts"

    @model_validator(mode="after")
    def disjoint(self):
        expected = 2 if self.study == "learning-speed" else 5
        if len(self.regime_digests) != expected or any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in self.regime_digests
        ):
            raise ValueError("protocol must bind every resolved recipe digest")
        if any(seed < 900000 for seed in self.paired_deals + self.anchor_deals):
            raise ValueError(
                "smoke evaluation seeds must use the reserved family >=900000"
            )
        if not self.paired_deals or not self.anchor_deals:
            raise ValueError("evaluation deal families must be nonempty")
        if set(self.paired_deals) & set(self.anchor_deals):
            raise ValueError("paired and anchor deals must be disjoint")
        if any(
            len(values) != len(set(values))
            for values in (self.paired_deals, self.anchor_deals)
        ):
            raise ValueError("duplicate evaluation deal")
        return self
