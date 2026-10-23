"""Bounded fitting of an exact-world belief model to a declared population."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from manabot.belief.encoding import BeliefEncodingSchema
from manabot.belief.population import (
    BeliefTrainingExample,
    ExactWorldBeliefModel,
    pack_belief_examples,
)
from manabot.belief.range import BeliefError
from manabot.infra.torch import deterministic_cpu


@dataclass(frozen=True, slots=True)
class BoundedPopulationTrainingResult:
    model: ExactWorldBeliefModel
    p0_nll: float
    initial_nll: float
    final_nll: float
    steps: int


def train_bounded_population(
    examples: Iterable[BeliefTrainingExample],
    schema: BeliefEncodingSchema,
    *,
    steps: int = 16,
    learning_rate: float = 0.01,
    seed: int = 197,
    hidden_dim: int = 4,
) -> BoundedPopulationTrainingResult:
    """Fit a fresh exact-world scorer to one pre-split population arm."""

    if steps < 1 or steps > 64:
        raise BeliefError("bounded population steps must be between 1 and 64")
    with deterministic_cpu():
        torch.manual_seed(seed)
        model = ExactWorldBeliefModel(schema, hidden_dim=hidden_dim)
        batch = pack_belief_examples(examples, schema)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            target_indexes = batch.offsets[:-1] + batch.target_world
            p0_nll = float((-batch.log_p0[target_indexes]).mean().item())
            initial_nll = float(model.negative_log_likelihood(batch).item())
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model.negative_log_likelihood(batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final_nll = float(model.negative_log_likelihood(batch).item())
    return BoundedPopulationTrainingResult(
        model=model,
        p0_nll=p0_nll,
        initial_nll=initial_nll,
        final_nll=final_nll,
        steps=steps,
    )
