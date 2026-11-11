"""Inspect the stage's accessible models without playing or updating weights.

This small executable operation exercises the module/artifact contract and
produces a useful admission receipt before more expensive stages.
"""

from collections.abc import Mapping

from pydantic import BaseModel

from manabot.infra.identity import StrictModel
from manabot.training.stages.contract import StageContext, StageDefinition


class ModelInventory(StrictModel):
    trainable_parameters: int
    frozen_parameters: dict[str, int]


def execute(context: StageContext, inputs: Mapping[str, BaseModel]) -> ModelInventory:
    context.check()
    return ModelInventory(
        trainable_parameters=sum(p.numel() for p in context.trainable.parameters())
        if context.trainable is not None
        else 0,
        frozen_parameters={
            name: sum(p.numel() for p in model.parameters())
            for name, model in context.frozen.items()
        },
    )


STAGE = StageDefinition(inputs={}, output=ModelInventory, execute=execute)
