"""Typed module-stage boundary: declared JSON artifacts and scoped model access.

Modules export a `STAGE` definition. Recipe admission resolves artifact types
before execution. Only a stage's named trainable model is passed live; frozen
readers receive detached copies, so their parameters and buffers cannot mutate
the models owned by the run.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from manabot.infra.files import atomic_json

if TYPE_CHECKING:
    from manabot.model.actor_critic import ActorCritic


@dataclass(frozen=True)
class StageContext:
    trainable: "ActorCritic | None"
    frozen: Mapping[str, "ActorCritic"]
    directory: Path
    check: Callable[[], None]


@dataclass(frozen=True)
class StageDefinition:
    inputs: Mapping[str, type[BaseModel]]
    output: type[BaseModel]
    execute: Callable[[StageContext, Mapping[str, BaseModel]], BaseModel]

    def run(self, context: StageContext, inputs: Mapping[str, Path]) -> Path:
        """Validate JSON at the module boundary and publish its typed result."""
        parsed = {
            name: schema.model_validate_json(inputs[name].read_bytes())
            for name, schema in self.inputs.items()
        }
        context.directory.mkdir()
        result = self.execute(context, parsed)
        if not isinstance(result, self.output):
            raise ValueError("module returned an incompatible artifact type")
        result = self.output.model_validate(result.model_dump())
        target = context.directory / "result.json"
        atomic_json(target, result.model_dump(mode="json"))
        return target


def load_stage(module: str) -> StageDefinition:
    definition = getattr(import_module(module), "STAGE", None)
    if not isinstance(definition, StageDefinition):
        raise ValueError(f"{module} must export a StageDefinition as STAGE")
    schemas = [definition.output, *definition.inputs.values()]
    if not all(
        isinstance(schema, type) and issubclass(schema, BaseModel) for schema in schemas
    ):
        raise ValueError(f"{module} artifact types must be Pydantic models")
    if not callable(definition.execute):
        raise ValueError(f"{module} must declare an execution callable")
    return definition
