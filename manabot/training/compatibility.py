"""Historical learning tags at the recipe input/serialization boundary.

Old serialized recipes participate in digests and recovery admission. Decode
one runtime algorithm, but preserve the original tag when reserializing those
inputs. Newly authored recipes use descriptive tags. No trainer dispatches on
an experiment name and no frozen bytes need migration.
"""

from typing import Self

from pydantic import (
    BaseModel,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class LearningInput(BaseModel):
    _historical_gradient: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def decode_gradient(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        if isinstance(value, dict) and value.get("gradient") == "ataraxos_move":
            result = handler({**value, "gradient": "damped_policy"})
            result._historical_gradient = True
            return result
        return handler(value)

    @model_serializer(mode="wrap")
    def retain_input_tag(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if self._historical_gradient and "gradient" in result:
            result["gradient"] = "ataraxos_move"
        return result
