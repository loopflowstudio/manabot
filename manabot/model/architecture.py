"""Derived architecture identities and parameter accounting for ordinary Agents.

<<<<<<< HEAD
AgentSpec remains the configuration authority. This receipt versions the executable
=======
ModelConfig remains the configuration authority. This receipt versions the executable
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
layout and binds resolved inputs; world/content and checkpoint bytes retain their
separate identities. Counting never constructs a model or advances its RNG.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

<<<<<<< HEAD
from manabot.arena.models import canonical_sha256
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
=======
from manabot.env import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.infra.identity import canonical_sha256
from manabot.model.actor_critic import ActorCritic
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f


class ParameterCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total: int = Field(ge=0)
    trainable: int = Field(ge=0)

    @model_validator(mode="after")
    def bounded(self) -> "ParameterCount":
        if self.trainable > self.total:
            raise ValueError("trainable parameters exceed total")
        return self


class ArchitectureReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameters: ParameterCount
    components: dict[str, ParameterCount]


<<<<<<< HEAD
def architecture_identity(spec: AgentSpec, space: ObservationSpace) -> str:
=======
def architecture_identity(spec: ModelConfig, space: ObservationSpace) -> str:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    """Hash resolved meaning, not recipe labels, device, seeds or weight bytes.

    Increment layout_version when equations or parameter ownership change, even
    if tensor shapes stay equal. Source/runtime receipts additionally bind code.
    """
<<<<<<< HEAD
    resolved = AgentSpec.model_validate(spec.model_dump())
=======
    resolved = ModelConfig.model_validate(spec.model_dump())
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    payload = resolved.model_dump(mode="json")
    if (
        resolved.attention_feedforward_dim
        == resolved.hidden_dim * resolved.num_attention_heads
    ):
        payload.pop("attention_feedforward_dim", None)
    return canonical_sha256(
        {
            "layout_version": 2,
            "agent": payload,
<<<<<<< HEAD
            "observation": space.encoder.hypers.model_dump(mode="json"),
=======
            "observation": space.encoder.config.model_dump(mode="json"),
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        }
    )


<<<<<<< HEAD
def architecture_receipt(agent: Agent) -> ArchitectureReceipt:
=======
def architecture_receipt(agent: ActorCritic) -> ArchitectureReceipt:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    """Count each distinct parameter once under its first named top-level owner.

    Buffers are excluded. Trainability describes the model at export, not optimizer
    exposures; admission compares structural totals, allowing frozen consumers.
    """
    totals: dict[str, int] = {}
    trainable: dict[str, int] = {}
    for name, parameter in agent.named_parameters():
        owner = name.split(".", 1)[0]
        totals[owner] = totals.get(owner, 0) + parameter.numel()
        trainable[owner] = trainable.get(owner, 0) + (
            parameter.numel() if parameter.requires_grad else 0
        )
    return ArchitectureReceipt(
<<<<<<< HEAD
        identity=architecture_identity(agent.hypers, agent.observation_space),
=======
        identity=architecture_identity(agent.config, agent.observation_space),
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        parameters=ParameterCount(
            total=sum(totals.values()), trainable=sum(trainable.values())
        ),
        components={
            name: ParameterCount(total=total, trainable=trainable[name])
            for name, total in sorted(totals.items())
        },
    )


<<<<<<< HEAD
def validate_architecture_receipt(agent: Agent, saved: object) -> None:
=======
def validate_architecture_receipt(agent: ActorCritic, saved: object) -> None:
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    """Reject contradictory new metadata; callers allow absent historical receipts."""
    receipt = ArchitectureReceipt.model_validate(saved)
    actual = architecture_receipt(agent)
    if receipt.parameters.trainable != sum(
        v.trainable for v in receipt.components.values()
    ):
        raise ValueError("checkpoint architecture trainable count mismatch")
    if (
        receipt.identity != actual.identity
        or receipt.parameters.total != actual.parameters.total
        or {k: v.total for k, v in receipt.components.items()}
        != {k: v.total for k, v in actual.components.items()}
    ):
        raise ValueError("checkpoint architecture receipt mismatch")
