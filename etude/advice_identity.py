"""Identity values shared by Etude's advice and testing-house contracts.

The legacy four-string identity remains import-compatible through
``etude.advice``.  ``AdvisorIdentity`` is the versioned, closed superset used
by the belief-conditioned provider.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from .experience_protocol import ProtocolModel, UInt32, UInt64

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class AdviceRequestIdentity(ProtocolModel):
    """Legacy GAM-6 replay/advisor/compute identity.

    It is intentionally unchanged so checked testing-house fixtures and
    existing imports continue to round-trip while callers migrate to
    :class:`AdvisorIdentity`.
    """

    source_replay_id: str
    match_id: str
    advisor_id: str
    compute_id: str


class AbiIdentity(ProtocolModel):
    name: str
    version: str
    sha256: str = Field(pattern=SHA256_PATTERN)


class CodeSourceArtifact(ProtocolModel):
    kind: Literal["code_source"]
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)


class CheckpointArtifact(ProtocolModel):
    kind: Literal["checkpoint"]
    checkpoint_id: str
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_bytes: UInt64
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    training_seed: UInt64
    observation_abi: AbiIdentity
    action_abi: AbiIdentity
    value_mode: Literal["neutral", "learned"]


AdvisorArtifact: TypeAlias = Annotated[
    CodeSourceArtifact | CheckpointArtifact,
    Field(discriminator="kind"),
]


class AdvisorComputeIdentity(ProtocolModel):
    id: str
    simulations_per_scenario: UInt64
    sampled_worlds: UInt32
    c_puct: float = Field(gt=0.0, allow_inf_nan=False)
    max_steps: UInt32 = Field(gt=0)
    branch_driver_id: str

    @model_validator(mode="after")
    def validate_world_budget(self) -> "AdvisorComputeIdentity":
        if self.sampled_worlds > self.simulations_per_scenario:
            raise ValueError("sampled worlds cannot exceed simulations")
        return self


class AdvisorSeedIdentity(ProtocolModel):
    plan_id: str
    root_seed: UInt64
    derivation_id: str


class AdvisorIdentity(ProtocolModel):
    """Fully pinned identity for one versioned advice computation."""

    kind: Literal["advisor_identity_v1"] = "advisor_identity_v1"
    source_replay_id: str
    source_replay_sha256: str = Field(pattern=SHA256_PATTERN)
    match_id: str
    world_id: str
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    observation_abi: AbiIdentity
    action_abi: AbiIdentity
    possible_world_abi: AbiIdentity
    information_boundary: Literal["historical_viewer"]
    planner_id: Literal["determinized_puct"]
    evaluator_id: Literal[
        "uniform_random_terminal",
        "checkpoint_policy_neutral_value",
    ]
    artifact: AdvisorArtifact
    compute: AdvisorComputeIdentity
    seed: AdvisorSeedIdentity

    @model_validator(mode="after")
    def validate_non_empty_identities(self) -> "AdvisorIdentity":
        required = (
            self.source_replay_id,
            self.match_id,
            self.world_id,
            self.observation_abi.name,
            self.action_abi.name,
            self.possible_world_abi.name,
            self.compute.id,
            self.seed.plan_id,
            self.seed.derivation_id,
        )
        if any(not value for value in required):
            raise ValueError("advisor identity fields must be non-empty")
        return self


AdviceIdentity: TypeAlias = AdviceRequestIdentity | AdvisorIdentity
