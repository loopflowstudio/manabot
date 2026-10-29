"""Frozen player identities and search semantics shared by simulation consumers."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from manabot.infra.identity import SHA256_PATTERN, StrictModel, canonical_sha256
from manabot.sim.execution import EvalCompute, EvalComputeV1
from manabot.sim.search.branch import branch_backend

type WorldVersion = Literal[
    "w2",
    "w3",
    "w4",
    "w5",
    "w6",
    "w7",
    "w8",
    "w9",
    "w10",
    "w11",
    "w12",
    "w13",
    "w14",
    "w15",
    "w16",
]


class SearchSemantics(StrictModel):
    branch_audit: bool = False
    root_prior: str
    leaf_evaluator: str


class PlayerRegistration(StrictModel):
    player_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    display_name: str
    role: Literal["anchor", "incumbent", "challenger"]
    runner_kind: Literal["code", "checkpoint"]
    player_spec: dict[str, Any]
    compute_class_id: str
    information_boundary: str
    world: WorldVersion
    content_suite: str
    observation_abi_sha256: str = Field(pattern=SHA256_PATTERN)
    action_abi_sha256: str = Field(pattern=SHA256_PATTERN)
    matchup_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_bytes: int | None = Field(default=None, gt=0)
    parameter_count: int | None = Field(default=None, gt=0)
    # Producer metadata may be unavailable for a standalone policy export.
    training_seed: int | None = None
    artifact_id: str | None = None
    evidence_class: Literal["production", "fixture"] = "production"
    player_seed_derivation_id: str
    search_call_seed_derivation_id: str | None = None
    search_semantics: SearchSemantics | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "PlayerRegistration":
        if self.runner_kind == "code":
            checkpoint_identity = (
                self.checkpoint_sha256,
                self.checkpoint_bytes,
                self.parameter_count,
                self.training_seed,
                self.artifact_id,
            )
            if not self.source_sha256 or any(
                value is not None for value in checkpoint_identity
            ):
                raise ValueError(
                    "code players require source_sha256 and no checkpoint identity"
                )
            if self.evidence_class != "production":
                raise ValueError("code players are production registrations")
        else:
            required = (
                self.checkpoint_sha256,
                self.checkpoint_bytes,
                self.parameter_count,
                self.artifact_id,
            )
            if any(value is None for value in required):
                raise ValueError("checkpoint players require complete byte identity")
            if self.source_sha256 is not None:
                raise ValueError("checkpoint players cannot claim source identity")
            if self.artifact_id and "latest" in self.artifact_id.lower():
                raise ValueError(
                    "checkpoint artifact identity cannot be a mutable alias"
                )
            kind = self.player_spec.get("kind")
            if kind == "checkpoint":
                expected_checkpoint_spec = {
                    "kind": "checkpoint",
                    "deterministic": self.player_spec.get("deterministic"),
                    "device": "cpu",
                    "batch_size": 1,
                }
                if "execution" in self.player_spec:
                    declared = self.player_spec["execution"]
                    execution: EvalCompute | EvalComputeV1
                    if declared.get("version") == "shared-eval-v1":
                        # Frozen v1 players fixed the batch size in their spec.
                        execution = EvalComputeV1.model_validate(declared)
                        expected_checkpoint_spec["batch_size"] = (
                            execution.max_batch_rows
                        )
                    else:
                        execution = EvalCompute.model_validate(declared)
                        del expected_checkpoint_spec["batch_size"]
                    expected_checkpoint_spec.update(
                        device=execution.device, execution=execution.model_dump()
                    )
                    if self.compute_class_id != execution.identity:
                        raise ValueError(
                            "checkpoint compute class differs from execution"
                        )
                if "allow_deck_repetition" in self.player_spec:
                    if self.player_spec["allow_deck_repetition"] is not True:
                        raise ValueError("deck repetition must be explicitly enabled")
                    expected_checkpoint_spec["allow_deck_repetition"] = True
                if (
                    self.player_spec != expected_checkpoint_spec
                    or type(self.player_spec.get("deterministic")) is not bool
                    or (self.world == "w2" and not self.player_spec["deterministic"])
                ):
                    raise ValueError("checkpoint inference spec must be fully explicit")
            elif kind == "local_update":
                from manabot.sim.search.local_update import (
                    LocalSearchConfig,
                    SamplerArtifact,
                )

                config = LocalSearchConfig.model_validate(
                    self.player_spec.get("config", {})
                )
                required = {
                    "kind",
                    "config",
                    "implementation_source_sha256",
                }
                if config.sampling == "learned":
                    required.add("sampler")
                    sampler = SamplerArtifact.model_validate(
                        self.player_spec.get("sampler")
                    )
                    if sampler.policy_identity != self.checkpoint_sha256:
                        raise ValueError(
                            "sampler generating policy differs from arena checkpoint"
                        )
                if set(self.player_spec) != required:
                    raise ValueError(
                        "local-update inference spec must be fully explicit"
                    )
                config = LocalSearchConfig.model_validate(self.player_spec["config"])
                if config.model_dump() != self.player_spec["config"]:
                    raise ValueError("local-update config must include every parameter")
                digest = self.player_spec["implementation_source_sha256"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("local-update source identity is required")
                if (
                    self.search_semantics is None
                    or self.search_semantics.model_dump()
                    != {
                        "branch_audit": True,
                        "root_prior": "two-kl-local-update/v1",
                        "leaf_evaluator": "frozen-viewer-policy-signed-value/v1",
                    }
                    or self.search_call_seed_derivation_id
                    != "local-policy-seed-times-1000003-plus-call-mod-2pow63/v1"
                ):
                    raise ValueError("local-update search and seed semantics differ")
            elif kind == "agent_puct":
                spec = self.player_spec
                required_keys = {
                    "kind",
                    "sims",
                    "worlds",
                    "c_puct",
                    "max_steps",
                    "branch_driver_id",
                    "branch_audit",
                    "device",
                    "batch_size",
                    "value_mode",
                    "implementation_source_sha256",
                }
                if set(spec) != required_keys:
                    raise ValueError("agent PUCT spec must be fully explicit")
                digest = spec["implementation_source_sha256"]
                if (
                    any(
                        type(spec[k]) is not int
                        for k in ("sims", "worlds", "max_steps")
                    )
                    or not 1 <= spec["worlds"] <= spec["sims"]
                    or spec["max_steps"] < 1
                    or type(spec["c_puct"]) not in (int, float)
                    or not math.isfinite(spec["c_puct"])
                    or spec["c_puct"] <= 0
                    or spec["device"] != "cpu"
                    or spec["batch_size"] != 1
                    or type(spec["branch_audit"]) is not bool
                    or spec["value_mode"] not in ("learned", "neutral")
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("invalid agent PUCT parameters or source identity")
                branch_backend(spec["branch_driver_id"])
                if (
                    self.search_semantics is None
                    or self.search_semantics.model_dump()
                    != {
                        "branch_audit": spec["branch_audit"],
                        "root_prior": "checkpoint-canonical-policy/v1",
                        "leaf_evaluator": "checkpoint-declared-value-win-score/v1"
                        if spec["value_mode"] == "learned"
                        else "neutral-win-score/v1",
                    }
                    or self.search_call_seed_derivation_id
                    != "mcts-mix-player-seed-decision-v1"
                ):
                    raise ValueError("agent PUCT search and seed semantics differ")
            elif kind == "policy_prior_puct":
                required_keys = {
                    "kind",
                    "sims",
                    "worlds",
                    "c_puct",
                    "max_steps",
                    "branch_driver_id",
                    "device",
                    "batch_size",
                    "deterministic",
                    "root_noise",
                    "implementation_source_sha256",
                }
                if set(self.player_spec) != required_keys:
                    raise ValueError("policy-prior PUCT spec must be fully explicit")
                if (
                    self.player_spec["sims"] not in {8, 32, 128}
                    or self.player_spec["worlds"] != 4
                    or self.player_spec["c_puct"] != 1.5
                    or self.player_spec["max_steps"] != 2000
                    or self.player_spec["branch_driver_id"]
                    != "full_clone/current_game_v1"
                    or self.player_spec["device"] != "cpu"
                    or self.player_spec["batch_size"] != 1
                    or self.player_spec["deterministic"] is not True
                    or self.player_spec["root_noise"] != "none"
                    or not isinstance(
                        self.player_spec["implementation_source_sha256"], str
                    )
                    or len(self.player_spec["implementation_source_sha256"]) != 64
                ):
                    raise ValueError("policy-prior PUCT parameters are frozen")
                if (
                    self.search_semantics is None
                    or self.search_semantics.model_dump()
                    != {
                        "branch_audit": False,
                        "root_prior": "checkpoint-policy-softmax-v1",
                        "leaf_evaluator": "uniform-random-terminal-v1",
                    }
                    or self.search_call_seed_derivation_id
                    != "mcts-mix-comparison-seed-decision-v1"
                ):
                    raise ValueError(
                        "policy-prior PUCT requires frozen search and seed semantics"
                    )
            else:
                raise ValueError("checkpoint inference spec must be fully explicit")
        if self.player_spec.get("kind") == "determinized_puct":
            required_keys = {
                "sims",
                "worlds",
                "c_puct",
                "max_steps",
                "branch_driver_id",
            }
            if set(self.player_spec) != required_keys | {"kind"}:
                raise ValueError("determinized PUCT spec must be fully explicit")
            if (
                self.search_semantics is None
                or self.search_call_seed_derivation_id is None
            ):
                raise ValueError("determinized PUCT requires search and seed semantics")
            if self.player_id == "determinized-puct-32-w4-v1":
                expected_spec = {
                    "kind": "determinized_puct",
                    "sims": 32,
                    "worlds": 4,
                    "c_puct": 1.5,
                    "max_steps": 2000,
                    "branch_driver_id": "full_clone/current_game_v1",
                }
                expected_semantics = {
                    "branch_audit": False,
                    "root_prior": "uniform-v1",
                    "leaf_evaluator": "uniform-random-terminal-v1",
                }
                if self.player_spec != expected_spec:
                    raise ValueError("arena-v1 PUCT parameters are frozen")
                if self.search_semantics.model_dump() != expected_semantics:
                    raise ValueError("arena-v1 PUCT search semantics are frozen")
                if (
                    self.compute_class_id != "dpuct-cpu-s32-w4-random-leaf-v1"
                    or self.player_seed_derivation_id != "arena-pair-deal-player-v1"
                    or self.search_call_seed_derivation_id
                    != "mcts-mix-player-seed-decision-v1"
                ):
                    raise ValueError(
                        "arena-v1 PUCT compute and seed identity is frozen"
                    )
        return self

    @model_validator(mode="after")
    def validate_demo(self) -> "PlayerRegistration":
        if self.player_spec.get("kind") == "demo_search":
            if self.runner_kind != "code" or self.player_spec != {
                "kind": "demo_search",
                "sims": 64,
                "rollouts_per_world": 4,
                "max_steps": 2000,
            }:
                raise ValueError("demo baseline requires exact Search-64 configuration")
        return self

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.model_dump())
