"""Strict identities and artifact schemas for the manabot skill arena."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SHA256_PATTERN = r"^[0-9a-f]{64}$"
BASE_ANCHOR_IDS = (
    "random-v1",
    "scripted-greedy-v1",
    "flat-mc-4-v1",
    "flat-mc-16-v1",
    "flat-mc-64-v1",
)


class ArenaKey(StrictModel):
    world: Literal["w2"]
    content_suite: str
    viewer_boundary: str
    arena_version: str
    rating_model_version: str
    rating_prior_sha256: str = Field(pattern=SHA256_PATTERN)
    anchor_cohort_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_compute_envelope_id: str


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
    world: Literal["w2"]
    content_suite: str
    observation_abi_sha256: str = Field(pattern=SHA256_PATTERN)
    action_abi_sha256: str = Field(pattern=SHA256_PATTERN)
    matchup_sha256: str = Field(pattern=SHA256_PATTERN)
    source_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    checkpoint_bytes: int | None = Field(default=None, gt=0)
    parameter_count: int | None = Field(default=None, gt=0)
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
                self.training_seed,
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
            expected_checkpoint_spec = {
                "kind": "checkpoint",
                "deterministic": True,
                "device": "cpu",
                "batch_size": 1,
            }
            if self.player_spec != expected_checkpoint_spec:
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

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.model_dump())


class Schedule(StrictModel):
    deal_seeds: tuple[int, ...]
    competency_seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_replicates: int

    @model_validator(mode="after")
    def validate_schedule(self) -> "Schedule":
        if not self.deal_seeds or len(set(self.deal_seeds)) != len(self.deal_seeds):
            raise ValueError("deal seeds must be nonempty and unique")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        return self


class PromotionRule(StrictModel):
    elo_margin: Literal[25.0] = 25.0
    elo_lower_bound: Literal[0.0] = 0.0
    latency_ratio_max: Literal[1.10] = 1.10
    throughput_ratio_min: Literal[0.90] = 0.90
    rss_ratio_max: Literal[1.10] = 1.10
    competency_margin: Literal[0.10] = 0.10
    playout_cap_rate_max: Literal[0.001] = 0.001


class RatingModel(StrictModel):
    model: Literal["seat-aware-gaussian-map-bradley-terry-v1"]
    anchor_player_id: Literal["random-v1"]
    prior_elo_std: Literal[400.0]
    elo_factor: Literal["400/ln(10)"]
    optimizer: Literal["newton-map-v1"]
    tolerance: Literal[1e-10]
    max_iterations: Literal[100]
    bootstrap_unit: Literal["global-deal-block-v1"]


class ProfileRoots(StrictModel):
    source_cell: Literal["random-v1__scripted-greedy-v1"]
    selection: Literal["deal-leg-revision-canonical-v1"]
    warmup: int = Field(ge=0)
    measured: int = Field(gt=0)
    sampler_interval_ms: Literal[5]


class ResourceCaps(StrictModel):
    outcome_workers: Literal[4]
    wall_hours: Literal[16.0]
    core_hours: Literal[64.0]
    artifact_bytes: Literal[4294967296]


class ArenaProfile(StrictModel):
    deal_blocks: int
    competency_runs: int
    bootstrap_replicates: int
    disposition: Literal["production", "engineering_smoke_non_promotion"]


class ArenaContract(StrictModel):
    schema_version: Literal[1]
    key: ArenaKey
    anchors: tuple[PlayerRegistration, ...]
    profiles: dict[str, ArenaProfile]
    schedules: dict[str, Schedule]
    promotion: PromotionRule
    rating_model: RatingModel
    profile_roots: ProfileRoots
    resource_caps: ResourceCaps
    source_paths: tuple[str, ...]
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime: dict[str, Any]

    @model_validator(mode="after")
    def validate_contract(self) -> "ArenaContract":
        expected_key = {
            "world": "w2",
            "content_suite": "w2-interactive-mirror-v1",
            "viewer_boundary": "acting-viewer-history-only-v1",
            "arena_version": "manabot-skill-arena-v1",
            "rating_model_version": "seat-aware-gaussian-map-bradley-terry-v1",
            "evaluation_compute_envelope_id": "apple-m4-max-cpu-four-workers-v1",
        }
        for field, expected in expected_key.items():
            if getattr(self.key, field) != expected:
                raise ValueError(f"frozen arena key drift: {field}")
        expected_runtime_keys = {
            "action_abi_sha256",
            "asset_manifest_hash",
            "content_digest",
            "content_manifest_sha256",
            "content_schema_version",
            "engine_extension_name",
            "engine_extension_sha256",
            "engine_source_sha256",
            "experience_content_hash",
            "experience_protocol_sha256",
            "inference_device",
            "matchup_sha256",
            "numpy_version",
            "observation_abi_sha256",
            "platform_machine",
            "platform_system",
            "psutil_version",
            "pydantic_version",
            "python_version",
            "torch_threads_per_player",
            "torch_version",
            "world",
        }
        if set(self.runtime) != expected_runtime_keys:
            raise ValueError("runtime identity fields are not closed")
        ids = [player.player_id for player in self.anchors]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate anchor player_id")
        if any(player.role != "anchor" for player in self.anchors):
            raise ValueError("contract anchors must have anchor role")
        if tuple(ids) != BASE_ANCHOR_IDS:
            raise ValueError("arena-v1 freezes exactly the five code-only anchors")
        if any(
            player.runner_kind != "code" or player.evidence_class != "production"
            for player in self.anchors
        ):
            raise ValueError("arena-v1 anchors must be code-only production players")
        expected_specs = {
            "random-v1": {"kind": "random"},
            "scripted-greedy-v1": {"kind": "scripted_greedy"},
            "flat-mc-4-v1": {
                "kind": "search",
                "sims": 4,
                "rollouts_per_world": 4,
                "max_steps": 2000,
            },
            "flat-mc-16-v1": {
                "kind": "search",
                "sims": 16,
                "rollouts_per_world": 4,
                "max_steps": 2000,
            },
            "flat-mc-64-v1": {
                "kind": "search",
                "sims": 64,
                "rollouts_per_world": 4,
                "max_steps": 2000,
            },
        }
        for player in self.anchors:
            if player.player_spec != expected_specs[player.player_id]:
                raise ValueError(f"frozen anchor spec drift: {player.player_id}")
            expected_compute_classes = {
                "random-v1": "random-cpu-v1",
                "scripted-greedy-v1": "scripted-greedy-cpu-v1",
                "flat-mc-4-v1": "flat-mc-cpu-s4-r4-v1",
                "flat-mc-16-v1": "flat-mc-cpu-s16-r4-v1",
                "flat-mc-64-v1": "flat-mc-cpu-s64-r4-v1",
            }
            if (
                player.compute_class_id != expected_compute_classes[player.player_id]
                or player.player_seed_derivation_id != "arena-pair-deal-player-v1"
            ):
                raise ValueError(f"anchor compute/seed drift: {player.player_id}")
            if (
                player.world != self.key.world
                or player.content_suite != self.key.content_suite
                or player.information_boundary != self.key.viewer_boundary
                or player.observation_abi_sha256
                != self.runtime.get("observation_abi_sha256")
                or player.action_abi_sha256 != self.runtime.get("action_abi_sha256")
                or player.matchup_sha256 != self.runtime.get("matchup_sha256")
            ):
                raise ValueError(f"anchor compatibility drift: {player.player_id}")
        if canonical_sha256([player.model_dump() for player in self.anchors]) != (
            self.key.anchor_cohort_sha256
        ):
            raise ValueError("anchor cohort digest mismatch")
        if canonical_sha256(self.rating_model.model_dump()) != (
            self.key.rating_prior_sha256
        ):
            raise ValueError("rating model/prior digest mismatch")
        if self.profile_roots.model_dump() != {
            "source_cell": "random-v1__scripted-greedy-v1",
            "selection": "deal-leg-revision-canonical-v1",
            "warmup": 16,
            "measured": 128,
            "sampler_interval_ms": 5,
        }:
            raise ValueError("matched-root corpus contract drift")
        if set(self.profiles) != {"smoke", "production"} or set(self.schedules) != {
            "smoke",
            "production",
        }:
            raise ValueError("arena-v1 requires exactly smoke and production profiles")
        expected_profiles = {
            "smoke": (2, 2, 100, "engineering_smoke_non_promotion"),
            "production": (24, 100, 2000, "production"),
        }
        for name, profile in self.profiles.items():
            schedule = self.schedules.get(name)
            if schedule is None:
                raise ValueError(f"profile {name} has no schedule")
            if len(schedule.deal_seeds) != profile.deal_blocks:
                raise ValueError(f"profile {name} deal-block mismatch")
            if len(schedule.competency_seeds) != profile.competency_runs:
                raise ValueError(f"profile {name} competency-run mismatch")
            if schedule.bootstrap_replicates != profile.bootstrap_replicates:
                raise ValueError(f"profile {name} bootstrap mismatch")
            actual = (
                profile.deal_blocks,
                profile.competency_runs,
                profile.bootstrap_replicates,
                profile.disposition,
            )
            if actual != expected_profiles[name]:
                raise ValueError(f"frozen profile drift: {name}")
        expected_schedules = {
            "smoke": {
                "deal_seeds": tuple(range(61001, 61003)),
                "competency_seeds": tuple(range(62001, 62003)),
                "bootstrap_seed": 63001,
                "bootstrap_replicates": 100,
            },
            "production": {
                "deal_seeds": tuple(range(61001, 61025)),
                "competency_seeds": tuple(range(62001, 62101)),
                "bootstrap_seed": 63001,
                "bootstrap_replicates": 2000,
            },
        }
        for name, expected in expected_schedules.items():
            if self.schedules[name].model_dump() != expected:
                raise ValueError(f"frozen schedule drift: {name}")
        expected_source_paths = (
            "experiments/runners/run_skill_arena.py",
            "manabot/arena/__init__.py",
            "manabot/arena/competency.py",
            "manabot/arena/match.py",
            "manabot/arena/models.py",
            "manabot/arena/players.py",
            "manabot/arena/profile.py",
            "manabot/arena/rating.py",
            "manabot/arena/replay.py",
        )
        if self.source_paths != expected_source_paths:
            raise ValueError("arena source identity paths are not closed")
        return self


class MatchRow(StrictModel):
    arena_key: ArenaKey
    cell_id: str
    deal_block: int = Field(ge=0)
    deal_seed: int
    deal_seed_set_sha256: str = Field(pattern=SHA256_PATTERN)
    leg: Literal[0, 1]
    player_a: str
    player_b: str
    player_a_registration_sha256: str
    player_b_registration_sha256: str
    player_a_compute_class: str
    player_b_compute_class: str
    player_a_seed: int = Field(ge=0)
    player_b_seed: int = Field(ge=0)
    player_a_seat: Literal[0, 1]
    winner: Literal[0, 1] | None
    score_a: float = Field(ge=0.0, le=1.0)
    terminated: bool
    truncated: bool
    termination_reason: Literal["terminal", "draw", "truncated"]
    decisions: int = Field(gt=0)
    game_trace_sha256: str = Field(pattern=SHA256_PATTERN)
    trace_path: str
    trace_sha256: str = Field(pattern=SHA256_PATTERN)
    trace_shard_sha256: str = Field(pattern=SHA256_PATTERN)
    replay_passed: bool
    integrity: dict[str, Any]
    latency: dict[str, Any]
    game_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_match_result(self) -> "MatchRow":
        if self.player_a == self.player_b:
            raise ValueError("arena match players must be distinct")
        if self.cell_id != "__".join(sorted((self.player_a, self.player_b))):
            raise ValueError("match cell identity mismatch")
        if self.player_a_seat != self.leg:
            raise ValueError("match leg and player-A seat mismatch")
        expected_score = (
            0.5 if self.winner is None else float(self.winner == self.player_a_seat)
        )
        if self.score_a != expected_score:
            raise ValueError("match score does not agree with winner and seat")
        if self.truncated != (self.termination_reason == "truncated"):
            raise ValueError("match truncation reason mismatch")
        if not self.truncated and not self.terminated:
            raise ValueError("nonterminal match cannot enter arena evidence")
        if self.trace_sha256 != self.game_trace_sha256:
            raise ValueError("match game-trace digest mismatch")
        trace_path = Path(self.trace_path)
        if trace_path.is_absolute() or ".." in trace_path.parts:
            raise ValueError("match trace locator must stay artifact-relative")
        if self.trace_path != f"traces/{self.cell_id}.commands.jsonl.gz":
            raise ValueError("match trace locator does not agree with cell identity")
        return self
