"""Checkpoint and anchor registration independent of training or placement.

Portable checkpoint admission owns architecture/world checks. This layer binds
those exact local bytes to arena setup and input identities. Monitoring adds its
producer metadata and restrictions without changing these registration facts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from manabot.env.match import Match
from manabot.infra.config import MatchConfig
from manabot.infra.identity import file_sha256
from manabot.infra.provenance import runtime_fingerprints, source_bundle_sha256
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim import players
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.registration import PlayerRegistration

AnchorKind = Literal["scripted_greedy", "heuristic", "random"]
OPPONENT_NAMES: dict[AnchorKind, str] = {
    "scripted_greedy": "Frozen scripted greedy",
    "heuristic": "Rule-of-thumb heuristic",
    "random": "Uniform legal random",
}


@dataclass(frozen=True)
class CheckpointAdmission:
    registration: PlayerRegistration
    identities: dict[str, JsonValue]


def register_checkpoint(
    path: Path,
    *,
    match: MatchConfig,
    content_suite: str,
    player_id: str,
    display_name: str,
    training_seed: int | None = None,
    artifact_id: str | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    allow_deck_repetition: bool = False,
) -> CheckpointAdmission:
    """Admit local bytes; a missing producer seed remains unknown, never invented."""
    digest, size = file_sha256(path), path.stat().st_size
    if (expected_sha256 is not None and digest != expected_sha256) or (
        expected_bytes is not None and size != expected_bytes
    ):
        raise ValueError("checkpoint differs from recorded artifact")
    agent, space = load_checkpoint(str(path))
    validate_agent_setup(
        agent, Match(match).to_rust(), allow_deck_repetition=allow_deck_repetition
    )
    runtime = runtime_fingerprints(
        197 if training_seed is None else training_seed,
        match_config=match,
        observation_space=space,
    )
    registration = PlayerRegistration(
        information_boundary="acting-viewer",
        world=runtime["world"],
        content_suite=content_suite,
        observation_abi_sha256=runtime["observation_abi_sha256"],
        action_abi_sha256=runtime["action_abi_sha256"],
        matchup_sha256=runtime["matchup_sha256"],
        player_seed_derivation_id="arena-pair-deal-player-v1",
        player_id=player_id,
        display_name=display_name,
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
            **({"allow_deck_repetition": True} if allow_deck_repetition else {}),
        },
        compute_class_id="policy-cpu-one-thread-one-pass",
        checkpoint_sha256=digest,
        checkpoint_bytes=size,
        parameter_count=sum(parameter.numel() for parameter in agent.parameters()),
        training_seed=training_seed,
        artifact_id=artifact_id or f"checkpoint/{digest}",
    )
    return CheckpointAdmission(registration, runtime)


def register_anchor(
    candidate: PlayerRegistration, kind: AnchorKind, *, player_id: str
) -> PlayerRegistration:
    """Keep frozen scripted/random source identities and the heuristic bundle."""
    source = (
        source_bundle_sha256([Path(path) for path in heuristic_source_paths()])
        if kind == "heuristic"
        else file_sha256(Path(players.__file__))
    )
    common = candidate.model_dump(
        include={
            "information_boundary",
            "world",
            "content_suite",
            "observation_abi_sha256",
            "action_abi_sha256",
            "matchup_sha256",
            "player_seed_derivation_id",
        }
    )
    return PlayerRegistration(
        **common,
        player_id=player_id,
        display_name=OPPONENT_NAMES[kind],
        role="anchor",
        runner_kind="code",
        player_spec={"kind": kind},
        compute_class_id=kind.replace("_", "-") + "-cpu-v1",
        source_sha256=source,
    )
