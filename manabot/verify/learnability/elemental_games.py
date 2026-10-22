"""Bounded Elemental arena worker, called by Etude's isolated evaluation job.

Manabot retains its released tracking SDK. JSON is the process boundary to the
new Etude evaluator; the arena still owns legality, Commands and replay.
"""

import json
from pathlib import Path
import sys

from pydantic import BaseModel, ConfigDict
import torch

from manabot.arena import players
from manabot.arena.match import play_cell
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.teacher1_evidence import runtime_fingerprints
import managym


class Player(BaseModel):
    kind: str
    path: str | None = None
    sha256: str | None = None


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    players: list[Player]
    role_a: str
    role_b: str
    deal: int
    mirrors: bool
    output: Path


def play(request: Request) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    matrix = AuthoredMatchup(
        pack_key="elemental-world-v1", role_a=request.role_a, role_b=request.role_b
    )
    fingerprints = runtime_fingerprints(match_config=matrix.match())
    registrations: list[PlayerRegistration] = []
    paths: dict[str, str] = {}
    for seat, player in enumerate(request.players):
        common = dict(
            player_id=f"elemental-player-{seat}",
            display_name=f"Elemental player {seat}",
            role="challenger" if seat == 0 else "anchor",
            world=managym.WORLD_VERSION,
            content_suite=matrix.content_suite,
            information_boundary="acting-viewer",
            observation_abi_sha256=fingerprints["observation_abi_sha256"],
            action_abi_sha256=fingerprints["action_abi_sha256"],
            matchup_sha256=canonical_sha256(matrix.match().model_dump()),
            player_seed_derivation_id="arena-pair-deal-player-v1",
            compute_class_id="policy-cpu-one-thread-one-pass",
        )
        if player.kind == "checkpoint":
            if player.path is None or file_sha256(player.path) != player.sha256:
                raise ValueError("checkpoint byte identity differs")
            model, _ = load_checkpoint_agent(player.path)
            paths[common["player_id"]] = player.path
            registrations.append(
                PlayerRegistration(
                    **common,
                    runner_kind="checkpoint",
                    player_spec={
                        "kind": "checkpoint",
                        "device": "cpu",
                        "batch_size": 1,
                        "deterministic": False,
                        "allow_deck_repetition": True,
                    },
                    checkpoint_sha256=player.sha256,
                    checkpoint_bytes=Path(player.path).stat().st_size,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    training_seed=17402,
                    artifact_id=f"elemental/{player.sha256}",
                )
            )
        elif player.kind == "scripted_greedy":
            registrations.append(
                PlayerRegistration(
                    **common,
                    runner_kind="code",
                    player_spec={"kind": player.kind},
                    source_sha256=file_sha256(Path(players.__file__)),
                )
            )
        else:
            raise ValueError("unsupported prospective anchor")
    key = ArenaKey(
        world=managym.WORLD_VERSION,
        content_suite=matrix.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="elemental-174-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(registrations[1].model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    request.output.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = play_cell(
        key=key,
        player_a=registrations[0],
        player_b=registrations[1],
        deal_seeds=(request.deal,),
        out_dir=request.output,
        checkpoint_paths=paths,
        game_seconds=120,
        max_commands=10000,
        authored_matchup=matrix,
        matchup_mode="mirrors" if request.mirrors else "cross",
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    return rows


def main() -> None:
    request = Request.model_validate_json(sys.stdin.read())
    rows = play(request)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
