"""Etude benchmarks backed by replay-verified Manabot simulation.

One Etude match contains all eight correlated deck/seat legs of one deal. Etude
owns scheduling, resume, tables and match uncertainty; sim owns bounded execution
and retained Commands. These inspected deals are development monitoring."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

from etude.eval.match import Row, Tables, agent
from etude.eval.records import Benchmark, Environment, Match, Matchup, Model
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
from pydantic import BaseModel

from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.eval.monitor import EvaluationRow, MonitorResult
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import selected_decks
from manabot.sim.registration import PlayerRegistration

BENCHMARK = "two-seat-ladder-heuristic-deck-seats"


class Conditions(BaseModel):
    deal: int
    key: EvaluationKey
    game_seconds: float
    max_commands: int


def benchmark(manifest: MonitorResult) -> Benchmark:
    """Freeze the arena protocol independently of checkpoint or training progress."""
    if (
        manifest.protocol.opponent != "heuristic"
        or not manifest.protocol.include_mirrors
    ):
        raise ValueError("Etude ladder benchmark requires heuristic and eight legs")
    # A model seat carries its fixed inference registration, with artifact facts
    # filled from Etude's checkpoint when played. Never put training coordinates
    # or host paths in the benchmark identity.
    registration = manifest.candidate.model_copy(
        update={
            "checkpoint_sha256": "0" * 64,
            "checkpoint_bytes": 1,
            "artifact_id": "monitor/checkpoint",
        }
    )
    code = [
        path
        for folder in (
            "manabot/eval",
            "manabot/model",
            "manabot/env",
            "manabot/sim",
            "manabot/infra",
        )
        for path in sorted(Path(folder).rglob("*.py"))
    ] + [Path(__file__)]
    opponent = agent(
        "heuristic",
        code=code,
        config={"registration": manifest.opponent.model_dump(mode="json")},
    )
    return Benchmark(
        name=BENCHMARK,
        environment=Environment(
            name=f"{manifest.key.world}-allies-lessons-development",
            files=canonical_sha256(manifest.evaluation_identities),
            config={"purpose": "repeated-development-monitoring"},
        ),
        matchups=(
            Matchup(
                name="checkpoint vs heuristic",
                players=(
                    Model(
                        code=code,
                        config={"registration": registration.model_dump(mode="json")},
                    ),
                    opponent,
                ),
                conditions=tuple(
                    Conditions(
                        deal=seed,
                        key=manifest.key,
                        game_seconds=manifest.protocol.game_seconds,
                        max_commands=manifest.protocol.max_commands,
                    ).model_dump(mode="json")
                    for seed in manifest.protocol.deal_seeds
                ),
                games=8,
            ),
        ),
    )


def play(match: Match) -> Tables:
    """Play a complete deal; failed integrity leaves arena evidence and fails Etude."""
    if len(match.agents) != 2 or match.games != 8:
        raise ValueError("expected one eight-leg checkpoint/heuristic deal")
    checkpoint, anchor = match.agents
    uri = urlsplit(checkpoint.uri or "")
    if checkpoint.kind != "checkpoint" or uri.scheme != "file" or uri.netloc:
        raise ValueError("checkpoint must be a verified local file")
    path = Path(unquote(uri.path))
    candidate = PlayerRegistration.model_validate(checkpoint.config["registration"])
    candidate = candidate.model_copy(
        update={
            "checkpoint_sha256": checkpoint.artifact,
            "checkpoint_bytes": path.stat().st_size,
            "artifact_id": f"monitor/{checkpoint.artifact}",
        }
    )
    verify_file(path, checkpoint.artifact, path.stat().st_size)
    opponent = PlayerRegistration.model_validate(anchor.config["registration"])
    conditions = Conditions.model_validate(match.conditions)
    out = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    out.mkdir(parents=True, exist_ok=True)
    rows, trace, replay = play_cell(
        key=conditions.key,
        player_a=candidate,
        player_b=opponent,
        deal_seeds=(conditions.deal,),
        out_dir=out,
        checkpoint_paths={candidate.player_id: str(path)},
        game_seconds=conditions.game_seconds,
        max_commands=conditions.max_commands,
        matchup_mode="all",
    )
    atomic_json(out / "rows.json", rows)
    atomic_json(out / "trace.json", trace)
    atomic_json(out / "replay.json", replay)
    return evaluation_tables([EvaluationRow.model_validate(row) for row in rows])


def evaluation_tables(rows: Sequence[EvaluationRow]) -> Tables:
    """Project one replay-checked deal using each player's actual authored deck.

    Cross-deck opponents exchange seats inside a pair of legs: the candidate
    plays Lessons in legs 0 and 3, Allies in 1 and 2. Leg pairs alone name the
    authored setup, not which deck the candidate controls.
    """
    if (
        {row.leg for row in rows} != set(range(8))
        or len(rows) != 8
        or not all(row.valid for row in rows)
    ):
        raise ValueError("arena deal incomplete or failed integrity; evidence retained")
    labels = {
        ("ur_lessons", "gw_allies"): "LvA",
        ("gw_allies", "ur_lessons"): "AvL",
        ("ur_lessons", "ur_lessons"): "LvL",
        ("gw_allies", "gw_allies"): "AvA",
    }
    plays: list[Row] = []
    for row in rows:
        decks = selected_decks(row.leg)
        extra = row.model_extra or {}
        if (
            extra.get("seat_decks") != list(decks)
            or extra.get("player_a_seat") != row.leg % 2
        ):
            raise ValueError("arena row differs from authored deck/seat assignment")
        score = row.score_a
        if score is None:
            raise ValueError("valid arena row requires a score")
        # Etude's player is a tournament participant, not the native on-play
        # seat. Deck labels and on-play flags describe that row's participant.
        for player in (0, 1):
            native_seat = (row.leg + player) % 2
            plays.append(
                {
                    "player": player,
                    "game": row.leg,
                    "score": score if player == 0 else 1.0 - score,
                    "matchup": labels[(decks[native_seat], decks[1 - native_seat])],
                    "on_play": native_seat == 0,
                    "valid": True,
                }
            )
    return {"plays": plays}


def evaluate(manifest: MonitorResult, directory: Path) -> None:
    """Hold one digest-bound benchmark; the parent owns the process deadline."""
    path = Path(manifest.artifact["path"]).resolve()
    verify_file(path, manifest.artifact["sha256"], manifest.artifact["bytes"])
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["ETUDE_OUTPUT_DIR"] = str(directory)
    os.environ["ETUDE_RUN"] = "checkpoint-" + manifest.artifact["sha256"]
    result = hold(
        benchmark(manifest),
        play=play,
        checkpoint=ArtifactRef(
            uri=path.as_uri(),
            sha256=manifest.artifact["sha256"],
            size_bytes=manifest.artifact["bytes"],
        ),
        directory=directory.parent.parent / "track",
        workers=1,
    )
    atomic_json(directory / "result.json", result.model_dump(mode="json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        MonitorResult.model_validate_json(args.manifest.read_bytes()),
        args.manifest.parent,
    )
