"""Etude benchmarks backed by the replay-checked Manabot arena.

One Etude match is one deal with all eight correlated deck/seat legs. Etude
owns match records, resume, Parquet tables and whole-match uncertainty. Arena
owns game limits, legal actions and retained command/replay evidence. These
repeatedly inspected deals are development monitoring, not a held-out study.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import cast

from etude.eval.records import (
    Benchmark,
    CheckpointRef,
    CheckpointSlot,
    Code,
    Composition,
    Environment,
    Match,
    Seating,
)
from etude.eval.result import Row, Tables, Value
from etude.eval.tournament import evaluating
from etude.records import ArtifactRef
from pydantic import BaseModel

from manabot.eval.agents import AGENTS, etude_registration
from manabot.eval.behaviour import checkpoint_counts
from manabot.eval.evaluation import (
    EvaluationRequest,
    GamePlayCounts,
    continue_evaluation,
    evaluate as evaluate_panel,
    evaluation_play_counts,
)
from manabot.eval.models import EvaluationKey
from manabot.eval.monitor import MonitorResult
from manabot.eval.results import EvaluationRow
from manabot.infra.artifacts import verify_file
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.sim.agent import GamePlayer
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup, selected_decks
from manabot.sim.registration import ETUDE_COMPUTE_CLASS, PlayerRegistration

BENCHMARK = "two-seat-ladder-heuristic-deck-seats-agents-v1"


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
    registration = etude_registration(manifest.candidate).model_copy(
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
    opponent = Code(
        name="heuristic",
        source=code,
        config={"registration": manifest.opponent.model_dump(mode="json")},
    )
    return Benchmark(
        name=BENCHMARK,
        environment=Environment(
            name=f"{manifest.key.world}-allies-lessons-development",
            files=canonical_sha256(manifest.evaluation_identities),
            config={"purpose": "repeated-development-monitoring"},
        ),
        seatings=tuple(
            Seating(
                name=f"checkpoint vs heuristic deal {seed}",
                players=(
                    Composition(
                        type="etude:configured",
                        fields={
                            "over": CheckpointSlot().model_dump(mode="json"),
                            "config": {
                                "registration": registration.model_dump(mode="json")
                            },
                        },
                    ),
                    opponent,
                ),
                config=Conditions(
                    deal=seed,
                    key=manifest.key.model_copy(
                        update={
                            "evaluation_compute_envelope_id": ETUDE_COMPUTE_CLASS,
                            "arena_version": "training-monitor-agents-v1",
                        }
                    ),
                    game_seconds=manifest.protocol.game_seconds,
                    max_commands=manifest.protocol.max_commands,
                ).model_dump(mode="json"),
                games=1,
            )
            for seed in manifest.protocol.deal_seeds
        ),
    )


def play(match: Match, agents: Sequence[object]) -> Tables:
    """Etude game callback: one eight-leg deal of the checkpoint against its anchor."""
    if len(match.players) != 2:
        raise ValueError("expected one eight-leg checkpoint/heuristic deal")
    checkpoint, anchor = match.players
    path = checkpoint.checkpoint.path
    sha256 = file_sha256(path)
    candidate = PlayerRegistration.model_validate(checkpoint.config["registration"])
    candidate = candidate.model_copy(
        update={
            "checkpoint_sha256": sha256,
            "checkpoint_bytes": path.stat().st_size,
            "artifact_id": f"monitor/{sha256}",
        }
    )
    opponent = PlayerRegistration.model_validate(anchor.config["registration"])
    return play_deal(
        match, candidate, opponent, {candidate.player_id: str(path)}, agents
    )


def play_deal(
    match: Match,
    candidate: PlayerRegistration,
    opponent: PlayerRegistration,
    checkpoint_paths: dict[str, str],
    agents: Sequence[object],
) -> Tables:
    """Play or continue the match's deal with Etude's agents, in seating order.

    Failed integrity leaves the arena evidence under the match's output
    directory and fails the Etude match.
    """
    conditions = Conditions.model_validate(match.config)
    out = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    request = EvaluationRequest(
        key=conditions.key,
        candidate=candidate,
        opponent=opponent,
        deal_seeds=(conditions.deal,),
        checkpoint_paths=checkpoint_paths,
        game_seconds=conditions.game_seconds,
        max_commands=conditions.max_commands,
        matchup_mode="all",
        purpose="repeated-development-comparison",
    )
    seated = {
        candidate.player_id: cast(GamePlayer, agents[0]),
        opponent.player_id: cast(GamePlayer, agents[1]),
    }
    result = (
        continue_evaluation(out, request=request, agents=seated)
        if out.exists()
        else evaluate_panel(request, out_dir=out, agents=seated)
    )
    return evaluation_tables(
        result.rows,
        play_counts=evaluation_play_counts(out)
        if opponent.runner_kind == "code"
        else None,
    )


def evaluation_tables(
    rows: Sequence[EvaluationRow],
    *,
    matchup: AuthoredMatchup | DeckMatchup | None = None,
    play_counts: GamePlayCounts | None = None,
    legs: int = 8,
) -> Tables:
    """Project one replay-checked deal using each player's actual authored deck.

    Cross-deck opponents exchange seats inside a pair of legs: with the selected
    Allies/Lessons pair the candidate plays Lessons in legs 0 and 3, Allies in 1
    and 2. Leg pairs alone name the authored setup, not which deck the candidate
    controls. Another authored `matchup` names its own pair, labelled
    "<deck> v <deck>", over the first `legs` of its matrix: the four cross legs,
    or all eight with the mirrors.
    """
    if {row.leg for row in rows} != set(range(legs)) or len(rows) != legs:
        raise ValueError("arena deal incomplete or failed integrity; evidence retained")
    labels = {
        ("ur_lessons", "gw_allies"): "LvA",
        ("gw_allies", "ur_lessons"): "AvL",
        ("ur_lessons", "ur_lessons"): "LvL",
        ("gw_allies", "gw_allies"): "AvA",
    }
    plays: list[Row] = []
    behaviour: list[Row] = []
    for row in rows:
        decks = selected_decks(row.leg) if matchup is None else matchup.decks(row.leg)
        extra = row.model_extra or {}
        if (
            extra.get("seat_decks") != list(decks)
            or extra.get("player_a_seat") != row.leg % 2
        ):
            raise ValueError("arena row differs from authored deck/seat assignment")
        score = row.score_a if row.valid else None
        # Etude's seat is a tournament participant, not the native on-play
        # seat. Deck labels and on-play flags describe that row's participant:
        # `deck` is the deck it actually played, `opponent_deck` the other's.
        for seat in (0, 1):
            game_seat = (row.leg + seat) % 2
            deck, opponent_deck = decks[game_seat], decks[1 - game_seat]
            participant: dict[str, Value] = {
                "seat": seat,
                "game": row.leg,
                "deck": deck,
                "opponent_deck": opponent_deck,
                "matchup": labels[deck, opponent_deck]
                if matchup is None
                else f"{deck} v {opponent_deck}",
                "game_seat": game_seat,
                "on_play": game_seat == 0,
            }
            if matchup is not None:
                participant["decklists"] = matchup.decklists
            plays.append(
                {
                    **participant,
                    "score": score if seat == 0 or score is None else 1.0 - score,
                    "valid": row.valid,
                    "failure": row.failure,
                }
            )
            if seat == 0 and row.valid and play_counts:
                pair = play_counts.get((row.deal_seed, row.leg))
                if pair is not None:
                    behaviour.append(
                        {**checkpoint_counts(pair, game_seat), **participant}
                    )
    tables = {"plays": plays}
    if behaviour:
        tables["behaviour"] = behaviour
    return tables


def evaluate(manifest: MonitorResult, directory: Path) -> None:
    """Hold one digest-bound benchmark; the parent owns the process deadline."""
    path = Path(manifest.artifact["path"]).resolve()
    verify_file(path, manifest.artifact["sha256"], manifest.artifact["bytes"])
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["ETUDE_OUTPUT_DIR"] = str(directory)
    os.environ["ETUDE_RUN"] = "checkpoint-" + manifest.artifact["sha256"]
    with evaluating(
        benchmark(manifest),
        experiment=BENCHMARK,
        # The training run's export, wherever its bytes are read from. A
        # manabot run is no Etude policy: its address is the key it was
        # always seated under, so retained matches keep their identity.
        checkpoint=CheckpointRef(
            address=f"{manifest.run_id}/checkpoints/{path.name}",
            artifact=ArtifactRef(
                uri=path.as_uri(),
                sha256=manifest.artifact["sha256"],
                size_bytes=manifest.artifact["bytes"],
            ),
        ),
        directory=directory.parent.parent / "track",
        resume=True,
    ) as active:
        result = active.play(play, agents=AGENTS)

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
