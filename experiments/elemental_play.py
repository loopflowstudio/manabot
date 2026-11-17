"""Elemental groups in Etude game processes, using its built agents.

Each match owns one deck/seat group. Native games run here, retain Commands,
verify replays and hand card evidence to Etude. There are no nested workers,
model downloads, device choices or private inference allocations.
"""

from collections.abc import Sequence
from functools import cache
import os
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Literal

from etude.eval.match import environment
from etude.eval.records import CheckpointPlayer, Code, Environment, Match
from etude.eval.result import Result, Row
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import (
    GAME_SECONDS,
    MAX_COMMANDS,
    PACK,
    GameRow,
    Group,
    _legs,
)
from experiments.elemental_long import H2HGame
from manabot.eval.admission import register_anchor
from manabot.eval.agents import game_players, registration
from manabot.eval.card_evaluation import trace_card_tables
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import ETUDE_COMPUTE_CLASS
import managym

ROOT = Path(__file__).resolve().parents[1]
HEURISTIC = "heuristic"
type Kind = Literal["full", "subset", "pair"]


def eval_environment() -> Environment:
    return environment(
        "elemental-world",
        files=[
            ROOT / "experiments/elemental_long.py",
            ROOT / "experiments/elemental_grid.py",
        ],
        config={
            "world": managym.WORLD_VERSION,
            "pack": PACK,
            "game_seconds": GAME_SECONDS,
            "max_commands": MAX_COMMANDS,
        },
    )


def evidence_directory() -> Path:
    """Arena evidence: a launched attempt's `$ETUDE_OUTPUT_DIR/arena`, else
    `.runs/elemental/arena` for local `etude eval` and `etude smoke`."""
    return (
        Path(os.environ.get("ETUDE_OUTPUT_DIR") or ROOT / ".runs/elemental") / "arena"
    )


@cache
def heuristic_code() -> Code:
    return Code(name=HEURISTIC, source=heuristic_source_paths())


class Conditions(BaseModel):
    """One row of a grid: player 0's deck against each opposing deck and seat.

    Each group is one (opposing deck, player 0's native seat) cell half of the
    frozen schedule, played by one bounded worker process.
    """

    model_config = ConfigDict(extra="forbid")
    # A heuristic grid (`full` or the declared `subset`) or a checkpoint pair.
    kind: Kind
    deck: str
    groups: tuple[Group, ...]


def h2h_groups(games: Sequence[H2HGame]) -> list[Group]:
    """The cross-checkpoint schedule as grid groups: the newer checkpoint is `model_*`."""
    found: dict[tuple[str, str, int], list[int]] = {}
    for game in games:
        key = (game.newer_deck, game.older_deck, game.newer_seat)
        found.setdefault(key, []).append(game.deal)
    return [
        Group(model_deck=x, heuristic_deck=y, model_seat=seat, deals=tuple(deals))  # type: ignore[arg-type]
        for (x, y, seat), deals in found.items()
    ]


def game_rows(rows: Sequence[GameRow], kind: Kind) -> list[Row]:
    """Each game once per player; player 0's row is the rated checkpoint's.

    `unit` is the match's kind: an endpoint on a subset cadence update plays
    both heuristic grids, and the report keeps the full one.
    `seat` indexes the Etude participants; `game_seat` is the native seat
    (0 is on the play). `score` counts a draw as half and is null for an invalid
    game, so Etude's `compare` and the report count only valid games.
    """
    found: list[Row] = []
    for row in rows:
        score = row.score if row.valid else None
        sides = (
            (row.model_deck, row.heuristic_deck, row.model_seat, score),
            (
                row.heuristic_deck,
                row.model_deck,
                1 - row.model_seat,
                None if score is None else 1.0 - score,
            ),
        )
        for seat, (deck, opponent_deck, game_seat, player_score) in enumerate(sides):
            found.append(
                {
                    "seat": seat,
                    "unit": kind,
                    "deal": row.deal,
                    "deck": deck,
                    "opponent_deck": opponent_deck,
                    "game_seat": game_seat,
                    "score": player_score,
                    "valid": row.valid,
                    "terminated": row.terminated,
                    "truncated": row.truncated,
                    "failure": row.failure,
                    "decisions": row.decisions,
                    "seconds": row.seconds,
                }
            )
    return found


def rows(kind: Kind, found: Sequence[Group]) -> list[Conditions]:
    return [Conditions(kind=kind, deck=g.model_deck, groups=(g,)) for g in found]


def play(match: Match[Conditions], agents: Sequence[object]) -> Result:
    conditions = Conditions.model_validate(match.config)
    if len(conditions.groups) != 1:
        raise ValueError("new Elemental matches contain exactly one group")
    [group] = conditions.groups
    if group.model_deck != conditions.deck:
        raise ValueError("group differs from its declared deck")
    role_a, role_b, leg = _legs(group)
    matchup = AuthoredMatchup(pack_key=PACK, role_a=role_a, role_b=role_b)
    pair = conditions.kind == "pair"
    names = ("newer", "older") if pair else ("model", "heuristic")
    first = match.players[0]
    if not isinstance(first, CheckpointPlayer):
        raise TypeError("Elemental's first seat must be a checkpoint")
    candidate = registration(
        first,
        name=names[0],
        match=matchup.match(),
        content_suite=matchup.content_suite,
        role="challenger",
    )
    if pair:
        second = match.players[1]
        if not isinstance(second, CheckpointPlayer):
            raise TypeError("a checkpoint pair needs two checkpoints")
        opponent = registration(
            second,
            name=names[1],
            match=matchup.match(),
            content_suite=matchup.content_suite,
            role="anchor",
        )
    else:
        opponent = register_anchor(candidate, "heuristic", player_id=names[1])
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="elemental-etude-agents-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id=ETUDE_COMPUTE_CLASS,
    )
    # One directory per invocation: a retry keeps earlier journals and traces,
    # including those of an interrupted attempt.
    attempts = evidence_directory() / match.key
    attempts.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="attempt-", dir=attempts))
    with TemporaryDirectory() as scratch:
        cards = Path(scratch)
        played, trace, _ = play_cell(
            key=key,
            player_a=candidate,
            player_b=opponent,
            deal_seeds=group.deals,
            out_dir=output,
            game_seconds=GAME_SECONDS,
            max_commands=MAX_COMMANDS,
            authored_matchup=matchup,
            matchup_mode="cross" if leg < 4 else "mirrors",
            legs=(leg,),
            card_evidence_dir=cards,
            agents=dict(zip(names, game_players(match.players, agents), strict=True)),
        )
        found = trace_card_tables(cards, trace, names)
        games = [
            GameRow(
                model_deck=group.model_deck,
                heuristic_deck=group.heuristic_deck,
                model_seat=r["player_a_seat"],
                deal=r["deal_seed"],
                score=r["score_a"],
                valid=r["failure"] is None and r["replay_passed"],
                terminated=r["terminated"],
                truncated=r["truncated"],
                failure=r["failure"],
                decisions=r["decisions"],
                model_decisions=r["latency"][names[0]]["count"],
                seconds=r["game_seconds"],
            )
            for r in played
        ]
        return Result(
            tables={"games": game_rows(games, conditions.kind), **found.tables},
            blobs=found.blobs,
        )
