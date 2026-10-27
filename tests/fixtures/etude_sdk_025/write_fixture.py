# /// script
# requires-python = ">=3.12"
# dependencies = ["etude-sdk[eval]==0.2.5"]
# ///
"""Write synthetic frozen records with the actual pre-migration SDK.

Run explicitly with `uv run --script tests/fixtures/etude_sdk_025/write_fixture.py`.
This archival producer intentionally uses 0.2.5 APIs. It imports no Manabot
execution code, loads no policies and plays no native games.
"""

from importlib.metadata import version
import os
from pathlib import Path
import tempfile
from typing import Literal

from etude.eval.match import Tables, evaluate
from etude.eval.records import Environment, Match
from pydantic import BaseModel, TypeAdapter


class Case(BaseModel):
    case: str
    experiment: str
    run: str
    matches: list[Match]


class Group(BaseModel):
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deals: tuple[int, ...]


class GridConditions(BaseModel):
    kind: str
    groups: tuple[Group, ...]


class CardConditions(BaseModel):
    deals: tuple[int, ...]


def outcomes(match: Match) -> Tables:
    rows: list[dict[str, int | float | str | bool]] = []
    if "groups" in match.conditions:
        conditions = GridConditions.model_validate(match.conditions)
        for group in conditions.groups:
            for deal in group.deals:
                for player in (0, 1):
                    rows.append(
                        {
                            "player": player,
                            "seat": (group.model_seat + player) % 2,
                            "unit": conditions.kind,
                            "deal": deal,
                            "deck": group.model_deck
                            if player == 0
                            else group.heuristic_deck,
                            "opponent_deck": group.heuristic_deck
                            if player == 0
                            else group.model_deck,
                            "score": float(1 - player),
                            "valid": True,
                            "terminated": True,
                            "truncated": False,
                            "decisions": 1,
                            "seconds": 0.01,
                        }
                    )
        return {"games": rows}
    if "deals" in match.conditions:
        for deal in CardConditions.model_validate(match.conditions).deals:
            for deck, opponent in (
                ("elemental_ur", "elemental_gw"),
                ("elemental_gw", "elemental_ur"),
            ):
                for seat in (0, 1):
                    rows.append(
                        {
                            "player": 0,
                            "seat": seat,
                            "deal": deal,
                            "deck": deck,
                            "opponent_deck": opponent,
                            "score": 1.0,
                            "valid": True,
                            "terminated": True,
                            "truncated": False,
                            "decisions": 1,
                            "model_decisions": 1,
                            "seconds": 0.01,
                        }
                    )
        return {"games": rows}
    return {
        "plays": [
            {
                "player": player,
                "game": leg,
                "score": float(1 - player),
                "on_play": (leg + player) % 2 == 0,
            }
            for leg in range(8)
            for player in (0, 1)
        ]
    }


def main() -> None:
    assert version("etude-sdk") == "0.2.5"
    root = Path(__file__).parent
    cases = TypeAdapter(list[Case]).validate_json((root / "input.json").read_bytes())
    with tempfile.TemporaryDirectory() as home:
        os.environ["ETUDE_HOME"] = home
        for case in cases:
            result = evaluate(
                experiment=case.experiment,
                environment=Environment(name="migration-fixture", files="a" * 64),
                matches=case.matches,
                play=outcomes,
                run=case.run,
                directory=root / case.case,
            )
            assert result.played == len(case.matches)


if __name__ == "__main__":
    main()
