"""Target evidence tests use synthetic viewer observations, never play games."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from manabot.study.allies_lessons.targeting_audit import audit
from manabot.study.record import Decision, GameRecord, read_games, write_games
from manabot.study.targeting import (
    TargetChoice,
    TargetCount,
    inspect_target,
    target_counts,
)
from managym.enums import ActionEnum, ActionSpaceEnum

if TYPE_CHECKING:
    from managym import Observation


def observation(
    target_id: int = 30,
    *,
    source_known: bool = True,
    decline: bool = False,
    controller_id: int = 10,
) -> Observation:
    # The native extension owns construction; expose exactly its viewer fields
    # here without requiring a native build or starting a game.
    return cast(
        "Observation",
        SimpleNamespace(
            agent=SimpleNamespace(id=10),
            opponent=SimpleNamespace(id=20),
            agent_cards=[SimpleNamespace(id=100, name="Removal")],
            opponent_cards=[],
            agent_permanents=[SimpleNamespace(id=30, controller_id=controller_id)],
            opponent_permanents=[SimpleNamespace(id=40, controller_id=20)],
            action_space=SimpleNamespace(
                action_space_type=ActionSpaceEnum.CHOOSE_TARGET,
                focus=[100] if source_known else [],
                actions=[
                    SimpleNamespace(
                        action_type=ActionEnum.DECLINE_CHOICE
                        if decline
                        else ActionEnum.CHOOSE_TARGET,
                        focus=[target_id],
                    )
                ],
            ),
        ),
    )


@pytest.mark.parametrize(
    ("target_id", "kind", "side"),
    [
        (10, "player", "own"),
        (20, "player", "opponent"),
        (30, "permanent", "own"),
        (40, "permanent", "opponent"),
        (99, "unresolved", None),
    ],
)
def test_target_side_uses_public_controller_identity(
    target_id: int, kind: str, side: str | None
) -> None:
    choice = inspect_target(observation(target_id), 0)
    assert choice is not None
    assert (choice.source, choice.target_kind, choice.side) == ("Removal", kind, side)


def test_decline_and_missing_source_stay_explicit() -> None:
    assert inspect_target(observation(decline=True), 0) == TargetChoice(
        "Removal", "declined", None
    )
    assert inspect_target(observation(source_known=False), 0) == TargetChoice(
        None, "permanent", "own"
    )


@pytest.mark.parametrize(
    ("controller_id", "expected"),
    [
        (20, TargetChoice("Removal", "permanent", "opponent")),
        (99, TargetChoice("Removal", "unresolved", None)),
    ],
)
def test_permanent_side_follows_controller_not_observation_list(
    controller_id: int, expected: TargetChoice
) -> None:
    assert inspect_target(observation(controller_id=controller_id), 0) == expected


def decision(target: TargetChoice | None = None) -> Decision:
    return Decision(
        turn=1,
        active=0,
        actor=0,
        phase="PRECOMBAT_MAIN",
        step="PRECOMBAT_MAIN_STEP",
        kind="CHOOSE_TARGET",
        offered=("CHOOSE_TARGET",),
        offer_labels=("Target",),
        chosen="CHOOSE_TARGET",
        chosen_index=0,
        label="Target",
        declared=None,
        subject=None,
        life=(20, 20),
        hand=3,
        mine=(),
        theirs=(),
        target=target,
    )


def game(target: TargetChoice | None = None) -> GameRecord:
    return GameRecord(
        "g",
        ("greedy", "other"),
        ("ur_lessons", "gw_allies"),
        1,
        decisions=[decision(target)],
    )


def test_record_round_trip_and_old_record_coverage(tmp_path: Path) -> None:
    record = game(TargetChoice("Removal", "permanent", "own"))
    path = tmp_path / "games.jsonl.gz"
    write_games(path, [record])
    assert read_games(path) == [record]
    legacy = asdict(record)
    del legacy["decisions"][0]["target"]
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(legacy) + "\n")
    rows = target_counts([record, *read_games(old)], "greedy")
    assert sum(row.count for row in rows) == 2
    assert {row.choice for row in rows} == {None, record.decisions[0].target}
    assert all(
        (row.deck, row.opponent_deck) == ("ur_lessons", "gw_allies") for row in rows
    )
    result = audit(path, "greedy")
    assert result.unfinished_games == 1
    assert result.records_sha256
    with pytest.raises(ValueError, match="no records"):
        audit(path, "missing")


def test_counts_use_actor_seat_and_keep_mirror_cells() -> None:
    choice = TargetChoice("Removal", "player", "opponent")
    d = decision(choice)
    record = GameRecord(
        "mirror",
        ("other", "greedy"),
        ("ur_lessons", "ur_lessons"),
        2,
        decisions=[d, replace(d, actor=1)],
    )
    rows = target_counts([record], "greedy")
    assert len(rows) == 1
    assert (rows[0].deck, rows[0].opponent_deck, rows[0].seat, rows[0].count) == (
        "ur_lessons",
        "ur_lessons",
        1,
        1,
    )


def test_audit_keeps_all_cells_with_missing_cohorts_explicit(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    write_games(path, [game()])
    result = audit(path, "greedy")
    cells = result.cells
    assert [cell.cell for cell in cells] == ["LvA", "AvL", "LvL", "AvA"]
    assert [cell.player_seats for cell in cells] == [1, 0, 0, 0]
    assert cells[0].counts == [TargetCount("ur_lessons", "gw_allies", 0, None, 1)]
    assert all(cell.counts == [] for cell in cells[1:])


def test_audit_counts_both_labelled_seats_but_each_game_once(tmp_path: Path) -> None:
    own = decision(TargetChoice("Removal", "permanent", "own"))
    mirror = replace(
        game(),
        labels=("greedy", "greedy"),
        decks=("ur_lessons", "ur_lessons"),
        decisions=[own, replace(own, actor=1)],
    )
    unrelated = replace(game(), labels=("other", "other"), decks=("unknown", "unknown"))
    path = tmp_path / "games.jsonl"
    write_games(path, [mirror, unrelated])

    result = audit(path, "greedy")
    assert result.games == result.unfinished_games == 1
    assert [cell.player_seats for cell in result.cells] == [0, 0, 2, 0]
    assert [cell.unfinished_player_seats for cell in result.cells] == [0, 0, 2, 0]
    assert result.cells[2].counts == [
        TargetCount("ur_lessons", "ur_lessons", 0, own.target, 1),
        TargetCount("ur_lessons", "ur_lessons", 1, own.target, 1),
    ]


def test_audit_rejects_unsupported_decks_for_labelled_seats(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    write_games(path, [replace(game(), decks=("unknown", "gw_allies"))])
    with pytest.raises(ValueError, match="Allies/Lessons"):
        audit(path, "greedy")


def test_non_target_prompt_does_not_produce_target_evidence() -> None:
    raw = cast(
        "Observation",
        SimpleNamespace(
            action_space=SimpleNamespace(
                action_space_type=ActionSpaceEnum.PRIORITY,
            )
        ),
    )
    assert inspect_target(raw, 0) is None
