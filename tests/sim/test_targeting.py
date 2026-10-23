"""Target inspection follows viewer-visible identity and current control."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from managym.enums import ActionEnum, ActionSpaceEnum
import pytest

from manabot.sim.action.targeting import TargetChoice, inspect_target

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
