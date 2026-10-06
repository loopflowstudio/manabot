"""Retained navigation positions come from ordinary selected-matchup Commands."""

from pathlib import Path

from pydantic import BaseModel
import pytest

from etude.experience_protocol import ExperienceFrame
from etude.server import GameSession


class PrefixStep(BaseModel):
    offer_id: int
    label: str


class NavigationPosition(BaseModel):
    config: dict[str, str | int | bool]
    prefix: list[PrefixStep]
    digest: str
    frame: ExperienceFrame


class NavigationFixture(BaseModel):
    positions: dict[str, NavigationPosition]


FIXTURE = Path(__file__).parents[2] / "etude/fixtures/choice-navigation.json"
POSITIONS = NavigationFixture.model_validate_json(FIXTURE.read_text()).positions


@pytest.mark.parametrize("family", list(POSITIONS))
def test_retained_choice_position(tmp_path: Path, family: str) -> None:
    # JSON is checked against the live producer, including full offers and projection.
    position = POSITIONS[family]
    session = GameSession(tmp_path)
    frame = session.new_game(position.config)["frame"]
    for ordinal, step in enumerate(position.prefix):
        offer = next(o for o in frame["offers"] if o["id"] == step.offer_id)
        assert offer["label"] == step.label
        result = session.hero_command(
            {
                "command_id": f"navigation.{ordinal}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert result["status"] == "accepted"
        frame = result["update"]["frame"]
    assert session.env is not None
    assert session.env.state_digest() == position.digest
    expected = position.frame.model_dump(mode="json", exclude_unset=True)
    for field in (
        "revision",
        "action_space",
        "offers",
        "projection",
        "content_hash",
        "asset_manifest_hash",
    ):
        assert frame[field] == expected[field], field

    # A stale click cannot mutate this retained position, even with a legal ID.
    result = session.hero_command(
        {
            "command_id": "navigation.stale",
            "match_id": frame["match_id"],
            "expected_revision": frame["revision"] - 1,
            "prompt_id": frame["prompt"]["id"],
            "offer_id": frame["offers"][0]["id"],
            "answers": [],
        }
    )
    assert result["status"] == "rejected"
    assert session.env.state_digest() == position.digest
