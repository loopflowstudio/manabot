"""Retained navigation positions come from ordinary selected-matchup Commands."""

from pathlib import Path

from pydantic import BaseModel
import pytest

from academy import server
from academy.server import GameSession
import managym
from managym.experience import ExperienceFrame


class PrefixStep(BaseModel):
    offer_id: int
    label: str


class NavigationPosition(BaseModel):
    config: dict[str, str | int | bool | dict[str, int]]
    prefix: list[PrefixStep]
    digest: str
    frame: ExperienceFrame


class NavigationFixture(BaseModel):
    positions: dict[str, NavigationPosition]


FIXTURE = Path(__file__).parents[2] / "academy/fixtures/choice-navigation-w11.json"
POSITIONS = NavigationFixture.model_validate_json(FIXTURE.read_text()).positions


@pytest.mark.parametrize("family", list(POSITIONS))
def test_retained_choice_position(
    tmp_path: Path, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # JSON is checked against the live producer, including full offers and projection.
    position = POSITIONS[family]
    # Reproduce this frozen evidence using its original native collapse mode.
    # Current Academy uses explicit decisions; the retained engine preset and
    # frozen frame bytes remain compatibility evidence, not current UX.
    native_env = managym.Env

    def historical_env(*, seed: int, skip_trivial: bool) -> managym.Env:
        return native_env(seed=seed, skip_trivial=True)

    monkeypatch.setattr(server.managym, "Env", historical_env)
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
    frame = dict(frame)
    frame["offers"] = [
        {**{k: v for k, v in offer.items() if k != "subjects"}, "source": None}
        for offer in frame["offers"]
    ]
    frame["projection"] = {
        k: v
        for k, v in frame["projection"].items()
        if k not in {"decision_context", "stack_objects"}
    }
    for player in ("agent", "opponent"):
        frame["projection"][player] = {
            k: v
            for k, v in frame["projection"][player].items()
            if k not in {"mana", "lesson_mana", "combat_mana", "inspected"}
        }
        # Historical frames omit optional grouping metadata. All original
        # object fields must still compare exactly.
        frame["projection"][player]["battlefield"] = [
            {k: v for k, v in permanent.items() if k not in {"types", "is_token"}}
            for permanent in frame["projection"][player]["battlefield"]
        ]
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


@pytest.mark.parametrize("family", ["CHOOSE_TARGET", "DISCARD"])
def test_current_explicit_choice_position(tmp_path: Path, family: str) -> None:
    current = NavigationFixture.model_validate_json(
        FIXTURE.with_name("choice-navigation-explicit-w10.json").read_text()
    ).positions[family]
    session = GameSession(tmp_path)
    try:
        frame = session.new_game(current.config)["frame"]
        for ordinal, step in enumerate(current.prefix):
            offer = next(
                offer for offer in frame["offers"] if offer["id"] == step.offer_id
            )
            assert offer["label"] == step.label
            result = session.hero_command(
                {
                    "command_id": f"current-navigation.{ordinal}",
                    "match_id": frame["match_id"],
                    "expected_revision": frame["revision"],
                    "prompt_id": frame["prompt"]["id"],
                    "offer_id": step.offer_id,
                    "answers": [],
                }
            )
            assert result["status"] == "accepted"
            frame = result["update"]["frame"]
        assert session.env.state_digest() == current.digest
        actual = ExperienceFrame.model_validate(frame)
        assert actual.action_space == family
        assert actual.offers == current.frame.offers
        assert actual.projection == current.frame.projection
    finally:
        session.close("test")
