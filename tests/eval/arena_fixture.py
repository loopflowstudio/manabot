"""The frozen INT-6 skill-arena key and code-only anchors shared by evaluation tests."""

from dataclasses import dataclass
import json
from pathlib import Path

from manabot.eval.models import EvaluationKey
from manabot.sim.registration import PlayerRegistration

ARENA_FIXTURE = Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json"


@dataclass(frozen=True)
class ArenaFixture:
    key: EvaluationKey
    anchors: tuple[PlayerRegistration, ...]


def arena_fixture() -> ArenaFixture:
    raw = json.loads(ARENA_FIXTURE.read_text())
    return ArenaFixture(
        key=EvaluationKey.model_validate(raw["key"]),
        anchors=tuple(PlayerRegistration.model_validate(p) for p in raw["anchors"]),
    )
