"""Current-world capacity witnesses reached through ordinary legal actions.

The frozen pilot tapes depend on former automatic source choices. These small
controlled games cover their transport boundaries without claiming to replay
that historical gameplay. Each test checks the resulting boundary explicitly.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import managym


class CapacityPosition(BaseModel):
    seed: int
    decks: list[dict[str, int]]
    actions: list[int]

    def setup(self, *, closed: bool = False) -> managym.GameSetup:
        return managym.GameSetup(
            [managym.PlayerConfig(f"p{i}", deck) for i, deck in enumerate(self.decks)],
            decklists="closed" if closed else "open",
        )


class CapacityWitnesses(BaseModel):
    engine_revision: str
    purpose: str
    positions: dict[str, CapacityPosition]


def capacity_position(
    name: Literal["land", "wan", "drill140", "drill270"],
) -> CapacityPosition:
    path = Path(__file__).parent / "fixtures/current-capacity-w17-smart-payment.json"
    return CapacityWitnesses.model_validate_json(path.read_bytes()).positions[name]
