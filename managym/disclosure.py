"""Game disclosure values and seeded per-game disclosure sampling."""

from typing import Literal

DecklistDisclosure = Literal["open", "closed"]
DecklistSampling = Literal[DecklistDisclosure, "mixed"]
