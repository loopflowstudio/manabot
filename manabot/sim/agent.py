"""The game-side player contract, independent of inference and orchestration."""

from collections.abc import Callable
from typing import Protocol

from manabot.env.env import Env
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import MatchConfig

type GameChoice = Callable[[Env, ObservationBuffers, int], int]


class GamePlayer(Protocol):
    @property
    def space(self) -> ObservationSpace | None: ...

    def admit(self, match: MatchConfig) -> None: ...

    def open_game(self, seed: int) -> GameChoice: ...
