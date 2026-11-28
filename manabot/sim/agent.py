"""The game-side player contract, independent of inference and orchestration."""

from typing import Protocol

from manabot.env.env import Env
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig
import managym


class GamePlayer(Protocol):
    @property
    def space(self) -> ObservationSpace | None: ...

    def admit(self, match: MatchConfig) -> None: ...

    def choose(self, env: Env, obs: managym.Observation, seed: int) -> int: ...
