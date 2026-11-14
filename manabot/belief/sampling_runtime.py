"""Bind scheduled belief capture to the existing Env and frozen player APIs.

Checkpoint loading stays with the caller. The same adapter can therefore use an
ordinary admitted historical player without substituting current native rules.
"""

from collections.abc import Callable, Mapping

import numpy as np

from manabot.belief.sampling_collection import (
    CaptureSlot,
    CaptureTransition,
    GameCaptureFailure,
    retained_offer_count,
)
from manabot.env.env import Env
from manabot.env.match import Match
from manabot.sim.search.flat_mc import MatchupPlayer
import managym
from managym.decision import Command, DecisionFrame, Observation, SemanticContractError


class EnvCaptureRuntime:
    def __init__(
        self,
        env: Env,
        player: MatchupPlayer,
        match: Callable[[CaptureSlot], Match],
        *,
        hand: Callable[[int], Mapping[str, int]] | None = None,
    ) -> None:
        self.env = env
        self.player = player
        self.match = match
        self.hand = hand
        self.obs: dict[str, np.ndarray] = {}
        self.retained_actions = 0

    def reset(self, slot: CaptureSlot) -> None:
        # Env.reset reseeds only when the numeric seed changes. Paired slots
        # deliberately reuse it, so restore the declared deal RNG every time.
        self.env._engine.set_seed(slot.seed)
        self.obs, _ = self.env.reset(
            seed=slot.seed, options={"match": self.match(slot)}
        )

    def observation(self, viewer: int) -> Observation:
        return Observation.from_json(self.env._engine.semantic_observation_json(viewer))

    def hidden_hand_constraints_json(self, viewer: int) -> str:
        return self.env._engine.hidden_hand_constraints_json(viewer)

    def hand_counts(self, player: int) -> Mapping[str, int]:
        if self.hand is not None:
            return self.hand(player)
        return self.env.hand_snapshot(player).cards

    def frame(self) -> DecisionFrame:
        return DecisionFrame.from_json(self.env._engine.semantic_decision_frame_json())

    def act(self) -> int:
        self.retained_actions = retained_offer_count(
            self.frame(),
            self.obs["actions_valid"].tolist(),
            overflow=self.env.obs_space.encoder.config.action_overflow,
        )
        action = self.player.act(self.env, self.obs)
        if action < 0 or action >= self.retained_actions:
            raise ValueError("frozen policy selected an action outside encoded support")
        return action

    def step(self, command: Command) -> CaptureTransition:
        # Only native rules failures during active play are game-local. Reset,
        # encoding capacity, stale Commands and policy/infrastructure errors stay fatal.
        try:
            self.obs, _, terminated, truncated, _, transition = self.env.step_semantic(
                command
            )
        except BaseException as error:
            native = (
                error.__cause__ if isinstance(error, SemanticContractError) else error
            )
            # PyO3's panic type is created at runtime, without an importable module.
            panic = (
                type(native).__module__ == "pyo3_runtime"
                and type(native).__name__ == "PanicException"
            )
            known_rules_failure = (
                isinstance(native, managym.AgentError)
                and str(native) == "no legal targets"
            )
            if panic or known_rules_failure:
                raise GameCaptureFailure(
                    f"{type(native).__name__}: {native}"
                ) from error
            raise
        return CaptureTransition(transition.receipt, terminated, truncated)
