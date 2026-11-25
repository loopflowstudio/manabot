"""
env.py
Environment wrapper around the managym Rust engine that conforms to the Gymnasium API.
"""

from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces

# Local imports
from manabot.infra.log import getLogger
import managym
from managym.decision import (
    Command,
    SemanticContractError,
    SemanticTransition,
)
from managym.recording import DecisionAnnotation

# Local directory imports
from .failures import CapturedSearch, ScalarCapture
from .match import Match, Reward
from .observation import ObservationSpace, encode_observation


def add_truncation_flags(
    raw_obs: managym.Observation,
    info: dict[str, Any],
    space: ObservationSpace,
) -> None:
    info["action_space_truncated"] = (
        len(raw_obs.action_space.actions) > space.config.max_actions
        and space.config.action_overflow == "error"
    )
    if space.config.action_overflow == "first-legal":
        info["policy_action_cutoff"] = max(
            0, len(raw_obs.action_space.actions) - space.config.max_actions
        )
    info["card_space_truncated"] = (
        len(raw_obs.agent_cards) > space.config.max_cards_per_player
        or len(raw_obs.opponent_cards) > space.config.max_cards_per_player
    )
    info["permanent_space_truncated"] = (
        len(raw_obs.agent_permanents) > space.config.max_permanents_per_player
        or len(raw_obs.opponent_permanents) > space.config.max_permanents_per_player
    )


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        match: Match,
        obs_space: ObservationSpace,
        reward: Reward,
        seed: int = 0,
        enable_profiler: bool = False,
        *,
        capture: ScalarCapture | None = None,
    ) -> None:
        """Gymnasium-compatible wrapper around one native game.

        A finished game stays finished: ``step`` returns the terminal
        observation and true done flags, and the caller resets explicitly.
        """
        super().__init__()
        self.seed = seed
        self._capture = (
            capture
            if capture is not None
            else ScalarCapture(
                seed,
                decision_memory=obs_space.config.max_decision_records > 0,
                decision_memory_include_own=obs_space.config.decision_memory_include_own,
            )
        )
        self.skip_trivial = True
        self.enable_profiler = enable_profiler
        logger = getLogger(__name__)
        logger.info(
            f"Initializing Env with seed={self.seed}, skip_trivial={self.skip_trivial}, enable_profiler={self.enable_profiler}"
        )
        self._engine = managym.Env(
            seed=self.seed,
            decision_memory=obs_space.config.max_decision_records > 0,
            decision_memory_include_own=obs_space.config.decision_memory_include_own,
            skip_trivial=self.skip_trivial,
            enable_profiler=self.enable_profiler,
        )

        # For when we need manabot.ObservationSpace
        self.failure_context = self._capture.context
        self._capture.identities["input_shapes"] = {
            name: list(shape) for name, shape in obs_space.shapes.items()
        }
        self.obs_space: ObservationSpace = obs_space
        # Type: gymnasium.Space
        self.observation_space = self.obs_space
        self.action_space = spaces.Discrete(self.obs_space.config.max_actions)

        self.match = match
        self.reward = reward

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[dict, dict]:
        """
        Resets the environment to an initial state and returns (observation, info).

        Args:
            seed: Optional seed for environment’s RNG.
            options: Optional `match` replaces the setup, including later resets.

        Returns:
            observation: A dictionary of numpy arrays (encoded from managym.Observation).
            info: Additional debug info from managym, as a dict of string->string.
        """
        # Gymnasium requires calling this for seeding (if you use self.np_random)
        super().reset(seed=seed)

        # Honor a new seed natively. Without reseeding the engine,
        # reset(seed=...) silently replays the identical deal (same
        # shuffles/opening hands) for every value of seed — which is what the
        # evaluation harness had been doing historically (see
        # experiments/reports/exp-06-newworld-training.md). The native set_seed
        # binding supersedes the engine-rebuild workaround exp-06 landed.
        if seed is not None and seed != self.seed:
            self.seed = seed
            self._engine.set_seed(seed)
            self._capture.seed = seed

        match = self.match if not options else options.get("match", self.match)

        # Get the initial managym observation
        raw_obs, raw_info = self._capture.reset(self._engine, match.to_rust())
        self.match = match
        self._last_obs = raw_obs
        # Encode to our dictionary-of-numpy format
        py_obs = encode_observation(self.obs_space, raw_obs)

        return py_obs, raw_info

    def step(
        self,
        action: int,
        *,
        annotation: DecisionAnnotation | dict[str, object] | None = None,
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Step the environment by applying `action` (int). A terminal step returns
        the terminal observation; the caller resets.

        Args:
            action: Chosen action index (within our placeholder discrete space).

        Returns:
            observation: Dictionary of numpy arrays.
            reward: Float reward for this step.
            terminated: Whether the episode ended because the game ended in a terminal state.
            truncated: Whether the episode ended due to a timelimit or external condition.
            info: Additional debug info from managym, e.g. partial game logs.
        """
        log = getLogger(__name__).getChild("step")
        self._record_policy_actions()
        raw_obs, raw_reward, terminated, truncated, info = self._capture.step(
            self._engine, action, annotation=annotation
        )
        reward = self.reward.compute(raw_reward, self._last_obs, raw_obs)
        add_truncation_flags(raw_obs, info, self.obs_space)

        log.debug(
            f"Stepped env. Step output: reward={raw_reward}, terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            log.info(f"Episode terminated: {terminated}, truncated: {truncated}")

        py_obs = encode_observation(self.obs_space, raw_obs)
        self._last_obs = raw_obs
        return py_obs, reward, terminated, truncated, info

    def step_semantic(
        self,
        command: Command,
        *,
        annotation: DecisionAnnotation | dict[str, object] | None = None,
    ) -> tuple[dict, float, bool, bool, dict, SemanticTransition]:
        """Execute one authoritative semantic Command.

        This evaluation path deliberately bypasses positional ``Env.step`` and
        consumes the native atomic semantic-step result, including its ordinary
        next-actor observation and canonical transition receipt.
        """

        self._record_policy_actions()
        try:
            transition_json, raw_obs, raw_reward, terminated, truncated, info = (
                self._capture.command(
                    self._engine,
                    command.to_json(),
                    transition_only=False,
                    annotation=annotation,
                )
            )
        except Exception as error:  # managym.AgentError
            message = str(error)
            code = message.split(":", 1)[0] if ":" in message else None
            raise SemanticContractError(message, code=code) from error

        transition = SemanticTransition.from_json(transition_json)
        reward = self.reward.compute(raw_reward, self._last_obs, raw_obs)
        winner = self._engine.winner_index()
        if winner is not None:
            info["winner_index"] = int(winner)
        add_truncation_flags(raw_obs, info, self.obs_space)
        self._last_obs = raw_obs
        return (
            encode_observation(self.obs_space, raw_obs),
            reward,
            terminated,
            truncated,
            info,
            transition,
        )

    def rollout_pool(
        self, worlds: int, rollouts: int, seed: int, max_steps: int
    ) -> CapturedSearch:
        return CapturedSearch(
            self._capture,
            self._engine,
            "rollout_pool",
            [worlds, rollouts, seed, max_steps],
        )

    def flat_mc_scores(
        self, worlds: int, rollouts: int, seed: int, max_steps: int
    ) -> tuple[list[float], int, int]:
        return CapturedSearch(
            self._capture,
            self._engine,
            "flat_mc_scores",
            [worlds, rollouts, seed, max_steps],
        ).native

    def scenario_refresh(self) -> tuple[dict, "managym.Observation"]:
        """Re-sync after managym scenario_* state injection.

        The competency harness (manabot/verify/competency.py) constructs
        positions via the engine's scenario_* helpers on ``self._engine``;
        those leave the cached observation and action space stale. This
        recomputes both and returns (encoded_obs, raw_obs).
        """
        raw_obs = self._engine.scenario_refresh()
        self._last_obs = raw_obs
        return encode_observation(self.obs_space, raw_obs), raw_obs

    def hand_snapshot(self, player: int) -> managym.HandSnapshot:
        """Read a native seat's authoritative hand, separate from policy inputs."""
        return self._engine.hand_snapshot(player)

    def render(self):
        pass

    def info(self) -> Dict[str, Any]:
        return self._engine.info()

    def content_pack_manifest(self) -> Dict[str, Any]:
        """Describe the exact immutable content pack retained by the match."""

        return self._engine.content_pack_manifest()

    def skip_trivial_count(self) -> int:
        """Trivial decision points auto-collapsed since the current game began."""
        return self._engine.skip_trivial_count()

    def _record_policy_actions(self) -> None:
        config = self.obs_space.config
        if config.action_overflow == "first-legal":
            self._capture.inputs.actor = self._engine.current_agent_index()
            self._capture.inputs.frame = self._engine.semantic_decision_frame_json()
            self._capture.record_policy_actions(
                len(self._last_obs.action_space.actions),
                str(self._last_obs.action_space.action_space_type).removeprefix(
                    "ActionSpaceEnum."
                ),
                config.max_actions,
                self._capture.inputs,
            )

    def close(self, *, reason: str = "capture closed") -> None:
        self._capture.close(reason=reason)
        self._capture.flush_policy_actions()

    @property
    def last_raw_obs(self):
        return self._last_obs
