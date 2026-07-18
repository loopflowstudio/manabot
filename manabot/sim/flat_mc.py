"""Flat determinized Monte Carlo player and seat-balanced matchup evaluation.

The searcher (exp-02, wave/intelligence C3) runs entirely on engine throughput: at
each of its decision points it asks the Rust engine (managym.Env.flat_mc_scores)
to evaluate every legal action by W determinized worlds x R uniformly-random
playouts per action, then plays the argmax. No network, no training.

Determinization (see managym/src/flow/search.rs): decklists are known to both
players, so hidden information is exactly the opponent's hand plus both library
orders. A sampled world replaces the opponent's hand with a uniform draw of
|hand| cards from their unseen pool (hand + library) and reshuffles both
libraries; all public state is preserved.

The matchup loop mirrors manabot.verify.util's seat balancing (hero alternates
between seat 0, on the play, and seat 1, on the draw) and its Wilson CI math,
but takes two arbitrary players instead of one agent + named opponent policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import (
    AgentHypers,
    MatchHypers,
    ObservationSpaceHypers,
    RewardHypers,
)
from manabot.model.agent import Agent
from manabot.verify.util import (
    INTERACTIVE_DECK,
    _select_agent_action,
    winner_from_info_or_obs,
)
from managym.decision import Command, DecisionFrame

DEFAULT_MAX_PLAYOUT_STEPS = 2000
DEFAULT_ROLLOUTS_PER_WORLD = 4

# managym ActionSpaceKind (managym/src/agent/action.rs), incl. the Stage-2/3
# mid-resolution decision kinds. Used for per-matchup decision profiles.
ACTION_SPACE_KIND_NAMES = {
    0: "game_over",
    1: "priority",
    2: "declare_attacker",
    3: "declare_blocker",
    4: "choose_target",
    5: "scry",
    6: "look_and_select",
    7: "pay_or_not",
    8: "modal",
    9: "discard_then_draw",
    10: "waterbend",
}


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a Bernoulli rate."""

    if total <= 0:
        return (0.0, 1.0)
    p = wins / total
    denom = 1.0 + (z**2) / total
    center = p + (z**2) / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)
    return (
        max(0.0, (center - margin) / denom),
        min(1.0, (center + margin) / denom),
    )


# -----------------------------------------------------------------------------
# Players
# -----------------------------------------------------------------------------


class MatchupPlayer(Protocol):
    """A player usable by play_games: picks an action index each decision."""

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int: ...


@dataclass
class SearchStats:
    """Accumulated cost/behavior of a search player across games."""

    decisions: int = 0
    seconds: float = 0.0
    simulations: int = 0
    cap_hits: int = 0
    decision_seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "decisions": float(self.decisions),
            "seconds": self.seconds,
            "simulations": float(self.simulations),
            "cap_hits": float(self.cap_hits),
        }


class FlatMCPlayer:
    """Flat determinized Monte Carlo with uniformly-random rollouts.

    ``sims`` is the strength dial N: simulations per legal action, split as
    W worlds x R rollouts (W = N / R, default R = 4). Worlds are shared across
    actions inside the engine (common random numbers).
    """

    def __init__(
        self,
        sims: int,
        *,
        rollouts_per_world: int = DEFAULT_ROLLOUTS_PER_WORLD,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        if sims < 1:
            raise ValueError("sims must be >= 1")
        self.sims = sims
        self.rollouts = max(1, min(rollouts_per_world, sims))
        self.worlds = max(1, sims // self.rollouts)
        self.max_steps = max_steps
        self._seed = seed
        self._calls = 0
        self.stats = SearchStats()
        #: Raw per-action playout scores from the most recent act() call
        #: (win-probability estimates in [0, 1], engine action order).
        self.last_scores: np.ndarray | None = None

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs  # search reads the raw engine state, not the encoding
        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        start = time.perf_counter()
        scores, simulations, cap_hits = env._engine.flat_mc_scores(
            self.worlds,
            self.rollouts,
            call_seed,
            self.max_steps,
        )
        elapsed = time.perf_counter() - start
        self.stats.decisions += 1
        self.stats.seconds += elapsed
        self.stats.simulations += int(simulations)
        self.stats.cap_hits += int(cap_hits)
        self.stats.decision_seconds.append(elapsed)
        self.last_scores = np.asarray(scores, dtype=np.float32)
        return int(np.argmax(scores))


class RandomMatchupPlayer:
    """Uniform random over valid actions in the encoded observation.

    Matches the RandomPolicy used for every historical vs-random baseline.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del env
        valid = np.flatnonzero(obs["actions_valid"] > 0)
        if len(valid) == 0:
            return 0
        return int(self._rng.choice(valid))


class AgentMatchupPlayer:
    """Trained policy player (stochastic, as in prior seat-balanced evals)."""

    def __init__(self, agent: Agent, deterministic: bool = False):
        self.agent = agent
        self.deterministic = deterministic
        agent.eval()

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del env
        return _select_agent_action(self.agent, obs, deterministic=self.deterministic)


# -----------------------------------------------------------------------------
# Player specs (picklable descriptions for multiprocessing workers)
# -----------------------------------------------------------------------------


def load_checkpoint_agent(path: str) -> tuple[Agent, ObservationSpace]:
    """Load an Agent from a training checkpoint, using its saved hypers."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hypers = checkpoint["hypers"]
    obs_space = ObservationSpace(ObservationSpaceHypers(**hypers["observation_hypers"]))
    agent = Agent(obs_space, AgentHypers(**hypers["agent_hypers"]))
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent, obs_space


def make_player(
    spec: dict[str, Any], seed: int
) -> tuple[MatchupPlayer, ObservationSpace | None]:
    """Build a player from a picklable spec.

    Specs:
        {"kind": "search", "sims": 64, "max_steps": 2000}
        {"kind": "determinized_puct", "sims": 64, "worlds": 4,
         "c_puct": 1.5, "max_steps": 2000}
        {"kind": "agent_puct", "sims": 16, "worlds": 4,
         "checkpoint": "/abs/visit-value.pt", "device": "cpu"}
        {"kind": "policy_search", "sims": 16, "checkpoint": "/abs/path.pt",
         "epsilon": 0.1, "rollouts_per_world": 1}
        {"kind": "random"}
        {"kind": "checkpoint", "path": "/abs/path/step_65536.pt"}
        {"kind": "value_greedy", "checkpoint": "/abs/value.pt", "device": "cpu"}
        {"kind": "value_search", "sims": 64, "checkpoint": "/abs/value.pt",
         "depth": 0, "rollouts_per_world": 1, "device": "cpu"}
        {"kind": "exact_range", "sims": 64, "checkpoint": "/abs/policy.pt",
         "checkpoint_sha256": "...", "epsilon": 0.05}
        {"kind": "uniform_range", ...same fields...}
    Returns (player, observation_space_or_None). Checkpoint players carry the
    ObservationSpace their encoder was trained with.
    """

    kind = spec["kind"]
    if kind == "search":
        return (
            FlatMCPlayer(
                int(spec["sims"]),
                rollouts_per_world=int(
                    spec.get("rollouts_per_world", DEFAULT_ROLLOUTS_PER_WORLD)
                ),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
            ),
            None,
        )
    if kind == "determinized_puct":
        from manabot.sim.mcts import DeterminizedPuctPlayer
        from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID

        return (
            DeterminizedPuctPlayer(
                int(spec["sims"]),
                worlds=int(spec.get("worlds", 4)),
                c_puct=float(spec.get("c_puct", 1.5)),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
                branch_driver_id=str(
                    spec.get("branch_driver_id", SELECTED_BRANCH_DRIVER_ID)
                ),
            ),
            None,
        )
    if kind == "agent_puct":
        from manabot.sim.mcts import AgentLeafEvaluator, DeterminizedPuctPlayer
        from manabot.sim.search_branch import SELECTED_BRANCH_DRIVER_ID

        if str(spec.get("device", "cpu")) != "cpu":
            raise ValueError("agent_puct inference is CPU-only")
        agent, obs_space = load_checkpoint_agent(spec["checkpoint"])
        return (
            DeterminizedPuctPlayer(
                int(spec["sims"]),
                worlds=int(spec.get("worlds", 4)),
                c_puct=float(spec.get("c_puct", 1.5)),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
                evaluator=AgentLeafEvaluator(agent, obs_space),
                branch_driver_id=str(
                    spec.get("branch_driver_id", SELECTED_BRANCH_DRIVER_ID)
                ),
            ),
            obs_space,
        )
    if kind == "policy_search":
        from manabot.sim.rollout import BatchedSampler, PolicyRolloutMCPlayer

        agent, obs_space = load_checkpoint_agent(spec["checkpoint"])
        sampler = BatchedSampler(
            agent,
            epsilon=float(spec.get("epsilon", 0.1)),
            temperature=float(spec.get("temperature", 1.0)),
            seed=seed,
            device=str(spec.get("device", "cpu")),
        )
        policy_plies = spec.get("policy_plies")
        return (
            PolicyRolloutMCPlayer(
                int(spec["sims"]),
                sampler,
                rollouts_per_world=int(spec.get("rollouts_per_world", 1)),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                policy_plies=int(policy_plies) if policy_plies is not None else None,
                seed=seed,
            ),
            obs_space,
        )
    if kind == "value_greedy":
        from manabot.sim.value import VGreedyPlayer, load_value_scorer

        scorer = load_value_scorer(
            spec["checkpoint"], device=str(spec.get("device", "cpu"))
        )
        return (
            VGreedyPlayer(
                scorer,
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
            ),
            None,
        )
    if kind == "value_search":
        from manabot.sim.value import ValueSearchPlayer, load_value_scorer

        scorer = load_value_scorer(
            spec["checkpoint"], device=str(spec.get("device", "cpu"))
        )
        return (
            ValueSearchPlayer(
                int(spec["sims"]),
                scorer,
                rollouts_per_world=int(spec.get("rollouts_per_world", 1)),
                depth=int(spec.get("depth", 0)),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
            ),
            None,
        )
    if kind in ("exact_range", "uniform_range"):
        from manabot.belief.likelihood import FrozenPolicyLikelihood
        from manabot.belief.player import ExactRangePlayer, RangeSampling

        likelihood = FrozenPolicyLikelihood(
            spec["checkpoint"],
            expected_sha256=str(spec["checkpoint_sha256"]),
            batch_size=int(spec.get("likelihood_batch_size", 256)),
            device=str(spec.get("device", "cpu")),
            counterfactual_seed=int(spec.get("counterfactual_seed", 0)),
        )
        return (
            ExactRangePlayer(
                int(spec["sims"]),
                likelihood=likelihood,
                sampling=(
                    RangeSampling.BELIEF
                    if kind == "exact_range"
                    else RangeSampling.COMPATIBLE_PRIOR
                ),
                epsilon=float(spec.get("epsilon", 0.05)),
                rollouts_per_world=int(
                    spec.get("rollouts_per_world", DEFAULT_ROLLOUTS_PER_WORLD)
                ),
                max_steps=int(spec.get("max_steps", DEFAULT_MAX_PLAYOUT_STEPS)),
                seed=seed,
            ),
            likelihood.obs_space,
        )
    if kind == "random":
        return RandomMatchupPlayer(seed=seed), None
    if kind == "checkpoint":
        agent, obs_space = load_checkpoint_agent(spec["path"])
        return (
            AgentMatchupPlayer(
                agent, deterministic=bool(spec.get("deterministic", False))
            ),
            obs_space,
        )
    raise ValueError(f"unknown player spec kind: {kind}")


def spec_name(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "search":
        return f"search-{spec['sims']}"
    if kind == "determinized_puct":
        return spec.get("name", f"dpuct-{spec['sims']}-w{spec.get('worlds', 4)}")
    if kind == "agent_puct":
        return spec.get("name", f"agent-puct-{spec['sims']}-w{spec.get('worlds', 4)}")
    if kind == "policy_search":
        return spec.get("name", f"psearch-{spec['sims']}")
    if kind == "checkpoint":
        return spec.get("name", spec["path"])
    if kind == "value_greedy":
        return spec.get("name", "vgreedy")
    if kind == "value_search":
        depth = int(spec.get("depth", 0))
        return spec.get("name", f"vsearch-{spec['sims']}-d{depth}")
    if kind in ("exact_range", "uniform_range"):
        default = "belief" if kind == "exact_range" else "uniform"
        return spec.get("name", f"{default}-{spec['sims']}")
    return kind


# -----------------------------------------------------------------------------
# Matchup loop
# -----------------------------------------------------------------------------


@dataclass
class GameRecord:
    game_index: int
    hero_seat: int
    hero_won: bool
    winner: int | None
    steps: int
    # Decision-profile instrumentation (exp-00 style): total turns and
    # surfaced decisions by ActionSpaceKind, split hero/villain.
    turns: int = 0
    hero_decisions: dict[str, int] = field(default_factory=dict)
    villain_decisions: dict[str, int] = field(default_factory=dict)


@dataclass
class MatchupResult:
    hero: str
    villain: str
    records: list[GameRecord]
    hero_search: SearchStats | None
    villain_search: SearchStats | None
    wall_seconds: float
    hero_evidence: dict[str, Any] | None = None
    villain_evidence: dict[str, Any] | None = None
    hero_known_truth: list[dict[str, Any]] = field(default_factory=list)
    villain_known_truth: list[dict[str, Any]] = field(default_factory=list)
    hero_replays: list[dict[str, Any]] = field(default_factory=list)
    villain_replays: list[dict[str, Any]] = field(default_factory=list)


def play_games(
    hero_spec: dict[str, Any],
    villain_spec: dict[str, Any],
    *,
    num_games: int,
    seed: int = 0,
    game_offset: int = 0,
    hero_deck: dict[str, int] | None = None,
    villain_deck: dict[str, int] | None = None,
) -> MatchupResult:
    """Play seat-balanced games between two player specs.

    Game ``i`` (globally indexed ``game_offset + i``) puts the hero in seat
    ``(game_offset + i) % 2``; seat 0 is on the play. The env is reseeded per
    game as in manabot.verify.util.
    """

    hero_player, hero_obs_space = make_player(hero_spec, seed=seed * 2 + 1)
    villain_player, villain_obs_space = make_player(villain_spec, seed=seed * 2 + 2)
    obs_space = hero_obs_space or villain_obs_space or ObservationSpace()

    match = Match(
        MatchHypers(
            hero=spec_name(hero_spec)[:32],
            villain=spec_name(villain_spec)[:32],
            hero_deck=hero_deck or dict(INTERACTIVE_DECK),
            villain_deck=villain_deck or dict(INTERACTIVE_DECK),
        )
    )
    match_swapped = match.swapped()
    env = Env(
        match,
        obs_space,
        Reward(RewardHypers()),
        seed=seed,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )

    records: list[GameRecord] = []
    hero_known_truth: list[Any] = []
    villain_known_truth: list[Any] = []
    start = time.perf_counter()
    for i in range(num_games):
        game_index = game_offset + i
        hero_seat = game_index % 2
        obs, _ = env.reset(
            seed=seed + game_index,
            options={"match": match_swapped} if hero_seat == 1 else None,
        )
        villain_seat = (hero_seat + 1) % 2
        for player, seat in (
            (hero_player, hero_seat),
            (villain_player, villain_seat),
        ):
            start_game = getattr(player, "start_game", None)
            if start_game is not None:
                start_game(env, seat)
        done = False
        steps = 0
        info: dict[str, Any] = {}
        hero_decisions: dict[str, int] = {}
        villain_decisions: dict[str, int] = {}
        while not done:
            raw = env.last_raw_obs
            acting = int(raw.agent.player_index)
            kind_id = int(raw.action_space.action_space_type)
            kind = ACTION_SPACE_KIND_NAMES.get(kind_id, f"unknown_{kind_id}")
            counts = hero_decisions if acting == hero_seat else villain_decisions
            counts[kind] = counts.get(kind, 0) + 1
            player = hero_player if acting == hero_seat else villain_player
            record_counts = {
                id(observer): len(tracker.records)
                for observer in (hero_player, villain_player)
                if (tracker := getattr(observer, "tracker", None)) is not None
            }
            action = player.act(env, obs)
            for observer in (hero_player, villain_player):
                prepare_step = getattr(observer, "prepare_step", None)
                if prepare_step is not None:
                    prepare_step(env, acting, action)
            semantic_execution = any(
                getattr(observer, "tracker", None) is not None
                for observer in (hero_player, villain_player)
            )
            transition = None
            if semantic_execution:
                command_id = f"flat-mc-{seed}-{game_index}-{steps}"
                command_for_action = getattr(player, "command_for_action", None)
                if command_for_action is not None:
                    command = command_for_action(
                        env._engine, action, command_id=command_id
                    )
                else:
                    frame = DecisionFrame.from_json(
                        env._engine.semantic_decision_frame_json()
                    )
                    if action < 0 or action >= len(frame.offers):
                        raise RuntimeError("action is outside the DecisionFrame")
                    command = Command(
                        command_id=command_id,
                        expected_revision=frame.revision,
                        offer_id=int(frame.offers[action]["id"]),
                    )
                obs, _, terminated, truncated, info, transition = env.step_semantic(
                    command
                )
            else:
                obs, _, terminated, truncated, info = env.step(action)
            for observer in (hero_player, villain_player):
                observe_step = getattr(observer, "observe_step", None)
                if observe_step is not None:
                    if transition is None:
                        raise RuntimeError(
                            "belief observer requires a semantic TransitionReceipt"
                        )
                    observe_step(env, acting, transition)
            for observer, audit_points in (
                (hero_player, hero_known_truth),
                (villain_player, villain_known_truth),
            ):
                tracker = getattr(observer, "tracker", None)
                if tracker is None or len(tracker.records) == record_counts.get(
                    id(observer), 0
                ):
                    continue
                from manabot.belief.audit import score_known_truth

                audit_points.append(
                    score_known_truth(
                        env._engine,
                        tracker,
                        game_index=game_index,
                        step=steps,
                    )
                )
            steps += 1
            done = bool(terminated or truncated)
        for player in (hero_player, villain_player):
            finish_game = getattr(player, "finish_game", None)
            if finish_game is not None:
                finish_game(game_index=game_index, seed=seed + game_index)
        winner = winner_from_info_or_obs(info, env.last_raw_obs)
        records.append(
            GameRecord(
                game_index=game_index,
                hero_seat=hero_seat,
                hero_won=winner == hero_seat,
                winner=winner,
                steps=steps,
                turns=int(env.last_raw_obs.turn.turn_number),
                hero_decisions=hero_decisions,
                villain_decisions=villain_decisions,
            )
        )
    wall_seconds = time.perf_counter() - start

    hero_evidence = (
        hero_player.evidence_stats() if hasattr(hero_player, "evidence_stats") else None
    )
    villain_evidence = (
        villain_player.evidence_stats()
        if hasattr(villain_player, "evidence_stats")
        else None
    )
    if hero_evidence is not None and hero_known_truth:
        from manabot.belief.audit import aggregate_known_truth

        hero_evidence["calibration"] = aggregate_known_truth(hero_known_truth)
    if villain_evidence is not None and villain_known_truth:
        from manabot.belief.audit import aggregate_known_truth

        villain_evidence["calibration"] = aggregate_known_truth(villain_known_truth)

    return MatchupResult(
        hero=spec_name(hero_spec),
        villain=spec_name(villain_spec),
        records=records,
        hero_search=getattr(
            hero_player, "search_stats", getattr(hero_player, "stats", None)
        ),
        villain_search=getattr(
            villain_player, "search_stats", getattr(villain_player, "stats", None)
        ),
        wall_seconds=wall_seconds,
        hero_evidence=hero_evidence,
        villain_evidence=villain_evidence,
        hero_known_truth=[point.to_dict() for point in hero_known_truth],
        villain_known_truth=[point.to_dict() for point in villain_known_truth],
        hero_replays=(
            hero_player.replay_receipts()
            if hasattr(hero_player, "replay_receipts")
            else []
        ),
        villain_replays=(
            villain_player.replay_receipts()
            if hasattr(villain_player, "replay_receipts")
            else []
        ),
    )


def aggregate_records(records: list[GameRecord]) -> dict[str, float]:
    """Overall and per-seat win rates with Wilson 95% intervals."""

    num_games = len(records)
    wins = sum(r.hero_won for r in records)
    lo, hi = wilson_interval(wins, num_games)
    metrics: dict[str, float] = {
        "num_games": float(num_games),
        "wins": float(wins),
        "win_rate": wins / num_games if num_games else 0.0,
        "win_ci_lower": lo,
        "win_ci_upper": hi,
        "mean_steps": (float(np.mean([r.steps for r in records])) if records else 0.0),
        "draws_or_caps": float(sum(r.winner is None for r in records)),
    }
    for seat in (0, 1):
        seat_records = [r for r in records if r.hero_seat == seat]
        seat_wins = sum(r.hero_won for r in seat_records)
        seat_lo, seat_hi = wilson_interval(seat_wins, len(seat_records))
        label = "play" if seat == 0 else "draw"
        metrics[f"games_on_{label}"] = float(len(seat_records))
        metrics[f"wins_on_{label}"] = float(seat_wins)
        metrics[f"win_rate_on_{label}"] = (
            seat_wins / len(seat_records) if seat_records else 0.0
        )
        metrics[f"win_ci_lower_on_{label}"] = seat_lo
        metrics[f"win_ci_upper_on_{label}"] = seat_hi
    return metrics
