"""Scalar game simulation and evaluation statistics recorded through Etude.

load_player admits local policies or exact Etude checkpoint records; simulate_models
admits each model against the actual match setup before play and returns one
`GameRecord` per game, which `simulation_summary` and `detailed_analysis` read.
run_simulation owns the evaluation tracking lifetime; game statistics never fit
a model.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Any

from etude.checkpoints import CheckpointRecord
from etude.records import Clock
from etude.run import declare, destinations
from etude.train.tracking import TrainingRun
import numpy as np

# Local imports
from manabot.backends import resolve
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import fetch_etude_checkpoint
from manabot.infra.config import ObservationConfig, RuntimeConfig, SimulationConfig
from manabot.infra.log import getLogger, setGlobalLogLevel
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import PolicyLoader, SimulationBackend
from manabot.sim.player import (
    DefaultPlayer,
    ModelPlayer,
    Player,
    RandomPlayer,
)

# -----------------------------------------------------------------------------
# Game Statistics
# -----------------------------------------------------------------------------


class GameOutcome(Enum):
    """Possible game outcomes."""

    HERO_WIN = "hero_win"
    VILLAIN_WIN = "villain_win"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class GameRecord:
    """One completed game with the environment's last profiler info.

    `profiler` maps a timer to its `total_time` seconds and call `count`.
    """

    outcome: GameOutcome
    steps: int
    duration: float
    thread_id: int | None = None
    profiler: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)


def simulation_summary(games: Sequence[GameRecord]) -> dict[str, Any]:
    """Win, timeout and length rates over every recorded game."""
    total_games = len(games)
    if total_games == 0:
        return {"total_games": 0}
    hero_wins = sum(g.outcome == GameOutcome.HERO_WIN for g in games)
    villain_wins = sum(g.outcome == GameOutcome.VILLAIN_WIN for g in games)
    steps_to_win = [g.steps for g in games if g.outcome == GameOutcome.HERO_WIN]
    return {
        "total_games": total_games,
        "hero_wins": hero_wins,
        "villain_wins": villain_wins,
        "timeouts": total_games - hero_wins - villain_wins,
        "hero_win_rate": hero_wins / total_games,
        "villain_win_rate": villain_wins / total_games,
        "avg_steps": sum(g.steps for g in games) / total_games,
        "avg_duration": sum(g.duration for g in games) / total_games,
        "avg_steps_to_win": np.mean(steps_to_win) if steps_to_win else 0,
    }


def detailed_analysis(games: Sequence[GameRecord]) -> dict[str, Any]:
    """Game-length distributions, win rate by length and profiler timings."""
    if not games:
        return {}
    turn_counts = [g.steps for g in games]
    hero_win_steps = [g.steps for g in games if g.outcome == GameOutcome.HERO_WIN]
    villain_win_steps = [g.steps for g in games if g.outcome == GameOutcome.VILLAIN_WIN]
    analysis: dict[str, Any] = {
        "turn_distribution": {
            "min": min(turn_counts),
            "max": max(turn_counts),
            "p25": np.percentile(turn_counts, 25),
            "p50": np.percentile(turn_counts, 50),
            "p75": np.percentile(turn_counts, 75),
        },
        "hero_win_turn_distribution": _percentiles(hero_win_steps),
        "villain_win_turn_distribution": _percentiles(villain_win_steps),
        "early_game_win_rate": _win_rate_by_turn_range(games, 0, 10),
        "mid_game_win_rate": _win_rate_by_turn_range(games, 11, 25),
        "late_game_win_rate": _win_rate_by_turn_range(games, 26, float("inf")),
    }
    if _profiler_entries(games):
        analysis["profiler"] = profiler_summary(games)
    return analysis


def _percentiles(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    fvalues = [float(v) for v in values]
    return {
        "count": len(fvalues),
        "min": min(fvalues),
        "max": max(fvalues),
        "p25": np.percentile(fvalues, 25),
        "p50": np.percentile(fvalues, 50),
        "p75": np.percentile(fvalues, 75),
    }


def _win_rate_by_turn_range(
    games: Sequence[GameRecord], min_turn: float, max_turn: float
) -> float:
    """Hero win rate among games that ended within an inclusive step range."""
    in_range = [g for g in games if min_turn <= g.steps <= max_turn]
    if not in_range:
        return 0.0
    return sum(g.outcome == GameOutcome.HERO_WIN for g in in_range) / len(in_range)


def _profiler_entries(
    games: Sequence[GameRecord],
) -> dict[str, list[tuple[float, int]]]:
    """Each timer's (total seconds, count) per game."""
    entries: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for game in games:
        for key, value in game.profiler.items():
            entries[key].append((float(value["total_time"]), int(value["count"])))
    return entries


def profiler_summary(games: Sequence[GameRecord]) -> dict[str, dict[str, float]]:
    """Per-game average total time and call count of each profiled timer."""
    summary: dict[str, dict[str, float]] = {}
    for key, entries in _profiler_entries(games).items():
        total_sum = sum(total for total, _ in entries)
        count_sum = sum(count for _, count in entries)
        summary[key] = {
            "avg_total_time": total_sum / len(entries),
            "avg_call_count": count_sum / len(entries),
            "avg_time_per_call": total_sum / count_sum if count_sum > 0 else 0,
        }
    return summary


def _log_profiler(games: Sequence[GameRecord], logger: logging.Logger) -> None:
    profiled = profiler_summary(games)
    if profiled:
        logger.info("Profiler Summary (Top 10 by total time):")
        top = sorted(
            profiled.items(), key=lambda x: x[1]["avg_total_time"], reverse=True
        )[:10]
        for key, stats in top:
            logger.info(f"  {key}:")
            logger.info(f"    Avg total time: {stats['avg_total_time']:.6f}s")
            logger.info(f"    Avg call count: {stats['avg_call_count']:.1f}")
            logger.info(f"    Avg time per call: {stats['avg_time_per_call']:.8f}s")


# -----------------------------------------------------------------------------
# Outcome Determination
# -----------------------------------------------------------------------------


def determine_outcome(
    info: dict, raw_obs, turn_count: int, max_steps: int
) -> GameOutcome:
    """
    Determine the outcome of a game with robust fallback logic.

    Args:
        info: Information dictionary from environment step
        raw_obs: Raw managym.Observation (not encoded)
        turn_count: Current turn count
        max_steps: Maximum steps before timeout

    Returns:
        Game outcome (HERO_WIN, VILLAIN_WIN, or TIMEOUT)
    """
    # Check for timeout first
    if turn_count >= max_steps:
        return GameOutcome.TIMEOUT

    # Primary source: winner field in info dict
    if "winner" in info:
        return GameOutcome.HERO_WIN if info["winner"] == 0 else GameOutcome.VILLAIN_WIN

    # Fallback: life totals from the raw observation
    hero_life = raw_obs.agent.life
    villain_life = raw_obs.opponent.life

    if hero_life <= 0:
        return GameOutcome.VILLAIN_WIN
    if villain_life <= 0:
        return GameOutcome.HERO_WIN

    return GameOutcome.TIMEOUT


# -----------------------------------------------------------------------------
# Main Simulation
# -----------------------------------------------------------------------------


def _shared_observation_config(*players: Player) -> ObservationConfig | None:
    """The one policy input every model player reads; the default when none does.

    Both seats are encoded with one space, so models bound to different inputs
    cannot meet in one game.
    """
    configs = [
        player.agent.observation_space.config
        for player in players
        if isinstance(player, ModelPlayer)
    ]
    if any(config != configs[0] for config in configs):
        raise ValueError("players have different observation ABIs")
    return configs[0] if configs else None


def simulate_models(
    hero_player: Player,
    villain_player: Player,
    sim_config: SimulationConfig | None = None,
) -> list[GameRecord]:
    """Run games in parallel threads after admitting each model to the match setup.

    Games are returned in completion order.
    """
    logger = getLogger("manabot.sim.sim").getChild("simulate_parallel")

    # Set up hyperparameters
    sim_config = sim_config or SimulationConfig()
    player_configs = Match(sim_config.match).to_rust()
    for player in (hero_player, villain_player):
        if isinstance(player, ModelPlayer):
            validate_agent_setup(player.agent, player_configs)
    observation_config = _shared_observation_config(hero_player, villain_player)

    # Determine number of threads (capped by games and CPU cores)
    num_threads = min(sim_config.num_threads, sim_config.num_games)
    logger.info(f"Starting simulation: {hero_player.name} vs {villain_player.name}")
    logger.info(
        f"Running {sim_config.num_games} games with max {sim_config.max_steps} steps each"
    )
    logger.info(f"Using {num_threads} parallel threads")

    games: list[GameRecord] = []
    games_lock = threading.Lock()
    completed_games = 0
    completed_lock = threading.Lock()

    # Start timing
    start_time = time.time()

    failures: list[BaseException] = []
    stopped = threading.Event()

    # Worker function that runs games until target count is reached
    def worker_thread(thread_id: int) -> None:
        try:
            run_worker(thread_id)
        except BaseException as error:
            with completed_lock:
                failures.append(error)
            stopped.set()

    def run_worker(thread_id: int) -> None:
        # Create environment for this thread
        match = Match(sim_config.match)
        observation_space = ObservationSpace(observation_config)
        reward = Reward(sim_config.reward)
        env = Env(
            match,
            observation_space,
            reward,
            enable_profiler=True,
        )

        nonlocal completed_games
        thread_games = 0

        try:
            while not stopped.is_set():
                # Check if we've completed enough games
                with completed_lock:
                    if completed_games >= sim_config.num_games:
                        break
                    completed_games += 1

                # Simulate a single game
                outcome, steps, duration, profiler_data = _simulate_game(
                    env, hero_player, villain_player, sim_config.max_steps
                )

                record = GameRecord(outcome, steps, duration, thread_id, profiler_data)
                with games_lock:
                    games.append(record)

                # Update thread counter
                thread_games += 1

                # Log progress occasionally
                if thread_games % 5 == 0:
                    logger.info(f"Thread {thread_id}: Completed {thread_games} games")

        finally:
            env.close()

        logger.info(f"Thread {thread_id} completed {thread_games} games")

    # Create and start worker threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(thread_id,))
        threads.append(thread)
        thread.start()

    # Monitor progress while threads are running
    while any(thread.is_alive() for thread in threads):
        with completed_lock:
            current = completed_games

        # Progress update every 5 seconds
        if current < sim_config.num_games:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = sim_config.num_games - current
            eta = remaining / rate if rate > 0 else "unknown"

            if isinstance(eta, float):
                eta_str = f"{eta:.1f} seconds"
            else:
                eta_str = str(eta)

            logger.info(
                f"Progress: {current}/{sim_config.num_games} games completed "
                f"({current / sim_config.num_games:.1%}) | "
                f"Rate: {rate:.2f} games/sec | ETA: {eta_str}"
            )

        # Sleep to avoid busy waiting
        time.sleep(5)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    if failures:
        raise failures[0]

    # End timing
    total_time = time.time() - start_time

    # Log results
    summary = simulation_summary(games)
    logger.info(
        f"Simulation complete: {summary['total_games']} games in {total_time:.2f} seconds"
    )
    logger.info(
        f"Overall performance: {summary['total_games'] / total_time:.2f} games/second"
    )
    logger.info(f"Hero wins: {summary['hero_wins']} ({summary['hero_win_rate']:.2%})")
    logger.info(
        f"Villain wins: {summary['villain_wins']} ({summary['villain_win_rate']:.2%})"
    )
    logger.info(
        f"Timeouts: {summary['timeouts']} ({summary['timeouts'] / summary['total_games']:.2%})"
    )
    logger.info(f"Average steps per game: {summary['avg_steps']:.1f}")

    detailed = detailed_analysis(games)
    logger.info("Detailed Analysis:")
    logger.info(f"Early game win rate: {detailed['early_game_win_rate']:.2f}")
    logger.info(f"Mid game win rate: {detailed['mid_game_win_rate']:.2f}")
    logger.info(f"Late game win rate: {detailed['late_game_win_rate']:.2f}")

    _log_profiler(games, logger)

    return games


def _simulate_game(
    env: Env, hero_player: Player, villain_player: Player, max_steps: int
) -> tuple[GameOutcome, int, float, dict[str, dict[str, float | int]]]:
    """
    Simulate a single game between two players.

    Args:
        env: Game environment
        hero_player: Player for hero position
        villain_player: Player for villain position
        max_steps: Maximum steps before timeout

    Returns:
        Tuple of (outcome, steps, duration, profiler_data)
    """
    logger = getLogger("manabot.sim.sim").getChild("simulate_game")
    # Reset environment
    start_time = time.time()
    obs, info = env.reset()
    done = False
    turn_count = 0

    # Track game state
    last_info = info

    # Main game loop
    while not done and turn_count < max_steps:
        # Get active player index from the raw observation.
        active_player_index = int(env.last_raw_obs.agent.player_index)

        # Select the appropriate player
        player = hero_player if active_player_index == 0 else villain_player

        action = player.get_action(obs)
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last_info = info
        turn_count += 1
        obs = new_obs

    # Calculate duration
    if not done:
        logger.warning("Game did not complete")

    duration = time.time() - start_time

    # Determine outcome
    outcome = determine_outcome(last_info, env.last_raw_obs, turn_count, max_steps)

    profiler_data = last_info.get("profiler", {})
    return outcome, turn_count, duration, profiler_data


def load_player(model_str: str, *, backend: PolicyLoader | None = None) -> Player:
    model_name = model_str.lower()
    if model_name == "random":
        return RandomPlayer("RandomPlayer")
    if model_name == "default":
        return DefaultPlayer("DefaultPlayer")

    if backend is None:
        raise ValueError("a checkpoint player requires an injected backend")
    path = Path(model_str).expanduser()
    if path.suffix == ".json":
        record = CheckpointRecord.model_validate_json(path.read_bytes())
        with tempfile.TemporaryDirectory(prefix="manabot-policy-") as directory:
            policy_path = fetch_etude_checkpoint(
                record.artifact, Path(directory) / "policy.pt"
            )
            agent, _ = backend.load_policy(str(policy_path))
    else:
        agent, _ = backend.load_policy(str(path))
    return ModelPlayer(f"Model_{path.stem}", agent)


# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------


def run_simulation(
    sim_config: SimulationConfig,
    runtime_config: RuntimeConfig,
    *,
    backend: SimulationBackend | None = None,
) -> None:
    """Run a simulation sweep for hero/villain model specs."""
    setGlobalLogLevel(getattr(logging, runtime_config.log_level.upper(), logging.INFO))
    random.seed(runtime_config.seed)
    np.random.seed(runtime_config.seed)
    if backend is not None:
        backend.seed(
            runtime_config.seed,
            deterministic=runtime_config.torch_deterministic,
        )
    previous_seed = os.environ.get("ETUDE_SEED")
    os.environ["ETUDE_SEED"] = str(runtime_config.seed)
    try:
        declared = declare(
            experiment=runtime_config.name,
            clocks=(
                Clock(name="games", kind="step", meaning="completed evaluation games"),
            ),
            config=sim_config.model_dump(mode="json"),
            directory=None
            if os.environ.get("ETUDE_TRACK_DIR")
            else runtime_config.runs_dir / runtime_config.name / "etude",
        )
    finally:
        if previous_seed is None:
            os.environ.pop("ETUDE_SEED", None)
        else:
            os.environ["ETUDE_SEED"] = previous_seed
    with TrainingRun(
        experiment=runtime_config.name,
        name=declared.name,
        root=declared.root,
        clocks=declared.record.clocks,
        destinations=destinations(declared.base, ()),
        strict=True,
        closed_clocks=True,
        inherited=None,
        seed=runtime_config.seed,
    ) as tracked:
        logger = getLogger("manabot.sim.sim")
        logger.info(
            "Simulation config: "
            f"hero={sim_config.hero}, villain={sim_config.villain}, "
            f"num_games={sim_config.num_games}, num_threads={sim_config.num_threads}, "
            f"max_steps={sim_config.max_steps}"
        )

        logger.info(f"Loading hero model: {sim_config.hero}")
        logger.info(f"Loading villain model: {sim_config.villain}")
        hero_player = load_player(sim_config.hero, backend=backend)
        villain_player = load_player(sim_config.villain, backend=backend)

        games = simulate_models(hero_player, villain_player, sim_config)
        for index, game in enumerate(games, start=1):
            tracked.log(
                {
                    "game/outcome": game.outcome.value,
                    "game/steps": game.steps,
                    "game/duration": game.duration,
                },
                games=index,
            )
        tracked.log({"evaluation": simulation_summary(games)}, games=len(games))


def main(argv: Sequence[str] | None = None) -> None:
    from manabot.config.load import load_sim_config

    parser = argparse.ArgumentParser(description="Run manabot model simulation")
    parser.add_argument("--preset", default="sim", help="Simulation preset name")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (repeatable key.path=value)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    sim_config, runtime_config = load_sim_config(
        preset=args.preset, set_overrides=args.set_values
    )
    run_simulation(sim_config, runtime_config, backend=resolve("torch"))


if __name__ == "__main__":
    main()
