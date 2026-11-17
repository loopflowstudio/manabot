"""Scalar game simulation and evaluation statistics recorded through Etude.

load_player admits local policies or exact Etude checkpoint records; simulate_models
admits each model against the actual match setup before play. run_simulation
owns the evaluation tracking lifetime; game statistics never fit a model.
"""

import argparse
from collections import defaultdict
from enum import Enum
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from etude.train.records import CheckpointRecord, Step
from etude.train.tracking import track
import numpy as np

# Local imports
from manabot.backends import resolve
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import fetch_etude_checkpoint
from manabot.infra.config import ObservationConfig, RuntimeConfig, SimulationConfig
from manabot.infra.log import getLogger, setGlobalLogLevel
from manabot.infra.runtime import seed_random
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import Backend
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


class GameStats:
    """Game statistics tracking with enhanced analysis capabilities."""

    def __init__(self):
        self.games = []
        self.hero_wins = 0
        self.villain_wins = 0
        self.timeouts = 0
        self.total_steps = 0
        self.total_duration = 0
        self.lock = threading.Lock()

        # Enhanced tracking
        self.steps_to_win = []  # Track steps taken in winning games
        self.phase_distributions = defaultdict(int)  # Track game phases
        self.game_records = []  # Detailed game records

        # Profiler and behavior tracking
        self.profiler_data = defaultdict(list)  # Store profiler data from each thread

    def record_game(
        self,
        outcome: GameOutcome,
        steps: int,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
        profiler_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a completed game with optional metadata, profiler data.

        Args:
            outcome: Game outcome (hero win, villain win, timeout)
            steps: Number of steps in the game
            duration: Time taken for the game in seconds
            metadata: Optional additional game data for analysis
            profiler_info: Profiler metrics from the environment
        """
        game_record = {"outcome": outcome, "steps": steps, "duration": duration}

        if metadata:
            game_record.update(metadata)

        with self.lock:
            self.games.append(game_record)
            self.game_records.append(game_record)

            if outcome == GameOutcome.HERO_WIN:
                self.hero_wins += 1
                self.steps_to_win.append(steps)
            elif outcome == GameOutcome.VILLAIN_WIN:
                self.villain_wins += 1
            else:
                self.timeouts += 1

            self.total_steps += steps
            self.total_duration += duration

            # Record profiler data if available
            if profiler_info:
                self._record_profiler_data(profiler_info)

    def _record_profiler_data(
        self, profiler_info: dict[str, dict[str, float | int]]
    ) -> None:
        for key, value in profiler_info.items():
            self.profiler_data[key].append(
                {"total": float(value["total_time"]), "count": int(value["count"])}
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_games = len(self.games)
        if total_games == 0:
            return {"total_games": 0}

        return {
            "total_games": total_games,
            "hero_wins": self.hero_wins,
            "villain_wins": self.villain_wins,
            "timeouts": self.timeouts,
            "hero_win_rate": self.hero_wins / total_games,
            "villain_win_rate": self.villain_wins / total_games,
            "avg_steps": self.total_steps / total_games,
            "avg_duration": self.total_duration / total_games,
            "avg_steps_to_win": np.mean(self.steps_to_win) if self.steps_to_win else 0,
        }

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of game patterns."""
        if not self.games:
            return {}

        # Analyze turn distribution
        turn_counts = [g["steps"] for g in self.games]
        hero_win_steps = [
            g["steps"] for g in self.games if g["outcome"] == GameOutcome.HERO_WIN
        ]
        villain_win_steps = [
            g["steps"] for g in self.games if g["outcome"] == GameOutcome.VILLAIN_WIN
        ]

        analysis = {
            "turn_distribution": {
                "min": min(turn_counts),
                "max": max(turn_counts),
                "p25": np.percentile(turn_counts, 25),
                "p50": np.percentile(turn_counts, 50),
                "p75": np.percentile(turn_counts, 75),
            },
            "hero_win_turn_distribution": self._get_percentiles(hero_win_steps),
            "villain_win_turn_distribution": self._get_percentiles(villain_win_steps),
            "early_game_win_rate": self._calculate_win_rate_by_turn_range(0, 10),
            "mid_game_win_rate": self._calculate_win_rate_by_turn_range(11, 25),
            "late_game_win_rate": self._calculate_win_rate_by_turn_range(
                26, float("inf")
            ),
        }

        # Add profiler data
        if self.profiler_data:
            analysis["profiler"] = self.get_profiler_summary()

        return analysis

    def _get_percentiles(self, values: List[int]) -> Dict[str, Any]:
        """Calculate percentiles for a list of values."""
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

    def _calculate_win_rate_by_turn_range(self, min_turn: int, max_turn: int) -> float:
        """Calculate win rate for games that ended within a turn range."""
        games_in_range = [g for g in self.games if min_turn <= g["steps"] <= max_turn]

        if not games_in_range:
            return 0.0

        hero_wins = sum(
            1 for g in games_in_range if g["outcome"] == GameOutcome.HERO_WIN
        )
        return hero_wins / len(games_in_range)

    def get_profiler_summary(self) -> Dict[str, Any]:
        """
        Get summary of profiler data.

        Returns:
            Dictionary with averaged profiler metrics
        """
        summary = {}

        for key, entries in self.profiler_data.items():
            if not entries:
                continue

            # Calculate summed totals and counts
            total_sum = sum(entry["total"] for entry in entries)
            count_sum = sum(entry["count"] for entry in entries)

            # Calculate averages
            avg_time = total_sum / len(entries)
            avg_count = count_sum / len(entries)

            summary[key] = {
                "avg_total_time": avg_time,
                "avg_call_count": avg_count,
                "avg_time_per_call": total_sum / count_sum if count_sum > 0 else 0,
            }

        return summary

    def log_profiler_summary(self, logger) -> None:
        """
        Log profiler summary statistics.

        Args:
            logger: Logger to use for output
        """
        # Log profiler summary
        profiler_summary = self.get_profiler_summary()
        if profiler_summary:
            logger.info("Profiler Summary (Top 10 by total time):")

            # Sort by average total time
            sorted_metrics = sorted(
                profiler_summary.items(),
                key=lambda x: x[1]["avg_total_time"],
                reverse=True,
            )[:10]

            for key, stats in sorted_metrics:
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
        player.agent.observation_space.encoder.config
        for player in players
        if isinstance(player, ModelPlayer)
    ]
    if any(config != configs[0] for config in configs):
        raise ValueError("players have different observation ABIs")
    return configs[0] if configs else None


def simulate_models(
    hero_player: Player,
    villain_player: Player,
    sim_config: Optional[SimulationConfig] = None,
) -> GameStats:
    """Run games in parallel threads after admitting each model to the match setup."""
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

    # Create shared stats object and counter
    stats = GameStats()
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

                # Record game with minimal metadata
                metadata = {
                    "thread_id": thread_id,
                }

                stats.record_game(
                    outcome=outcome,
                    steps=steps,
                    duration=duration,
                    metadata=metadata,
                    profiler_info=profiler_data,
                )

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
    summary = stats.get_summary()
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

    detailed = stats.get_detailed_analysis()
    logger.info("Detailed Analysis:")
    logger.info(f"Early game win rate: {detailed['early_game_win_rate']:.2f}")
    logger.info(f"Mid game win rate: {detailed['mid_game_win_rate']:.2f}")
    logger.info(f"Late game win rate: {detailed['late_game_win_rate']:.2f}")

    # Log profiler summary
    stats.log_profiler_summary(logger)

    return stats


def _simulate_game(
    env: Env, hero_player: Player, villain_player: Player, max_steps: int
) -> Tuple[GameOutcome, int, float, Dict[str, Any]]:
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

    # Extract profiler data from the environment info
    profiler_data = last_info.get("profiler", {})

    return outcome, turn_count, duration, profiler_data


def load_player(model_str: str, *, backend: Backend | None = None) -> Player:
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
    backend: Backend | None = None,
) -> None:
    """Run a simulation sweep for hero/villain model specs."""
    setGlobalLogLevel(getattr(logging, runtime_config.log_level.upper(), logging.INFO))
    seed_random(
        runtime_config.seed,
        backend=backend,
        torch_deterministic=runtime_config.torch_deterministic,
    )
    with track(
        experiment=runtime_config.name,
        clocks=[Step("games", "completed evaluation games")],
        seed=runtime_config.seed,
        config=sim_config.model_dump(mode="json"),
        directory=None
        if os.environ.get("ETUDE_TRACK_DIR")
        else runtime_config.runs_dir / runtime_config.name / "etude",
        strict=True,
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

        stats = simulate_models(hero_player, villain_player, sim_config)
        for index, game in enumerate(stats.games, start=1):
            tracked.log(
                {
                    "game/outcome": game["outcome"].value,
                    "game/steps": game["steps"],
                    "game/duration": game["duration"],
                },
                games=index,
            )
        tracked.log({"evaluation": stats.get_summary()}, games=len(stats.games))


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
