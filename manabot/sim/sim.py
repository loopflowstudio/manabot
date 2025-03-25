"""
evaluate.py
Simplified and consolidated evaluation package for manabot.

This module provides:
1. Loading models from wandb
2. Player abstractions for model inference
3. Basic game simulation and statistics tracking
4. Action distribution and decision analysis
"""

import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import wandb
import threading
import logging
from collections import defaultdict, Counter
import hydra
from omegaconf import DictConfig, OmegaConf

from manabot.env import Env, Match, Reward, ObservationSpace
from manabot.env.observation import get_agent_indices
from manabot.infra.hypers import SimulationHypers, ExperimentHypers
from manabot.infra.log import getLogger
from manabot.infra.experiment import Experiment
from manabot.sim.player import Player, ModelPlayer, RandomPlayer, DefaultPlayer, load_model_from_wandb

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
        self.hero_behavior = defaultdict(list)  # Store hero behavior metrics
        self.villain_behavior = defaultdict(list)  # Store villain behavior metrics
    
    def record_game(
        self, 
        outcome: GameOutcome, 
        steps: int, 
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
        profiler_info: Optional[Dict[str, Any]] = None,
        behavior_info: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Record a completed game with optional metadata, profiler and behavior data.
        
        Args:
            outcome: Game outcome (hero win, villain win, timeout)
            steps: Number of steps in the game
            duration: Time taken for the game in seconds
            metadata: Optional additional game data for analysis
            profiler_info: Profiler metrics from the environment
            behavior_info: Behavior tracking metrics for hero and villain
        """
        game_record = {
            "outcome": outcome,
            "steps": steps,
            "duration": duration
        }
        
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
            
            # Record profiler and behavior data if available
            if profiler_info:
                self._record_profiler_data(profiler_info)
                
            if behavior_info:
                if "hero" in behavior_info:
                    self._record_behavior_data(behavior_info["hero"], self.hero_behavior)
                if "villain" in behavior_info:
                    self._record_behavior_data(behavior_info["villain"], self.villain_behavior)
    
    def _record_profiler_data(self, profiler_info: Dict[str, str]) -> None:
        """
        Record profiler data from the environment.
        
        Args:
            profiler_info: Dictionary of profiler metrics
        """
        for key, value in profiler_info.items():
            # Extract total time and count from the string format
            if 'total=' in value and 'count=' in value:
                parts = value.split(', ')
                total_part = parts[0].replace('total=', '').replace('s', '')
                count_part = parts[1].replace('count=', '')
                
                try:
                    total = float(total_part)
                    count = int(count_part)
                    
                    # Store both total and count for proper averaging later
                    self.profiler_data[key].append({
                        "total": total,
                        "count": count
                    })
                except (ValueError, TypeError):
                    # Skip if parsing fails
                    continue
    
    def _record_behavior_data(self, behavior_info: Dict[str, str], target_dict: Dict[str, list]) -> None:
        """
        Record behavior tracking data.
        
        Args:
            behavior_info: Dictionary of behavior metrics
            target_dict: Target dictionary to store the data
        """
        for key, value in behavior_info.items():
            try:
                # Try to convert to float for numerical metrics
                val = float(value)
                target_dict[key].append(val)
            except (ValueError, TypeError):
                # Keep as string if not convertible
                target_dict[key].append(value)
    
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
        hero_win_steps = [g["steps"] for g in self.games 
                          if g["outcome"] == GameOutcome.HERO_WIN]
        villain_win_steps = [g["steps"] for g in self.games 
                             if g["outcome"] == GameOutcome.VILLAIN_WIN]
        
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
            "late_game_win_rate": self._calculate_win_rate_by_turn_range(26, float('inf')),
        }
        
        # Add profiler and behavior data
        if self.profiler_data:
            analysis["profiler"] = self.get_profiler_summary()
            
        if self.hero_behavior or self.villain_behavior:
            analysis["behavior"] = self.get_behavior_summary()
        
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
        games_in_range = [g for g in self.games 
                         if min_turn <= g["steps"] <= max_turn]
        
        if not games_in_range:
            return 0.0
            
        hero_wins = sum(1 for g in games_in_range 
                        if g["outcome"] == GameOutcome.HERO_WIN)
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
                "avg_time_per_call": total_sum / count_sum if count_sum > 0 else 0
            }
            
        return summary
        
    def get_behavior_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of behavior data.
        
        Returns:
            Dictionary with averaged behavior metrics for hero and villain
        """
        hero_summary = {}
        villain_summary = {}
        
        # Process hero behavior
        for key, values in self.hero_behavior.items():
            if not values:
                continue
                
            # Calculate average
            if all(isinstance(v, (int, float)) for v in values):
                hero_summary[key] = sum(values) / len(values)
            else:
                # For non-numeric values, just take the most common
                hero_summary[key] = Counter(values).most_common(1)[0][0]
                
        # Process villain behavior
        for key, values in self.villain_behavior.items():
            if not values:
                continue
                
            # Calculate average
            if all(isinstance(v, (int, float)) for v in values):
                villain_summary[key] = sum(values) / len(values)
            else:
                # For non-numeric values, just take the most common
                villain_summary[key] = Counter(values).most_common(1)[0][0]
                
        return {
            "hero": hero_summary,
            "villain": villain_summary
        }
    
    def log_profiler_and_behavior_summary(self, logger) -> None:
        """
        Log profiler and behavior summary statistics.
        
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
                reverse=True
            )[:10]
            
            for key, stats in sorted_metrics:
                logger.info(f"  {key}:")
                logger.info(f"    Avg total time: {stats['avg_total_time']:.6f}s")
                logger.info(f"    Avg call count: {stats['avg_call_count']:.1f}")
                logger.info(f"    Avg time per call: {stats['avg_time_per_call']:.8f}s")
        
        # Log behavior summary
        behavior_summary = self.get_behavior_summary()
        if behavior_summary:
            # Hero behavior
            if behavior_summary["hero"]:
                logger.info("Hero Behavior Metrics:")
                for key, value in sorted(behavior_summary["hero"].items()):
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.2f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Villain behavior
            if behavior_summary["villain"]:
                logger.info("Villain Behavior Metrics:")
                for key, value in sorted(behavior_summary["villain"].items()):
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.2f}")
                    else:
                        logger.info(f"  {key}: {value}")
    
    def log_to_wandb(self, run) -> None:
        """Log summary statistics and visualizations to wandb."""
        summary = self.get_summary()
        
        # Basic metrics
        metrics = {
            "eval/hero_win_rate": summary["hero_win_rate"],
            "eval/villain_win_rate": summary["villain_win_rate"],
            "eval/timeout_rate": summary["timeouts"] / summary["total_games"],
            "eval/avg_steps": summary["avg_steps"],
            "eval/avg_duration": summary["avg_duration"],
        }
        
        if self.steps_to_win:
            metrics["eval/avg_steps_to_win"] = summary["avg_steps_to_win"]
        
        run.log(metrics)
        
        # Game records table
        games_table = wandb.Table(
            columns=["outcome", "steps", "duration"],
            data=[[g["outcome"].value, g["steps"], g["duration"]] for g in self.games]
        )
        run.log({"eval/games": games_table})
        
        # Turn distribution histogram
        if self.games:
            turn_counts = [g["steps"] for g in self.games]
            turn_histogram = np.histogram(turn_counts, bins=20)
            run.log({
                "eval/turn_distribution": wandb.Histogram(
                    np_histogram=turn_histogram
                )
            })
            
            # Win rates by turn
            detailed = self.get_detailed_analysis()
            run.log({
                "eval/early_game_win_rate": detailed["early_game_win_rate"],
                "eval/mid_game_win_rate": detailed["mid_game_win_rate"],
                "eval/late_game_win_rate": detailed["late_game_win_rate"],
            })
            
            # Log behavior metrics
            behavior_summary = self.get_behavior_summary()
            
            # Hero behavior metrics
            for key, value in behavior_summary["hero"].items():
                if isinstance(value, (int, float)):
                    run.log({f"eval/hero_{key}": value})
                    
            # Villain behavior metrics
            for key, value in behavior_summary["villain"].items():
                if isinstance(value, (int, float)):
                    run.log({f"eval/villain_{key}": value})

# -----------------------------------------------------------------------------
# Outcome Determination
# -----------------------------------------------------------------------------

def determine_outcome(
    info: dict, 
    last_obs: dict, 
    turn_count: int, 
    max_steps: int
) -> GameOutcome:
    """
    Determine the outcome of a game with robust fallback logic.
    
    Args:
        info: Information dictionary from environment step
        last_obs: Last observation from environment
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
        
    # Fallback: game_over and won fields in observation
    if last_obs.get("game_over", False):
        return GameOutcome.HERO_WIN if last_obs.get("won", -1) == 0 else GameOutcome.VILLAIN_WIN
        
    # Fallback: Check life totals if available
    hero_life = _extract_life(last_obs, player_index=0)
    villain_life = _extract_life(last_obs, player_index=1)
    
    if hero_life is not None and villain_life is not None:
        if hero_life <= 0:
            return GameOutcome.VILLAIN_WIN
        if villain_life <= 0:
            return GameOutcome.HERO_WIN
    
    # If we can't determine a winner, consider it a timeout
    return GameOutcome.TIMEOUT

def _extract_life(obs: dict, player_index: int) -> Optional[float]:
    """Extract life total for a player from observation if possible."""
    try:
        # This is a simplified example - adapt based on your actual observation structure
        if player_index == 0 and "agent_player" in obs:
            return obs["agent_player"][0, 2]  # Assuming life is at index 2
        elif player_index == 1 and "opponent_player" in obs:
            return obs["opponent_player"][0, 2]  # Assuming life is at index 2
    except (IndexError, KeyError):
        pass
    return None

# -----------------------------------------------------------------------------
# Main Simulation
# -----------------------------------------------------------------------------

def simulate_models(
    hero_player: Player,
    villain_player: Player,
    sim_hypers: Optional[SimulationHypers] = None,
) -> GameStats:
    """
    Evaluate models in parallel by running multiple games simultaneously.
    
    Args:
        hero_player: Player for hero position
        villain_player: Player for villain position
        eval_hypers: Evaluation hyperparameters
        
    Returns:
        GameStats with results
    """
    logger = getLogger("manabot.sim.sim").getChild("simulate_parallel")
    
    # Set up hyperparameters
    sim_hypers = sim_hypers or SimulationHypers()
    
    # Determine number of threads (capped by games and CPU cores)
    num_threads = min(sim_hypers.num_threads, sim_hypers.num_games)
    logger.info(f"Starting simulation: {hero_player.name} vs {villain_player.name}")
    logger.info(f"Running {sim_hypers.num_games} games with max {sim_hypers.max_steps} steps each")
    logger.info(f"Using {num_threads} parallel threads")
    
    # Create shared stats object and counter
    stats = GameStats()
    completed_games = 0
    completed_lock = threading.Lock()
    
    # Start timing
    start_time = time.time()
    
    # Worker function that runs games until target count is reached
    def worker_thread(thread_id):
        # Create environment for this thread
        match = Match(sim_hypers.match)
        observation_space = ObservationSpace()
        reward = Reward(sim_hypers.reward)
        env = Env(match, observation_space, reward, auto_reset=False, enable_profiler=True, enable_behavior_tracking=True)
        
        nonlocal completed_games
        thread_games = 0
        
        while True:
            # Check if we've completed enough games
            with completed_lock:
                if completed_games >= sim_hypers.num_games:
                    break
                # Claim this game
                game_id = completed_games
                completed_games += 1
            
            # Simulate a single game
            outcome, steps, duration, profiler_data, behavior_data = _simulate_game(
                env, hero_player, villain_player, sim_hypers.max_steps)
            
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
                behavior_info=behavior_data
            )
            
            # Update thread counter
            thread_games += 1
            
            # Log progress occasionally
            if thread_games % 5 == 0:
                logger.info(f"Thread {thread_id}: Completed {thread_games} games")
        
        # Clean up
        info = env.info()
        logger.info(f"Thread {thread_id} info: {info}") 
        env.close()
        logger.info(f"Thread {thread_id} completed {thread_games} games")
    
    # Create and start worker threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=worker_thread,
            args=(thread_id,)
        )
        threads.append(thread)
        thread.start()
    
    # Monitor progress while threads are running
    while any(thread.is_alive() for thread in threads):
        with completed_lock:
            current = completed_games
        
        # Progress update every 5 seconds
        if current < sim_hypers.num_games:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = sim_hypers.num_games - current
            eta = remaining / rate if rate > 0 else "unknown"
            
            if isinstance(eta, float):
                eta_str = f"{eta:.1f} seconds"
            else:
                eta_str = str(eta)
                
            logger.info(f"Progress: {current}/{sim_hypers.num_games} games completed "
                       f"({current / sim_hypers.num_games:.1%}) | "
                       f"Rate: {rate:.2f} games/sec | ETA: {eta_str}")
        
        # Sleep to avoid busy waiting
        time.sleep(5)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # End timing
    total_time = time.time() - start_time
    
    # Log results
    summary = stats.get_summary()
    logger.info(f"Simulation complete: {summary['total_games']} games in {total_time:.2f} seconds")
    logger.info(f"Overall performance: {summary['total_games'] / total_time:.2f} games/second")
    logger.info(f"Hero wins: {summary['hero_wins']} ({summary['hero_win_rate']:.2%})")
    logger.info(f"Villain wins: {summary['villain_wins']} ({summary['villain_win_rate']:.2%})")
    logger.info(f"Timeouts: {summary['timeouts']} ({summary['timeouts'] / summary['total_games']:.2%})")
    logger.info(f"Average steps per game: {summary['avg_steps']:.1f}")
    
    detailed = stats.get_detailed_analysis()
    logger.info("Detailed Analysis:")
    logger.info(f"Early game win rate: {detailed['early_game_win_rate']:.2f}")
    logger.info(f"Mid game win rate: {detailed['mid_game_win_rate']:.2f}")
    logger.info(f"Late game win rate: {detailed['late_game_win_rate']:.2f}")
    
    # Log profiler and behavior summary
    stats.log_profiler_and_behavior_summary(logger)
    
    return stats

def _simulate_game(
    env: Env,
    hero_player: Player,
    villain_player: Player,
    max_steps: int
) -> Tuple[GameOutcome, int, float, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Simulate a single game between two players.
    
    Args:
        env: Game environment
        hero_player: Player for hero position
        villain_player: Player for villain position
        max_steps: Maximum steps before timeout
        
    Returns:
        Tuple of (outcome, steps, duration, profiler_data, behavior_data)
    """
    logger = getLogger("manabot.sim.sim").getChild("simulate_game")
    # Reset environment
    start_time = time.time()
    obs, info = env.reset()
    done = False
    turn_count = 0
    
    # Track game state
    last_obs = obs
    last_info = info
    
    # Main game loop
    while not done and turn_count < max_steps:
        # Get active player index from observation
        active_player_index = get_agent_indices(
            {k: torch.tensor(v)[None, ...] for k, v in obs.items()}
        )[0].item()
        
        # Select the appropriate player
        player = hero_player if active_player_index == 0 else villain_player
        
        # Get action from player
        try:
            action = player.get_action(obs)
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            # Fallback to random action if model fails
            valid_actions = np.where(obs["actions_valid"] > 0)[0]
            if len(valid_actions) == 0:
                logger.error("No valid actions available")
                break
            action = int(np.random.choice(valid_actions))
        
        # Step environment
        try:
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_obs = new_obs
            last_info = info
            turn_count += 1
            
            # Update observation for next step
            obs = new_obs
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            break
    
    # Calculate duration
    if not done:
        logger.warning("Game did not complete")

    duration = time.time() - start_time
    
    # Determine outcome
    outcome = determine_outcome(last_info, last_obs, turn_count, max_steps)
    
    # Extract profiler and behavior data from the environment info
    profiler_data = last_info.get("profiler", {})
    behavior_data = last_info.get("behavior", {})
    
    return outcome, turn_count, duration, profiler_data, behavior_data

def load_player(model_str: str) -> Player:
    if model_str.lower() == "random":
        return RandomPlayer("RandomPlayer")
    elif model_str.lower() == "default":
        return DefaultPlayer("DefaultPlayer")
    else:
        if ":" in model_str:
            model, version = model_str.split(":")
        else:
            model = model_str
            version = "latest"
        return ModelPlayer(f"Model_{model_str}", load_model_from_wandb(model, version, device="cpu"))

# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf/sim", config_name="sim")
def main(cfg: DictConfig) -> None:
    """
    Hydra-powered simulation entry point.
    Expects a configuration file (e.g. conf/sim/sim.yaml) with fields:
      - hero: str
      - villain: str
      - num_games: int
      - num_threads: int
      - max_steps: int
      - match: (dictionary matching MatchHypers)
    """
    # Convert the config to a dictionary.
    sim_hypers = OmegaConf.to_object(cfg.sim)
    experiment_hypers = OmegaConf.to_object(cfg.experiment)
    assert isinstance(sim_hypers, SimulationHypers)
    assert isinstance(experiment_hypers, ExperimentHypers)

    # Initialize the logging etc
    _ = Experiment(experiment_hypers)

    logger = getLogger("manabot.sim.sim")
    logger.info("Simulation configuration:\n" + OmegaConf.to_yaml(cfg))

    logger.info(f"Loading hero model: {sim_hypers.hero}")
    logger.info(f"Loading villain model: {sim_hypers.villain}")
    hero_player = load_player(sim_hypers.hero)
    villain_player = load_player(sim_hypers.villain)
    
    simulate_models(hero_player, villain_player, sim_hypers)

if __name__ == "__main__":
    main()
