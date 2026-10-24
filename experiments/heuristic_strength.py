"""Heuristic strength against a checkpoint, and what search over it buys.

Jack Heart's question (2026-10-24): how far can the `heuristic` anchor improve
as UR Lessons against a checkpoint strong against it, without slower decisions;
then, what does search on top of it cost per unit of strength. Results are in
experiments/heuristic-strength.md.

Each deal plays twice, once with each deck on the play. Deal and checkpoint
sampling seeds are fixed, so revisions and search budgets meet identical games.
`HeuristicSearchPlayer` is flat determinized Monte Carlo with the heuristic as
both seats' rollout policy; `--search-worlds 0` is the plain heuristic.

    uv run python -m experiments.heuristic_strength --checkpoint CKPT \\
        --deals 25 --search-worlds 4 --out .runs/heuristic-strength/w4.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from managym.enums import ActionEnum, ActionSpaceEnum
from pydantic import BaseModel, ConfigDict
import torch

from manabot.arena.heuristic import HeuristicPlayer
from manabot.env import Env, Match, Reward
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.sim.search.flat_mc import make_player

PACK = "elemental-world-v2"
FIRST_SEED = 2_026_102_400
MAX_DECISIONS = 5_000

# Decisions the search revisits; the rest are the heuristic's alone.
_SEARCHED_SPACES = {
    ActionSpaceEnum.DECLARE_ATTACKER,
    ActionSpaceEnum.DECLARE_BLOCKER,
    ActionSpaceEnum.CHOOSE_TARGET,
}
_PRIORITY_CHOICES = {
    ActionEnum.PRIORITY_CAST_SPELL,
    ActionEnum.PRIORITY_ACTIVATE_ABILITY,
}


class HeuristicSearchPlayer(HeuristicPlayer):
    """Flat Monte Carlo over every offered action, the heuristic rolling out.

    At a decision with a real choice (a cast or ability on offer, an attack, a
    block, a target), each offered action is played in `worlds` copies of the
    game whose hidden cards are resampled from this seat's view, and the
    heuristic plays both seats to the end (a capped or drawn rollout scores
    half). The best mean result wins; ties keep the heuristic's own choice.
    """

    def __init__(self, worlds: int, *, seed: int, max_steps: int = 600) -> None:
        super().__init__()
        self.worlds = worlds
        self.max_steps = max_steps
        self._seed = seed
        self._calls = 0

    def act(self, env: Env, obs: dict[str, Any]) -> int:
        own = super().act(env, obs)
        raw = env.last_raw_obs
        kind = int(raw.action_space.action_space_type)
        actions = raw.action_space.actions
        searched = kind in _SEARCHED_SPACES or (
            kind == ActionSpaceEnum.PRIORITY
            and any(int(a.action_type) in _PRIORITY_CHOICES for a in actions)
        )
        if self.worlds == 0 or not searched:
            return own
        self._calls += 1
        seat = int(raw.agent.player_index)
        best, best_value = own, -1.0
        for index in [own, *(i for i in range(len(actions)) if i != own)]:
            value = 0.0
            for world in range(self.worlds):
                seed = (self._seed * 1_000_003 + self._calls * 9_973 + world) % 2**63
                value += self._rollout(env._engine, seat, index, seed)
            if value > best_value:
                best, best_value = index, value
        return best

    def _rollout(self, root: Any, seat: int, action: int, seed: int) -> float:
        game = root.clone_env()
        game.determinize(seed, seat)
        players = (HeuristicPlayer(), HeuristicPlayer())
        observation, _, terminated, truncated, _ = game.step(action)
        for _ in range(self.max_steps):
            if terminated or truncated:
                break
            actor = int(observation.agent.player_index)
            # The heuristic reads only `last_raw_obs` from its environment.
            view = SimpleNamespace(last_raw_obs=observation)
            choice = players[actor].act(view, {})  # type: ignore[arg-type]
            observation, _, terminated, truncated, _ = game.step(choice)
        winner = game.winner_index() if terminated else None
        if winner is None:
            return 0.5
        return 1.0 if int(winner) == seat else 0.0


class GameResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    deal: int
    heuristic_on_play: bool
    # None when the game hit the decision cap or drew.
    heuristic_won: bool | None
    turns: int
    heuristic_decisions: int
    heuristic_seconds: float


def play(
    checkpoint: Path,
    deal: int,
    heuristic_on_play: bool,
    heuristic_deck: str,
    agent_deck: str,
    worlds: int,
) -> GameResult:
    """One game: the checkpoint samples its policy, the heuristic decides."""
    torch.set_num_threads(1)
    seed = FIRST_SEED + deal
    agent, space = make_player(
        {"kind": "checkpoint", "path": str(checkpoint)}, seed=seed
    )
    if space is None:
        raise ValueError("checkpoint player has no observation space")
    heuristic = HeuristicSearchPlayer(worlds, seed=seed)
    heuristic_seat = 0 if heuristic_on_play else 1
    decks = (heuristic_deck, agent_deck)
    match = MatchConfig.authored(
        PACK,
        *(decks if heuristic_on_play else decks[::-1]),
        hero="seat-0",
        villain="seat-1",
    )
    env = Env(Match(match), space, Reward(RewardConfig()), seed=seed, auto_reset=False)
    obs, _ = env.reset(seed=seed)
    start_game = getattr(agent, "start_game", None)
    if start_game is not None:
        start_game(env, 1 - heuristic_seat)
    decisions, seconds = 0, 0.0
    winner: int | None = None
    for ordinal in range(MAX_DECISIONS):
        if int(env.last_raw_obs.agent.player_index) == heuristic_seat:
            started = time.perf_counter()
            action = heuristic.act(env, obs)
            seconds += time.perf_counter() - started
            decisions += 1
        else:
            torch.manual_seed(seed * 7_919 + ordinal)
            action = agent.act(env, obs)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            winner = env._engine.winner_index() if terminated else None
            break
    turns = int(env.last_raw_obs.turn.turn_number)
    env.close()
    return GameResult(
        deal=deal,
        heuristic_on_play=heuristic_on_play,
        heuristic_won=None if winner is None else int(winner) == heuristic_seat,
        turns=turns,
        heuristic_decisions=decisions,
        heuristic_seconds=seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deals", type=int, default=25)
    parser.add_argument("--first-deal", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--heuristic-deck", default="elemental_ur")
    parser.add_argument("--agent-deck", default="elemental_gw")
    parser.add_argument("--search-worlds", type=int, default=0)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    deals = range(args.first_deal, args.first_deal + args.deals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("w") as handle:
        futures = [
            pool.submit(
                play,
                checkpoint,
                deal,
                on_play,
                args.heuristic_deck,
                args.agent_deck,
                args.search_worlds,
            )
            for deal in deals
            for on_play in (True, False)
        ]
        results = [future.result() for future in futures]
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    wins = sum(bool(r.heuristic_won) for r in results)
    decisions = sum(r.heuristic_decisions for r in results)
    seconds = sum(r.heuristic_seconds for r in results)
    print(
        f"heuristic wins {wins}/{len(results)} = {wins / len(results):.3f}; "
        f"{1e6 * seconds / decisions:.0f} us per heuristic decision, "
        f"{seconds / len(results):.2f} s per game; "
        f"{time.perf_counter() - started:.0f} s wall"
    )


if __name__ == "__main__":
    main()
