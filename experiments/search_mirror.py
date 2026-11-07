"""Magnetic-mirror local update on one oracle world, over MTG-183's endpoint.

Each searched decision tries every legal action once in each world holding
the opponent's true hand (libraries reshuffled, once per world), plays `depth` decisions by
policy sampling, scores the position with the network, and samples the move from

    p ∝ exp((Q + alpha log uniform + beta log policy) / (alpha + beta))

(`manabot.sim.search.local_update.regularized_update`). Each Q averages one
rollout per world. The move is sampled, so the matching no-search baseline is the
sampled policy. These games reuse search_scaling's first five deals per grid
cell, so scores pair with every condition of that sweep.

    uv run python -m experiments.search_mirror play --checkpoint PT --out DIR \
        --worker 0 --workers 7
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

from experiments.elemental_grid import PACK, Game, schedule
from experiments.search_scaling import (
    CHECKPOINT_SHA256,
    MAX_STEPS,
    Condition,
    Opponent,
    _done,
    _opponent,
)
from manabot.sim.search.sampling import (
    HandBatch,
    HandSample,
    PhysicalHandSampler,
    PreparedHands,
)

DEALS_PER_CELL = 5
# Depth runs at a step size that moves the policy; beta runs at one depth. At the
# default beta 1 the target stays within about 0.005 total variation of the policy.
# Four worlds keep the true hand and draw fresh shuffles and rollouts, so each Q
# averages four rollouts. Settings are (depth, beta, worlds).
SETTINGS = (
    (2, 0.1, 1),
    (10, 0.1, 1),
    (20, 0.1, 1),
    (40, 0.1, 1),
    (20, 1.0, 1),
    (20, 0.03, 1),
    (20, 0.1, 4),
    (20, 0.03, 4),
)
# Deep rollouts must finish; local update raises rather than truncating a search.
DECISION_SECONDS = 600.0


@dataclass(frozen=True)
class Arm:
    opponent: Opponent
    depth: int
    beta: float
    worlds: int = 1

    @property
    def key(self) -> str:
        suffix = f"-w{self.worlds}" if self.worlds > 1 else ""
        return f"{self.opponent}/mirror-d{self.depth}-b{self.beta:g}{suffix}"


ARMS = tuple(
    Arm(opponent, depth, beta, worlds)
    for depth, beta, worlds in SETTINGS
    for opponent in ("policy", "heuristic")
)


class OracleHandSampler(PhysicalHandSampler):
    """Every sampled world holds the opponent's true hand at the current root."""

    identity = "oracle-hand/v1"

    def __init__(self) -> None:
        self._root: Any = None
        self._viewer = -1

    def prepare(self, engine: Any, history: Any) -> PreparedHands:
        self._root, self._viewer = engine, history.viewer
        return super().prepare(engine, history)

    def sample(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        del check
        authority = self._root.observation_for_player(1 - self._viewer)
        hand = Counter(
            str(card.name) for card in authority.agent_cards if int(card.zone) == 1
        )
        # A point mass: its log probability is zero.
        sample = HandSample(tuple(sorted(hand.items())), 0.0)
        return HandBatch(prepared, seed, (sample,) * count)


@dataclass(frozen=True)
class MirrorRow:
    key: str
    condition: str
    opponent: str
    depth: int
    beta: float
    model_deck: str
    opponent_deck: str
    model_seat: int
    deal: int
    score: float | None
    failure: str | None
    turns: int
    decisions: int
    hero_decisions: int
    hero_seconds: float
    host: str


@dataclass(frozen=True)
class MirrorDecision:
    """One search: `q` is signed, None where an action had no rollout."""

    key: str
    index: int
    turn: int
    phase: int
    step: int
    kind: int
    chosen: int
    base: list[float]
    q: list[float | None]
    target: list[float]
    terminal_rollouts: int
    seconds: float


def _teacher(arm: Arm, checkpoint: Path) -> Any:
    from manabot.sim.search.local_update import LocalSearchConfig, LocalUpdateTeacher

    teacher = LocalUpdateTeacher(
        checkpoint,
        CHECKPOINT_SHA256,
        LocalSearchConfig(
            worlds=arm.worlds,
            depth=arm.depth,
            beta=arm.beta,
            decision_seconds=DECISION_SECONDS,
            sampling="compatible_prior",
        ),
    )
    # The constructor admits only declared distributions; this one is deliberately
    # outside them, so it replaces the sampler under its own identity.
    oracle = OracleHandSampler()
    teacher._sampler_authority = oracle
    teacher.hand_sampler = oracle
    return teacher


def play_game(
    arm: Arm, game: Game, teacher: Any, checkpoint: Path, host: str
) -> tuple[MirrorRow, list[MirrorDecision]]:
    import torch

    from managym.decision import Command, DecisionFrame
    from manabot.env import Env, Match, Reward
    from manabot.infra.config import MatchConfig, RewardConfig
    from manabot.sim.search.local_update import LocalUpdatePlayer

    torch.manual_seed(game.deal)
    key = f"{arm.key}/{game.deal}"
    hero = LocalUpdatePlayer(teacher, seed=game.deal)
    opponent = _opponent(
        Condition(arm.opponent, "policy"), checkpoint, seed=game.deal + 1
    )
    decks = [game.heuristic_deck, game.heuristic_deck]
    decks[game.model_seat] = game.model_deck
    setup = MatchConfig.authored(
        PACK, decks[0], decks[1], hero="seat-0", villain="seat-1"
    )
    env = Env(
        Match(setup.model_copy(update={"content_pack": PACK})),
        teacher.space,
        Reward(RewardConfig()),
        seed=game.deal,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    log: list[MirrorDecision] = []
    failure: str | None = None
    terminated = truncated = False
    steps = turns = 0
    hero_seconds = 0.0
    try:
        obs, _ = env.reset(seed=game.deal)
        hero.start_game(env, game.model_seat)
        start = getattr(opponent, "start_game", None)
        if start is not None:
            start(env, 1 - game.model_seat)
        while steps < MAX_STEPS:
            acting = int(env.last_raw_obs.agent.player_index)
            if acting == game.model_seat:
                started = time.perf_counter()
                action = int(hero.act(env, obs))
                elapsed = time.perf_counter() - started
                hero_seconds += elapsed
                receipt, turn = hero.last_receipt, env.last_raw_obs.turn
                log.append(
                    MirrorDecision(
                        key=key,
                        index=len(log),
                        turn=int(turn.turn_number),
                        phase=int(turn.phase),
                        step=int(turn.step),
                        kind=int(env.last_raw_obs.action_space.action_space_type),
                        chosen=action,
                        base=[round(float(p), 5) for p in receipt.base],
                        q=[None if v is None else round(v, 4) for v in receipt.values],
                        target=[round(float(p), 5) for p in receipt.target],
                        terminal_rollouts=sum(r.terminal for r in receipt.rollouts),
                        seconds=round(elapsed, 5),
                    )
                )
            else:
                action = int(opponent.act(env, obs))
            hero.prepare_step(env, acting, action)
            frame = DecisionFrame.from_json(env._engine.semantic_decision_frame_json())
            command = Command(
                f"mirror-{steps}", frame.revision, int(frame.offers[action]["id"])
            )
            obs, _, terminated, truncated, _, transition = env.step_semantic(command)
            steps += 1
            if terminated or truncated:
                break
            hero.observe_step(env, acting, transition)
        turns = int(env.last_raw_obs.turn.turn_number)
    except Exception:  # retained as an invalid row, never replayed or replaced
        failure = traceback.format_exc(limit=8)
    winner = env._engine.winner_index() if terminated and failure is None else None
    if failure is None and not terminated:
        failure = "truncated" if truncated else "step bound"
    score = None
    if failure is None:
        score = 0.5 if winner is None else float(int(winner) == game.model_seat)
    return (
        MirrorRow(
            key=key,
            condition=arm.key,
            opponent=arm.opponent,
            depth=arm.depth,
            beta=arm.beta,
            model_deck=game.model_deck,
            opponent_deck=game.heuristic_deck,
            model_seat=game.model_seat,
            deal=game.deal,
            score=score,
            failure=failure,
            turns=turns,
            decisions=steps,
            hero_decisions=len(log),
            hero_seconds=round(hero_seconds, 4),
            host=host,
        ),
        log,
    )


def play(
    checkpoint: Path,
    out: Path,
    worker: int,
    workers: int,
    limit: int | None,
    arms: tuple[Arm, ...] = ARMS,
) -> None:
    """Play this worker's share, deals leading so a stopped run stays balanced."""
    import torch

    from manabot.infra.identity import file_sha256

    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint bytes differ from MTG-183's final export")
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    host = os.uname().nodename
    done = _done(out)
    games = sorted(
        (g for g in schedule() if g.index < DEALS_PER_CELL),
        key=lambda g: (g.index, g.deal),
    )
    units = [(game, arm) for game in games for arm in arms]
    teachers: dict[tuple[int, float, int], Any] = {}
    played = 0
    for index, (game, arm) in enumerate(units):
        if index % workers != worker or f"{arm.key}/{game.deal}" in done:
            continue
        if limit is not None and played >= limit:
            return
        setting = (arm.depth, arm.beta, arm.worlds)
        teacher = teachers.get(setting)
        if teacher is None:
            teacher = teachers[setting] = _teacher(arm, checkpoint)
        row, log = play_game(arm, game, teacher, checkpoint, host)
        with gzip.open(out / f"decisions-{host}-{worker}.jsonl.gz", "at") as file:
            for entry in log:
                file.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        with (out / f"games-{host}-{worker}.jsonl").open("a") as file:
            file.write(json.dumps(asdict(row)) + "\n")
            file.flush()
            os.fsync(file.fileno())
        played += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("play")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--worker", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    play(args.checkpoint, args.out, args.worker, args.workers, args.limit)


if __name__ == "__main__":
    main()
