"""How many rollouts does a mirror Q need? Measured on fixed positions.

The hero plays the sampled policy, so positions come from the no-search
baseline's own games. At a random subset of its decisions with at least two
policy-supported moves, it runs a depth-20 local update with ROLLOUTS oracle
worlds (true opponent hand, fresh shuffle and rollout per world) and keeps every
rollout's signed value per move. It then plays the policy's sampled move, so the
measurement never steers the game.

`analyze` subsamples n of those rollouts per move and compares the resulting
target with the one from the held-out rest, for several step sizes. The
estimate costs nothing extra in games: each position is one 64-world search.

    uv run python -m experiments.search_mirror_noise play --checkpoint PT \
        --out DIR --worker 0 --workers 1 --positions 200
    uv run python -m experiments.search_mirror_noise analyze DIR
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gzip
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from experiments.elemental_grid import PACK, Game, schedule
from experiments.search_mirror import DEALS_PER_CELL, Arm, _teacher
from experiments.search_scaling import (
    CHECKPOINT_SHA256,
    MAX_STEPS,
    Condition,
    Opponent,
    _opponent,
)

ROLLOUTS = 64
DEPTH = 20
# About one searched position per ten eligible decisions spreads positions over
# many games and turns instead of clustering them in a few.
SEARCH_PROBABILITY = 0.1
SUBSAMPLES = (1, 2, 4, 8, 16, 32)
BETAS = (0.01, 0.03, 0.1, 0.3, 1.0)
ALPHA = 0.02


@dataclass(frozen=True)
class Position:
    """`values[a]` lists every rollout's signed value for move a; None off support."""

    key: str
    opponent: str
    deal: int
    turn: int
    kind: int
    base: list[float]
    values: list[list[float] | None]
    terminal: list[int]
    seconds: float


def _positions_in_game(
    opponent_kind: Opponent, game: Game, teacher: Any, checkpoint: Path, budget: int
) -> list[Position]:
    from managym.decision import Command, DecisionFrame
    import torch

    from manabot.env import Env, Match, Reward
    from manabot.infra.config import MatchConfig, RewardConfig
    from manabot.sim.search.local_update import LocalUpdatePlayer

    torch.manual_seed(game.deal)
    rng = np.random.default_rng(game.deal ^ 0x5EED)
    hero = LocalUpdatePlayer(teacher, seed=game.deal)
    opponent = _opponent(
        Condition(opponent_kind, "policy"), checkpoint, seed=game.deal + 1
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
    found: list[Position] = []
    obs, _ = env.reset(seed=game.deal)
    hero.start_game(env, game.model_seat)
    start = getattr(opponent, "start_game", None)
    if start is not None:
        start(env, 1 - game.model_seat)
    for steps in range(MAX_STEPS):
        acting = int(env.last_raw_obs.agent.player_index)
        if acting == game.model_seat:
            base, _ = teacher.predict(env._engine)
            if (
                len(found) < budget
                and int((base > 0).sum()) >= 2
                and rng.random() < SEARCH_PROBABILITY
            ):
                assert hero.history is not None
                started = time.perf_counter()
                receipt = teacher.search(
                    env._engine, hero.history, seed=int(rng.integers(0, 2**62))
                )
                values: list[list[float] | None] = [None] * len(base)
                terminal = [0] * len(base)
                for rollout in receipt.rollouts:
                    row = values[rollout.action_index]
                    if row is None:
                        row = values[rollout.action_index] = []
                    row.append(round(float(rollout.signed_value), 4))
                    terminal[rollout.action_index] += int(rollout.terminal)
                turn = env.last_raw_obs.turn
                found.append(
                    Position(
                        key=f"{opponent_kind}/{game.deal}/{len(found)}",
                        opponent=opponent_kind,
                        deal=game.deal,
                        turn=int(turn.turn_number),
                        kind=int(env.last_raw_obs.action_space.action_space_type),
                        base=[float(p) for p in receipt.base],
                        values=values,
                        terminal=terminal,
                        seconds=round(time.perf_counter() - started, 3),
                    )
                )
            action = int(rng.choice(len(base), p=base / base.sum()))
        else:
            action = int(opponent.act(env, obs))
        hero.prepare_step(env, acting, action)
        frame = DecisionFrame.from_json(env._engine.semantic_decision_frame_json())
        command = Command(
            f"noise-{steps}", frame.revision, int(frame.offers[action]["id"])
        )
        obs, _, terminated, truncated, _, transition = env.step_semantic(command)
        if terminated or truncated:
            break
        hero.observe_step(env, acting, transition)
    return found


def play(
    checkpoint: Path, out: Path, worker: int, workers: int, positions: int
) -> None:
    """Search positions from games after the mirror run's deals until `positions`."""
    import torch

    from manabot.infra.identity import file_sha256

    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint bytes differ from MTG-183's final export")
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    host = os.uname().nodename
    path = out / f"positions-{host}-{worker}.jsonl.gz"
    done_path = out / f"games-{host}-{worker}.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    have = 0
    if path.exists():
        with gzip.open(path, "rt") as file:
            have = sum(1 for _ in file)
    teacher = _teacher(Arm("policy", DEPTH, 1.0, ROLLOUTS), checkpoint)
    games = sorted(
        (g for g in schedule() if g.index >= DEALS_PER_CELL),
        key=lambda g: (g.index, g.deal),
    )
    units = [(game, opponent) for game in games for opponent in ("policy", "heuristic")]
    for index, (game, opponent) in enumerate(units):
        if have >= positions:
            return
        key = f"{opponent}/{game.deal}"
        if index % workers != worker or key in done:
            continue
        try:
            found = _positions_in_game(
                opponent, game, teacher, checkpoint, positions - have
            )
        except Exception:
            print(f"{key} failed\n{traceback.format_exc(limit=8)}", flush=True)
            found = []
        with gzip.open(path, "at") as file:
            for position in found:
                file.write(json.dumps(asdict(position), separators=(",", ":")) + "\n")
        with done_path.open("a") as file:
            file.write(key + "\n")
        have += len(found)
        print(f"{key}: {len(found)} positions, {have}/{positions}", flush=True)


# ---------------------------------------------------------------------------
# Analysis


def _target(base: np.ndarray, q: np.ndarray, beta: float) -> np.ndarray:
    """The local update's closed form over supported moves (uniform reference)."""
    logits = (
        q + ALPHA * np.log(1.0 / len(base)) + beta * np.log(np.maximum(base, 1e-12))
    ) / (ALPHA + beta)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


@dataclass(frozen=True)
class NoiseRow:
    rollouts: int
    beta: float
    positions: int
    argmax_agreement: float
    total_variation: float
    gain_over_policy: float
    gain_with_reference_q: float
    gain_low: float
    gain_high: float


def analyze(out: Path, draws: int = 20) -> tuple[dict[str, float], list[NoiseRow]]:
    """Subsample n rollouts per move; judge against the held-out remainder.

    For each position, a draw picks n of the 64 rollouts per move as the
    estimate and averages the other 64 - n as the reference Q. Reported per
    (n, beta): how often the estimate's best move matches the reference's,
    total variation between their targets, and the reference-Q gain of the
    estimated target over the policy in signed value units, not game-win points.
    The finite remainder is noisy and optimizing against itself is optimistic.
    Percentile intervals resample whole source deals (both opponents together),
    retaining all their positions and draws; these are not independent learners.
    """
    positions: list[Position] = []
    for path in sorted(out.glob("positions-*.jsonl.gz")):
        with gzip.open(path, "rt") as file:
            positions += [Position(**json.loads(line)) for line in file]
    rng = np.random.default_rng(0)
    sums: dict[tuple[int, float], list[float]] = {}
    clusters: dict[tuple[int, float], dict[int, list[float]]] = defaultdict(dict)
    for position in positions:
        support = [i for i, v in enumerate(position.values) if v]
        base = np.asarray([position.base[i] for i in support])
        base = base / base.sum()
        rollouts = np.asarray([position.values[i] for i in support])
        width = rollouts.shape[1]
        for n in SUBSAMPLES:
            for _ in range(draws):
                order = rng.permuted(
                    np.tile(np.arange(width), (len(support), 1)), axis=1
                )
                pick = np.take_along_axis(rollouts, order, axis=1)
                estimate, reference = pick[:, :n].mean(1), pick[:, n:].mean(1)
                agree = float(np.argmax(estimate) == np.argmax(reference))
                for beta in BETAS:
                    est, ref = (
                        _target(base, estimate, beta),
                        _target(base, reference, beta),
                    )
                    row = sums.setdefault((n, beta), [0.0, 0.0, 0.0, 0.0, 0.0])
                    row[0] += 1
                    row[1] += agree
                    row[2] += 0.5 * float(np.abs(est - ref).sum())
                    row[3] += float((est - base) @ reference)
                    row[4] += float((ref - base) @ reference)
                    cluster = clusters[n, beta].setdefault(position.deal, [0.0, 0.0])
                    cluster[0] += 1
                    cluster[1] += float((est - base) @ reference)
    deals = sorted({p.deal for p in positions})
    # One common resampling matrix preserves covariance across all settings.
    picks = np.random.default_rng(26901).integers(0, len(deals), (10000, len(deals)))
    intervals: dict[tuple[int, float], tuple[float, float]] = {}
    for key, grouped in clusters.items():
        counts, totals = np.asarray([grouped[d] for d in deals]).T
        samples = totals[picks].sum(1) / counts[picks].sum(1)
        low, high = np.quantile(samples, [0.025, 0.975])
        intervals[key] = float(low), float(high)
    rows = [
        NoiseRow(
            n,
            beta,
            len(positions),
            round(s[1] / s[0], 3),
            round(s[2] / s[0], 4),
            round(s[3] / s[0], 4),
            round(s[4] / s[0], 4),
            *intervals[n, beta],
        )
        for (n, beta), s in sorted(sums.items())
    ]
    spread = [float(np.std([np.mean(v) for v in p.values if v])) for p in positions]
    single = [float(np.std(v)) for p in positions for v in p.values if v]
    summary = {
        "positions": len(positions),
        "source_deals": len(deals),
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 26901,
        "moves_per_position": round(
            float(np.mean([sum(1 for v in p.values if v) for p in positions])),
            2,
        ),
        "seconds_per_position": round(
            float(np.mean([p.seconds for p in positions])), 1
        ),
        "between_move_sd_of_q": round(float(np.mean(spread)), 3),
        "single_rollout_sd": round(float(np.mean(single)), 3),
        "single_rollout_sd_rms": round(math.sqrt(float(np.mean(np.square(single)))), 3),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("play")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--worker", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--positions", type=int, default=200)
    report = commands.add_parser("analyze")
    report.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "play":
        play(args.checkpoint, args.out, args.worker, args.workers, args.positions)
    else:
        summary, rows = analyze(args.out)
        print(json.dumps(summary))
        for row in rows:
            print(json.dumps(asdict(row)))


if __name__ == "__main__":
    main()
