"""Test-time compute on a frozen policy: agent_puct budgets over MTG-183's endpoint.

The question: how much does determinized PUCT search add to a well-understood
policy, and how much of what it could add is limited by beliefs about the
opponent's hand? The policy is MTG-183's final raw checkpoint (update 5,950,
w13). Every condition plays the same fixed deals from the 10×10 Elemental grid
(`experiments.elemental_grid.schedule`), so conditions are paired by deal.

Conditions (`CONDITIONS`):

- `policy`: the bare checkpoint sampling its action, as every MTG-183 evaluation
  played it; `greedy`: the same checkpoint taking its most likely action.
- `puct`: `agent_puct` at a simulation budget, with each world's opponent hand
  from `uniform` (known hand plus a uniform draw from the unseen pool) or
  `oracle` (the true hand). Both reshuffle libraries, so neither knows future
  draws. Oracle bounds what any belief over the hand could add to this search;
  it also removes strategy fusion over the hand, so the bound is loose.

Each is played against two opponents: the bare checkpoint (`policy`) and the
w13 `heuristic`. A game row records the outcome; a searched decision records
what the search saw and spent (`DecisionLog`), so later work can study where
search changes the policy's choice and what that costs.

    uv run python -m experiments.search_scaling plan --games-per-cell 10 --out plan.jsonl
    uv run python -m experiments.search_scaling play --plan plan.jsonl \
        --checkpoint PATH --out DIR --worker 0 --workers 8
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
import gzip
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, Literal

from experiments.elemental_grid import DECKS, PACK, Game, schedule

CHECKPOINT_SHA256 = "4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4"
# The deals' frozen grid; a plan uses the first `games_per_cell` deals of each cell.
GRID_SCHEDULE_SHA256 = "ecc487e4708a8565b9c787068ff676f62b0c35420b179204fbbf72056f436264"
MAX_STEPS = 10_000
WORLDS = 4
C_PUCT = 1.5

Hero = Literal["policy", "greedy", "puct"]
Opponent = Literal["policy", "heuristic"]
Source = Literal["uniform", "oracle"]


@dataclass(frozen=True)
class Condition:
    """One way of playing the hero against one opponent."""

    opponent: Opponent
    hero: Hero
    sims: int = 0
    source: Source | None = None

    @property
    def key(self) -> str:
        if self.hero != "puct":
            return f"{self.opponent}/{self.hero}"
        return f"{self.opponent}/puct-{self.sims}-{self.source}"


BUDGETS = (16, 64, 256, 1024)
CONDITIONS: tuple[Condition, ...] = tuple(
    [
        Condition(opponent, hero)
        for opponent in ("policy", "heuristic")
        for hero in ("policy", "greedy")
    ]
    + [
        Condition(opponent, "puct", sims, source)
        for sims in BUDGETS
        for opponent in ("policy", "heuristic")
        for source in ("uniform", "oracle")
    ]
)


@dataclass(frozen=True)
class Unit:
    """One game of one condition on one scheduled deal."""

    condition: Condition
    game: Game

    @property
    def key(self) -> str:
        return f"{self.condition.key}/{self.game.deal}"

    def to_json(self) -> str:
        return json.dumps(
            {"condition": asdict(self.condition), "game": asdict(self.game)},
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> Unit:
        raw = json.loads(line)
        return cls(Condition(**raw["condition"]), Game(**raw["game"]))


# The largest budget costs about as much as all others together, so it plays the
# first half of each cell's deals; every deal it plays is also played by the rest.
DEALS_PER_CELL = {1024: 5}


def plan(games_per_cell: int, conditions: Iterable[Condition] = CONDITIONS) -> list[Unit]:
    """Units cheapest condition first, deals round-robin across cells within it.

    Workers take units in this order, so a partial run holds complete low budgets
    and balanced cells for the budget in progress.
    """
    by_round = sorted(schedule(), key=lambda game: (game.index, game.deal))
    ordered = sorted(conditions, key=lambda c: (c.sims, c.hero != "policy", c.key))
    return [
        Unit(condition, game)
        for condition in ordered
        for game in by_round
        if game.index < min(games_per_cell, DEALS_PER_CELL.get(condition.sims, games_per_cell))
    ]


# ---------------------------------------------------------------------------
# Records


@dataclass(frozen=True)
class GameRow:
    key: str
    condition: str
    opponent: str
    hero: str
    sims: int
    source: str | None
    model_deck: str
    opponent_deck: str
    model_seat: int
    deal: int
    score: float | None
    winner: int | None
    terminated: bool
    truncated: bool
    failure: str | None
    turns: int
    decisions: int
    hero_decisions: int
    hero_seconds: float
    opponent_seconds: float
    game_seconds: float
    searched_changed: int
    forward_calls: int
    host: str


@dataclass(frozen=True)
class DecisionLog:
    """What one root search saw and spent; values are hero win scores in [0, 1]."""

    key: str
    index: int
    turn: int
    phase: int
    step: int
    kind: int
    actions: int
    chosen: int
    prior_argmax: int
    priors: list[float]
    visits: list[int]
    q: list[float]
    root_value: float
    world_visits: list[list[int]]
    world_q: list[list[float]]
    world_root_values: list[float]
    tree_nodes: int
    max_depth: int
    cap_hits: int
    seconds: float


# ---------------------------------------------------------------------------
# Players


class _Timed:
    """Wrap a player to charge its decisions and, for PUCT, log each root."""

    def __init__(self, player: Any, key: str, log: list[DecisionLog] | None) -> None:
        self.player = player
        self.key = key
        self.log = log
        self.seconds = 0.0
        self.decisions = 0
        self.changed = 0

    def start_game(self, env: Any, seat: int) -> None:
        start = getattr(self.player, "start_game", None)
        if start is not None:
            start(env, seat)

    def act(self, env: Any, obs: Any) -> int:
        started = time.perf_counter()
        action = int(self.player.act(env, obs))
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.decisions += 1
        if self.log is not None:
            result = self.player.last_result
            priors = self.player.evaluator.last_root_priors
            prior_argmax = int(max(range(len(priors)), key=lambda i: priors[i]))
            self.changed += int(action != prior_argmax)
            turn = env.last_raw_obs.turn
            self.log.append(
                DecisionLog(
                    key=self.key,
                    index=self.decisions - 1,
                    turn=int(turn.turn_number),
                    phase=int(turn.phase),
                    step=int(turn.step),
                    kind=int(env.last_raw_obs.action_space.action_space_type),
                    actions=len(priors),
                    chosen=action,
                    prior_argmax=prior_argmax,
                    priors=[round(float(p), 5) for p in priors],
                    visits=[int(v) for v in result.visit_counts],
                    q=[round(float(v), 5) for v in result.q_values],
                    root_value=round(float(result.root_value), 5),
                    world_visits=[[int(v) for v in row] for row in result.world_visit_counts],
                    world_q=[[round(float(v), 4) for v in row] for row in result.world_q_values],
                    world_root_values=[round(float(v), 5) for v in result.world_root_values],
                    tree_nodes=int(result.tree_nodes),
                    max_depth=int(result.max_depth),
                    cap_hits=int(result.cap_hits),
                    seconds=round(elapsed, 5),
                )
            )
        return action


def _hero(condition: Condition, checkpoint: Path, seed: int) -> tuple[Any, Any]:
    from manabot.sim.search.flat_mc import make_player

    if condition.hero == "puct":
        return make_player(
            {
                "kind": "agent_puct",
                "sims": condition.sims,
                "worlds": min(WORLDS, condition.sims),
                "c_puct": C_PUCT,
                "checkpoint": str(checkpoint),
                "world_source": condition.source,
            },
            seed=seed,
        )
    return make_player(
        {
            "kind": "checkpoint",
            "path": str(checkpoint),
            "deterministic": condition.hero == "greedy",
        },
        seed=seed,
    )


def _opponent(condition: Condition, checkpoint: Path, seed: int) -> Any:
    if condition.opponent == "heuristic":
        from manabot.sim.heuristic import HeuristicPlayer

        return HeuristicPlayer()
    from manabot.sim.search.flat_mc import make_player

    return make_player({"kind": "checkpoint", "path": str(checkpoint)}, seed=seed)[0]


# ---------------------------------------------------------------------------
# One game


def play_unit(unit: Unit, checkpoint: Path, host: str) -> tuple[GameRow, list[DecisionLog]]:
    import torch

    from manabot.env import Env, Match, Reward
    from manabot.infra.config import MatchConfig, RewardConfig

    condition, game = unit.condition, unit.game
    # Policy sampling draws from torch's global stream; seed it from the deal so
    # every condition replays the same opponent randomness given the same moves.
    torch.manual_seed(game.deal)
    hero_player, space = _hero(condition, checkpoint, seed=game.deal)
    log: list[DecisionLog] | None = [] if condition.hero == "puct" else None
    hero = _Timed(hero_player, unit.key, log)
    opponent = _Timed(_opponent(condition, checkpoint, seed=game.deal + 1), unit.key, None)
    decks = [game.heuristic_deck, game.heuristic_deck]
    decks[game.model_seat] = game.model_deck
    setup = MatchConfig.authored(PACK, decks[0], decks[1], hero="seat-0", villain="seat-1")
    env = Env(
        Match(setup.model_copy(update={"content_pack": PACK})),
        space,
        Reward(RewardConfig()),
        seed=game.deal,
        auto_reset=False,
        enable_profiler=False,
        enable_behavior_tracking=False,
    )
    started = time.perf_counter()
    failure: str | None = None
    terminated = truncated = False
    steps = 0
    turns = 0
    try:
        obs, _ = env.reset(seed=game.deal)
        hero.start_game(env, game.model_seat)
        opponent.start_game(env, 1 - game.model_seat)
        while steps < MAX_STEPS:
            acting = int(env.last_raw_obs.agent.player_index)
            player = hero if acting == game.model_seat else opponent
            obs, _, terminated, truncated, _ = env.step(player.act(env, obs))
            steps += 1
            if terminated or truncated:
                break
        turns = int(env.last_raw_obs.turn.turn_number)
    except Exception:  # retained as an invalid row, never replayed or replaced
        failure = traceback.format_exc(limit=8)
    winner = env._engine.winner_index() if terminated and failure is None else None
    if failure is None and not terminated:
        failure = "truncated" if truncated else "step bound"
    score = None
    if failure is None:
        score = 0.5 if winner is None else float(int(winner) == game.model_seat)
    forward_calls = getattr(getattr(hero_player, "evaluator", None), "forward_calls", 0)
    row = GameRow(
        key=unit.key,
        condition=condition.key,
        opponent=condition.opponent,
        hero=condition.hero,
        sims=condition.sims,
        source=condition.source,
        model_deck=game.model_deck,
        opponent_deck=game.heuristic_deck,
        model_seat=game.model_seat,
        deal=game.deal,
        score=score,
        winner=None if winner is None else int(winner),
        terminated=bool(terminated),
        truncated=bool(truncated),
        failure=failure,
        turns=turns,
        decisions=steps,
        hero_decisions=hero.decisions,
        hero_seconds=round(hero.seconds, 4),
        opponent_seconds=round(opponent.seconds, 4),
        game_seconds=round(time.perf_counter() - started, 4),
        searched_changed=hero.changed,
        forward_calls=int(forward_calls),
        host=host,
    )
    return row, log or []


# ---------------------------------------------------------------------------
# Workers


def _done(out: Path) -> set[str]:
    done: set[str] = set()
    for path in out.glob("games-*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    return done


def _units(path: Path) -> Iterator[Unit]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield Unit.from_json(line)


def play(
    plan_path: Path,
    checkpoint: Path,
    out: Path,
    worker: int,
    workers: int,
    sims: tuple[int, int] = (0, max(BUDGETS)),
) -> None:
    """Play this worker's share of the plan's units within `sims`, resuming.

    Hosts split the plan by budget; workers on a host split it by unit index.
    """
    import torch

    from manabot.infra.identity import file_sha256

    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint bytes differ from MTG-183's final export")
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    host = os.uname().nodename
    done = _done(out)
    games = out / f"games-{host}-{worker}.jsonl"
    decisions = out / f"decisions-{host}-{worker}.jsonl.gz"
    # Within a budget, deals lead and conditions interleave, so a stopped host
    # still holds every condition of its budget on the same deals.
    units = sorted(
        (u for u in _units(plan_path) if sims[0] <= u.condition.sims <= sims[1]),
        key=lambda u: (u.condition.sims, u.game.index, u.game.deal, u.condition.key),
    )
    for index, unit in enumerate(units):
        if index % workers != worker or unit.key in done:
            continue
        row, log = play_unit(unit, checkpoint, host)
        if log:
            with gzip.open(decisions, "at") as file:
                for entry in log:
                    file.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        with games.open("a") as file:
            file.write(json.dumps(asdict(row)) + "\n")
            file.flush()
            os.fsync(file.fileno())


# ---------------------------------------------------------------------------
# Rented hosts (Etude)

REVISION_FILE = "REVISION"
RUST_TOOLCHAIN = "1.98.1"
CHECKPOINT_KEY = "jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt"
CHECKPOINT_BYTES = 1_071_166
# A share's workers stop this long before Etude's deadline so it can deliver.
STOP_MARGIN_SECONDS = 600


@dataclass(frozen=True)
class Share:
    """One rented host's part of the plan: a budget range and a worker slice."""

    name: str
    min_sims: int
    max_sims: int
    first_worker: int
    workers: int
    total_workers: int


SHARES = (
    Share("s256", 256, 256, 0, 32, 32),
    Share("s1024a", 1024, 1024, 0, 32, 64),
    Share("s1024b", 1024, 1024, 32, 32, 64),
)


def job() -> None:
    """On a rented host: play one share with every worker, stop before the deadline.

    Completed games are durable rows in ETUDE_OUTPUT_DIR; a game in progress
    at the stop is lost and replayed by any later run of the same share.
    """
    import signal
    import subprocess
    import sys

    from datetime import datetime

    share = Share(**json.loads(Path(os.environ["ETUDE_CONFIG"]).read_text()))
    out = Path(os.environ["ETUDE_OUTPUT_DIR"])
    checkpoint = Path(os.environ["ETUDE_INPUTS_DIR"]) / "policy.pt"
    stop = Path(os.environ["ETUDE_STOP_FILE"])
    given = os.environ.get("ETUDE_DEADLINE")
    deadline = datetime.fromisoformat(given).timestamp() if given else math.inf
    children = [
        subprocess.Popen(
            [
                sys.executable, "-m", "experiments.search_scaling", "play",
                "--plan", "experiments/search-scaling/plan-v1.jsonl",
                "--checkpoint", str(checkpoint), "--out", str(out),
                "--worker", str(share.first_worker + i),
                "--workers", str(share.total_workers),
                "--min-sims", str(share.min_sims), "--max-sims", str(share.max_sims),
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        for i in range(share.workers)
    ]
    started = time.time()
    while any(child.poll() is None for child in children):
        if stop.exists() or time.time() > deadline - STOP_MARGIN_SECONDS:
            for child in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
            break
        time.sleep(10)
    for child in children:
        child.wait()
    rows = sum(1 for path in out.glob("games-*.jsonl") for _ in path.open())
    (out / "etude-result.json").write_text(
        json.dumps({"clocks": {"steps": rows, "wall_time_ms": int(1000 * (time.time() - started))}})
    )


def runs(revision: str = "HEAD", hours: float = 4.5) -> list[Any]:
    """Etude jobs for every share; construction launches nothing."""
    from decimal import Decimal

    from etude.launch.records import Input, Job

    setup = (
        "set -eu\n"
        'test -x "$HOME/.cargo/bin/rustup" || curl --proto =https --tlsv1.2 -sSf '
        "https://sh.rustup.rs | sh -s -- -y --profile minimal "
        f"--default-toolchain {RUST_TOOLCHAIN}\n"
        f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}\n'
        f"rustup toolchain install {RUST_TOOLCHAIN} --profile minimal\n"
        "uv sync --locked --python 3.12\n"
    )
    return [
        Job(
            name=f"search-scaling-{share.name}",
            setup=setup,
            command=(
                f'export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_TOOLCHAIN={RUST_TOOLCHAIN}\n'
                "uv run --locked python -m experiments.search_scaling job"
            ),
            config=json.dumps(asdict(share)),
            repo=str(Path(__file__).resolve().parents[1]),
            revision=revision,
            max_hours=hours,
            # $0.96/hour ceiling $1.00, plus Etude's 15-minute setup allowance.
            max_usd=Decimal(str(round(hours + 0.5, 2))),
            max_attempts=1,
            inputs=(
                Input(
                    name="policy.pt",
                    key=CHECKPOINT_KEY,
                    sha256=CHECKPOINT_SHA256,
                    size_bytes=CHECKPOINT_BYTES,
                ),
            ),
        )
        for share in SHARES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan")
    make.add_argument("--games-per-cell", type=int, default=10)
    make.add_argument("--max-sims", type=int, default=max(BUDGETS))
    make.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("play")
    commands.add_parser("job")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--worker", type=int, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--min-sims", type=int, default=0)
    run.add_argument("--max-sims", type=int, default=max(BUDGETS))
    args = parser.parse_args()
    if args.command == "plan":
        from experiments.elemental_grid import schedule_sha256

        if schedule_sha256() != GRID_SCHEDULE_SHA256:
            raise ValueError("grid schedule differs from the frozen deals")
        chosen = [c for c in CONDITIONS if c.sims <= args.max_sims]
        units = plan(args.games_per_cell, chosen)
        args.out.write_text("".join(unit.to_json() + "\n" for unit in units))
        print(f"{len(units)} units, {len(chosen)} conditions, {len(DECKS) ** 2} cells")
    elif args.command == "job":
        job()
    else:
        play(
            args.plan,
            args.checkpoint,
            args.out,
            args.worker,
            args.workers,
            (args.min_sims, args.max_sims),
        )


if __name__ == "__main__":
    main()
