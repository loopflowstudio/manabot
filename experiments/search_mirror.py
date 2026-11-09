"""Magnetic-mirror local updates over MTG-183's frozen endpoint.

Each searched decision tries every supported legal action once in each world,
plays `depth` decisions by
policy sampling, scores the position with the network, and samples the move from

    p ∝ exp((Q + alpha log uniform + beta log policy) / (alpha + beta))

(`manabot.sim.search.local_update.regularized_update`). Each Q averages one
rollout per world. The move is sampled, so the matching no-search baseline is the
sampled policy. These games reuse search_scaling's first five deals per grid
cell, so scores pair with every condition of that sweep.

The default v1 schedule holds the true opponent hand. V2 compares oracle and
uniform physical hands at fixed beta and 4/8/16 worlds, against raw sampled
policy only. Both reshuffle libraries. V2 workers require an admitted Etude
allocation and a shared absolute deadline; their private attempt streams retain
pending frames, Commands and search receipts separately from score summaries.

    uv run python -m experiments.search_mirror play --checkpoint PT --out DIR \
        --worker 0 --workers 7
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import gzip
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator, Literal

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
    sampler: Literal["oracle", "uniform"] = "oracle"
    study: Literal["v1", "v2", "v2-smoke"] = "v1"

    @property
    def key(self) -> str:
        suffix = f"-w{self.worlds}" if self.worlds > 1 else ""
        if self.study != "v1":
            return (
                f"{self.opponent}/mirror-{self.study}-{self.sampler}"
                f"-d{self.depth}-b{self.beta:g}{suffix}"
            )
        if self.sampler != "oracle":
            raise ValueError("v1 keys are reserved for oracle hands")
        return f"{self.opponent}/mirror-d{self.depth}-b{self.beta:g}{suffix}"


ARMS = tuple(
    Arm(opponent, depth, beta, worlds)
    for depth, beta, worlds in SETTINGS
    for opponent in ("policy", "heuristic")
)

# The extension is a fixed-beta factorial against the raw sampled policy only.
V2_ARMS = tuple(
    Arm("policy", 20, 0.01, worlds, sampler, "v2")
    for sampler in ("oracle", "uniform")
    for worlds in (4, 8, 16)
)


def planned_units(arms: tuple[Arm, ...] = ARMS) -> list[tuple[Game, Arm]]:
    """Keep the original deal-leading assignment, including within each v2 stage."""
    games = sorted(
        (g for g in schedule() if g.index < DEALS_PER_CELL),
        key=lambda g: (g.index, g.deal),
    )
    return [(game, arm) for game in games for arm in arms]


def extension_arms(
    stage: Literal["oracle", "uniform"], *, smoke: bool = False
) -> tuple[Arm, ...]:
    return tuple(
        Arm(
            a.opponent,
            a.depth,
            a.beta,
            a.worlds,
            a.sampler,
            "v2-smoke" if smoke else "v2",
        )
        for a in V2_ARMS
        if a.sampler == stage
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
    worlds: int | None = None  # Historical rows encode worlds only in their key.
    sampler: str = "oracle"
    study: str = "v1"
    game_seconds: float | None = None


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
    if arm.sampler == "oracle":
        oracle = OracleHandSampler()
        teacher._sampler_authority = oracle
        teacher.hand_sampler = oracle
    return teacher


def play_game(
    arm: Arm,
    game: Game,
    teacher: Any,
    checkpoint: Path,
    host: str,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[MirrorRow, list[MirrorDecision]]:
    from managym.decision import Command, DecisionFrame
    import torch

    from manabot.env import Env, Match, Reward
    from manabot.infra.config import MatchConfig, RewardConfig
    from manabot.sim.search.local_update import LocalUpdatePlayer

    started_game = time.perf_counter()
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
            if record is not None:
                record(
                    {
                        "event": "pending",
                        "step": steps,
                        "acting": acting,
                        "frame": json.loads(env._engine.semantic_decision_frame_json()),
                        "observation": json.loads(
                            env._engine.semantic_observation_json(acting)
                        ),
                    }
                )
            if acting == game.model_seat:
                started = time.perf_counter()
                action = int(hero.act(env, obs))
                elapsed = time.perf_counter() - started
                hero_seconds += elapsed
                receipt, turn = hero.last_receipt, env.last_raw_obs.turn
                if record is not None:
                    record(
                        {
                            "event": "search",
                            "step": steps,
                            "receipt": json.loads(receipt.to_json()),
                        }
                    )
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
            if record is not None:
                record(
                    {
                        "event": "command",
                        "step": steps,
                        "command": json.loads(command.to_json()),
                    }
                )
            obs, _, terminated, truncated, _, transition = env.step_semantic(command)
            steps += 1
            if terminated or truncated:
                break
            hero.observe_step(env, acting, transition)
        turns = int(env.last_raw_obs.turn.turn_number)
    except BaseException as error:  # PyO3 PanicException is not an Exception.
        if isinstance(error, (CampaignInterrupted, KeyboardInterrupt, SystemExit)):
            raise
        failure = traceback.format_exc()
        if record is not None:
            record(
                {
                    "event": "failure",
                    "step": steps,
                    "error": failure,
                    "torch_rng": torch.get_rng_state().tolist(),
                }
            )
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
            worlds=arm.worlds,
            sampler=arm.sampler,
            study=arm.study,
            game_seconds=time.perf_counter() - started_game,
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
    deadline: float | None = None,
) -> None:
    """Play this worker's share, deals leading so a stopped run stays balanced."""
    import torch

    from manabot.infra.identity import file_sha256

    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint bytes differ from MTG-183's final export")
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    host = os.uname().nodename
    if not 0 <= worker < workers <= 7:
        raise ValueError("use at most seven workers with distinct indexes")
    extension = any(a.study != "v1" for a in arms)
    units = planned_units(arms)
    if extension:
        if deadline is None or not math.isfinite(deadline) or deadline <= time.time():
            raise ValueError(
                "extension requires an unexpired absolute campaign deadline"
            )
        bind_extension(
            out, workers, deadline, smoke=all(a.study == "v2-smoke" for a in arms)
        )
        owned = {
            f"{a.key}/{g.deal}"
            for i, (g, a) in enumerate(units)
            if i % workers == worker
        }
        done = reconcile_extension(
            out,
            smoke=all(a.study == "v2-smoke" for a in arms),
            owned_keys=owned,
            game_file=f"games-{host}-{worker}.jsonl",
        )
        if arms == extension_arms("uniform"):
            required = {
                f"{a.key}/{g.deal}" for g, a in planned_units(extension_arms("oracle"))
            }
            # No oracle workers may remain when the owning job advances stages.
            if not required <= _done(out):
                raise ValueError(
                    "uniform stage requires reconciled oracle outcomes first"
                )
    else:
        done = _done(out)
    teachers: dict[tuple[int, float, int, str], Any] = {}
    played = 0
    for index, (game, arm) in enumerate(units):
        if index % workers != worker or f"{arm.key}/{game.deal}" in done:
            continue
        if limit is not None and played >= limit:
            return
        if deadline is not None and time.time() >= deadline:
            return
        setting = (arm.depth, arm.beta, arm.worlds, arm.sampler)
        teacher = teachers.get(setting)
        if teacher is None:
            teacher = teachers[setting] = _teacher(arm, checkpoint)
        if extension:
            assert deadline is not None
            row, log = retained_game(
                arm, game, teacher, checkpoint, host, out, deadline
            )
        else:
            row, log = play_game(arm, game, teacher, checkpoint, host)
        with gzip.open(out / f"decisions-{host}-{worker}.jsonl.gz", "at") as file:
            for entry in log:
                file.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        with (out / f"games-{host}-{worker}.jsonl").open("a") as file:
            file.write(json.dumps(asdict(row)) + "\n")
            file.flush()
            os.fsync(file.fileno())
        played += 1


class CampaignInterrupted(BaseException):
    """An allocation stop is an interrupted attempt, never a failed game score."""


def bind_extension(out: Path, workers: int, deadline: float, *, smoke: bool) -> None:
    """Freeze source, plan, worker assignment and deadline without racing other workers.

    The owning Etude allocation must already be admitted. This receipt does not
    acquire a machine or recover an interrupted attempt.
    """
    from manabot.infra.identity import file_sha256
    from manabot.managym import _managym as native

    root = Path(__file__).resolve().parents[1]
    sources = [
        p
        for directory in ("manabot", "managym/src", "experiments")
        for p in (root / directory).rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".rs", ".json", ".yaml", ".toml")
        and "data" not in p.relative_to(root).parts
    ]
    sources.extend(
        root / name
        for name in (
            "uv.lock",
            "pyproject.toml",
            "managym/Cargo.toml",
            "managym/Cargo.lock",
            "experiments/protocols/search-mirror-v2.md",
        )
    )
    native_sha = file_sha256(Path(native.__file__))
    if native_sha != "7c22a1e101874ae4818ba8250326277b2fb3b0a55260edcb9da0c162080baf25":
        raise ValueError("extension requires the retained mini w13 native build")
    arms = extension_arms("oracle", smoke=smoke) + extension_arms(
        "uniform", smoke=smoke
    )
    plan = [{"arm": asdict(a), "game": asdict(g)} for g, a in planned_units(arms)]
    identity = {
        "schema": "mirror-execution/v2",
        "workers": workers,
        "deadline": deadline,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "native_sha256": native_sha,
        "python": sys.version,
        "host": os.uname().nodename,
        "packages": {
            name: version(name) for name in ("torch", "numpy", "pydantic", "etude-sdk")
        },
        "source_files": {
            str(p.relative_to(root)): file_sha256(p) for p in sorted(sources)
        },
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest(),
        "planned_games": len(plan),
        "training_updates": 0,
    }
    path = out / "identity.json"
    encoded = json.dumps(identity, sort_keys=True, indent=2) + "\n"
    if deadline > time.time() + 168 * 3600:
        raise ValueError("campaign deadline exceeds the seven-day ceiling")
    with tempfile.NamedTemporaryFile(mode="w", dir=out, delete=False) as temporary:
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        try:
            os.link(temporary.name, path)  # Publish complete bytes, exclusively.
        except FileExistsError:
            if path.read_text() != encoded:
                raise ValueError(
                    "source, plan, shard assignment or deadline changed"
                ) from None
    finally:
        Path(temporary.name).unlink()


@contextmanager
def campaign_deadline(deadline: float) -> Iterator[None]:
    """Bound this foreground worker even during a slow search; Etude owns the job."""

    def stop(signum: int, frame: object) -> None:
        raise CampaignInterrupted(f"campaign stop signal {signum}")

    previous = {s: signal.signal(s, stop) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def reconcile_extension(
    out: Path,
    *,
    smoke: bool = False,
    owned_keys: set[str] | None = None,
    game_file: str = "games-*.jsonl",
) -> set[str]:
    """Fail closed on duplicates, changed identities or uncommitted attempt evidence."""
    arms = extension_arms("oracle", smoke=smoke) + extension_arms(
        "uniform", smoke=smoke
    )
    expected = {f"{a.key}/{g.deal}": (g, a) for g, a in planned_units(arms)}
    done: set[str] = set()
    rows: dict[str, MirrorRow] = {}
    for path in sorted(out.glob(game_file)):
        for line in path.read_text().splitlines():
            row = MirrorRow(**json.loads(line))
            if row.key in done or row.key not in expected:
                raise ValueError(f"duplicate or unexpected outcome: {row.key}")
            if row.score not in (None, 0, 0.5, 1) or (row.score is None) != (
                row.failure is not None
            ):
                raise ValueError(f"invalid outcome status: {row.key}")
            game, arm = expected[row.key]
            if (
                row.condition,
                row.deal,
                row.model_deck,
                row.opponent_deck,
                row.model_seat,
                row.opponent,
                row.depth,
                row.beta,
                row.worlds,
                row.sampler,
                row.study,
            ) != (
                arm.key,
                game.deal,
                game.model_deck,
                game.heuristic_deck,
                game.model_seat,
                arm.opponent,
                arm.depth,
                arm.beta,
                arm.worlds,
                arm.sampler,
                arm.study,
            ):
                raise ValueError(f"extension identity mismatch: {row.key}")
            done.add(row.key)  # Includes failures; never replace them with successes.
            rows[row.key] = row
    owned_digests = (
        None
        if owned_keys is None
        else {hashlib.sha256(key.encode()).hexdigest() for key in owned_keys}
    )
    for path in (out / "attempts").glob("*.jsonl.gz"):
        if (
            owned_digests is not None
            and path.name.removesuffix(".jsonl.gz") not in owned_digests
        ):
            continue
        with gzip.open(path, "rt") as stream:
            start = json.loads(next(stream))
            last = start
            for line in stream:
                last = json.loads(line)
        if start["key"] not in done:
            raise ValueError(f"unreconciled attempt (not an ordinary failure): {path}")
        if last.get("event") != "outcome" or last.get("row") != asdict(
            rows[start["key"]]
        ):
            raise ValueError(f"incomplete or conflicting attempt evidence: {path}")
    return done


def retained_game(
    arm: Arm,
    game: Game,
    teacher: Any,
    checkpoint: Path,
    host: str,
    out: Path,
    deadline: float,
) -> tuple[MirrorRow, list[MirrorDecision]]:
    """Stream private replay evidence before each effect, retaining interrupted attempts."""
    key = f"{arm.key}/{game.deal}"
    directory = out / "attempts"
    directory.mkdir(exist_ok=True)
    path = directory / f"{hashlib.sha256(key.encode()).hexdigest()}.jsonl.gz"
    with gzip.open(path, "xt") as stream:

        def record(event: dict[str, Any]) -> None:
            # JSON is the native receipt boundary; reject nonfinite values here.
            stream.write(
                json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
            )
            stream.flush()

        record(
            {
                "event": "start",
                "key": key,
                "arm": asdict(arm),
                "game": asdict(game),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "runtime_sha256": teacher.runtime_sha256,
                "search_source_sha256": teacher.source_sha256,
                "world_binding": teacher.agent.world_binding,
                "config": teacher.config.model_dump(mode="json"),
                "observation_config": teacher.space.encoder.config.model_dump(
                    mode="json"
                ),
                "hero_seed": game.deal,
                "opponent_seed": game.deal + 1,
                "started": time.time(),
                "deadline": deadline,
                "host": host,
            }
        )
        try:
            result = play_game(arm, game, teacher, checkpoint, host, record)
            record({"event": "outcome", "row": asdict(result[0])})
            return result
        except BaseException:
            record(
                {
                    "event": "interrupted",
                    "error": traceback.format_exc(),
                    "stopped": time.time(),
                }
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("play")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--worker", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--study", choices=("v1", "v2", "v2-smoke"), default="v1")
    run.add_argument("--stage", choices=("oracle", "uniform"))
    run.add_argument(
        "--deadline", type=float, help="absolute campaign stop, Unix seconds"
    )
    args = parser.parse_args()
    if (args.study == "v1") != (args.stage is None):
        parser.error("--stage is required exactly for v2 or v2-smoke")
    arms = (
        ARMS
        if args.study == "v1"
        else extension_arms(args.stage, smoke=args.study == "v2-smoke")
    )
    # The OS releases the lock on interruption. This does not authorize replaying
    # an incomplete attempt: reconciliation above still requires explicit recovery.
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"worker-{args.worker}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.study == "v1":
            play(args.checkpoint, args.out, args.worker, args.workers, args.limit)
        else:
            if args.deadline is None or not math.isfinite(args.deadline):
                parser.error("extension requires --deadline")
            with campaign_deadline(args.deadline):
                play(
                    args.checkpoint,
                    args.out,
                    args.worker,
                    args.workers,
                    args.limit,
                    arms,
                    args.deadline,
                )


if __name__ == "__main__":
    main()
