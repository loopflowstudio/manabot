"""The 10×10 Elemental deck grid: one checkpoint against the heuristic.

Jack Heart's evaluation design (MTG-181, 2026-10-23): the model plays every
Elemental deck (x axis) against the heuristic playing every Elemental deck
(y axis), mirrors included, 50 games per cell. Each game's seat, and so whether
the model is on the play (seat 0 takes the first turn), comes from one recorded
random seed; deals are fixed per game. Every checkpoint therefore plays the same
5,000 games, so checkpoints and training seeds are compared on paired deals.

`schedule` is the frozen game list and `SCHEDULE_SHA256` pins it. `evaluate`
plays one checkpoint by running bounded subprocess workers (`python -m
experiments.elemental_grid`), one per (cell, seat) group, through the
replay-checked arena `play_cell`. `summarize` turns retained rows into cell,
row, column and overall scores with 95% Wilson intervals; `grid_table` renders
that summary as HTML for the training report.

Scores count a win as 1, a draw as 0.5 and a loss as 0. Only valid games count:
terminated, no failure, and the trace replayed. Invalid games are reported
beside the score, never replaced or replayed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from html import escape
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from manabot.infra.identity import canonical_sha256
from manabot.sim.execution import EvalCompute, EvalExecution
from manabot.sim.threads import child_environment

PACK = "elemental-world-v2"
DECKS = (
    "elemental_bg",
    "elemental_gr",
    "elemental_gw",
    "elemental_rb",
    "elemental_ub",
    "elemental_ug",
    "elemental_ur",
    "elemental_uw",
    "elemental_wb",
    "elemental_wr",
)
GRID_SEED = 18_190_001
GAMES_PER_CELL = 50
DEAL_BASE = 1_819_000_000
GAME_SECONDS = 600.0
MAX_COMMANDS = 10_000
# One group is about 25 games; a group past this wall time is killed and its
# games count as invalid. The bound protects the queue, not the score.
GROUP_SECONDS = 3_600
WORKERS = 10


@dataclass(frozen=True)
class Game:
    model_deck: str
    heuristic_deck: str
    index: int
    deal: int
    model_seat: Literal[0, 1]


def schedule() -> list[Game]:
    """All 5,000 games in cell order; one RNG stream assigns every seat."""
    rng = random.Random(GRID_SEED)
    games: list[Game] = []
    for x, model_deck in enumerate(DECKS):
        for y, heuristic_deck in enumerate(DECKS):
            cell = x * len(DECKS) + y
            for index in range(GAMES_PER_CELL):
                seat: Literal[0, 1] = 0 if rng.random() < 0.5 else 1
                games.append(
                    Game(
                        model_deck,
                        heuristic_deck,
                        index,
                        DEAL_BASE + cell * 1_000 + index,
                        seat,
                    )
                )
    return games


def schedule_sha256() -> str:
    return canonical_sha256([vars(game) for game in schedule()])


SCHEDULE_SHA256 = "ecc487e4708a8565b9c787068ff676f62b0c35420b179204fbbf72056f436264"


# ---------------------------------------------------------------------------
# Worker: one (model deck, heuristic deck, model seat) group in one process


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deals: tuple[int, ...]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: Path
    sha256: str
    training_seed: int
    group: Group
    output: Path
    # The eval run's folder: when given, each game's card evidence is retained there.
    card_evidence: Path | None = None
    # Shared batched inference for the group's games; None is the frozen scalar path.
    compute: EvalCompute | None = None


class GameRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_deck: str
    heuristic_deck: str
    model_seat: Literal[0, 1]
    deal: int
    score: float | None
    valid: bool
    terminated: bool
    truncated: bool
    failure: str | None
    decisions: int
    model_decisions: int
    seconds: float


def groups(games: Iterable[Game]) -> list[Group]:
    found: dict[tuple[str, str, int], list[int]] = {}
    for game in games:
        key = (game.model_deck, game.heuristic_deck, game.model_seat)
        found.setdefault(key, []).append(game.deal)
    return [
        Group(model_deck=x, heuristic_deck=y, model_seat=seat, deals=tuple(deals))  # type: ignore[arg-type]
        for (x, y, seat), deals in found.items()
    ]


def _legs(group: Group) -> tuple[str, str, int]:
    """Authored pair and arena leg that put the model's deck in its seat.

    Cross legs 0 and 3 give player A (the model) role A's deck in seats 0 and
    1; mirror legs 4 and 5 give both seats role A's deck. A mirror needs some
    distinct role B to form the authored pair; it never plays.
    """
    x, y = group.model_deck, group.heuristic_deck
    if x != y:
        return x, y, (0, 3)[group.model_seat]
    partner = DECKS[(DECKS.index(x) + 1) % len(DECKS)]
    return x, partner, (4, 5)[group.model_seat]


def play_group(request: Request) -> list[GameRow]:
    import torch

    from manabot.eval.models import EvaluationKey
    from manabot.infra.identity import file_sha256
    from manabot.infra.provenance import runtime_fingerprints, source_bundle_sha256
    from manabot.model.checkpoint import load_checkpoint
    from manabot.sim.heuristic import heuristic_source_paths
    from manabot.sim.matchup import AuthoredMatchup
    from manabot.sim.registration import PlayerRegistration
    import managym

    torch.set_num_threads(1)
    role_a, role_b, leg = _legs(request.group)
    matchup = AuthoredMatchup(pack_key=PACK, role_a=role_a, role_b=role_b)
    common = dict(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        matchup_sha256=canonical_sha256(matchup.match().model_dump()),
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    if file_sha256(request.checkpoint) != request.sha256:
        raise ValueError("checkpoint byte identity differs")
    model, space = load_checkpoint(request.checkpoint)
    # The checkpoint's own observation space (its saved policy-input version)
    # names its ABI; served admission rejects any other. The heuristic plays
    # in the runtime's default space.
    own = runtime_fingerprints(match_config=matchup.match(), observation_space=space)
    default = runtime_fingerprints(match_config=matchup.match())
    candidate = PlayerRegistration(
        **common,
        observation_abi_sha256=own["observation_abi_sha256"],
        action_abi_sha256=own["action_abi_sha256"],
        player_id="model",
        display_name="Checkpoint",
        role="challenger",
        compute_class_id="policy-cpu-one-thread-one-pass",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "device": "cpu",
            "batch_size": 1,
            "deterministic": False,
            "allow_deck_repetition": True,
        },
        checkpoint_sha256=request.sha256,
        checkpoint_bytes=request.checkpoint.stat().st_size,
        parameter_count=sum(p.numel() for p in model.parameters()),
        training_seed=request.training_seed,
        artifact_id=f"mtg181/{request.sha256}",
    )
    heuristic = PlayerRegistration(
        **common,
        observation_abi_sha256=default["observation_abi_sha256"],
        action_abi_sha256=default["action_abi_sha256"],
        player_id="heuristic",
        display_name="Heuristic",
        role="anchor",
        compute_class_id="heuristic-cpu-v1",
        runner_kind="code",
        player_spec={"kind": "heuristic"},
        source_sha256=source_bundle_sha256(
            [Path(path) for path in heuristic_source_paths()]
        ),
    )
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="mtg181-elemental-grid-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(heuristic.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    request.output.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = group_cell(request.compute)(
        key=key,
        player_a=candidate,
        player_b=heuristic,
        deal_seeds=request.group.deals,
        out_dir=request.output,
        checkpoint_paths={"model": str(request.checkpoint)},
        game_seconds=GAME_SECONDS,
        max_commands=MAX_COMMANDS,
        authored_matchup=matchup,
        matchup_mode="cross" if leg < 4 else "mirrors",
        legs=(leg,),
        card_evidence_dir=request.card_evidence,
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    result = []
    for row in rows:
        decks = row["seat_decks"]
        seat = row["player_a_seat"]
        if decks[seat] != request.group.model_deck or decks[1 - seat] != (
            request.group.heuristic_deck
        ):
            raise ValueError("arena leg did not assign the scheduled decks")
        if seat != request.group.model_seat:
            raise ValueError("arena leg did not assign the scheduled seat")
        result.append(
            GameRow(
                model_deck=request.group.model_deck,
                heuristic_deck=request.group.heuristic_deck,
                model_seat=seat,
                deal=row["deal_seed"],
                score=row["score_a"],
                valid=bool(
                    row["terminated"]
                    and not row["truncated"]
                    and row["failure"] is None
                    and row["replay_passed"]
                ),
                terminated=row["terminated"],
                truncated=row["truncated"],
                failure=None if row["failure"] is None else str(row["failure"]),
                decisions=row["decisions"],
                model_decisions=row["latency"]["model"]["count"],
                seconds=row["game_seconds"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# One checkpoint: every group, durable per group, bounded workers


# A machine-wide shared allocation's endpoint (`endpoint_environment`): when set,
# every group plays its cell through it, so all games share one model batch.
ENDPOINT = "MANABOT_EVAL_ENDPOINT"
Cell = tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]


def endpoint_environment(endpoint: Any) -> str:
    """An `InferenceEndpoint` as the `ENDPOINT` value group workers read."""
    found = asdict(endpoint)
    return json.dumps(
        {
            "address": found["address"],
            "authkey": found["authkey"].hex(),
            "execution": endpoint.execution.model_dump(mode="json"),
        }
    )


def group_cell(compute: EvalCompute | None) -> Callable[..., Cell]:
    """How a group plays its cell: through the machine's allocation when
    `ENDPOINT` names one, else in this process (batched per group under `compute`)."""
    from manabot.eval.endpoint import InferenceEndpoint
    from manabot.eval.match import play_cell

    found = os.environ.get(ENDPOINT)
    if found:
        if compute is not None:
            raise ValueError("a group plays under one compute contract")
        data = json.loads(found)
        return InferenceEndpoint(
            address=data["address"],
            authkey=bytes.fromhex(data["authkey"]),
            execution=EvalExecution.model_validate(data["execution"]),
        ).play_cell
    return partial(play_cell, execution=group_execution(compute))


def group_execution(compute: EvalCompute | None) -> EvalExecution | None:
    """One group's allocation under `compute`, bounded by the group's wall time."""
    if compute is None:
        return None
    return EvalExecution(
        **compute.model_dump(), deadline_unix=time.time() + GROUP_SECONDS
    )


def _group_name(group: Group) -> str:
    return f"{group.model_deck}--{group.heuristic_deck}--seat{group.model_seat}"


def evaluate(
    checkpoint: Path,
    sha256: str,
    training_seed: int,
    directory: Path,
    *,
    workers: int = WORKERS,
    root: Path | None = None,
    games: Sequence[Game] | None = None,
) -> list[GameRow]:
    """Play the whole grid (or a declared subset of its games); completed groups
    are reused after an interruption.

    A group's retained `games.json` is its durable result. A group that failed
    or timed out is retained as `failed.json` and counts its games as invalid;
    it is never replayed, including after a restart.
    """
    root = root or Path(__file__).resolve().parents[1]
    directory.mkdir(parents=True, exist_ok=True)
    games = schedule() if games is None else games
    pending = []
    for group in groups(games):
        target = directory / _group_name(group)
        if (
            not (target / "games.json").exists()
            and not (target / "failed.json").exists()
        ):
            pending.append(group)

    def run(group: Group) -> None:
        target = directory / _group_name(group)
        attempt = target / f"attempt-{time.time_ns()}"
        target.mkdir(parents=True, exist_ok=True)
        request = Request(
            checkpoint=checkpoint,
            sha256=sha256,
            training_seed=training_seed,
            group=group,
            output=attempt,
        )
        started = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.elemental_grid"],
                cwd=root,
                input=request.model_dump_json(),
                text=True,
                capture_output=True,
                timeout=GROUP_SECONDS,
                env=child_environment(),
            )
            error = None if completed.returncode == 0 else completed.stderr[-4000:]
            output = completed.stdout
        except subprocess.TimeoutExpired:
            error, output = "group wall-time bound reached", ""
        if error is None:
            rows = [GameRow.model_validate(r) for r in json.loads(output)]
            _atomic(target / "games.json", [r.model_dump(mode="json") for r in rows])
        else:
            _atomic(
                target / "failed.json",
                {"error": error, "started": started, "finished": time.time()},
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, pending))
    return collect(directory, games)


def collect(directory: Path, games: Sequence[Game] | None = None) -> list[GameRow]:
    """Retained rows for every scheduled game; failed groups become invalid rows."""
    rows: list[GameRow] = []
    for group in groups(schedule() if games is None else games):
        target = directory / _group_name(group)
        if (target / "games.json").exists():
            found = [
                GameRow.model_validate(r)
                for r in json.loads((target / "games.json").read_text())
            ]
            if sorted(r.deal for r in found) != sorted(group.deals):
                raise ValueError(f"{target} differs from the frozen schedule")
            rows.extend(found)
        elif (target / "failed.json").exists():
            error = json.loads((target / "failed.json").read_text())["error"]
            rows.extend(
                GameRow(
                    model_deck=group.model_deck,
                    heuristic_deck=group.heuristic_deck,
                    model_seat=group.model_seat,
                    deal=deal,
                    score=None,
                    valid=False,
                    terminated=False,
                    truncated=False,
                    failure=f"group failed: {error[-200:]}",
                    decisions=0,
                    model_decisions=0,
                    seconds=0.0,
                )
                for deal in group.deals
            )
    return rows


def _atomic(path: Path, value: object) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value))
    partial.replace(path)


# ---------------------------------------------------------------------------
# Summary


class Score(BaseModel):
    score: float | None
    low: float | None
    high: float | None
    valid: int
    invalid: int


class Summary(BaseModel):
    cells: dict[str, dict[str, Score]]
    model_decks: dict[str, Score]
    heuristic_decks: dict[str, Score]
    model_seats: dict[str, Score]
    overall: Score
    games: int
    decisions: int
    model_decisions: int
    seconds: float


def wilson(rows: Sequence[GameRow]) -> Score:
    """Mean score with a 95% Wilson interval, treating a draw as half a win."""
    valid = [r.score for r in rows if r.valid and r.score is not None]
    n = len(valid)
    invalid = len(rows) - n
    if n == 0:
        return Score(score=None, low=None, high=None, valid=0, invalid=invalid)
    p = sum(valid) / n
    z = 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Score(
        score=p,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        valid=n,
        invalid=invalid,
    )


def summarize(rows: Sequence[GameRow]) -> Summary:
    def where(**match: object) -> list[GameRow]:
        return [r for r in rows if all(getattr(r, k) == v for k, v in match.items())]

    return Summary(
        cells={
            x: {y: wilson(where(model_deck=x, heuristic_deck=y)) for y in DECKS}
            for x in DECKS
        },
        model_decks={x: wilson(where(model_deck=x)) for x in DECKS},
        heuristic_decks={y: wilson(where(heuristic_deck=y)) for y in DECKS},
        model_seats={
            "play": wilson(where(model_seat=0)),
            "draw": wilson(where(model_seat=1)),
        },
        overall=wilson(rows),
        games=len(rows),
        decisions=sum(r.decisions for r in rows),
        model_decisions=sum(r.model_decisions for r in rows),
        seconds=sum(r.seconds for r in rows),
    )


def _short(deck: str) -> str:
    return deck.removeprefix("elemental_").upper()


def _cell(score: Score, *, strong: bool = False) -> str:
    if score.score is None:
        return '<td class="na">—</td>'
    shade = round(255 - 90 * score.score)
    text = f"{score.score:.0%}"
    tag = "strong" if strong else "span"
    invalid = f" · {score.invalid} invalid" if score.invalid else ""
    return (
        f'<td style="background:rgb({shade},{shade + (255 - shade) // 3},255)" '
        f'title="{score.score:.3f} [{score.low:.3f}, {score.high:.3f}], '
        f'{score.valid} valid{invalid}"><{tag}>{text}</{tag}>'
        f"<small>{score.low:.0%}–{score.high:.0%}</small></td>"
    )


def grid_table(summary: Summary, caption: str) -> str:
    """Rows are the model's deck, columns the heuristic's deck; margins aggregate."""
    head = "".join(f"<th>{_short(y)}</th>" for y in DECKS)
    body = "".join(
        f"<tr><th>{_short(x)}</th>"
        + "".join(_cell(summary.cells[x][y]) for y in DECKS)
        + _cell(summary.model_decks[x], strong=True)
        + "</tr>"
        for x in DECKS
    )
    foot = (
        "<tr><th>All</th>"
        + "".join(_cell(summary.heuristic_decks[y], strong=True) for y in DECKS)
        + _cell(summary.overall, strong=True)
        + "</tr>"
    )
    return (
        f'<table class="grid"><caption>{escape(caption)}</caption>'
        f"<tr><th>Model ↓ / Heuristic →</th>{head}<th>All</th></tr>{body}{foot}</table>"
    )


def main() -> None:
    request = Request.model_validate_json(sys.stdin.read())
    rows = play_group(request)
    print(json.dumps([r.model_dump(mode="json") for r in rows]))


if __name__ == "__main__":
    main()
