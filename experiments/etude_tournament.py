"""Play manabot agents in Etude eval runs: one Etude match is one arena deal.

`play(match)` is the function Etude's spawned workers import by name. It seats
`match.agents[0]` as arena player A and `[1]` as B and plays the deal
`match.conditions["deal"]` over all eight legs of the w6 Allies/Lessons suite
(`manabot.arena.match.play_cell`, `matchup_mode="all"`): adjacent legs swap
seats, legs 0-3 cross the decks and 4-7 are mirrors. Seat swaps therefore live
here, as Etude asks, and `match.games` must be 8.

Tables returned to Etude:

- `plays`, one row per game and Etude player (0/1 is the agent's slot in the
  Etude match, not its in-game seat): `game` (leg), `score` (1/0.5/0, null when
  a game failed without blaming a player), `won`, `seat` (in-game seat;
  seat 0 is on the play), `on_play`, `deck`, `opponent_deck`, `decisions`.
- `games`, one row per game in group `all`: `deal`, `termination_reason`,
  `game_seconds`, `decisions`, `winner` (Etude player or null).

An agent's config is `PlayerConfig`. A checkpoint agent's bytes are found by
its artifact digest in `CHECKPOINTS/index.json`, which also supplies the
training seed the arena registration requires. Code agents hash the files
that define their behaviour, and that digest is the registration's source
identity. Command traces stay under `TRACES/<match key>/`.

Identity files are named relative to the repository root, and commands run
from it: the launcher's source archive has no `.git`, and Etude names a file
outside a git checkout by the path it was given. Relative paths give the same
agent and environment identities, and so the same games, on every machine.

`main` holds the round robin, the benchmark, and prints standings and curves:

    uv run --no-sync python -m experiments.etude_tournament round-robin --deals 2

On a rented machine (`experiments/etude_tournament_job.py`), setup runs
`install-managym` and `fetch-checkpoints`, then `hold-share` plays the share of
the round robin its `ETUDE_CONFIG` names into the launched eval run.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Literal
import urllib.request
import zipfile

from etude.eval.match import Tables, agent, environment
from etude.eval.query import connect
from etude.eval.records import (
    Agent,
    Benchmark,
    Environment,
    Match,
    Matchup,
    Model,
    RoundRobin,
)
from etude.eval.tournament import curves, hold, standings
from etude.run.records import ArtifactRef
from etude.run.storage import Storage, storage_from_uri
from pydantic import BaseModel, ConfigDict

from experiments.etude_tournament_job import ShareConfig
from manabot.arena import players
from manabot.arena.heuristic import heuristic_source_paths
from manabot.arena.match import SELECTED_SUITE, play_cell, selected_match
from manabot.arena.models import (
    ArenaKey,
    PlayerRegistration,
    canonical_sha256,
    file_sha256,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.teacher1_evidence import REPO_ROOT, runtime_fingerprints
from managym import WORLD_VERSION

RUNS = REPO_ROOT / ".runs-etude-tournament"
CHECKPOINTS = RUNS / "checkpoints"
TRACES = RUNS / "traces"
ETUDE = RUNS / "etude"

# Every leg of the selected suite: two seatings of four deck pairs.
LEGS = 8
# MTG-153's held-out deal family for the GPU 25k endpoints.
DEALS = tuple({"deal": 1_226_102_000 + d} for d in range(25))

type PlayerKind = Literal["checkpoint", "heuristic", "scripted_greedy"]


class PlayerConfig(BaseModel):
    """How an Etude agent plays in the arena; part of the agent's identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    player: PlayerKind
    # Checkpoints only: argmax instead of sampling the policy.
    deterministic: bool = False


class _Indexed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    training_seed: int


# Fixed, so a pair's policy seeds depend only on the pair and the deal.
KEY = ArenaKey(
    world=WORLD_VERSION,  # type: ignore[arg-type]  # validated against the Literal by pydantic
    content_suite=SELECTED_SUITE,
    viewer_boundary="acting-viewer",
    arena_version="etude-tournament-v1",
    rating_model_version="unrated",
    rating_prior_sha256=canonical_sha256({}),
    anchor_cohort_sha256=canonical_sha256({}),
    evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
)


# -----------------------------------------------------------------------------
# Agents and the environment
# -----------------------------------------------------------------------------


def _relative(paths: Sequence[Path]) -> list[Path]:
    """Paths below the repository root, for identities alike with or without `.git`."""
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from the repository root, {REPO_ROOT}")
    return [path.resolve().relative_to(REPO_ROOT) for path in paths]


def code_agent(kind: Literal["heuristic", "scripted_greedy"]) -> Agent:
    """A code player, identified by the files that define its behaviour."""
    files = (
        heuristic_source_paths() if kind == "heuristic" else [Path(players.__file__)]
    )
    return agent(kind, code=_relative(files), config=PlayerConfig(player=kind))


def checkpoint_ref(file: str) -> ArtifactRef:
    path = CHECKPOINTS / file
    return ArtifactRef(
        uri=path.as_uri(), sha256=file_sha256(path), size_bytes=path.stat().st_size
    )


def checkpoint_agent(name: str, file: str) -> Agent:
    return agent(
        name, checkpoint=checkpoint_ref(file), config=PlayerConfig(player="checkpoint")
    )


def world() -> Environment:
    """w6 Allies/Lessons: the engine, its content and the arena's leg layout."""
    return environment(
        SELECTED_SUITE,
        files=_relative(
            [
                REPO_ROOT / "managym" / "src",
                REPO_ROOT / "managym" / "Cargo.lock",
                REPO_ROOT / "content" / "semantic",
                REPO_ROOT / "manabot" / "arena" / "match.py",
                REPO_ROOT / "manabot" / "arena" / "replay.py",
            ]
        ),
        config={"world": WORLD_VERSION},
    )


# -----------------------------------------------------------------------------
# Playing a match
# -----------------------------------------------------------------------------


@cache
def _index() -> dict[str, _Indexed]:
    raw = json.loads((CHECKPOINTS / "index.json").read_text())
    return {sha: _Indexed.model_validate(entry) for sha, entry in raw.items()}


@cache
def _checkpoint(sha256: str) -> tuple[Path, int, int]:
    """The verified file, training seed and parameter count of a checkpoint digest."""
    indexed = _index().get(sha256)
    if indexed is None:
        raise FileNotFoundError(f"checkpoint {sha256} is not in {CHECKPOINTS}")
    path = CHECKPOINTS / indexed.file
    if file_sha256(path) != sha256:
        raise ValueError(f"{path} does not have digest {sha256}")
    model, _ = load_checkpoint_agent(str(path))
    return path, indexed.training_seed, sum(p.numel() for p in model.parameters())


@cache
def _common() -> dict[str, str]:
    runtime = runtime_fingerprints(match_hypers=selected_match())
    return {
        "information_boundary": KEY.viewer_boundary,
        "world": KEY.world,
        "content_suite": KEY.content_suite,
        "observation_abi_sha256": runtime["observation_abi_sha256"],
        "action_abi_sha256": runtime["action_abi_sha256"],
        "matchup_sha256": runtime["matchup_sha256"],
        "player_seed_derivation_id": "arena-pair-deal-player-v1",
    }


def _registration(
    seated: Agent, player_id: str
) -> tuple[PlayerRegistration, Path | None]:
    config = PlayerConfig.model_validate(seated.config)
    common: dict[str, Any] = {
        **_common(),
        "player_id": player_id,
        "display_name": seated.name,
        "compute_class_id": "policy-cpu-one-thread-one-pass",
    }
    if config.player == "checkpoint":
        if seated.kind != "checkpoint":
            raise ValueError(f"{seated.name}: checkpoint config on a code agent")
        path, training_seed, parameters = _checkpoint(seated.artifact)
        registration = PlayerRegistration(
            **common,
            role="challenger",
            runner_kind="checkpoint",
            player_spec={
                "kind": "checkpoint",
                "deterministic": config.deterministic,
                "device": "cpu",
                "batch_size": 1,
                # Legs 4-7 are mirrors.
                "allow_deck_repetition": True,
            },
            checkpoint_sha256=seated.artifact,
            checkpoint_bytes=path.stat().st_size,
            parameter_count=parameters,
            training_seed=training_seed,
            artifact_id=f"etude/{seated.artifact}",
        )
        return registration, path
    if seated.kind != "code" or config.deterministic:
        raise ValueError(f"{seated.name}: {config.player} is a plain code agent")
    registration = PlayerRegistration(
        **common,
        role="anchor",
        runner_kind="code",
        player_spec={"kind": config.player},
        source_sha256=seated.artifact,
    )
    return registration, None


def play(match: Match) -> Tables:
    """Play one deal over all eight legs between the match's two agents."""
    if len(match.agents) != 2 or match.games != LEGS:
        raise ValueError(f"a match is two agents over {LEGS} legs")
    deal = match.conditions.get("deal")
    if type(deal) is not int:
        raise ValueError("conditions need an integer 'deal'")
    # Player ids seed policies; an agent's own identity keeps them stable.
    ids = [
        f"{PlayerConfig.model_validate(a.config).player.replace('_', '-')}-{a.identity[:12]}"
        for a in match.agents
    ]
    if ids[0] == ids[1]:
        ids[1] += "-b"
    (a, path_a), (b, path_b) = (
        _registration(seated, player_id)
        for seated, player_id in zip(match.agents, ids, strict=True)
    )
    paths = {
        registration.player_id: str(path)
        for registration, path in ((a, path_a), (b, path_b))
        if path is not None
    }
    out = TRACES / match.key
    out.mkdir(parents=True, exist_ok=True)
    rows, _, replay = play_cell(
        key=KEY,
        player_a=a,
        player_b=b,
        deal_seeds=(deal,),
        out_dir=out,
        checkpoint_paths=paths,
        matchup_mode="all",
    )
    if not replay["passed"]:
        raise RuntimeError(f"deal {deal}: command replay mismatched")
    plays: list[dict[str, bool | int | float | str | None]] = []
    games: list[dict[str, bool | int | float | str | None]] = []
    for row in rows:
        leg = int(row["leg"])
        seat_a = int(row["player_a_seat"])
        failed = row["failure"] is not None
        score_a = row["score_a"]
        winner = row["winner"]
        for player, (registration, seat) in enumerate(((a, seat_a), (b, 1 - seat_a))):
            score = (
                None if score_a is None else (score_a if player == 0 else 1 - score_a)
            )
            plays.append(
                {
                    "player": player,
                    "game": leg,
                    "score": score,
                    "won": None if failed else winner == seat,
                    "seat": seat,
                    "on_play": seat == 0,
                    "deck": row["seat_decks"][seat],
                    "opponent_deck": row["seat_decks"][1 - seat],
                    "decisions": int(row["latency"][registration.player_id]["count"]),
                }
            )
        games.append(
            {
                "group": "all",
                "game": leg,
                "deal": deal,
                "termination_reason": row["termination_reason"],
                "game_seconds": float(row["game_seconds"]),
                "decisions": int(row["decisions"]),
                "winner": None if winner is None else int(winner != seat_a),
            }
        )
    return {"plays": plays, "games": games}


# -----------------------------------------------------------------------------
# Tournaments
# -----------------------------------------------------------------------------


def _workers() -> int:
    # os.sched_getaffinity does not exist on macOS.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _league(deals: int) -> RoundRobin:
    return RoundRobin(
        name=f"w6-finals-{deals}",
        environment=world(),
        conditions=DEALS[:deals],
        games=LEGS,
    )


def _entrants() -> list[Agent]:
    return [
        checkpoint_agent("gpu-12261", "gpu-14353d64c8d3-final.pt"),
        checkpoint_agent("gpu-12262", "gpu-cb14c42a3338-final.pt"),
        code_agent("heuristic"),
        code_agent("scripted_greedy"),
    ]


def _round_robin(deals: int, workers: int) -> None:
    league = _league(deals)
    started = time.monotonic()
    run = hold(league, enter=_entrants(), play=play, workers=workers, directory=ETUDE)
    print(f"{run.name}: played {run.played}, skipped {run.skipped}, "
          f"{time.monotonic() - started:.0f} s")  # fmt: skip
    print_standings(ETUDE, league.name)


def print_standings(source: Path | Storage, name: str) -> None:
    db = connect(source)
    held = db.sql(
        "select run from runs where tournament = ? order by run", params=[name]
    ).fetchall()
    print(f"{name}: eval runs {[run for (run,) in held]}")
    table = standings(db, name, table="plays", value="score")
    print(f"held {table.held}/{table.scheduled}")
    names = {entry.agent: entry.name for entry in table.entries}
    for entry in table.entries:
        against = "field" if entry.opponent is None else names[entry.opponent]
        print(entry.name, against, f"{entry.value:.3f}", entry.interval,
              entry.rows, entry.matches)  # fmt: skip


def _benchmark(deals: int, workers: int) -> None:
    progress = Benchmark(
        name=f"w6-anchors-{deals}",
        environment=world(),
        matchups=tuple(
            Matchup(
                name=f"model vs {opponent}",
                players=(
                    Model(config=PlayerConfig(player="checkpoint").model_dump()),
                    code_agent(opponent),
                ),
                conditions=DEALS[:deals],
                games=LEGS,
            )
            for opponent in ("heuristic", "scripted_greedy")
        ),
    )
    for file in (
        "gpu-14353d64c8d3-initial.pt",
        "gpu-14353d64c8d3-final.pt",
        "gpu-cb14c42a3338-initial.pt",
        "gpu-cb14c42a3338-final.pt",
    ):
        started = time.monotonic()
        run = hold(
            progress,
            checkpoint=checkpoint_ref(file),
            play=play,
            workers=workers,
            directory=ETUDE,
        )
        print(f"{file}: {run.name} played {run.played}, "
              f"{time.monotonic() - started:.0f} s")  # fmt: skip
    for line in curves(connect(ETUDE), progress.name, table="plays", value="score"):
        print(line.matchup, line.run)
        for point in line.points:
            print(" ", point.checkpoint[:12], point.ordinal, point.clocks,
                  f"{point.value:.3f}", point.interval, point.matches)  # fmt: skip


# -----------------------------------------------------------------------------
# On a launched machine
# -----------------------------------------------------------------------------


def _share_config() -> ShareConfig:
    return ShareConfig.model_validate_json(Path(os.environ["ETUDE_CONFIG"]).read_text())


def _install_managym() -> None:
    """Place the native module from the wheel setup built beside the package."""
    wheels = sorted((REPO_ROOT / "managym" / "target" / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one managym wheel, found {wheels}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        (native,) = [n for n in wheel.namelist() if "/_managym." in n]
        with (
            wheel.open(native) as source,
            open(REPO_ROOT / "managym" / Path(native).name, "wb") as target,
        ):
            shutil.copyfileobj(source, target)


def _fetch_checkpoints() -> None:
    """Download each checkpoint the config names and write the index; refuse a wrong digest."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    index: dict[str, _Indexed] = {}
    for sha, download in _share_config().checkpoints.items():
        path = CHECKPOINTS / download.file
        partial = path.with_name(path.name + ".partial")
        with (
            urllib.request.urlopen(download.url) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        found = hashlib.sha256(partial.read_bytes()).hexdigest()
        if found != sha:
            raise ValueError(f"{download.file}: digest {found}, expected {sha}")
        partial.replace(path)
        index[sha] = _Indexed(file=download.file, training_seed=download.training_seed)
    (CHECKPOINTS / "index.json").write_text(
        json.dumps({sha: entry.model_dump() for sha, entry in index.items()})
    )


def _hold_share() -> None:
    """Play this run's share; under `etude launch` the eval run is the launched run."""
    config = _share_config()
    started = time.monotonic()
    run = hold(
        _league(config.deals),
        enter=_entrants(),
        play=play,
        workers=_workers(),
        share=config.share,
    )
    print(f"{run.name}: share {config.share}, played {run.played}, "
          f"{_workers()} workers, {time.monotonic() - started:.0f} s")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=(
            "round-robin",
            "benchmark",
            "standings",
            "install-managym",
            "fetch-checkpoints",
            "hold-share",
        ),
    )
    parser.add_argument("--deals", type=int, default=2)
    parser.add_argument("--workers", type=int, default=_workers())
    # standings: a directory or s3://BUCKET/PREFIX holding the eval runs.
    parser.add_argument("--source", default=str(ETUDE))
    args = parser.parse_args()
    if args.command == "round-robin":
        _round_robin(args.deals, args.workers)
    elif args.command == "benchmark":
        _benchmark(args.deals, args.workers)
    elif args.command == "standings":
        print_standings(storage_from_uri(args.source), f"w6-finals-{args.deals}")
    elif args.command == "install-managym":
        _install_managym()
    elif args.command == "fetch-checkpoints":
        _fetch_checkpoints()
    else:
        _hold_share()


if __name__ == "__main__":
    main()
