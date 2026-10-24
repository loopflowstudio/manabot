"""Evaluate MTG-183's rated checkpoints on local CPU while the seed trains.

Windows w1 and w2 trained under the retired cohort controller; the follower
that watched them admitted their exports into `<directory>/ledger.json`, and
those entries are kept as they are. The seed continues in exactly one Etude
segment, w3 (`experiments.elemental_long.segment_job`): the follower reads that
run's tracked checkpoint records from Etude storage, fetches digest-checked bytes
and admits the rated checkpoints of `experiments.elemental_long.rated_updates`.
It then plays two kinds of evaluation unit, oldest newer-checkpoint first:

- `heuristic`: a checkpoint's 10×10 grid against the heuristic
  (`experiments.elemental_grid`, full or declared 10-game subset);
- `pair`: a newer checkpoint against an older one on the 10×10 deck grid
  (`h2h_schedule`), from `pair_design`.

The endpoint (the segment's final update) is admitted only once the run is
terminal; its pairs follow. Group results under `<directory>/units/`
are the durable authority: a completed group is reused after a restart, a
failed group counts its games as invalid and is never replayed. The follower
never rents, cancels or changes training, and waits while MTG-181's follower
is still evaluating on this machine.

    uv run --extra artifacts python -m experiments.elemental_long_follow \
        --directory .runs/mtg183/follow --until <unix-seconds> \
        --etude-source <tracking storage URI> --run <etude run>
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from etude.launch.client import status
from etude.launch.records import Status
from etude.train.read import read
from etude.train.records import CheckpointRecord
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import (
    GAME_SECONDS,
    GROUP_SECONDS,
    MAX_COMMANDS,
    PACK,
    SCHEDULE_SHA256,
    WORKERS,
    Group,
    _legs,
    evaluate,
    schedule_sha256,
)
from experiments.elemental_long import (
    H2H_SCHEDULE_SHA256,
    SEED,
    STAGE,
    H2HGame,
    h2h_schedule,
    h2h_schedule_sha256,
    heuristic_games,
    pair_design,
    rated_updates,
)
from manabot.infra.files import atomic_json
from manabot.model.checkpoint import fetch_etude_checkpoint

ROOT = Path(__file__).resolve().parents[1]
# MTG-181's local follower; this one yields the CPU until it exits.
YIELD_TO = "experiments.elemental_two_seat_follow"


class Checkpoint(BaseModel):
    updates: int
    job: str
    training_run: str
    path: str
    sha256: str
    coordinates: dict[str, object]


class Unit(BaseModel):
    kind: Literal["heuristic", "pair"]
    newer: int
    older: int | None = None
    grid: str | None = None
    role: str | None = None
    phase: Literal["queued", "running", "done"] = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    restarts: int = 0

    @property
    def key(self) -> str:
        if self.kind == "heuristic":
            return f"heuristic-{self.newer:06d}-{self.grid}"
        return f"pair-{self.newer:06d}-v-{self.older:06d}"


class Ledger(BaseModel):
    heuristic_schedule_sha256: str
    h2h_schedule_sha256: str
    checkpoints: dict[int, Checkpoint] = {}
    units: dict[str, Unit] = {}
    # The segment's terminal Etude outcome, e.g. "failed (deadline)"; None
    # while the run is live. Any outcome settles the segment.
    segment_outcome: str | None = None
    endpoint: int | None = None
    error: str | None = None
    observed_at: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Cross-checkpoint worker: one (newer deck, older deck, newer seat) group


class H2HGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    newer_deck: str
    older_deck: str
    newer_seat: Literal[0, 1]
    deals: tuple[int, ...]


class H2HRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    newer: Checkpoint
    older: Checkpoint
    group: H2HGroup
    output: Path


class H2HRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    newer_deck: str
    older_deck: str
    newer_seat: Literal[0, 1]
    deal: int
    score: float | None
    valid: bool
    terminated: bool
    truncated: bool
    failure: str | None
    decisions: int
    seconds: float


def h2h_groups(games: list[H2HGame]) -> list[H2HGroup]:
    found: dict[tuple[str, str, int], list[int]] = {}
    for game in games:
        key = (game.newer_deck, game.older_deck, game.newer_seat)
        found.setdefault(key, []).append(game.deal)
    return [
        H2HGroup(newer_deck=x, older_deck=y, newer_seat=seat, deals=tuple(deals))  # type: ignore[arg-type]
        for (x, y, seat), deals in found.items()
    ]


def play_h2h_group(request: H2HRequest) -> list[H2HRow]:
    """Newer checkpoint as arena player A, through the grid's deck/seat legs."""
    import torch

    from manabot.arena.match import play_cell
    from manabot.arena.matchup import AuthoredMatchup
    from manabot.arena.models import (
        ArenaKey,
        PlayerRegistration,
        canonical_sha256,
        file_sha256,
    )
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.model.checkpoint import load_checkpoint
    import managym

    torch.set_num_threads(1)
    group = request.group
    role_a, role_b, leg = _legs(
        Group(
            model_deck=group.newer_deck,
            heuristic_deck=group.older_deck,
            model_seat=group.newer_seat,
            deals=group.deals,
        )
    )
    matchup = AuthoredMatchup(pack_key=PACK, role_a=role_a, role_b=role_b)
    fingerprints = runtime_fingerprints(match_config=matchup.match())
    common = dict(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        information_boundary="acting-viewer",
        observation_abi_sha256=fingerprints["observation_abi_sha256"],
        action_abi_sha256=fingerprints["action_abi_sha256"],
        matchup_sha256=canonical_sha256(matchup.match().model_dump()),
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )

    def register(name: str, checkpoint: Checkpoint, role: str) -> PlayerRegistration:
        path = Path(checkpoint.path)
        if file_sha256(path) != checkpoint.sha256:
            raise ValueError("checkpoint byte identity differs")
        model, _ = load_checkpoint(path)
        return PlayerRegistration(
            **common,
            player_id=name,
            display_name=f"Update {checkpoint.updates}",
            role=role,
            compute_class_id="policy-cpu-one-thread-one-pass",
            runner_kind="checkpoint",
            player_spec={
                "kind": "checkpoint",
                "device": "cpu",
                "batch_size": 1,
                "deterministic": False,
                "allow_deck_repetition": True,
            },
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_bytes=path.stat().st_size,
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_seed=SEED,
            artifact_id=f"mtg183/{checkpoint.sha256}",
        )

    newer = register("newer", request.newer, "challenger")
    older = register("older", request.older, "anchor")
    key = ArenaKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version="mtg183-elemental-h2h-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(older.model_dump()),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
    )
    request.output.mkdir(parents=True, exist_ok=False)
    rows, trace, replay = play_cell(
        key=key,
        player_a=newer,
        player_b=older,
        deal_seeds=group.deals,
        out_dir=request.output,
        checkpoint_paths={"newer": request.newer.path, "older": request.older.path},
        game_seconds=GAME_SECONDS,
        max_commands=MAX_COMMANDS,
        authored_matchup=matchup,
        matchup_mode="cross" if leg < 4 else "mirrors",
        legs=(leg,),
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    result = []
    for row in rows:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        if (
            decks[seat] != group.newer_deck
            or decks[1 - seat] != group.older_deck
            or seat != group.newer_seat
        ):
            raise ValueError("arena leg did not assign the scheduled decks and seat")
        result.append(
            H2HRow(
                newer_deck=group.newer_deck,
                older_deck=group.older_deck,
                newer_seat=seat,
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
                seconds=row["game_seconds"],
            )
        )
    return result


def _group_name(group: H2HGroup) -> str:
    return f"{group.newer_deck}--{group.older_deck}--seat{group.newer_seat}"


def evaluate_pair(
    newer: Checkpoint, older: Checkpoint, directory: Path, *, workers: int = WORKERS
) -> list[H2HRow]:
    """Play one pair's 1,000 games with the grid's durable group semantics."""
    directory.mkdir(parents=True, exist_ok=True)
    groups = h2h_groups(h2h_schedule())
    pending = [
        g
        for g in groups
        if not (directory / _group_name(g) / "games.json").exists()
        and not (directory / _group_name(g) / "failed.json").exists()
    ]

    def run(group: H2HGroup) -> None:
        target = directory / _group_name(group)
        target.mkdir(parents=True, exist_ok=True)
        request = H2HRequest(
            newer=newer,
            older=older,
            group=group,
            output=target / f"attempt-{time.time_ns()}",
        )
        started = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.elemental_long_follow", "worker"],
                cwd=ROOT,
                input=request.model_dump_json(),
                text=True,
                capture_output=True,
                timeout=GROUP_SECONDS,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            error = None if completed.returncode == 0 else completed.stderr[-4000:]
            output = completed.stdout
        except subprocess.TimeoutExpired:
            error, output = "group wall-time bound reached", ""
        if error is None:
            rows = [H2HRow.model_validate(r) for r in json.loads(output)]
            atomic_json(
                target / "games.json", [r.model_dump(mode="json") for r in rows]
            )
        else:
            atomic_json(
                target / "failed.json",
                {"error": error, "started": started, "finished": time.time()},
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, pending))
    return collect_pair(directory)


def collect_pair(directory: Path) -> list[H2HRow]:
    """Retained rows for every scheduled game; failed groups become invalid rows."""
    rows: list[H2HRow] = []
    for group in h2h_groups(h2h_schedule()):
        target = directory / _group_name(group)
        if (target / "games.json").exists():
            found = [
                H2HRow.model_validate(r)
                for r in json.loads((target / "games.json").read_text())
            ]
            if sorted(r.deal for r in found) != sorted(group.deals):
                raise ValueError(f"{target} differs from the frozen schedule")
            rows.extend(found)
        elif (target / "failed.json").exists():
            error = json.loads((target / "failed.json").read_text())["error"]
            rows.extend(
                H2HRow(
                    newer_deck=group.newer_deck,
                    older_deck=group.older_deck,
                    newer_seat=group.newer_seat,
                    deal=deal,
                    score=None,
                    valid=False,
                    terminated=False,
                    truncated=False,
                    failure=f"group failed: {error[-200:]}",
                    decisions=0,
                    seconds=0.0,
                )
                for deal in group.deals
            )
    return rows


# ---------------------------------------------------------------------------
# Admission


@dataclass(frozen=True)
class SegmentSource:
    """The segment's Etude run: its tracked checkpoints and its status."""

    run: str
    checkpoints: Callable[[], list[CheckpointRecord]]
    status: Callable[[], Status]


def segment_source(source: str, run: str) -> SegmentSource:
    return SegmentSource(
        run=run,
        checkpoints=lambda: list(read(source, STAGE, run).checkpoints),
        status=lambda: status(run),
    )


def _admit(directory: Path, ledger: Ledger, segment: SegmentSource) -> None:
    """Record the segment's tracked exports and whether its run has ended.

    Any terminal Etude state settles the segment: a run that stops short of
    its target still ends the seed at its last tracked export, and its state
    stays in `ledger.segment_outcome` for the report.
    """
    try:
        state: Status | None = segment.status()
        records = segment.checkpoints()
        ledger.error = None
    except Exception as error:
        state, records = None, []
        ledger.error = f"{segment.run}: {type(error).__name__}"
    cache = directory / "cache" / segment.run
    for record in records:
        updates = int(record.at["updates"])
        # The segment's initial export repeats w2's endpoint.
        if updates in ledger.checkpoints:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        path = fetch_etude_checkpoint(record, cache / f"{updates:06d}.pt")
        assert record.artifact.sha256 is not None
        ledger.checkpoints[updates] = Checkpoint(
            updates=updates,
            job=segment.run,
            training_run=segment.run,
            path=str(path),
            sha256=record.artifact.sha256,
            coordinates=dict(record.at),
        )
    if state is not None and state.done:
        cause = f" ({state.cause})" if state.cause else ""
        ledger.segment_outcome = f"{state.outcome}{cause}"
    if ledger.segment_outcome and ledger.checkpoints and ledger.endpoint is None:
        ledger.endpoint = max(ledger.checkpoints)
    latest = max(ledger.checkpoints, default=0)
    rated = [
        u for u in rated_updates(latest, ledger.endpoint) if u in ledger.checkpoints
    ]
    for updates in rated:
        grid, _ = heuristic_games(updates, ledger.endpoint)
        unit = Unit(kind="heuristic", newer=updates, grid=grid)
        ledger.units.setdefault(unit.key, unit)
    for pair in pair_design(tuple(rated), ledger.endpoint):
        unit = Unit(kind="pair", newer=pair.newer, older=pair.older, role=pair.role)
        ledger.units.setdefault(unit.key, unit)


def _yielding() -> bool:
    found = subprocess.run(["pgrep", "-f", YIELD_TO], capture_output=True, text=True)
    return bool(found.stdout.strip())


def _evaluate(unit: Unit, directory: Path, ledger: Ledger, path: Path) -> None:
    target = directory / "units" / unit.key
    if unit.phase == "running":
        unit.restarts += 1  # completed groups are kept; failed groups never replay
    unit.phase = "running"
    unit.started_at = unit.started_at or time.time()
    atomic_json(path, ledger.model_dump(mode="json"))
    newer = ledger.checkpoints[unit.newer]
    if unit.kind == "heuristic":
        _, games = heuristic_games(unit.newer, ledger.endpoint)
        rows: list[BaseModel] = list(
            evaluate(
                Path(newer.path),
                newer.sha256,
                SEED,
                target / "groups",
                games=games,
            )
        )
    else:
        assert unit.older is not None
        rows = list(
            evaluate_pair(newer, ledger.checkpoints[unit.older], target / "groups")
        )
    atomic_json(target / "rows.json", [r.model_dump(mode="json") for r in rows])
    unit.phase = "done"
    unit.finished_at = time.time()
    atomic_json(path, ledger.model_dump(mode="json"))


def follow(directory: Path, until: float, segment: SegmentSource) -> None:
    if schedule_sha256() != SCHEDULE_SHA256 or h2h_schedule_sha256() != (
        H2H_SCHEDULE_SHA256
    ):
        raise ValueError("a game schedule differs from its frozen digest")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ledger.json"
    with (directory / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = (
            Ledger.model_validate_json(path.read_bytes())
            if path.exists()
            else Ledger(
                heuristic_schedule_sha256=SCHEDULE_SHA256,
                h2h_schedule_sha256=H2H_SCHEDULE_SHA256,
            )
        )
        polled = 0.0
        while time.time() < until and not ledger.done:
            if time.time() - polled > 600:
                polled = time.time()
                try:
                    _admit(directory, ledger, segment)
                except Exception as error:
                    ledger.error = f"admission: {type(error).__name__}: {error}"
                ledger.observed_at = time.time()
                atomic_json(path, ledger.model_dump(mode="json"))
            pending = sorted(
                (u for u in ledger.units.values() if u.phase != "done"),
                key=lambda u: (u.newer, u.kind != "heuristic", -(u.older or 0)),
            )
            if pending and not _yielding():
                _evaluate(pending[0], directory, ledger, path)
                continue
            ledger.done = ledger.endpoint is not None and not pending
            if not ledger.done:
                time.sleep(60)
        atomic_json(path, ledger.model_dump(mode="json"))


def main() -> None:
    if sys.argv[1:] == ["worker"]:
        request = H2HRequest.model_validate_json(sys.stdin.read())
        print(json.dumps([r.model_dump(mode="json") for r in play_h2h_group(request)]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--until", type=float, required=True)
    parser.add_argument("--etude-source", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    follow(
        args.directory.resolve(),
        args.until,
        segment_source(args.etude_source, args.run),
    )


if __name__ == "__main__":
    main()
