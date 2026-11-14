"""How one Elemental evaluation match is played: `play`, for `experiments/elemental.py`.

A match is one row of a grid: player 0's deck against every opposing deck and
seat. `Conditions` is its config, built by `rows` from the frozen heuristic
schedule (`experiments.elemental_grid`) or the cross-checkpoint schedule
(`experiments.elemental_long`, through `h2h_groups`). Player 0 is the rated (or
newer) checkpoint and player 1 the heuristic (or older checkpoint).

`play` runs each (opposing deck, seat) group in a bounded worker process: the
grid's own worker against the heuristic, this module's `worker` entry for a
checkpoint pair. It returns table `games`, one row per game and player
(`game_rows`), with the card tables and evidence blobs of each game's
replay-checked card evidence. A group that fails or reaches its bound is stored
with invalid rows (score null) and never replayed; Etude fails a match only for
an integrity error (wrong bytes, wrong legs).

Arena evidence lands under `$ETUDE_OUTPUT_DIR/arena` (a launched job's output),
else `.runs/elemental/arena`. `ELEMENTAL_EVAL_COMPUTE` holds an `EvalAllocation`
as JSON for shared batched inference; `ELEMENTAL_EVAL_GROUP_WORKERS` is how many
of a match's groups play at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import TYPE_CHECKING, Literal

from etude.eval.match import environment
from etude.eval.records import CheckpointPlayer, Code, Environment, Match, Player
from etude.eval.run import Result, Row
from pydantic import BaseModel, ConfigDict

from experiments.elemental_grid import (
    GAME_SECONDS,
    GROUP_SECONDS,
    MAX_COMMANDS,
    PACK,
    GameRow,
    Group,
    Request,
    _legs,
    group_cell,
)
from experiments.elemental_long import SEED, H2HGame
from manabot.data import cache_directory
from manabot.eval.card_evaluation import trace_card_tables
from manabot.infra.artifacts import fetch_etude_checkpoint
from manabot.infra.identity import file_sha256
from manabot.sim.execution import EvalAllocation
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.threads import child_environment
import managym

if TYPE_CHECKING:
    from manabot.sim.serving_protocol import InferenceRuntime

ROOT = Path(__file__).resolve().parents[1]
HEURISTIC = "heuristic"
# Inference provenance is versioned independently of player identity.
INFERENCE_CODE = (
    ROOT / "manabot/model",
    ROOT / "manabot/env",
    ROOT / "manabot/sim/match.py",
    ROOT / "manabot/eval/match.py",
    ROOT / "experiments/elemental_grid.py",
)
# An `EvalAllocation` as JSON: when set, groups play through shared batched inference.
COMPUTE = "ELEMENTAL_EVAL_COMPUTE"
# How many of a match's groups play at once (default one, in turn); rows keep
# the groups' order either way.
GROUP_WORKERS = "ELEMENTAL_EVAL_GROUP_WORKERS"


# ---------------------------------------------------------------------------
# Players and matches


def eval_environment() -> Environment:
    return environment(
        "elemental-world",
        files=[
            ROOT / "experiments/elemental_long.py",
            ROOT / "experiments/elemental_grid.py",
        ],
        config={
            "world": managym.WORLD_VERSION,
            "pack": PACK,
            "game_seconds": GAME_SECONDS,
            "max_commands": MAX_COMMANDS,
            "group_seconds": GROUP_SECONDS,
        },
    )


@cache
def heuristic_code() -> Code:
    return Code(name=HEURISTIC, source=heuristic_source_paths())


type Kind = Literal["full", "subset", "pair"]


class Conditions(BaseModel):
    """One row of a grid: player 0's deck against each opposing deck and seat.

    Each group is one (opposing deck, player 0's native seat) cell half of the
    frozen schedule, played by one bounded worker process.
    """

    model_config = ConfigDict(extra="forbid")
    # A heuristic grid (`full` or the declared `subset`) or a checkpoint pair.
    kind: Kind
    deck: str
    groups: tuple[Group, ...]


def h2h_groups(games: Sequence[H2HGame]) -> list[Group]:
    """The cross-checkpoint schedule as grid groups: the newer checkpoint is `model_*`."""
    found: dict[tuple[str, str, int], list[int]] = {}
    for game in games:
        key = (game.newer_deck, game.older_deck, game.newer_seat)
        found.setdefault(key, []).append(game.deal)
    return [
        Group(model_deck=x, heuristic_deck=y, model_seat=seat, deals=tuple(deals))  # type: ignore[arg-type]
        for (x, y, seat), deals in found.items()
    ]


def rows(kind: Kind, found: Sequence[Group]) -> list[Conditions]:
    """The groups as grid rows: one per player-0 deck, each one match's conditions."""
    by_deck: dict[str, list[Group]] = {}
    for group in found:
        by_deck.setdefault(group.model_deck, []).append(group)
    return [
        Conditions(kind=kind, deck=deck, groups=tuple(row))
        for deck, row in by_deck.items()
    ]


# ---------------------------------------------------------------------------
# Playing one match


class LocalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The seated player's name, such as `u000200` or `learner_transitions-512.pt`.
    name: str
    path: Path
    sha256: str


class H2HRequest(BaseModel):
    """A pair group for the worker: the newer checkpoint is arena player A."""

    model_config = ConfigDict(extra="forbid")
    newer: LocalCheckpoint
    older: LocalCheckpoint
    group: Group
    output: Path
    card_evidence: Path | None = None
    compute: EvalAllocation | None = None


def _local(seated: Player) -> LocalCheckpoint:
    """A model's verified checkpoint bytes, downloaded once per digest."""
    if not isinstance(seated, CheckpointPlayer):
        raise ValueError(f"{seated.name} is not a checkpoint with a digest")
    reference = seated.checkpoint.artifact
    if reference.sha256 is None:
        raise ValueError(f"{seated.name} is not a checkpoint with a digest")
    # A local file is verified in place; a remote object is cached by digest.
    path = cache_directory() / reference.sha256
    if (
        reference.uri.startswith("file:")
        or not path.exists()
        or file_sha256(path) != reference.sha256
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path = fetch_etude_checkpoint(reference, path)
    return LocalCheckpoint(name=seated.name, path=path, sha256=reference.sha256)


def evidence_directory() -> Path:
    """Where `play` keeps arena evidence: a launched attempt's output
    (`$ETUDE_OUTPUT_DIR/arena`), else `.runs/elemental/arena`."""
    if os.environ.get("ETUDE_OUTPUT_DIR"):
        return Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena"
    return ROOT / ".runs/elemental/arena"


def play(match: Match[Conditions]) -> Result:
    """Play one grid row, a bounded worker process per group; see the module docstring.

    Workers hand each played game's card evidence over through a scratch
    folder; its bytes leave as blobs of the match beside the card tables. A
    failed group has none.
    """
    conditions = Conditions.model_validate(match.config)
    newer = _local(match.players[0])
    older = None if conditions.kind != "pair" else _local(match.players[1])
    # The registered player IDs the workers seat, in Etude row order.
    participants = ("model", "heuristic") if older is None else ("newer", "older")
    rows: list[GameRow] = []
    tables: dict[str, list[Row]] = {}
    blobs: list[bytes] = []
    if any(group.model_deck != conditions.deck for group in conditions.groups):
        raise ValueError("a group plays another deck than its match")
    concurrent = int(os.environ.get(GROUP_WORKERS, "1"))
    with TemporaryDirectory() as scratch:
        cards = Path(scratch)

        def run(group: Group) -> tuple[list[GameRow], dict[str, object] | None]:
            output = evidence_directory() / match.key / f"attempt-{time.time_ns()}"
            return _play_group(group, newer, older, output, cards)

        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            played_groups = list(pool.map(run, conditions.groups))
        for played, trace in played_groups:
            rows += played
            if trace is not None:
                found = trace_card_tables(cards, trace, participants)
                blobs += found.blobs
                for name, card_rows in found.tables.items():
                    tables.setdefault(name, []).extend(card_rows)
    return Result(
        tables={"games": game_rows(rows, conditions.kind), **tables}, blobs=blobs
    )


def _play_group(
    group: Group,
    newer: LocalCheckpoint,
    older: LocalCheckpoint | None,
    output: Path,
    cards: Path,
) -> tuple[list[GameRow], dict[str, object] | None]:
    """A group's rows and arena trace from its worker; invalid rows and no trace
    when it fails or times out."""
    found = os.environ.get(COMPUTE)
    compute = None if not found else EvalAllocation.model_validate_json(found)
    if older is None:
        command = ["experiments.elemental_grid"]
        request: BaseModel = Request(
            checkpoint=newer.path,
            sha256=newer.sha256,
            training_seed=SEED,
            group=group,
            output=output,
            card_evidence=cards,
            compute=compute,
        )
    else:
        command = ["experiments.elemental_play", "worker"]
        request = H2HRequest(
            newer=newer,
            older=older,
            group=group,
            output=output,
            card_evidence=cards,
            compute=compute,
        )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", *command],
            cwd=ROOT,
            input=request.model_dump_json(),
            text=True,
            capture_output=True,
            timeout=GROUP_SECONDS,
            env=child_environment(),
        )
        failure = None if completed.returncode == 0 else completed.stderr[-4000:]
    except subprocess.TimeoutExpired:
        completed, failure = None, "group wall-time bound reached"
    if failure is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "failed.txt").write_text(failure)
        return [_invalid(group, deal, failure) for deal in group.deals], None
    assert completed is not None
    rows = [GameRow.model_validate(r) for r in json.loads(completed.stdout)]
    if sorted(r.deal for r in rows) != sorted(group.deals):
        raise ValueError("worker rows differ from the scheduled deals")
    return rows, json.loads((output / "trace.json").read_text())


def _invalid(group: Group, deal: int, failure: str) -> GameRow:
    return GameRow(
        model_deck=group.model_deck,
        heuristic_deck=group.heuristic_deck,
        model_seat=group.model_seat,
        deal=deal,
        score=None,
        valid=False,
        terminated=False,
        truncated=False,
        failure=f"group failed: {failure[-200:]}",
        decisions=0,
        model_decisions=0,
        seconds=0.0,
    )


def game_rows(rows: Sequence[GameRow], kind: Kind) -> list[Row]:
    """Each game once per player; player 0's row is the rated checkpoint's.

    `unit` is the match's kind: an endpoint on a subset cadence update plays
    both heuristic grids, and the report keeps the full one.
    `seat` indexes the Etude participants; `game_seat` is the native seat
    (0 is on the play). `score` counts a draw as half and is null for an invalid
    game, so Etude's `compare` and the report count only valid games.
    """
    found: list[Row] = []
    for row in rows:
        score = row.score if row.valid else None
        sides = (
            (row.model_deck, row.heuristic_deck, row.model_seat, score),
            (
                row.heuristic_deck,
                row.model_deck,
                1 - row.model_seat,
                None if score is None else 1.0 - score,
            ),
        )
        for seat, (deck, opponent_deck, game_seat, player_score) in enumerate(sides):
            found.append(
                {
                    "seat": seat,
                    "unit": kind,
                    "deal": row.deal,
                    "deck": deck,
                    "opponent_deck": opponent_deck,
                    "game_seat": game_seat,
                    "score": player_score,
                    "valid": row.valid,
                    "terminated": row.terminated,
                    "truncated": row.truncated,
                    "failure": row.failure,
                    "decisions": row.decisions,
                    "seconds": row.seconds,
                }
            )
    return found


def play_h2h_group(
    request: H2HRequest, *, inference: InferenceRuntime | None = None
) -> list[GameRow]:
    """Newer checkpoint as arena player A, through the grid's deck/seat legs.

    An optional shared allocation serves both saved policies. Each registration
    uses its own checkpoint observation schema.
    Rows use the grid's `GameRow`: `model_*` is the newer checkpoint and
    `heuristic_deck` the older checkpoint's deck.
    """
    import torch

    from manabot.backends.torch.checkpoint import load_checkpoint
    from manabot.eval.models import EvaluationKey
    from manabot.infra.identity import canonical_sha256
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.sim.matchup import AuthoredMatchup
    from manabot.sim.registration import PlayerRegistration

    torch.set_num_threads(1)
    group = request.group
    role_a, role_b, leg = _legs(group)
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

    def register(
        name: str, checkpoint: LocalCheckpoint, role: str
    ) -> PlayerRegistration:
        if file_sha256(checkpoint.path) != checkpoint.sha256:
            raise ValueError("checkpoint byte identity differs")
        model, space = load_checkpoint(str(checkpoint.path))
        policy_fingerprints = runtime_fingerprints(
            match_config=matchup.match(), observation_space=space
        )
        return PlayerRegistration(
            **{
                **common,
                "observation_abi_sha256": policy_fingerprints["observation_abi_sha256"],
            },
            player_id=name,
            display_name=checkpoint.name,
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
            checkpoint_bytes=checkpoint.path.stat().st_size,
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_seed=SEED,
            artifact_id=f"mtg183/{checkpoint.sha256}",
        )

    newer = register("newer", request.newer, "challenger")
    older = register("older", request.older, "anchor")
    key = EvaluationKey(
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
    rows, trace, replay = group_cell(request.compute)(
        key=key,
        player_a=newer,
        player_b=older,
        deal_seeds=group.deals,
        out_dir=request.output,
        checkpoint_paths={
            "newer": str(request.newer.path),
            "older": str(request.older.path),
        },
        game_seconds=GAME_SECONDS,
        max_commands=MAX_COMMANDS,
        authored_matchup=matchup,
        matchup_mode="cross" if leg < 4 else "mirrors",
        legs=(leg,),
        inference=inference,
        card_evidence_dir=request.card_evidence,
    )
    for name, value in (("rows", rows), ("trace", trace), ("replay", replay)):
        (request.output / f"{name}.json").write_text(json.dumps(value, indent=2))
    result = []
    for row in rows:
        seat = row["player_a_seat"]
        decks = row["seat_decks"]
        if (
            decks[seat] != group.model_deck
            or decks[1 - seat] != group.heuristic_deck
            or seat != group.model_seat
        ):
            raise ValueError("arena leg did not assign the scheduled decks and seat")
        result.append(
            GameRow(
                model_deck=group.model_deck,
                heuristic_deck=group.heuristic_deck,
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
                model_decisions=row["latency"]["newer"]["count"],
                seconds=row["game_seconds"],
            )
        )
    return result


def main() -> None:
    """The pair worker: an `H2HRequest` on stdin, its `GameRow`s on stdout."""
    if sys.argv[1:] != ["worker"]:
        raise SystemExit("usage: python -m experiments.elemental_play worker")
    request = H2HRequest.model_validate_json(sys.stdin.read())
    rows = play_h2h_group(request)
    print(json.dumps([r.model_dump(mode="json") for r in rows]))


if __name__ == "__main__":
    main()
