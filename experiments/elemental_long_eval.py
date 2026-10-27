"""MTG-183's evaluation as one Etude eval run: heuristic grids and the pairing graph.

Jack Heart's direction (2026-10-24): evaluate on Etude eval, tables in R2,
anchored on heuristic v2. The design is frozen in `experiments.elemental_long`
and unchanged; only where it is played and stored moved.

Checkpoints come from two places, merged by absolute update:

- windows w1/w2: `experiments/protocols/elemental-183-windows.json`, the
  retired cohort follower's digest-checked exports, copied to R2 at the keys
  the old S3 layout used (`jobs/<job>/runtime/artifacts/sha256/<sha256>`);
- segments w3, w4, …: each one-rental Etude run's tracked checkpoints and its
  `state.json` in the tracking bucket (`etude.train.read`). Reading the bucket,
  not the launcher, lets this run on a machine other than the controller. The
  last named run's end settles the seed's endpoint unless the seed is declared
  to continue (`--continues`), in which case the next segment's run is named on
  restart.

Each admitted checkpoint is an Etude checkpoint agent `uNNNNNN`, keyed by its
window or segment run and update; the heuristic is the code agent `heuristic`,
whose version is the digest of `heuristic_source_paths()`. Under etude-sdk 0.2,
which the running follower uses, an agent's identity was its digest instead, so
v1 and v2 heuristic results never pooled.

`play`, `Conditions` and `game_rows` are also what `experiments/elemental.py`
plays; the follower below remains only until MTG-183's evaluation ends, then
that file's `eval(state)` replaces it.

`units` turns the design into Etude matches, one per row of a heuristic grid
or pair schedule (player 0's deck), ten per unit. Player 0 is the rated (or
newer) checkpoint and player 1 the heuristic (or older checkpoint); conditions
carry the row's (opposing deck, player 0's native seat, deals) groups. `play`
runs each group in the grid's subprocess worker under its wall bound and
returns table `games`: one row per game and player, plus the card tables
(`card_plays`, `card_counts`, `cards`) from each game's replay-checked card
evidence, retained in the eval run's folder. A group that fails or reaches its bound is stored
with invalid rows (score null) and never replayed, as the protocol requires;
Etude fails a match only for an integrity error (wrong bytes, wrong legs).

`follow` holds the eval run `mtg183-eval-cards`, one unit (a checkpoint's grid or one
pair, oldest newer checkpoint first) at a time, delivering every match's tables
to the given storage, and writes `<directory>/state.json` (`EvalState`): the
admitted checkpoints, the final segment's outcome and the endpoint. The report
reads that state and the tables (`etude.eval.query.statsdb`). It never rents,
cancels or changes training, and yields while any MTG-181 experiment runs on this
machine. It runs as an Etude job on the Mac mini (`experiments/protocols/elemental-183-eval-job.json`).

    uv run --extra artifacts --extra evaluation python -m experiments.elemental_long_eval \
        --directory ~/.manabot/mtg183-eval-cards --until <unix-seconds> \
        --windows experiments/protocols/elemental-183-windows.json \
        --storage r2://manabot/evaluations --etude-source <tracking storage> \
        --run <w3 run> --run <w4 run> [--continues]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from etude.eval.match import Row, Tables, agent, environment, evaluate
from etude.eval.records import (
    Agent,
    AgentKey,
    CheckpointKey,
    Environment,
    Match,
    Version,
    version_of,
)
from etude.run.records import ArtifactRef
from etude.run.storage import Storage, run_directory, storage_from_uri
from etude.train.read import read
from etude.train.records import CheckpointRecord
from pydantic import BaseModel, ConfigDict, JsonValue

from experiments.elemental_grid import (
    GAME_SECONDS,
    GROUP_SECONDS,
    MAX_COMMANDS,
    PACK,
    SCHEDULE_SHA256,
    WORKERS,
    GameRow,
    Group,
    Request,
    _legs,
    group_cell,
    groups,
    schedule_sha256,
)
from experiments.elemental_long import (
    H2H_SCHEDULE_SHA256,
    SEED,
    STAGE,
    H2HGame,
    Pair,
    h2h_schedule,
    h2h_schedule_sha256,
    heuristic_games,
    pair_design,
    rated_updates,
)
from manabot.data import cache_directory
from manabot.eval.card_evaluation import trace_card_tables
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.model.checkpoint import fetch_etude_checkpoint
from manabot.sim.execution import EvalCompute
from manabot.sim.heuristic import heuristic_source_paths
import managym

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mtg183-elemental-eval"
# The run that plays the design with card data (amendment 2026-10-25, card data).
# `mtg183-eval` played its first units without it on etude-sdk 0.2.2, whose
# match keys this code does not reproduce, so this run replays them.
RUN = "mtg183-eval-cards"
HEURISTIC = "heuristic"
# The code that turns a checkpoint's bytes into play; a change makes new agents.
INFERENCE_CODE = (
    ROOT / "manabot/model",
    ROOT / "manabot/env",
    ROOT / "manabot/sim/match.py",
    ROOT / "manabot/eval/match.py",
    ROOT / "experiments/elemental_grid.py",
)
# Any experiment process from MTG-181's checkout (its follower, its later
# evaluation scripts and their grid workers); this one yields the CPU until none
# runs. Matching the checkout, not a script name, survives MTG-181 switching scripts.
YIELD_TO = r"manabot\.mtg-181-trial-two-seat/.* -m experiments\."
# Where `play` writes arena evidence; `follow` sets it for its worker processes.
EVIDENCE = "MTG183_EVAL_EVIDENCE"
# The eval run's folder, where workers retain card evidence; `follow` sets it.
CARDS = "MTG183_EVAL_CARDS"
# An `EvalCompute` as JSON: when set, groups play through shared batched inference.
COMPUTE = "MTG183_EVAL_COMPUTE"
# How many of a match's groups play at once (default one, in turn); rows keep
# the groups' order either way.
GROUP_WORKERS = "MTG183_EVAL_GROUP_WORKERS"


def player(updates: int) -> str:
    return f"u{updates:06d}"


# ---------------------------------------------------------------------------
# Checkpoints and state


class Checkpoint(BaseModel):
    updates: int
    # The window's job (w1/w2) or the segment's Etude run (w3, w4, …).
    run: str
    artifact: ArtifactRef
    coordinates: dict[str, JsonValue]


class EvalState(BaseModel):
    heuristic_schedule_sha256: str = SCHEDULE_SHA256
    h2h_schedule_sha256: str = H2H_SCHEDULE_SHA256
    checkpoints: dict[int, Checkpoint] = {}
    # The final segment's tracked end, "finished" or "failed"; None while it is
    # live or the seed continues. Either settles the seed.
    segment_outcome: str | None = None
    endpoint: int | None = None
    error: str | None = None
    observed_at: float = 0.0
    done: bool = False


def window_checkpoints(path: Path) -> list[Checkpoint]:
    """w1/w2 exports as the retired cohort follower admitted them."""
    return [Checkpoint.model_validate(entry) for entry in json.loads(path.read_bytes())]


@dataclass(frozen=True)
class Observed:
    """What a segment's tracking folder shows: its exports and how it ended."""

    checkpoints: list[CheckpointRecord]
    # The latest attempt's `state.json` ("finished" or "failed"); None while it
    # runs, or when it never finished tracking (killed, or its machine lost).
    ended: str | None


@dataclass(frozen=True)
class SegmentSource:
    """One segment's Etude run, observed through its tracking bucket."""

    run: str
    observe: Callable[[], Observed]


def segment_source(source: str, run: str) -> SegmentSource:
    # `read` takes a bare string for a local directory, so name the bucket's storage.
    tracking = storage_from_uri(source)

    def observe() -> Observed:
        found = read(tracking, STAGE, run)
        return Observed(
            checkpoints=list(found.checkpoints),
            ended=found.state.state if found.state is not None else None,
        )

    return SegmentSource(run=run, observe=observe)


def admit(
    state: EvalState,
    windows: Sequence[Checkpoint],
    segments: Sequence[SegmentSource],
    *,
    continues: bool = False,
) -> None:
    """Record every known export and whether the seed's last segment has ended.

    Windows come first, then segments in launch order: each segment's initial
    export repeats its parent's endpoint and is skipped. A segment trained if
    it exported past everything before it; one that ended without training is
    an error and settles nothing, since its relaunch is a new run, named on
    restart. A run whose machine is lost never records an end, so it also
    waits for that restart. Earlier segments ending is the chain's normal
    course. The last segment's end settles the seed at its last tracked export
    (even short of its target, with its end kept in `segment_outcome` for the
    report) unless `continues` declares a successor.
    """
    for checkpoint in windows:
        state.checkpoints.setdefault(checkpoint.updates, checkpoint)
    state.error = None
    parent = max((checkpoint.updates for checkpoint in windows), default=-1)
    ended: str | None = None
    for segment in segments:
        try:
            observed = segment.observe()
        except Exception as error:
            state.error = f"{segment.run}: {type(error).__name__}: {error}"
            return
        ended, reached = observed.ended, -1
        for record in observed.checkpoints:
            updates = int(record.at["updates"])
            reached = max(reached, updates)
            if updates not in state.checkpoints:
                state.checkpoints[updates] = Checkpoint(
                    updates=updates,
                    run=segment.run,
                    artifact=record.artifact,
                    coordinates=dict(record.at),
                )
        if ended is not None and reached <= parent:
            state.error = f"{segment.run}: ended {ended} without training"
            return
        parent = max(parent, reached)
    if ended is not None and not continues:
        state.segment_outcome = ended
    if state.segment_outcome and state.checkpoints and state.endpoint is None:
        state.endpoint = max(state.checkpoints)


# ---------------------------------------------------------------------------
# Agents and matches


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
def _inference_version() -> Version:
    return version_of(INFERENCE_CODE)


def checkpoint_agent(checkpoint: Checkpoint) -> Agent:
    """As `etude.eval.match.agent` makes one, with the code's version computed once.

    Keyed by its window or segment run and update (`RUN/checkpoints/uNNNNNN`),
    so where its bytes lie is not part of what it is.
    """
    return Agent(
        key=AgentKey(
            checkpoint=CheckpointKey(
                run=checkpoint.run, name=player(checkpoint.updates)
            )
        ),
        name=player(checkpoint.updates),
        artifact=checkpoint.artifact,
        version=_inference_version(),
    )


@cache
def heuristic_agent() -> Agent:
    return agent(HEURISTIC, code=heuristic_source_paths())


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


@dataclass(frozen=True)
class Unit:
    """A checkpoint's heuristic grid, or one pair; the matches it plays."""

    key: str
    newer: int
    # The pair's older checkpoint; None for a heuristic grid.
    older: int | None
    matches: tuple[Match, ...]


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


def _matches(
    kind: Kind,
    agents: tuple[Agent, Agent],
    found: Sequence[Group],
) -> tuple[Match, ...]:
    return tuple(
        Match(
            agents=agents,
            conditions=row.model_dump(mode="json"),
            games=sum(len(group.deals) for group in row.groups),
        )
        for row in rows(kind, found)
    )


def units(state: EvalState) -> list[Unit]:
    """The design over admitted rated checkpoints, oldest newer checkpoint first."""
    latest = max(state.checkpoints, default=0)
    rated = tuple(
        u for u in rated_updates(latest, state.endpoint) if u in state.checkpoints
    )
    agents = {u: checkpoint_agent(state.checkpoints[u]) for u in rated}
    anchor = heuristic_agent()
    found = []
    for updates in rated:
        grid, games = heuristic_games(updates, state.endpoint)
        found.append(
            Unit(
                key=f"heuristic-{updates:06d}-{grid}",
                newer=updates,
                older=None,
                matches=_matches(grid, (agents[updates], anchor), groups(games)),
            )
        )
    pairs = h2h_groups(h2h_schedule())
    design = pair_design(rated, state.endpoint)
    for pair in [*design, *halving_pairs(rated, design)]:
        found.append(
            Unit(
                key=f"pair-{pair.newer:06d}-v-{pair.older:06d}",
                newer=pair.newer,
                older=pair.older,
                matches=_matches(
                    "pair", (agents[pair.newer], agents[pair.older]), pairs
                ),
            )
        )
    # Stable: a checkpoint's grid before its pairs, nearer older checkpoints first.
    return sorted(
        found,
        key=lambda u: (u.newer, u.older is not None, -(u.older or 0)),
    )


def halving_pairs(rated: tuple[int, ...], design: Sequence[Pair]) -> list[Pair]:
    """Each rated checkpoint against the one at exactly half its updates.

    Jack Heart's direction (2026-10-27): the score of `u` against `u/2` traces
    what the second half of training gains at every scale. These pairs extend
    MTG-183's played design only; pairs it already holds keep their role, and
    the shared `pair_design` (MTG-181's too) is unchanged.
    """
    held = {(p.newer, p.older) for p in design}
    return [
        Pair(newer, newer // 2, "halving")
        for newer in rated
        if newer > 0 and newer % 2 == 0 and newer // 2 in rated
        and (newer, newer // 2) not in held
    ]


# ---------------------------------------------------------------------------
# Playing one match


class LocalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The seated agent's name, such as `u000200` or `learner_transitions-512.pt`.
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
    compute: EvalCompute | None = None


def _local(seated: Agent) -> LocalCheckpoint:
    """A checkpoint agent's verified bytes, downloaded once per digest."""
    reference = seated.artifact
    if reference is None or reference.sha256 is None:
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
    """Where `play` keeps arena evidence: `$MTG183_EVAL_EVIDENCE`, else the
    launched attempt's output (`$ETUDE_OUTPUT_DIR/arena`), else `.runs/elemental/arena`."""
    if os.environ.get(EVIDENCE):
        return Path(os.environ[EVIDENCE])
    if os.environ.get("ETUDE_OUTPUT_DIR"):
        return Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena"
    return ROOT / ".runs/elemental/arena"


def play(match: Match) -> Tables:
    """Play one grid row, a bounded worker process per group; see the module docstring.

    Each played group's card evidence lands in the eval run's folder (`CARDS`),
    and its card tables are delivered beside `games`. A failed group has none.
    """
    conditions = Conditions.model_validate(match.conditions)
    newer = _local(match.agents[0])
    older = None if conditions.kind != "pair" else _local(match.agents[1])
    # The registered player IDs the workers seat, in Etude row order.
    participants = ("model", "heuristic") if older is None else ("newer", "older")
    cards = Path(os.environ[CARDS])
    rows: list[GameRow] = []
    tables: dict[str, list[Row]] = {}
    if any(group.model_deck != conditions.deck for group in conditions.groups):
        raise ValueError("a group plays another deck than its match")

    def run(group: Group) -> tuple[list[GameRow], dict[str, object] | None]:
        output = evidence_directory() / match.key / f"attempt-{time.time_ns()}"
        return _play_group(group, newer, older, output, cards)

    concurrent = int(os.environ.get(GROUP_WORKERS, "1"))
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        played_groups = list(pool.map(run, conditions.groups))
    for played, trace in played_groups:
        rows += played
        if trace is not None:
            for name, found in trace_card_tables(cards, trace, participants).items():
                tables.setdefault(name, []).extend(found)
    return {"games": game_rows(rows, conditions.kind), **tables}


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
    compute = None if not found else EvalCompute.model_validate_json(found)
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
        command = ["experiments.elemental_long_eval", "worker"]
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
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
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
    `seat` is the player's native seat (0 is on the play); `score` counts a
    draw as half and is null for an invalid game, so Etude's `compare` and the
    report count only valid games.
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
        for player, (deck, opponent_deck, seat, player_score) in enumerate(sides):
            found.append(
                {
                    "player": player,
                    "unit": kind,
                    "deal": row.deal,
                    "deck": deck,
                    "opponent_deck": opponent_deck,
                    "seat": seat,
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


def play_h2h_group(request: H2HRequest) -> list[GameRow]:
    """Newer checkpoint as arena player A, through the grid's deck/seat legs.

    Rows use the grid's `GameRow`: `model_*` is the newer checkpoint and
    `heuristic_deck` the older checkpoint's deck.
    """
    import torch

    from manabot.eval.models import EvaluationKey
    from manabot.infra.identity import canonical_sha256
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.model.checkpoint import load_checkpoint
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
        model, _ = load_checkpoint(str(checkpoint.path))
        return PlayerRegistration(
            **common,
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


# ---------------------------------------------------------------------------
# Following training


def done_keys(directory: Path) -> set[str]:
    """Match keys this eval run has stored as done (`matches/KEY.json`)."""
    _, root = run_directory(directory, EXPERIMENT, RUN)
    folder = root / "matches"
    if not folder.is_dir():
        return set()
    return {
        path.name.removesuffix(".json")
        for path in folder.glob("*.json")
        if not path.name.endswith(".failed.json")
    }


def _yielding() -> bool:
    found = subprocess.run(["pgrep", "-f", YIELD_TO], capture_output=True, text=True)
    return bool(found.stdout.strip())


def follow(
    directory: Path,
    until: float,
    *,
    windows: Sequence[Checkpoint],
    segments: Sequence[SegmentSource],
    continues: bool = False,
    storages: Sequence[Storage],
    play_match: Callable[[Match], Tables] = play,
    workers: int = WORKERS,
    poll_seconds: float = 600,
    yielding: Callable[[], bool] = _yielding,
) -> EvalState:
    """Admit exports and play pending units until the endpoint's are done or `until`."""
    if schedule_sha256() != SCHEDULE_SHA256 or h2h_schedule_sha256() != (
        H2H_SCHEDULE_SHA256
    ):
        raise ValueError("a game schedule differs from its frozen digest")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[EVIDENCE] = str(directory / "arena")
    track = directory / "track"
    os.environ[CARDS] = str(run_directory(track, EXPERIMENT, RUN)[1].resolve())
    path = directory / "state.json"
    state = (
        EvalState.model_validate_json(path.read_bytes())
        if path.exists()
        else EvalState()
    )
    known = eval_environment()
    polled = 0.0
    while time.time() < until and not state.done:
        if time.time() - polled > poll_seconds:
            polled = time.time()
            try:
                admit(state, windows, segments, continues=continues)
            except Exception as error:
                state.error = f"admission: {type(error).__name__}: {error}"
            state.observed_at = time.time()
            atomic_json(path, state.model_dump(mode="json"))
        stored = done_keys(track)
        pending = [
            u for u in units(state) if any(m.key not in stored for m in u.matches)
        ]
        if pending and not yielding():
            evaluate(
                experiment=EXPERIMENT,
                environment=known,
                matches=pending[0].matches,
                play=play_match,
                workers=workers,
                run=RUN,
                directory=track,
                storages=storages,
            )
            continue
        state.done = state.endpoint is not None and not pending
        if not state.done:
            time.sleep(min(60.0, poll_seconds))
    atomic_json(path, state.model_dump(mode="json"))
    return state


def main() -> None:
    if sys.argv[1:] == ["worker"]:
        request = H2HRequest.model_validate_json(sys.stdin.read())
        rows = play_h2h_group(request)
        print(json.dumps([r.model_dump(mode="json") for r in rows]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--until", type=float, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--etude-source")
    # Segment runs in launch order; repeat for each segment.
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument(
        "--continues",
        action="store_true",
        help="another segment follows the last --run; its end does not settle the seed",
    )
    args = parser.parse_args()
    if (args.etude_source is None) != (not args.run):
        parser.error("--etude-source and --run go together")
    # Imported by name so spawned workers unpickle `play` from this module, not __main__.
    from experiments import elemental_long_eval as this

    this.follow(
        args.directory.expanduser().resolve(),
        args.until,
        windows=this.window_checkpoints(args.windows.expanduser()),
        segments=[this.segment_source(args.etude_source, run) for run in args.run],
        continues=args.continues,
        storages=[storage_from_uri(args.storage)],
        play_match=this.play,
    )


if __name__ == "__main__":
    main()
