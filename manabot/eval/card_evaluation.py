"""Etude scheduling and delivery for replay-derived card statistics.

One match is the existing eight-leg authored arena matrix for one deal. Etude
owns completion, retries and scalar tables. Evidence is stored before table
admission; publication explicitly copies it because SDK 0.2 sync omits it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Literal
from urllib.parse import unquote, urlsplit

from etude.eval.match import EvalRun, Row, Tables, agent
from etude.eval.records import Agent, Environment, Match, RoundRobin
from etude.eval.tournament import hold
from etude.run.declaration import declare
from etude.run.delivery import sync
from etude.run.listing import places
from etude.run.records import ArtifactRef
from etude.run.storage import DirectoryStorage, Storage
from pydantic import Field, model_validator

from manabot.eval.card_metrics import METRIC_VERSION, CardMatchError, reduce_cards
from manabot.eval.card_query import evidence_prefix, evidence_refs
from manabot.eval.card_schedule import (
    RUNTIME_FIELDS,
    CardConditions,
    card_match_players,
)
from manabot.eval.evaluate import (
    validate_player_compatibility,
    validate_registered_player,
)
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.eval.monitor import EvaluationRow
from manabot.infra.artifacts import verify_file
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.sim.card_evidence import CAPTURE_VERSION, EvidenceRef, read_evidence
from manabot.sim.matchup import SELECTED_SUITE, AuthoredMatchup, selected_match
from manabot.sim.registration import PlayerRegistration

EXPERIMENT = "gih-card-evaluation"


class CardEvaluationSpec(StrictModel):
    run: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,199}$")
    key: EvaluationKey
    players: tuple[PlayerRegistration, ...] = Field(min_length=2)
    checkpoints: dict[str, Path] = Field(default_factory=dict)
    deals: tuple[int, ...] = Field(min_length=1)
    authored_matchup: AuthoredMatchup | None = None
    game_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    max_commands: int = Field(default=10000, gt=0)
    workers: int = Field(default=1, ge=1)
    capture_version: Literal[1] = CAPTURE_VERSION
    metric_version: Literal["gih-copies-v1"] = METRIC_VERSION

    @model_validator(mode="after")
    def unique(self) -> CardEvaluationSpec:
        if len(set(self.deals)) != len(self.deals) or any(
            not 0 <= seed < 2**64 for seed in self.deals
        ):
            raise ValueError("deal seeds must be unique uint64 values")
        if set(self.checkpoints) != {
            p.player_id for p in self.players if p.runner_kind == "checkpoint"
        }:
            raise ValueError(
                "checkpoint paths must name exactly the checkpoint entrants"
            )
        if len({p.player_id for p in self.players}) != len(self.players):
            raise ValueError("player IDs must be unique")
        if self.key.content_suite != (
            self.authored_matchup.content_suite
            if self.authored_matchup
            else SELECTED_SUITE
        ):
            raise ValueError("card evaluation requires an authored arena matrix")
        return self


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _entrants(spec: CardEvaluationSpec) -> tuple[Agent, ...]:
    setup = spec.authored_matchup.match() if spec.authored_matchup else selected_match()
    # Bind all Python inference code as well as the separately bound native
    # runtime; a registration's human label never establishes byte identity.
    code = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    spaces = []
    for player in spec.players:
        _, space = validate_registered_player(
            player, spec.checkpoints.get(player.player_id)
        )
        if space is not None:
            spaces.append(space)
    if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
        raise ValueError("players have different observation ABIs")
    runtime = runtime_fingerprints(
        match_config=setup, observation_space=spaces[0] if spaces else None
    )
    entrants = []
    for player in spec.players:
        validate_player_compatibility(player, spec.key, runtime)
        checkpoint = None
        if player.runner_kind == "checkpoint":
            path = spec.checkpoints[player.player_id].resolve()
            checkpoint = ArtifactRef(
                uri=path.as_uri(),
                sha256=player.checkpoint_sha256,
                size_bytes=player.checkpoint_bytes,
            )
        entrants.append(
            agent(
                player.player_id,
                checkpoint=checkpoint,
                code=code,
                config={"registration": player.model_dump(mode="json")},
            )
        )
    return tuple(entrants)


def play_card_match(match: Match) -> Tables:
    players = card_match_players(match)
    conditions = CardConditions.model_validate(match.conditions)
    paths: dict[str, str] = {}
    for entrant, player in zip(match.agents, players, strict=True):
        if entrant.kind == "checkpoint":
            uri = urlsplit(entrant.uri or "")
            if (
                uri.scheme != "file"
                or uri.netloc
                or entrant.artifact != player.checkpoint_sha256
            ):
                raise ValueError("checkpoint must be admitted local bytes")
            path = Path(unquote(uri.path))
            verify_file(path, entrant.artifact, player.checkpoint_bytes)
            paths[player.player_id] = str(path)
    root = Path(os.environ["MANABOT_CARD_OUTPUT"])
    attempts = root / "arena" / match.key
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
    rows, trace, _ = play_cell(
        key=conditions.key,
        player_a=players[0],
        player_b=players[1],
        deal_seeds=(conditions.deal,),
        out_dir=attempt,
        checkpoint_paths=paths,
        game_seconds=conditions.game_seconds,
        max_commands=conditions.max_commands,
        matchup_mode="all",
        authored_matchup=conditions.authored_matchup,
        card_evidence_dir=root,
    )
    if (
        len(rows) != 8
        or {r["leg"] for r in rows} != set(range(8))
        or not all(EvaluationRow.model_validate(r).valid for r in rows)
    ):
        excluded: Counter[str] = Counter()
        for row in rows:
            reason = (
                "replay_failed"
                if not row["replay_passed"]
                else "timeout"
                if row["termination_reason"] == "timeout"
                else "truncated"
                if row["truncated"]
                else "crash"
                if row["failure"]
                else "failed_match"
            )
            excluded[reason] += 1
        excluded["unobserved"] += max(0, match.games - len(rows))
        raise CardMatchError(dict(excluded))
    tables: dict[str, list[Row]] = {"plays": [], "card_games": [], "card_evidence": []}
    participants = (players[0].player_id, players[1].player_id)
    for raw in trace["card_evidence"]:
        ref = EvidenceRef.model_validate(raw)
        game = read_evidence((root / ref.path).read_bytes(), ref)
        if not game.replay["passed"]:
            raise ValueError("replay failed; card tables unavailable")
        for row in reduce_cards(game, participants):
            tables["card_games"].append(row.model_dump())
        for player, participant in enumerate(participants):
            seat = game.seat_players.index(participant)
            tables["card_evidence"].append(
                {
                    "player": player,
                    **ref.model_dump(),
                    "capture_version": CAPTURE_VERSION,
                    "metric_version": METRIC_VERSION,
                    "trace_sha256": game.trace_sha256,
                    "replay_sha256": game.replay_sha256,
                }
            )
            tables["plays"].append(
                {
                    "player": player,
                    "game": game.game,
                    "leg": game.leg,
                    "score": 0.5 if game.winner is None else float(game.winner == seat),
                    "on_play": seat == 0,
                    "deck": game.seat_decks[seat],
                    "opponent_deck": game.seat_decks[1 - seat],
                    "valid": True,
                }
            )
    return tables


def evaluate_cards(spec: CardEvaluationSpec, directory: Path) -> EvalRun:
    entrants = _entrants(spec)
    setup = spec.authored_matchup.match() if spec.authored_matchup else selected_match()
    runtime = runtime_fingerprints(match_config=setup)
    # File names and the compiled extension's platform bytes are not environment semantics.
    identity = {key: runtime[key] for key in RUNTIME_FIELDS}
    tournament = RoundRobin(
        name=EXPERIMENT,
        environment=Environment(
            name=EXPERIMENT,
            files=canonical_sha256(identity),
            config={
                **identity,
                "capture_version": CAPTURE_VERSION,
                "metric_version": METRIC_VERSION,
            },
        ),
        conditions=tuple(
            CardConditions(
                deal=deal,
                key=spec.key,
                authored_matchup=spec.authored_matchup,
                game_seconds=spec.game_seconds,
                max_commands=spec.max_commands,
            ).model_dump(mode="json")
            for deal in spec.deals
        ),
        games=8,
    )
    declared = declare(
        experiment=EXPERIMENT,
        clocks=(),
        run=spec.run,
        kind="evaluation",
        directory=directory,
    )
    with _environment(
        {"MANABOT_CARD_OUTPUT": str(declared.root.resolve()), "ETUDE_RUN": spec.run}
    ):
        return hold(
            tournament,
            experiment=EXPERIMENT,
            play=play_card_match,
            enter=entrants,
            workers=spec.workers,
            directory=directory,
        )


def publish_cards(directory: Path, destination: Storage) -> None:
    """Copy validated evidence first, then Etude tables and completion records.

    SDK 0.2 sync does not include arbitrary artifacts. A repeated publication is
    harmless: content-addressed evidence is immutable and Etude owns table delivery.
    No network destination is selected implicitly.
    """
    source = DirectoryStorage(directory)
    run_directories: set[Path] = set()
    for run in places(source):
        prefix = evidence_prefix(source, run)
        run_directories.add(directory / Path(prefix).parent)
        for ref in evidence_refs(source, run):
            body = source.get(prefix + ref.path)
            read_evidence(body, ref)
            destination.put(f"{run}/{ref.path}", body)
            read_evidence(destination.get(f"{run}/{ref.path}"), ref)
    # Query discovery is recursive; SDK sync expects immediate run children.
    failures = {
        run: errors
        for parent in sorted(run_directories)
        for run, errors in sync(parent, storages=(destination,)).items()
    }
    if failures:
        raise RuntimeError(f"evaluation publication incomplete: {failures}")
