"""Etude scheduling and delivery for replay-derived card statistics.

One match is the existing eight-leg authored arena matrix for one deal. Etude
owns completion, retries and scalar tables; every eval run names the experiment
whose database it joins. Each game's evidence is written under the eval run's
folder, which Etude delivery carries with its tables. Games are admitted one at
a time: a crash, timeout or failed replay is an ineligible `card_plays` row,
never a reason to drop the rest of its deal."""

from __future__ import annotations

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
from etude.run.records import ArtifactRef
from pydantic import Field, model_validator

from manabot.eval.card_metrics import METRIC_VERSION, normalize, reduce_cards
from manabot.eval.card_schedule import (
    RUNTIME_FIELDS,
    CardConditions,
    admit_card_game,
    card_match_players,
)
from manabot.eval.evaluate import (
    validate_player_compatibility,
    validate_registered_player,
)
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.infra.artifacts import verify_file
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.infra.source import source_commit, source_dirty
from manabot.sim.card_evidence import (
    CAPTURE_VERSION,
    EvidenceRef,
    GameEvidence,
    read_evidence,
)
from manabot.sim.matchup import SELECTED_SUITE, AuthoredMatchup, selected_match
from manabot.sim.registration import PlayerRegistration

NAME = r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,199}$"


class CardEvaluationSpec(StrictModel):
    experiment: str = Field(pattern=NAME)
    run: str = Field(pattern=NAME)
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
    _, trace, _ = play_cell(
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
    return card_tables(
        match,
        [
            (ref, read_evidence((root / ref.path).read_bytes(), ref))
            for ref in map(EvidenceRef.model_validate, trace["card_evidence"])
        ],
    )


def card_tables(match: Match, games: list[tuple[EvidenceRef, GameEvidence]]) -> Tables:
    """Admit each retained game and project it into Etude tables.

    `plays` is the shared outcome table; the card tables are normalized by
    `card_metrics.normalize`, with each game's evidence reference and source
    commit on its `card_plays` rows.
    """
    tables: dict[str, list[Row]] = {
        "plays": [],
        "card_plays": [],
        "card_counts": [],
        "cards": [],
    }
    provenance = {"source_commit": source_commit(), "source_dirty": source_dirty()}
    for ref, game in games:
        participants = admit_card_game(game, match)
        rows = normalize(reduce_cards(game, participants))
        evidence = {
            "evidence": ref.path,
            "evidence_sha256": ref.sha256,
            "evidence_bytes": ref.bytes,
            "trace_sha256": game.trace_sha256,
            "replay_sha256": game.replay_sha256,
            "capture_version": CAPTURE_VERSION,
            **provenance,
        }
        tables["card_plays"].extend({**play, **evidence} for play in rows.plays)
        tables["card_counts"].extend(rows.counts)
        tables["cards"].extend(rows.cards)
        for play in rows.plays:
            tables["plays"].append(
                {
                    "player": play["player"],
                    "game": game.game,
                    "score": 0.5 if game.winner is None else float(play["win"] or 0),
                    "on_play": play["on_play"],
                    "deck": play["deck"],
                    "opponent_deck": play["opponent_deck"],
                    "valid": play["eligible"] or game.outcome == "draw",
                }
            )
    if not tables["card_plays"]:
        raise ValueError("match retained no card evidence")
    return tables


def evaluate_cards(spec: CardEvaluationSpec, directory: Path) -> EvalRun:
    entrants = _entrants(spec)
    setup = spec.authored_matchup.match() if spec.authored_matchup else selected_match()
    runtime = runtime_fingerprints(match_config=setup)
    # File names and the compiled extension's platform bytes are not environment semantics.
    identity = {key: runtime[key] for key in RUNTIME_FIELDS}
    tournament = RoundRobin(
        name=spec.experiment,
        environment=Environment(
            name=spec.experiment,
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
        experiment=spec.experiment,
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
            experiment=spec.experiment,
            play=play_card_match,
            enter=entrants,
            workers=spec.workers,
            directory=directory,
        )
