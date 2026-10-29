"""Etude scheduling and delivery for replay-derived card statistics.

One match is the existing eight-leg authored arena matrix for one deal. Etude
owns completion, retries and scalar tables; every eval run names the experiment
whose database it joins. The replay worker hands each game's evidence over in a
scratch folder; `play` returns it to Etude as a blob of its match
(`etude.eval.match.Result`), named in `card_plays` by its SHA256, and the run
declares `CARD_ROLLUPS` so its blocks carry the totals card queries read. Games
are admitted one at a time: a crash, timeout or failed replay is an ineligible
`card_plays` row, never a reason to drop the rest of its deal."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Literal

from etude.eval.match import EvalRun, Result, Row, Tables
from etude.eval.records import (
    CheckpointKey,
    CheckpointRef,
    Code,
    Environment,
    Inference,
    Match,
    Model,
    Player,
    RoundRobin,
)
from etude.eval.tournament import hold
from etude.run.records import ArtifactRef
from pydantic import Field, model_validator

from manabot.eval.card_metrics import (
    CARD_ROLLUPS,
    METRIC_VERSION,
    normalize,
    reduce_cards,
)
from manabot.eval.card_schedule import (
    RUNTIME_FIELDS,
    CardConditions,
    admit_card_game,
    card_match_players,
)
from manabot.eval.evaluate import (
    source_digest,
    validate_player_compatibility,
    validate_registered_player,
)
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.infra.source import source_commit, source_dirty
from manabot.model.checkpoint import local_checkpoint_file
from manabot.sim.card_evidence import (
    CAPTURE_VERSION,
    EvidenceRef,
    GameEvidence,
    read_evidence,
)
from manabot.sim.matchup import SELECTED_SUITE, AuthoredMatchup, selected_match
from manabot.sim.registration import PlayerRegistration

VERIFY_EXPERIMENT = "card-verification"

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


def verification_spec(games: int) -> CardEvaluationSpec:
    """Build the bounded scripted schedule without importing report dependencies."""
    if games < 8 or games % 8:
        raise ValueError("games must be a positive multiple of eight")
    matrix = AuthoredMatchup(
        pack_key="elemental-world-v2", role_a="elemental_ug", role_b="elemental_bg"
    )
    runtime = runtime_fingerprints(match_config=matrix.match())
    players = tuple(
        PlayerRegistration(
            player_id=name,
            display_name=name,
            role="anchor",
            runner_kind="code",
            player_spec={"kind": "scripted_greedy"},
            compute_class_id="scripted-greedy-cpu-v1",
            information_boundary="acting-viewer",
            world=runtime["world"],
            content_suite=matrix.content_suite,
            observation_abi_sha256=runtime["observation_abi_sha256"],
            action_abi_sha256=runtime["action_abi_sha256"],
            matchup_sha256=runtime["matchup_sha256"],
            source_sha256=source_digest(["manabot/sim/anchors.py"]),
            player_seed_derivation_id="arena-pair-deal-player-v1",
        )
        for name in ("verify-one", "verify-two")
    )
    return CardEvaluationSpec(
        experiment=VERIFY_EXPERIMENT,
        run=f"verify-{games}",
        key=EvaluationKey(
            world=runtime["world"],
            content_suite=matrix.content_suite,
            viewer_boundary="acting-viewer",
            arena_version="card-evaluation-v1",
            rating_model_version="unrated-card-evaluation",
            rating_prior_sha256=canonical_sha256({}),
            anchor_cohort_sha256=canonical_sha256([p.model_dump() for p in players]),
            evaluation_compute_envelope_id="scripted-greedy-cpu-v1",
        ),
        players=players,
        deals=tuple(range(230000, 230000 + games // 8)),
        authored_matchup=matrix,
        game_seconds=60,
        max_commands=10000,
    )


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


def _entrants(spec: CardEvaluationSpec) -> tuple[Player, ...]:
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
    entrants: list[Player] = []
    for player in spec.players:
        validate_player_compatibility(player, spec.key, runtime)
        config = {"registration": player.model_dump(mode="json")}
        if player.runner_kind == "checkpoint":
            path = spec.checkpoints[player.player_id].resolve()
            checkpoint = CheckpointRef(
                key=CheckpointKey(address=path.as_uri()),
                artifact=ArtifactRef(
                    uri=path.as_uri(),
                    sha256=player.checkpoint_sha256,
                    size_bytes=player.checkpoint_bytes,
                ),
            )
            entrants.append(
                Model(
                    name=player.player_id,
                    checkpoint=checkpoint,
                    inference=Inference(source=code, config=config),
                )
            )
        else:
            entrants.append(Code(name=player.player_id, source=code, config=config))
    return tuple(entrants)


def play_card_match(match: Match) -> Result:
    players = card_match_players(match)
    conditions = CardConditions.model_validate(match.conditions)
    paths: dict[str, str] = {}
    for entrant, player in zip(match.players, players, strict=True):
        if isinstance(entrant, Model):
            artifact = entrant.checkpoint.artifact
            if (
                artifact.sha256 != player.checkpoint_sha256
                or artifact.size_bytes != player.checkpoint_bytes
            ):
                raise ValueError("checkpoint must be admitted local bytes")
            paths[player.player_id] = str(local_checkpoint_file(entrant))
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
    games = _handed_over(root, trace)
    return Result(
        tables=card_tables(match, [(ref, game) for _, ref, game in games]),
        blobs=[body for body, _, _ in games],
    )


def _handed_over(
    root: Path, trace: Mapping[str, object]
) -> list[tuple[bytes, EvidenceRef, GameEvidence]]:
    """Each game's evidence bytes the replay worker wrote under `root`, verified.

    `root` is a scratch folder: each file is removed once read, since its bytes
    leave as a blob of the match.
    """
    references = trace.get("card_evidence")
    if not isinstance(references, list):
        raise ValueError("play_cell retained no card evidence; pass card_evidence_dir")
    games = []
    for raw in references:
        ref = EvidenceRef.model_validate(raw)
        body = (root / ref.path).read_bytes()
        games.append((body, ref, read_evidence(body, ref)))
        (root / ref.path).unlink()
    return games


def card_tables(match: Match, games: list[tuple[EvidenceRef, GameEvidence]]) -> Tables:
    """A card evaluation's tables: each game admitted against its match, plus `plays`."""
    tables: dict[str, list[Row]] = {"plays": []}
    for ref, game in games:
        participants = admit_card_game(game, match)
        game_tables = _game_card_tables(ref, game, participants)
        for name, rows in game_tables.items():
            tables.setdefault(name, []).extend(rows)
        for play in game_tables["card_plays"]:
            tables["plays"].append(
                {
                    "seat": play["seat"],
                    "game_seat": play["game_seat"],
                    "game": game.game,
                    "score": 0.5 if game.winner is None else float(play["win"] or 0),
                    "on_play": play["on_play"],
                    "deck": play["deck"],
                    "opponent_deck": play["opponent_deck"],
                    "valid": play["eligible"] or game.outcome == "draw",
                }
            )
    if not tables["plays"]:
        raise ValueError("match retained no card evidence")
    return tables


def trace_card_tables(
    root: Path, trace: Mapping[str, object], participants: tuple[str, str]
) -> Result:
    """Card tables and evidence blobs for the games one `play_cell(card_evidence_dir=root)` call retained.

    This lets any evaluation deliver card data beside its own outcome tables:
    merge the result's tables and blobs into the `Result` its `play` returns,
    and declare `CARD_ROLLUPS` with the run. `root` is a scratch folder (see
    `_handed_over`). `participants` names the two registered players in Etude
    row order. Games are not bound to a card-evaluation schedule; each must
    seat exactly these players.
    """
    tables: dict[str, list[Row]] = {}
    blobs = []
    for body, ref, game in _handed_over(root, trace):
        blobs.append(body)
        for name, rows in _game_card_tables(ref, game, participants).items():
            tables.setdefault(name, []).extend(rows)
    return Result(tables=tables, blobs=blobs)


def _game_card_tables(
    ref: EvidenceRef, game: GameEvidence, participants: tuple[str, str]
) -> dict[str, list[Row]]:
    """One game's normalized card rows, with its evidence and provenance on `card_plays`."""
    rows = normalize(reduce_cards(game, participants))
    evidence = {
        "evidence_sha256": ref.sha256,
        "evidence_bytes": ref.bytes,
        "trace_sha256": game.trace_sha256,
        "replay_sha256": game.replay_sha256,
        "capture_version": CAPTURE_VERSION,
        "source_commit": source_commit(),
        "source_dirty": source_dirty(),
    }
    return {
        "card_plays": [
            {**play, **evidence, "participant": participants[int(play["seat"] or 0)]}
            for play in rows.plays
        ],
        "card_counts": list(rows.counts),
        "cards": list(rows.cards),
    }


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
    # Workers' attempt folders and evidence handoff; nothing here is kept.
    with (
        tempfile.TemporaryDirectory(prefix="manabot-cards-") as scratch,
        _environment({"MANABOT_CARD_OUTPUT": scratch, "ETUDE_RUN": spec.run}),
    ):
        return hold(
            tournament,
            experiment=spec.experiment,
            play=play_card_match,
            enter=entrants,
            workers=spec.workers,
            directory=directory,
            rollups=CARD_ROLLUPS,
        )
