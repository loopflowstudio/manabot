"""Play a paired deal through the replay-checked arena.

The candidate is a checkpoint. The opponent can be a checkpoint, the native
heuristic, or a registered code player. The caller owns the experimental
schedule; Etude owns the model forwards and supplies the built agents. This
module admits players, retains the arena trace and returns each game's card
evidence as a blob of the Etude match.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Literal, cast

from etude.eval.agents import ModelForward
from etude.eval.records import CheckpointPlayer, Code, Match
from etude.eval.result import Result

from manabot.eval.admission import register_anchor
from manabot.eval.agents import (
    CheckpointReader,
    ScalarSampler,
    checkpoint_reader,
    local_scalar_sampler,
    registration,
    require_heuristic_version,
)
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.card_evaluation import trace_card_tables
from manabot.eval.evaluate import validate_registration_source
from manabot.eval.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    attempt_play_counts,
    evaluate,
)
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.sim.agent import GamePlayer
from manabot.sim.matchup import AuthoredMatchup, DeckMatchup
from manabot.sim.registration import (
    ETUDE_COMPUTE_CLASS,
    EvalPrecision,
    PlayerRegistration,
)

PARTICIPANTS = ("candidate", "opponent")


@dataclass(frozen=True)
class PairingSpec:
    """Arena parameters for one paired-deal match."""

    matchup: AuthoredMatchup | DeckMatchup
    deal: int
    matchup_mode: Literal["cross", "all"]
    game_seconds: float
    arena: str
    precision: EvalPrecision = "float32"
    max_commands: int = 10_000
    bootstrap_seed: int = 6510
    keep_game_logs: bool = False


@dataclass(frozen=True)
class AdmittedPairing[Seat]:
    """Validated pairing request, built game players and output destination."""

    request: EvaluationRequest
    agents: dict[str, Seat]
    retained: Path


def admit_pairing[Seat](
    match: Match,
    agents: Sequence[Seat],
    spec: PairingSpec,
    *,
    output_dir: Path | None = None,
    read: CheckpointReader | None = None,
) -> AdmittedPairing[Seat]:
    """Validate match players and build registrations, request and game players."""
    legs = 4 if spec.matchup_mode == "cross" else 8
    # An indexed Etude match is one deal. A legacy multi-game match declares legs.
    expected_games = legs if match.index is None else 1
    if match.games != expected_games or len(match.players) != 2:
        raise ValueError("a pairing requires two players and every declared leg")
    players: list[PlayerRegistration] = []
    paths: dict[str, str] = {}
    for name, player in zip(PARTICIPANTS, match.players, strict=True):
        if name == "opponent" and isinstance(player, Code):
            if "registration" in player.config:
                opponent = PlayerRegistration.model_validate(
                    player.config["registration"]
                )
                if opponent.runner_kind != "code" or opponent.player_id != name:
                    raise ValueError("code pairing registration must name the opponent")
                validate_registration_source(opponent)
            elif player.name == "heuristic":
                require_heuristic_version(player)
                opponent = register_anchor(
                    players[0],
                    "heuristic",
                    player_id=name,
                    heuristic_actions=player.config.get("max_actions"),
                )
            else:
                raise ValueError("a pairing's code opponent requires a registration")
            players.append(opponent)
            continue
        if not isinstance(player, CheckpointPlayer):
            raise ValueError(
                "pairing requires a checkpoint candidate and an admitted opponent"
            )
        players.append(
            registration(
                player,
                name=name,
                match=spec.matchup.match(),
                content_suite=spec.matchup.content_suite,
                role="challenger" if name == "candidate" else "anchor",
                precision=spec.precision,
                read=read,
            )
        )
        paths[name] = str(player.checkpoint.path)
    candidate, opponent = players
    anchor = opponent.model_dump(mode="json")
    if opponent.runner_kind == "checkpoint":
        # The cohort identifies the opponent; the envelope identifies precision.
        anchor["compute_class_id"] = ETUDE_COMPUTE_CLASS
    key = EvaluationKey(
        content_suite=spec.matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=spec.arena,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(anchor),
        evaluation_compute_envelope_id=candidate.compute_class_id,
        seed_compute_envelope_id=(
            ETUDE_COMPUTE_CLASS if spec.precision != "float32" else None
        ),
    )
    root = (
        output_dir
        if output_dir is not None
        else Path(os.environ.get("ETUDE_OUTPUT_DIR") or Path(".runs") / spec.arena)
    )
    retained = root / "arena" / match.key
    request = EvaluationRequest(
        key=key,
        candidate=candidate,
        opponent=opponent,
        checkpoint_paths=paths,
        deal_seeds=(spec.deal,),
        matchup_mode=spec.matchup_mode,
        authored_matchup=spec.matchup,
        game_seconds=spec.game_seconds,
        max_commands=spec.max_commands,
        bootstrap_seed=spec.bootstrap_seed,
        purpose=spec.arena,
    )
    seated = dict(zip(PARTICIPANTS, agents, strict=True))
    return AdmittedPairing(
        request=request,
        agents=seated,
        retained=retained,
    )


def pairing_result(
    result: EvaluationResult,
    *,
    out: Path,
    cards: Path,
) -> Result:
    """Assemble Etude tables and card blobs from a settled pairing attempt."""
    if result.error:
        raise ValueError(result.error)
    counts = (
        attempt_play_counts(out, result.rows)
        if result.request.opponent.runner_kind == "code"
        and result.status == "completed"
        else None
    )
    legs = len(result.request.legs)
    tables = evaluation_tables(
        result.rows,
        matchup=result.request.authored_matchup,
        legs=legs,
        play_counts=counts,
    )
    trace = json.loads((out / "trace.json").read_bytes())
    evidence = trace_card_tables(cards, trace, PARTICIPANTS)
    if len(evidence.blobs) != legs:
        raise ValueError("pairing requires card evidence for every leg")
    return Result(tables={**tables, **evidence.tables}, blobs=evidence.blobs)


def play_pairing(
    match: Match,
    agents: Sequence[object],
    *,
    matchup: AuthoredMatchup | DeckMatchup,
    deal: int,
    matchup_mode: Literal["cross", "all"],
    game_seconds: float,
    arena: str,
    precision: EvalPrecision = "float32",
) -> Result:
    """Return the paired deal's outcome, behaviour and replay-derived card data.

    Missing legs or card evidence raise after the arena diagnostics are retained.
    An invalid leg with evidence keeps its null score.
    """
    read = checkpoint_reader(len(match.players))
    seated = [
        local_scalar_sampler(player, cast(ModelForward, agent), read)
        if isinstance(player, CheckpointPlayer) and not isinstance(agent, ScalarSampler)
        else cast(GamePlayer, agent)
        for player, agent in zip(match.players, agents, strict=True)
    ]
    admitted = admit_pairing(
        match,
        seated,
        PairingSpec(
            matchup=matchup,
            deal=deal,
            matchup_mode=matchup_mode,
            game_seconds=game_seconds,
            arena=arena,
            precision=precision,
        ),
        read=read,
    )
    with pairing_attempt(admitted.retained) as (out, cards):
        result = evaluate(
            admitted.request,
            out_dir=out,
            card_evidence_dir=cards,
            agents=admitted.agents,
        )
        return pairing_result(result, out=out, cards=cards)


@contextmanager
def pairing_attempt(retained: Path) -> Iterator[tuple[Path, Path]]:
    """Keep deal diagnostics locally and leave card evidence to Etude's blobs."""
    journal_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with TemporaryDirectory(dir=journal_root) as journal:
        out = Path(journal) / "deal"
        cards = Path(journal) / "cards"
        try:
            yield out, cards
        finally:
            if out.is_dir():
                shutil.copytree(out, retained, dirs_exist_ok=True)
