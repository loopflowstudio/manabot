"""Play a paired deal through the replay-checked arena.

The candidate is a checkpoint; the opponent is a checkpoint or the native
heuristic (`Code(name="heuristic")`). The caller owns the experimental
schedule; Etude owns the model forwards and supplies the built agents. This
module admits players, retains the arena trace and returns each game's card
evidence as a blob of the Etude match.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from etude.eval.records import CheckpointPlayer, Code, Match
from etude.eval.result import Result

from manabot.eval.admission import register_anchor
from manabot.eval.agents import game_players, registration
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.card_evaluation import trace_card_tables
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


@dataclass(frozen=True)
class AdmittedPairing:
    """Validated pairing request, built game players and output destination."""

    request: EvaluationRequest
    agents: dict[str, GamePlayer]
    legs: int
    participants: tuple[str, str]
    retained: Path


def admit_pairing(
    match: Match,
    agents: Sequence[object],
    *,
    matchup: AuthoredMatchup | DeckMatchup,
    deal: int,
    matchup_mode: Literal["cross", "all"],
    game_seconds: float,
    arena: str,
    precision: EvalPrecision = "float32",
    max_commands: int = 10_000,
    bootstrap_seed: int = 6510,
    output_dir: Path | None = None,
) -> AdmittedPairing:
    """Validate match players and build registrations, request and game players."""
    legs = 4 if matchup_mode == "cross" else 8
    # An indexed Etude match is one deal. A legacy multi-game match declares legs.
    expected_games = legs if match.index is None else 1
    if match.games != expected_games or len(match.players) != 2:
        raise ValueError("a pairing requires two players and every declared leg")
    players: list[PlayerRegistration] = []
    paths: dict[str, str] = {}
    participants = ("candidate", "opponent")
    for name, player in zip(participants, match.players, strict=True):
        if name == "opponent" and isinstance(player, Code):
            if player.name != "heuristic":
                raise ValueError("a pairing's code opponent must be the heuristic")
            players.append(
                register_anchor(
                    players[0],
                    "heuristic",
                    player_id=name,
                    heuristic_actions=player.config.get("max_actions"),
                )
            )
            continue
        if not isinstance(player, CheckpointPlayer):
            raise ValueError("pairing players are checkpoints or a heuristic opponent")
        players.append(
            registration(
                player,
                name=name,
                match=matchup.match(),
                content_suite=matchup.content_suite,
                role="challenger" if name == "candidate" else "anchor",
                precision=precision,
            )
        )
        paths[name] = str(player.checkpoint.path)
    candidate, opponent = players
    anchor = opponent.model_dump(mode="json")
    if opponent.runner_kind == "checkpoint":
        # The cohort identifies the opponent; the envelope identifies precision.
        anchor["compute_class_id"] = ETUDE_COMPUTE_CLASS
    key = EvaluationKey(
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=arena,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(anchor),
        evaluation_compute_envelope_id=candidate.compute_class_id,
        seed_compute_envelope_id=(
            ETUDE_COMPUTE_CLASS if precision != "float32" else None
        ),
    )
    root = (
        output_dir
        if output_dir is not None
        else Path(os.environ.get("ETUDE_OUTPUT_DIR") or Path(".runs") / arena)
    )
    retained = root / "arena" / match.key
    request = EvaluationRequest(
        key=key,
        candidate=candidate,
        opponent=opponent,
        checkpoint_paths=paths,
        deal_seeds=(deal,),
        matchup_mode=matchup_mode,
        authored_matchup=matchup,
        game_seconds=game_seconds,
        max_commands=max_commands,
        bootstrap_seed=bootstrap_seed,
        purpose=arena,
    )
    seated = dict(zip(participants, game_players(match.players, agents), strict=True))
    return AdmittedPairing(
        request=request,
        agents=seated,
        legs=legs,
        participants=participants,
        retained=retained,
    )


def pairing_result(
    result: EvaluationResult,
    *,
    out: Path,
    cards: Path,
    matchup: AuthoredMatchup | DeckMatchup,
    legs: int,
    participants: tuple[str, str],
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
    tables = evaluation_tables(
        result.rows, matchup=matchup, legs=legs, play_counts=counts
    )
    trace = json.loads((out / "trace.json").read_bytes())
    evidence = trace_card_tables(cards, trace, participants)
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
    admitted = admit_pairing(
        match,
        agents,
        matchup=matchup,
        deal=deal,
        matchup_mode=matchup_mode,
        game_seconds=game_seconds,
        arena=arena,
        precision=precision,
    )
    journal_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with TemporaryDirectory(dir=journal_root) as journal:
        out = Path(journal) / "deal"
        # Only the deal directory is retained. Etude stores card evidence as blobs.
        cards = Path(journal) / "cards"
        try:
            result = evaluate(
                admitted.request,
                out_dir=out,
                card_evidence_dir=cards,
                agents=admitted.agents,
            )
            return pairing_result(
                result,
                out=out,
                cards=cards,
                matchup=matchup,
                legs=admitted.legs,
                participants=admitted.participants,
            )
        finally:
            if out.is_dir():
                shutil.copytree(out, admitted.retained, dirs_exist_ok=True)
