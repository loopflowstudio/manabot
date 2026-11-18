"""Play a paired authored deal through the replay-checked arena.

The candidate is a checkpoint; the opponent is a checkpoint or the native
heuristic (`Code(name="heuristic")`). The caller owns the experimental
schedule; Etude owns the model forwards and supplies the built agents. This
module owns player admission, retained arena evidence and disclosure labels on
the resulting tables.
"""

from collections.abc import Sequence
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from etude.eval.records import CheckpointPlayer, Code, Match
from etude.eval.result import Tables

from manabot.eval.admission import register_anchor
from manabot.eval.agents import game_players, registration
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import ETUDE_COMPUTE_CLASS, PlayerRegistration
import managym


def play_pairing(
    match: Match,
    agents: Sequence[object],
    *,
    matchup: AuthoredMatchup,
    deal: int,
    matchup_mode: Literal["cross", "all"],
    game_seconds: float,
    arena: str,
) -> Tables:
    """Play all legs; an incomplete or invalid pairing raises after retaining evidence."""
    legs = 4 if matchup_mode == "cross" else 8
    if match.games != legs or len(match.players) != 2:
        raise ValueError("a pairing requires two players and every declared leg")
    players: list[PlayerRegistration] = []
    paths: dict[str, str] = {}
    for index, player in enumerate(match.players):
        if index == 1 and isinstance(player, Code):
            if player.name != "heuristic":
                raise ValueError("a pairing's code opponent must be the heuristic")
            players.append(
                register_anchor(players[0], "heuristic", player_id="opponent")
            )
            continue
        if not isinstance(player, CheckpointPlayer):
            raise ValueError("pairing players are checkpoints or a heuristic opponent")
        name = "candidate" if index == 0 else "opponent"
        players.append(
            registration(
                player,
                name=name,
                match=matchup.match(),
                content_suite=matchup.content_suite,
                role="challenger" if index == 0 else "anchor",
            )
        )
        paths[name] = str(player.checkpoint.path)
    candidate, opponent = players
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=arena,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id=ETUDE_COMPUTE_CLASS,
    )
    retained = Path(os.environ["ETUDE_OUTPUT_DIR"]) / "arena" / match.key
    journal_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with TemporaryDirectory(dir=journal_root) as journal:
        out = Path(journal) / "deal"
        try:
            result = evaluate(
                EvaluationRequest(
                    key=key,
                    candidate=candidate,
                    opponent=opponent,
                    checkpoint_paths=paths,
                    deal_seeds=(deal,),
                    matchup_mode=matchup_mode,
                    authored_matchup=matchup,
                    game_seconds=game_seconds,
                    max_commands=10_000,
                    bootstrap_seed=6510,
                    purpose=arena,
                ),
                out_dir=out,
                agents=dict(
                    zip(
                        ("candidate", "opponent"),
                        game_players(match.players, agents),
                        strict=True,
                    )
                ),
            )
        finally:
            if out.is_dir():
                shutil.copytree(out, retained, dirs_exist_ok=True)
    if result.error:
        raise ValueError(result.error)
    tables = evaluation_tables(result.rows, matchup=matchup, legs=legs)
    for rows in tables.values():
        for row in rows:
            row["decklists"] = matchup.decklists
    return tables
