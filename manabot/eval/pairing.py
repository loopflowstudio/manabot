"""Play a paired authored deal through the replay-checked arena.

The candidate is a checkpoint; the opponent is a checkpoint or the native
heuristic (`Code(name="heuristic")`). The caller owns the experimental schedule
and backend. This module owns player admission, retained arena evidence and
disclosure labels on the resulting tables.
"""

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from etude.eval.match import Tables
from etude.eval.records import CheckpointPlayer, Code, Match

from manabot.eval.admission import register_anchor, register_checkpoint
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.eval.models import EvaluationKey
from manabot.infra.artifacts import local_checkpoint_file
from manabot.infra.identity import canonical_sha256
from manabot.policy.protocol import Backend
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.registration import PlayerRegistration
import managym


def play_pairing(
    match: Match,
    *,
    matchup: AuthoredMatchup,
    deal: int,
    matchup_mode: Literal["cross", "all"],
    game_seconds: float,
    arena: str,
    backend: Backend,
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
        blank = player.inference.config.get("blank_decision_memory", False)
        if not isinstance(blank, bool):
            raise ValueError("blank_decision_memory must be a boolean")
        name = "candidate" if index == 0 else "opponent"
        path = local_checkpoint_file(player)
        admission = register_checkpoint(
            path,
            match=matchup.match(),
            content_suite=matchup.content_suite,
            player_id=name,
            display_name=player.name,
            role="challenger" if index == 0 else "anchor",
            artifact_prefix=arena,
            allow_deck_repetition=True,
            backend=backend,
            blank_decision_memory=blank,
        )
        players.append(admission.registration)
        paths[name] = str(path)
    candidate, opponent = players
    key = EvaluationKey(
        world=managym.WORLD_VERSION,
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=arena,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
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
