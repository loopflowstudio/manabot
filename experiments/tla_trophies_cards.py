"""Play a trophy held-out deal and keep each game's card evidence.

The same pairing as `tla_trophies.play`, with card evidence handed to Etude
beside the outcome tables so `manabot.eval.card_page` can read the games.
Benchmarks that use it declare `CARD_ROLLUPS` and name this file in their
environment. Retained trophy benchmarks keep `tla_trophies.play` so their
games keep their identities.
"""

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from etude.eval.match import environment
from etude.eval.records import CheckpointPlayer, Code, Environment, Match
from etude.eval.result import Result

from experiments import tla_trophies as trophies
from manabot.eval.admission import register_anchor
from manabot.eval.agents import game_players, registration
from manabot.eval.benchmark import evaluation_tables
from manabot.eval.card_evaluation import trace_card_tables
from manabot.eval.evaluation import EvaluationRequest, attempt_play_counts, evaluate
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import ETUDE_COMPUTE_CLASS, PlayerRegistration

ARENA = "tla-trophies"
NAMES = ("candidate", "opponent")
LEGS = 4


def card_environment() -> Environment:
    return environment(
        ARENA,
        files=[
            Path(__file__),
            Path(trophies.__file__),
            trophies.ROOT / "manabot/eval/pairing.py",
        ],
        config={"pools": trophies.corpus_input().sha256, "cards": True},
    )


def play(match: Match, agents: Sequence[object]) -> Result:
    """All four cross legs of one deal, with card tables and evidence blobs."""
    deal = trophies.Deal.model_validate(match.config)
    pools = trophies.playable("held_out")
    a, b = pools[deal.first], pools[deal.second]
    matchup = DeckMatchup(
        population_sha256=trophies.corpus_input().sha256,
        entries=(a.id, b.id),
        lists=(trophies.deck(a), trophies.deck(b)),
        decklists="closed",
    )
    if match.games != 1 or len(match.players) != 2:
        raise ValueError("a pairing requires two players and one indexed deal")
    players: list[PlayerRegistration] = []
    paths: dict[str, str] = {}
    for index, player in enumerate(match.players):
        if index == 1 and isinstance(player, Code):
            players.append(
                register_anchor(
                    players[0],
                    "heuristic",
                    player_id="opponent",
                    heuristic_actions=player.config.get("max_actions"),
                )
            )
            continue
        if not isinstance(player, CheckpointPlayer):
            raise ValueError("pairing players are checkpoints or a heuristic opponent")
        players.append(
            registration(
                player,
                name=NAMES[index],
                match=matchup.match(),
                content_suite=matchup.content_suite,
                role="challenger" if index == 0 else "anchor",
            )
        )
        paths[NAMES[index]] = str(player.checkpoint.path)
    candidate, opponent = players
    key = EvaluationKey(
        content_suite=matchup.content_suite,
        viewer_boundary="acting-viewer",
        arena_version=ARENA,
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump(mode="json")),
        evaluation_compute_envelope_id=ETUDE_COMPUTE_CLASS,
    )
    retained = (
        Path(os.environ.get("ETUDE_OUTPUT_DIR") or Path(".runs") / ARENA)
        / "arena"
        / match.key
    )
    journal_root = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with TemporaryDirectory(dir=journal_root) as journal:
        out = Path(journal) / "deal"
        cards = Path(journal) / "cards"
        cards.mkdir()
        try:
            result = evaluate(
                EvaluationRequest(
                    key=key,
                    candidate=candidate,
                    opponent=opponent,
                    checkpoint_paths=paths,
                    deal_seeds=(deal.seed,),
                    matchup_mode="cross",
                    authored_matchup=matchup,
                    game_seconds=120,
                    max_commands=10_000,
                    bootstrap_seed=6510,
                    purpose=ARENA,
                ),
                out_dir=out,
                agents=dict(
                    zip(NAMES, game_players(match.players, agents), strict=True)
                ),
                card_evidence_dir=cards,
            )
            if result.error:
                raise ValueError(result.error)
            counts = (
                attempt_play_counts(out, result.rows)
                if opponent.runner_kind == "code" and result.status == "completed"
                else None
            )
            tables = evaluation_tables(
                result.rows, matchup=matchup, legs=LEGS, play_counts=counts
            )
            found = trace_card_tables(
                cards, json.loads((out / "trace.json").read_bytes()), NAMES
            )
            return Result(tables={**tables, **found.tables}, blobs=found.blobs)
        finally:
            if out.is_dir():
                shutil.copytree(out, retained, dirs_exist_ok=True)
