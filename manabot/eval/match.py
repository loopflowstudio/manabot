"""Paired evaluation cells: seeded deck/seat schedules over bounded simulation."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from manabot.eval.models import EvaluationKey
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.sim.card_evidence import GameTrace, capture_verified_game, retain_evidence
from manabot.sim.match import play_game
from manabot.sim.matchup import (
    SELECTED_SUITE,
    AuthoredMatchup,
    selected_decks,
    selected_match,
)
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import (
    replay_games,
    write_trace,
)
from manabot.verify.util import INTERACTIVE_DECK
from managym import WORLD_VERSION

MatchupMode = Literal["cross", "mirrors", "all"]


def derive_seed(
    key: EvaluationKey,
    pair: tuple[str, str],
    deal_seed: int,
    player_id: str,
    *,
    comparison_seed_aliases: dict[str, str] | None = None,
) -> int:
    aliases = comparison_seed_aliases or {}
    identity = canonical_sha256(
        {
            "arena_key": key.model_dump(),
            "pair": sorted(aliases.get(member, member) for member in pair),
            "deal_seed": deal_seed,
            "player_id": aliases.get(player_id, player_id),
        }
    )
    return int(identity[:16], 16)


def play_cell(
    *,
    key: EvaluationKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    comparison_seed_aliases: dict[str, str] | None = None,
    game_seconds: float = 120.0,
    max_commands: int = 10_000,
    matchup_mode: MatchupMode = "cross",
    authored_matchup: AuthoredMatchup | None = None,
    card_evidence_dir: Path | None = None,
    legs: tuple[int, ...] | None = None,
    keep_game_logs: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Play every deal across the mode's deck/seat legs, or only the given subset.

    Each game's Command journal under `out_dir/failures/` is kept only when the
    game fails, unless `keep_game_logs`; the cell trace keeps every game.

    `legs` selects a subset of the mode's legs for every deal, for designs that
    assign one seat and deck pair per game instead of playing whole deals.
    """
    if not math.isfinite(game_seconds) or game_seconds <= 0 or max_commands < 1:
        raise ValueError("game limits must be finite and positive")
    if player_a.player_id == player_b.player_id:
        raise ValueError("arena players must be distinct")
    selected = key.content_suite == SELECTED_SUITE
    authored = selected or authored_matchup is not None
    if (
        authored_matchup is not None
        and key.content_suite != authored_matchup.content_suite
    ):
        raise ValueError("arena suite differs from explicit authored matrix")
    if not authored and key.content_suite != "w2-interactive-mirror-v1":
        raise ValueError("unknown arena content suite")
    if not deal_seeds or len(set(deal_seeds)) != len(deal_seeds):
        raise ValueError("deal seeds must be nonempty and unique")
    checkpoint_paths = checkpoint_paths or {}
    match = (
        authored_matchup.match()
        if authored_matchup is not None
        else selected_match()
        if selected
        else MatchConfig(
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    if authored:
        if key.world != WORLD_VERSION:
            raise ValueError("selected matchup requires corrected-world identity")
        for player in (player_a, player_b):
            if (
                player.world != key.world
                or player.content_suite != key.content_suite
                or player.information_boundary != key.viewer_boundary
                or player.matchup_sha256 != canonical_sha256(match.model_dump())
            ):
                raise ValueError("player is not bound to the selected matchup")
    if matchup_mode not in ("cross", "mirrors", "all"):
        raise ValueError("unknown arena matchup mode")
    if matchup_mode in ("mirrors", "all"):
        if not authored:
            raise ValueError("mirrors require an explicit authored deck roster")
        for player in (player_a, player_b):
            if (
                player.runner_kind == "checkpoint"
                and player.player_spec.get("allow_deck_repetition") is not True
            ):
                raise ValueError(
                    "mirrors require explicit checkpoint deck repetition admission"
                )
    mode_legs = (
        range(8)
        if matchup_mode == "all"
        else range(4, 8)
        if matchup_mode == "mirrors"
        else range(4 if authored else 2)
    )
    if legs is not None and (not legs or not set(legs) <= set(mode_legs)):
        raise ValueError("selected legs must be a nonempty subset of the mode's legs")
    played_legs = tuple(mode_legs) if legs is None else tuple(sorted(set(legs)))
    games, rows = [], []
    pair = (player_a.player_id, player_b.player_id)
    cell_id = "__".join(sorted(pair))
    trace_relative_path = str(Path("traces") / f"{cell_id}.commands.jsonl.gz")
    trace_path = out_dir / trace_relative_path
    deal_seed_set_sha256 = canonical_sha256(list(deal_seeds))
    for block, deal_seed in enumerate(deal_seeds):
        seeds = {
            p.player_id: derive_seed(
                key,
                pair,
                deal_seed,
                p.player_id,
                comparison_seed_aliases=comparison_seed_aliases,
            )
            for p in (player_a, player_b)
        }
        for leg in played_legs:
            seat_players = (
                [player_a, player_b] if leg % 2 == 0 else [player_b, player_a]
            )
            seat_decks = (
                authored_matchup.decks(leg) if authored_matchup else selected_decks(leg)
            )
            setup = (
                authored_matchup.match(leg)
                if authored_matchup is not None
                else MatchConfig.authored("ur-lessons-vs-gw-allies", *seat_decks)
                if selected
                else match.model_copy(deep=True)
            )
            setup.hero, setup.villain = [p.player_id for p in seat_players]
            game = {
                "match_id": f"{key.arena_version}:{cell_id}:{deal_seed}:{leg}",
                "cell_id": cell_id,
                "deal_block": block,
                "deal_seed": deal_seed,
                "leg": leg,
                "seat_players": [p.player_id for p in seat_players],
                "match_hypers": setup.model_dump(),
                "player_seeds": seeds,
                "winner": None,
                "terminated": False,
                "truncated": False,
                "termination_reason": "crash",
                "failed_player_id": None,
                "failure": None,
                "decisions": [],
                "integrity": {
                    name: 0
                    for name in (
                        "illegal_actions",
                        "truncations",
                        "root_mutations",
                        "private_exposures",
                        "offer_binding_failures",
                        "command_fabrications",
                        "replay_mismatches",
                    )
                },
            }
            if authored:
                game["seat_decks"] = list(seat_decks)
            started = time.perf_counter()
            game = play_game(
                game,
                seat_players,
                checkpoint_paths,
                max_commands,
                game_seconds,
                out_dir / "failures" / f"{cell_id}.{deal_seed}.{leg}.jsonl",
                keep_completed=keep_game_logs,
            )
            seconds = time.perf_counter() - started
            game["game_trace_sha256"] = canonical_sha256(game)
            games.append(game)
            # Durable completed attempts survive failure of a subsequent game.
            trace_receipt = write_trace(trace_path, games)
            if game["failure"] is not None:
                score_a = (
                    None
                    if game["failed_player_id"] is None
                    else float(game["failed_player_id"] == player_b.player_id)
                )
            else:
                score_a = (
                    0.5 if game["winner"] is None else float(game["winner"] == leg % 2)
                )
            row = {
                "arena_key": key.model_dump(),
                "cell_id": game["cell_id"],
                "deal_block": block,
                "deal_seed": deal_seed,
                "deal_seed_set_sha256": deal_seed_set_sha256,
                "leg": leg,
                "player_a": player_a.player_id,
                "player_b": player_b.player_id,
                "player_a_registration_sha256": player_a.identity_sha256,
                "player_b_registration_sha256": player_b.identity_sha256,
                "player_a_compute_class": player_a.compute_class_id,
                "player_b_compute_class": player_b.compute_class_id,
                "player_a_seed": seeds[player_a.player_id],
                "player_b_seed": seeds[player_b.player_id],
                "player_a_seat": leg % 2,
                "score_a": score_a,
                **{
                    name: game[name]
                    for name in (
                        "winner",
                        "terminated",
                        "truncated",
                        "termination_reason",
                        "failed_player_id",
                        "failure",
                    )
                },
                "decisions": len(game["decisions"]),
                "game_trace_sha256": game["game_trace_sha256"],
                "trace_path": trace_relative_path,
                "trace_sha256": game["game_trace_sha256"],
                "replay_passed": False,
                "integrity": deepcopy(game["integrity"]),
                "latency": {},
                "game_seconds": seconds,
            }
            if authored:
                row["seat_decks"] = list(seat_decks)
            for player_id in pair:
                values = [
                    d["latency_seconds"]
                    for d in game["decisions"]
                    if d["player_id"] == player_id
                ]
                row["latency"][player_id] = {
                    "count": len(values),
                    "seconds": float(sum(values)),
                    "p50": float(np.percentile(values, 50)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                }
            rows.append(row)
    trace_receipt["artifact_path"] = trace_relative_path
    replay_started = time.perf_counter()
    if card_evidence_dir is None:
        receipts = [replay_games([game]).to_dict() for game in games]
    else:
        receipts = []
        references = []
        for game in games:
            evidence = capture_verified_game(GameTrace(data=game))
            references.append(retain_evidence(card_evidence_dir, evidence).model_dump())
            receipts.append(dict(evidence.replay))
        trace_receipt["card_evidence"] = references
    replay_payload = {
        name: sum(r[name] for r in receipts) for name in receipts[0] if name != "passed"
    }
    replay_payload["passed"] = all(r["passed"] for r in receipts)
    replay_payload["seconds"] = time.perf_counter() - replay_started
    for row, receipt in zip(rows, receipts, strict=True):
        row["replay_passed"] = receipt["passed"]
        row["trace_shard_sha256"] = trace_receipt["sha256"]
        row["integrity"]["replay_mismatches"] = sum(
            v for k, v in receipt.items() if k not in {"games", "decisions", "passed"}
        )
    return rows, trace_receipt, replay_payload
