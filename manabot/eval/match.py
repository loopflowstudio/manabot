"""Paired evaluation cells: seeded deck/seat schedules over bounded simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
from pydantic import TypeAdapter

from manabot.eval.game import play_game as play_in_process, replay_in_process
from manabot.eval.models import EvaluationKey
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.policy.protocol import BackendName
from manabot.sim.agent import GamePlayer
from manabot.sim.evaluation_records import finalize_game, set_execution_limits
from manabot.sim.match import play_game
from manabot.sim.matchup import (
    SELECTED_SUITE,
    AuthoredMatchup,
    DeckMatchup,
    selected_decks,
    selected_match,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import HISTORICAL_MIRROR_SUITE, PlayerRegistration
from manabot.sim.replay import ReplayReceipt, write_trace
from manabot.sim.supervision import failure_record, replay_retained
from manabot.verify.games import INTERACTIVE_DECK
from managym.recording import atomic_private_json
from managym.statistics import Reducer, Reduction

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
    seed_key = key.model_dump()
    if key.seed_compute_envelope_id is not None:
        seed_key.pop("seed_compute_envelope_id")
        seed_key["evaluation_compute_envelope_id"] = key.seed_compute_envelope_id
    identity = canonical_sha256(
        {
            "arena_key": seed_key,
            "pair": sorted(aliases.get(member, member) for member in pair),
            "deal_seed": deal_seed,
            "player_id": aliases.get(player_id, player_id),
        }
    )
    return int(identity[:16], 16)


@dataclass(frozen=True)
class CellWorkItem:
    game: dict[str, Any]
    registrations: list[PlayerRegistration]
    evidence_path: Path


@dataclass(frozen=True)
class CellPlan:
    pair: tuple[str, str]
    cell_id: str
    authored: bool
    played_legs: tuple[int, ...]
    trace_relative_path: str
    deal_seed_set_sha256: str
    work: list[CellWorkItem]


def build_cell_plan(
    *,
    key: EvaluationKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    comparison_seed_aliases: dict[str, str] | None = None,
    game_seconds: float = 120.0,
    max_commands: int = 10_000,
    matchup_mode: MatchupMode = "cross",
    authored_matchup: AuthoredMatchup | DeckMatchup | None = None,
    legs: tuple[int, ...] | None = None,
) -> CellPlan:
    """Validate cell registrations/schedule and construct per-leg game records."""
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
    if not authored and key.content_suite != HISTORICAL_MIRROR_SUITE:
        raise ValueError("unknown arena content suite")
    if not deal_seeds or len(set(deal_seeds)) != len(deal_seeds):
        raise ValueError("deal seeds must be nonempty and unique")
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
        for player in (player_a, player_b):
            if (
                player.content_suite != key.content_suite
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
    if any("execution" in p.player_spec for p in (player_a, player_b)):
        raise ValueError(
            "retained shared registrations require their recorded revision"
        )
    pair = (player_a.player_id, player_b.player_id)
    cell_id = "__".join(sorted(pair))
    trace_relative_path = str(Path("games") / f"{cell_id}.jsonl")
    deal_seed_set_sha256 = canonical_sha256(list(deal_seeds))
    work: list[CellWorkItem] = []
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
            game: dict[str, Any] = {
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
            work.append(
                CellWorkItem(
                    game=game,
                    registrations=seat_players,
                    evidence_path=out_dir
                    / "failures"
                    / f"{cell_id}.{deal_seed}.{leg}.jsonl",
                )
            )
    return CellPlan(
        pair=pair,
        cell_id=cell_id,
        authored=authored,
        played_legs=played_legs,
        trace_relative_path=trace_relative_path,
        deal_seed_set_sha256=deal_seed_set_sha256,
        work=work,
    )


def _evaluation_row(
    game: dict[str, Any],
    *,
    key: EvaluationKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    plan: CellPlan,
) -> dict[str, Any]:
    block, deal_seed, leg = game["deal_block"], game["deal_seed"], game["leg"]
    seeds = game["player_seeds"]
    seat_decks = game.get("seat_decks")
    seconds = game["execution_seconds"]
    if game["failure"] is not None:
        score_a = None
    else:
        score_a = 0.5 if game["winner"] is None else float(game["winner"] == leg % 2)
    row: dict[str, Any] = {
        "arena_key": key.model_dump(),
        "cell_id": game["cell_id"],
        "deal_block": block,
        "deal_seed": deal_seed,
        "deal_seed_set_sha256": plan.deal_seed_set_sha256,
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
        "trace_path": plan.trace_relative_path,
        "trace_sha256": game["game_trace_sha256"],
        "replay_passed": False,
        "integrity": deepcopy(game["integrity"]),
        "latency": {},
        "game_seconds": seconds,
    }
    if plan.authored:
        row["seat_decks"] = list(seat_decks)
    if (cutoff := game.get("policy_action_cutoff")) is not None:
        counts = cutoff["counts_by_kind"].values()
        row["policy_actions"] = {
            "decisions": sum(c["decisions"] for c in counts),
            "dropped_actions": sum(c["dropped_actions"] for c in counts),
        }
    for player_id in plan.pair:
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
    return row


def finalize_settled_cell(
    settled_games: Sequence[dict[str, Any]],
    *,
    plan: CellPlan,
    key: EvaluationKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    game_seconds: float = 120.0,
    card_evidence_dir: Path | None = None,
    in_process_replay: bool = True,
    deadline: float | None = None,
    reducers: Mapping[str, Reducer[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Replay, shard and persist an already-settled set of cell games."""
    reducers = dict(reducers or {})
    reduction = Reduction(tuple(reducers.values()))
    games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    replay_errors: list[str] = []
    card_evidence: list[dict[str, Any]] = []
    replayed_blocks: set[int] = set()
    replay_seconds = 0.0
    perf = EvalPerformance("cell")
    trace_path = out_dir / plan.trace_relative_path

    def replay_block(block: int) -> None:
        """Publish replayed rows only after their immutable shard is retained."""
        nonlocal replay_seconds
        block_games = sorted(
            (g for g in games if g["deal_block"] == block), key=lambda g: g["leg"]
        )
        block_rows = sorted(
            (r for r in rows if r["deal_block"] == block), key=lambda r: r["leg"]
        )
        # Shards sit below games/ so reading that directory yields each game once.
        relative = (
            Path("games")
            / "deals"
            / f"{plan.cell_id}.{block_games[0]['deal_seed']}.jsonl"
        )
        shard = write_trace(out_dir / relative, block_games)
        started = time.perf_counter()
        if in_process_replay:
            outcome = replay_in_process(
                block_games, card_evidence_dir, reduction if reducers else None
            )
        else:
            outcome = replay_retained(
                block_games,
                game_seconds * len(block_games)
                if deadline is None
                else min(
                    game_seconds * len(block_games), max(0, deadline - time.time())
                ),
                card_evidence_dir=card_evidence_dir,
                reducers=reduction.reducers,
            )
            if outcome.statistics_rows and reduction.error is None:
                for combined, reducer_rows in zip(
                    reduction.rows, outcome.statistics_rows, strict=True
                ):
                    combined.extend(reducer_rows)
        receipts = outcome.receipts
        if outcome.analysis_error is not None:
            reduction.fail(outcome.analysis_error)
        if outcome.error is not None:
            replay_errors.append(outcome.error)
        if receipts is None:
            receipts = [
                {**ReplayReceipt().to_dict(), "passed": False} for _ in block_games
            ]
        replay_receipts.extend(receipts)
        card_evidence.extend(ref.model_dump() for ref in outcome.card_evidence)
        for row, receipt in zip(block_rows, receipts, strict=True):
            row["trace_path"] = str(relative)
            row["trace_shard_sha256"] = shard["sha256"]
            row["replay_passed"] = receipt["passed"]
            row["integrity"]["replay_mismatches"] = sum(
                v
                for k, v in receipt.items()
                if k not in {"games", "decisions", "passed"}
            )
        replay_seconds += time.perf_counter() - started
        perf.record("replay", started)
        replayed_blocks.add(block)
        commits = out_dir / "blocks"
        commits.mkdir(exist_ok=True)
        atomic_json(
            commits / f"{block_games[0]['deal_seed']}.json",
            {
                "rows_sha256": canonical_sha256(block_rows),
            },
        )
        # Atomic snapshots contain only fully finalized blocks. A killed
        # coordinator cannot erase a previously validated complete block.
        atomic_json(
            out_dir / "rows.json",
            [r for r in rows if r["deal_block"] in replayed_blocks],
        )

    for game in settled_games:
        finalize_game(game)
        games.append(game)
        rows.append(
            _evaluation_row(
                game, key=key, player_a=player_a, player_b=player_b, plan=plan
            )
        )
        block = game["deal_block"]
        if sum(g["deal_block"] == block for g in games) == len(plan.played_legs):
            replay_block(block)

    if games:
        games.sort(key=lambda g: (g["deal_block"], g["leg"]))
        with perf.span("output_retention"):
            trace_receipt = write_trace(trace_path, games)
        trace_receipt["artifact_path"] = plan.trace_relative_path
    rows.sort(key=lambda r: (r["deal_block"], r["leg"]))
    attempted = {g["match_id"] for g in games}
    unattempted = [
        item.game["match_id"]
        for item in plan.work
        if item.game["match_id"] not in attempted
    ]
    perf.counters["unattempted_games"] = len(unattempted)
    atomic_json(out_dir / "unattempted.json", unattempted)
    if not games:
        perf.write(out_dir / "performance.json")
        raise RuntimeError("no game started")
    for block in sorted({g["deal_block"] for g in games} - replayed_blocks):
        replay_block(block)
    if card_evidence_dir is not None:
        trace_receipt["card_evidence"] = card_evidence
    replay_payload: dict[str, Any] = {
        name: sum(r[name] for r in replay_receipts)
        for name in replay_receipts[0]
        if name != "passed"
    }
    replay_payload["passed"] = all(r["passed"] for r in replay_receipts)
    replay_payload["seconds"] = replay_seconds
    if replay_errors:
        replay_payload["error"] = "; ".join(replay_errors)
    perf.count("expected_games", len(plan.work))
    perf.count("legs_per_block", len(plan.work) // len(deal_seeds))
    perf.count("attempted_games", len(rows))
    perf.count(
        "valid_games",
        sum(r["failure"] is None and r["replay_passed"] for r in rows),
    )
    perf.count("failed_games", sum(r["failure"] is not None for r in rows))
    perf.count("decisions", sum(r["decisions"] for r in rows))
    perf.write(out_dir / "performance.json")
    atomic_json(out_dir / "rows.json", rows)
    if reducers:
        tables = None
        if not replay_errors and reduction.error is None:
            try:
                tables = TypeAdapter(dict[str, list[Any]]).dump_python(
                    dict(zip(reducers, reduction.finish(), strict=True)),
                    mode="json",
                )
            except Exception as exc:
                reduction.fail(f"analysis table encoding: {type(exc).__name__}: {exc}")
        if reduction.error is not None:
            tables = None
        statistics_path = out_dir / "statistics.json"
        atomic_private_json(
            statistics_path,
            {
                "tables": tables,
                "games": len(rows),
                "complete_verified_games": sum(
                    r["failure"] is None and r["replay_passed"] and r["terminated"]
                    for r in rows
                ),
                "replay_passed": replay_payload["passed"],
                "error": reduction.error or replay_payload.get("error"),
                "analysis_error": reduction.error,
            },
        )
        trace_receipt["statistics_path"] = str(statistics_path)
    return rows, trace_receipt, replay_payload


def play_cell(
    *,
    key: EvaluationKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    backend_name: BackendName = "torch",
    comparison_seed_aliases: dict[str, str] | None = None,
    game_seconds: float = 120.0,
    max_commands: int = 10_000,
    matchup_mode: MatchupMode = "cross",
    authored_matchup: AuthoredMatchup | DeckMatchup | None = None,
    card_evidence_dir: Path | None = None,
    legs: tuple[int, ...] | None = None,
    keep_game_logs: bool = False,
    agents: Mapping[str, GamePlayer] | None = None,
    deadline: float | None = None,
    reducers: Mapping[str, Reducer[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Play complete deals or an explicit leg subset under bounded supervision.

    Card capture uses verified replay within the same deadline and retains the
    upstream evidence references for trace_card_tables.

    Each game's capture journal under `out_dir/failures/` is kept only when the
    game fails, unless `keep_game_logs`; `games/<cell-id>.jsonl` keeps every
    settled canonical record. Block shards under `games/deals/` preserve
    finalized progress. `reducers` run inside the verification replay; their
    tables go to the private `statistics.json`, and a reducer failure is an
    `analysis_error` that voids no game.

    `legs` selects a subset of the mode's legs for every deal, for designs that
    assign one seat and deck pair per game instead of playing whole deals.
    """
    checkpoint_paths = checkpoint_paths or {}
    reducers = dict(reducers or {})
    reduction = Reduction(tuple(reducers.values()))
    plan = build_cell_plan(
        key=key,
        player_a=player_a,
        player_b=player_b,
        deal_seeds=deal_seeds,
        out_dir=out_dir,
        comparison_seed_aliases=comparison_seed_aliases,
        game_seconds=game_seconds,
        max_commands=max_commands,
        matchup_mode=matchup_mode,
        authored_matchup=authored_matchup,
        legs=legs,
    )
    games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    replay_errors: list[str] = []
    card_evidence: list[dict[str, Any]] = []
    replayed_blocks: set[int] = set()
    replay_seconds = 0.0
    perf = EvalPerformance("cell")
    trace_path = out_dir / plan.trace_relative_path

    def run(item: CellWorkItem) -> dict[str, Any]:
        game, registrations, evidence = (
            item.game,
            item.registrations,
            item.evidence_path,
        )
        allowance = (
            game_seconds
            if deadline is None
            else min(game_seconds, deadline - time.time())
        )
        if allowance <= 0:
            set_execution_limits(game, registrations, max_commands, game_seconds)
            game["execution_seconds"] = 0.0
            failure_record(game, "timeout", "game wall-clock budget exhausted", None)
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(["settled", game]) + "\n")
            return game
        if agents is not None:
            return play_in_process(
                game,
                registrations,
                agents,
                max_commands,
                allowance,
                evidence,
                keep_completed=keep_game_logs,
            )
        return play_game(
            game,
            registrations,
            checkpoint_paths,
            max_commands,
            allowance,
            evidence,
            keep_completed=keep_game_logs,
            backend_name=backend_name,
        )

    def replay_block(block: int) -> None:
        """Publish replayed rows only after their immutable shard is retained."""
        nonlocal replay_seconds
        block_games = sorted(
            (g for g in games if g["deal_block"] == block), key=lambda g: g["leg"]
        )
        block_rows = sorted(
            (r for r in rows if r["deal_block"] == block), key=lambda r: r["leg"]
        )
        # Shards sit below games/ so reading that directory yields each game once.
        relative = (
            Path("games")
            / "deals"
            / f"{plan.cell_id}.{block_games[0]['deal_seed']}.jsonl"
        )
        shard = write_trace(out_dir / relative, block_games)
        started = time.perf_counter()
        if agents is not None:
            outcome = replay_in_process(
                block_games, card_evidence_dir, reduction if reducers else None
            )
        else:
            outcome = replay_retained(
                block_games,
                game_seconds * len(block_games)
                if deadline is None
                else min(
                    game_seconds * len(block_games), max(0, deadline - time.time())
                ),
                card_evidence_dir=card_evidence_dir,
                reducers=reduction.reducers,
            )
            if outcome.statistics_rows and reduction.error is None:
                for combined, reducer_rows in zip(
                    reduction.rows, outcome.statistics_rows, strict=True
                ):
                    combined.extend(reducer_rows)
        receipts = outcome.receipts
        if outcome.analysis_error is not None:
            reduction.fail(outcome.analysis_error)
        if outcome.error is not None:
            replay_errors.append(outcome.error)
        if receipts is None:
            receipts = [
                {**ReplayReceipt().to_dict(), "passed": False} for _ in block_games
            ]
        replay_receipts.extend(receipts)
        card_evidence.extend(ref.model_dump() for ref in outcome.card_evidence)
        for row, receipt in zip(block_rows, receipts, strict=True):
            row["trace_path"] = str(relative)
            row["trace_shard_sha256"] = shard["sha256"]
            row["replay_passed"] = receipt["passed"]
            row["integrity"]["replay_mismatches"] = sum(
                v
                for k, v in receipt.items()
                if k not in {"games", "decisions", "passed"}
            )
        replay_seconds += time.perf_counter() - started
        perf.record("replay", started)
        replayed_blocks.add(block)
        commits = out_dir / "blocks"
        commits.mkdir(exist_ok=True)
        atomic_json(
            commits / f"{block_games[0]['deal_seed']}.json",
            {
                "rows_sha256": canonical_sha256(block_rows),
            },
        )
        # Atomic snapshots contain only fully finalized blocks. A killed
        # coordinator cannot erase a previously validated complete block.
        atomic_json(
            out_dir / "rows.json",
            [r for r in rows if r["deal_block"] in replayed_blocks],
        )

    def finalize(game: dict[str, Any]) -> None:
        finalize_game(game)
        games.append(game)
        rows.append(
            _evaluation_row(
                game, key=key, player_a=player_a, player_b=player_b, plan=plan
            )
        )
        block = game["deal_block"]
        if sum(g["deal_block"] == block for g in games) == len(plan.played_legs):
            replay_block(block)

    try:
        for item in plan.work:
            finalize(run(item))
    finally:
        # Block shards and capture journals retain progress. Build the
        # diagnostic aggregate once, including on coordinator failure,
        # rather than recompressing the growing panel after every game.
        if games:
            games.sort(key=lambda g: (g["deal_block"], g["leg"]))
            with perf.span("output_retention"):
                trace_receipt = write_trace(trace_path, games)
            trace_receipt["artifact_path"] = plan.trace_relative_path
    rows.sort(key=lambda r: (r["deal_block"], r["leg"]))
    attempted = {g["match_id"] for g in games}
    unattempted = [
        item.game["match_id"]
        for item in plan.work
        if item.game["match_id"] not in attempted
    ]
    perf.counters["unattempted_games"] = len(unattempted)
    atomic_json(out_dir / "unattempted.json", unattempted)
    if not games:
        perf.write(out_dir / "performance.json")
        raise RuntimeError("no game started")
    # Incomplete blocks keep replay diagnostics too, but cannot contribute
    # to an evaluation until every requested leg is valid.
    for block in sorted({g["deal_block"] for g in games} - replayed_blocks):
        replay_block(block)
    if card_evidence_dir is not None:
        trace_receipt["card_evidence"] = card_evidence
    replay_payload = {
        name: sum(r[name] for r in replay_receipts)
        for name in replay_receipts[0]
        if name != "passed"
    }
    replay_payload["passed"] = all(r["passed"] for r in replay_receipts)
    replay_payload["seconds"] = replay_seconds
    if replay_errors:
        replay_payload["error"] = "; ".join(replay_errors)
    perf.count("expected_games", len(plan.work))
    perf.count("legs_per_block", len(plan.work) // len(deal_seeds))
    perf.count("attempted_games", len(rows))
    perf.count(
        "valid_games",
        sum(r["failure"] is None and r["replay_passed"] for r in rows),
    )
    perf.count("failed_games", sum(r["failure"] is not None for r in rows))
    perf.count("decisions", sum(r["decisions"] for r in rows))
    perf.write(out_dir / "performance.json")
    atomic_json(out_dir / "rows.json", rows)
    if reducers:
        # Tables contain private facts. Keep them separate from viewer
        # records, and aggregate once across all deal blocks.
        tables = None
        if not replay_errors and reduction.error is None:
            try:
                tables = TypeAdapter(dict[str, list[Any]]).dump_python(
                    dict(zip(reducers, reduction.finish(), strict=True)),
                    mode="json",
                )
            except Exception as exc:
                reduction.fail(f"analysis table encoding: {type(exc).__name__}: {exc}")
        if reduction.error is not None:
            tables = None
        statistics_path = out_dir / "statistics.json"
        atomic_private_json(
            statistics_path,
            {
                "tables": tables,
                "games": len(rows),
                "complete_verified_games": sum(
                    r["failure"] is None and r["replay_passed"] and r["terminated"]
                    for r in rows
                ),
                "replay_passed": replay_payload["passed"],
                "error": reduction.error or replay_payload.get("error"),
                "analysis_error": reduction.error,
            },
        )
        trace_receipt["statistics_path"] = str(statistics_path)
    return rows, trace_receipt, replay_payload
