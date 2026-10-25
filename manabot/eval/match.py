"""Paired evaluation cells: seeded deck/seat schedules over bounded simulation."""

from __future__ import annotations

from concurrent.futures import Future, as_completed
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import islice
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from manabot.eval.execution import bind_execution
from manabot.eval.models import EvaluationKey
from manabot.infra.config import MatchConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256
from manabot.sim.execution import EvalExecution
from manabot.sim.match import play_game
from manabot.sim.matchup import (
    SELECTED_SUITE,
    AuthoredMatchup,
    selected_decks,
    selected_match,
)
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import ReplayReceipt, write_trace
from manabot.sim.serving import (
    InferenceFailure,
    InferenceRuntime,
    PolicyAdmission,
)
from manabot.sim.supervision import replay_retained
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
    execution: EvalExecution | None = None,
    inference: InferenceRuntime | None = None,
    card_evidence_dir: Path | None = None,
    legs: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Play complete deals or an explicit leg subset under bounded supervision.

    Card capture uses verified replay within the same deadline and retains the
    upstream evidence references for trace_card_tables.
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
    if inference is not None:
        if execution is not None and execution != inference.execution:
            raise ValueError("conflicting execution declaration")
        execution = inference.execution
    if execution is not None:
        key, (player_a, player_b) = bind_execution(key, (player_a, player_b), execution)
    elif any("execution" in p.player_spec for p in (player_a, player_b)):
        raise ValueError("shared registration requires its declared runtime")
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
    games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    replay_errors: list[str] = []
    card_evidence: list[dict[str, Any]] = []
    replayed_blocks: set[int] = set()
    replay_seconds = 0.0
    work: list[tuple[dict[str, Any], list[PlayerRegistration], Path]] = []
    perf = EvalPerformance("cell", execution.instrumentation if execution else True)
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
            work.append(
                (
                    game,
                    seat_players,
                    out_dir / "failures" / f"{cell_id}.{deal_seed}.{leg}.jsonl",
                )
            )
    owner = (
        InferenceRuntime(execution, out_dir / "allocation")
        if execution is not None and inference is None
        else nullcontext(inference)
    )
    with owner as runtime:
        configs: dict[str, dict[str, Any]] = {}
        try:
            if runtime is not None:
                # Every deal repeats the same deck/seat legs. Admit each setup
                # once per policy, regardless of the panel's deal count.
                setups = tuple(
                    MatchConfig.model_validate(game["match_hypers"])
                    for game, _, _ in work[: len(played_legs)]
                )
                for registration in (player_a, player_b):
                    if registration.runner_kind == "checkpoint":
                        config = runtime.admit(
                            PolicyAdmission(
                                registration,
                                checkpoint_paths[registration.player_id],
                                setups,
                            )
                        )
                        configs[registration.player_id] = config.model_dump()

        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                out_dir / "admission-failure.json",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "key": key.model_dump(),
                    "registrations": [p.model_dump() for p in (player_a, player_b)],
                    "unattempted_games": [g["match_id"] for g, _, _ in work],
                },
            )
            raise

        def run(
            item: tuple[dict[str, Any], list[PlayerRegistration], Path],
        ) -> dict[str, Any]:
            game, registrations, evidence = item
            if runtime is not None:
                game["served_observations"] = configs
                game["execution"] = runtime.execution.model_dump()
                game["allocation_id"] = runtime.identity
                game["instrumentation"] = runtime.execution.instrumentation
            return play_game(
                game,
                registrations,
                checkpoint_paths,
                max_commands,
                game_seconds,
                evidence,
                inference=runtime,
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
            relative = (
                Path("traces")
                / f"{cell_id}.{block_games[0]['deal_seed']}.commands.jsonl.gz"
            )
            shard = write_trace(out_dir / relative, block_games)
            started = time.perf_counter()
            outcome = replay_retained(
                block_games,
                min(game_seconds * len(block_games), runtime.execution.remaining())
                if runtime is not None
                else game_seconds * len(block_games),
                card_evidence_dir=card_evidence_dir,
            )
            receipts = outcome.receipts
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
            block, deal_seed, leg = game["deal_block"], game["deal_seed"], game["leg"]
            seeds = game["player_seeds"]
            seat_decks = game.get("seat_decks")
            seconds = game["execution_seconds"]
            game["game_trace_sha256"] = canonical_sha256(game)
            games.append(game)
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
            if sum(g["deal_block"] == block for g in games) == len(played_legs):
                replay_block(block)

        try:
            if runtime is None:
                for item in work:
                    finalize(run(item))
            else:
                # Bounded submission: only active slots have futures. Never create a
                # thread or an IPC queue entry per unstarted game in a large panel.
                pending_work = iter(work)
                futures: set[Future[dict[str, Any]]] = set()
                for item in islice(pending_work, execution.max_active_games):
                    try:
                        futures.add(runtime.submit_game(partial(run, item)))
                    except InferenceFailure:
                        break
                while futures:
                    future = next(as_completed(futures))
                    futures.remove(future)
                    try:
                        finalize(future.result())
                    except InferenceFailure:
                        pass  # The final identity set counts all unattempted games.
                    # submit_game checks allocation health before admitting work.
                    item = next(pending_work, None)
                    if item is not None:
                        try:
                            futures.add(runtime.submit_game(partial(run, item)))
                        except InferenceFailure:
                            pass
        finally:
            # Block shards and Command journals retain progress. Build the
            # diagnostic aggregate once, including on coordinator failure,
            # rather than recompressing the growing panel after every game.
            if games:
                games.sort(key=lambda g: (g["deal_block"], g["leg"]))
                with perf.span("output_retention"):
                    trace_receipt = write_trace(trace_path, games)
                trace_receipt["artifact_path"] = trace_relative_path
        rows.sort(key=lambda r: (r["deal_block"], r["leg"]))
        attempted = {g["match_id"] for g in games}
        unattempted = [
            g["match_id"] for g, _, _ in work if g["match_id"] not in attempted
        ]
        perf.counters["unattempted_games"] = len(unattempted)
        atomic_json(out_dir / "unattempted.json", unattempted)
        if not games:
            perf.write(out_dir / "performance.json")
            raise InferenceFailure("allocation stopped before any game started")
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
        perf.count("expected_games", len(work))
        perf.count("legs_per_block", len(work) // len(deal_seeds))
        perf.count("attempted_games", len(rows))
        perf.count(
            "valid_games",
            sum(r["failure"] is None and r["replay_passed"] for r in rows),
        )
        perf.count("failed_games", sum(r["failure"] is not None for r in rows))
        perf.count("decisions", sum(r["decisions"] for r in rows))
        perf.write(out_dir / "performance.json")
        atomic_json(out_dir / "rows.json", rows)
        if runtime is not None:
            atomic_json(
                out_dir / "report-run.json",
                {
                    "job": cell_id,
                    "seed": deal_seeds[0],
                    "serving_path": str((runtime.directory / "serving.json").resolve()),
                },
            )
        return rows, trace_receipt, replay_payload
