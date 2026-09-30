"""Paired arena matches with bounded execution and retained current Commands."""

from __future__ import annotations

from copy import deepcopy
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from etude.server import ASSET_MANIFEST_HASH, CONTENT_HASH
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import MatchHypers
from manabot.sim.teacher1_evidence import build_command, build_viewer_frame
from manabot.verify.util import INTERACTIVE_DECK
from managym import WORLD_VERSION

from .guidance import build_arena_player
from .models import ArenaKey, PlayerRegistration, canonical_sha256
from .replay import replay_environment, replay_games, write_trace

SELECTED_SUITE = f"{WORLD_VERSION}-allies-lessons-v1"


def selected_match() -> MatchHypers:
    return MatchHypers.authored(
        "ur-lessons-vs-gw-allies",
        "ur_lessons",
        "gw_allies",
        hero="arena-seat-0",
        villain="arena-seat-1",
    )


def derive_seed(
    key: ArenaKey,
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


def _failure(game: dict, reason: str, message: str, player_id: str | None) -> None:
    game.setdefault("integrity", {})["execution_failures"] = 1
    game.update(
        termination_reason=reason,
        failure=message,
        failed_player_id=player_id,
        winner=None,
        terminated=False,
        truncated=reason == "truncated",
    )


def _execute_game(
    game: dict,
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    send,
) -> None:
    import torch

    torch.set_num_threads(1)
    responsible = None
    try:
        built = {}
        for registration in registrations:
            responsible = registration.player_id
            send(("phase", responsible))
            built[responsible] = build_arena_player(
                registration,
                seed=game["player_seeds"][responsible],
                checkpoint_path=checkpoint_paths.get(responsible),
            )
        responsible = None
        send(("phase", None))
        spaces = [space for _, space in built.values() if space is not None]
        if spaces and any(space.shapes != spaces[0].shapes for space in spaces):
            raise ValueError("players have different observation ABIs")
        space = spaces[0] if spaces else ObservationSpace()
        game["observation_hypers"] = space.encoder.hypers.model_dump()
        send(("observation", game["observation_hypers"]))
        env, obs = replay_environment(game, space)
        game["initial_state_digest"] = env._engine.state_digest()
        send(("initial", game["initial_state_digest"]))
        decision_counts = dict.fromkeys(built, 0)
        for revision in range(max_commands):
            raw = env.last_raw_obs
            actor = int(raw.agent.player_index)
            player_id = game["seat_players"][actor]
            frame = build_viewer_frame(
                raw,
                match_id=game["match_id"],
                revision=revision,
                content_hash=CONTENT_HASH,
                asset_manifest_hash=ASSET_MANIFEST_HASH,
            )
            if frame["projection"]["opponent"].get("hand"):
                game["integrity"]["private_exposures"] += 1
                raise RuntimeError("viewer frame exposes a private hand")
            pre_digest = env._engine.state_digest()
            responsible = player_id
            send(("phase", player_id))
            ordinal = decision_counts[player_id]
            policy_seed = (game["player_seeds"][player_id] + ordinal) % (2**63)
            if registrations[actor].runner_kind == "checkpoint":
                torch.manual_seed(policy_seed)
            started = time.perf_counter()
            action = int(built[player_id][0].act(env, obs))
            elapsed = time.perf_counter() - started
            if env._engine.state_digest() != pre_digest:
                game["integrity"]["root_mutations"] += 1
                raise RuntimeError("player mutated the authoritative root")
            offers = {int(offer["id"]): offer for offer in frame["offers"]}
            if action not in offers:
                game["integrity"]["illegal_actions"] += 1
                raise RuntimeError(f"illegal offer {action}")
            command = build_command(frame, action)
            responsible = None
            send(("phase", None))
            obs, _, terminated, truncated, info = env.step(action)
            decision = {
                "revision": revision,
                "actor": actor,
                "player_id": player_id,
                "player_seed": game["player_seeds"][player_id],
                "player_decision_ordinal": ordinal,
                "policy_rng_seed": policy_seed,
                "action_space_kind": frame["action_space"],
                "frame_sha256": canonical_sha256(frame),
                "command": command,
                "command_sha256": canonical_sha256(command),
                "chosen_offer": offers[action],
                "pre_state_digest": pre_digest,
                "post_state_digest": env._engine.state_digest(),
                "latency_seconds": elapsed,
            }
            game["decisions"].append(decision)
            decision_counts[player_id] += 1
            send(("decision", decision))
            if truncated or any(
                info.get(name)
                for name in (
                    "action_space_truncated",
                    "card_space_truncated",
                    "permanent_space_truncated",
                )
            ):
                game["integrity"]["truncations"] += 1
                _failure(
                    game, "truncated", "environment or observation truncated", None
                )
                break
            if terminated:
                winner = env._engine.winner_index()
                game.update(
                    winner=winner,
                    terminated=True,
                    termination_reason="draw" if winner is None else "terminal",
                )
                break
        else:
            _failure(game, "command_cap", "game exceeded its Command cap", None)
    except Exception as exc:
        _failure(game, "crash", f"{type(exc).__name__}: {exc}", responsible)
    send(("result", game))


def _game_worker(connection, game, registrations, checkpoint_paths, max_commands):
    try:
        _execute_game(
            game, registrations, checkpoint_paths, max_commands, connection.send
        )
    finally:
        connection.close()


def _bounded_game(
    game: dict,
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
) -> dict:
    """Keep completed Commands even if native code hangs or the worker dies."""
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_game_worker,
        args=(
            sender,
            game,
            registrations,
            checkpoint_paths,
            max_commands,
        ),
    )
    responsible = None
    started = time.monotonic()
    process.start()
    sender.close()
    try:
        while True:
            remaining = game_seconds - (time.monotonic() - started)
            if remaining <= 0:
                _failure(game, "timeout", "game wall-clock budget exhausted", None)
                return game
            if not receiver.poll(min(remaining, 0.1)):
                if not process.is_alive():
                    _failure(
                        game, "crash", f"worker exited {process.exitcode}", responsible
                    )
                    return game
                continue
            try:
                kind, value = receiver.recv()
            except EOFError:
                _failure(game, "crash", "worker exited without a result", responsible)
                return game
            if kind == "phase":
                responsible = value
            elif kind == "initial":
                game["initial_state_digest"] = value
            elif kind == "observation":
                game["observation_hypers"] = value
            elif kind == "decision":
                game["decisions"].append(value)
            elif kind == "result":
                return value
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        receiver.close()


def play_cell(
    *,
    key: ArenaKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    comparison_seed_aliases: dict[str, str] | None = None,
    game_seconds: float = 120.0,
    max_commands: int = 10_000,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not math.isfinite(game_seconds) or game_seconds <= 0 or max_commands < 1:
        raise ValueError("game limits must be finite and positive")
    if player_a.player_id == player_b.player_id:
        raise ValueError("arena players must be distinct")
    selected = key.content_suite == SELECTED_SUITE
    if not selected and key.content_suite != "w2-interactive-mirror-v1":
        raise ValueError("unknown arena content suite")
    if not deal_seeds or len(set(deal_seeds)) != len(deal_seeds):
        raise ValueError("deal seeds must be nonempty and unique")
    checkpoint_paths = checkpoint_paths or {}
    match = (
        selected_match()
        if selected
        else MatchHypers(
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    if selected:
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
        for leg in range(4 if selected else 2):
            seat_players = (
                [player_a, player_b] if leg % 2 == 0 else [player_b, player_a]
            )
            # Same per-seat setup/deal within each pair; reverse deck starting seats
            # for the second pair. Each player gets every deck/seat combination.
            setup = (
                Match(match).swapped().hypers
                if leg >= 2
                else match.model_copy(deep=True)
            )
            seat_decks = ["ur_lessons", "gw_allies"]
            if leg >= 2:
                seat_decks.reverse()
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
            if selected:
                game["seat_decks"] = seat_decks
            started = time.perf_counter()
            game = _bounded_game(
                game, seat_players, checkpoint_paths, max_commands, game_seconds
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
            if selected:
                row["seat_decks"] = seat_decks
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
    receipts = [replay_games([game]).to_dict() for game in games]
    replay_payload = {
        name: sum(r[name] for r in receipts) for name in receipts[0] if name != "passed"
    }
    replay_payload["passed"] = all(r["passed"] for r in receipts)
    for row, receipt in zip(rows, receipts, strict=True):
        row["replay_passed"] = receipt["passed"]
        row["trace_shard_sha256"] = trace_receipt["sha256"]
        row["integrity"]["replay_mismatches"] = sum(
            v for k, v in receipt.items() if k not in {"games", "decisions", "passed"}
        )
    return rows, trace_receipt, replay_payload
