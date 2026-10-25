"""Bounded game execution and retained Commands, independent of evaluation policy."""

from __future__ import annotations

from collections.abc import Callable
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
import traceback
from typing import Any

from managym.decision import Command, DecisionFrame
from managym.experience import UNVERSIONED_CONTENT_HASH

from manabot.env import ObservationSpace
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.infra.provenance import (
    RECORDED_ASSET_MANIFEST_HASH,
    runtime_fingerprints,
    source_bundle_sha256,
)
from manabot.sim.guidance import build_registered_player
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import (
    apply_game_event,
    replay_environment,
)
from manabot.sim.search.local_update import LocalUpdatePlayer
from manabot.sim.search.puct import DeterminizedPuctPlayer
from manabot.sim.search.teacher1_evidence import build_command, build_viewer_frame


def _failure(
    game: dict[str, Any], reason: str, message: str, player_id: str | None
) -> None:
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
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    send: Callable[[tuple[str, Any]], None],
) -> None:
    import torch

    torch.set_num_threads(1)
    responsible = None
    try:
        built = {}
        for registration in registrations:
            responsible = registration.player_id
            send(("phase", responsible))
            built[responsible] = build_registered_player(
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
        game["observation_hypers"] = space.encoder.config.model_dump()
        send(("observation", game["observation_hypers"]))
        game["runtime_identities"] = runtime_fingerprints(
            game["deal_seed"],
            match_config=MatchConfig.model_validate(game["match_hypers"])
            if "match_hypers" in game
            else None,
            observation_space=space,
        )
        game["runtime_identities"]["arena_source_sha256"] = source_bundle_sha256(
            sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        )
        send(("identities", game["runtime_identities"]))
        env, obs = replay_environment(game, space)
        game["initial_state_digest"] = env._engine.state_digest()
        send(("initial", game["initial_state_digest"]))
        for seat, player_id in enumerate(game["seat_players"]):
            player = built[player_id][0]
            if isinstance(player, (LocalUpdatePlayer, DeterminizedPuctPlayer)):
                player.start_game(env, seat)
        local_players = [
            player
            for player, _ in built.values()
            if isinstance(player, LocalUpdatePlayer)
        ]
        decision_counts = dict.fromkeys(built, 0)
        for revision in range(max_commands):
            raw = env.last_raw_obs
            actor = int(raw.agent.player_index)
            player_id = game["seat_players"][actor]
            frame = build_viewer_frame(
                raw,
                match_id=game["match_id"],
                revision=revision,
                content_hash=UNVERSIONED_CONTENT_HASH,
                asset_manifest_hash=RECORDED_ASSET_MANIFEST_HASH,
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
                "latency_seconds": elapsed,
            }
            # The parent acknowledges durable pre-command evidence before native
            # execution. Successful Commands alone cannot reproduce a rejection.
            game["pending_decision"] = decision
            send(("pending", decision))
            if local_players:
                for player in local_players:
                    player.prepare_step(env, actor, action)
                semantic_frame = DecisionFrame.from_json(
                    env._engine.semantic_decision_frame_json()
                )
                semantic_command = Command(
                    f"arena-{revision}",
                    semantic_frame.revision,
                    int(semantic_frame.offers[action]["id"]),
                )
                obs, _, terminated, truncated, info, transition = env.step_semantic(
                    semantic_command
                )
            else:
                obs, _, terminated, truncated, info = env.step(action)
            decision["post_state_digest"] = env._engine.state_digest()
            player = built[player_id][0]
            if (
                isinstance(player, LocalUpdatePlayer)
                and player.last_receipt is not None
            ):
                # Arena traces are private. Advice uses the explicit public projection.
                decision["local_update_receipt"] = player.last_receipt.to_json()
                decision["latency_seconds"] += player.last_receipt.belief_seconds
                decision["belief_model_mismatch"] = (
                    "opponent_behavior_not_assumed_equal_to_frozen_policy"
                )
            game.pop("pending_decision", None)
            game["decisions"].append(decision)
            decision_counts[player_id] += 1
            send(("decision", decision))
            for player in local_players:
                player.observe_step(env, actor, transition)
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
    except BaseException as exc:
        if not isinstance(exc, Exception) and type(exc).__name__ != "PanicException":
            raise
        game["failure_trace"] = traceback.format_exc()
        game["error_class"] = type(exc).__name__
        _failure(game, "crash", f"{type(exc).__name__}: {exc}", responsible)
    send(("result", game))


def _game_worker(
    connection: Connection,
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
) -> None:
    def send(event: tuple[str, Any]) -> None:
        connection.send(event)
        if event[0] == "pending" and connection.recv() != "retained":
            raise RuntimeError("pre-command evidence was not retained")

    try:
        _execute_game(game, registrations, checkpoint_paths, max_commands, send)
    finally:
        connection.close()


def play_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    keep_completed: bool = False,
) -> dict[str, Any]:
    """Retain attempted Commands before native code can hang or kill its worker.

    The journal at `evidence_path` is written as the game runs, so a hung or
    killed worker (or a lost parent) still leaves its attempted Commands. A game
    that settles without failure deletes its journal unless `keep_completed`:
    its trace already holds every decision, and per-game journals dominate
    evaluation disk use.
    """
    started = time.monotonic()

    def retain(kind: str, value: Any) -> None:
        if evidence_path is not None:
            with evidence_path.open("a") as evidence:
                evidence.write(json.dumps([kind, value], allow_nan=False) + "\n")
                evidence.flush()
                os.fsync(evidence.fileno())

    game["execution_limits"] = {"commands": max_commands, "seconds": game_seconds}
    game["registrations"] = [
        registration.model_dump(mode="json") for registration in registrations
    ]
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.touch(mode=0o600, exist_ok=False)
        retain("game", game)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=True)
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
            retain(kind, value)
            game = apply_game_event(game, kind, value)
            if kind == "phase":
                responsible = value
            elif kind == "pending":
                try:
                    receiver.send("retained")
                except (BrokenPipeError, ConnectionResetError):
                    _failure(
                        game, "crash", "worker exited before acknowledgement", None
                    )
                    return game
            elif kind == "result":
                return game
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        receiver.close()
        game["execution_seconds"] = time.monotonic() - started
        retain("settled", game)
        if (
            evidence_path is not None
            and game["failure"] is None
            and not keep_completed
        ):
            evidence_path.unlink()
