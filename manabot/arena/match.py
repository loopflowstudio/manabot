"""Same-deal seat-paired arena matches with retained current Commands."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from etude.server import ASSET_MANIFEST_HASH, CONTENT_HASH
from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import MatchHypers, RewardHypers
from manabot.sim.teacher1_evidence import build_command, build_viewer_frame
from manabot.verify.util import INTERACTIVE_DECK, winner_from_info_or_obs

from .guidance import build_arena_player
from .models import ArenaKey, PlayerRegistration, canonical_sha256
from .replay import replay_games, write_trace


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


def _cell_id(first: str, second: str) -> str:
    return "__".join(sorted((first, second)))


def _trace_name(first: str, second: str) -> str:
    return f"{_cell_id(first, second)}.commands.jsonl.gz"


def play_cell(
    *,
    key: ArenaKey,
    player_a: PlayerRegistration,
    player_b: PlayerRegistration,
    deal_seeds: tuple[int, ...],
    out_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    comparison_seed_aliases: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    checkpoint_paths = checkpoint_paths or {}
    deal_seed_set_sha256 = canonical_sha256(list(deal_seeds))
    games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    pair = (player_a.player_id, player_b.player_id)
    for block, deal_seed in enumerate(deal_seeds):
        seeds = {
            registration.player_id: derive_seed(
                key,
                pair,
                deal_seed,
                registration.player_id,
                comparison_seed_aliases=comparison_seed_aliases,
            )
            for registration in (player_a, player_b)
        }
        for leg in (0, 1):
            seat_players = [player_a, player_b] if leg == 0 else [player_b, player_a]
            built = {
                registration.player_id: build_arena_player(
                    registration,
                    seed=seeds[registration.player_id],
                    checkpoint_path=checkpoint_paths.get(registration.player_id),
                )
                for registration in (player_a, player_b)
            }
            obs_space = (
                built[player_a.player_id][1]
                or built[player_b.player_id][1]
                or ObservationSpace()
            )
            match = Match(
                MatchHypers(
                    hero=seat_players[0].player_id,
                    villain=seat_players[1].player_id,
                    hero_deck=dict(INTERACTIVE_DECK),
                    villain_deck=dict(INTERACTIVE_DECK),
                )
            )
            env = Env(
                match,
                obs_space,
                Reward(RewardHypers()),
                seed=deal_seed,
                auto_reset=False,
            )
            obs, _ = env.reset(seed=deal_seed)
            match_id = f"{key.arena_version}:{_cell_id(*pair)}:{deal_seed}:{leg}"
            decisions: list[dict[str, Any]] = []
            latencies: dict[str, list[float]] = {
                player_a.player_id: [],
                player_b.player_id: [],
            }
            player_decision_ordinals = {
                player_a.player_id: 0,
                player_b.player_id: 0,
            }
            integrity = {
                "illegal_actions": 0,
                "truncations": 0,
                "root_mutations": 0,
                "private_exposures": 0,
                "offer_binding_failures": 0,
                "command_fabrications": 0,
                "replay_mismatches": 0,
            }
            info: dict[str, Any] = {}
            done = False
            revision = 0
            game_started = time.perf_counter()
            while not done:
                raw = env.last_raw_obs
                actor_seat = int(raw.agent.player_index)
                registration = seat_players[actor_seat]
                player = built[registration.player_id][0]
                frame = build_viewer_frame(
                    raw,
                    match_id=match_id,
                    revision=revision,
                    content_hash=CONTENT_HASH,
                    asset_manifest_hash=ASSET_MANIFEST_HASH,
                )
                if frame["projection"]["opponent"].get("hand"):
                    integrity["private_exposures"] += 1
                pre_digest = env._engine.state_digest()
                started = time.perf_counter()
                action = int(player.act(env, obs))
                elapsed = time.perf_counter() - started
                latencies[registration.player_id].append(elapsed)
                if env._engine.state_digest() != pre_digest:
                    integrity["root_mutations"] += 1
                legal = {int(offer["id"]) for offer in frame["offers"]}
                if action not in legal:
                    integrity["illegal_actions"] += 1
                    raise RuntimeError(
                        f"{registration.player_id} returned illegal offer {action}"
                    )
                command = build_command(frame, action)
                chosen_offer = next(
                    offer for offer in frame["offers"] if int(offer["id"]) == action
                )
                obs, _, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                integrity["truncations"] += int(
                    bool(
                        info.get("action_space_truncated")
                        or info.get("card_space_truncated")
                        or info.get("permanent_space_truncated")
                        or truncated
                    )
                )
                decisions.append(
                    {
                        "revision": revision,
                        "actor": actor_seat,
                        "player_id": registration.player_id,
                        "player_seed": seeds[registration.player_id],
                        "player_decision_ordinal": player_decision_ordinals[
                            registration.player_id
                        ],
                        "action_space_kind": frame["action_space"],
                        "frame_sha256": canonical_sha256(frame),
                        "command": command,
                        "command_sha256": canonical_sha256(command),
                        "chosen_offer": chosen_offer,
                        "pre_state_digest": pre_digest,
                        "post_state_digest": env._engine.state_digest(),
                        "latency_seconds": elapsed,
                    }
                )
                player_decision_ordinals[registration.player_id] += 1
                revision += 1
            game_seconds = time.perf_counter() - game_started
            winner = winner_from_info_or_obs(info, env.last_raw_obs)
            termination_reason = (
                "truncated" if truncated else "draw" if winner is None else "terminal"
            )
            game = {
                "match_id": match_id,
                "cell_id": _cell_id(*pair),
                "deal_block": block,
                "deal_seed": deal_seed,
                "leg": leg,
                "seat_players": [
                    registration.player_id for registration in seat_players
                ],
                "player_seeds": seeds,
                "winner": winner,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": termination_reason,
                "decisions": decisions,
                "integrity": integrity,
            }
            game["game_trace_sha256"] = canonical_sha256(game)
            games.append(game)
            player_a_seat = 0 if leg == 0 else 1
            score_a = 0.5 if winner is None else float(winner == player_a_seat)
            rows.append(
                {
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
                    "player_a_seat": player_a_seat,
                    "winner": winner,
                    "score_a": score_a,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": termination_reason,
                    "decisions": len(decisions),
                    "game_trace_sha256": game["game_trace_sha256"],
                    "trace_path": str(Path("traces") / _trace_name(*pair)),
                    "trace_sha256": game["game_trace_sha256"],
                    "replay_passed": False,
                    "integrity": integrity,
                    "latency": {
                        player_id: {
                            "count": len(values),
                            "seconds": float(sum(values)),
                            "p50": float(np.percentile(values, 50)) if values else None,
                            "p95": float(np.percentile(values, 95)) if values else None,
                        }
                        for player_id, values in latencies.items()
                    },
                    "game_seconds": game_seconds,
                }
            )
    trace_path = out_dir / "traces" / _trace_name(*pair)
    trace_receipt = write_trace(trace_path, games)
    trace_receipt["artifact_path"] = str(Path("traces") / _trace_name(*pair))
    replay = replay_games(games)
    replay_payload = replay.to_dict()
    mismatch_count = sum(
        int(value)
        for key, value in replay_payload.items()
        if key not in {"games", "decisions", "passed"}
    )
    for row in rows:
        row["replay_passed"] = replay.passed
        row["trace_shard_sha256"] = trace_receipt["sha256"]
        row["integrity"] = {
            **row["integrity"],
            "replay_mismatches": mismatch_count,
        }
    return rows, trace_receipt, replay_payload
