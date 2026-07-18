"""Serialized matched-root latency, throughput, and isolated RSS evidence."""

from __future__ import annotations

import math
import multiprocessing
import platform
import resource
import threading
import time
from typing import Any

import numpy as np
import psutil

from .models import PlayerRegistration, ProfileRoots, canonical_sha256
from .players import build_player
from .replay import replay_prefix


def native_gameplay_profiles(
    rows: list[dict[str, Any]], *, worker_count: int = 1
) -> dict[str, Any]:
    """Aggregate contended match timing without treating it as promotion cost."""

    players: dict[str, dict[str, Any]] = {}
    for row in rows:
        for player_id, block in row.get("latency", {}).items():
            player = players.setdefault(
                player_id,
                {
                    "decisions": 0,
                    "decision_seconds": 0.0,
                    "games": 0,
                    "game_seconds": 0.0,
                    "cell_p50_seconds": [],
                    "cell_p95_seconds": [],
                },
            )
            player["decisions"] += int(block["count"])
            player["decision_seconds"] += float(block["seconds"])
            player["games"] += 1
            player["game_seconds"] += float(row["game_seconds"])
            if block["p50"] is not None:
                player["cell_p50_seconds"].append(float(block["p50"]))
                player["cell_p95_seconds"].append(float(block["p95"]))
    for player in players.values():
        player["decisions_per_second"] = (
            player["decisions"] / player["decision_seconds"]
            if player["decision_seconds"]
            else None
        )
        player["games_per_second"] = (
            player["games"] / player["game_seconds"] if player["game_seconds"] else None
        )
    return {
        "method": "native-contended-gameplay-v1",
        "promotion_authority": False,
        "worker_count": worker_count,
        "players": players,
    }


def select_profile_roots(
    games: list[dict[str, Any]], *, warmup: int, measured: int
) -> list[dict[str, Any]]:
    ordered_games = sorted(
        games,
        key=lambda game: (
            int(game["deal_seed"]),
            int(game["leg"]),
            str(game["match_id"]),
        ),
    )
    roots = [
        {
            "root_id": canonical_sha256(
                [game["game_trace_sha256"], int(decision["revision"])]
            ),
            "game": game,
            "revision": int(decision["revision"]),
        }
        for game in ordered_games
        for decision in game["decisions"]
    ]
    required = warmup + measured
    if len(roots) < required:
        raise ValueError(
            f"matched-root corpus has {len(roots)} roots; {required} are required"
        )
    return roots[:required]


def _ru_maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _profile_worker(
    connection: Any,
    registration_payload: dict[str, Any],
    roots: list[dict[str, Any]],
    warmup: int,
    checkpoint_path: str | None,
    sampler_interval_ms: int,
) -> None:
    try:
        import torch

        torch.set_num_threads(1)
        registration = PlayerRegistration.model_validate(registration_payload)
        process = psutil.Process()
        baseline_rss = int(process.memory_info().rss)
        player_seed = int(
            canonical_sha256([registration.player_id, "matched-root-profile-v1"])[:16],
            16,
        )
        player, observation_space = build_player(
            registration, seed=player_seed, checkpoint_path=checkpoint_path
        )
        post_load_rss = int(process.memory_info().rss)
        peak_rss = max(baseline_rss, post_load_rss)
        stop = threading.Event()

        def sample_rss() -> None:
            nonlocal peak_rss
            while not stop.wait(sampler_interval_ms / 1000.0):
                peak_rss = max(peak_rss, int(process.memory_info().rss))

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        samples = []
        root_mutations = 0
        illegal_actions = 0
        for root_index, root in enumerate(roots):
            env, observation = replay_prefix(
                root["game"], root["revision"], observation_space
            )
            pre_digest = env._engine.state_digest()
            stats = getattr(player, "stats", None)
            before_simulations = int(getattr(stats, "simulations", 0))
            before_cap_hits = int(getattr(stats, "cap_hits", 0))
            started = time.perf_counter()
            action = int(player.act(env, observation))
            elapsed = time.perf_counter() - started
            simulations = int(getattr(stats, "simulations", 0)) - before_simulations
            cap_hits = int(getattr(stats, "cap_hits", 0)) - before_cap_hits
            mutated = env._engine.state_digest() != pre_digest
            root_mutations += int(mutated)
            legal = 0 <= action < len(env.last_raw_obs.action_space.actions)
            illegal_actions += int(not legal)
            if root_index >= warmup:
                samples.append(
                    {
                        "root_id": root["root_id"],
                        "decision_ordinal": root_index,
                        "latency_seconds": elapsed,
                        "action": action,
                        "simulations": simulations,
                        "cap_hits": cap_hits,
                        "root_mutated": mutated,
                        "legal": legal,
                    }
                )
        stop.set()
        sampler.join(timeout=1.0)
        peak_rss = max(peak_rss, int(process.memory_info().rss), _ru_maxrss_bytes())
        connection.send(
            {
                "ok": True,
                "player_id": registration.player_id,
                "player_seed": player_seed,
                "checkpoint_bytes": registration.checkpoint_bytes,
                "parameter_count": registration.parameter_count,
                "samples": samples,
                "root_mutations": root_mutations,
                "illegal_actions": illegal_actions,
                "baseline_rss_bytes": baseline_rss,
                "post_load_rss_bytes": post_load_rss,
                "peak_rss_bytes": peak_rss,
                "ru_maxrss_bytes": _ru_maxrss_bytes(),
            }
        )
    except BaseException as error:
        connection.send(
            {"ok": False, "error": type(error).__name__, "message": str(error)}
        )
    finally:
        connection.close()


def _summarize_isolated(result: dict[str, Any]) -> dict[str, Any]:
    samples = result["samples"]
    latencies = np.asarray(
        [float(sample["latency_seconds"]) for sample in samples], dtype=np.float64
    )
    elapsed = float(np.sum(latencies))
    simulations = sum(int(sample["simulations"]) for sample in samples)
    cap_hits = sum(int(sample["cap_hits"]) for sample in samples)
    return {
        **result,
        "p50_seconds": float(np.percentile(latencies, 50)),
        "p95_seconds": float(np.percentile(latencies, 95)),
        "decisions_per_second": len(samples) / elapsed if elapsed else math.inf,
        "simulations_per_second": simulations / elapsed if elapsed else None,
        "playout_cap_rate": cap_hits / simulations if simulations else 0.0,
        "peak_rss_delta_bytes": max(
            0, int(result["peak_rss_bytes"]) - int(result["baseline_rss_bytes"])
        ),
    }


def profile_players(
    registrations: list[PlayerRegistration],
    *,
    source_games: list[dict[str, Any]],
    profile_roots: ProfileRoots,
    checkpoint_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Profile each registration in a fresh child process on identical roots."""

    checkpoint_paths = checkpoint_paths or {}
    roots = select_profile_roots(
        source_games, warmup=profile_roots.warmup, measured=profile_roots.measured
    )
    root_ids = [root["root_id"] for root in roots]
    context = multiprocessing.get_context("spawn")
    players = {}
    for registration in registrations:
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_profile_worker,
            args=(
                child,
                registration.model_dump(),
                roots,
                profile_roots.warmup,
                checkpoint_paths.get(registration.player_id),
                profile_roots.sampler_interval_ms,
            ),
        )
        process.start()
        child.close()
        result = parent.recv()
        process.join()
        if process.exitcode != 0 or not result.get("ok"):
            raise RuntimeError(
                f"matched-root profile failed for {registration.player_id}: "
                f"{result.get('error')}: {result.get('message')}"
            )
        result.pop("ok")
        players[registration.player_id] = _summarize_isolated(result)
    return {
        "schema_version": 1,
        "method": "isolated-matched-root-v1",
        "promotion_authority": True,
        "source_cell": profile_roots.source_cell,
        "selection": profile_roots.selection,
        "warmup_roots": profile_roots.warmup,
        "measured_roots": profile_roots.measured,
        "root_ids": root_ids,
        "root_corpus_sha256": canonical_sha256(root_ids),
        "sampler_interval_ms": profile_roots.sampler_interval_ms,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "torch_threads_per_player": 1,
        },
        "rss_limitations": "shared pages may be counted; sub-5ms spikes may be missed",
        "players": players,
    }


def verify_profile(profile: dict[str, Any]) -> None:
    """Recompute stored summaries from samples without invoking a player."""

    if profile.get("method") != "isolated-matched-root-v1":
        raise ValueError("profile is not matched-root authority")
    if canonical_sha256(profile["root_ids"]) != profile["root_corpus_sha256"]:
        raise ValueError("matched-root corpus digest mismatch")
    expected_count = int(profile["measured_roots"])
    warmup_count = int(profile["warmup_roots"])
    expected_root_ids = profile["root_ids"][warmup_count:]
    if len(expected_root_ids) != expected_count:
        raise ValueError("matched-root corpus count mismatch")
    corpus_ids: list[str] | None = None
    for player_id, stored in profile["players"].items():
        if len(stored["samples"]) != expected_count:
            raise ValueError(f"profile sample count mismatch: {player_id}")
        root_ids = [sample["root_id"] for sample in stored["samples"]]
        if root_ids != expected_root_ids:
            raise ValueError(f"profile root identity mismatch: {player_id}")
        if corpus_ids is None:
            corpus_ids = root_ids
        elif root_ids != corpus_ids:
            raise ValueError("players were not profiled on matched roots")
        recomputed = _summarize_isolated(
            {
                key: value
                for key, value in stored.items()
                if key
                not in {
                    "p50_seconds",
                    "p95_seconds",
                    "decisions_per_second",
                    "simulations_per_second",
                    "playout_cap_rate",
                    "peak_rss_delta_bytes",
                }
            }
        )
        for key in (
            "p50_seconds",
            "p95_seconds",
            "decisions_per_second",
            "simulations_per_second",
            "playout_cap_rate",
            "peak_rss_delta_bytes",
        ):
            if recomputed[key] != stored[key]:
                raise ValueError(f"profile summary mismatch: {player_id}/{key}")


def summarize_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility name for native outcome-worker summaries."""

    return native_gameplay_profiles(rows)
