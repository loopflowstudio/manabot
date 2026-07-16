"""Generate attributable, incrementally durable search self-play shards.

Workers write small atomic shards and matching summary sidecars. The parent
atomically updates a progress manifest after every completed shard. A resumed
run accepts only artifacts carrying the exact run fingerprint, so interruption
cannot silently mix teachers, seeds, code versions, or target contracts.

Usage:
    uv run experiments/runners/run_distill_datagen.py --games 480 --workers 4 \
        --sims 64 --games-per-shard 8 --out-dir .runs/exp03/dataset
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    from manabot.sim.distill import _git_commit as current_commit

    return current_commit()


def _normalized_teacher_spec(
    teacher_spec: dict[str, Any] | None, sims: int
) -> dict[str, Any]:
    if teacher_spec is None:
        return {"kind": "search", "sims": sims}
    return {key: value for key, value in teacher_spec.items() if key != "device"}


def _run_contract(
    args: argparse.Namespace, teacher_spec: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "games": args.games,
        "games_per_shard": args.games_per_shard,
        "round": args.round,
        "seed": args.seed,
        "sims": args.sims,
        "source_commit": _git_commit(),
        "teacher_spec": _normalized_teacher_spec(teacher_spec, args.sims),
    }


def _chunks(
    args: argparse.Namespace,
    teacher_spec: dict[str, Any] | None,
    run_fingerprint: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    offset = 0
    shard_index = 0
    while offset < args.games:
        chunk_games = min(args.games_per_shard, args.games - offset)
        chunks.append(
            {
                "num_games": chunk_games,
                "sims": args.sims,
                "teacher_spec": teacher_spec,
                "round_index": args.round,
                "seed": args.seed + shard_index * 1_000_000,
                "game_offset": offset,
                "out_path": str(out_dir / f"shard_{shard_index:05d}.npz"),
                "run_fingerprint": run_fingerprint,
                "shard_index": shard_index,
            }
        )
        offset += chunk_games
        shard_index += 1
    return chunks


def _worker(args: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from manabot.sim.distill import generate_selfplay_shard

    summary = generate_selfplay_shard(
        num_games=args["num_games"],
        sims=args["sims"],
        teacher_spec=args.get("teacher_spec"),
        seed=args["seed"],
        game_offset=args["game_offset"],
        out_path=args["out_path"],
        round_index=args.get("round_index", 0),
        dataset_run_fingerprint=args["run_fingerprint"],
    )
    shard_path = Path(args["out_path"])
    summary["sha256"] = _sha256(shard_path)
    summary["run_fingerprint"] = args["run_fingerprint"]
    summary["shard_index"] = args["shard_index"]
    _atomic_json(shard_path.with_suffix(".json"), summary)
    return summary


def _validate_summary(
    summary: dict[str, Any], chunk: dict[str, Any], shard_path: Path
) -> None:
    provenance = summary.get("provenance") or {}
    expected_summary = {
        "run_fingerprint": chunk["run_fingerprint"],
        "shard_index": chunk["shard_index"],
        "num_games": chunk["num_games"],
    }
    mismatches = [
        f"{key}={summary.get(key)!r} (expected {value!r})"
        for key, value in expected_summary.items()
        if summary.get(key) != value
    ]
    expected_provenance = {
        "dataset_run_fingerprint": chunk["run_fingerprint"],
        "game_offset": chunk["game_offset"],
        "num_games": chunk["num_games"],
        "round": chunk["round_index"],
        "seed": chunk["seed"],
        "teacher_spec": _normalized_teacher_spec(
            chunk.get("teacher_spec"), chunk["sims"]
        ),
    }
    mismatches.extend(
        f"provenance.{key}={provenance.get(key)!r} (expected {value!r})"
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    )
    if not shard_path.exists():
        mismatches.append("shard file is missing")
    elif summary.get("sha256") != _sha256(shard_path):
        mismatches.append("shard digest does not match its durable summary")
    if mismatches:
        raise SystemExit(f"cannot resume {shard_path.name}: " + "; ".join(mismatches))


def _completed_summaries(
    out_dir: Path, chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_shards = {Path(chunk["out_path"]).name for chunk in chunks}
    expected_summaries = {
        Path(name).with_suffix(".json").name for name in expected_shards
    }
    unexpected = sorted(
        path.name
        for path in [*out_dir.glob("shard_*.npz"), *out_dir.glob("shard_*.json")]
        if path.name not in expected_shards | expected_summaries
    )
    if unexpected:
        raise SystemExit(
            "dataset directory contains shards outside this run contract: "
            + ", ".join(unexpected)
        )

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for chunk in chunks:
        shard_path = Path(chunk["out_path"])
        summary_path = shard_path.with_suffix(".json")
        if shard_path.exists() != summary_path.exists():
            raise SystemExit(
                f"cannot resume {shard_path.name}: shard and durable summary "
                "must either both exist or both be absent"
            )
        if not shard_path.exists():
            pending.append(chunk)
            continue
        summary = json.loads(summary_path.read_text())
        _validate_summary(summary, chunk, shard_path)
        completed.append(summary)
    return completed, pending


def _record_progress(
    manifest: dict[str, Any], summaries: list[dict[str, Any]], total_shards: int
) -> None:
    manifest["shards"] = sorted(summaries, key=lambda item: item["shard_index"])
    manifest["progress"] = {
        "decisions_completed": int(sum(item["decisions"] for item in summaries)),
        "games_completed": int(sum(item["num_games"] for item in summaries)),
        "shards_completed": len(summaries),
        "shards_total": total_shards,
    }
    manifest["updated_at"] = datetime.now(UTC).isoformat()


def _base_manifest(
    args: argparse.Namespace,
    contract: dict[str, Any],
    run_fingerprint: str,
    summaries: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "running",
        "started_at": started_at,
        "run_contract": contract,
        "run_fingerprint": run_fingerprint,
        "teacher": summaries[0]["teacher"] if summaries else None,
        "provenance": summaries[0].get("provenance") if summaries else None,
        "policy_target_kind": (
            summaries[0].get("policy_target_kind") if summaries else None
        ),
        "value_target_kind": (
            summaries[0].get("value_target_kind") if summaries else None
        ),
        "round": args.round,
        "games": args.games,
        "games_per_shard": args.games_per_shard,
        "workers": args.workers,
        "seed": args.seed,
        "sims": args.sims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=480)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument(
        "--games-per-shard",
        type=int,
        default=8,
        help="durable checkpoint granularity; each completed shard is resumable",
    )
    parser.add_argument(
        "--teacher-json",
        type=str,
        default=None,
        help='full teacher spec as JSON, e.g. \'{"kind": "policy_search", '
        '"sims": 16, "checkpoint": "/abs/x.pt"}\' (overrides --sims)',
    )
    parser.add_argument(
        "--round", type=int, default=0, help="expert-iteration round tag"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=".runs/exp03/dataset")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only shards matching the exact recorded run fingerprint",
    )
    args = parser.parse_args()

    if args.games < 1 or args.workers < 1 or args.sims < 1:
        raise SystemExit("games, workers, and sims must be positive")
    if args.games_per_shard < 1:
        raise SystemExit("games-per-shard must be positive")

    os.environ.setdefault("WANDB_MODE", "disabled")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_spec = json.loads(args.teacher_json) if args.teacher_json else None
    contract = _run_contract(args, teacher_spec)
    run_fingerprint = _json_sha256(contract)
    chunks = _chunks(args, teacher_spec, run_fingerprint, out_dir)
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                "dataset manifest already exists; pass --resume or choose a new directory"
            )
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_fingerprint") != run_fingerprint:
            raise SystemExit(
                "dataset run fingerprint mismatch; refusing to mix stale shards"
            )
        started_at = previous.get("started_at") or datetime.now(UTC).isoformat()
    else:
        if args.resume:
            raise SystemExit("--resume requires an existing dataset manifest")
        if list(out_dir.glob("shard_*")):
            raise SystemExit("dataset shards exist without a run manifest")
        started_at = datetime.now(UTC).isoformat()

    summaries, pending = _completed_summaries(out_dir, chunks)
    manifest = _base_manifest(args, contract, run_fingerprint, summaries, started_at)
    _record_progress(manifest, summaries, len(chunks))
    _atomic_json(manifest_path, manifest)
    print(
        f"resume: {len(summaries)}/{len(chunks)} shards durable; "
        f"{len(pending)} pending",
        flush=True,
    )

    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    try:
        if pending:
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(pending)), mp_context=ctx
            ) as pool:
                futures = {pool.submit(_worker, chunk): chunk for chunk in pending}
                for future in as_completed(futures):
                    summaries.append(future.result())
                    _record_progress(manifest, summaries, len(chunks))
                    _atomic_json(manifest_path, manifest)
                    progress = manifest["progress"]
                    print(
                        f"progress: {progress['games_completed']}/{args.games} games; "
                        f"{progress['shards_completed']}/{len(chunks)} shards",
                        flush=True,
                    )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        _record_progress(manifest, summaries, len(chunks))
        _atomic_json(manifest_path, manifest)
        raise
    wall_seconds = time.perf_counter() - start

    summaries.sort(key=lambda item: item["shard_index"])
    engine_seconds = float(sum(item["wall_seconds"] for item in summaries))
    decisions = int(sum(item["decisions"] for item in summaries))
    steps = [step for item in summaries for step in item["steps_per_game"]]
    winners = [winner for item in summaries for winner in item["winners"]]
    manifest.update(
        status="completed",
        finished_at=datetime.now(UTC).isoformat(),
        teacher=summaries[0]["teacher"] if summaries else None,
        provenance=summaries[0].get("provenance") if summaries else None,
        policy_target_kind=(
            summaries[0].get("policy_target_kind") if summaries else None
        ),
        value_target_kind=(
            summaries[0].get("value_target_kind") if summaries else None
        ),
        games=int(sum(item["num_games"] for item in summaries)),
        decisions=decisions,
        wall_seconds=wall_seconds,
        engine_seconds=engine_seconds,
        engine_core_hours=engine_seconds / 3600.0,
        mean_steps_per_game=float(np.mean(steps)) if steps else 0.0,
        seat0_win_rate=(
            float(np.mean([winner == 0 for winner in winners])) if winners else 0.0
        ),
        search={
            key: float(sum(item["search"][key] for item in summaries))
            for key in ("decisions", "seconds", "simulations", "cap_hits")
        },
    )
    if summaries and "tree_nodes" in summaries[0]["search"]:
        for key in ("tree_nodes", "worlds_sampled", "max_depth_sum"):
            manifest["search"][key] = float(
                sum(item["search"][key] for item in summaries)
            )
        manifest["search"]["max_depth_max"] = float(
            max(item["search"]["max_depth_max"] for item in summaries)
        )
        manifest["search"]["mean_max_depth"] = manifest["search"][
            "max_depth_sum"
        ] / max(1.0, decisions)
    _record_progress(manifest, summaries, len(chunks))
    _atomic_json(manifest_path, manifest)
    print(
        f"done: {manifest['games']} games, {decisions} decisions, "
        f"wall {wall_seconds:.0f}s, engine {engine_seconds:.0f}s "
        f"({manifest['engine_core_hours']:.2f} core-hours) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
