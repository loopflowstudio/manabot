"""Project retained arena measurements into the existing experiment Perf surface.

No synthetic timings or policy executions occur here. A cell is a job/seed run;
shared serving receipts are shown once per allocation, with unallocated cost.
Trees partition sequential host phases only. Worker sums and allocation elapsed
are different resources, and request queue intervals live only on the timeline.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from manabot.sim.performance import allocation_cost
from manabot.sim.replay import read_trace


def _tree(
    name: str, elapsed: float, phases: dict[str, float], source: str
) -> dict[str, Any]:
    children = [
        {
            "name": key,
            "seconds": value,
            "status": "recorded",
            "category": "eval",
            "source": source,
        }
        for key, value in phases.items()
    ]
    remainder = elapsed - sum(phases.values())
    if remainder < -1e-6:
        raise ValueError("overlapping phases cannot partition a timing tree")
    children.append(
        {
            "name": "unattributed",
            "seconds": max(0, remainder),
            "status": "derived",
            "category": "unattributed",
            "source": "elapsed minus compatible measured phases",
            "note": "Includes unmeasured work and waits; no pure FFI or GPU execution attribution.",
        }
    )
    return {
        "name": name,
        "seconds": elapsed,
        "status": "recorded",
        "category": "run",
        "source": source,
        "children": children,
    }


def project_evaluations(directories: list[Path]) -> dict[str, Any]:
    if len({p.resolve() for p in directories}) != len(directories):
        raise ValueError("duplicate evaluation evidence is not an independent run")
    runs, timings, details, series = [], [], [], []
    allocations: dict[str, dict[str, Any]] = {}
    for index, directory in enumerate(directories):
        receipt_path = directory / "performance.json"
        receipt = json.loads(receipt_path.read_text())
        metadata_path = directory / "report-run.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        games = [
            game
            for path in sorted((directory / "traces").glob("*.commands.jsonl.gz"))
            for game in read_trace(path)
        ]
        rows = json.loads((directory / "rows.json").read_text())
        elapsed = receipt["elapsed_seconds"]
        counters = receipt["counters"]
        decisions = counters.get("decisions", 0)
        phases = {k: v["seconds"] for k, v in receipt["measurements"].items()}
        timings.append(
            {
                "run": index,
                "resource": "Cell elapsed",
                "decisions": decisions,
                "tree": _tree("cell", elapsed, phases, str(receipt_path)),
            }
        )
        worker_phases: dict[str, float] = defaultdict(float)
        worker_wall, worker_cpu = 0.0, 0.0
        worker_decisions = 0
        spans = list(receipt["spans"])
        coverage: dict[str, int] = defaultdict(int)
        for game in games:
            worker = game.get("worker_performance")
            if worker is None:
                continue
            worker_decisions += len(game["decisions"])
            worker_wall += worker["elapsed_seconds"]
            worker_cpu += worker["process_cpu_seconds"]
            spans.extend(worker["spans"])
            for name, measure in worker["measurements"].items():
                worker_phases[name] += measure["seconds"]
                coverage[name] += 1
        if worker_wall:
            timings.append(
                {
                    "run": index,
                    "resource": "Worker elapsed (sum, overlaps)",
                    "decisions": worker_decisions,
                    "tree": _tree(
                        "workers",
                        worker_wall,
                        dict(worker_phases),
                        "retained game worker_performance",
                    ),
                }
            )
            timings.append(
                {
                    "run": index,
                    "resource": "Worker CPU (sum)",
                    "decisions": worker_decisions,
                    "tree": _tree(
                        "process CPU",
                        worker_cpu,
                        {},
                        "process_time; no phase CPU attribution",
                    ),
                }
            )
        serving_path = metadata.get("serving_path")
        candidates = (
            [Path(serving_path)]
            if serving_path
            else [directory / "allocation" / "serving.json"]
        )
        allocation_ids = []
        for path in candidates:
            if not path.exists():
                continue
            serving = json.loads(path.read_text())
            identity = serving["identity"]
            allocation_ids.append(identity)
            if identity not in allocations:
                execution_path = path.parent / "execution.json"
                execution = (
                    json.loads(execution_path.read_text())
                    if execution_path.exists()
                    else {}
                )
                allocation_path = path.parent / "allocation.json"
                allocation = (
                    json.loads(allocation_path.read_text())
                    if allocation_path.exists()
                    else {}
                )
                serving_phases = {
                    k: v["seconds"]
                    for k, v in serving["measurements"].items()
                    if k != "batch_queue"
                }
                allocations[identity] = {
                    "identity": identity,
                    "serving": serving,
                    "elapsed_seconds": allocation.get("elapsed_seconds"),
                    "execution": execution.get("execution"),
                    "cost": execution.get(
                        "cost", allocation_cost(serving["elapsed_seconds"])
                    ),
                    "tree": _tree(
                        "serving", serving["elapsed_seconds"], serving_phases, str(path)
                    ),
                    "evidence": str(path),
                }

        def valid(row: dict[str, Any]) -> bool:
            return (
                row["failure"] is None
                and row["terminated"]
                and row["replay_passed"]
                and not any(row["integrity"].values())
            )

        valid_games = sum(valid(row) for row in rows)
        blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            blocks[row["deal_seed"]].append(row)
        expected_legs = counters.get("legs_per_block")
        valid_blocks = sum(
            len(block) == expected_legs and all(valid(row) for row in block)
            for block in blocks.values()
        )
        failure_count = sum(row["failure"] is not None for row in rows)
        mechanisms = {
            "native_step_and_encoding": "Profile native rules and encoding on these retained roots; inference rewrites do not remove this work.",
            "inference_round_trip": "Join request/batch spans to distinguish queue, IPC, transfers and forward execution before choosing a serving intervention.",
            "command_durable_acknowledgement": "Measure evidence write/fsync waits while preserving the pre-Command durability boundary.",
            "scalar_policy": "Compare residency, catalog preparation and batching with the same roots and complete panels.",
            "player_startup": "Amortize checkpoint loading over the admitted residence interval, charging cold load and idle capacity.",
        }
        serial = bool(games) and all("execution" not in g for g in games)
        opportunities = [
            {
                "phase": name,
                "observed_seconds": worker_phases[name],
                "coverage_games": coverage[name],
                "game_population": len(games),
                "mechanism": mechanism,
                "dependency": "Each game's next decision waits for this phase; concurrent allocation critical path is unmeasured.",
                "serial_elimination_bound": amdahl_bound(
                    worker_phases[name] / elapsed, float("inf")
                )
                if serial
                and coverage[name] == len(games)
                and worker_phases[name] < elapsed
                else None,
                "status": "serial cell bound"
                if serial and coverage[name] == len(games)
                else "concurrent hypothesis, not a wall-time speedup prediction",
            }
            for name, mechanism in mechanisms.items()
            if worker_phases.get(name, 0) > 0
        ]
        details.append(
            {
                "run": index,
                "evidence": str(receipt_path),
                "allocation_ids": allocation_ids,
                "valid_games": valid_games,
                "valid_blocks": valid_blocks,
                "games_per_wall_hour": valid_games * 3600 / elapsed,
                "blocks_per_wall_hour": valid_blocks * 3600 / elapsed,
                "games_per_dollar": None,
                "failed_games": failure_count,
                "unattempted_games": counters.get("unattempted_games", 0),
                "interrupted_games": sum(
                    row["termination_reason"] == "timeout" for row in rows
                ),
                "coverage": dict(coverage),
                "game_population": len(games),
                "spans": spans,
                "sample_coverage": receipt["sample_coverage"],
                "opportunities": opportunities,
                "optimization": {
                    "status": "hypothesis; no critical path inferred from overlapping totals",
                    "observation": "Compare native/encoding, durable acknowledgement, serving and IPC intervals using the timeline.",
                    "mechanism": "Amortize repeated preparation with residency/batching; profile engine or transfers only if they delay completion.",
                    "validation": "Interleaved equal-work serving_benchmark plan with declared tolerances and diagnostic traces.",
                    "coverage": "bounded host timeline; pure FFI, device execution and exact synchronization cost unavailable",
                    "evidence": str(receipt_path),
                },
            }
        )
        runs.append(
            {
                "id": str(directory),
                "label": metadata.get("job", directory.name),
                "job_name": metadata.get("job", directory.name),
                "seed": metadata.get("seed", index),
                "evals": [
                    {
                        "updates": 1,
                        "decisions": decisions,
                        "hours": elapsed / 3600,
                        "usd": None,
                    }
                ],
            }
        )
        series.append(
            {
                "run": index,
                "updates": [1],
                "metrics": {
                    "eval/games_per_wall_hour": [valid_games * 3600 / elapsed],
                    "eval/blocks_per_wall_hour": [valid_blocks * 3600 / elapsed],
                    "eval/failed_games": [failure_count],
                },
            }
        )
    for identity, allocation in allocations.items():
        cells = [r for r in details if identity in r["allocation_ids"]]
        seconds = allocation["elapsed_seconds"]
        valid_games = sum(r["valid_games"] for r in cells)
        valid_blocks = sum(r["valid_blocks"] for r in cells)
        cost = allocation["cost"]
        allocation.update(
            valid_games=valid_games,
            valid_blocks=valid_blocks,
            included_cells=len(cells),
            games_per_allocation_hour=valid_games * 3600 / seconds if seconds else None,
            blocks_per_allocation_hour=valid_blocks * 3600 / seconds
            if seconds
            else None,
            games_per_measured_dollar=valid_games / cost["measured_usd"]
            if cost["measured_usd"]
            else None,
            games_per_estimated_dollar=valid_games / cost["estimated_usd"]
            if cost["estimated_usd"]
            else None,
        )
    return {
        "id": "evaluation-performance",
        "title": "Evaluation performance",
        "question": "What did this evaluation capacity buy?",
        "source": "Recorded arena receipts",
        "synthetic": False,
        "decision_unit": "evaluation",
        "decks": [],
        "deck_names": [],
        "anchors": [],
        "runs": runs,
        "cells": [],
        "direct": [],
        "baseline": runs[0]["job_name"],
        "training": {"series": series},
        "timing": timings,
        "eval_performance": {
            "runs": details,
            "allocations": list(allocations.values()),
        },
        "context": {
            "status": "completed",
            "world": "recorded arena",
            "device": "see execution receipts",
            "heartbeat_at": None,
            "captured_at": None,
            "caveat": "Host launch timers are not GPU compute. Shared allocation cost is shown once and remains unallocated.",
        },
        "layout": {
            "clock": "updates",
            "evals": [],
            "ml": [],
            "perf": [{"kind": "flame", "title": "Evaluation time", "scale": "total"}],
        },
    }


def amdahl_bound(serial_fraction: float, phase_speedup: float) -> float:
    """Only applicable after evidence establishes serial, nonoverlapping work."""
    if not 0 <= serial_fraction <= 1 or phase_speedup < 1:
        raise ValueError("invalid serial fraction or phase speedup")
    denominator = 1 - serial_fraction + serial_fraction / phase_speedup
    return 1 / denominator if denominator else float("inf")
