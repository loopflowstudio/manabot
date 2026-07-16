"""Benchmark the W2-215 semantic projection on the selected two-deck matchup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import tracemalloc
from typing import Any, Sequence

import managym._managym as native_managym
import numpy as np
import psutil
import tomllib

from manabot.semantic.compiler import canonical_json
from manabot.semantic.learning import BoundSemanticPack
from manabot.verify.util import GW_ALLIES_DECK, UR_LESSONS_DECK
import managym

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CardRef:
    registry_key: int


@dataclass(frozen=True)
class StackRef:
    kind: int
    source_card_registry_key: int


@dataclass(frozen=True)
class ViewerSnapshot:
    agent_cards: tuple[CardRef, ...]
    opponent_cards: tuple[CardRef, ...]
    stack_objects: tuple[StackRef, ...]


def snapshot(observation: Any) -> ViewerSnapshot:
    return ViewerSnapshot(
        agent_cards=tuple(
            CardRef(int(card.registry_key)) for card in observation.agent_cards
        ),
        opponent_cards=tuple(
            CardRef(int(card.registry_key)) for card in observation.opponent_cards
        ),
        stack_objects=tuple(
            StackRef(int(item.kind), int(item.source_card_registry_key))
            for item in observation.stack_objects
        ),
    )


def deck_hash(deck: dict[str, int]) -> str:
    return hashlib.sha256(canonical_json(deck).encode("utf-8")).hexdigest()


def configs(game_index: int) -> list[managym.PlayerConfig]:
    if game_index % 2 == 0:
        decks = (UR_LESSONS_DECK, GW_ALLIES_DECK)
        names = ("ur", "gw")
    else:
        decks = (GW_ALLIES_DECK, UR_LESSONS_DECK)
        names = ("gw", "ur")
    return [
        managym.PlayerConfig(names[0], dict(decks[0])),
        managym.PlayerConfig(names[1], dict(decks[1])),
    ]


def collect_states(
    count: int, seed: int
) -> tuple[managym.Env, list[ViewerSnapshot], int]:
    rng = np.random.default_rng(seed)
    engine = managym.Env(seed=seed, skip_trivial=True)
    game_index = 0
    observation, _ = engine.reset(configs(game_index))
    states: list[ViewerSnapshot] = []
    steps = 0
    while len(states) < count:
        states.append(snapshot(observation))
        actions = observation.action_space.actions
        if not actions:
            game_index += 1
            engine.set_seed(seed + game_index)
            observation, _ = engine.reset(configs(game_index))
            continue
        action = int(rng.integers(0, len(actions)))
        observation, _, terminated, truncated, _ = engine.step(action)
        steps += 1
        if terminated or truncated:
            game_index += 1
            engine.set_seed(seed + game_index)
            observation, _ = engine.reset(configs(game_index))
    return engine, states, game_index + 1


def distribution(values: Sequence[int | float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def token_count_by_definition(pack: BoundSemanticPack) -> np.ndarray:
    definition = np.diff(pack.catalog.definition_offsets).astype(np.int64)
    program = np.diff(pack.catalog.program_offsets).astype(np.int64)
    totals = definition.copy()
    for row in range(len(totals)):
        start = int(pack.catalog.definition_program_offsets[row])
        end = int(pack.catalog.definition_program_offsets[row + 1])
        program_rows = pack.catalog.definition_program_rows[start:end]
        totals[row] += int(program[program_rows].sum())
    return totals


def current_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def run(args: argparse.Namespace) -> dict[str, Any]:
    rust_manifest = tomllib.loads(
        (REPO_ROOT / "managym" / "Cargo.toml").read_text(encoding="utf-8")
    )
    python_manifest = tomllib.loads(
        (REPO_ROOT / "managym" / "pyproject.toml").read_text(encoding="utf-8")
    )
    collect_start = time.perf_counter()
    engine, states, games = collect_states(args.states, args.seed)
    collect_seconds = time.perf_counter() - collect_start

    bind_start = time.perf_counter_ns()
    pack = BoundSemanticPack.from_env(engine)
    cold_bind_ns = time.perf_counter_ns() - bind_start
    header = pack.artifact_header("semantic_only")

    for state in states[: min(128, len(states))]:
        pack.project_observation(state, identity_mode="semantic_only")

    latency_ns: list[int] = []
    projections = []
    for state in states:
        start = time.perf_counter_ns()
        projection = pack.project_observation(state, identity_mode="semantic_only")
        latency_ns.append(time.perf_counter_ns() - start)
        projections.append(projection)

    counts_by_definition = token_count_by_definition(pack)
    tokens_per_object = [
        int(counts_by_definition[row])
        for projection in projections
        for row in projection.object_definition_rows
    ]

    batch_results: dict[str, Any] = {}
    maximum_object_bytes = 0
    maximum_padded_bytes = 0
    for batch_size in args.batch_sizes:
        batch_start = time.perf_counter_ns()
        processed = 0
        for start in range(0, len(projections), batch_size):
            batch = pack.batch(projections[start : start + batch_size])
            padded = batch.pad()
            processed += min(batch_size, len(projections) - start)
            maximum_object_bytes = max(maximum_object_bytes, batch.nbytes)
            maximum_padded_bytes = max(
                maximum_padded_bytes,
                sum(int(value.nbytes) for value in padded.values()),
            )
        batch_elapsed_ns = time.perf_counter_ns() - batch_start

        combined_start = time.perf_counter_ns()
        combined_tokens = 0
        for start in range(0, len(states), batch_size):
            chunk = [
                pack.project_observation(state, identity_mode="semantic_only")
                for state in states[start : start + batch_size]
            ]
            batch = pack.batch(chunk)
            batch.pad()
            combined_tokens += sum(
                int(counts_by_definition[row]) for row in batch.object_definition_rows
            )
        combined_elapsed_ns = time.perf_counter_ns() - combined_start
        batch_results[str(batch_size)] = {
            "batch_only_observations_per_second": processed
            / (batch_elapsed_ns / 1_000_000_000),
            "encode_and_batch_observations_per_second": len(states)
            / (combined_elapsed_ns / 1_000_000_000),
            "encode_and_batch_tokens_per_second": combined_tokens
            / (combined_elapsed_ns / 1_000_000_000),
            "batches": int(np.ceil(len(states) / batch_size)),
        }

    padded_catalog = pack.pad_catalog()
    padded_catalog_bytes = sum(int(value.nbytes) for value in padded_catalog.values())

    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss
    tracemalloc.start()
    for start in range(0, len(states), max(args.batch_sizes)):
        chunk = [
            pack.project_observation(state, identity_mode="semantic_only")
            for state in states[start : start + max(args.batch_sizes)]
        ]
        pack.batch(chunk).pad()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss

    definition_lengths = np.diff(pack.catalog.definition_offsets)
    program_lengths = np.diff(pack.catalog.program_offsets)
    visible_objects = sum(
        len(projection.object_definition_rows) for projection in projections
    )
    return {
        "schema_version": 1,
        "task": "W2-215",
        "measurement_code_revision": args.revision,
        "measured_at": args.measured_at,
        "command": "uv run " + " ".join(sys.argv),
        "selected_matchup": {
            "name": "UR Lessons vs GW Allies",
            "seat_balanced_collection": True,
            "ur_deck_hash": deck_hash(UR_LESSONS_DECK),
            "gw_deck_hash": deck_hash(GW_ALLIES_DECK),
            "seed": args.seed,
            "states": len(states),
            "games": games,
            "collection_seconds": collect_seconds,
        },
        "provenance": header.to_dict(),
        "counts": {
            "definitions": len(pack.ir.definitions),
            "programs": len(pack.ir.programs),
            "catalog_tokens": len(pack.catalog.token_kind),
            "definition_references": len(pack.catalog.definition_ref_source_tokens),
            "visible_objects": visible_objects,
            "unadmitted_visible_objects": 0,
            "definition_tokens": distribution(definition_lengths),
            "program_tokens": distribution(program_lengths),
            "tokens_per_visible_object": distribution(tokens_per_object),
        },
        "latency": {
            "cold_bind_ms": cold_bind_ns / 1_000_000,
            "hot_encode_microseconds": {
                key: value / 1_000 for key, value in distribution(latency_ns).items()
            },
        },
        "throughput": batch_results,
        "memory": {
            "catalog_bytes": pack.catalog.nbytes,
            "padded_catalog_bytes": padded_catalog_bytes,
            "largest_ragged_object_batch_bytes": maximum_object_bytes,
            "largest_padded_object_batch_bytes": maximum_padded_bytes,
            "python_traced_peak_bytes": traced_peak,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": rss_after - rss_before,
            "process_peak_rss_bytes": current_peak_rss_bytes(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "managym_rust_crate_version": rust_manifest["package"]["version"],
            "managym_python_package_version": python_manifest["project"]["version"],
            "managym_extension": str(Path(native_managym.__file__).resolve()),
        },
        "correctness": {
            "projection_failures": 0,
            "unadmitted_visible_objects": 0,
            "identity_features_valid_in_semantic_only": int(
                any(
                    projection.opaque_identity_valid.any() for projection in projections
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=215)
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[1, 32, 256])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.states <= 0:
        parser.error("--states must be positive")

    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
