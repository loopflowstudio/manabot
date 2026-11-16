"""Controlled storage accounting at retained census counts, without game load.

This is a size proof, not a throughput or CUDA admission benchmark. It builds
one valid synthetic prefix at the measured maxima and measures actual ownership,
selected raw gather, and host preparation. Larger envelopes are arithmetic;
no full padded rollout or resource-pressure fixture is allocated.
"""

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
from typing import Literal

import numpy as np
import psutil
import torch

from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match
from manabot.env.memory_store import (
    MEMORY_FIELDS,
    CompactObservation,
    MemoryStore,
    Prefix,
    PrefixIdentity,
    PrefixObservation,
    archive_memory,
    gather_bytes,
    restore_memory,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
import managym


def residency(device: torch.device) -> dict[str, int | None]:
    """Phase-end process-tree RSS plus process high-water RSS; never add CUDA."""
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    rss = sum(p.memory_info().rss for p in processes if p.is_running())
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "host_tree_rss_bytes": rss,
        "host_self_peak_rss_bytes": int(
            high_water * (1 if sys.platform == "darwin" else 1024)
        ),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }


def execution_proof(
    space: ObservationSpace,
    retained: PrefixObservation,
    device: torch.device,
) -> list[dict[str, object]]:
    """One complete maximum prefix per forward, four arms, sequentially.

    This bounds execution without changing the experiment's streams or optimizer
    batches. Full job sizing remains separate. CPU high-water is cumulative;
    CUDA peaks reset per arm and include model, input, activations and Adam.
    """
    results: list[dict[str, object]] = []
    encoders: tuple[Literal["transformer", "gru"], ...] = ("transformer", "gru")
    for encoder in encoders:
        for summaries in (1, 4):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            phases = {"before_model": residency(device)}
            model = ActorCritic(
                space,
                ModelConfig(
                    world=managym.WORLD_SETUP,
                    hidden_dim=64,
                    opponent_memory=encoder,
                    opponent_memory_summaries=summaries,
                    attention_on=True,
                    attention_layers=2,
                    policy_film=True,
                    value_aggregation="masked_mean",
                ),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            phases["model"] = residency(device)
            # Admission and preparation see only temporary dense input; the next
            # observation owner continues to contain ordinary fields + prefixes.
            host = model.prepare(retained.materialize())
            phases["prepared"] = residency(device)
            prefetched = model.prepare(retained.materialize())
            phases["two_host_prepared"] = residency(device)
            prepared = on_device(host, device)
            phases["device_input"] = residency(device)
            logits, value = model.forward_distribution(prepared)
            legal_logits = logits[prepared.legal > 0]
            assert torch.isfinite(legal_logits).all() and torch.isfinite(value).all()
            loss = legal_logits.square().mean() + value.square().mean()
            phases["forward"] = residency(device)
            loss.backward()
            assert model.opponent_memory is not None
            gradients = [
                p.grad for p in model.opponent_memory.parameters() if p.grad is not None
            ]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(g.abs().sum() > 0 for g in gradients)
            phases["backward"] = residency(device)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            phases["optimizer"] = residency(device)
            assert not MEMORY_FIELDS.intersection(retained)
            results.append(
                {
                    "encoder": encoder,
                    "summaries": summaries,
                    "execution_rows": len(retained.prefixes),
                    "host_prepared_batches": 2,
                    "finite_forward_backward_adam": True,
                    "phases": phases,
                }
            )
            del (
                loss,
                legal_logits,
                logits,
                value,
                gradients,
                prepared,
                host,
                prefetched,
                optimizer,
                model,
            )
    return results


def measure(
    source: Path,
    *,
    streams: int = 256,
    selected_rows: int = 256,
    device_name: Literal["cpu", "cuda"] = "cpu",
) -> dict[str, object]:
    """Use retained counts and a valid controlled fixture; never claim replay bytes."""
    device = torch.device(device_name)
    census = json.loads(source.read_text())
    record_count = int(census["terminal_records"]["maximum"])
    detail_count = int(census["terminal_details"]["maximum"])
    capacities = (int(census["record_capacity"]), int(census["detail_capacity"]))
    space = ObservationSpace(
        ObservationConfig(
            max_opponent_records=capacities[0], max_opponent_details=capacities[1]
        )
    )
    env = managym.Env(seed=111)
    env.reset(Match().to_rust())
    buffers = {
        key: value[None]
        for key, value in space.encode(
            env.observation_for_player(env.current_agent_index())
        ).items()
    }
    headers, details = buffers["opponent_records"][0], buffers["opponent_details"][0]
    headers.fill(0)
    details.fill(0)

    def words(value: int) -> tuple[int, int]:
        return value >> 16, value & 65535

    kind = words(managym.catalog_token_value("OpponentMemory::Priority"))
    field = words(managym.catalog_token_value("OpponentMemory::Life"))
    bounds = np.linspace(0, detail_count, record_count + 1, dtype=np.int64)
    for ordinal, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        headers[ordinal] = [
            *words(ordinal),
            0,
            1,
            0,
            3,
            *words(ordinal),
            *words(ordinal),
            1,
            3,
            *kind,
            start,
            stop - start,
            2,
            1,
        ]
    details[:detail_count, 2:4] = field
    details[:detail_count, 4] = -1
    # Distinct exact numeric words prevent interning an unrealistically repeated
    # record body; these are synthetic facts, not retained gameplay evidence.
    details[:detail_count, 5] = np.arange(detail_count) >> 16
    details[:detail_count, 6] = np.arange(detail_count) & 65535
    store = MemoryStore(capacities)
    start_time = time.perf_counter()
    prefix = store.capture(
        buffers, 0, PrefixIdentity("controlled-census-max", 0, "root")
    )
    capture_seconds = time.perf_counter() - start_time
    # Retain all intermediate prefix versions, then 64 decisions per stream.
    tails = []
    tail = prefix.tail
    while tail is not None:
        tails.append(tail)
        tail = tail.previous
    references = [
        Prefix(
            PrefixIdentity(f"controlled-{row // 64}", row % 2, "root"),
            tails[row % len(tails)],
        )
        for row in range(streams * 64)
    ]
    ownership = store.measure(references)
    ordinary = {k: v for k, v in buffers.items() if k not in MEMORY_FIELDS}
    tracemalloc.start()
    compact = CompactObservation(ordinary, [prefix], capacities)
    gather_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    torch.set_num_threads(1)
    model = ActorCritic(
        space,
        ModelConfig(
            world=managym.WORLD_SETUP,
            hidden_dim=64,
            opponent_memory="gru",
            opponent_memory_summaries=4,
            attention_on=True,
            attention_layers=2,
            policy_film=True,
            value_aggregation="masked_mean",
        ),
    )
    tracemalloc.start()
    started = time.perf_counter()
    prepared = model.prepare(compact)
    prepare_seconds = time.perf_counter() - started
    preparation_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    prepared_arrays = array_fields(prepared, np.ndarray)
    memory_arrays = array_fields(prepared.memory, np.ndarray)
    prepared_bytes = sum(a.nbytes for a in memory_arrays.values())
    incident = archive_memory([prefix] * 16, capacities)
    restored = restore_memory(incident)
    assert all(p.tail is not None and p.tail.key == prefix.tail.key for p in restored)
    terminal_details = 0
    terminal_records = 0
    receipt = source.with_name("native.jsonl")
    for line in receipt.read_text().splitlines():
        game = json.loads(line)
        for terminal in game.get("terminal", []):
            terminal_details += terminal["details"]
            terminal_records += terminal["records"]
    native_bytes = streams * sum(buffers[k].nbytes for k in MEMORY_FIELDS)
    gather_selected = gather_bytes([prefix] * selected_rows)
    # At the preparation overlap: prior Prepared + next raw and Prepared.
    # Temporaries, ordinary arrays and activations remain separate, not a fit.
    host_two = gather_selected + 2 * selected_rows * prepared_bytes
    device_input = selected_rows * prepared_bytes
    host_owned = (
        native_bytes
        + host_two
        + sum(
            ownership[key]
            for key in (
                "unique_chunk_bytes",
                "python_metadata_bytes",
                "intern_index_bytes",
            )
        )
    )
    memory_array_bytes = {k: a.nbytes for k, a in memory_arrays.items()}
    all_field_bytes = sum(a.nbytes for a in prepared_arrays.values())
    retained = PrefixObservation(ordinary, [prefix], capacities)
    del compact, prepared, prepared_arrays, memory_arrays, model
    execution = execution_proof(space, retained, device)
    return {
        "source_sha256": {
            name: sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "manabot/env/memory_store.py",
                "scripts/verify_opponent_memory_storage.py",
                "manabot/policy/opponent_memory.py",
                "manabot/policy/prepare.py",
                "manabot/policy/admission.py",
                "manabot/training/rollout/collection.py",
                "manabot/backends/torch/state.py",
                "manabot/backends/torch/rollout.py",
                "manabot/sim/rollout.py",
            )
        },
        "fixture": "controlled valid distinct facts at census terminal maxima; no replay corpus",
        "census": {
            "games": census["games"],
            "decisions": census["decisions"],
            "step_digest": census["step_digest"],
        },
        "capacities": list(capacities),
        "records": record_count,
        "details": detail_count,
        "streams": streams,
        "selected_rows": selected_rows,
        "retained_fixture": ownership,
        "census_terminal_payload_bytes_without_interning": terminal_records * 72
        + terminal_details * 32,
        "capture_seconds": capture_seconds,
        "gather_one_peak_bytes": gather_peak,
        "prepare_one_peak_bytes": preparation_peak,
        "prepare_seconds": prepare_seconds,
        "prepared_memory_array_bytes": memory_array_bytes,
        "prepared_memory_bytes_one": prepared_bytes,
        "prepared_all_fields_bytes_one": all_field_bytes,
        "native_memory_buffers_bytes": native_bytes,
        "gather_selected_worst_case_bytes": gather_selected,
        "host_two_batches_worst_case_bytes": host_two,
        "device_memory_inputs_worst_case_bytes": device_input,
        "incident_16_complete_prefixes_bytes": incident.nbytes,
        "incident_within_32_mib_evidence_budget": incident.nbytes <= 32 * 1024**2,
        "archive_identity_round_trip": True,
        "host_owned_memory_components_bytes": host_owned,
        "host_limit_bytes": 40 * 1024**3,
        "device_one_width64_detail_activation_bytes": selected_rows
        * detail_count
        * 64
        * 4,
        "device": device_name,
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "device_limit_bytes": torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
        "execution_proof": execution,
        "execution_row_counts": {
            "controlled_forward_backward": 1,
            "controlled_host_prefetch_batches": 2,
            "collector_recovery_test_streams": 1,
            "collector_recovery_test_quota": 16,
            "real_update_test_learner_rows": 4,
            "pilot_streams_not_executed": streams,
            "pilot_selected_rows_not_executed": selected_rows,
        },
        "accounting_note": "Host ownership arithmetic excludes temporary preparation and activation peaks; never summed with device bytes or used to admit a job. Phase RSS samples and cumulative self high-water are measured independently. CUDA peaks are null on CPU. Fixture shares tails and does not estimate independent games. Bootstrap raw arrays expire after preparation; two learner Prepared host batches can coexist with one temporary raw gather. No rollout schedule changed.",
        "excluded": [
            "other native/rollout fields",
            "temporary preparation beyond retained arrays",
            "model and optimizer",
            "attention/activations/gradients",
            "CUDA allocator and transfer staging",
            "native game logs",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--selected-rows", type=int, default=256)
    args = parser.parse_args()
    if min(args.streams, args.selected_rows) < 1:
        parser.error("stream and selected row counts must be positive")
    result = measure(
        args.source,
        streams=args.streams,
        selected_rows=args.selected_rows,
        device_name=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "host_owned_memory_components_bytes",
                    "host_limit_bytes",
                    "device",
                    "device_limit_bytes",
                    "archive_identity_round_trip",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
