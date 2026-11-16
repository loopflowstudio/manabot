"""Opponent-memory ownership and bounded execution measurements.

Default mode uses one controlled census-maximum prefix. --independent-games
instead retains real Elemental games beside 256 native buffers, then executes
one logical minibatch for all five arms. Neither mode establishes throughput,
learning or CUDA worker admission. No artificial resource pressure is induced.
"""

import argparse
from collections.abc import Callable
from copy import deepcopy
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from threading import Event, Thread
import time
import tracemalloc
from typing import Literal

import numpy as np
import psutil
import torch

import manabot.backends.torch as backend
from manabot.backends.torch.arrays import array_fields
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
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
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.profiler import Profiler
from manabot.sim.rollout import BatchedSampler, RandomBatchController, allocate_buffers
from manabot.training.regime import Learning
from manabot.training.rollout.actions import ACTION_FIELDS, PackedActions
from manabot.training.rollout.collection import (
    RecordedObservations,
    RolloutBatch,
    _ObservationRows,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.producers import ShardedVectorEnv
from manabot.training.updates.ppo import update_iteration
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


def measured[**P, T](
    operation: Callable[P, T],
    phases: dict[str, dict[str, int | None]],
    name: str,
    device: torch.device,
) -> Callable[P, T]:
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        result = operation(*args, **kwargs)
        phases[name] = residency(device)
        return result

    return run


class HostPeak:
    """Sample simultaneous process-tree RSS through every model/update lifetime."""

    def __init__(self) -> None:
        self.stop = Event()
        self.peak = 0
        self.thread = Thread(target=self._sample, name="memory-rss", daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while True:
            processes = [process, *process.children(recursive=True)]
            self.peak = max(
                self.peak, sum(p.memory_info().rss for p in processes if p.is_running())
            )
            if self.stop.wait(0.02):
                return

    def __enter__(self) -> "HostPeak":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join()


def execution_proof(
    space: ObservationSpace,
    retained: PrefixObservation,
    device: torch.device,
    *,
    collection_rows: int,
    bootstrap_rows: int,
    learner_rows: int,
    independent: bool = False,
) -> list[dict[str, object]]:
    """Exercise collection, frozen inference, bootstrap and PPO chunk paths.

    Three policies coexist with Adam and actual one-ahead host preparation.
    Default mode repeats a controlled maximum prefix and a final short chunk;
    independent mode uses the supplied cohort without repeating any prefix.
    Its caller retains the native buffers and all recorded rows throughout.
    """
    results: list[dict[str, object]] = []
    logical_rows = (
        len(retained.prefixes)
        if independent
        else max(collection_rows, bootstrap_rows, learner_rows) * 2 + 1
    )
    rows = (
        retained
        if independent
        else PrefixObservation(
            {k: np.repeat(v, logical_rows, axis=0) for k, v in retained.items()},
            retained.prefixes * logical_rows,
            retained.memory_capacities,
        )
    )
    encoders: tuple[Literal["none", "transformer", "gru"], ...] = (
        ("none", "transformer", "gru") if independent else ("transformer", "gru")
    )
    for encoder in encoders:
        for summaries in (4,) if encoder == "none" else (1, 4):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            with HostPeak() as peak:
                phases = {"before_models": residency(device)}
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
                ema, frozen = deepcopy(model), deepcopy(model)
                learner = Learner(
                    model,
                    device,
                    minibatch_rows=logical_rows,
                    execution_rows=learner_rows,
                    bootstrap_rows=bootstrap_rows,
                )
                phases["three_models"] = residency(device)
                session = CollectionSession(device=str(device), seed=111)
                with session.scope(model) as scope:
                    sampled = scope.sample(
                        rows, Profiler(), execution_rows=collection_rows
                    )
                phases["collection"] = residency(device)
                BatchedSampler(
                    frozen, device=str(device), execution_rows=collection_rows
                ).select(rows, np.arange(logical_rows))
                phases["frozen_opponent"] = residency(device)
                batch = RolloutBatch(
                    RecordedObservations(
                        {k: v for k, v in rows.items() if k not in ACTION_FIELDS},
                        np.arange(logical_rows),
                        memory=rows.prefixes,
                        memory_capacities=rows.memory_capacities,
                    ),
                    sampled.actions[None],
                    sampled.logprobs[None],
                    np.linspace(-1, 1, logical_rows)[None],
                    np.zeros((1, logical_rows), bool),
                    sampled.values[None],
                    rows,
                    np.zeros(logical_rows, bool),
                    PackedActions.pack(
                        {k: v for k, v in rows.items() if k in ACTION_FIELDS},
                        sampled.log_probabilities,
                        validator=backend.rollout,
                    ),
                )
                # Sample residency at actual preparation/forward/backward/Adam boundaries.
                model.prepare_admitted = measured(
                    model.prepare_admitted, phases, "prepared_or_prefetched", device
                )
                model.forward_distribution = measured(
                    model.forward_distribution, phases, "forward", device
                )
                diagnostics = update_iteration(
                    backend,
                    learner,
                    batch,
                    Learning(filter_scope="actor", epochs=1, minibatches=1),
                    0.0,
                    np.random.default_rng(1),
                    bootstrap_agent=ema,
                )
                del model.prepare_admitted, model.forward_distribution
                phases["after_adam"] = residency(device)
                gradients = []
                if model.opponent_memory is not None:
                    gradients = [
                        p.grad
                        for p in model.opponent_memory.parameters()
                        if p.grad is not None
                    ]
                    assert gradients and all(torch.isfinite(g).all() for g in gradients)
                    assert any(g.abs().sum() > 0 for g in gradients)
                assert all(torch.isfinite(p).all() for p in model.parameters())
                assert all(
                    state["step"].item() == 1
                    for state in learner.optimizer.state.values()
                )
            results.append(
                dict(
                    encoder=encoder,
                    summaries=summaries,
                    logical_rows=logical_rows,
                    execution_limits=dict(
                        collection_rows=collection_rows,
                        bootstrap_rows=bootstrap_rows,
                        learner_rows=learner_rows,
                    ),
                    host_prepared_batches=2,
                    simultaneous_policy_copies=3,
                    finite_forward_backward_adam=True,
                    optimizer_exposures=diagnostics["optimizer_exposures"],
                    host_tree_peak_sampled_bytes=peak.peak,
                    host_tree_sample_interval_seconds=0.02,
                    phases=phases,
                )
            )
            if independent:
                print(
                    json.dumps(
                        dict(
                            encoder=encoder,
                            summaries=summaries,
                            execution_complete=True,
                        )
                    ),
                    flush=True,
                )
            del (
                model,
                learner,
                ema,
                frozen,
                session,
                batch,
                sampled,
                gradients,
            )
    return results


def measure(
    source: Path,
    *,
    streams: int = 256,
    selected_rows: int = 256,
    device_name: Literal["cpu", "cuda"] = "cpu",
    collection_rows: int = 2,
    bootstrap_rows: int = 2,
    learner_rows: int = 2,
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
    execution = execution_proof(
        space,
        retained,
        device,
        collection_rows=collection_rows,
        bootstrap_rows=bootstrap_rows,
        learner_rows=learner_rows,
    )
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
                "manabot/backends/torch/collection.py",
                "manabot/backends/torch/learner.py",
                "manabot/backends/torch/policy.py",
                "manabot/training/execution.py",
                "manabot/training/updates/ppo.py",
                "manabot/training/updates/damped_policy.py",
                "manabot/training/regime.py",
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
        "bounded_execution_host_memory_components_bytes": native_bytes
        + max(
            collection_rows * (gather_bytes([prefix]) + prepared_bytes),
            bootstrap_rows * (gather_bytes([prefix]) + prepared_bytes),
            learner_rows * (gather_bytes([prefix]) + 2 * prepared_bytes),
        )
        + sum(
            ownership[key]
            for key in (
                "unique_chunk_bytes",
                "python_metadata_bytes",
                "intern_index_bytes",
            )
        ),
        "execution_row_counts": {
            "collection_limit": collection_rows,
            "bootstrap_limit": bootstrap_rows,
            "learner_limit": learner_rows,
            "logical_optimizer_rows": max(collection_rows, bootstrap_rows, learner_rows)
            * 2
            + 1,
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


def independent_measure(
    source: Path,
    *,
    device_name: str,
    collection_rows: int,
    bootstrap_rows: int,
    learner_rows: int,
) -> dict[str, object]:
    """Measure real independent Elemental histories beside full native buffers.

    Uniform legal play generates resource inputs, not learning evidence. Keep
    every prefix through a 64-decision floor and complete-game closure in each
    stream. Execute one 256-row logical minibatch, selecting each stream's largest
    prefix. The full retained cohort and four native producers stay live through
    all five arms. This does not admit an on-policy CUDA worker or its throughput.
    """
    streams, floor = 256, 64
    host_limit = 40 * 1024**3
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    census = json.loads(source.read_text())
    capacities = (int(census["record_capacity"]), int(census["detail_capacity"]))
    space = ObservationSpace(
        ObservationConfig(
            max_opponent_records=capacities[0],
            max_opponent_details=capacities[1],
        )
    )
    torch.set_num_threads(1)
    match = Match(
        MatchConfig.authored(
            "elemental-world-v2",
            "elemental_ug",
            "elemental_bg",
        )
    )
    buffers = allocate_buffers(space, streams)
    store = MemoryStore(capacities)
    # Use the production recorder, including legal action fields for later
    # resampling. It excludes the dense history axes exactly as collection does.
    floor_rows = streams * floor
    recorded = _ObservationRows(
        buffers, list(space.keys()), floor_rows + floor_rows // 4
    )
    native = ShardedVectorEnv(streams, 39101, 4)
    random = RandomBatchController(39101)
    counts = np.zeros(streams, np.int64)
    generations = np.zeros(streams, np.int64)
    completed = np.zeros(streams, np.int64)
    active = np.ones(streams, bool)
    largest = np.zeros(streams, np.int64)
    sizes = np.full(streams, -1, np.int64)
    phases: dict[str, dict[str, int | None]] = {}
    started = time.perf_counter()
    try:
        with HostPeak() as peak:
            native.set_buffers(buffers)
            native.set_uniform_roster(elemental_roster(match))
            native.reset_all_into_buffers(match.to_rust())
            phases["native_reset"] = residency(device)
            # Refuse a broken control contract before generating the cohort.
            control = ActorCritic(space, ModelConfig(world=managym.WORLD_SETUP))
            control.prepare(
                {
                    key: value[:1]
                    for key, value in buffers.items()
                    if key in space.keys()
                }
            )
            del control
            rounds = 0
            while active.any():
                if rounds >= 2000:
                    raise RuntimeError(
                        "independent cohort exceeded 2000 decisions; no truncation admitted"
                    )
                if residency(device)["host_tree_rss_bytes"] > host_limit:
                    raise RuntimeError("independent cohort exceeded 40 GiB host limit")
                indexes = np.flatnonzero(active)
                viewers = native.current_agent_indices()
                prefixes = tuple(
                    store.capture(
                        buffers,
                        int(row),
                        PrefixIdentity(
                            f"stream-{row}-game-{generations[row]}",
                            viewers[row],
                            "root",
                        ),
                    )
                    for row in indexes
                )
                start = recorded.append(indexes, prefixes=prefixes)
                for offset, (row, prefix) in enumerate(
                    zip(indexes, prefixes, strict=True)
                ):
                    if prefix.detail_count > sizes[row]:
                        largest[row], sizes[row] = start + offset, prefix.detail_count
                actions = np.zeros(streams, np.int64)
                actions[indexes] = random.select(buffers, indexes)
                native.step_into_buffers(actions.tolist(), active.tolist())
                counts[indexes] += 1
                done = active & (
                    (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
                )
                if (active & (buffers["truncated"] > 0)).any():
                    raise RuntimeError(
                        "independent game truncated; cohort not admitted"
                    )
                completed[done] += 1
                generations[done] += 1
                active[done & (counts >= floor)] = False
                rounds += 1
                if rounds % 32 == 0:
                    print(
                        json.dumps(
                            dict(
                                round=rounds,
                                rows=recorded.rows,
                                active=int(active.sum()),
                            )
                        ),
                        flush=True,
                    )
            phases["retained_complete_games"] = residency(device)
            arrays = recorded.arrays()
            selected = PrefixObservation(
                {key: value[largest] for key, value in arrays.items()},
                [recorded.memory[int(i)] for i in largest],
                capacities,
            )
            ownership = store.measure(recorded.memory)
            execution = execution_proof(
                space,
                selected,
                device,
                collection_rows=collection_rows,
                bootstrap_rows=bootstrap_rows,
                learner_rows=learner_rows,
                independent=True,
            )
            phases["after_execution"] = residency(device)
        return dict(
            fixture="independent Elemental uniform legal games; no synthetic prefixes",
            source_sha256={
                str(path): sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__).relative_to(Path.cwd()),
                    Path("manabot/env/memory_store.py"),
                    Path("manabot/training/rollout/collection.py"),
                    Path("manabot/training/rollout/producers.py"),
                    Path("manabot/backends/torch/policy.py"),
                    Path("manabot/policy/admission.py"),
                    Path("manabot/policy/prepare.py"),
                )
            },
            world=managym.WORLD_VERSION,
            seed=39101,
            device=device_name,
            cuda_available=torch.cuda.is_available(),
            torch_version=str(torch.__version__),
            capacities=list(capacities),
            streams=streams,
            native_producers=4,
            minimum_decisions_per_stream=floor,
            stream_decisions=counts.tolist(),
            stream_completed_games=completed.tolist(),
            retained_rows=recorded.rows,
            selected_logical_rows=streams,
            selected_detail_counts=sizes.tolist(),
            retained_prefixes=ownership,
            native_buffer_bytes=sum(v.nbytes for v in buffers.values()),
            native_memory_buffer_bytes=sum(buffers[k].nbytes for k in MEMORY_FIELDS),
            ordinary_retained_allocated_bytes=sum(
                v.nbytes for v in recorded._arrays.values()
            ),
            host_tree_peak_sampled_bytes=peak.peak,
            host_limit_bytes=host_limit,
            host_tree_sample_interval_seconds=0.02,
            sampled_host_peak_within_limit=peak.peak <= host_limit,
            elapsed_seconds=time.perf_counter() - started,
            phases=phases,
            execution_proof=execution,
            limitations=[
                "Random legal actions size host ownership; no on-policy learning or throughput claim.",
                "One 256-row logical minibatch executes while the whole cohort stays live; not a whole worker update.",
                "Sampled RSS may miss transients; allocator retention carries across arms.",
                "CUDA peaks, actual worker update and checkpoint reload remain required before Etude wiring.",
                "Learned-game tails may exceed these independent random games; full census capacity is unchanged.",
            ],
        )
    finally:
        native.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-games", action="store_true")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--selected-rows", type=int, default=256)
    parser.add_argument("--collection-rows", type=int, default=2)
    parser.add_argument("--bootstrap-rows", type=int, default=2)
    parser.add_argument("--learner-rows", type=int, default=2)
    args = parser.parse_args()
    if (
        min(
            args.streams,
            args.selected_rows,
            args.collection_rows,
            args.bootstrap_rows,
            args.learner_rows,
        )
        < 1
    ):
        parser.error("stream and selected row counts must be positive")
    if args.independent_games and (args.streams != 256 or args.selected_rows != 256):
        parser.error("independent pilot proof requires 256 streams and selected rows")
    result = (
        independent_measure(
            args.source,
            device_name=args.device,
            collection_rows=args.collection_rows,
            bootstrap_rows=args.bootstrap_rows,
            learner_rows=args.learner_rows,
        )
        if args.independent_games
        else measure(
            args.source,
            streams=args.streams,
            selected_rows=args.selected_rows,
            device_name=args.device,
            collection_rows=args.collection_rows,
            bootstrap_rows=args.bootstrap_rows,
            learner_rows=args.learner_rows,
        )
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
                if key in result
            }
        )
    )


if __name__ == "__main__":
    main()
