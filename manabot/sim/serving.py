"""The serving child: one resident inference process per evaluation allocation.

`InferenceRuntime` (`manabot.sim.serving_protocol`) launches `serve` as a
`ChildProcess`; only this process imports Torch and loads served weights. Game
children send owned arrays through their supervisor, which multiplexes them
here. Requests group by `PolicyKey` (checkpoint and numerics, not matchup);
FIFO groups flush independently at `max_batch_rows` or `max_delay_seconds`.
Each batch moves to the device in one copy per dtype, and each row samples on
CPU from its own retained decision seed, so actions do not depend on batch
composition.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
import psutil
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.inference import PolicyInference, transfer
from manabot.model.world import validate_agent_setup
from manabot.sim.execution import EvalExecution
from manabot.sim.performance import EvalPerformance
from manabot.sim.serving_protocol import (
    DecisionReply,
    DecisionRequest,
    InferenceFailure,
    PolicyAdmission,
    PolicyKey,
)
from manabot.sim.threads import SERVING_BUDGET


@dataclass
class ResidentPolicy:
    agent: ActorCritic
    space: ObservationSpace
    scope: PolicyInference
    bytes: int

    def close(self) -> None:
        self.scope.__exit__(None, None, None)


class PolicyStore:
    """Serial service owner: eviction occurs only between completed batches."""

    def __init__(self, execution: EvalExecution, perf: EvalPerformance) -> None:
        self.execution = execution
        self.perf = perf
        self.admissions: dict[PolicyKey, PolicyAdmission] = {}
        self.policies: OrderedDict[PolicyKey, ResidentPolicy] = OrderedDict()

    def admit(self, admission: PolicyAdmission) -> ObservationConfig:
        player = admission.registration
        path = Path(admission.path)
        if (
            path.stat().st_size != player.checkpoint_bytes
            or file_sha256(path) != player.checkpoint_sha256
        ):
            raise ValueError("checkpoint byte identity drift")
        if player.player_spec.get("execution") != self.execution.compute.model_dump():
            raise ValueError("registration differs from serving execution")
        if not admission.setups:
            raise ValueError("policy admission requires an execution setup")
        key = PolicyKey.from_registration(player, self.execution)
        # Registrations of one checkpoint for other matchups share its resident
        # policy. Each registration's own setups are validated against it.
        admitted = self.admissions.setdefault(key, admission)
        resident = self.get(key)
        if (
            sum(p.numel() for p in resident.agent.parameters())
            != player.parameter_count
        ):
            raise ValueError("checkpoint parameter count drift")
        runtime = runtime_fingerprints(
            match_config=admission.setups[0], observation_space=resident.space
        )
        if (
            runtime["world"] != player.world
            or runtime["action_abi_sha256"] != player.action_abi_sha256
        ):
            raise ValueError("checkpoint action/world ABI drift")
        for setup in admission.setups:
            validate_agent_setup(
                resident.agent,
                Match(setup).to_rust(),
                allow_deck_repetition=player.player_spec.get(
                    "allow_deck_repetition", False
                ),
            )
        if admitted is not admission:
            self.admissions[key] = replace(
                admitted, setups=admitted.setups + admission.setups
            )
        return resident.space.encoder.config

    def get(self, key: PolicyKey) -> ResidentPolicy:
        if key in self.policies:
            self.perf.count("resident_hits")
            self.policies.move_to_end(key)
            return self.policies[key]
        admission = self.admissions[key]
        player, path = admission.registration, Path(admission.path)
        if (
            path.stat().st_size != player.checkpoint_bytes
            or file_sha256(path) != key.checkpoint
        ):
            raise ValueError("checkpoint byte identity drift")
        with self.perf.span("model_load"):
            agent, space = load_checkpoint(str(path))
            agent.eval()
            if agent.config.compound_decisions or agent.belief_count_buckets:
                raise ValueError(
                    "shared inference rejects compound or belief checkpoints"
                )
            if sum(p.numel() for p in agent.parameters()) != player.parameter_count:
                raise ValueError("checkpoint parameter count drift")
            schema = {
                name: {"shape": list(shape), "dtype": "float32"}
                for name, shape in sorted(space.shapes.items())
            }
            if canonical_sha256(schema) != player.observation_abi_sha256:
                raise ValueError("checkpoint observation ABI drift")
            # Reserve the cached semantic catalog too; temporary activations are
            # separately measured, not claimed to fit this resident-weights cap.
            size = sum(
                t.numel() * t.element_size()
                for t in (*agent.parameters(), *agent.buffers())
            )
            if agent.semantic_cards is not None:
                size += (
                    agent.semantic_cards.definition_rows.numel()
                    * agent.config.hidden_dim
                    * 4
                )
            if size > self.execution.resident_model_bytes:
                raise ValueError("policy exceeds resident-model byte limit")
            while (
                self.policies
                and sum(p.bytes for p in self.policies.values()) + size
                > self.execution.resident_model_bytes
            ):
                _, old = self.policies.popitem(last=False)
                old.close()
                self.perf.count("evictions")
            agent.to(self.execution.device)
            resident = ResidentPolicy(agent, space, PolicyInference(agent), size)
            self.policies[key] = resident
            self.perf.count("model_loads")
            self.perf.counters["resident_bytes"] = sum(
                p.bytes for p in self.policies.values()
            )
        if self.execution.cache_catalog and agent.semantic_cards is not None:
            with self.perf.span("catalog_preparation"):
                resident.scope.catalog()
            self.perf.count("catalog_builds")
        return resident

    def close(self) -> None:
        for policy in self.policies.values():
            policy.close()
        self.policies.clear()


def sample_rows(
    logits: torch.Tensor, masks: torch.Tensor, seeds: list[int], deterministic: bool
) -> list[int]:
    """Preserve scalar Categorical sampling, independent of arrival/batch order."""
    if logits.device.type != "cpu" or masks.shape != logits.shape:
        raise ValueError("sampling requires matching CPU logits and masks")
    valid = masks != 0
    if not valid.any(1).all() or not torch.isfinite(logits[valid]).all():
        raise ValueError("invalid legal logits or empty action support")
    actions = []
    for row, seed in enumerate(seeds):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            values = logits[row : row + 1]
            action = (
                values.argmax(-1)
                if deterministic
                else torch.distributions.Categorical(logits=values).sample()
            )
            selected = int(action.item())
        if not valid[row, selected]:
            raise ValueError("policy selected a masked offer")
        actions.append(selected)
    return actions


def _infer(
    store: PolicyStore, requests: list[DecisionRequest], batch: str
) -> list[DecisionReply]:
    perf = store.perf
    policy = store.get(requests[0].policy)
    for request in requests:
        if request.deadline_unix <= time.time():
            raise InferenceFailure("request deadline exhausted")
        perf.record(
            "batch_queue", request.submitted, request=request.request_id, batch=batch
        )
    with perf.span("input_assembly", batch=batch):
        tensors = {
            key: torch.from_numpy(
                np.stack([r.observation[key] for r in requests])
            ).float()
            for key in requests[0].observation
        }
        if any(set(r.observation) != set(policy.space.shapes) for r in requests):
            raise ValueError("request observation fields differ from admitted ABI")
        if any(
            tuple(t.shape[1:]) != tuple(policy.space.shapes[k])
            for k, t in tensors.items()
        ):
            raise ValueError("request observation shape differs from admitted ABI")
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        perf.count("input_bytes", size)
    with torch.no_grad():
        with perf.span("input_admission_and_transfer_host", batch=batch):
            if store.execution.cache_catalog:
                prepared = policy.scope.prepare(tensors)
                transfers = prepared.transfers
            else:
                prepared, transfers = transfer(
                    tensors, torch.device(store.execution.device)
                )
        if store.execution.device == "cuda":
            perf.count("host_to_device_bytes", size)
            perf.count("host_to_device_calls", transfers)
        with perf.span("forward_host", batch=batch):
            logits, _ = policy.agent(prepared)
        with perf.span("output_transfer_host", batch=batch):
            logits = logits.cpu()
        perf.count("output_bytes", logits.numel() * logits.element_size())
        if store.execution.device == "cuda":
            perf.count("device_to_host_calls")
            perf.count("required_output_waits")
        with perf.span("sampling", batch=batch):
            actions = sample_rows(
                logits,
                tensors["actions_valid"],
                [r.seed for r in requests],
                requests[0].policy.deterministic,
            )
    perf.count("forward_calls")
    perf.count(f"batch_rows/{len(requests)}")
    perf.count("decisions", len(requests))
    if not store.execution.cache_catalog and policy.agent.semantic_cards is not None:
        perf.count("catalog_builds")
    replies = [
        DecisionReply(
            r.request_id, r.game_id, r.actor, r.revision, r.ordinal, action, batch
        )
        for r, action in zip(requests, actions, strict=True)
    ]
    for request in requests:
        perf.latency(time.monotonic() - request.submitted)
    if not store.execution.resident:
        store.close()
    return replies


def serve(
    connection: Connection, execution: EvalExecution, directory: Path, allocation: str
) -> None:
    """Serving child entry; exceeding its thread budget fails the allocation."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    perf = EvalPerformance(allocation, execution.instrumentation)
    store = PolicyStore(execution, perf)
    pending: deque[tuple[str, DecisionRequest]] = deque()
    diagnostics = 0
    published = time.monotonic()
    try:
        while execution.remaining() > 0:
            # Once a group is due, continuous admissions cannot starve its flush.
            due = bool(pending) and (
                time.monotonic() - pending[0][1].submitted
                >= execution.max_delay_seconds
                or sum(r.policy == pending[0][1].policy for _, r in pending)
                >= execution.max_batch_rows
            )
            if not due and connection.poll(
                0 if pending else min(0.1, execution.remaining())
            ):
                identity, kind, payload = connection.recv()
                if kind == "close":
                    break
                if kind == "infer":
                    pending.append((identity, payload))
                else:
                    try:
                        result = (
                            store.admit(payload) if kind == "admit" else perf.snapshot()
                        )
                        connection.send((identity, result, None))
                    except Exception as exc:
                        perf.count("failed_admissions")
                        connection.send(
                            (identity, None, f"{type(exc).__name__}: {exc}")
                        )
                continue
            if not pending:
                continue
            first = pending[0][1]
            compatible = [item for item in pending if item[1].policy == first.policy]
            if (
                len(compatible) < execution.max_batch_rows
                and time.monotonic() - first.submitted < execution.max_delay_seconds
            ):
                connection.poll(min(execution.max_delay_seconds, 0.001))
                continue
            chosen = compatible[: execution.max_batch_rows]
            ids = {item[0] for item in chosen}
            pending = deque(item for item in pending if item[0] not in ids)
            batch = uuid4().hex
            try:
                if diagnostics < execution.diagnostic_batches:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if execution.device == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(
                        activities=activities, record_shapes=True
                    ) as profile:
                        if execution.device == "cuda":
                            begin, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            begin.record()
                        replies = _infer(store, [r for _, r in chosen], batch)
                        if execution.device == "cuda":
                            end.record()
                            end.synchronize()
                            perf.count("diagnostic_synchronizations")
                            perf.device_events.append(
                                {
                                    "batch": batch,
                                    "seconds": begin.elapsed_time(end) / 1000,
                                    "clock": "CUDA event duration; no host epoch alignment",
                                    "resource": "device stream elapsed",
                                    "scope": "diagnostic batch, includes stream gaps; not pure kernel compute",
                                }
                            )
                    trace = directory / f"diagnostic-{diagnostics}.json"
                    profile.export_chrome_trace(str(trace))
                    perf.traces.append(str(trace))
                    perf.count("diagnostic_batches")
                    diagnostics += 1
                else:
                    replies = _infer(store, [r for _, r in chosen], batch)
                # Pools start lazily (CUDA, profiler); count after each batch
                # and fail its requests, never attribute them to a player.
                SERVING_BUDGET.account(perf.counters)
                for (identity, _), reply in zip(chosen, replies, strict=True):
                    connection.send((identity, reply, None))
            except Exception as exc:
                perf.count("failed_requests", len(chosen))
                for identity, _ in chosen:
                    connection.send((identity, None, f"{type(exc).__name__}: {exc}"))
            perf.counters["rss_bytes"] = psutil.Process().memory_info().rss
            if execution.device == "cuda":
                perf.counters["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            if time.monotonic() - published >= 1:
                perf.write(directory / "serving.json")
                published = time.monotonic()
    finally:
        store.close()
        perf.write(directory / "serving.json")
        connection.close()
