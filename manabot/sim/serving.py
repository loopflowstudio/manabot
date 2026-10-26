"""One resident inference process per evaluation allocation.

Game processes exchange owned arrays through their existing supervisor pipe.
Only this process loads served weights. The parent multiplexes bounded requests
and grants game slots across callbacks. FIFO policy groups flush independently;
each row samples on CPU from its own retained decision seed.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import queue
import threading
import time
from types import TracebackType
from typing import Any
from uuid import uuid4

import numpy as np
import psutil
import torch

from manabot.env import Match, ObservationSpace
from manabot.infra.config import MatchConfig, ObservationConfig
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.inference import PolicyInference
from manabot.model.world import validate_agent_setup
from manabot.sim.execution import EvalExecution
from manabot.sim.performance import EvalPerformance, allocation_cost
from manabot.sim.registration import PlayerRegistration


class InferenceFailure(RuntimeError):
    """Infrastructure failure, never an attributed player forfeit."""


@dataclass(frozen=True)
class PolicyKey:
    checkpoint: str
    input_binding: str
    device: str
    dtype: str
    deterministic: bool
    implementation: str = "policy-inference-cpu-row-sampling-v1"

    @classmethod
    def from_registration(
        cls, player: PlayerRegistration, execution: EvalExecution
    ) -> PolicyKey:
        return cls(
            str(player.checkpoint_sha256),
            canonical_sha256(
                {
                    "observation": player.observation_abi_sha256,
                    "action": player.action_abi_sha256,
                    "world": player.world,
                    "content": player.content_suite,
                    "matchup": player.matchup_sha256,
                }
            ),
            execution.device,
            execution.dtype,
            bool(player.player_spec["deterministic"]),
        )


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    game_id: str
    actor: int
    revision: int
    policy: PolicyKey
    ordinal: int
    seed: int
    observation: dict[str, np.ndarray]
    deadline_unix: float
    submitted: float


@dataclass(frozen=True)
class DecisionReply:
    request_id: str
    game_id: str
    actor: int
    revision: int
    ordinal: int
    action: int
    batch_id: str

    def validate(self, request: DecisionRequest) -> None:
        if (self.request_id, self.game_id, self.actor, self.revision, self.ordinal) != (
            request.request_id,
            request.game_id,
            request.actor,
            request.revision,
            request.ordinal,
        ):
            raise InferenceFailure("stale or mismatched inference reply")


@dataclass(frozen=True)
class PolicyAdmission:
    registration: PlayerRegistration
    path: str
    setups: tuple[MatchConfig, ...]


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
        key = PolicyKey.from_registration(player, self.execution)
        self.admissions[key] = admission
        resident = self.get(key)
        if (
            sum(p.numel() for p in resident.agent.parameters())
            != player.parameter_count
        ):
            raise ValueError("checkpoint parameter count drift")
        if not admission.setups:
            raise ValueError("policy admission requires an execution setup")
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
            prepared = (
                policy.scope.prepare(tensors)
                if store.execution.cache_catalog
                else {k: t.to(store.execution.device) for k, t in tensors.items()}
            )
        if store.execution.device == "cuda":
            perf.count("host_to_device_bytes", size)
            perf.count("host_to_device_calls", len(tensors))
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


def _service(
    connection: Connection, execution: EvalExecution, directory: Path, allocation: str
) -> None:
    torch.set_num_threads(1)
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


class InferenceRuntime:
    """Allocation supervisor, shared by concurrent local Etude callbacks."""

    def __init__(self, execution: EvalExecution, directory: Path) -> None:
        if execution.remaining() <= 0:
            raise ValueError("allocation deadline has expired")
        self.execution = execution
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.identity = uuid4().hex
        self.performance = EvalPerformance(self.identity, execution.instrumentation)
        self.slots = threading.BoundedSemaphore(execution.max_active_games)
        self._game_capacity = threading.BoundedSemaphore(execution.max_active_games * 2)
        self._games = ThreadPoolExecutor(max_workers=execution.max_active_games)
        self._capacity = threading.BoundedSemaphore(execution.max_active_games + 1)
        self._lock = threading.RLock()
        self._futures: dict[str, Future[Any]] = {}
        self._error: str | None = None
        self._closed = False
        self._closing = False
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self.process = context.Process(
            target=_service, args=(child, execution, directory, self.identity)
        )
        self.process.start()
        child.close()
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()
        self._outgoing: queue.Queue[tuple[str, str, object] | None] = queue.Queue(
            execution.max_active_games + 2
        )
        # Pipe.send can block while a service is stopped. Keep it off the
        # futures lock and caller thread so request deadlines and fail() work.
        self._sender = threading.Thread(target=self._send, daemon=True)
        self._sender.start()

    def check(self) -> None:
        if (
            self._error
            or self._closed
            or self._closing
            or self.execution.remaining() <= 0
            or not self.process.is_alive()
        ):
            raise InferenceFailure(self._error or "inference allocation stopped")

    def submit_game(self, run: Callable[[], dict[str, Any]]) -> Future[dict[str, Any]]:
        """Bound both threads and queued games across all callbacks."""
        while not self._game_capacity.acquire(
            timeout=min(0.1, self.execution.remaining())
        ):
            self.check()
        try:
            self.check()
            future = self._games.submit(run)
        except BaseException:
            self._game_capacity.release()
            raise
        future.add_done_callback(lambda _: self._game_capacity.release())
        return future

    def _send(self) -> None:
        try:
            while (message := self._outgoing.get()) is not None:
                self._connection.send(message)
        except (OSError, EOFError) as exc:
            self.fail(f"inference send failed: {exc}")

    def _receive(self) -> None:
        try:
            while True:
                identity, result, error = self._connection.recv()
                with self._lock:
                    future = self._futures.pop(identity, None)
                if future is not None:
                    if error:
                        future.set_exception(InferenceFailure(error))
                    else:
                        future.set_result(result)
        except (EOFError, OSError) as exc:
            if not self._closing:
                self.fail(f"inference service exited: {exc}")

    def fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            for future in self._futures.values():
                if not future.done():
                    future.set_exception(InferenceFailure(message))
            self._futures.clear()

    def _call(self, kind: str, payload: object, timeout: float | None = None) -> Any:
        timeout = min(
            self.execution.request_seconds,
            self.execution.request_seconds if timeout is None else timeout,
            self.execution.remaining(),
        )
        if timeout <= 0:
            raise InferenceFailure("inference request deadline exhausted")
        deadline = time.monotonic() + timeout
        if not self._capacity.acquire(timeout=timeout):
            raise InferenceFailure("inference backpressure deadline exhausted")
        try:
            self.check()
            identity = uuid4().hex
            future: Future[Any] = Future()
            with self._lock:
                self._futures[identity] = future
                self._outgoing.put_nowait((identity, kind, payload))
            try:
                return future.result(timeout=max(0, deadline - time.monotonic()))
            except TimeoutError as exc:
                self.fail("inference service request timed out")
                raise InferenceFailure("inference service request timed out") from exc
        finally:
            self._capacity.release()

    def admit(self, admission: PolicyAdmission) -> ObservationConfig:
        result = self._call("admit", admission)
        if not isinstance(result, ObservationConfig):
            raise InferenceFailure("invalid observation admission reply")
        return result

    def infer(self, request: DecisionRequest) -> DecisionReply:
        try:
            result = self._call("infer", request, request.deadline_unix - time.time())
            if not isinstance(result, DecisionReply):
                raise InferenceFailure("invalid inference reply")
            result.validate(request)
            if time.time() >= request.deadline_unix:
                raise InferenceFailure("inference reply arrived after its deadline")
            return result
        except InferenceFailure as exc:
            self.fail(str(exc))
            raise

    def snapshot(self) -> dict[str, Any]:
        result = self._call("snapshot", None)
        if not isinstance(result, dict):
            raise InferenceFailure("invalid serving receipt")
        return result

    def __enter__(self) -> InferenceRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        if self._futures:
            self.fail("allocation closed with pending requests")
        self._games.shutdown(wait=True, cancel_futures=True)
        try:
            self._outgoing.put_nowait(("close", "close", None))
        except queue.Full:
            self.fail("inference shutdown queue saturated")
        self.process.join(timeout=min(5, self.execution.remaining()))
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self._reader.join(timeout=1)
        try:
            self._outgoing.put_nowait(None)
        except queue.Full:
            pass
        self._sender.join(timeout=1)
        self._closed = True
        self._connection.close()
        self.performance.write(self.directory / "allocation.json")

        atomic_json(
            self.directory / "execution.json",
            {
                "allocation_id": self.identity,
                "execution": self.execution.model_dump(),
                "cost": allocation_cost(time.monotonic() - self.performance.started),
                "error": self._error,
            },
        )
