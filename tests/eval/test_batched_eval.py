"""Shared inference preserves admission, request RNG and native arena evidence."""

from dataclasses import replace
import json
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import threading
import time
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.eval.evaluation import EvaluationRequest
from manabot.eval.execution import bind_execution
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationContract, EvaluationKey
from manabot.infra.config import MatchConfig, ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.model.inference import transfer
from manabot.sim import child, serving
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import SELECTED_SUITE, AuthoredMatchup, selected_match
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import read_trace, replay_environment, replay_prefix
from manabot.sim.serving import (
    PolicyStore,
    _infer,
    _PendingRequest,
    _record_dispatch,
    sample_rows,
)
from manabot.sim.serving_protocol import (
    DecisionRequest,
    InferenceClient,
    InferenceFailure,
    InferenceRuntime,
    PolicyAdmission,
    PolicyKey,
)
from managym import WORLD_VERSION


def fixture_policy(
    directory: Path, seed: int = 83, *, policy_input_version: Literal[1, 2] = 2
) -> tuple[EvaluationKey, PlayerRegistration, PlayerRegistration, Path]:
    contract = EvaluationContract.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json").read_bytes()
    )
    space = ObservationSpace(
        ObservationConfig(policy_input_version=policy_input_version)
    )
    torch.manual_seed(seed)
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        ),
    )
    path = directory / f"policy-{seed}.pt"
    save_checkpoint(
        agent, space, path, player_configs=Match(selected_match()).to_rust()
    )
    runtime = runtime_fingerprints(
        match_config=selected_match(), observation_space=space, world=WORLD_VERSION
    )
    key = contract.key.model_copy(
        update={"world": WORLD_VERSION, "content_suite": SELECTED_SUITE}
    )
    anchor = contract.anchors[0].model_copy(
        update={
            "world": WORLD_VERSION,
            "content_suite": SELECTED_SUITE,
            "matchup_sha256": canonical_sha256(selected_match().model_dump()),
        }
    )
    player = PlayerRegistration.model_validate(
        {
            **anchor.model_dump(),
            "player_id": f"policy-{seed}",
            "runner_kind": "checkpoint",
            "role": "challenger",
            "source_sha256": None,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "parameter_count": sum(p.numel() for p in agent.parameters()),
            "training_seed": seed,
            "artifact_id": f"fixture-{seed}",
            "evidence_class": "fixture",
            "player_spec": {
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
            },
            "observation_abi_sha256": runtime["observation_abi_sha256"],
            "action_abi_sha256": runtime["action_abi_sha256"],
        }
    )
    return key, player, anchor, path


def request_for(
    player: PlayerRegistration,
    execution: EvalExecution,
    seed: int = 5,
    setup: MatchConfig | None = None,
) -> DecisionRequest:
    _, obs = replay_environment(
        {
            "deal_seed": 83,
            "seat_players": ["a", "b"],
            "match_hypers": (setup or selected_match()).model_dump(),
        },
        observation_space=ObservationSpace(),
    )
    return DecisionRequest(
        str(seed),
        "fixture-game",
        0,
        seed,
        PolicyKey.from_registration(player, execution),
        seed,
        seed,
        obs,
        execution.deadline_unix,
        time.monotonic(),
    )


def test_row_sampling_is_independent_of_neighbors_and_preserves_rng() -> None:
    logits = torch.tensor([[1.0, 2.0, -1e8], [3.0, 4.0, -1e8]])
    masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    before = torch.random.get_rng_state()
    batched = sample_rows(logits, masks, [5, 7], False)
    assert batched == [
        sample_rows(logits[i : i + 1], masks[i : i + 1], [seed], False)[0]
        for i, seed in enumerate([5, 7])
    ]
    assert torch.equal(before, torch.random.get_rng_state())
    assert batched[::-1] == sample_rows(logits.flip(0), masks.flip(0), [7, 5], False)
    with pytest.raises(ValueError, match="invalid legal"):
        sample_rows(logits, torch.zeros_like(masks), [5, 7], False)


@pytest.mark.parametrize("version", (1, 2))
def test_served_checkpoint_preserves_input_version(
    tmp_path: Path, version: Literal[1, 2]
) -> None:
    key, player, anchor, path = fixture_policy(tmp_path, policy_input_version=version)
    _, trace, replay = play_cell(
        key=key,
        player_a=player,
        player_b=anchor,
        deal_seeds=(831,),
        out_dir=tmp_path / "panel",
        checkpoint_paths={player.player_id: str(path)},
        execution=EvalExecution(deadline_unix=time.time() + 120, max_active_games=1),
        max_commands=2,
    )
    assert replay["passed"]
    for game in read_trace(Path(trace["path"])):
        assert len(game["decisions"]) == 2, game.get("failure")
        assert game["observation_hypers"].get("policy_input_version", 1) == version
        assert ("policy_input_version" in game["observation_hypers"]) == (version == 2)
        env, observation = replay_prefix(game, 1)
        try:
            assert ("relations" in observation) == (version == 2)
        finally:
            env.close()


def test_residency_catalog_batching_parity_and_eviction(tmp_path: Path) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 120)
    _, (player,) = bind_execution(key, (player,), execution)
    perf = EvalPerformance("test")
    store = PolicyStore(execution, perf)
    store.admit(PolicyAdmission(player, str(path), (selected_match(),)))
    requests = [request_for(player, execution, seed) for seed in (5, 7)]
    replies = _infer(store, requests, "batch")
    assert perf.counters["model_loads"] == 1
    assert perf.counters["catalog_builds"] == 1
    assert perf.counters["forward_calls"] == 1
    assert perf.counters["batch_rows/2"] == 1
    resident = store.get(requests[0].policy)
    tensor = {
        k: torch.from_numpy(np.stack([r.observation[k] for r in requests])).float()
        for k in requests[0].observation
    }
    with torch.no_grad():
        batched, _ = resident.agent(resident.scope.prepare(tensor))
        scalar = torch.cat(
            [
                resident.agent(
                    resident.scope.prepare({k: v[i : i + 1] for k, v in tensor.items()})
                )[0]
                for i in range(2)
            ]
        )
    torch.testing.assert_close(batched, scalar, rtol=1e-5, atol=1e-5)
    assert [r.action for r in replies] == sample_rows(
        scalar, tensor["actions_valid"], [5, 7], False
    )
    _, second, _, second_path = fixture_policy(tmp_path, 84)
    _, (second,) = bind_execution(key, (second,), execution)
    store.execution = execution.model_copy(
        update={"resident_model_bytes": resident.bytes}
    )
    second = PlayerRegistration.model_validate(
        {
            **second.model_dump(),
            "compute_class_id": store.execution.identity,
            "player_spec": {
                **second.player_spec,
                "execution": store.execution.compute.model_dump(),
            },
        }
    )
    store.admit(PolicyAdmission(second, str(second_path), (selected_match(),)))
    assert perf.counters["evictions"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        resident.scope.catalog()
    assert PolicyKey.from_registration(second, execution) != requests[0].policy
    store.close()


def test_matchups_of_one_checkpoint_share_its_policy_and_batches(
    tmp_path: Path,
) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    mirror = MatchConfig.authored(
        "ur-lessons-vs-gw-allies",
        "ur_lessons",
        "ur_lessons",
        hero="arena-seat-0",
        villain="arena-seat-1",
    )
    other = player.model_copy(
        update={
            "player_spec": {**player.player_spec, "allow_deck_repetition": True},
            "matchup_sha256": canonical_sha256(mirror.model_dump()),
            "content_suite": AuthoredMatchup(
                pack_key="ur-lessons-vs-gw-allies",
                role_a="ur_lessons",
                role_b="gw_allies",
            ).content_suite,
        }
    )
    execution = EvalExecution(deadline_unix=time.time() + 120)
    _, (player, other) = bind_execution(key, (player, other), execution)
    assert PolicyKey.from_registration(player, execution) == (
        PolicyKey.from_registration(other, execution)
    )
    perf = EvalPerformance("test")
    store = PolicyStore(execution, perf)
    store.admit(PolicyAdmission(player, str(path), (selected_match(),)))
    store.admit(PolicyAdmission(other, str(path), (mirror,)))
    assert perf.counters["model_loads"] == 1
    assert len(store.admissions) == 1
    requests = [
        request_for(player, execution, 5),
        request_for(other, execution, 7, mirror),
    ]
    mixed = _infer(store, requests, "mixed")
    assert perf.counters["batch_rows/2"] == 1
    single = [_infer(store, [request], "single")[0] for request in requests]
    assert [r.action for r in mixed] == [r.action for r in single]
    assert perf.counters["model_loads"] == 1
    store.close()


def _client(runtime: InferenceRuntime) -> InferenceClient:
    """A direct service connection held in this process, as a game child holds one."""
    return InferenceClient(runtime.connect().detach())


def test_service_batches_a_backlog_of_overdue_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 60, max_batch_rows=4)
    _, (player,) = bind_execution(key, (player,), execution)
    # Submitted long before the service reads them, as when it lags.
    requests = [
        replace(request_for(player, execution, seed), submitted=time.monotonic() - 1)
        for seed in range(6)
    ]
    control, service_control = Pipe()
    handoff, service_handoff = socket.socketpair()
    control.send(
        ("admission", "admit", PolicyAdmission(player, str(path), (selected_match(),)))
    )
    clients, handed_over = [], []
    for request in requests:
        client, service_end = socket.socketpair()
        # Room for the whole request before the service reads any of it.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2**20)
        service_end.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)
        socket.send_fds(handoff, [b"c"], [service_end.fileno()])
        # Kept open until the service holds it, as `InferenceRuntime.connect` does.
        handed_over.append(service_end)
        clients.append(Connection(client.detach()))
        clients[-1].send(request)
    # The pytest process keeps its own pools; the budget belongs to a child.
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(
        serving, "SERVING_BUDGET", replace(serving.SERVING_BUDGET, threads=10**6)
    )
    service = threading.Thread(
        target=serving.serve,
        args=(
            service_control,
            execution,
            tmp_path,
            "backlog",
            service_handoff.detach(),
        ),
    )
    service.start()
    try:
        assert all(client.poll(30) for client in clients)
        replies = [client.recv() for client in clients]
    finally:
        control.send(("close", "close", None))
        service.join(30)
        for service_end in handed_over:
            service_end.close()
    assert handoff.recv(len(requests)) == b"a" * len(requests)
    assert [(identity, error) for identity, _, error in replies] == [
        (r.request_id, None) for r in requests
    ]
    counters = json.loads((tmp_path / "serving.json").read_bytes())["counters"]
    assert {k: v for k, v in counters.items() if k.startswith("batch_rows/")} == {
        "batch_rows/4": 1,
        "batch_rows/2": 1,
    }
    assert counters["client_connections"] == len(requests)


def test_packed_transfer_makes_one_copy_per_dtype() -> None:
    tensors = {
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b": torch.ones(2, 4, 5),
        "rows": torch.arange(4).reshape(2, 2),
    }
    moved, copies = transfer(tensors, torch.device("meta"))
    assert copies == 2
    assert {k: (v.shape, v.dtype, v.device.type) for k, v in moved.items()} == {
        k: (v.shape, v.dtype, "meta") for k, v in tensors.items()
    }
    assert transfer(tensors, torch.device("cpu")) == (tensors, 0)


def test_service_partial_flush_stale_replies_and_death(tmp_path: Path) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 60, max_delay_seconds=0.01)
    _, (player,) = bind_execution(key, (player,), execution)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        runtime.admit(PolicyAdmission(player, str(path), (selected_match(),)))
        client = _client(runtime)
        request = request_for(player, execution)
        reply = client.infer(request)
        reply.validate(request)
        with pytest.raises(InferenceFailure, match="stale"):
            replace(reply, revision=-1).validate(request)
        receipt = runtime.snapshot()
        runtime.process.kill()
        runtime.process.join()
        with pytest.raises(InferenceFailure, match="exited"):
            client.infer(request)
        with pytest.raises(InferenceFailure):
            runtime.connect()
        client.close()
    assert receipt["counters"]["batch_rows/1"] == 1
    assert receipt["counters"]["completed_requests"] == 1
    assert "batch_queue" not in receipt["measurements"]
    assert receipt["measurements"]["dispatch_queue"]["population"] == 1
    lifecycle = json.loads((tmp_path / "allocation/allocation.json").read_bytes())
    assert lifecycle["measurements"]["service_start_to_ready"]["population"] == 1
    assert lifecycle["measurements"]["admission_round_trip"]["population"] == 1
    assert lifecycle["measurements"]["cleanup"]["population"] == 1
    # Decisions bypass the supervisor; it only hands over connections.
    assert lifecycle["counters"]["client_connections"] == 1
    assert "requests" not in lifecycle["counters"]
    retained = json.loads((tmp_path / "allocation/serving.json").read_bytes())
    assert retained["counters"]["model_loads"] == 1
    assert retained["counters"]["rss_samples"] >= 1
    assert retained["counters"]["peak_sampled_rss_bytes"] > 0
    assert not runtime.process.is_alive()


def test_queue_boundary_excludes_cold_load_and_overlapping_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 60)
    _, (player,) = bind_execution(key, (player,), execution)
    perf = EvalPerformance("controlled", started=0)
    store = PolicyStore(execution, perf)
    store.admissions[PolicyKey.from_registration(player, execution)] = PolicyAdmission(
        player, str(path), (selected_match(),)
    )
    requests = [
        replace(request_for(player, execution, seed), submitted=1) for seed in (5, 7)
    ]
    clock = [10.0]
    load = serving.load_checkpoint

    def delayed_load(filename: str) -> tuple[ActorCritic, ObservationSpace]:
        result = load(filename)
        clock[0] += 40  # Inject the duration; never create resource pressure.
        return result

    monkeypatch.setattr(serving, "load_checkpoint", delayed_load)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    client, _ = Pipe()
    _record_dispatch(perf, [_PendingRequest(r, client, 3) for r in requests], "batch")
    try:
        replies = _infer(store, requests, "batch")
        assert len(replies) == 2
        receipt = perf.snapshot()
        measures = receipt["measurements"]
        assert measures["dispatch_queue"]["maximum_seconds"] == 7
        assert measures["submission_to_receipt"]["maximum_seconds"] == 2
        assert measures["model_load"]["seconds"] == 40
        phases = {
            k: m["seconds"] for k, m in measures.items() if m["partitions_elapsed"]
        }
        assert sum(phases.values()) == 40
        assert receipt["elapsed_seconds"] == 50
        assert receipt["counters"]["resident_misses"] == 1
    finally:
        store.close()


def test_failed_service_request_has_no_success_latency(tmp_path: Path) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 60)
    _, (player,) = bind_execution(key, (player,), execution)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        runtime.admit(PolicyAdmission(player, str(path), (selected_match(),)))
        request = replace(request_for(player, execution), observation={})
        with pytest.raises(InferenceFailure, match="observation fields"):
            _client(runtime).infer(request)
    receipt = json.loads((tmp_path / "allocation/serving.json").read_bytes())
    assert receipt["counters"]["received_requests"] == 1
    assert receipt["counters"]["failed_requests"] == 1
    assert receipt["counters"]["unanswered_requests"] == 0
    assert receipt["request_p95_seconds"] is None
    assert receipt["failed_request_p95_seconds"] is not None
    assert receipt["first_request_outcome"] == "failed"
    assert not runtime.process.is_alive()


def test_service_start_failure_retains_unready_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_start(*args: object, **kwargs: object) -> None:
        raise OSError("injected spawn failure")

    monkeypatch.setattr(child.subprocess, "Popen", fail_start)
    execution = EvalExecution(deadline_unix=time.time() + 60)
    directory = tmp_path / "allocation"
    with pytest.raises(OSError, match="injected spawn failure"):
        InferenceRuntime(execution, directory)
    lifecycle = json.loads((directory / "allocation.json").read_bytes())
    assert lifecycle["counters"]["failed_service_starts"] == 1
    assert "service_start_to_ready" not in lifecycle["measurements"]
    assert lifecycle["measurements"]["failed_service_start"]["population"] == 1
    execution_receipt = json.loads((directory / "execution.json").read_bytes())
    assert execution_receipt["error"] == "OSError: injected spawn failure"
    assert execution_receipt["cost"]["measured_usd"] is None


def test_games_reach_the_service_directly_and_its_failure_is_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import signal

    key, player, anchor, path = fixture_policy(tmp_path)
    execution = EvalExecution(
        deadline_unix=time.time() + 120, max_active_games=2, request_seconds=10
    )
    _, (bound,) = bind_execution(key, (player,), execution)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        connect = runtime.connect
        connected = []

        def connect_then_stop() -> socket.socket:
            # Both games hold their connections; every decision then meets a
            # stopped service and times out in the game child.
            connected.append(connect())
            if len(connected) == execution.max_active_games:
                os.kill(runtime.process.pid, signal.SIGSTOP)
            return connected[-1]

        monkeypatch.setattr(runtime, "connect", connect_then_stop)
        rows, trace, _ = play_cell(
            key=key,
            player_a=player,
            player_b=anchor,
            deal_seeds=(831,),
            out_dir=tmp_path / "panel",
            checkpoint_paths={player.player_id: str(path)},
            execution=execution,
            inference=runtime,
            max_commands=4,
        )
        lifecycle = runtime.performance.snapshot()["counters"]
        os.kill(runtime.process.pid, signal.SIGCONT)
    assert len(rows) == 2 and all(
        row["failure"] == "inference service request timed out"
        and row["failed_player_id"] is None
        and row["score_a"] is None
        for row in rows
    )
    games = read_trace(Path(trace["path"]))
    assert all(game["infrastructure_failure"] for game in games)
    # No decision passes through the runtime.
    assert "requests" not in lifecycle


def test_concurrent_deal_blocks_retain_commands_and_one_resident_model(
    tmp_path: Path,
) -> None:
    key, player, anchor, path = fixture_policy(tmp_path)
    execution = EvalExecution(
        deadline_unix=time.time() + 180, max_active_games=4, max_delay_seconds=0.05
    )
    rows, trace, replay = play_cell(
        key=key,
        player_a=player,
        player_b=anchor,
        deal_seeds=(831, 832),
        out_dir=tmp_path / "panel",
        checkpoint_paths={player.player_id: str(path)},
        execution=execution,
        max_commands=4,
    )
    assert len(rows) == 8
    assert replay["passed"]
    games = read_trace(Path(trace["path"]))
    assert all(g["decisions"] and g["allocation_id"] for g in games)
    assert all(
        d["policy_rng_seed"]
        == (d["player_seed"] + d["player_decision_ordinal"]) % 2**63
        for g in games
        for d in g["decisions"]
    )
    assert len(list((tmp_path / "panel/failures").glob("*.jsonl"))) == 8
    receipt = json.loads((tmp_path / "panel/allocation/serving.json").read_text())
    assert receipt["counters"]["model_loads"] == 1
    assert receipt["counters"]["catalog_builds"] == 1
    assert all(
        g["worker_performance"]["latency_sample_count"]
        == sum(d["player_id"] == player.player_id for d in g["decisions"])
        for g in games
    )
    assert all(row["score_a"] is None for row in rows)
    # The group worker's own census, taken while its game threads are alive.
    cell = json.loads((tmp_path / "panel/performance.json").read_text())["counters"]
    assert cell["threads/group_worker"] >= 1
    assert cell["threads_budget/group_worker"] == execution.max_active_games + 7

    from experiments.report_prototype.build import write_evaluations
    from manabot.eval.serving_benchmark import BenchmarkPlan, Panel, numerical_check

    report = write_evaluations([tmp_path / "panel"], tmp_path / "report")
    assert report.exists()
    plan = BenchmarkPlan(
        panels=(
            Panel(
                name="fixture",
                request=EvaluationRequest(
                    key=key,
                    candidate=player,
                    opponent=anchor,
                    checkpoint_paths={player.player_id: str(path)},
                    deal_seeds=(831, 832),
                    game_seconds=120,
                    max_commands=4,
                    purpose="serving-calibration",
                ),
            ),
        ),
        configurations={
            "cold_scalar": execution.model_copy(
                update={
                    "max_active_games": 1,
                    "max_batch_rows": 1,
                    "resident": False,
                    "cache_catalog": False,
                }
            ),
            "resident_only": execution.model_copy(
                update={"max_batch_rows": 1, "cache_catalog": False}
            ),
            "resident_catalog": execution.model_copy(update={"max_batch_rows": 1}),
            "batched": execution,
        },
        repetitions=3,
        measured_roots=16,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
        minimum_action_agreement=1,
        minimum_speedup=1,
        maximum_instrumentation_overhead=1,
        process_tree_bytes=4 * 2**30,
        deadline_unix=execution.deadline_unix,
    )
    numerical = numerical_check(plan, plan.panels[0], Path(trace["path"]))
    assert numerical["passed"], numerical
    assert len(numerical["results"]) == 4
    assert all(r["instrumentation_actions_match"] for r in numerical["results"])


def test_execution_binding_preserves_seed_authority_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    from manabot.eval.match import derive_seed

    key, player, anchor, _ = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 30)
    bound, players = bind_execution(key, (player, anchor), execution)
    assert bound != key
    assert derive_seed(
        bound, (player.player_id, anchor.player_id), 7, player.player_id
    ) == derive_seed(key, (player.player_id, anchor.player_id), 7, player.player_id)
    assert (
        PlayerRegistration.model_validate_json(players[0].model_dump_json())
        == players[0]
    )
    later = execution.model_copy(
        update={
            "deadline_unix": execution.deadline_unix + 3600,
            "instrumentation": False,
            "diagnostic_batches": 2,
        }
    )
    assert later != execution
    assert later.identity == execution.identity
    assert bind_execution(bound, players, later) == (bound, players)
    with pytest.raises(ValueError, match="conflicting"):
        bind_execution(
            bound, players, execution.model_copy(update={"max_batch_rows": 1})
        )


def test_private_callback_endpoint_runs_complete_replayed_block(tmp_path: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    from manabot.eval.endpoint import CallbackEndpoint

    contract = EvaluationContract.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json").read_bytes()
    )
    execution = EvalExecution(deadline_unix=time.time() + 120, max_active_games=2)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        with CallbackEndpoint(runtime) as endpoint:
            with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
                rows, _, replay = pool.submit(
                    endpoint.play_cell,
                    key=contract.key,
                    player_a=contract.anchors[0],
                    player_b=contract.anchors[1],
                    deal_seeds=(777,),
                    out_dir=tmp_path / "cell",
                ).result(timeout=100)
    assert len(rows) == 2
    assert all(row["terminated"] and row["failure"] is None for row in rows)
    assert replay["passed"]


def test_expired_allocation_starts_no_games_and_search_stays_unbound(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="deadline"):
        InferenceRuntime(
            EvalExecution(deadline_unix=time.time() - 1), tmp_path / "expired"
        )
    assert not (tmp_path / "expired").exists()
    contract = EvaluationContract.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/int-6-skill-arena-v1.json").read_bytes()
    )
    # Search players keep their registration and play in their game child.
    _, (search,) = bind_execution(
        contract.key,
        (contract.anchors[2],),
        EvalExecution(deadline_unix=time.time() + 30),
    )
    assert search == contract.anchors[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA runner")
def test_cuda_retained_input_parity(tmp_path: Path) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    logits = []
    for device in ("cpu", "cuda"):
        execution = EvalExecution(device=device, deadline_unix=time.time() + 120)
        _, (bound,) = bind_execution(key, (player,), execution)
        store = PolicyStore(execution, EvalPerformance(device))
        store.admit(PolicyAdmission(bound, str(path), (selected_match(),)))
        request = request_for(bound, execution)
        policy = store.get(request.policy)
        with torch.no_grad():
            inputs = {
                k: torch.from_numpy(v).float().unsqueeze(0)
                for k, v in request.observation.items()
            }
            logits.append(policy.agent(policy.scope.prepare(inputs))[0].cpu())
        store.close()
    torch.testing.assert_close(logits[0], logits[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("deadline_seconds", [0.1, 60.0])
def test_two_policies_never_share_a_forward_and_hung_service_is_bounded(
    tmp_path: Path,
    deadline_seconds: float,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import os
    import signal

    key, first, _, first_path = fixture_policy(tmp_path)
    _, second, _, second_path = fixture_policy(tmp_path, 84)
    execution = EvalExecution(
        deadline_unix=time.time() + 90,
        request_seconds=10,
        max_batch_rows=2,
        max_delay_seconds=0.1,
    )
    _, (first, second) = bind_execution(key, (first, second), execution)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        for player, path in ((first, first_path), (second, second_path)):
            runtime.admit(PolicyAdmission(player, str(path), (selected_match(),)))
        requests = [
            request_for(player, execution, seed)
            for player, seed in ((first, 1), (second, 2), (first, 3), (second, 4))
        ]
        # One connection per game, each with one request in flight.
        clients = [_client(runtime) for _ in requests]
        with ThreadPoolExecutor(4) as pool:
            replies = list(
                pool.map(lambda pair: pair[0].infer(pair[1]), zip(clients, requests))
            )
        batches: dict[str, set[PolicyKey]] = {}
        for request, reply in zip(requests, replies, strict=True):
            batches.setdefault(reply.batch_id, set()).add(request.policy)
        # A loaded host may flush partial batches before every caller arrives;
        # compatibility is mandatory, coalescing these particular rows is not.
        assert all(len(policies) == 1 for policies in batches.values())
        counters = runtime.snapshot()["counters"]
        assert counters["model_loads"] == 2
        assert counters["forward_calls"] == len(batches)
        os.kill(runtime.process.pid, signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(InferenceFailure, match="timed out"):
            # Exceed socket capacity while the reader is stopped. The send
            # itself is bounded, not only the wait for a reply.
            clients[0].infer(
                replace(
                    requests[0],
                    # A game bounds each request by the allocation's request
                    # allowance, as `execute_game` builds it.
                    deadline_unix=time.time()
                    + min(deadline_seconds, execution.request_seconds),
                    observation={"blocked_pipe": np.zeros(2**22, dtype=np.float32)},
                )
            )
        assert (
            time.monotonic() - started
            < min(deadline_seconds, execution.request_seconds) + 5
        )
    assert not runtime.process.is_alive()
    assert not runtime._sender.is_alive()
    assert not runtime._reader.is_alive()


def test_benchmark_missing_plan_fields_fail_before_execution(tmp_path: Path) -> None:
    from manabot.eval.serving_benchmark import run

    plan = tmp_path / "plan.json"
    plan.write_text('{"repetitions": 3}')
    with pytest.raises(ValueError):
        run(plan, tmp_path / "result")
    assert not (tmp_path / "result").exists()
