"""Shared inference preserves admission, request RNG and native arena evidence."""

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env import Match, ObservationSpace
from manabot.eval.evaluation import EvaluationRequest
from manabot.eval.execution import bind_execution
from manabot.eval.match import play_cell
from manabot.eval.models import EvaluationContract, EvaluationKey
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import runtime_fingerprints
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.execution import EvalExecution
from manabot.sim.matchup import SELECTED_SUITE, selected_match
from manabot.sim.performance import EvalPerformance
from manabot.sim.registration import PlayerRegistration
from manabot.sim.replay import read_trace, replay_environment, replay_prefix
from manabot.sim.serving import (
    DecisionRequest,
    InferenceFailure,
    InferenceRuntime,
    PolicyAdmission,
    PolicyKey,
    PolicyStore,
    _infer,
    sample_rows,
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
    player: PlayerRegistration, execution: EvalExecution, seed: int = 5
) -> DecisionRequest:
    _, obs = replay_environment(
        {
            "deal_seed": 83,
            "seat_players": ["a", "b"],
            "match_hypers": selected_match().model_dump(),
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


def test_service_partial_flush_stale_replies_and_death(tmp_path: Path) -> None:
    key, player, _, path = fixture_policy(tmp_path)
    execution = EvalExecution(deadline_unix=time.time() + 60, max_delay_seconds=0.01)
    _, (player,) = bind_execution(key, (player,), execution)
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        runtime.admit(PolicyAdmission(player, str(path), (selected_match(),)))
        request = request_for(player, execution)
        reply = runtime.infer(request)
        reply.validate(request)
        with pytest.raises(InferenceFailure, match="stale"):
            replace(reply, revision=-1).validate(request)
        receipt = runtime.snapshot()
        runtime.process.kill()
        runtime.process.join()
        with pytest.raises(InferenceFailure):
            runtime.infer(request)
    assert receipt["counters"]["batch_rows/1"] == 1


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
    assert any(
        int(k.split("/")[1]) > 1
        for k in receipt["counters"]
        if k.startswith("batch_rows/")
    )
    assert all(row["score_a"] is None for row in rows)

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


def test_expired_allocation_and_unsupported_players_start_no_games(
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
    with pytest.raises(ValueError, match="stateless"):
        bind_execution(
            contract.key,
            (contract.anchors[2],),
            EvalExecution(deadline_unix=time.time() + 30),
        )


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
        with ThreadPoolExecutor(4) as pool:
            replies = list(pool.map(runtime.infer, requests))
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
            # Exceed pipe capacity while the reader is stopped. A synchronous
            # send under the futures lock would prevent the deadline firing.
            runtime.infer(
                replace(
                    requests[0],
                    deadline_unix=time.time() + deadline_seconds,
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
