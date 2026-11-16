"""MPS serving admission, frozen compute identities and bounded replay proof."""

import json
from pathlib import Path
import time
from typing import Literal

import pytest
import torch

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.eval.admission import register_anchor, register_checkpoint
from manabot.eval.evaluation import EvaluationRequest, evaluate
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint
from manabot.sim.execution import EvalCompute, EvalComputeV1, EvalExecution
from manabot.sim.matchup import DeckMatchup
from manabot.sim.performance import EvalPerformance
from manabot.sim.replay import read_trace
from manabot.sim.serving import PolicyStore, _sample_memory
from manabot.sim.serving_protocol import InferenceRuntime, PolicyKey
import managym
from tests.eval.arena_fixture import arena_fixture


@pytest.mark.parametrize(
    ("device", "v2_hash", "v1_hash"),
    [
        (
            "cpu",
            "e541f51084923189acbe3b94057a34bdaa8a9f53ca8c81838b29810904e83bba",
            "52748afa9c9b375ac633fae58247670b7ad96b781938d756de86af5a04db7ffd",
        ),
        (
            "cuda",
            "e4cf71348a370839681795773be1a166adcb892869aa38c07706f88e8c6daae2",
            "3da6111477294136fa3e8b514d36d25c29146c6bd67eacb2a79bf35972b8243c",
        ),
    ],
)
def test_saved_compute_bytes_and_identities_unchanged(
    device: Literal["cpu", "cuda"], v2_hash: str, v1_hash: str
) -> None:
    # Captured from the parent implementation before extending its device enum.
    payload = f'{{"version":"shared-eval-v2","device":"{device}","dtype":"float32"}}'
    compute = EvalCompute(device=device)
    assert compute.model_dump_json() == payload
    assert compute.identity == "shared-eval-" + v2_hash
    execution = EvalExecution(device=device, deadline_unix=1)
    assert execution.compute.model_dump_json() == payload
    assert execution.identity == compute.identity
    frozen = (
        f'{{"version":"shared-eval-v1","device":"{device}","dtype":"float32",'
        '"max_active_games":64,"max_batch_rows":64,"max_delay_seconds":0.005,'
        '"resident_model_bytes":1073741824,"request_seconds":60.0,'
        '"cache_catalog":true,"resident":true}'
    )
    v1 = EvalComputeV1.model_validate_json(frozen)
    assert v1.model_dump_json() == frozen
    assert v1.identity == "shared-eval-" + v1_hash
    with pytest.raises(ValueError):
        EvalComputeV1.model_validate({**v1.model_dump(), "device": "mps"})


def test_mps_compute_is_explicit_float32() -> None:
    compute = EvalCompute(device="mps")
    assert EvalCompute.model_validate_json(compute.model_dump_json()) == compute
    assert compute.dtype == "float32"
    assert len({EvalCompute(device=d).identity for d in ("cpu", "cuda", "mps")}) == 3
    for dtype in ("float16", "bfloat16", "tf32"):
        with pytest.raises(ValueError):
            EvalCompute.model_validate({"device": "mps", "dtype": dtype})


@pytest.mark.parametrize("available", [False, True])
def test_mps_serving_rejects_unavailable_or_fallback(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    store = PolicyStore(
        EvalExecution(device="mps", deadline_unix=time.time() + 30),
        EvalPerformance("unavailable"),
    )
    message = (
        "forbids PYTORCH_ENABLE_MPS_FALLBACK"
        if available
        else "MPS requested but unavailable"
    )
    with pytest.raises(ValueError, match=message):
        store.get(PolicyKey("unused", "unused", "mps", "float32", False))
    assert not store.policies


def test_mps_memory_samples_are_not_allocator_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf = EvalPerformance("memory")
    execution = EvalExecution(device="mps", deadline_unix=1)
    for allocated, driver in ((200, 500), (100, 400)):
        monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: allocated)
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: driver)
        _sample_memory(perf, execution)
    assert perf.counters["allocated_mps_bytes"] == 100
    assert perf.counters["driver_mps_bytes"] == 400
    assert perf.counters["peak_sampled_allocated_mps_bytes"] == 200
    assert perf.counters["peak_sampled_driver_mps_bytes"] == 500
    assert not any("cuda" in name for name in perf.counters)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_tiny_mps_service_completes_valid_games_and_replay(tmp_path: Path) -> None:
    """Four short deck-out games, one resident model, sequential game slots."""
    torch.set_num_threads(1)
    matchup = DeckMatchup(
        population_sha256="a" * 64,
        entries=("plains", "mountains"),
        lists=(managym.Deck({"Plains": 10}), managym.Deck({"Mountain": 10})),
    )
    space = ObservationSpace(
        ObservationConfig(max_cards_per_player=16, max_permanents_per_player=16)
    )
    torch.manual_seed(83)
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world="world"),
    )
    path = tmp_path / "policy.pt"
    save_checkpoint(agent, space, path, player_configs=Match(matchup.match()).to_rust())
    player = register_checkpoint(
        path,
        match=matchup.match(),
        content_suite=matchup.content_suite,
        player_id="tiny-mps",
        display_name="Tiny MPS",
        allow_deck_repetition=True,
    ).registration
    opponent = register_anchor(player, "heuristic", player_id="heuristic")
    request = EvaluationRequest(
        key=arena_fixture().key.model_copy(
            update={
                "world": managym.WORLD_VERSION,
                "content_suite": matchup.content_suite,
                "viewer_boundary": player.information_boundary,
            }
        ),
        candidate=player,
        opponent=opponent,
        checkpoint_paths={player.player_id: str(path)},
        deal_seeds=(21390001,),
        authored_matchup=matchup,
        game_seconds=30,
        max_commands=100,
    )
    execution = EvalExecution(
        device="mps",
        max_active_games=1,
        max_batch_rows=4,
        deadline_unix=time.time() + 90,
        diagnostic_batches=1,
    )
    with InferenceRuntime(execution, tmp_path / "allocation") as runtime:
        result = evaluate(request, out_dir=tmp_path / "evaluation", inference=runtime)
        receipt = runtime.snapshot()
    assert not runtime.process.is_alive()
    assert result.status == "completed", result.error
    assert len(result.rows) == 4 and all(row.valid for row in result.rows)
    assert all(row.replay_passed for row in result.rows)
    games = read_trace(tmp_path / "evaluation" / result.rows[0].trace_path)
    assert len(games) == 4 and all(game["decisions"] for game in games)
    counts = receipt["counters"]
    assert counts["model_loads"] == counts["catalog_builds"] == 1
    assert counts["completed_requests"] > 0
    assert counts["required_output_waits"] == counts["forward_calls"]
    assert counts["host_to_device_calls"] > 0
    assert counts["allocated_mps_bytes"] > 0
    assert counts["driver_mps_bytes"] >= counts["allocated_mps_bytes"]
    assert counts["diagnostic_synchronizations"] == 1
    retained = json.loads((tmp_path / "allocation/serving.json").read_text())
    assert not retained["device_events"]
