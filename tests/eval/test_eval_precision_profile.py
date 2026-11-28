"""Exercise the offline forward profiler on real decisions from a small paired deal."""

import json
from pathlib import Path
from statistics import mean

from etude.checkpoints import CheckpointRef
from etude.eval.records import Code, Match, checkpoint_player
import numpy as np
import pytest
import torch

from experiments import eval_precision_profile as experiment
from manabot.backends.torch.checkpoint import save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.eval.agent_loader import load_torch_policy
from manabot.eval.pairing import PairingSpec
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.infra.identity import file_sha256
from manabot.policy.batching import concatenate
from manabot.sim.heuristic import heuristic_source_paths
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import EvalPrecision
import managym


def test_cuda_trace_charges_nested_casts_to_attention_and_kernel_overlap_once() -> None:
    def event(
        name: str,
        start: float,
        duration: float,
        category: str,
        *,
        gpu: bool = False,
        correlation: int | None = None,
    ) -> experiment.TraceEvent:
        return {
            "name": name,
            "cat": category,
            "pid": int(gpu),
            "tid": 1,
            "ts": start,
            "dur": duration,
            "args": {} if correlation is None else {"correlation": correlation},
        }

    events = [
        event(experiment.FORWARD, 0, 100, "user_annotation"),
        event(experiment.FORWARD, 30, 50, "gpu_user_annotation", gpu=True),
        event("aten::to", 2, 5, "cpu_op"),
        event("cudaLaunchKernel", 3, 1, "cuda_runtime", correlation=6),
        event(experiment.ATTENTION, 10, 25, "user_annotation"),
        event("aten::to", 15, 10, "cpu_op"),
        event("cudaLaunchKernel", 20, 2, "cuda_runtime", correlation=7),
        event("attention_kernel", 30, 50, "kernel", gpu=True, correlation=7),
        event("copy_kernel", 40, 10, "kernel", gpu=True, correlation=6),
        event("outside_forward", 110, 10, "kernel", gpu=True, correlation=8),
    ]
    wall, cuda, kernels = experiment.bucket_trace(events)
    elapsed = sum(scope["dur"] for scope in experiment.forward_scopes(events)) / 1e6
    assert sum(wall.values()) == pytest.approx(elapsed)
    assert wall == pytest.approx(
        {"other": 25e-6, "casts_and_copies": 15e-6, "attention": 60e-6}
    )
    assert cuda == pytest.approx({"attention": 40e-6, "casts_and_copies": 10e-6})
    assert kernels == {"attention_kernel": 1}


def test_capture_replays_arena_logits_and_builds_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deal_fixture(
        ref: CheckpointRef, count: int, precision: EvalPrecision
    ) -> tuple[Code, dict[str, PairingSpec], list[Match]]:
        assert count == 1
        roles = ("ur_lessons", "gw_allies")
        a, b = (
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", role)
            for role in roles
        )
        matchup = DeckMatchup(
            population_sha256="a" * 64,
            entries=roles,
            lists=(
                managym.Deck(a.decklist, a.sideboard),
                managym.Deck(b.decklist, b.sideboard),
            ),
            decklists="closed",
        )
        heuristic = Code(
            name="heuristic",
            source=heuristic_source_paths(),
            config={"max_actions": 64},
        )
        match = Match(players=(checkpoint_player(ref), heuristic), config={"deal": 123})
        spec = PairingSpec(
            matchup=matchup,
            deal=123,
            matchup_mode="cross",
            game_seconds=60,
            arena="profile-fixture",
            precision=precision,
        )
        return heuristic, {match.key: spec}, [match]

    monkeypatch.setattr(experiment, "trophy_deals", deal_fixture)
    monkeypatch.setenv("MANABOT_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        check_profile(tmp_path)
    finally:
        torch.set_num_threads(previous)


def check_profile(tmp_path: Path) -> None:
    torch.manual_seed(42)
    space = ObservationSpace(
        ObservationConfig(
            mana_payment="auto",
            mana_floats="usable",
            max_actions=64,
            action_overflow="first-legal",
        )
    )
    config = ModelConfig(
        world=managym.WORLD_SETUP,
        hidden_dim=16,
        num_attention_heads=4,
        attention_layers=1,
    )
    checkpoint = tmp_path / "policy.safetensors"
    save_checkpoint(ActorCritic(space, config), space, checkpoint, player_configs=None)
    batches = tmp_path / "captured.batches"
    experiment.capture(
        checkpoint, batches, tmp_path / "capture", device="cpu", deals=1, slots=4
    )

    device = torch.device("cpu")
    sync = experiment.device_sync(device)
    policy = load_torch_policy(checkpoint, "float32", device)
    with experiment.open_capture(batches) as (_, calls):
        multi = None
        for call in calls:
            logits, _ = experiment.run_call(policy, call.batch, "float32", device, sync)
            np.testing.assert_array_equal(logits, call.logits)
            if call.batch.batch > 1 and multi is None:
                multi = call
        assert multi is not None
    joined = concatenate(list(multi.singles))
    logits, _ = experiment.run_call(policy, joined, "float32", device, sync)
    np.testing.assert_array_equal(logits, multi.logits)
    for actual, single in zip(logits, multi.singles, strict=True):
        expected, _ = experiment.run_call(policy, single, "float32", device, sync)
        np.testing.assert_allclose(actual, expected[0], atol=2e-5, rtol=1e-5)

    alternate = tmp_path / "alternate.safetensors"
    experiment.random_checkpoint(alternate, 32, 4, template=checkpoint)
    digest = file_sha256(alternate)
    widened = load_torch_policy(alternate, "float32", device)
    refused = tmp_path / "refused"
    with pytest.raises(ValueError, match="exact bytes"):
        experiment.replay(
            alternate, batches, refused, device_name="cpu", variant="float32"
        )
    assert not refused.exists()
    (batch,) = experiment.bound_inputs(batches, widened, digest, 4, 1, 0)
    assert batch.batch == 4
    assert experiment.run_call(widened, batch, "float32", device, sync)[0].shape[0] == 4

    receipt = experiment.replay(
        checkpoint,
        batches,
        tmp_path / "replay",
        device_name="cpu",
        variant="float32",
        limit=2,
        repeats=2,
        profile_calls=1,
    )
    buckets = receipt["profile_estimated_bucket_mean_seconds"]
    assert isinstance(buckets, dict)
    assert buckets["attention"] > 0
    assert sum(buckets.values()) == pytest.approx(
        receipt["profile_synced_sample_mean_seconds"]
    )
    assert receipt["cuda_buckets_seconds"] is None
    # With seed 1 the second call replaces the first in the one-call sample.
    calls = [
        json.loads(line)
        for line in (tmp_path / "replay/calls.jsonl").read_text().splitlines()
    ]
    assert receipt["profile_synced_sample_mean_seconds"] == mean(
        call["forward_seconds"]
        for call in calls
        if call["repeat"] and call["call"] == 1
    )
    report = tmp_path / "report.md"
    experiment.render_report(tmp_path, report)
    assert "| replay |" in report.read_text()
