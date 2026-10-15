"""Replay keeps real collection semantics and rejects changed policy evidence."""

import json
from pathlib import Path

import pytest
import torch

from experiments.runners.pipeline_benchmark import (
    PreparedReplay,
    capture,
    load_services,
    replay,
)
from experiments.runners.run_training_regimes import smoke_recipe


def test_policy_tape_replays_real_collector_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    tape = capture(smoke_recipe("direct-self-play"), tmp_path, 2, 8, 10831)
    audited = replay(tmp_path, verify_inputs=True)
    measured = replay(tmp_path, verify_inputs=False, request_timings=True)
    assert audited.batch_sha256 == measured.batch_sha256 == tape.batch_sha256
    assert audited.journal_sha256 == measured.journal_sha256 == tape.journal_sha256
    assert audited.buffers_sha256 == measured.buffers_sha256 == tape.buffers_sha256
    assert measured.game_seeds == tape.game_seeds
    assert measured.demand.decisions == sum(frame.rows for frame in tape.frames)
    assert measured.demand.learner_transitions == 16
    assert 0 < measured.native_seconds < measured.seconds
    assert measured.native_calls > 0
    assert measured.truncated_games == 0
    assert measured.request_host_seconds is not None
    assert measured.tail_host_seconds is not None
    assert len(measured.request_host_seconds) == len(tape.frames)
    assert min(measured.request_host_seconds) >= 0
    assert measured.tail_host_seconds >= 0
    assert sum(
        measured.request_host_seconds
    ) + measured.tail_host_seconds == pytest.approx(measured.seconds, abs=1e-12)
    original = (tmp_path / "tape.json").read_text()
    data = json.loads(original)
    data["frames"][0]["input_sha256"] = "corrupt"
    (tmp_path / "tape.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="acting-view policy input diverged"):
        replay(tmp_path, verify_inputs=True)
    (tmp_path / "tape.json").write_text(original)
    with (tmp_path / "outputs.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replay(tmp_path, verify_inputs=True)


def test_failed_gpu_cells_are_never_extrapolated(tmp_path: Path) -> None:
    source = tmp_path / "hardware.json"
    common = dict(
        card="fixture",
        capacity="w384-d8",
        kind="model",
        batch=64,
        inference_observations_per_second=6400,
        optimizer_observations_per_second=640,
        peak_allocated_bytes=1_000_000,
    )
    source.write_text(
        json.dumps(
            {
                "hardware": [
                    dict(common, cell="b64", status="completed"),
                    dict(common, cell="b1024", batch=1024, status="failed"),
                    dict(
                        common,
                        cell="b256",
                        batch=256,
                        status="completed",
                        inference_observations_per_second=None,
                    ),
                ]
            }
        )
    )
    services = load_services(source)
    assert len(services) == 1
    assert services[0].batch == 64
    assert services[0].inference_seconds == 0.01
    assert services[0].optimizer_seconds == 0.1
    assert len(services[0].evidence_sha256) == 64


def test_resident_actors_keep_sampling_identity_and_reject_reuse(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    recipe = smoke_recipe("direct-self-play")
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first_tape = capture(recipe, first_path, 2, 8, 10831)
    second_tape = capture(recipe, second_path, 2, 8, 10832)
    first = PreparedReplay(first_path, verify_inputs=True)
    second = PreparedReplay(second_path, verify_inputs=True)
    with pytest.raises(RuntimeError, match="no completed collection"):
        first.verify()
    # Initialization and unrelated sampling must not contaminate another actor.
    torch.rand(173)
    second.collect()
    first.collect()
    assert first.verify().batch_sha256 == first_tape.batch_sha256
    assert second.verify().batch_sha256 == second_tape.batch_sha256
    assert first.batch is not None and second.batch is not None
    with pytest.raises(RuntimeError, match="already attempted"):
        first.collect()
